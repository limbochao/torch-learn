---
title: 已推送的八类 Inductor 修复：问题、前后 DSL 与正确性依据
published: false
---

# 已推送的八类 Inductor 修复：问题、前后 DSL 与正确性依据

本文补充[四模型修复记录](model-fixes-20261020.md)，逐项解释已经推送的八类修复。
框架基线为 `0b4b636555`，整合分支为 `debug/cicd-all-fixes-20261020`、`5b18fac653`。
这里只说明已有修改，不为[本轮三个模型定界](three-model-boundary-20261021.md)增加框架修复。

DSL 指 Inductor 生成、交给 Triton 编译的 kernel 程序；IR 是生成 kernel 之前的图或调度表示。
下面标为“原始摘录”的代码保留产物原文；标为“语义示意”的代码用于解释，不冒充捕获产物。
调度、stride 证明和 range 拆分的原失败发生在 DSL 生成之前，因此其“修复前 DSL”为未生成。
这三项提供失败 IR/表达式、修复前后代码以及成功生成/执行的证据，不伪造一份失败 kernel。

## 修复覆盖与故障层级

| 修复分支 | 涉及模型/用例 | 故障层级 | 整合提交 |
| --- | --- | --- | --- |
| `bugfix/codegen-reduction-mask` | GPT2ForSequenceClassification、BERT_pytorch 前向 | 各模型第一层 | `a514111f10` |
| `bugfix/codegen-reduction-axes` | BERT_pytorch 静态训练 bias gradient | 第二层 | `22b6b739eb` |
| `bugfix/codegen-negative-index` | 独立负索引回归，未证实四模型触发 | 不分配模型层级 | `c32818e10e` |
| `bugfix/codegen-load-liveness` | AlbertForMaskedLM 多路参数梯度 | 第一层，与 tiling 共同解决 | `7381eb8188` |
| `bugfix/tiling-resource-budget` | AlbertForMaskedLM、nfnet_l0 | 两者第一层；另覆盖 Albert softmax 第二层 | `2d147618bc` |
| `bugfix/scheduler-symbolic-groups` | nfnet_l0 动态训练 | 第二层 | `e1429cc917` |
| `bugfix/layout-symbolic-strides` | NFNet 派生小图第二组动态 shape | 第三层，仅小图 | `c74c45c960` |
| `bugfix/codegen-range-split` | 同一小图的初次 backward 编译 | 第四层，仅小图 | `5b18fac653` |

层级表示前一阻塞解决之后实际暴露的下一问题，不表示组件栈。
各独立分支的 SHA 与整合提交不同，准确对应关系见
[branches.json](../../../artifacts/cicd/model_fixes_20261020/branches.json)。
八份完整补丁保存在 [patches](../../../artifacts/cicd/model_fixes_20261020/patches/)。

## 1. Persistent reduction：计算后必须重新屏蔽无效 lane

**问题描述。** GPT2/BERT 的 embedding + norm 在隐藏维度 768 上计算均值、方差。
实际 SIMT tile 扩到 1024；多出的 256 个 lane 不属于输入。
load 设置 `other=0` 并不足以保证后续表达式的无效 lane 仍是规约单位元。
例如减去均值再平方后，尾部变成 `mean²`，错误方差额外包含 `256 * mean² / 768`。

**问题 DSL，原始 GPT2 kernel 摘录：**

```python
        tmp8 = tmp2 + tmp7
        tmp10 = tl.sum(tmp8, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = tl.full([1, 1], 768.0, tl.float32)
        tmp12 = (tmp10 / tmp11)
        tmp13 = tmp8 - tmp12
        tmp14 = tmp13 * tmp13
        tmp16 = tl.sum(tmp14, 1).reshape(X0BLOCK_SUB, 1)
```

**修复逻辑。** `_codegen_mask` 与 `filter_masks` 保留不能证明恒真的条件；
`reduction()` 在执行融合表达式之后、实际归约之前，用 `tl.where` 选择单位元。
sum 为 0、prod 为 1、max 为负无穷、min 为正无穷，整数极值按实际 dtype 转换。
mask 与 value 采用同一轴顺序、permute 和 reshape，不能把某个轴的条件广播到另一轴。
只有长度能被实际 `BLOCK_SUB` 整除时才消去 mask；SIMT 的二次幂扩展也计入这个证明。

**修复后 DSL，同一模型生成路径的原始摘录：**

```python
        tmp8 = tmp2 + tmp7
        tmp10 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1]) & r1_mask.reshape([1, R1BLOCK_SUB])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp8, 0)
        tmp11 = tl.sum(tmp10, 1).reshape(X0BLOCK_SUB, 1)
        tmp12 = tl.full([1, 1], 768.0, tl.float32)
        tmp13 = (tmp11 / tmp12)
        tmp14 = tmp8 - tmp13
        tmp15 = tmp14 * tmp14
        tmp17 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1]) & r1_mask.reshape([1, R1BLOCK_SUB])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp15, 0)
        tmp18 = tl.sum(tmp17, 1).reshape(X0BLOCK_SUB, 1)
```

**为什么可以这样修。** 对合法 lane，`where` 仍选择原表达式；对非法 lane，选择运算单位元，
不会改变合法元素集合的规约结果。应屏蔽的是最终待规约值，而不只是最初 load。
这也能阻断下层 masked indirect load 错误尾值对归约的影响，但没有修复那个独立 TA 缺陷。
其状态仍为“缺陷未修复，影响被屏蔽；不阻塞当前模型验证”。

证据：[原 DSL][mask-old]、[修复后完整 DSL][mask-new]、[补丁][mask-patch]。
验证：mask 单元 4/4、静态/动态数值回归 22/22；GPT2 累计 8/8，BERT 前向恢复后还暴露下一项。

## 2. 多轴归约：压平 reduction axes，保留输出 channel

**问题描述。** BERT 静态反向 bias gradient 应对两个批次/序列轴求和，保留 768 个 channel。
原输入重排对应 `reshape(4,12,128,64).permute(0,2,1,3).reshape(512,768).sum(0)`。
其中 X4=12、X5=64 是输出维，R2=4、R3=128 是规约维。
旧代码把 R2、R3、X5 全压成一维，产生一个标量，再把相同标量写给全部 X5。

**问题 DSL，原始摘录：**

```python
                tmp0 = tl.load(in_ptr0 + (x5 + 64*r3 + 8192*x4 + 98304*r2), x5_mask, other=0.0)
                tmp1 = tl.reshape(tmp0, [R2BLOCK_SUB* R3BLOCK_SUB* X5BLOCK_SUB])
                tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
                tl.store(out_ptr1 + (x5 + 64*x4 ), tmp3, x5_mask)
```

**修复逻辑。** 以 `golden_var_list` 的实际 value 布局为准，把 preserved axes 放前面、
reduction axes 放后面，只合并后者；沿合并后的 reduction dim 求和，再恢复 broadcast 维。
同时修正 CSE 表达式的 shape 元数据，使后续 layout 分析看到的形状与已生成 DSL 一致。

**修复后 DSL，原始摘录：**

```python
                tmp1 = tl.reshape(tmp0.permute([2, 0, 1]), [X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB])
                tmp3 = tl.where(tl.reshape((tl.broadcast_to((x5_mask.reshape([1, 1, X5BLOCK_SUB])), [R2BLOCK_SUB, R3BLOCK_SUB, X5BLOCK_SUB])).permute([2, 0, 1]), [X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB]), tmp1, 0)
                tmp4 = tl.sum(tmp3, 1).reshape(1, 1, X5BLOCK_SUB)
                tl.store(out_ptr1 + (x5 + 64*x4 ), tmp4, x5_mask)
```

**为什么可以这样修。** permute/reshape 只重排索引，没有增加或丢失元素；
每个 X5 对应的 R2×R3 元素集合保持独立，求和后仍有 X5BLOCK_SUB 个结果。
外层 X4 循环再覆盖 12 组，合计 12×64=768 个输出，符合 bias gradient 定义。
反过来，原来的单标量不可能表示任意 64 个独立 channel 的梯度。

证据：[原 DSL][axes-old]、[修复后完整 DSL][axes-new]、[补丁][axes-patch]。
验证：bias gradient/分离规约轴的静态和动态回归 4/4，BERT 静态训练 accuracy 恢复通过。
此分支依赖 mask 修复；整合后的产物同时包含两项，不能把所有差异都归因于 axes。

## 3. 负索引：dump/check 开关不能取消 wrap

**问题描述。** 对长度 8 的 tensor，索引 -1 应访问第 7 个元素。
旧 `CSEProxy.indirect_indexing()` 在设置 dump/check 环境变量时强制 `wrap_neg=False`，
使相同图因调试开关不同而改变访问地址。此项来自专门回归，未证明原四模型输入触发。

**问题与修复后 DSL 语义示意：** 以下是由实际 codegen 分支导出的等价语义，不是模型原始产物。

```python
# 旧：dump/check 强行禁用 wrap，idx=-1 仍访问负地址
value = tl.load(table + idx * 16 + column)

# 新：保持调用方 wrap_neg=True 的要求
wrapped = tl.where(idx < 0, idx + 8, idx)
value = tl.load(table + wrapped * 16 + column)
```

**修复逻辑。** 删除环境变量修改 `wrap_neg` 的分支，遵守调用者的语义。
另修正 bounds 传播：只有实际 wrap 才计算加 size 后的范围，no-wrap 必须保留原 bounds。
例如 `[-2,3]` 在 wrap 模式变为 `[0,7]` 的保守范围，no-wrap 模式仍为 `[-2,3]`；
若后者也标成非负，后续可能错误消除 lower-bound check。CSE 同时保留 dtype/shape。

**为什么可以这样修。** wrap 是索引算子的行为，调试开关只应增加观察/检查，不能改变算子行为。
对于明确要求 no-wrap 的调用，修复也没有强制 wrap，而是保留负范围及对应边界检查。
这是两种调用协议各自的一致性修复，不是给所有负地址加 size。

证据：[完整前后 codegen 补丁][negative-patch]、[no-wrap 单元结果][negative-unit]、
[静态 dump 数值结果][negative-result]。
验证：静态/动态 × none/dump/check 共 6 项数值通过，no-wrap bounds 专项通过。
修复后的 v8 真实静态 kernel 已从原回归缓存补充归档，明确保留了 `idx<0` 的转换：

```python
            tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 8, tl.int32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp0 < 0
            tmp4 = tl.where(tmp3, tmp2, tmp0)
            tl.device_assert(((0 <= tmp4) & (tmp4 < 8)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 8")
            tmp6 = tl.load(in_ptr1 + (x1 + 16*tmp4), x1_mask & y0_mask)
```

见[负索引完整 DSL][negative-dsl]。修复前没有留存同一回归的完整 kernel，
旧行为以补丁删除的环境变量分支和上面的语义示意为证，没有将示意标成实测产物。

## 4. Load 生存期：不要让 81 个输入 tile 同时存活

**问题描述。** Albert LayerNorm 参数梯度融合 12 路贡献，生成 81 次输入 load。
旧 codegen 把 load 都放入 `self.loads`，把消费运算放入后续 compute buffer；
较多中间 tile 同时占用片上 UB，候选编译因资源不足失败。

**问题 DSL，原始 load 序列开头：**

```python
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), None)
        tmp4 = tl.load(in_ptr3 + (r1), None)
        tmp11 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp12 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
```

这里继续加载其余输入后才执行 `tmp1 - tmp2` 等计算；完整顺序见原始文件。

**修复逻辑。** reduction 内普通输入 load 写入 compute，与首次消费计算相邻；
依赖计算结果的 indirect/tmpmask load 同样进 compute；可提升的循环不变量仍留在 prefix。
load 后的 layout transform 也放入相同 buffer，保证先定义、后使用。

**修复后 DSL，原始摘录：**

```python
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), None)
        tmp3 = tmp1 - tmp2
        tmp4 = tl.load(in_ptr3 + (r1), None)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp8 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp6, 0)
        tmp9 = tl.sum(tmp8, 0).reshape(1, X0BLOCK_SUB)
        tmp11 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp0, 0)
        tmp12 = tl.sum(tmp11, 0).reshape(1, X0BLOCK_SUB)
        tmp13 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp15 = tmp13 + tmp14
```

**为什么可以这样修。** 不改 load 的地址、mask、dtype，也不改各路梯度的表达式和贡献集合；
改变的是互不依赖指令之间的排布。消费者及时计算后，前一组输入不必全部保留到最后一组加载。
有真实数据依赖的 load 不会被提前，循环公共值也没有被错误移入循环。
UB 估计不是精确编译器模型，最终仍需固定候选编译和数值对照。

证据：[原 DSL][load-old]、[修复后完整 DSL][load-new]、[补丁][load-patch]、
[固定候选数值结果][load-numeric]。deferred 静态 performance 对照最大差约 `3.66e-4/2.44e-4`，
在 `atol=0.002、rtol=0.001` 下通过。Albert 四项训练失败还需要下一项提供合法 tile，
不能声称仅重排 load 就独立解决整网所有候选。

## 5. Tiling：搜索必须覆盖设备能编译的较小 tile

**问题描述。** NFNet dynamic pooling 的 SIMT 候选受到每 tile 8 KiB 的限制。
FP32 对应最多 2048 个元素，而旧搜索在 4096 元素处已停止，未枚举任何合法 SIMT 小 tile。
Albert 多输入梯度还受到 pointer 数量被截断为 3、persistent 最小 tile 未保留的影响。
切换 template/SIMT 时资源上限不更新，`self.config` 拼写错误又没有清空真实 `self.configs`。

**问题 DSL 与修复后 DSL。** NFNet pooling 的函数体没有修改，下面是两边共有的原始摘录：

```python
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
```

实际差异在 constexpr 和编译 options，而不是把四点均值换一种算法：

| 阶段 | 生成/选择的候选 | 结果 |
| --- | --- | --- |
| 修复前 | SIMT 搜索未进入 `XBLOCK<=2048` 范围；template 候选编译失败 | 无合法候选 |
| 修复后 | 正常 autotune 选中 `XBLOCK=512, compile_mode=simt_only` | 三个 dynamic pooling kernel 均可执行 |

**修复逻辑。** 保存真实 pointer 数量，切换 kernel type 时重算容量和搜索停止点；
SIMT 搜索上限与已有 8 KiB 检查一致；每种 mode 独立清空候选。
persistent/no-loop 轴不可继续切分，保守的 pointer 估计不应提前丢弃最小候选；SIMT 的硬资源限制仍保留，实际合法性由编译器验证。

Albert 第一层解决后又暴露动态 softmax 第二层：`rebuild_dynamic_axis()` 漏算完整 R3，
并用 `axis_name == 'r'` 判断实际名字 `r3`，把 X2 错误设为不可继续切分。
修复先乘入全部 persistent reduction 轴，使用 `startswith('r')`，再决定其他轴能否免循环。

**Softmax 的前后 DSL 参数化方式不变，原始摘录：**

```python
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
```

独立固定 DSL 对照中，`X2BLOCK_SUB=512、R3BLOCK_SUB=512` 编译超 UB，
仅将 X2 改为 8 后编译及 eager 比较通过，最大差 `6.98e-10`。
这证明存在合法小 tile；框架修复是恢复枚举能力，没有在框架里写死 X2=8。

**为什么可以这样修。** XBLOCK/X2BLOCK_SUB 只改变分块和循环次数，原 mask 仍限制合法索引，
所有输出元素仍被覆盖，每行 softmax 的 reduction 轴仍完整。
资源阈值只筛选候选，不改变数学运算。下层合理拒绝超 UB 的候选，不等于下层有 bug。
这也没有在 Inductor 中过滤某个模型名或固定绕过特定 TA 编译错误。

证据：[完整补丁][tile-patch]、[修复前 pooling DSL][pool-old]、[修复后 pooling DSL][pool-new]、
[实际选中配置][pool-config]、[softmax DSL][softmax-new]、[固定候选失败][softmax-fail]、
[小 tile 数值通过][softmax-pass]。tiling 单元 3/3；Albert 累计 8/8、NFNet 推理恢复通过，
后者动态训练继续暴露下一项。

## 6. Scheduling：比较符号尺寸的数学值，而非表达式结构

**问题描述。** NFNet dynamic training 的融合组包含 pointwise producer、空间 sum、epilogue。
令 `q=FloorDiv(H-1,32)`、`A=1536*batch`，真实调度 IR 可写为：

```text
producer group = (A*q*q + 2*A*q + A, 1)
reduction group = (A, q*q + 2*q + 1)
epilogue group = (A, 1)
```

**修复前 DSL：未生成。** 旧调度器用结构 `==` 比较 producer_numel 与 `numel*rnumel`；
展开式与因式分解式在 SymPy 中不是同一结构，错误抛出 `unexpected group`，尚未调用 Triton。

**修复逻辑，前后代码语义：**

```python
# 旧匹配路径
node_numel == numel * rnumel
# 新：NumelList 先转换为其总元素数；结构快路径后尝试证明相等
sizevars.statically_known_equals(node_numel, numel * rnumel)
```

两条 NPU 调度路径统一复用该匹配，不再只对 Welford 融合启用符号证明。
无法证明相等的组仍拒绝，不追加新的 shape guard，也不把所有组强行判为可融合。

**修复后真实 DSL，NFNet 反向 producer → reduction → epilogue 的摘录：**

```python
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tl.full([1, 1], 1.0, tl.float32)
    tmp31 = tmp30 - tmp29
    tmp32 = tmp29 * tmp31
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr2 + (x0), tmp33, xmask)
```

**为什么可以这样修。** `A*(q²+2q+1)=Aq²+2Aq+A` 是恒等式，两个组的元素数确实相等。
修复仅识别原本合法的尺寸关系，没有改变依赖顺序、归约循环开闭或融合策略。
生成后 `tmp26` 沿空间轴求和，再计算 sigmoid 梯度 epilogue，符合原 IR。

证据：[原始完整调度 IR][schedule-ir]、[修复后完整 DSL][schedule-new]、[补丁][schedule-patch]。
验证：等价组、非等价组拒绝及无新增 guard 的单元 2/2；NFNet dynamic training 两项通过。
同一小图扩展到第二组动态 shape 后另暴露下一项，不能把第一组通过当成两组都通过。

## 7. Symbolic stride：整数整除证明不能引入倒数

**问题描述。** NFNet 派生训练小图第二组输入 `(3,16,256,256)` 在 convolution backward
layout 证明时失败。实际 stride 为 `a=q+1`、`b=(q+1)²`，`q=FloorDiv(H-1,32)`、H≥33。
旧 SymPy `%` 会把 `Mod(a,b)` 变为包含 `b/a` 的表达式；展开后残留倒数，
整数值域分析遇到指数 -1，抛出 `Invalid ranges [0:-1]`。

**修复前 DSL：未生成。** 异常发生于 `Layout._stride_expr_ge_or_false()`，
不分配 NPU tensor 的 ShapeEnv 元数据调用也可复现；下层编译器尚未参与。

**修复前后符号/IR 对照：**

```python
# 旧：SymPy 通用 Mod 可以引入有理式
Eq(left % right, 0)
# 原问题中间形式：a * Mod(1, b/a)

# 新：表达整数取模，不在该边界引入倒数
Eq(PythonMod(left, right), 0)
```

**修复逻辑。** 保留原先的 right=0、left=0、大小证明和整除证明顺序，
只将整除表达式替换成社区已有的 `PythonMod`；仍由 sizevars 证明真假。
在 NPU `_sizevars.patch_simplify` 安装补丁并防止重复安装。
baseline 保存/恢复使用 `inspect.getattr_static`，保留 staticmethod 描述符，避免恢复后错误绑定 self。

**修复后 DSL 的变化。** 这项不改任何 Triton 计算或访存模板，修复的是生成之前能否证明布局合法。
对先前已能编译的输入，没有需要替换的 DSL 表达式；对失败的第二组 shape，
修复后才继续生成 convolution backward 等 kernel。已从实际 v8 缓存补充[双 shape 小图的反向 DSL][training-dsl]，它不使用前一项 NFNet 整网 v5 的产物。
其中一个下游 kernel 的原始摘录为：

```python
    tmp6 = tmp1 * tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = (ks0).to(tl.float32)
    tmp9 = (tmp8).to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp2 + tmp10
    tl.store(out_ptr0 + (x2), tmp11, xmask)
```

这里 `ks0` 是空间归约长度，修复后仍按原图对空间 mean 的梯度除以该长度，
再与逐点梯度相加；没有把符号证明替换成新的数值算法。

**为什么可以这样修。** stride 是整数；对非零 right，`left` 能被 `right` 整除等价于整数取模为 0。
替换不改变这条判定，只避免符号化简进入整数分析不支持的有理式域。
实际元数据对照中 `compare(a,b)=False`、`compare(b,a)=True`，符合 a≥2 时 a<b 的关系。
不捕获异常后强制放行，不针对某个模型或 shape 做特判。

证据：[补丁及四项框架检查][stride-patch]、[旧版元数据异常][stride-old]、
[最终 stride 检查][stride-new]、[双 shape 训练日志][training-new]。
v8 静态/动态各两组 shape 的前向、输入梯度、权重梯度均通过；原 NFNet 模型矩阵没有观察到此新失败。

## 8. Range 拆分：合法拒绝应返回“不兼容”，不能变成 TypeError

**问题描述。** 同一双 shape 小图在 stride 修复后最终能得到正确结果，
但第二组初次 backward 编译仍报 `CantSplit.__init__()` 缺少 `expr/remaining`。
原因是社区异常要求两个操作数，NPU 四个拒绝分支却使用无参 `raise CantSplit`；
另一分支在判断 group 耗尽之前访问 `remaining[current_group]`，先触发 IndexError。

**修复前 DSL：这个失败的融合候选没有生成 DSL。** `is_compatible()` 本应捕获 `CantSplit`
并返回 False，旧版实际抛出的 TypeError/IndexError 不在捕获协议内。
后续 AOT 重试能数值通过，也不能证明第一次检查正确。

**修复逻辑，前后代码：**

```python
# 旧
raise CantSplit
# 新：各分支传入当前不能整除的实际操作数
raise CantSplit(remaining[i], expr)

# 新：在读取 remaining[current_group] 之前检查 group 是否耗尽
if current_group >= len(remaining):
    raise CantSplit(size, 0)
```

**修复后 DSL/索引行为。** 被拒绝的融合候选仍不生成 DSL；`is_compatible()` 正常返回 False，
让已有调度流程选择合法安排。合法的多组拆分算法保留，例如 groups `[2,3,5]` 合成长度 30 时，
重建索引仍是 `15*i0 + 5*i1 + i2`，没有将 NPU 多组算法替换成仅支持两组的算法。
这项改变异常协议，不改变成功候选的 load/store/归约公式。
整图成功路径实际生成的 v8 反向 kernel 见[完整 DSL][training-dsl]；
不能将被拒绝的融合候选写成“修复后强行生成了 kernel”。

**为什么可以这样修。** 无法整除本来就意味着该候选不兼容；传参使既定拒绝信号能被既定调用者捕获，
而不是把不合法融合强行编译。提前检查耗尽条件也只是将 Python 越界变成同一拒绝协议。
正常索引重建的对照证明没有破坏合法三组拆分。

证据：[完整补丁与三项检查][range-patch]、[旧版五个拒绝错误][range-old]、
[修复后检查][range-new]、[最终动态双 shape 训练][training-new]。
v8 range 检查 3/3，双 shape 日志不再出现 CantSplit TypeError。
原 NFNet 整网没有观察到这个新错误，因此不把它写成 NFNet 整网第四层失败。

## 验证范围与剩余问题

四个历史模型累计 32/32 场景通过，但结果来自不同修复阶段，不能称为 v8 整套矩阵重新跑过。
最终 v8 的证据是 32 项数值回归、17 项针对性框架检查，以及静态/动态双 shape 训练小图。
准确版本和通过范围以[原修复日志](model-fixes-20261020.md)及
[模型矩阵](../../../artifacts/cicd/model_fixes_20261020/model_matrix.json)为准。

TA masked indirect load 的 template 独立用例仍失败；上面的 Inductor mask 修复屏蔽了它对已验证模型归约结果的影响。
它没有被这些分支修复，模型 PASS 与缺陷仍存在分别记录。
本轮 MobileNet、fastNLP BERT、ConvNeXt 新发现的问题也不属于这八个已修复条目，
其状态和复现见[三模型定界记录](three-model-boundary-20261021.md)。

[mask-old]: ../../../artifacts/cicd/four_model_matrix_20261020/kernel_sources/GPT2ForSequenceClassification/s_infer_acc/GPT2ForSequenceClassification__0_inference_0.0/triton_per_fused_add_addmm_arange_embedding_na_3.py
[mask-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/gpt2_embedding_norm.py
[mask-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-reduction-mask.patch
[axes-old]: ../../../artifacts/cicd/four_model_matrix_20261020/kernel_sources/BERT_pytorch/s_train_acc/BERT_pytorch__0_backward_3.1/triton_per_fused_clone_matmul_backward_sum_tra_11.py
[axes-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/bert_bias_gradient.py
[axes-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-reduction-axes.patch
[negative-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-negative-index.patch
[negative-unit]: ../../../artifacts/cicd/model_fixes_20261020/final_checks/negative.txt
[negative-result]: ../../../artifacts/cicd/model_fixes_20261020/regressions_v8/negative_static_dump.json
[load-old]: ../../../artifacts/cicd/four_model_matrix_20261020/kernel_sources/AlbertForMaskedLM/s_train_perf/triton_per_fused_add_native_layer_norm_backwar_14.py
[load-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/albert_backward.py
[load-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-load-liveness.patch
[load-numeric]: ../../../artifacts/cicd/four_model_matrix_20261020/analysis/Albert_configs/numeric_validation.json
[tile-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/tiling-resource-budget.patch
[pool-old]: ../../../artifacts/cicd/four_model_matrix_20261020/kernel_sources/nfnet_l0/d_infer_perf/triton_poi_fused__native_batch_norm_legit_avg_pool2d_convolution_view_21.py
[pool-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/nfnet_pooling_v4.py
[pool-config]: ../../../artifacts/cicd/model_fixes_20261020/nfnet_pooling_selected_v4.json
[softmax-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/albert_softmax_v2.py
[softmax-fail]: ../../../artifacts/cicd/model_fixes_20261020/softmax_minimal_original.txt
[softmax-pass]: ../../../artifacts/cicd/model_fixes_20261020/softmax_minimal_fixed.txt
[schedule-ir]: ../../../artifacts/cicd/model_fixes_20261020/scheduler/nfnet_backward_fused_group_ir.txt
[schedule-new]: ../../../artifacts/cicd/model_fixes_20261020/generated/nfnet_backward_v5.py
[schedule-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/scheduler-symbolic-groups.patch
[stride-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/layout-symbolic-strides.patch
[stride-old]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v7/baseline_unit.txt
[stride-new]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_stride_unit.txt
[training-new]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_training.txt
[range-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-range-split.patch
[range-old]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/baseline_range_unit.txt
[range-new]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_range_unit.txt

[negative-dsl]: ../../../artifacts/cicd/model_fixes_20261020/explanation_20261021/negative_index_v8.py
[training-dsl]: ../../../artifacts/cicd/model_fixes_20261020/explanation_20261021/training_backward_v8.py
