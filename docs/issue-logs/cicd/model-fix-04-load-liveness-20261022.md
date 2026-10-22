---
title: 修复 ④：load 为什么从 loads 移到 compute，怎样降低 UB 峰值
published: false
---

# 修复 ④：load 为什么从 loads 移到 compute，怎样降低 UB 峰值

对应整合提交 `7381eb8188823769432d6d6cce6a505c067b6eee`，主题为
`fix(inductor): emit reduction loads beside their consumers`。
该提交只修改 `torch_npu/_inductor/codegen/triton.py` 的
`NPUIndexTritonKernel.load()`，共三个逻辑改动，完整内容见[补丁][patch]。
八项关系见[总览][overview]。

## `load()` 返回的不是已经读到的设备数据

模型图中的读取被 codegen 解释时，`load(name, index)` 接收 buffer 名称和地址索引表达式，
分析 mask、dtype、块形状，构造 `tl.load(...)` 字符串，返回一个 CSE 临时量对象。
这个对象可能叫 `tmp0`，表示将来 kernel 执行时才读到的 tile。

`self.loads` 和 `self.compute` 都是 Python 的 `IndentedBuffer`，保存待输出的源码行。
名字不同意味着最后拼接的位置不同，并不表示一个分配在全局内存、一个分配在 UB。
UB 是 NPU 用于保存计算数据的片上存储，容量有限。
实际存储分配由后面的编译器完成，生成顺序会影响它看到的临时量生存期。

| 缓冲区 | 在相关 codegen 路径中的职责 |
| --- | --- |
| `prefix` | 为循环公共读取、累加器初始化等提供前置位置 |
| `loads` | 在对应计算段之前集中输出的读取 |
| `compute` | 按图解释顺序写入的算术和依赖计算 |
| `stores` 等 | 在适当层级输出结果写回 |

`codegen_body()` 按循环层级拼接这些缓冲区；相关块内会先拼 loads、后拼 compute。
循环外公共值还要经 prefix 输出，实际代码也有归约后专用缓冲区，
因此不能把上表简化成所有 kernel 都只有四段、完全相同的循环层级。

## 为什么 Python 调用顺序正确，DSL 仍可能把所有 load 放在前面

假设图的解释顺序是“读第一路、算第一路、读第二路、算第二路”。
如果读取写入 `loads`、计算写入 `compute`，最终输出会是：

```python
# 语义示意：旧缓冲区分流后的结果
a0 = tl.load(ptr_a0 + offsets)
b0 = tl.load(ptr_b0 + offsets)
a1 = tl.load(ptr_a1 + offsets)
b1 = tl.load(ptr_b1 + offsets)
g0 = a0 * b0
s0 = tl.sum(g0, 0)
g1 = a1 * b1
s1 = tl.sum(g1, 0)
```

在第二路刚加载完时，第一路的 a0/b0 仍未使用，第二路也要存着，多个 tile 同时存活。
把普通归约读取写入 compute 后，生成顺序可变成：

```python
# 语义示意：读取和消费计算进入同一个缓冲区
a0 = tl.load(ptr_a0 + offsets)
b0 = tl.load(ptr_b0 + offsets)
g0 = a0 * b0
s0 = tl.sum(g0, 0)
a1 = tl.load(ptr_a1 + offsets)
b1 = tl.load(ptr_b1 + offsets)
g1 = a1 * b1
s1 = tl.sum(g1, 0)
```

完成 s0 后，若 a0/b0/g0 没有其他用途，其存储就有机会被第二路复用。
生存期指从值被定义到最后一次使用的区间；这里关注多个区间的重叠峰值。
Albert 参数梯度融合了多路贡献，原 kernel 有 81 次输入 load，问题因此被放大。
修复不要求取消融合，只调整融合 kernel 内部的读取位置。

## 修改前先看已有分支

`load()` 在构造 `line = "tl.load(...)"` 后，根据这些条件选择写到哪个缓冲区：

| 条件 | 含义 |
| --- | --- |
| `has_tmpmask` | load 的 mask 依赖前面算出的临时值 |
| `indirect_indexing` | load 的地址依赖计算出的索引 |
| `inside_reduction` | 当前正在生成归约体，而不是纯逐点/归约后部分 |
| `range_trees[-1].is_loop` | 归约轴存在需要输出的循环 |
| `has_rindex` | 地址使用归约索引，会随归约循环变化 |

这些条件不是模型名或固定 shape，决定的是读取与前序计算、循环变量之间的依赖。

## 三处修改逐行解释

### 1. `if has_tmpmask` 扩展为 `if has_tmpmask or indirect_indexing`

```python
if has_tmpmask or indirect_indexing:
    # Indirect loads and their masks depend on computed values.
    load_buffer = self.compute
```

`has_tmpmask` 原来已经要求先算 mask、再读取。
新增 `indirect_indexing`：即便没有临时 mask，只要地址来自计算值，也必须先定义这个地址。
若把依赖 tmp 索引的 load 放到集中 loads 段，可能早于定义 tmp 的 compute 段。
这个分支排在最前面，优先满足真实数据依赖，也防止它进入后面的循环外提升分支。
其范围不限于 reduction。

### 2. 保留循环公共 load 的 prefix 分支，再新增普通归约分支

完整相关条件为：

```python
elif (
    self.inside_reduction
    and self.range_trees[-1].is_loop
    and not indirect_indexing
    and not has_rindex
):
    load_buffer = self.prefix
elif self.inside_reduction:
    load_buffer = self.compute
else:
    load_buffer = self.loads
```

第一段是既有逻辑：在归约循环里，地址不依赖归约索引、也不是间接读取的公共值可以提升。
例如按 channel 读取的系数在遍历多个归约块时保持不变，留在 prefix 可避免重复加载。
因为它放在新增 reduction 分支前面，修复没有把这类公共读取全部塞回循环。

新增的 `elif self.inside_reduction` 捕获其余归约输入。
这些输入进入 compute，跟解释图时紧随其后的消费运算保持在同一个缓冲区里。
非归约的普通读取仍进 loads；“所有 load 都改成 compute”不符合代码。

### 3. load 的 permute/reshape 结果必须跟在 load 之后

旧代码在 `index_analyze.need_permute` 分支中，总把布局转换写入 `self.loads`。
若只改前两处，可能生成“先 permute(tmp0)，后 load 定义 tmp0”的倒序。
新增选择是：

```python
transform_buffer = self.compute if load_buffer is self.compute else self.loads
result_var = self.cse.generate(transform_buffer, line, dtype=dtype, shape=result_var.shape)
```

若 load 在 compute，消费它的布局转换也进 compute，维持定义先于使用。
若 load 在 prefix，转换仍在 loads，位于定义之后；若 load 本来在 loads，转换也在 loads。
这里比较对象身份 `is`，判断的是同一个代码缓冲区，不是检查 tile 的值。

同一函数内，退化/标量读取后的 `append_broadcast` 分支原本已使用 `load_buffer`，
不需要重复修改；block pointer 的 advance 也仍写到对应 load buffer。
后续 layout 元数据处理属于已有代码，本提交没有增加新的地址、mask 或梯度公式。

## 对照真实 Albert DSL

原始 kernel 先集中读取多路输入，之后才生成消费者，完整产物入口见[总览][overview]第 4 项。
[修复后的 kernel][new-dsl]中能看到如下原始片段：

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
```

前三次读取之后立即出现减法、乘法和归约；后续各路不再全部抢在这些计算之前读取。
片段中的 where 来自已整合的 mask 修复，不能算成本提交新增的第四处修改。

## 为什么在 `load()` 改，而不是修改 scheduler 或直接增大 UB

这里正是代码生成器决定某次读取输出到哪一段的位置，并且已有索引依赖分析。
scheduler 安排的是图节点如何融合、归约前后如何执行；即便节点顺序合理，
load/compute 分流仍可能把节点内部的读取重新集中起来，所以仅改节点顺序不一定触及问题。
硬件 UB 容量固定；编译器也不能凭空为超容量配置提供更多片上空间。

本修复缩短的是生成顺序上的生存期，下层编译器仍可能重新排程。
因此源码解释只能说明为什么有机会降低峰值，实际是否缓解必须看固定配置编译结果。

## tile 修复能否替代它，是否为模型测试必需

两项作用不同：tile 修复减少每个临时 tile 的大小或恢复小候选的枚举，
本提交减少同时存活的临时 tile 数量。固定 Albert kernel 的既有对照为：

| 场景 | 仅把 X tile 改成 1 | 仅重排 load，保留原 tile |
| --- | --- | --- |
| 静态 performance，原 X=1 | 仍失败：需要 6,030,080 bits，只有 2,031,616 bits | 编译、数值通过 |
| 静态 accuracy，原 X=16 | 编译、数值通过 | 编译通过 |
| 动态 performance，原 X=80、R=128 | 编译、数值通过 | 仍编译失败 |
| 动态 accuracy，原 X=8、R=512 | 编译、数值通过 | 编译通过 |

数据见[静态对照][static]、[动态对照][dynamic]、[数值对照][numeric]。
静态 performance 已经 X=1，persistent R 在该 DSL 中固定为 2048，
既有拆分规则无法再靠缩小 X 解决；该场景有 load 重排必要性的直接证据。
动态 performance 则说明 load 重排不是所有 UB overflow 的充分修复。
把 persistent 算法改成其他分段归约算法属于额外方案，不在本提交范围内。

尚未对最终整合版做“只撤销本提交”的完整模型矩阵，
因此不能把本提交三个分支改动分别称为所有 Albert 场景的必要条件。
已有数值验证包含 deferred 静态 performance，对照在 `atol=0.002, rtol=0.001` 下通过。

## 影响范围和风险

普通归约输入、间接读取和布局转换的排布都可能受影响。
读取地址、mask、dtype、各路贡献的数学公式没有被本提交改写；
但排布变化会影响预取、访存与计算重叠、存储复用，可能降低某些 kernel 的访存并行度。
减少峰值 UB 不等于所有模型都加速，也不保证所有配置编译成功。
已有主要证据来自多输入 reduction，没有证实新增 indirect 分支是 Albert 原失败的直接原因。
本次补充文档未运行新的设备测试。

[overview]: model-fix-dsl-explained-20261021.md
[patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-load-liveness.patch
[new-dsl]: ../../../artifacts/cicd/model_fixes_20261020/generated/albert_backward.py
[static]: ../../../artifacts/cicd/four_model_matrix_20261020/analysis/Albert_configs/static_results.json
[dynamic]: ../../../artifacts/cicd/four_model_matrix_20261020/analysis/Albert_configs/results.json
[numeric]: ../../../artifacts/cicd/four_model_matrix_20261020/analysis/Albert_configs/numeric_validation.json
