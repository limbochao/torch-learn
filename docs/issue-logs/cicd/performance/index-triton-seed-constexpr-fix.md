---
title: NPU Index codegen：seed constexpr 常量遗漏修复与性能验证
---

# NPU Index codegen：seed constexpr 常量遗漏修复与性能验证

`NPUIndexTritonKernel.codegen_kernel` 遗漏静态等于 1 的参数常量绑定，使 Electra Train 首层
attention 的 `load_seed_offset` 在 signature 中为 `constexpr`、在 constants 中却没有值。
Index checker 生成 TTIR 失败后进入 no-linear 回退，首层 kernel 从正常约 0.974 ms 增至 16.790 ms。
补齐与社区 codegen 一致的元数据规则后，首层恢复多轴生成，整步设备耗时减少 **14.983 ms / 9.55%**。

本文直接给出错误、关键 DSL、修复规则和性能结果；完整日志与完整 kernel 作为辅助证据。
模型历史分析仍维护在 [Electra 原章节](models/ElectraForCausalLM.md#attention-constexpr-validation)。

## 修复分支与验证版本

| 项目 | 版本 / 范围 |
| --- | --- |
| 分支基线 | 2026-11-04 获取的 `Ascend/pytorch:master`，`6c43e4db2375daa873829c175976602088fe392e` |
| 修复分支 | [`fix/inductor-index-constexpr-constants`][branch] |
| 合并后唯一提交 | [`20c8be6356b7ebd532da050277c8a0ee709d87d2`][commit]，包含修复和回归测试 |
| PR / Issue | [PR !49024](https://gitcode.com/Ascend/pytorch/merge_requests/49024) → `Ascend/pytorch:master`；已结构化关联 [Issue #5855](https://gitcode.com/Ascend/pytorch/issues/5855) |
| 修改范围 | `torch_npu/_inductor/codegen/triton.py` 增加 5 行；新增 3 个元数据回归测试 |
| 性能与模型精度实测 | 2026-11-03，`test/cicd_benchmark@b3bd74035189` 对应安装文件，进程内补齐常量 |
| master 回归 | 在 `pta_213_cicd` 中加载 master 修复前后完整 `codegen_kernel` 方法，执行同一份回归测试 |

**下文性能数值来自 2026-11-03 的临时补丁 A/B，不是重建整个最新 master 后的整网实测。**
两次使用的修复核心均为同一段 `equal_1_arg_indices` 常量绑定；master 回归验证该正式改动的元数据行为。
原 `4515e5af0c`、`ca3407d2e4` 已 rebase 合并为上述单提交；合并前后文件树完全一致，原回归证据仍对应相同源码。
原性能环境另有既存 matmul backward decomposition 修改，两组保持一致，见
[安装文件身份][identity]和[既存差异][decomp]。

## 原因：constexpr 声明缺少对应的值

seed offset 是随机种子数组的下标。本场景的 FX 为 `inductor_lookup_seed(seeds, 1)`，生成 DSL 为：

```python
                tmp27 = tl.load(in_ptr1 + load_seed_offset)
                tmp28 = r3 + 512*x2 + 262144*y1 + 1048576*z0
                tmp29 = tl.rand(tmp27, (tmp28).to(tl.uint32))
```

`in_ptr1` 指向种子数组，`load_seed_offset=1` 选择其中一个元素；真正的随机 seed 仍在运行时读取。
社区 `signature_of` 将静态等于 1 的 `SizeArg` 标记为 `constexpr`，codegen 必须同时提供常量值。

原 Index 路径留下如下矛盾：

```python
signature = {"load_seed_offset": "constexpr"}  # 只展示相关参数
constants = {}
```

下面是精简后的实际错误链，省略个人路径、重复 traceback 和候选配置长列表：

```text
NPUCombinedScheduling.codegen_node
  -> NPUTritonScheduling.codegen_node_schedule
  -> make_ttir_for_check -> _triton_make_ttir -> ast_to_ttir
  -> tmp27 = tl.load(in_ptr1 + load_seed_offset)
                            ^
AttributeError: 'NoneType' object has no attribute 'type'
  -> index codegen failed, falling back to no-linear codegen
```

checker 提前编译多轴候选以检查是否合法。`load_seed_offset` 声明为 constexpr，却没有拿到常量值，
因此无法完成这里的地址表达式；异常被 scheduling 捕获，模型继续运行备用 kernel，整网不会因此直接报错。
失败发生在 Triton 前端，修复入口是 Inductor 元数据构造；还没有进入该失败候选的设备 IR/npuir 优化。
辅助证据：[完整失败记录][before-failures]、[历史 checker DSL 与 metadata][history]。

**原始组最终可执行的 Grid1D kernel 已经含有 `constants={'load_seed_offset': 1}`**，因为备用 codegen
有常量补齐。遗漏发生在回退前的 Index codegen，不能拿最终备用 kernel 的 metadata 代替失败 checker。

## 正式修改与上游对应关系

在 `triton_utils` 的文件级 import 中增加 `equal_1_arg_indices`，并在生成 signature 后、
将 `triton_meta` 交给后续 codegen/checker 前加入：

```python
for arg_num in equal_1_arg_indices(signature):
    triton_meta["constants"][signature[arg_num].name] = 1
```

### `equal_1_arg_indices` 返回什么

它检查 kernel 参数的元信息，返回**表达式为整数 1 的 SizeArg 在 signature 中的位置编号**。
它本身不修改 signature、不读取 tensor 内容，也不填充 constants；后续循环根据这些编号完成绑定。
实际判断要求参数属于 `SizeArg`、表达式属于 Python `int` 或 `sympy.Integer`，且静态等于 1。
因此，尚未还原的符号参数不会仅凭名字或运行时值进入结果。

下面用一个参数表说明：

| signature 位置 | 参数 | 表达式 | 是否返回该位置 |
| --- | --- | --- | --- |
| 0 | `load_seed_offset` | `sympy.Integer(1)` | 是 |
| 1 | `static_size` | `1` | 是 |
| 2 | `dynamic_size` | 符号 `s0` | 否 |
| 3 | `other_offset` | `2` | 否 |

此时 `equal_1_arg_indices(signature)` 返回 `(0, 1)`，修复循环生成：

```python
constants = {"load_seed_offset": 1, "static_size": 1}
```

seed 数组中的随机值仍由 `tl.load` 读取；只有“选择数组哪一项”这个静态下标获得了编译期绑定。
precomputed symbol 必须先还原为整数表达式，随后这条规则才能识别它，这也是修复放置位置与回归用例的依据。

完整修改及回归测试见[修复 patch][patch]和[修复提交][commit]。
社区 CUDA 的 `TritonKernel.codegen_kernel` 已有相同循环；本次补齐的是 NPU Index 路径。
社区参考：
[seed load](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton.py)、
[signature / equal-to-one 判断](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton_utils.py)。

NPU 其他路径的具体位置如下，链接固定到本修复分支的 `20c8be6356`。这三处在本次修复前已存在：

| NPU 路径 | 类 / 函数和源码位置 | 如何获得常量绑定 |
| --- | --- | --- |
| no-linear，正是本问题的备用路径 | [`codegen/triton.py::NPUTritonKernel.codegen_kernel`][npu-no-linear]，第 976 行 | 调用 `super().codegen_kernel()`，由社区父类 `TritonKernel.codegen_kernel` 填充 constants，随后只补充 NPU inductor metadata |
| `triton_experimental` | [`triton_experimental/codegen/triton.py::NPUTritonKernel.codegen_kernel`][npu-experimental]，第 4731–4732 行 | 自己显式执行上面的 `equal_1_arg_indices` 循环 |
| template 的 metadata 生成，`use_jit=False` | [`select_algorithm.py::NPUTritonTemplateKernel.jit_lines`][npu-template]，第 1159–1160 行 | 自己显式执行同一循环；`use_jit=True` 分支提前返回，不经过这里 |

no-linear 对应的调度类为 `NPUNoLinearTritonScheduling`，其 `kernel_type=NPUTritonKernel`；
Index 调度的 `kernel_type=NPUIndexTritonKernel`，见[调度类映射][npu-scheduling]。
因此，“旧路径已有处理”在这里具体指 **no-linear 继承父类的实现**，并非同文件中又显式写了一份循环。
Index 类独立重写了元数据构造，遗漏了父类中的这一步，这也解释了为什么备用 kernel 能正常绑定同一个 offset。

规则处理所有经过该 Index codegen、静态等于 1 的 `SizeArg`，并不限于名字为 `load_seed_offset` 的参数。
值为 0、2 等的 seed offset 及动态参数不由该循环加入 constants；原有随机数公式和 seed 数组选择保持其语义。
常量检查在 precomputed symbol 替换后执行，使已还原为 1 的参数也得到绑定。
无需改动 DSL 数学公式、Triton_ascend 或 npuir。

## 修复前后的完整 kernel

| 对照 | 完整 kernel 名 | 原始生成产物 |
| --- | --- | --- |
| 修复前首层 | `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4` | [DSL + metadata][before-kernel]、[调用位置][before-calls] |
| 修复后首层 | `triton_per_fused__softmax_all_amax_div_isfinit_4` | [DSL + metadata][after-kernel]、[调用位置][after-calls] |
| 两组后续层控制 | `triton_per_fused__softmax_all_amax_div_isfinit_7` | 同两份完整产物中的第二个 kernel |

链接保留完整原始格式，没有重新排版 generated DSL。首层名称中的 `matmul` 为来源标记；
该 kernel 自身执行 softmax / 有限值检查 / dropout 等计算，QK 矩阵乘是前一个独立设备调用。

### 直接对比关键 DSL

下面保留原始 DSL 的局部格式，省略两路 max、有限值检查和其他输出写回。
这里对比的是**最终执行的慢备用 kernel**与**修复后执行的多轴 kernel**；失败的 checker 候选本身没有执行。

修复前，Grid1D 将 batch/head/query 合并为 `x0`，共有 `32×4×512=65536` 行，每行归约 512 个 key：

```python
    xnumel = 65536
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
```

该备用实现的 softmax 归约、seed 读取及 dropout 输出如下。`tmp29` 是经过 mask 的指数值，
`tmp30` 是行和，`tmp37` 是 softmax 概率，`tmp40` 是 dropout 后结果：

```python
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl.load(in_ptr1 + load_seed_offset)
    tmp32 = (r0_1 + 512*x0).to(tl.int32)
    tmp33 = tl.rand(tmp31, (tmp32).to(tl.uint32))
    tmp34 = tl.full([1, 1], 0.1, tl.float32)
    tmp35 = tmp33 > tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = (tmp26 / tmp30)
    tmp38 = tmp36 * tmp37
    tmp39 = tl.full([1, 1], 1.1111111111111112, tl.float32)
    tmp40 = tmp38 * tmp39
    tl.store(out_ptr5 + (r0_1 + 512*x0), tmp40, r0_mask & xmask)
```

修复后，GridNpu 保留 batch=`z0`、head=`y1`、query=`x2`、key/reduction=`r3`，
分别映射和分块；以下是原始 grid/index 和输入读取片段：

```python
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    x2_offset = tl.program_id(2) * X2BLOCK
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:,None]
                x2_mask = x2 < min(X2BLOCK+x2_offset, x2_numel)
                r3 = base_r3[None,None,None,:]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (r3 + 512*x2 + 262144*y1 + 1048576*z0), x2_mask & y1_mask & z0_mask, other=0.0)
```

相同计算在多轴实现中的归约、seed 读取与 dropout 输出如下。`tmp25` 对应 masked 指数值，
`tmp26` 对应行和，`tmp33` 对应 softmax 概率，`tmp36` 对应 dropout 后结果：

```python
                tmp26 = tl.sum(tmp25, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp27 = tl.load(in_ptr1 + load_seed_offset)
                tmp28 = r3 + 512*x2 + 262144*y1 + 1048576*z0
                tmp29 = tl.rand(tmp27, (tmp28).to(tl.uint32))
                tmp30 = tl.full([1, 1, 1, 1], 0.1, tl.float32)
                tmp31 = tmp29 > tmp30
                tmp32 = tmp31.to(tl.float32)
                tmp33 = (tmp23 / tmp26)
                tmp34 = tmp32 * tmp33
                tmp35 = tl.full([1, 1, 1, 1], 1.1111111111111112, tl.float32)
                tmp36 = tmp34 * tmp35
                tl.store(out_ptr5 + (r3 + 512*x2 + 262144*y1 + 1048576*z0), tmp36, x2_mask & y1_mask & z0_mask)
```

这两段都读取同一个 seed 数组下标，计算 `softmax × dropout_mask / 0.9`，并写回完整结果。
修复首先让 Index checker 得到正确的编译期常量，随后恢复正常多轴生成及其调度选择；
它没有删除 softmax 或 dropout 的计算。具体因果链为：

```text
修改前：Index constants={} -> TTIR 检查失败 -> no-linear / Grid1D -> 首层 16.790 ms
修改后：constants={'load_seed_offset': 1} -> 检查通过 -> Index / GridNpu -> 首层 0.974 ms
```

| 生成行为 | 修复前 | 修复后 |
| --- | --- | --- |
| 首层路径 | no-linear 备用生成 | Index 多轴生成 |
| grid 类型 | `Grid1D` | `GridNpu` |
| 迭代域 | `x=65536, r0_=512` | `z0=32,y1=4,x2=512,r3=512` |
| metadata 标记 | `requires_no_linear_block_remap=True` | 按多轴映射生成 |
| seed checker 失败数 | 1 | 0 |
| 全部 Index 失败事件数 | 37 | 36 |

修改后恢复多轴路径及其现有 tiling/autotune 选择，避免运行慢的备用实现。
本实验没有拆分多轴映射、tile 和编译模式各自贡献的比例；确认的因果链是常量绑定补齐后
该 checker 失败消失，首层生成路径变化并获得收益。其他 36 个失败事件不属于本次修复。

## 设备侧性能和精度

测试场景：NPU_A5 的 Ascend950PR、`pta_213_cicd`、物理卡 2；Electra Train、FP32、static、
batch=32、sequence=512、attention shape=`[32,4,512,512]`，关闭 aclgraph。
torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`、transformers `4.36.0`。
每组各采样 10 个稳定 step，以下为设备 kernel duration 求和，不包含编译或 warmup，不使用主机 wall time。

| 项目，均为 ms/step | 修改前 | 修改后 | 减少 |
| --- | ---: | ---: | ---: |
| 首层 `_4`，1 次/step | 16.790405 | 0.974007 | **15.816398** |
| 后续 `_7`，11 次/step 合计 | 10.704739 | 10.704843 | -0.000104 |
| Inductor 全部 kernel 合计 | 156.810495 | 141.827617 | **14.982878** |
| Eager 控制 | 126.474829 | 126.288455 | 0.186374 |

首层耗时下降 **94.20%**，加速 **17.24 倍**；修复后接近后续层单次约 0.973168 ms。
其余 kernel 合计增加 0.833520 ms，因此整网净收益为 **14.982878 ms，Inductor 整步下降 9.55%**。
稳定采样的整步范围分别为 156.761–156.887 ms、141.678–142.120 ms。

每组各自 Eager 计算的编译提升比为 **-23.99% → -12.30%**。
归因修改贡献时固定以原始 Eager=126.474829 ms 为分母，提升比增量为
`100 × 14.982878 / 126.474829 = 11.847` 个百分点，统一分母后为 **-23.99% → -12.14%**。
Eager 的波动没有计入修复收益。本次未重测 GPU，也未实测 LayoutLM 的修复收益。

独立整网 accuracy 为 FP32、batch=32、训练 2 步、原 runner 默认 `tol=0.01`，结果为 **`pass_accuracy`**：
logits 最大绝对误差 1.013279e-6；梯度 204/204 通过、最大绝对误差 8.5565262e-9；
更新后参数 204/204、buffers 2/2 均通过。accuracy 的编译计时不用于性能结论。

证据：[10 步汇总][summary]、[修改前逐步统计][before-steps]、[修改后逐步统计][after-steps]、
[修改后失败记录][after-failures]、[accuracy 完整日志][accuracy]。

## master 回归与复现

回归文件为 [`test/_inductor/test_index_triton_constants.py`][test]。
它调用真实的 `NPUIndexTritonKernel.codegen_kernel`，在 metadata 构造完成后停止，
mock 设备属性读取而不启动设备 kernel。覆盖 seed offset=0/1/2、其他静态 1 与动态参数混合、
precomputed symbol 还原为 1 三类条件。

### 用例中的 `max="2.13"` 是 CI 分流标记

新增用例使用仓库自己的 `@runIfVersion(max="2.13")`，声明的版本范围为 `(-∞, 2.13]`。
`test/version_mark.py` 根据 `CI_VERSION_MIN/MAX` 或 CI 设置的范围决定单个用例是否跳过，
`ci/access_control/version_filter.py` 还会用 AST 在文件层面筛选它；这不是依据 `torch.__version__` 自动判断。

| 执行方式 / CI 目标 | 实际行为 |
| --- | --- |
| 直接运行测试文件、没有配置 CI 版本范围 | 装饰器不跳过，用例照常执行 |
| 版本化 CI 目标恰为 2.13 | 范围相交，保留并运行 |
| 版本化 CI 目标恰为 2.14 | 范围不相交，过滤 / 跳过 |
| CI 没传 `--between_version` | 文件级筛选会排除带版本标记的用例文件 |

这里的上限是新增用例时沿用了同目录内部 API 用例的标记，**不代表已经证明 PyTorch 2.14 不兼容**。
本次只实际验证了 2.13；“只验证过 2.13”本身不能证明应把未来版本排除，也不证明所有较低版本兼容。
常量绑定修复本身没有 `<=2.13` 条件。当前 PR 保留该原始标记，版本范围应按实际 API 兼容性与 CI 分流策略确定。

在完整安装该修复版本的兼容环境中，可直接运行：

```bash
python test/_inductor/test_index_triton_constants.py
```

本次保留原安装版本，使用[方法级回放脚本][metadata-repro]加载 master 修复前后方法执行相同测试；
结果为修复前 **3 项失败、0 error**，修复后 **3 项通过、0 skipped**；安装源码 SHA256 未变化。
见[回归汇总][metadata-result]、[修改前日志][metadata-before]、[修改后日志][metadata-after]和
[提交与文件 SHA256][source-identity]。这验证正式补丁的常量绑定行为，不代表完整 master 构建回归。
模型 A/B 的采样和 accuracy 使用[既有实验脚本][model-repro]，基线选择 `baseline`，
修复组选择 `bind_constexpr`；各组使用独立 cache / debug 目录，保留生成代码。

[branch]: https://gitcode.com/luqichao/pytorch_new/tree/fix/inductor-index-constexpr-constants
[commit]: https://gitcode.com/luqichao/pytorch_new/commit/20c8be6356b7ebd532da050277c8a0ee709d87d2
[test]: https://gitcode.com/luqichao/pytorch_new/blob/20c8be6356b7ebd532da050277c8a0ee709d87d2/test/_inductor/test_index_triton_constants.py
[npu-no-linear]: https://gitcode.com/luqichao/pytorch_new/blob/20c8be6356b7ebd532da050277c8a0ee709d87d2/torch_npu/_inductor/codegen/triton.py#L976
[npu-experimental]: https://gitcode.com/luqichao/pytorch_new/blob/20c8be6356b7ebd532da050277c8a0ee709d87d2/torch_npu/_inductor/triton_experimental/codegen/triton.py#L4731
[npu-template]: https://gitcode.com/luqichao/pytorch_new/blob/20c8be6356b7ebd532da050277c8a0ee709d87d2/torch_npu/_inductor/select_algorithm.py#L1159
[npu-scheduling]: https://gitcode.com/luqichao/pytorch_new/blob/20c8be6356b7ebd532da050277c8a0ee709d87d2/torch_npu/_inductor/codegen/scheduling.py#L158
[patch]: ../../../../artifacts/cicd/performance/index-triton-seed-constexpr-fix/fix.patch
[metadata-result]: ../../../../artifacts/cicd/performance/index-triton-seed-constexpr-fix/result.json
[metadata-before]: ../../../../artifacts/cicd/performance/index-triton-seed-constexpr-fix/before.txt
[metadata-after]: ../../../../artifacts/cicd/performance/index-triton-seed-constexpr-fix/after.txt
[source-identity]: ../../../../artifacts/cicd/performance/index-triton-seed-constexpr-fix/source_identity.json
[metadata-repro]: ../../../../scripts/repro/cicd/performance/fma_seed_validation/validate_index_constants.py
[model-repro]: ../../../../scripts/repro/cicd/performance/fma_seed_validation/README.md
[history]: electra-layoutlm-train-first-attention-20261031.md
[identity]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/installed_identity.json
[decomp]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/decomposition_difference.patch
[before-kernel]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/selected_kernels.txt
[after-kernel]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/selected_kernels.txt
[before-calls]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/wrapper_calls.txt
[after-calls]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/wrapper_calls.txt
[before-failures]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/codegen_failures.json
[after-failures]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/codegen_failures.json
[summary]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/summary.json
[before-steps]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/dynamo_by_step.csv
[after-steps]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/dynamo_by_step.csv
[accuracy]: ../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr_accuracy/run.log
