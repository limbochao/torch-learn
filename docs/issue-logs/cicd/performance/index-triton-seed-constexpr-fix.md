---
title: NPU Index codegen：seed constexpr 常量遗漏修复与性能验证
---

# NPU Index codegen：seed constexpr 常量遗漏修复与性能验证

`NPUIndexTritonKernel.codegen_kernel` 遗漏静态等于 1 的参数常量绑定，使 Electra Train 首层
attention 的 `load_seed_offset` 在 signature 中为 `constexpr`、在 constants 中却没有值。
Index checker 生成 TTIR 失败后进入 no-linear 回退，首层 kernel 从正常约 0.974 ms 增至 16.790 ms。
补齐与社区 codegen 一致的元数据规则后，首层恢复多轴生成，整步设备耗时减少 **14.983 ms / 9.55%**。

本页按修复请求整理分支、原始 kernel 和实测证据；模型历史分析仍维护在
[Electra 原章节](models/ElectraForCausalLM.md#attention-constexpr-validation)。

## 修复分支与验证版本

| 项目 | 版本 / 范围 |
| --- | --- |
| 分支基线 | 2026-11-04 获取的 `Ascend/pytorch:master`，`6c43e4db2375daa873829c175976602088fe392e` |
| 修复分支 | [`fix/inductor-index-constexpr-constants`][branch] |
| codegen 修复提交 | [`4515e5af0c7388374a2ea17599523cac8c4b4c31`][commit] |
| 分支最终提交 | `ca3407d2e4209c62106df0cfdc23b89e1446b987`，包含最终回归测试 |
| 修改范围 | `torch_npu/_inductor/codegen/triton.py` 增加 5 行；新增 3 个元数据回归测试 |
| 性能与模型精度实测 | 2026-11-03，`test/cicd_benchmark@b3bd74035189` 对应安装文件，进程内补齐常量 |
| master 回归 | 在 `pta_213_cicd` 中加载 master 修复前后完整 `codegen_kernel` 方法，执行同一份回归测试 |

**下文性能数值来自 2026-11-03 的临时补丁 A/B，不是重建整个最新 master 后的整网实测。**
两次使用的修复核心均为同一段 `equal_1_arg_indices` 常量绑定；master 回归验证该正式改动的元数据行为。
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
signature = {"load_seed_offset": "constexpr", ...}
constants = {}
```

checker 在 TTIR 生成期间执行到上述 seed load 时出现
`AttributeError: 'NoneType' object has no attribute 'type'`，异常被 scheduling 捕获，进入 no-linear。
见[本次失败记录][before-failures]；[历史 checker 完整取证页][history]保存了失败 checker 的 DSL 和 metadata。
失败发生在 Triton 前端，不能由 traceback 出现在 Triton 就推定为 Triton_ascend/npuir 的优化问题。

**原始组最终可执行的 Grid1D kernel 已经含有 `constants={'load_seed_offset': 1}`**，因为备用 codegen
有常量补齐。遗漏发生在回退前的 Index codegen，不能拿最终备用 kernel 的 metadata 代替失败 checker。

## 正式修改与上游对应关系

在 `triton_utils` 的文件级 import 中增加 `equal_1_arg_indices`，并在生成 signature 后、
将 `triton_meta` 交给后续 codegen/checker 前加入：

```python
for arg_num in equal_1_arg_indices(signature):
    triton_meta["constants"][signature[arg_num].name] = 1
```

完整修改及回归测试见[修复 patch][patch]和[修复提交][commit]。
社区 CUDA 的 `TritonKernel.codegen_kernel` 已有相同循环，NPU 旧 codegen 路径也已有这类处理；
本次补齐的是 Index 路径。社区参考：
[seed load](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton.py)、
[signature / equal-to-one 判断](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton_utils.py)。

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
[commit]: https://gitcode.com/luqichao/pytorch_new/commit/4515e5af0c7388374a2ea17599523cac8c4b4c31
[test]: https://gitcode.com/luqichao/pytorch_new/blob/ca3407d2e4/test/_inductor/test_index_triton_constants.py
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
