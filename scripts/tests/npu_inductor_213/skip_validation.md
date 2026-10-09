# 跳过用例的 NPU 临时适配验证

2026-10-08 完成原基线全部 14 个跳过逻辑用例的环境验证。
首轮为 7 通过、3 失败、4 跳过；第 201 项关闭卷积 HF32 后补测通过，
最终为 **8 通过、2 失败、4 跳过**，对应 16 个有效实例：8 通过、3 失败、5 跳过。
CSV 同时保留首轮结果、补充验证结果和最终临时结果；不覆盖原始 247 项基线。
使用独立目录下的原测试副本，仅调整明确的设备条件和内存检查；没有修改原测试文件或已安装模块。
87 个失败项的修复建议仍属于源码评审，不计入本轮已验证范围。

- [逐项结果 CSV](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation.csv)
- [累计修复评审 CSV](../../../artifacts/npu-inductor-213-batches-20261008/repair-triage.csv)
- [补丁与原文件校验值](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/manifest.json)
- [运行统计](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/summary.json)
- [真实能力和内存检查](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/observations.json)

## 第一批：6 个优先候选全部通过

| 原序号 | 用例 | 临时修改 | 实际结果 |
|---|---|---|---|
| 163 | `test_large_tensor_reduction` | 用 NPU API 检查原 4 GiB 内存门槛 | 原 `2^32` 个 int8 元素及末尾最大值断言通过 |
| 165 | `test_large_pointwise` | 同上，保留 4 GiB 门槛 | 原 `2^31+1` 元素和全量比较通过 |
| 166 | `test_large_offset_pointwise` | 同上，保留 3 GiB 门槛 | 原 `2^31+1` 存储、`2^30` offset 和全量比较通过 |
| 167 | `test_large_strided_reduction` | 同上，保留 2 GiB 门槛 | 原 `2^31+1` 存储、stride=32 和最大值断言通过 |
| 204 | `test_upsample_cat_conv` | CPU-only 条件增加 NPU 例外 | 原模型、输入及 common 数值和低精度检查通过 |
| 245 | `test_float_item_return` | FBCODE 条件增加 NPU 例外；补标量值断言 | fullgraph、标量捕获及返回值 `3.0` 通过 |

四次内存检查均使用当前 NPU 的真实 free/total、进程比例配额和 reserved，保留 cpp-wrapper 的双倍估算。
实际可用内存约为 71–122 GiB，超过原门槛。这里未缩小大索引测试的规模。
建议先提交这 6 项的测试侧适配；本轮证据只覆盖下述已验证环境，不能外推全部版本和设备。

## Conv/BN：两项均可适配，其中一项需要精度设置

第 202 项 `test_conv_functional_bn_fuse` 仅放开 NPU 设备条件后通过，保留 common 原数值与 fp16 检查。
该项没有单独检查融合图，不能仅根据用例名称把数值通过当作融合策略已验证。

第 201 项 `test_conv_bn_fuse` 首轮在第一个 Conv1d 配置的 fp32 比较失败：
输入 `(1,3,112)`、输出通道 32、kernel=1、bias=True、dilation=1、groups=1，
112/3584 个元素不符，最大绝对误差 `0.0013867617`，原允许绝对误差为 `1e-5`。
后续 47 组在首轮未执行，见 [首轮失败报告](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-201.json)。

使用 [精度验证入口](run_conv_precision_validation.py)，仅通过已有
`torch_npu.npu.aclnn.flags(allow_hf32=False)` 在测试进程内关闭卷积 HF32 后，
**原 48 组 1D/2D/3D 配置和 fp16 检查全部通过**，没有修改 dtype、容差、freezing 或模型。
[补测报告](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-201-hf32off.json)
完整记录 setup/call/teardown；能力观察记录证明开关为 `True → False → True`，恢复检查通过。
HF32 是卷积计算允许使用较低有效精度的设置，关闭它可以保留此用例严格的 fp32 比较目标。
因此建议在 NPU 测试分支局部设置并恢复该开关；这不是修复开启 HF32 时的算子误差。

该补测还记录了 96 份生成模块，对应 48 组配置的 fp32/fp16 路径，见
[生成代码证据](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-201-hf32off-generated.json)。
生成代码使用冻结后的权重/偏置调用 convolution，没有独立 BatchNorm 调用，支持本次融合路径的判断。

## 动态 Linear：解除门槛后仍有两个阻塞

第 197 项 `test_linear_dynamic_maxautotune` 保留 `max_autotune=True`、TRITON-only 和动态输入后失败。
[完整报告](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-197.json)
显示，AOT joint graph 优化的 `pad_mm` 测时调用默认设备推断，CUDA 迁移别名与 NPU 同时被认为可用，
触发 `len(avail_gpus) <= 1` 断言。这对应已评审的 R04，不能通过关闭 padding 或被测模式作为功能修复。

此外，[已生成的动态前向代码](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-197-generated-1.txt)
实际调用 `torch.ops.aten.addmm.default(...)`，没有 Triton GEMM 模板。
当前安装的 `torch_npu/_inductor/kernel/mm.py::tuned_addmm` 在
`not (static_shape_tmp and is_nonzero_tmp)` 时直接调用 `fallback_handler(aten.addmm.default)`，
发生在模板候选构造之前。相关安装文件校验值和判断分支见
[源码证据](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/source-evidence.json)。
同时，[另一份反向产物](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-197-generated-2.txt)
确实生成了 Triton `mm` 模板；这里的缺口特指动态前向 `addmm`，不代表所有矩阵乘都回退。

因此，修好 benchmark 别名问题也不等于已经覆盖原 Triton-only 目标。
该项需要继续适配动态 `addmm` 模板，并用生成代码确认实际路径，暂不列为仅改 skip 就能完成的修复。

## progressive：两个实例均被子进程序列化兼容阻塞

第 96 项的 `TestSubprocess` 和 `GPUTests` 均失败，见
[逐实例报告](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/case-96.json)。
后台子进程恢复 `LoweringSerializer` 状态时调用 `lowering.force_fallback(k)`，
但 `k` 不满足 `OpOverload` 类型要求，触发 `Only OpOverload to make the clean up easier`。
这是已评审 R22 的实际复现，首错尚未记录具体违规 key，不能直接断言删除哪个注册就能修好。

当前没有完成优化版切换、计数和性能断言，也没有发生原 300 秒限制的超时。
需要先适配 NPU lowering 与社区 `compile_fx_ext.py` / `lowering.py` 的序列化协议，
可能涉及允许范围之外的社区模块。不能跳过状态恢复或关闭 progressive 来记为通过。

## 仍跳过的四项

| 原序号 | 用例 | 实际结果与建议 |
|---|---|---|
| 50 | `test_ensure_fp4_dtype_registered` | 缺少真实 `cutlass_api`，仍因依赖条件跳过；没有安装或模拟依赖 |
| 42 | `test_tma_lookup_table_entry` | mm/addmm 两个参数化实例都因缺少 TMA 支持跳过 |
| 45 | `test_multiple_configs_same_template` | 同上，不能仅把 TMA 条件改为真 |
| 46 | `test_mixed_template_configs` | 同上；NPU 等价的多模板 lookup 接入仍需设计 |

第 50 项即使后续具备依赖并通过，也只证明主机端类型映射，不能证明 NPU FP4 kernel。
第 42/45/46 项仍保留真实能力限制，没有把兼容 capability 改为 9.0。

## 运行环境与复现

环境为 Ascend950PR_9579，torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0`。
使用当前环境默认 NPU Inductor 后端，保留已有安装状态，不假定安装内容与提交完全一致。
每个逻辑用例独立进程，串行运行；NPU 当前卡为 0，全部 8 卡可见，编译和 NPU 预编译线程均为 4。
`run_batch.py` 选择原生 NPU 实例及中立测试类，保留 setup/call/teardown 和所有收集到的参数化实例。

准备独立副本目录 `<overlay>/test/inductor`，从原目录复制以下 5 个文件：

```text
test_torchinductor.py
test_torchinductor_dynamic_shapes.py
test_compile_subprocess.py
test_utils.py
test_lookup_table.py
```

校验原文件 SHA256 与 manifest 一致，再在副本目录应用归档的三个 `.patch`（`patch -p1`）。
复制本目录的 [run_batch.py](run_batch.py) 和 [skip_validation_support.py](skip_validation_support.py)。
为避免动态测试导入原目录中未适配的模块，在副本目录建立 `__init__.py`：

```python
import os
__path__.append(os.environ["NPU_SKIP_ORIGINAL_TEST_DIR"])
```

在副本目录运行；以下路径变量应指向容器工作目录内的位置，`NPU_TEST_PYTHON` 应指向目标环境 Python：

```bash
timeout --signal=TERM --kill-after=30s 1800 env \
  TORCH_TRANSFER_TO_NPU=1 TORCH_NPU_DEVICE_CAPABILITY=8.0 \
  TORCHINDUCTOR_COMPILE_THREADS=4 PYTORCH_TEST_WITH_SLOW=1 \
  NPU_SKIP_ORIGINAL_TEST_DIR="$NPU_ORIGINAL_TEST_DIR" \
  NPU_SKIP_OBSERVATIONS="$NPU_OVERLAY/observations.jsonl" \
  TORCHINDUCTOR_CACHE_DIR="$NPU_OVERLAY/cache" TMPDIR="$NPU_OVERLAY" \
  "$NPU_TEST_PYTHON" run_batch.py --case-names --precompile-workers 4 --device-index 0 \
  --output "$NPU_OVERLAY/result.json" test_torchinductor.py::test_large_tensor_reduction
```

其他用例替换最后的 `文件::用例`；第 197、96 项额外设置 `TORCH_LOGS=output_code`。
第 201 项补测复制 `run_conv_precision_validation.py` 到同一副本目录，用它替换上述入口，
并额外设置 `TORCHNPU_PRECOMPILE_THREADS=4`、`TORCH_LOGS=output_code`；缓存使用独立的 `precision_cache`。
原 48 组配置在同一个进程中完整执行，原测试文件内容与首轮一致。
1800 秒是进程上限，progressive 原用例内部的 300 秒限制和性能断言仍保留。
没有安装 CUTLASS、伪造 TMA 能力、修改 dtype/容差或删除性能检查。

验证结束后，5 个原测试文件的 SHA256 均与预检一致，cgroup 的 `oom_kill` 计数仍为 3，没有新增 OOM。
见 [结束校验](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/postflight.json)。
测试副本和证据保留用于复现；没有替换已安装模块，临时精度设置已恢复。
