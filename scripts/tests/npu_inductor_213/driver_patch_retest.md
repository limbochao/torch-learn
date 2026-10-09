# 完整 PR 加回原 Triton driver patch 的影响验证

本轮之后的 100 个未全通过项已继续评审和修复；最新结论见 [剩余用例修复报告](remaining_repairs.md)。
下文保留本次 driver 实验的原始统计。

2026-10-09 已完成 247 项原用例及 14 项适配复测。加回补丁后，17 项原回归全部恢复通过，
适配后的 `test_upsample_cat_conv` 也从失败转为通过；本轮范围内未发现新增功能回归。
驱动误选得到修复，但其余失败、导入顺序遗漏和重复构造 driver 的行为仍需分别处理。

本轮在 PR #47384 的完整 HEAD `16437be0ffc9d67c8b27855a234243e35385c58b` 上，
加回基线中的 `_patch_triton_driver()` 及 `_init()` 调用。
函数体与原安装版本一致：直接选择 Ascend driver，并将上游 `set_driver_to_gpu` 替换为该函数。
PR 自带的 NVIDIA 禁用和 benchmark 默认设备适配继续保留。

- [本轮补丁](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/restore-driver.patch)
- [安装文件校验值](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/installation.json)
- [247 项总表及通过对应修改](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/results.csv)
- [14 项适配用例对比](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/skip-validation.csv)
- [最终统计](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/progress.json)
- [覆盖及一致性检查](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/validation.json)

## 最终结果

| 范围 | PR 原样 | PR 加回 driver patch | 变化 |
|---|---|---|---|
| 247 项原用例 | 130 通过、99 失败、14 跳过、4 项含预期失败 | 147 通过、82 失败、14 跳过、4 项含预期失败 | 17 项转为通过，230 项状态不变 |
| 14 项适配复测 | 8 通过、2 失败、4 跳过 | 9 通过、1 失败、4 跳过 | 1 项转为通过，13 项状态不变 |

4 项含预期失败具体为 2 项部分通过/含预期失败、2 项预期失败。
原用例展开为 283 个有效实例：167 通过、94 失败、16 跳过、6 xfail；
适配复测展开为 16 个有效实例：9 通过、2 失败、5 跳过。
实例名称与上轮完全一致，逐实例比较未发现原先通过的参数组合退化。
14 项适配复测单独统计，不加进 247 项原用例的通过数。

总表的“本轮验证修改”说明实际安装的 PR 和 driver patch；“通过变化类型”“通过对应修改”
区分基线已通过、采用 PR 后通过、加回 driver patch 后恢复，以及尚未全部通过。
“通过修改证据”指向对应补丁。14 项适配复测并列在原序号行，明确 skip 条件调整、
HF32-off 配置、适配结果及日志，不将适配版通过写成原用例通过。

恢复的 17 项为 `arange1–7`、`arange9`、`builtins_round_int_ndigits_pos`、
`builtins_round_int_ndigits_zero`、`div6`、`div_precision`、`linspace1`、`nan_to_num`、
`one_hot`、`tensor3` 和 `to_device_constant`。
[定向阶段报告](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/targeted-results.json)
保留这些实例的完整 setup/call/teardown。

适配后的 `test_upsample_cat_conv` 原先报 `libcuda.so cannot found!`，本轮原断言通过；
对应[适配报告](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/skip/case-204.json)。
仍失败的 `test_progressive` 两个实例均维持 `Only OpOverload to make the clean up easier` 错误。
第 201 项默认配置仍有精度失败，沿用此前 HF32-off 配置后通过，不归功于本补丁。

原用例 94 个失败实例中，93 个异常首行与 PR 原样一致；唯一变化为下面的
`mark_unbacked_slice`。本轮原用例报告中已没有 `libcuda.so cannot found!`，
但不能据此推断所有调用方式均已覆盖，导入顺序探针给出了反例。

## 验证范围

沿用 torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0` 和 Ascend950PR_9579。
17 个原基线通过、PR 后失败的用例先单独执行，并计入 247 项；其余 230 项按原批次执行。
第 93–95 项继续以独立进程结果为准，避免注册清理影响。
14 项 skip 适配沿用之前的测试副本和断言，第 201 项最终采用相同的 HF32-off 配置。
原测试输入、容差和断言不变。

启动参数包含 `TORCH_TRANSFER_TO_NPU=1`、`TORCH_NPU_DEVICE_CAPABILITY=8.0`，
编译线程和 NPU 预编译线程均为 4，使用本轮的新缓存目录。
capability 参数只用于兼容社区测试的条件判断，不表示 NPU 硬件能力。
测试批次并行执行，部分定向测试和第 0 队列共用 NPU 0；多设备批次等待其余队列完成。
本轮用于功能回归，不用于比较耗时或证明没有性能开销。

本轮保留 PR 原有 5 项相对旧环境的改善，因此 147 个通过项比旧环境的 142 项多 5 项。
通过数包含 CPU、负向断言及主机逻辑等测试；各行的实际 NPU 覆盖范围仍以 CSV 为准。

## 仍失败的后续路径

`test_mark_unbacked_slice_npu` 在 PR 原样下于加载生成模块时进入 NVIDIA driver，报 `libcuda.so cannot found!`。
加回补丁后越过该处，随后在 `check_caching_allocator_for_cudagraphs()` 中访问
`torch._C._cuda_cudaCachingAllocator_is_enabled()`，报缺少属性。
用例使用 `torch.compile(mode="reduce-overhead")`，会进入 cudagraph 检查；
迁移后的 `torch.cuda.is_available()` 为真，但 CPU 版 torch 没有对应 CUDA C++ 接口。
这是同一失败用例暴露出的后续兼容性问题，不能记成恢复通过，也不能记成原来通过的用例新增回归。

证据保存在本轮与原 PR 的 `batch-14-results.json`；
[失败对比](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/failure-comparison.json)
记录具体节点和前后错误。

## 驱动和 worker 证据

[主进程与新进程探针](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/driver-probe.json)
确认 helper 指向 `_patch_triton_driver()`，连续调用都保持 NPUDriver 类型。
但每次都会创建不同的 driver 实例，原补丁没有复用已有实例。

[真实编译 worker 探针](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/worker-probe.json)
使用 PyTorch `SubprocPool`，确认编译 worker 同样获得替换后的 helper。
这项结论依赖启动时的迁移环境变量；仅在主进程替换函数不等于 worker 已获得修复。

[arange 图补充验证](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/arange-artifact.json)
保存 float32/float16 输入的生成模块和 eager 对比结果。
这两个独立小例子均未生成 `set_driver_to_gpu()` 调用，不能单独作为原失败路径被覆盖的证据。
原始用例是否恢复，以 `targeted-results.json` 中完整的 setup/call/teardown 为准。

## 导入顺序遗漏

[导入顺序探针](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/import-order-probe.json)
确认另一种调用方式仍未修复：

1. 以 `TORCH_TRANSFER_TO_NPU=0` 启动，先导入 NPU 的 `triton_helpers`。
2. 手动导入 `transfer_to_npu`，启用迁移和原 driver patch。
3. 调用上游 helper 成功，但调用先前导入的 NPU helper 仍报 `libcuda.so cannot found!`。

NPU helper 通过 `from torch._inductor.runtime.triton_helpers import *` 保存函数引用。
之后替换上游模块属性不会更新已经保存的引用，旧函数仍执行 NVIDIA 兜底判断。
这是原补丁未覆盖的调用顺序，不是相对 PR 原样新增的回归。

复现脚本：

```bash
TORCH_TRANSFER_TO_NPU=0 python probe_driver_import_order.py --output import-order-probe.json
```

## 重现与汇总

在目标容器的原 Inductor 测试目录，安装本轮补丁后可运行原用例：

```bash
export TORCH_TRANSFER_TO_NPU=1 TORCH_NPU_DEVICE_CAPABILITY=8.0
export TORCHINDUCTOR_COMPILE_THREADS=4 TORCHNPU_PRECOMPILE_THREADS=4
python <torch-learn>/scripts/tests/npu_inductor_213/run_batch.py \
  --case-names --precompile-workers 4 --device-index 0 \
  --output <output>/arange1.json test_torchinductor.py::test_arange1
```

补充图和 worker 验证使用 [probe_restored_driver.py](probe_restored_driver.py)。
从 torch-learn 根目录重新生成对比表：

```bash
python scripts/tests/npu_inductor_213/summarize_driver_retest.py \
  --raw-dir artifacts/npu-inductor-pr47384-restored-driver-20261009 \
  --output-dir artifacts/npu-inductor-pr47384-restored-driver-20261009
```

## 验证收尾

本轮与原 PR 验证的 176 份原始测试文件、196 份 NPU Inductor 文件及 5 份适配测试源码一致。
运行前后检查的 223 份安装源码、176 份原始测试文件、9 份适配目录文件均无变化。
所有有效报告中的迁移文件校验值与本轮补丁相符；未通过修改测试输入、容差或断言改变结果。

全部队列、独立进程补测及多设备批次结束后，已将容器中的 `transfer_to_npu.py`
恢复为 PR 原文件，SHA-256 与安装前一致；完整 PR checkout 保持干净，未残留本轮验证进程。
详情见[环境恢复检查](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/postflight.json)。
补丁和去敏报告已保留，可按需应用；本轮没有更新远程 PR。

本地 Python 语法、脚本和文档的 `git diff --check`、报告链接及个人路径扫描均通过。
使用已归档输入重新汇总后，51 份产物逐字节一致，可复现 CSV 和 JSON 统计。
