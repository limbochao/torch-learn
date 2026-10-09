# PR #47384 迁移文件的 247 项重测

本轮按 [PR #47384](https://gitcode.com/Ascend/pytorch/pull/47384) 的提交
`16437be0ffc9d67c8b27855a234243e35385c58b`，替换验证环境已安装的
`torch_npu/contrib/transfer_to_npu.py`。测试范围为原始 247 行清单，另复测此前 14 个跳过项的临时适配版本。
两组结果单独统计，原基线保留。

本报告的运行范围是仅替换迁移文件。2026-10-09 已另外完成 PR 全部 6 个提交及 3 个净变更文件的同步和核对，
见 [完整 PR 核对与驱动复现](full_pr_review.md)。迁移文件与完整 HEAD 字节一致，另外两处变更为 contrib 测试 skip。
本报告的 247 项统计没有冒充为完整同步之后重新执行的结果。

全部测试于 2026-10-08 23:09（北京时间）结束，2026-10-09 完成剩余证据核对和归档。
247 项均已分析，无待测试或待分析项。主测试相对基线 **5 项转为通过、17 项回归**。

| 统计口径 | 基线 | PR 完整迁移文件 |
|---|---:|---:|
| 通过 | 142 | 130 |
| 失败 | 87 | 99 |
| 跳过 | 14 | 14 |
| 部分通过且含预期失败 | 2 | 2 |
| 预期失败 | 2 | 2 |
| 参数化有效实例 | 283 | 283 |

283 个实例为 150 passed、111 failed、16 skipped、6 xfail。
130 个通过项包含 CPU、主机端、负向及测试适配路径，不等于 130 项 NPU 功能正向验收通过。
第 93–95 项以独立进程补测为最终结果，原批量报告保留，但不重复计数。

- [本轮逐项对比 CSV](../../../artifacts/npu-inductor-pr47384-20261008/results.csv)
- [247 项测试目的和 NPU 覆盖建议](../../../artifacts/npu-inductor-pr47384-20261008/test-objectives.csv)
- [14 个跳过项的适配复测](../../../artifacts/npu-inductor-pr47384-20261008/skip-validation.csv)
- [分批进度与实例统计](../../../artifacts/npu-inductor-pr47384-20261008/progress.json)
- [完整性与环境校验](../../../artifacts/npu-inductor-pr47384-20261008/validation-summary.json)
- [PR 版本信息](../../../artifacts/npu-inductor-pr47384-20261008/pr-metadata.json)
- [安装前后校验](../../../artifacts/npu-inductor-pr47384-20261008/installation.json)
- [相对安装前的完整差异](../../../artifacts/npu-inductor-pr47384-20261008/installed-transfer.patch)
- [PR 自身差异](../../../artifacts/npu-inductor-pr47384-20261008/pr-transfer.patch)
- [原始基线](../../../artifacts/npu-inductor-213-batches-20261008/results.csv)

## 版本和统计口径

环境为 torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0`、Ascend950PR_9579。
沿用默认 NPU Inductor backend，原测试文件不改动，编译及 NPU 预编译线程均为 4。
使用新缓存目录，设置 `TORCH_TRANSFER_TO_NPU=1` 和 `TORCH_NPU_DEVICE_CAPABILITY=8.0`。
后者仅用于社区 CUDA capability 比较，不代表 NPU 硬件能力。

安装文件 SHA-256 为 `d40b55f8358273e51d58a897c49fa57298977dae9c81b4aed0876494db83a92a`，
与上述 PR 提交中的完整目标文件一致。仅替换该文件，不代表应用 PR 内其他文件的改动。
每份原始报告记录实际版本、默认设备、源码校验值和 PR 提交；不以 wheel 版本代替源码标识。

CSV 以原始清单的 247 行为单位，参数化实例另列。完成的实例必须有完整阶段记录；
skip、xfail、CPU 专用检查、负向测试和未完成实例不计为 NPU 正向通过。
`状态不变` 只表示结果类别相同，首错和覆盖范围仍逐项复核。

`测试目的` 描述源码试图验证的行为，不代表本轮已执行到该检查。
`NPU覆盖建议` 根据功能契约和后端适用范围给出建议，不以当前 pass/fail 决定删测。
基础数值、别名和索引边界建议保留；static/fast launcher、lookup、异步编译等按承诺的功能范围验收；
CUDA 资源、NVTX、CUTLASS 和固定代码字符串应转换成 NPU 等价覆盖。
大规模/穷举用例可安排定期回归，但不应通过缩小关键边界改变原测试目的。
每项均提供 `目的源码证据`；源数据保存在 `objectives.json`。

表内术语：

| 术语 | 含义 |
|---|---|
| launcher | 将参数和 stream 传给设备 kernel 的调用器；static/fast 是特定调用实现，数值通过不证明它被使用 |
| reinplace / functionalization | 将原地更新改写成函数式表示，再在别名和生命周期允许时恢复原地更新，以减少复制 |
| Meta | 不执行真实计算、只描述输出形状和 stride 等信息的算子实现；其描述必须与真实输出一致 |
| backed / unbacked 符号维度 | 前者来自输入 shape 等可获得提示值的信息；后者来自 item/nonzero 等数据依赖结果 |
| autotune / lookup | 前者试测候选实现后选优；后者按设备、算子和输入 key 查询预设配置 |
| TRITON-only | 配置要求使用 Triton GEMM 候选；只调用 ATEN 回退不能证明该目标已覆盖 |
| TMA / NVTX / CUTLASS | 分别是 CUDA 专项的数据传输能力、性能标注接口、模板计算库，不能仅改设备名称验收 NPU |
| freezing / HF32 | 前者将推理中不变的参数折叠以优化图；后者是 NPU 卷积允许较低有效精度的设置 |

## 驱动选择回归

PR 将 `CudaDriver.is_active` 改为恒 False，但当前 PyTorch 的
`torch/_inductor/runtime/triton_helpers.py::_is_backend_active` 在该检查为 False 后还有兜底分支：

```python
if name == "nvidia":
    import torch
    return torch.cuda.is_available() and torch.version.hip is None
```

迁移模式下 `torch.cuda.is_available()` 是 NPU 可用性的别名，返回 True；CPU torch 的 `hip` 为 None。
因此该 helper 仍把 NVIDIA 后端判为可用，并在 `set_driver_to_gpu` 中尝试构造 `CudaDriver`。
生成模块导入或编译 worker 随即因 `libcuda.so cannot found!` 失败。

[独立机制探针](../../../artifacts/npu-inductor-pr47384-20261008/driver-probe.json) 实测：
`CudaDriver.is_active() == False`、已注册的 driver 类确实是被修改的类、
`_is_backend_active("nvidia", ...) == True`；主动调用 `set_driver_to_gpu()` 复现缺少 libcuda 的断言。
这排除了“补丁没有安装”和“patch 到另一份 driver 类”两种解释。
该探针未修改生产实现，记录中包含实际安装源码及行号。

原安装版本同时适配了 `triton_helpers.set_driver_to_gpu`；换成 PR 完整文件后该适配消失。
因此，PR 的 driver 自检修改不足以适配当前 PyTorch helper 的兜底行为。
修复方向是让迁移模式下的完整 driver 选择流程及编译 worker 都保持实际 NPU 后端。
这涉及 `torch_npu/contrib/transfer_to_npu.py`，超出仅改测试或 `torch_npu._inductor` 的范围。

17 个从通过转为失败的原序号为：102–110、116、137、138、141、142、148、159、227。
全部在本轮报告中出现相同驱动错误；未在重测中加入额外修复，因此这些结果对应指定 PR 文件本身。

## Benchmark 默认设备

同一探针确认 `_get_default_gpu_device_type()` 返回 `npu`，PR 的默认设备修改已经生效。
单个用例能否通过仍以原断言为准：即使绕过原来的 CUDA/NPU 重复计数断言，
也可能继续遇到驱动选择、benchmark 配置契约或测试预期不匹配。

| 原序号 | 转为通过的用例 | 实际覆盖 |
|---|---|---|
| 74 | `test_mlp` | 原三次调用完成；测试未显式比较数值 |
| 183、184 | `test_linear1`、`test_linear2` | 原 common 前向数值及低精度比较通过 |
| 240 | `test_equivalent_backed_unbacked` | 符号维度统一后的 BMM fullgraph/eager 比较通过 |
| 241 | `test_unbacked_linear_layer_norm_input` | 数据依赖维度下三个前向输出的 fullgraph/eager 比较通过 |

同样失败的项也存在首错变化：第 82 项 profiler 现在因 NPU profile 不具有 `profiler` 属性而失败，
见 [profiler 源码证据](../../../artifacts/npu-inductor-pr47384-20261008/profiler-source-evidence.json)。
第 83 项 Conv/BN 已进入数值比较，520/6016 个元素超差；本项未做 HF32 对照，不能套用第 201 项的归因。
第 187 项通过首个小 K BMM 数值比较后，生成代码仍有 `extern_kernels.bmm`，未满足分解策略断言，
见 [完整生成模块](../../../artifacts/npu-inductor-pr47384-20261008/case-187-generated.txt)。

## 独立扩展测试和跳过项复测

第 93–95 项在批量执行时会受到 runtime 注册清理影响，因此各自在新进程中完成补测。
第 93 项仍是 PrivateUse1 已被 NPU 注册；第 94 项出现异步 `aclnnInplaceCopy` 错误，
第 95 项仍报告 AIVEC error 334（非法 GM 地址或跨设备访问超时）。
异步异常出现的位置不能直接确定故障 kernel；尚无证据支持给出简单的通用修复。
第 94 项的基线 worker 导入问题也不能据此判为已解决。
[结果替换映射](../../../artifacts/npu-inductor-pr47384-20261008/overrides.json) 指向三份完整独立报告。

14 个原跳过项使用与基线完全相同的测试副本和适配补丁，未缩小 shape、调整容差或模拟 TMA 能力。
首轮为 7 通过、3 失败、4 跳过；第 201 项固定采用 HF32-off 补测配置后，
最终为 **8 通过、2 失败、4 跳过**，对应 16 个有效实例：8 passed、3 failed、5 skipped。
补测首轮和最终配置均在 CSV 中保留，独立于 247 项主测试统计。

| 原序号 | 本轮结果 | 相对基线适配结果及覆盖边界 |
|---|---|---|
| 163、165、166、167 | 通过 | 原大索引/stride/offset 边界全部保留，使用真实 NPU 内存检查 |
| 197 | 通过 | 原先失败；动态 Linear 原断言通过，前向仍走 ATEN `addmm` |
| 201 | 首轮失败，补测通过 | 与基线一致；局部关闭卷积 HF32 后通过 48 组配置及 fp16 检查 |
| 202、245 | 通过 | 与基线一致；分别覆盖 Conv/functional BN 数值和捕获标量返回值 |
| 204 | 失败 | 原先通过；解除 skip 后被本轮 CUDA driver 误选阻塞 |
| 96 | 失败 | 两个 progressive 实例仍被 lowering 状态恢复的 OpOverload 类型约束阻塞 |
| 42、45、46、50 | 跳过 | 真实 TMA 能力或 CUTLASS 依赖仍不满足 |

第 197 项 [8 份生成模块](../../../artifacts/npu-inductor-pr47384-20261008/skip/case-197-generated.json)
包含 5 份调用 `aten.addmm` 的前向和 3 份含 Triton mm 模板的反向。
因此原断言通过不代表动态前向 TRITON-only 目标已实现；仍需处理 addmm lowering 的 fallback 分支。
对应源码已在基线 [安装源码证据](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation/source-evidence.json)
中保留，本轮结束校验确认该源码没有改动。

第 201 项首轮仍为 112/3584 个元素超差，最大绝对误差约 0.001387；补测没有改变输入或原容差。
[96 份生成模块](../../../artifacts/npu-inductor-pr47384-20261008/skip/case-201-hf32off-generated.json)
对应原 48 组配置的 fp32/fp16 路径。
[能力与精度观察](../../../artifacts/npu-inductor-pr47384-20261008/skip/observations.json)
记录 HF32 开关 `True → False → True`，恢复检查通过。

## 结束校验

预检记录的 176 个原测试文件、196 个已安装 NPU Inductor Python 文件的 SHA-256 均未变化；
本清单涉及的 15 个测试文件也已与基线逐一核对。
5 个适配测试副本的 SHA-256 与基线 manifest 一致，基线记录的 120 个文件内容未变。
安装的迁移文件仍与指定 PR 完整文件一致；没有额外修改测试断言或后端来改善通过率。
cgroup `oom_kill` 仍为 3，本轮没有新增 OOM，所有目标测试和队列进程均已退出。
详见 [结束源码校验](../../../artifacts/npu-inductor-pr47384-20261008/postflight.json)。

实际执行器以 [文本证据](../../../artifacts/npu-inductor-pr47384-20261008/executed-runner.txt) 保留。
仓库执行器另增加默认 benchmark 设备查询的异常记录保护，
见 [差异](../../../artifacts/npu-inductor-pr47384-20261008/runner-diff.patch)；
本轮查询成功返回 `npu`，测试选择和断言没有变化。
各批 `*-invocation.json` 保留命令、起止时间和实际设备索引（路径已去敏）。

## 复现

在 Docker 内的原 Inductor 测试目录激活上述 Python 环境，`LEARN_REPO` 指向 torch-learn，
`RETEST_OUTPUT` 指向容器工作目录内的新输出目录，使用归档的 PR 文件内容安装后运行：

```bash
test -f /.dockerenv
mkdir -p "$RETEST_OUTPUT"
export TORCH_TRANSFER_TO_NPU=1 TORCH_NPU_DEVICE_CAPABILITY=8.0
export TORCHINDUCTOR_COMPILE_THREADS=4 TORCHNPU_PRECOMPILE_THREADS=4
export NPU_TRANSFER_PR_HEAD=16437be0ffc9d67c8b27855a234243e35385c58b
export TMPDIR="$RETEST_OUTPUT" TORCHINDUCTOR_CACHE_DIR="$RETEST_OUTPUT/cache"
python "$LEARN_REPO/scripts/tests/npu_inductor_213/run_batch.py" \
  --case-names --precompile-workers 4 --device-index 0 \
  --output "$RETEST_OUTPUT/arange1.json" test_torchinductor.py::test_arange1
```

[批次计划](../../../artifacts/npu-inductor-pr47384-20261008/plan.json) 保留全部行号和请求参数。
第 1 批使用精确 nodeid，不加 `--case-names`；其余批次按方法选择全部对应设备实例。

从 torch-learn 根目录重新生成 CSV：

```bash
python scripts/tests/npu_inductor_213/summarize_pr_retest.py \
  --raw-dir artifacts/npu-inductor-pr47384-20261008 \
  --output-dir artifacts/npu-inductor-pr47384-20261008
```

人工审核结论保存在 `review.json`；汇总器检查完成实例与 PR 校验值，不将旧基线结果当作本轮实测。
适配复测结论保存在 `skip-review.json`；适配方法和三个补丁见基线 [复现说明](skip_validation.md)。
