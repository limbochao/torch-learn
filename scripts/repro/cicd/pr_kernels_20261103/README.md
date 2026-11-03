# 三个 PR 的 7 个问题

按实际历史触发记录整理，每个问题只保留一个入口，共 **7 个问题、7 份 eager + compile、4 份问题 DSL**。
文件按 PR 和问题编号放在同一个交付目录，可整体复制。
已移除补充 softmax 用例和重复的 GPT2 mask 用例；mask 使用有同环境 PR 前后对照的 split 案例。

| 编号 | PR / 问题 | 实际触发场景与失败特征 | 文件 |
| --- | --- | --- | --- |
| 01 | [!48545](https://gitcode.com/Ascend/pytorch/pull/48545)：多输入 persistent reduction 资源压力 | Albert 参数梯度，81 个输入，R=2048、X=4096；原静态 performance 的 X tile 已是 1，仍 UB overflow | [eager + compile](pr48545/01_persistent_threshold/eager_compile.py)、[问题 DSL](pr48545/01_persistent_threshold/problem_dsl.py) |
| 02 | [!48475](https://gitcode.com/Ascend/pytorch/pull/48475)：连续多轴 sum 布局错误 | BERT bias gradient；旧 DSL 将应保留的 X5 channel 轴并入 R2/R3，错误归约 | [eager + compile](pr48475/02_contiguous_sum/eager_compile.py)、[问题 DSL](pr48475/02_contiguous_sum/problem_dsl.py) |
| 03 | !48475：UB overflow 后遗漏更小 tile | NFNet pooling；原 20 个候选失败，实际 retry 追加 XBLOCK=2048/1024/512/256 后通过 | [eager + compile](pr48475/03_ub_retry_pooling/eager_compile.py)、[实际重放的 DSL](pr48475/03_ub_retry_pooling/problem_dsl.py) |
| 04 | !48475：no-linear 回退未复核融合 | NFNet 派生动态训练图；旧 combined scheduler 在第一组 shape 实际报 unexpected group | [eager + compile](pr48475/04_nolinear_fusion/eager_compile.py) |
| 05 | [!48448](https://gitcode.com/Ascend/pytorch/pull/48448)：persistent reduction 尾部 mask 丢失 | split sizes=(7,16,9)，首段 sum；旧 DSL 生成 r0_mask 但 load/sum 未使用；同环境 PR 前失败、PR 后通过 | [eager + compile](pr48448/05_reduction_mask/eager_compile.py)、[问题 DSL](pr48448/05_reduction_mask/problem_dsl.py) |
| 06 | !48448：symbolic stride 证明异常 | 同源训练图第二组 shape 的 backward；旧 Layout 取模证明实际报 Invalid ranges [0:-1] | [eager + compile](pr48448/06_symbolic_strides/eager_compile.py) |
| 07 | !48448：range split 拒绝协议异常 | 同源训练图在 stride 修复后实际出现 CantSplit 缺参 TypeError；候选异常可能被捕获 | [eager + compile](pr48448/07_range_split/eager_compile.py) |

[extraction_manifest.json](extraction_manifest.json) 记录逐问题来源、输入条件、失败特征和校验值；
[historical_triggers.json](historical_triggers.json) 保存对应历史记录的来源、SHA256 和关键结果摘录。
四份 DSL 保留生成代码原文；03 已替换为实际 retry 重放使用的普通 pointwise DSL，未沿用之前的 index-path DSL。
04、06、07 在目标 DSL 完整生成前失败，按要求只保留 eager + compile 文件。

## 运行

每个 eager 文件默认执行 eager；增加 `--compile` 执行 `torch.compile` 并比较数值。
CPU、CUDA 路径无需 torch_npu；只有 NPU 路径才注册 NPU 设备。

```bash
python pr48475/02_contiguous_sum/eager_compile.py --device cpu --compile
python pr48475/02_contiguous_sum/eager_compile.py --device cuda --compile
python pr48475/02_contiguous_sum/eager_compile.py --device npu:0 --compile
```

在目标 NPU 环境复现时，应使用对应修复前实现，并用新的缓存目录，避免成功缓存跳过问题路径：

```bash
run_dir=$(mktemp -d "$PWD/pr_kernel.XXXXXX")
TORCH_COMPILE_DEBUG=1 \
TORCH_COMPILE_DEBUG_DIR="$run_dir/debug" \
TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" \
TRITON_CACHE_DIR="$run_dir/triton_cache" \
TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
python pr48475/04_nolinear_fusion/eager_compile.py --device npu:0 --compile --index-debug
```

## 各问题的触发条件

**01：保留 Albert 完整规模。** 81 个输入、57 个大矩阵及 12 路梯度贡献全部保留，矩阵输入约 1.78 GiB。
直接执行已分解的 backward ATen 图，返回两个 `[4096]` 参数梯度，不再调用 backward。
Ascend950、R=2048 且保持 persistent 选择开启，才对应 !48545 的高读取压力条件。
`--no-persistent` 仅用于循环归约对照。历史证据证明原 persistent kernel UB overflow；
本次没有重新执行 !48545 父提交/修复提交的 NPU 对照，不能声称本次验证了阈值切换。

**02：保留 channel 轴及两个实际输出。** 输入 `[48,128,64]`，经过 reshape、permute、clone，
返回 `[512,768]` 重排结果和 `[1,768]` sum。检查旧 DSL 的
`tl.reshape(tmp0, [R2BLOCK_SUB* R3BLOCK_SUB* X5BLOCK_SUB])` 与随后 `tl.sum(tmp1, 0)`；
这把保留轴一起求和，和 eager 的按 batch/sequence 归约不同。

**03：重放实际 all-UB 候选集。** 默认输入 `[128,256,72,72]`，保持 FP32 和 dynamic=True。
普通 `--compile` 用于从 ATen 重新 codegen；新版本可能选择其他 kernel，不能据此认定触发了原 retry。
要使用有历史触发记录的 DSL 和普通 NPUCachingAutotuner，执行：

```bash
python pr48475/03_ub_retry_pooling/eager_compile.py --device npu:0 --compile --replay-ub
```

该模式先计算 eager，再加载同目录的实际问题 DSL，关闭其 autotune 缓存，执行原候选预编译。
旧 runtime 会在原始候选全部失败后报错；含修复 runtime 的历史结果为追加 4 个小 tile 后通过，
并完成逐候选 launch 数值检查。脚本打印原始候选、`retry_attempted` 和追加后的候选数。
若安装版本已生成更小候选，或编译器不再触发 overflow，脚本会明确报告原触发条件不成立。
该显式 replay 模式依赖配对 DSL，普通 eager / torch.compile 模式仍可单文件运行。

**04：自然触发 no-linear 回退。** 使用旧 combined scheduler `dc67771ff2` 对应实现，
其余 Python 模块对齐历史验证分支。1×1 conv、stride=32、空间 mean/sigmoid/mul 训练图
在第一组 `(2,16,224,224)` 已实际报 `NotImplementedError: unexpected group`。
使用 `--index-debug` 显示真实捕获的 index 异常，不人为注入 codegen 失败。

**05：保留运行时 split 长度。** FP32 `[32]` 输入和 CPU int64 sizes `(7,16,9)`，
保留 `tolist()`、`capture_scalar_outputs=True` 和 scalar 总长度输出，未替换成固定切片。
2026-10-29 同环境只替换 PR 前后 `codegen/triton.py`：父版本失败，PR 版本和当前安装版本通过。
历史原测试的 expected/actual 为 `-5.140399/-6.655652`；本文件 seed=0 的合成输入不同，
不要求复现相同数值，但保持相同长度和无效尾部参与 sum 的触发条件。

**06：第二组 shape 的真实 backward。** 前置 scheduler 阻塞已解决，再恢复旧 Layout modulo 证明。
同一 compiled callable 依次处理 `(2,16,224,224)`、`(3,16,256,256)`，都执行 backward；
第二组历史实际出现 `ValueRangeError: Invalid ranges [0:-1]`。

**07：观察候选异常，不能只看最终 PASS。** 在 stride 证明已修正、range split 仍为旧实现时，
同源训练图实际记录了 `CantSplit.__init__()` 缺少 `expr/remaining` 的 TypeError。
某些版本捕获该失败候选后继续执行，因此最后数值 PASS 仍可能包含原异常；
用 `--index-debug` 检查异常日志。越界 IndexError 另有直接 helper 输入证据，未声称它阻塞原模型。

04、06、07 共用同一个训练图，但需要分别恢复对应旧实现并解决前置阻塞，
不能在同一套未修复基线上期望一次执行同时到达三个阶段。
原模型、捕获 DSL 和 ATen 子图来自历史 torch 2.13 / torch_npu 环境；版本变化会影响融合和候选路径。

## 验证范围

[cpu_validation.json](cpu_validation.json) 记录所选 7 个问题的 CPU eager / compile 检查。
本次仅在本地 CPU 检查交付文件；没有重新运行远程 NPU/GPU。
历史实际触发证据与本地 CPU 数值检查分别记录，CPU PASS 不替代目标路径触发证明。
新增 `--replay-ub` 的调用依据历史重放流程整理，本轮未在 NPU 上重新执行这个入口。
