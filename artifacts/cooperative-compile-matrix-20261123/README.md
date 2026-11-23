# Compile / DSL 多卡测试包的小范围验证

交付入口为 [cooperative-compile-matrix.tar.gz](cooperative-compile-matrix.tar.gz)，
使用方法见[多卡一键测试](../../scripts/tests/cooperative_reductions/COMPILE_MATRIX_README.md)。
设备上执行的采集器、DSL helper 和配置记录器与交付包逐文件 SHA256 一致。
`bundle_manifest.json` 记录包内文件摘要与源码提交，压缩包摘要见相邻 `.sha256` 文件。

本目录验证测试流程、精度过滤、配置及产物保存；全量性能仍待用户在目标机器采集。
既有 A5/L20 固定 BLOCK 矩阵继续位于独立目录，不与这些 compile 数据拼接。

## 验证范围与结果

| 项目 | CUDA | NPU |
| --- | --- | --- |
| 环境 | GPU_4090，`pytorch-dev-gpu67`，RTX 4090 两卡 | NPU_A5 Docker，`pta_213_case`，物理卡 2 |
| PyTorch | `2.13.0+cu126`，提交 `cf30153c4c13` | `2.13.0+cpu`，同一 PyTorch 提交 |
| Triton / torch_npu | Triton `3.7.1`，CUDA `12.6` | Triton `3.6.0`，torch_npu `2.13.0+git93d7d01` |
| NPU 编译器 | — | BiShengIR 2.0.0，NPUIR `395c9c5646f9`，LLVM 19.1.7 |
| 操作 / 形状 | sum、amax；X=1；R=4K、32K | sum、amax；X=7；R=4K |
| DSL 搜索 | GPU 由 Inductor 选择配置 | RS=1/8/16 × BLOCK=512/2048；origin/debug 独立搜索 |
| 独立复测 | 每组 2 轮，plain/cooperative 交替 | 每组 2 轮，plain 与入选 DSL 配置交替 |
| 主结果 | 8/8 PASS | 6/6 PASS |
| 全候选 | 16/16 PASS | 36 条：32 PASS、4 个预期的超核 debug WRONG |
| 自然选择 | 4/4 PASS；R=4K 普通路径、R=32K cooperative | 不测试尚未接入的 cooperative 选择 |

CUDA 的完整形状组轮流分给两张卡，同一形状的 plain、cooperative 与自然选择始终在同卡执行。
两张卡分别对应物理卡 6/7，UUID 为 `GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7` 和
`GPU-19ae2c1b-628b-0861-d177-a8fe4d3894a4`；每张卡各产出 4 个主结果。
再次执行相同参数加 `--resume`，没有重新执行形状组，直接汇总已有 8 个主结果。

NPU 的有效核限额为 56 Vector。X=7、RS=8 时恰好 56 个 program，入选 debug 配置检查通过。
RS=16 时有 112 个 program，两个 BLOCK、两种操作的 debug 整数交换均失败，
4 条记录保留约 9～10 µs 的诊断耗时，且未进入最快正确配置和接入候选。
同配置的 origin 在本组检查通过。这只验证过滤流程，不将单次超核通过推广为通用同步保证。
RS=1 另保留协议开销，最佳实际 cooperative 配置只从 RS>1 中选择。

NPU 使用共享卡，本轮复用本任务已有编译缓存；这些热数据小矩阵不用于制定产品触发阈值。
独立冷编译尝试曾在 300 秒墙钟时限内未完成，证据在 `diagnostics/npu-cold-init/`；
最终验证使用新结果目录、`--init-timeout 900`，全部主结果通过，不将前一超时算作精度错误或延迟。

## 实际执行命令

以下命令均在指定容器的任务目录中执行，包目录内的脚本不依赖源码仓库。
CUDA 在 `/home/gaojg/dev` 下执行：

```bash
bash run_matrix.sh --backend cuda --devices 0,1 --smoke --confirm-rounds 2 --no-plot \
  --out /mnt/raid1/gaojg/data/dev/cooperative-compile-matrix-20261123/cuda-package-v2
```

NPU 在 `/home/l30023782/PTA_213` 下执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 NPU_DEVICE_LIMIT=28,56 \
  bash run_matrix.sh --backend npu --devices 2 --smoke --rows 7 --rnumels 4096 \
  --splits 1,8,16 --confirm-rounds 2 --init-timeout 900 --no-plot \
  --out /home/l30023782/PTA_213/cooperative-results/compile-matrix-20261123/npu-package-v2
```

## 数据与产物

| 路径 | 内容 |
| --- | --- |
| `cuda-smoke/`、`npu-smoke/` | 原始 run、主结果、候选、自然选择与逐形状记录 |
| `cases/**/output_code_*.py` | 实际 compile wrapper 与其中的 DSL |
| `cases/**/ir/` | 选中 kernel 的完整 metadata；CUDA TTIR/TTGIR/LLVM/PTX，NPU TTIR/adapter |
| DSL 各候选目录 | reduction/exchange DSL、逐 RS 的 IR 与 metadata，含错误配置 |
| `figures/` | 无卡机器合并生成的 CSV 与 SVG/PNG；14 个主结果、52 个候选、4 个自然选择 |
| `diagnostics/npu-cold-init/` | 单次冷编译超时的原始 job、result 与日志 |

编译候选及选中 tiling 保存在各 compile 记录的 `selected_configs`、`kernels` 与逐轮配置中。
全部记录的检查包括首次输出、最后一次 Graph replay 输出和两组变化输入；DSL 另有严格整数交换。
导出目录已移除可再生成的编译缓存、瞬时状态和正常重复日志。

离线刷新：

```bash
python scripts/tests/cooperative_reductions/plot_compile_matrix.py \
  --input artifacts/cooperative-compile-matrix-20261123/npu-smoke \
  --input artifacts/cooperative-compile-matrix-20261123/cuda-smoke \
  --out artifacts/cooperative-compile-matrix-20261123/figures
```
