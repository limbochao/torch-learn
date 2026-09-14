---
title: DLRM embedding kernel 性能复现与收益口径
---

# DLRM embedding kernel 性能复现与收益口径

## 问题

DLRM 在 CUDA 上去掉公共算子后，Inductor 仍有明显收益；在 NPU 上，去掉相同公共算子后反而劣化。需要把问题拆成两个可独立复现的 case：

1. `embedding_select`：26 个 `[128, 16]` lookup。
2. `embedding_dense_backward`：26 个 `[128, 16]` grad 输入，以及其前面的 `add + slice`。

两个 repro 都保留 `torch.compile(..., backend="inductor")`，不改变 fallback 策略。

## Repro

脚本位于：

- [`scripts/repro/dlrm_embedding_select_perf.py`](../../../scripts/repro/dlrm_embedding_select_perf.py)
- [`scripts/repro/dlrm_embedding_dense_backward_perf.py`](../../../scripts/repro/dlrm_embedding_dense_backward_perf.py)

在目标 NPU 上运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2
python scripts/repro/dlrm_embedding_select_perf.py
python scripts/repro/dlrm_embedding_dense_backward_perf.py
```

脚本默认使用：

```text
BATCH=128
TABLES=26
SLOTS=27
EMBEDDING_DIM=16
embedding_select rows=1460
embedding_dense_backward rows=4096
dtype=float32，indices=int64
dynamic=False
```

`embedding_dense_backward` repro 输出三组时间：

- `eager_full`：先计算完整 `[128, 27, 16]` add，再切出 26 个 slice 后调用 ATen fallback。
- `eager_split`：每个 table 单独计算一个 `add + slice` 后调用 ATen fallback。
- `inductor_split`：对 `eager_split` 做 Inductor compile，生成每个 table 的 fused 前处理 kernel，fallback 仍是 ATen。

可用 `--tables 1` 缩小到单个 table，或用 `--rows` 调整 embedding table 行数。脚本会检查 eager/compiled 输出一致性，并按：

```text
improvement = 1 - inductor_time / eager_time
```

打印提升比。

## Generated code 语义

原始 DLRM 产物中，`triton_poi_fused_add_embedding_dense_backward__N` 的名字包含 `embedding_dense_backward`，但该 Triton kernel 实际完成的是对应 slice 的 add/load/store。其后才是独立的：

```python
torch.ops.aten.embedding_dense_backward.default(...)
```

因此需要区分：

- kernel 名字中的 provenance；
- fused `add + slice` 前处理；
- 真正的 ATen `embedding_dense_backward` fallback。

## A5 验证结果

在 `Ascend950PR_9579` 上，固定一个 `[128, 16]` grad 输入单独测量 fallback：

| 路径 | median |
|---|---:|
| Eager `aten.embedding_dense_backward` | 18.633 us |
| Inductor 仅调用该 fallback | 18.778 us |

差异为 `-0.78%`，在测量误差范围内。因此没有证据表明 fallback 本体是主要劣化来源。

包含完整前处理的等价图结果为：

| 路径 | median |
|---|---:|
| Eager：完整 add + slice + fallback | 24.028 us |
| Inductor：Triton add/permute + fallback | 25.975 us |

对应提升比为：

```text
1 - 25.975 / 24.028 = -8.1%
```

该差异来自 fused 前处理 kernel、wrapper 和调度形态，而不是 fallback 算法发生变化。

## 预期收益

### kernel 级目标

原始 profile 去掉公共算子后，两类 NPU kernel 的总耗时约为 `1939.176 us`，对应 CUDA 为 `691.150 us`：

| kernel 类别 | NPU | CUDA | 目标减少 |
|---|---:|---:|---:|
| `embedding_select` | 858.777 us | 415.367 us | 443.410 us，约 51.6% |
| `add_embedding_dense_backward` 前处理 | 1080.399 us | 275.783 us | 804.616 us，约 74.5% |
| 合计 | 1939.176 us | 691.150 us | 1248.026 us，约 64.4% |

这里的“目标”是让 NPU 两类 kernel 达到当前 CUDA profile 的对应耗时，属于 kernel 级目标，不是保证值。

### 模型 residual 级目标

去掉公共算子后：

```text
NPU Eager     = 3063.807 us
NPU Inductor  = 3497.165 us
CUDA Eager    = 3248.420 us
CUDA Inductor = 2150.900 us
```

CUDA 的提升比为 `33.79%`。若 NPU 只把上述两类 kernel 优化到 CUDA 的耗时，NPU residual 预计为：

```text
3497.165 - (1939.176 - 691.150) = 2249.139 us
```

对应：

```text
1 - 2249.139 / 3063.807 = 26.59%
```

这比 CUDA 的 `33.79%` 仍少约 `7.20` 个百分点，约还需要减少 `220 us`。这部分应来自 launch/wrapper/queue gap、其他 kernel 差异或内存调度，而不能归因于 `embedding_dense_backward` fallback 本体。

## 复现时的判断标准

- `embedding_select`：先看单 table，再看 26 table 的总耗时和 kernel 数量。
- `embedding_dense_backward`：同时报告 `eager_full`、`eager_split` 和 `inductor_split`，不要只比较 Inductor 与 split eager。
- 若 `inductor_split` 慢而单独 fallback 接近 eager，优化重点应放在 fused `add + slice` 和小 kernel 调度。
- 若修改后 fallback 单独耗时变化，才需要继续检查 ATen fallback 实现或其输入 layout。

## 风险

`embedding_dense_backward` repro 默认将 table 行数设为 4096，以控制 26 个输出 grad table 的显存占用；`[128,16]` 的 grad shape、stride 和 ATen 调用保持不变。要复现某一个真实大 table，可使用 `--tables 1 --rows <真实行数>`。不同 NPU 型号、CANN、torch_npu 和 Triton-Ascend 版本需要重新生成 output code，不能跨 SoC 直接复用生成 kernel 做绝对性能比较。
