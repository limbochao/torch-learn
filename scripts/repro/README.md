# repro

本目录存放最小复现脚本。

建议脚本文件名包含问题关键词，例如：

- `inductor_dynamic_shape_guard.py`
- `dynamo_graph_break_case.py`

每个脚本尽量在文件头部说明复现目标、依赖环境和运行命令。

已归档脚本：

- `emb_opt_bs400/`：embedding 模型 `bs=400` 的 NPU 动态 shape/group 与静态编译复现，详细用法见子目录 README。
- `simt_template_fused_reduction_precision_repro.py`：融合 add、ReLU、exp、cos 和 sum 的 SIMT template reduction 精度问题复现。
- `rms_norm_simd_multi_reduction_repro.py`：RMSNorm weight grad 的 SIMD 多 reduction 轴 codegen 复现脚本，默认使用原始大 shape。
- `rms_norm_simd_multi_reduction_pass_case.py`：同一表达式的可通过 case，默认使用 `BATCH=2 SEQ=128 HEADS=8 HEAD_DIM=128 CHECK=1`。
- `rms_norm_simd_multi_reduction_manual_tiling.py`：同一表达式的手动 tiling case，默认使用 `R2BLOCK_SUB=7`，覆盖非整除 reduction tile。
- `symbolic_mask_perf/`：对比无 `x2` mask、`x2 < 200` 和显式 full-shape 恒真 mask 的三份 SIMT kernel。
