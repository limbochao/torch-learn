# repro

本目录存放最小复现脚本。

建议脚本文件名包含问题关键词，例如：

- `inductor_dynamic_shape_guard.py`
- `dynamo_graph_break_case.py`

每个脚本尽量在文件头部说明复现目标、依赖环境和运行命令。

已归档脚本：

- `rms_norm_simd_multi_reduction_repro.py`：RMSNorm weight grad 的 SIMD 多 reduction 轴 codegen 复现脚本，默认使用原始大 shape。
- `rms_norm_simd_multi_reduction_pass_case.py`：同一表达式的可通过 case，默认使用 `BATCH=2 SEQ=128 HEADS=8 HEAD_DIM=128 CHECK=1`。
- `rms_norm_simd_multi_reduction_manual_tiling.py`：同一表达式的手动 tiling case，默认使用 `R2BLOCK_SUB=7`，覆盖非整除 reduction tile。
