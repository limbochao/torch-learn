# 脚本索引

仓库中的测试、复现、分析和辅助工具脚本及其说明统一放在本目录。其它目录只引用这里，
不重复维护脚本说明。

Skill 内部需要随安装包分发的 bundled scripts 保留在对应 `skills/<name>/scripts/` 中，不属于仓库级公共脚本。

## 目录

- `repro/`: 最小复现脚本，优先保证单文件可运行。
- `tests/`: 功能验证、实验验证或回归测试脚本。
- `tools/`: 被测试、复现和分析脚本复用的辅助工具。

## 当前内容

- `repro/rms_norm_simd_multi_reduction_repro.py`: 复现 RMSNorm weight grad SIMD 多 reduction 精度问题。
- `repro/rms_norm_simd_multi_reduction_pass_case.py`: 提供缩小 shape 后的正确性对照。
- `repro/rms_norm_simd_multi_reduction_manual_tiling.py`: 使用手工 tiling 验证问题。
- `tests/pointwise_op_cost_cases.py`: 验证并对比 elementwise Aten 算子的 NPU kernel 开销。
- `tools/npu_profiler.py`: 采集并解析 torch_npu profiler 结果，详细用法见 `tools/README.md`。

新增或调整脚本时，在对应脚本、同级 `README.md` 或本索引中维护用途、运行命令、输入输出和验证结论。
