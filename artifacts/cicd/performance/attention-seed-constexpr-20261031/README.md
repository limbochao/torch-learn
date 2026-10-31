# 两个 Train 首层 attention 性能证据

对应 [分析文档][analysis]；可执行复现的说明维护在 [scripts/repro][repro]。
这里保存分析证据，不将生成代码作为长期维护的可执行脚本。

- `*_first_attention.py` / `*_later_attention.py`：从真实生成 wrapper 提取的完整 kernel DSL。
- `failed_first_attention_checker.py`：回退前失败的多轴 checker，包含完整计算体和 metadata。
- `source_evidence.txt`：安装环境中 signature、常量绑定、TTIR 检查和回退分支的源码摘录及行号。
- `*_fallback_traceback.txt`、`ttir_original_failure.txt`：触发回退和独立 TTIR 检查的关键日志。
- `ttir_check_results.json`：原始、常量绑定和运行时参数三组对照。
- `electra_kernel_timings.json`、`electra_selected_tiling.json`：原始首层和后续层的采样及实际配置。
- `attention_profile_windows.csv`：原始 profile 中 attention 区间的逐项耗时和原 CSV 行号。
- `profile_summary.json`：四个场景基线和两个 Train 的首层 attention 汇总。
- `manifest.json`：版本、采集范围、原始 CSV 和源码校验值。

个人路径、trace run ID 和完整 backend_hash 已去敏；DSL 计算体、索引和原始格式保留。
原始 CSV 的环境构建版本未知；2026-10-31 的版本只标识重新采集 codegen / 单 kernel 的环境。
runner HEAD 不代替安装框架版本，源码校验值用于区分实际安装内容。

[analysis]:
  ../../../../docs/issue-logs/cicd/performance/electra-layoutlm-train-first-attention-20261031.md
[repro]:
  ../../../../scripts/repro/cicd/performance/attention_seed_constexpr_20261031/README.md
