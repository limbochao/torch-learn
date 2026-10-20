# 四模型修复验证证据

分析与复现命令见 [修复审查](../../../docs/issue-logs/cicd/model-fixes-20261020.md)
和 [脚本说明](../../../scripts/cicd-model-fixes-20261020.md)。本目录随复测更新，RUNNING 不表示通过。

- `branches.json`、`patches/`：五类修复分支及最终整合分支的精确提交和代码差异。
- `environment.json`：实际 Python/package 版本、runner hash、未被修改的原安装文件 hash。
- `overlay*_manifest.json`：各次测试实际加载的修复文件 hash。
- `matrix/`：首轮矩阵（规划 32 项）；`matrix_v2/`：修复 softmax tiling 后的 Albert 动态训练 performance。
- `matrix_v4/`：最新 master 整合分支上的 NFNet 动态场景复核。
- `matrix_v4_accuracy/`：同一整合分支，在另一空闲设备上并行执行的 NFNet 动态精度场景。
- `model_matrix.json`：按场景合并各阶段最新结果；每个场景保留来源阶段，未完成项不会自动记为 PASS。
- `regressions_v4/`：最新整合分支的 32 项数值回归；8 项框架单元检查保留在 `*_unit_v4.txt`。
- `generated/`：从真实模型 output_code 抽取的完整 kernel；是分析证据，执行入口在 scripts/repro。
- `nfnet_pooling_selected_v4.json`：真实 autotune cache 中三个 dynamic pooling kernel 的选中配置。
- `ta_after_inductor_v4/`：仅运行独立 Triton kernel 得到的结果、TTIR 和 ttadapter。
- `softmax_capture_v1/journal.jsonl`：skill 单 kernel 捕获的原始候选记录（已去敏）。
- `softmax_direct_export.txt`、`softmax_minimal_original.txt`：独立导出及精简用例的原候选编译失败。
- `softmax_minimal_fixed.txt`：同一 DSL 的小 tile 编译和 eager 数值对照。

版本对应关系：初版矩阵使用 `overlay_manifest.json`，Albert 第二轮使用 `overlay_v2_manifest.json`；
最终数值回归、TA 独立用例、NFNet 动态复核使用 `overlay_v4_manifest.json`。
`overlay_v3_manifest.json` 补充 no-wrap bounds 修复；v3/v4 的三个目标文件内容一致，
v4 另加载最新 master 的 decomposition.py。早期 `regressions/`、`regressions_v3/` 不计入最终回归总数。

归档移除了个人路径、进程标识、完整 backend hash；异常日志仅保留末尾 65 行。
generated kernel body 保持生成格式，metadata 中的环境标识已替换为占位。
TA 源码和安装包均未修改。
