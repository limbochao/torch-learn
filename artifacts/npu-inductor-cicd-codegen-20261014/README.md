# CICD kernel codegen 精度复测汇总

[summary.csv](summary.csv) 保存 2026-10-14 的 44 个 kernel 用例复测及问题分类结果，
覆盖 13 个模型，包含随机复测、定点 DSL 修正复测、误差统计和源码定位信息。

来源为 `CICD/summary.csv`，按原文件逐字节保存，保留 UTF-8 BOM 和原始列顺序。
表中的 `bench_acc_nodata/`、`retest/` 路径均相对于原 CICD 工作目录；对应原始用例、
详细结果和源码快照未随本表导入。

“定点 DSL 修正复测”记录的是手动修改生成内核后的诊断实验，
不是后续三个 codegen 修复分支重新编译验证的结果。
