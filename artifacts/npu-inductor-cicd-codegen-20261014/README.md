# CICD kernel codegen 精度复测汇总

[summary.csv](summary.csv) 保存 2026-10-14 的 44 个 kernel 用例复测及问题分类结果，
覆盖 13 个模型，包含随机复测、定点 DSL 修正复测、误差统计和源码定位信息。

来源为 `CICD/summary.csv`，在原始 19 列基础上增加“原因分类”列，置于“原因分析”之前。
原有字段内容、行顺序、UTF-8 BOM 和 CRLF 换行保持不变。
表中的 `bench_acc_nodata/`、`retest/` 路径均相对于原 CICD 工作目录；对应原始用例、
详细结果和源码快照未随本表导入。

“定点 DSL 修正复测”记录的是手动修改生成内核后的诊断实验，
不是后续三个 codegen 修复分支重新编译验证的结果。

## 原因分类

| 原因分类 | 用例数 | 对应修复分支 |
|---|---:|---|
| 归约尾块 mask 缺失 | 16 | `bugfix/codegen-reduction-mask` |
| 多轴归约错误合并保留轴 | 25 | `bugfix/codegen-reduction-axes` |
| 调试开关破坏负索引回绕 | 1 | `bugfix/codegen-negative-index` |
| 回放／精度比较布局误报 | 2 | 不属于上述三个 codegen 修复分支 |

尾块 mask 类包括 Softmax 2 例和归一化 14 例；多轴归约类包括 BatchNorm 23 例和
clone + sum 2 例。分类依据原有“原因分析”、源码定位以及逐例复现记录，
通过原始文件 SHA256 和用例路径核对记录对应关系。
