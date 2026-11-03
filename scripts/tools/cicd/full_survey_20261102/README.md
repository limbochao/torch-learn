# CICD 全量性能粗筛：2026-11-02

仅使用 Python 标准库，在本地解析 profile 与生成代码，不导入 torch、不运行模型、不连接设备。
这是当前数据布局的分析快照，不是通用性能归因器；候选和条件性预算必须结合模型文档阅读。

输入目录需包含 `summary_NPU.csv`、`summary_GPU.csv`，以及每个 `<model>_<Infer|Train>/` 下的：

- `<case>_<Eager|Inductor>_<NPU|GPU>.csv` 四份数据。
- `NPU_g*_output_code.py`、`GPU_g*_output_code.py` 及同批 FX 文件。

在仓库根目录运行，设置三个变量为输入数据、独立工作目录和文档仓库：

```bash
CICD_INPUT='<profile-root>'
CICD_WORK='<analysis-work-dir>'
CICD_REPO="$PWD"
python scripts/tools/cicd/full_survey_20261102/collect.py \
  --input-root "$CICD_INPUT" --work-dir "$CICD_WORK"
python scripts/tools/cicd/full_survey_20261102/analyze.py \
  --input-root "$CICD_INPUT" --work-dir "$CICD_WORK"
python scripts/tools/cicd/full_survey_20261102/render.py \
  --input-root "$CICD_INPUT" --work-dir "$CICD_WORK" --repo "$CICD_REPO"
```

`collect.py` 输出 `inventory.json`，包含四份数据的归一化汇总和 kernel 原始 ATen/FX 来源。
`analyze.py` 输出 `analysis.json`，保留每项选择的 Eager/kernel 名称与预算公式所需数值。
`render.py` 会更新 `docs/issue-logs/cicd/performance/models/` 的模型文档、全量总览和对应 artifacts；
MobileBERT 的逐 kernel 深挖直接维护在对应模型文档的 `MOBILEBERT REDUCTION ANALYSIS` 标记区间内，
重新生成时原样保留该区间；不再另设专项分析页面。不会改写历史分析或性能分类索引。
FMA/attention 的实测补充维护在模型文档的 `CICD VALIDATION <case>:<kind>` 标记区间，
总览实测摘要维护在 `CICD RUNTIME VALIDATION` 区间；重新生成时同样保留。
只针对本次 40 模型/78 场景快照使用，手工补充的具体解释也属于该快照。

NPU 按 CSV 中实际 Step Id 数量归一化，GPU 的聚合表按本批固定 10 步归一化。
输入改变时应先检查 summary 的 profile_steps、batch、模型集合、schema 和截断 kernel 名映射，
不能无条件沿用“10 步”或当前 kernel 后缀。

局部预算为 `N_I - N_E × G_I/G_E`；不用整网 Eager 比值缩放某个 kernel。
含多形状/融合边界的操作族采用文档所述保守边界，不能把它读作单调用的精确映射。
首层 attention 使用同设备后续层控制；FMA 使用已知重复执行窗口，不混入有效计算成本。
候选未填写预算；已列预算不代表实际修复可完全回收，重叠预算不可相加。

本次验证仅包括本地全量重算、公式和覆盖检查、证据/链接及去敏检查。
尚未做修复、固定配置跑卡、数值精度或端到端性能回归。
