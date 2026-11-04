# GNN 长归约与 scatter 融合验证

分析结论直接维护在 `docs/issue-logs/cicd/performance/models/` 下的
`basic_gnn_edgecnn.md`、`basic_gnn_gcn.md`、`basic_gnn_sage.md`、`lennard_jones.md` 和 `nfnet_l0.md`。
这里仅说明复现入口。

`run.py` 保留原 shape / stride / FP32 和实际输出边界，比较原始 generated kernel、
同 DSL 改 tile、两阶段归约以及 eager；scatter 场景比较 eager、原 fallback 编译小图和手写融合 DSL。
原始 kernel 由 `scripts/tools/extract_triton_kernel.py` 提取，保存在
`artifacts/cicd/performance/gnn-validation/original/`。不手工改写原 DSL。

```bash
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_NPU_BACKEND=triton \
python <torch-learn>/scripts/repro/cicd/performance/gnn_validation/run.py \
  --device 3 --case gcn --output <output>/micro_gcn
```

`--case` 也可为 `edgecnn`、`scatter` 或 `pool`。设备必须事先确认空闲且符合使用限制。
小图预热后以 NPUGraph 捕获 16 次调用，计时 9 次 replay，报告每次调用的设备 Event 中位数。
捕获后把输出填成 NaN，再 replay 并检查输出，避免计到空 graph。
正确性参考为同输入 CPU FP64，逐元素检查 `rtol=1e-4, atol=2e-3`；不改模型 runner 精度阈值。
EdgeCNN 原 kernel 原地覆盖激活，因此小图回放额外恢复输入，报告中明确包含 copy 成本；
整网 profile 无此额外 copy，最终整网收益以整网 profile 为准。

`model_run.py` 运行原 TorchBench runner，仅在进程内安装一种实验改动：

- `baseline`：原安装实现。
- `split`：修改 `Reduction.num_splits` 的决策；对 NPU sum、输出元素数 2–128、
  归约长度至少 8192 的情况，尝试 `min(256, ceil(R/512))` 份部分和，再归约部分和。
  `--split-min-reduction` 可改变实验下限；Lennard-Jones 使用 4096。
  `--split-relu-only` 进一步限制到输入为 `aten.where.self` 的 sum，隔离 EdgeCNN ReLU backward。
- `scatter`：导入前只移除 fallback list 中 `aten.scatter_add` 的 packet/default/out，
  以及 `aten.scatter_add_` 的 packet/default，共 5 项，恢复社区 lowering。
- `both`：同时启用 split 和 scatter。
- `pool-fallback`：只替换 `aten.avg_pool2d.default` lowering，比较现有 ACLNN 实现。

这是验证优化方向的临时策略，不能据此直接提交同样阈值的生产补丁。
正式策略需要结合输出并行度、访存、临时 buffer、第二阶段开销以及 mutation/fusion 约束决定。

```bash
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_NPU_BACKEND=triton \
python <torch-learn>/scripts/repro/cicd/performance/gnn_validation/model_run.py \
  --runner <runner-dir>/torchbench.py --variant split --split-relu-only \
  --output <output>/metadata -- --only basic_gnn_edgecnn --device npu \
  --training --performance --float32 --batch-size 1 --backend inductor --npu-backend triton \
  --disable-aclgraph --iterations 20 --repeat 3 --enable-profiler 0 \
  --prof-output-path <output>/profile --output <output>/benchmark.csv
```

GCN 使用 `--only basic_gnn_gcn`，不传 `--split-relu-only`；推理改为 `--inference`。
各组使用独立 `TORCHINDUCTOR_CACHE_DIR` / `TORCH_COMPILE_DEBUG_DIR`，关闭 FX graph cache。
设备 profile 按真实 Step Id 汇总 kernel duration，不用主机 wall time 替代设备时间。
精度验证使用相同变体与原 runner 的 `--accuracy --accu-summary`。
`environment.json` 保存实际软件版本、原安装源码 SHA256、实验拆分事件和安装文件未修改检查。

GCN Train 另按 baseline→split→scatter→both→baseline 复测，复用各变体第一轮的 kernel cache
和最终配置，单独保存新 debug/profile。不能把不同批次的局部节省量直接相加；开始和结束 baseline
用于显示未改 kernel 的波动范围。`selected_configs.json` 记录最终 autotune winner，
复测对应原变体缓存中的记录；`source_identity.json` 对齐安装版与指定分支，
`runner_identity.json` 单独记录 runner checkout 和文件哈希。
`selected_configs.json` 的 `used_by_graphs` 将各 cache source 的配置绑定到实际 performance graph；
同名 kernel 可能因 accuracy/其他图而有多个缓存，不能只按名字任取一个配置。

归档只保留 `micro_*_verified`：早期小图在 graph capture 外取得 raw stream，输出检查未通过，
已经排除其计时。修正为在 capture 内取 stream 后，重新检查 NaN 覆盖、replay 输出与 CPU FP64 参考。
`export_results.py` 去敏个人路径和 backend hash，保持 generated DSL 排版；
`manifest.json` 同时记录远端原文件与去敏文件 SHA256。精度组只保留 CSV、日志和环境，避免重复保存生成图。


2026-11-05 的小改动扩展沿用本脚本，不覆盖 11-04 结果：

- SAGE Infer/Train：仅 `scatter`，保持五项 fallback 过滤，比较原安装版。
- Lennard-Jones Train：`split --split-min-reduction 4096`，其余分段策略不变。
- NFNet Train：`pool-fallback` 临时把 `aten.avg_pool2d.default` lowering 换为现有
  `fallback_handler`，作为实现选择对照，不作为最终生产策略。

`launch_followup.sh <job> small|pool` 从已激活的专用 TorchBench workspace 启动，
两组分别固定使用物理卡 3 / 2，启动前须确认这两张卡空闲。
脚本依次保存原始组/变体的 performance 和 accuracy；shape、batch、dtype 不因性能而缩小。


NFNet 整网基线在正式采样前主动停止，状态单独保存为 `status.json`；不能用其未完成结果计算整网收益。
`run.py --case pool` 抽取原 `_avg_pool2d_42`，固定输入 `[128,1536,14,14]`，
比较原 kernel、同语义 eager、重新 codegen，以及同 DSL 的 SIMD template / SIMT-only 配置。
最后进程内替换一个 lowering，并关闭 FX graph cache，生成真正调用 ACLNN 的 compiled fallback 小图。
仍检查 CPU FP64、NaN 覆盖和 graph replay，单 kernel 时间不冒充整网 A/B。

`launch_followup.sh <job> small-recheck <device>` 复测 SAGE Train 和 Lennard-Jones Train，
各按 baseline→variant→baseline 运行，复用对应原始变体缓存。`device` 仅允许 1、2、3，须先确认空闲。
11-05 第一轮使用物理卡 3；卡 3 后来被其他任务占用，复测改用独占物理卡 2，
每个复测组内原始/改动使用同一卡，不把不同物理卡的两个耗时配成一组。

11-05 证据位于 `artifacts/cicd/performance/small-change-validation-20261105/`。
模型 harness 对应 `f141dbb`（`model_run.py` SHA256 见 `source_identity.json`），
微测 compiled fallback 对应 `4e3ae58`，配置与 performance graph 的绑定对应 `993109a`。
NFNet 原 kernel 与 compiled fallback 的同批对照以 `micro_pool_fallback_verified/result.json` 为准；
`micro_pool_verified` 保留更早一批七组对照，两批时间不混算。
模型测试分别为 SAGE batch 1 / Lennard-Jones batch 4096；NFNet 原定 batch 128 的整网测试未完成。
