# GNN 长归约与 scatter 融合验证

分析结论直接维护在 `docs/issue-logs/cicd/performance/models/` 下的
`basic_gnn_edgecnn.md`、`basic_gnn_gcn.md`。这里仅说明复现入口。

`run.py` 保留原 shape / stride / FP32 和实际输出边界，比较原始 generated kernel、
同 DSL 改 tile、两阶段归约以及 eager；scatter 场景比较 eager、原 fallback 编译小图和手写融合 DSL。
原始 kernel 由 `scripts/tools/extract_triton_kernel.py` 提取，保存在
`artifacts/cicd/performance/gnn-validation/original/`。不手工改写原 DSL。

```bash
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_NPU_BACKEND=triton \
python <torch-learn>/scripts/repro/cicd/performance/gnn_validation/run.py \
  --device 3 --case gcn --output <output>/micro_gcn
```

`--case` 也可为 `edgecnn` 或 `scatter`。设备必须事先确认空闲且符合使用限制。
小图预热后以 NPUGraph 捕获 16 次调用，计时 9 次 replay，报告每次调用的设备 Event 中位数。
捕获后把输出填成 NaN，再 replay 并检查输出，避免计到空 graph。
正确性参考为同输入 CPU FP64，逐元素检查 `rtol=1e-4, atol=2e-3`；不改模型 runner 精度阈值。
EdgeCNN 原 kernel 原地覆盖激活，因此小图回放额外恢复输入，报告中明确包含 copy 成本；
整网 profile 无此额外 copy，最终整网收益以整网 profile 为准。

`model_run.py` 运行原 TorchBench runner，仅在进程内安装一种实验改动：

- `baseline`：原安装实现。
- `split`：修改 `Reduction.num_splits` 的决策；对 NPU sum、输出元素数 2–128、
  归约长度至少 8192 的情况，尝试 `min(256, ceil(R/512))` 份部分和，再归约部分和。
  `--split-relu-only` 进一步限制到输入为 `aten.where.self` 的 sum，隔离 EdgeCNN ReLU backward。
- `scatter`：导入前只移除 fallback list 中 `aten.scatter_add` 的 packet/default/out，
  以及 `aten.scatter_add_` 的 packet/default，共 5 项，恢复社区 lowering。
- `both`：同时启用 split 和 scatter。

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

归档只保留 `micro_*_verified`：早期小图在 graph capture 外取得 raw stream，输出检查未通过，
已经排除其计时。修正为在 capture 内取 stream 后，重新检查 NaN 覆盖、replay 输出与 CPU FP64 参考。
`export_results.py` 去敏个人路径和 backend hash，保持 generated DSL 排版；
`manifest.json` 同时记录远端原文件与去敏文件 SHA256。精度组只保留 CSV、日志和环境，避免重复保存生成图。
