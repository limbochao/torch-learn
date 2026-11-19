# A5 原始 x_grid_barrier 对照证据（2026-11-19）

本目录保存对上游 `x_grid_barrier` 的新增运行证据，结论和性能表见
[需求分析第 3.3、3.4、6.1 节](../../docs/requirements-design/cooperative-reduction/requirements.md)。
原始 JSON、helper 源码与编译产物保持采集格式；不包含编译缓存和传输用 Git bundle。

实验在 NPU_A5 Docker 的 `/home/l30023782/PTA_213` 执行，激活 `pta_213_case`，
使用经授权共享的物理卡 1。性能脚本通过 CANN API 确认可见逻辑设备 0 映射到物理 1。
设备为 Ascend950PR_9579，28 Cube / 56 Vector core、128 MiB L2、216 KiB UB。

| 项目 | 实际运行版本 |
| --- | --- |
| PyTorch | `2.13.0+cpu` |
| torch_npu | `2.13.0+git93d7d01`，metadata 与 `__version__` 一致 |
| Triton / Triton-Ascend | `3.6.0` / `3.6.0+dev20261112182052` |
| 实际设备编译器 | Triton 包内 BiShengIR 2.0.0，AscendNPU-IR `395c9c5646f9` |
| atomic 探针提交 | `bd9b7c9d251425b6ad1606cbeccf08570b90e0e9` |
| 同步/性能探针提交 | `b8839c73636f22ea6e199dbd9c2c904e0383b2f8` |

两份 manifest 记录脚本和 helper 的 SHA256；同步/性能 manifest 另存实际编译器路径与版本。
采集的 helper 与 PyTorch `v2.13.0` 提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`
中的 `x_grid_barrier` 源码一致。torch_npu 与 11 月 18 日不同，不跨轮计算性能比值。
脚本使用方式统一见[探针 README](../../scripts/tests/cooperative_reductions/README.md)。

## 功能检查

- [atomic 记录](atomic/results.json)：1/2/8/32 个 program，release/acquire/acq_rel 三种参数，
  每项重复 8 次；共 12 项、96 次全部通过计数和旧值检查。该检查不含自旋。
- [交换记录](barrier/visibility.json)：有限轮询诊断版、原始 helper、原生 `all_vector` 各覆盖
  `(X,S)=(1,1)/(1,2)/(1,8)/(1,32)/(4,8)`；共 15 项，每项 8 轮交换均通过。
- 每轮写值、同步、读取同组值，再同步后覆盖 workspace。有限轮询版本仅作诊断，全部通过
  后才调用原始 helper；性能数据没有使用诊断版替代 helper。

这组证据限定于固定 SIMD、`num_warps=4`、`launch_cooperative_grid=True` 和受测参与者。
它不证明 template、pure SIMT、任意 Grid 映射或完整 CUDA cooperative launch 契约。

## 性能记录与编译产物

[bench.json](barrier/bench.json) 有 5 个 shape、3 种策略共 15 条记录，正确性均通过。
`single` 实际 `S=1`；`x_grid_barrier` 与 `native_all_vector` 使用同一 sum DSL，
分片数、`B=1024`、peer 合并及最终输出归属相同，仅同步调用不同。
各条记录保存 5 个 Graph + Event 样本及中位数，每个样本归一化 320 次调用。

测量对象为整段手写 sum 调用，未剥离 launcher 隐式状态处理，不能将总耗时差值标为 barrier 延迟。
例如 `(1,1048576),S=32` 的原始 helper 和原生同步分别为 537.085 µs、7.720 µs；
对应产物可直接对照：

- [原始 helper 的 metadata](barrier/sum-x_grid_barrier-x1-r1048576.metadata.json) 与
  [原生同步 metadata](barrier/sum-native_all_vector-x1-r1048576.metadata.json)。
  `auto_blockify_enabled` 分别为 false/true；前者含 `sync_block_lock_layout=8589934592`。
- [原始 helper 的 adapter](barrier/sum-x_grid_barrier-x1-r1048576.ttadapter) 与
  [原生同步 adapter](barrier/sum-native_all_vector-x1-r1048576.ttadapter)。
  前者可见 atomic store 和轮询，后者可见 `ALL_VECTOR` 同步。
- [运行日志](barrier/run.log) 保留原始功能结果、性能样本与编译器提示。
  原始 helper 路径出现 atomic 导致 AutoBlockify 关闭的提示；尚未量化各项差异的成本。

每个配置均保留 TTIR 和 adapter，metadata 保存实际编译选项；编译器未返回 LLIR 的项没有
对应文件。`start-time.txt` 与 `npu-smi-before.txt` 分别保存启动时间与启动前的占用信息。
本轮是共享卡、热数据、固定配置的协议实现对照，尚未在新的 torch_npu 版本上重测完整
Inductor cooperative 路径，也不作为产品阈值或模型收益证据。
