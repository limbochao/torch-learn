# A5 x_grid_barrier 慢例：分量与编译证据

本目录保存 2026-11-20 的原始证据。结论、计时表与边界统一维护在
[需求分析第 6.1 节（5）](../../docs/requirements-design/cooperative-reduction/requirements.md#61-已有性能证据及边界)。
脚本说明与复现入口见[同步探针 README](../../scripts/tests/cooperative_reductions/README.md)。

## 运行与版本对应

- NPU_A5 Docker，`pta_213_case`，经授权共享物理卡 2；[启动时间](start-time.txt)与
  [启动前占用](npu-smi-before.txt)保留原始记录，CANN API 的物理卡映射检查在探针启动时通过。
- [manifest](manifest.json)记录包版本、实际芯片、编译器路径/版本与 helper SHA256。
  此次分量脚本对应提交 `5c9e52009b8337e5aea879366d80722fca60fcca`。
- [计时与正确性记录](components.json)共 44 项，每项先检查输出，再记录五组 Graph + Event 样本及中位数。
  数字是完整 Triton 调用的设备耗时，包含隐式状态处理，不等于裸 atomic 或裸 barrier 的硬件延迟。
- [原始 helper](x_grid_barrier.py)直接用于计时；[有限轮询计数](poll-counts.json)来自单独诊断版，
  不参与原始 helper 性能统计，也不能当作其每次执行的固定轮询次数。
- 各项 `*.ttir`、`*.ttadapter` 和 `*.metadata.json` 与计时记录的 `name` 一一对应。
  `poll-counts-p*` 属于四项额外计数诊断；[run.log](run.log)保留编译提示与运行记录。
- [compiler.py](compiler.py)、[launcher.py](launcher.py)是实测安装包的只读快照。
  两者与 11 月 19 日快照逐文件 SHA256 一致；未修改安装目录文件。

## 匹配设备编译器版本的源码

实际设备编译器报告 `AscendNPU-IR 395c9c5646f9 2026-11-12`，对应完整提交
`395c9c5646f94119a591794a02961b5801f057c0`。`source/` 从该固定提交提取，
[source-map.json](source/source-map.json)记录上游原路径、固定提交 URL 与逐文件 SHA256。

| 文件 | 用于核对的事实 |
| --- | --- |
| [DecomposeAtomicSync.cpp](source/DecomposeAtomicSync.cpp) | `addSyncForReturnedValue` 识别旧值快照，在读取快照前加锁、atomic 后解锁 |
| [SyncBlockLock.cpp](source/SyncBlockLock.cpp) | 无序锁使用 Bakery 算法；取号和等待扫描参与者，状态访问带 `dcci`，并使用 `PIPE_ALL` |
| [SyncBlockHoisting.cpp](source/SyncBlockHoisting.cpp) | 无序锁留在循环内部，不能将轮询循环整体看作只取一次锁 |
| [锁布局编码源码][lock-layout] | `sync_block_lock_layout` 的高 32 位编码无序锁数量 |

## 实际 lowering 路径核对

使用 `dump_adapter.py`，提交 `f492f2d1c18f3dab3234e4efe93851f02276a079`，将归档的
adapter 与 metadata 交给同一安装的设备编译器，开启 debug 保存中间产物；仅编译，没有 launch。
各目录 `manifest.json` 记录输入和脚本 SHA256、编译返回 metadata 及 `launched=false`。
保留 MLIR/LLVM IR 和编译命令日志，未提交临时缓存与设备二进制。

| 产物 | 输入与观察 |
| --- | --- |
| [sum LLVM IR][sum-llvm] | 输入为[长 sum adapter][sum-input]；到达与轮询各有锁调用，后者位于循环内 |
| [atomic LLVM IR][atomic-llvm] | 输入为[取旧值 atomic adapter][atomic-input]；生成取锁、nocache 读取、store、解锁 |
| [sum MLIR][sum-mlir]、[atomic MLIR][atomic-mlir] | 交叉核对锁参数、workspace 参数和读写顺序 |
| [sum manifest][sum-manifest]、[atomic manifest][atomic-manifest] | 输入及脚本哈希、最终 metadata；哈希已与原件核对 |
| [sum 编译日志][sum-log]、[atomic 编译日志][atomic-log] | 实际目标为 `Ascend950`，记录设备编译器参数 |

LLVM IR 中的锁体是待模板库链接的调用，锁内部的循环、`dcci` 和流水屏障来自上面匹配版本的源码；
本目录没有把这些源码结论描述为已反汇编观察到的最终设备指令。
当前尚未分别量化隐式初始化、锁体、GM/缓存维护与内部等待的占比。

[lock-layout]: source/InsertInferSyncBlockLockNumAndInitFunc.cpp
[sum-llvm]: final-sum/kernel.ll
[atomic-llvm]: final-atomic/kernel.ll
[sum-input]: ../npu-cooperative-reductions-20261119-xgrid/barrier/sum-x_grid_barrier-x1-r1048576.ttadapter
[atomic-input]: atomic_old_shared-p32.ttadapter
[sum-mlir]: final-sum/module.hivm.opt.mlir
[atomic-mlir]: final-atomic/module.hivm.opt.mlir
[sum-manifest]: final-sum/manifest.json
[atomic-manifest]: final-atomic/manifest.json
[sum-log]: final-sum/compile.log
[atomic-log]: final-atomic/compile.log
