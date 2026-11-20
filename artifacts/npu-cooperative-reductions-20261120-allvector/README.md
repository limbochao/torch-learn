# all_vector 改写 x_grid_barrier 的验证证据

本目录保存 2026-11-20 NPU_A5 Docker、`pta_213_case`、共享物理卡 2 上的原始产物。
功能和性能结论见[需求分析第 6.1 节（6）][requirements]，helper 代码和适用条件见
[方案第 1.3 节][design]；运行方法统一维护在[探针 README][scripts]。

## 源码与环境

源码提交为 `fd8f2c733a25beb304676588b876a05e3da069bb`。
[manifest](manifest.json)记录实际包版本、设备映射、编译器版本、调用参数与源码 SHA256。
`source_sha256` 对应 `barrier_bench.py`，`npu_helper_file_sha256` 对应完整 NPU helper 模块。
[原始 helper 快照](x_grid_barrier.py)与[改写版函数快照](npu_x_grid_barrier.py)另有独立 SHA256。
脚本和 helper 的文件哈希均已与该提交核对。

[启动时间](start-time.txt)、[结束时间](end-time.txt)、[启动前卡占用](npu-smi-before.txt)和
[运行日志](run.log)保留原始记录。未修改已安装的 PyTorch、torch_npu 或 Triton 文件。

## 数据和编译产物

| 入口 | 内容 |
| --- | --- |
| [visibility.json](visibility.json) | 四种实现、五种参与者配置，共 20 项检查；每项三次不同 epoch、每次八轮数据交换 |
| [bench.json](bench.json) | 五个 shape、四种 sum 策略的 20 项计时；每项保留精度检查状态、五个样本和中位数 |
| [改写版长 sum adapter][adapter] | 实际生成为 `ALL_VECTOR`、事件 0，不含原始 semaphore atomic 协议 |
| [改写版长 sum metadata][metadata] | 自动 block 映射开启，无返回值 atomic 对应的软件锁布局 |

`visibility.json` 中 `kind=0/1/2/3` 分别表示有限轮询诊断版、原始 helper、直接原生同步、
NPU helper 改写版。改写版在写后读、读完再覆盖两个位置都调用同一个 helper，均使用事件 0；
传入的 semaphore 初值为 123，检查其内容保持不变。
多组检查要求各组执行相同轮次，不能用于证明按组独立推进的控制流也适合全 Vector 同步。

各项 `*.ttir`、`*.ttadapter`、`*.metadata.json` 按检查名或策略/shape 命名。
计时均为整个手写 sum 的 Graph + Event 设备耗时，包含局部归约、同步、peer 合并与输出；
未将差值称为裸 barrier 延迟，也未经过 Inductor schedule/autotune。

## 多输出组的定向复测

主记录中直接原生同步的 `(8,1048576)` 样本出现明显波动，原始样本保留不替换。
`repeat-x8/` 使用同一源码、同一物理卡和主实验的 Triton 编译缓存，单独复测该 shape：
[复测计时](repeat-x8/bench.json)、[复测可见性检查](repeat-x8/visibility.json)、
[复测版本与参数](repeat-x8/manifest.json)、[复测日志](repeat-x8/run.log)。
两种原生调用方式的 metadata 除 hash 外一致；复测亦用于确认首次波动不能解释为封装带来加速。
复测保存 JSON metadata 和时间/占用记录，不重复归档相同源码下的 IR。

## 本核写回与跨核等待的后端产物

`final-native/` 使用本目录已归档的改写版 `(1,1048576),S=32` adapter 和 metadata，
通过仓库现有 `dump_adapter.py` 在同一环境仅编译导出；没有启动 kernel，没有新增性能样本。
编译器为 NPUIR `395c9c5646f94119a591794a02961b5801f057c0`、LLVM 19.1.7。

- [后端 MLIR](final-native/module.hivm.opt.mlir)：partial store 后生成 `pipe_barrier[PIPE_ALL]`，
  随后为 MTE3 生产流水的跨核 set/wait，再执行 nocache peer load。
- [LLVM IR](final-native/kernel.ll)：保留 `llvm.hivm.BARRIER`、`llvm.hivm.SET.CROSS.CORE`
  和 `llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM`；这是降低后的 IR，不是最终机器码反汇编。
- [编译 manifest](final-native/manifest.json)：`launched=false`；记录输入 adapter、metadata
  和 `dump_adapter.py` 的 SHA256，均已与归档输入及仓库脚本核对。
- [编译日志](final-native/compile.log)、[编译器版本](final-native/compiler-version.txt)：保留原始输出。

[requirements]: ../../docs/requirements-design/cooperative-reduction/requirements.md#61-已有性能证据及边界
[design]: ../../docs/requirements-design/cooperative-reduction/design.md#13-kernel-内协议与运行流程
[scripts]: ../../scripts/tests/cooperative_reductions/README.md
[adapter]: sum-npu_x_grid_barrier-x1-r1048576.ttadapter
[metadata]: sum-npu_x_grid_barrier-x1-r1048576.metadata.json
