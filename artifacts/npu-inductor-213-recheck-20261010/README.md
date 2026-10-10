# PyTorch 2.13 NPU Inductor 复测结果

本目录记录会话 `01a12536-1178-7af2-b4af-473911ac7a81` 的 331 行输入清单复测结果。测试在 NPU_A5 的 `pta_213_case` 环境执行，用例目录为 `/home/l30023782/PTA_213/Community-Use-Case-Testing/v2.13.0/pytorch/test/inductor`，测试源码未修改。

每行输入用例使用一个独立 pytest 进程，普通用例并行使用 NPU 0、1、2；`test_linspace4`、`test_multi_device`、`test_multi_gpu_device`、`test_multi_gpu_recompile_on_index` 使用全卡环境。基础环境变量为 `TORCH_TRANSFER_TO_NPU=1`、`TORCH_NPU_DEVICE_CAPABILITY=8.0`，并设置 `TORCHINDUCTOR_COMPILE_THREADS=2`、`TORCHNPU_PRECOMPILE_THREADS=2`、`OMP_NUM_THREADS=2` 和独立缓存目录。

`results.csv` 按输入清单原顺序排列，结果值包括 `PASS`、`FAIL`、`SKIP`、`XFAIL`、`PROCESS_ERROR`、`TIMEOUT` 和 `NOT_COLLECTED`。`PROCESS_ERROR` 表示 pytest 在收集或初始化阶段以非测试断言退出；动态 shape 后段的 58 行均属于此类。`NOT_COLLECTED` 是 `test_conv_bn_eval`：当前源码中它是 `test_basic` 内部函数，pytest 没有独立 nodeid。`TIMEOUT` 使用单行 900 秒上限，保留已产生的日志和 JSON。

`node-results.json` 保留逐节点结果、跳过/XFAIL 原因和 pytest 证据；`manifest.json` 保存输入清单哈希、环境和覆盖范围；`preflight.json` 保存容器版本与源码状态。
