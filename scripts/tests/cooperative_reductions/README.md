# Cooperative reduction 调研探针

`probe.py` 用于 A5 上的定向研究，结果解释见
[需求分析](../../../docs/requirements-design/cooperative-reduction/requirements.md)。
不修改已安装的 PyTorch、torch_npu 或 Triton。

在 Docker 工作目录激活 `pta_213_case`，按物理卡限制和占用情况设置
`ASCEND_RT_VISIBLE_DEVICES` 后运行。`--device 0` 是可见设备中的逻辑号。

```bash
# 只读版本、源码 SHA256 和对应源码片段，不初始化 NPU。
python scripts/tests/cooperative_reductions/probe.py inspect --out ../cooperative-results/inspect

# 显式指定目标，仅编译，不 launch CUDA semaphore 自旋屏障。
python scripts/tests/cooperative_reductions/probe.py compile --out ../cooperative-results/compile

# 现有 Inductor 路径、开启开关和强制开关的差异，保存实际 selected config。
python scripts/tests/cooperative_reductions/probe.py inductor \
  --shapes 1x1048576 --out ../cooperative-results/inductor

# 动态 shape 编译，并在同一个 callable 上验证缩小 R 和非整除尾块。
python scripts/tests/cooperative_reductions/probe.py inductor --dynamic \
  --shapes 1x1048576 --inductor-modes baseline,enabled,forced \
  --reuse-r 32768,1048593 \
  --out ../cooperative-results/dynamic

# 手写 SIMD 原型：GM 可见性与重复屏障检查，然后比较三种策略。
python scripts/tests/cooperative_reductions/probe.py bench --graph \
  --shapes 1x4096,1x32768,1x1048576,1x16777216,8x1048576,32x1048576 \
  --out ../cooperative-results/bench
```

`compile` 阶段使用固定 target、signature、constexpr 和 compile mode；每个结果明确记录
`launched: false`。`bench` 只运行原生 `all_vector` 屏障，且要求 `X * S` 不超过 vector core 数。
它验证的是手写原型，不是已适配的 Inductor cooperative reduction。
`native_visibility` 在每一阶段写新值、全核同步、读其他 program 的值，并在覆盖 workspace
前再次同步，覆盖 8 个阶段。没有自动运行可能永久等待的 CUDA semaphore 协议。

性能结果保留五组原始样本；`--graph` 在 NPU Graph 内捕获 32 次调用，计时 10 次 replay，
先预热三次 replay，以 320 次调用归一化。
未指定 `--graph` 时记录事件加 Python enqueue 时间，不能直接混比。
原型固定 `B=1024`、SIMD，没有 autotune；默认拆分数取 core 数与 X 的比例并限制为 64。
因此结果只用于判断收益区间，不能作为最终触发阈值。
