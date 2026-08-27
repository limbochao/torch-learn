# Symbolic mask performance kernels

该复现将同一个 SIMT pointwise kernel 构造成三种只在 `x2` mask 表达形式上不同的版本：

- A `no_x2_mask`：对应当前 `new_kernel`，load/store 只使用 `y1_mask` 和 `z0_mask`，mask 在 `x2` 维通过 broadcast 生效。
- B `x2_lt_200`：恢复 `x2_mask = x2 < 200`，并将 `x2_mask` 加入两个 load 和一个 store。
- C `full_true_x2_mask`：使用 `x2_mask = tl.full((1, 1, 200), True, tl.int1)`，显式构造覆盖完整 `x2` 维的恒真 mask。

三份 kernel 共用相同的输入 shape、地址表达式、计算逻辑、pointwise metadata 和启动参数。A/B 可以复现删除
`x2_mask` 前后的差异；B/C 用于区分性能变化来自 `x2 < 200` 比较本身，还是来自显式 full-shape mask 对后端
访存分析和 lowering 的影响。

在 NPU 环境运行：

```bash
python scripts/repro/symbolic_mask_perf/compare_mask_kernels.py
```

默认使用 `s0=256`、20 次 warmup 和 100 次计时迭代，并检查三份 kernel 的输出一致性。可通过环境变量调整：

```bash
S0=256 WARMUP=50 REPEAT=500 CHECK=1 \
  python scripts/repro/symbolic_mask_perf/compare_mask_kernels.py
```

设置 `PRINT_KERNEL_SOURCE=1` 可以打印替换完成后的三份完整 Triton DSL。
