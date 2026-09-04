# Triton 参数性能扫描

`num_warps_superblock_factor_perf.py` 复现给定的 Triton atomic-add kernel，扫描
`num_warps` 和 `superblock_factor` 的笛卡尔积。两个参数都取 `1, 2, 4, 8, 16, 32, 64`，共 49 组。

脚本必须在安装 `torch_npu`、Triton 且有 NPU 的环境运行：

```bash
python scripts/tests/triton_parameter_perf/num_warps_superblock_factor_perf.py \
    --warmup 10 --repeat 20 --profile-root prof_log/num_warps_superblock_factor
```

每组先完成 `warmup` 次未采样 launch，再使用 `torch_npu.profiler` 采集 `repeat` 次。
`summary.csv` 中的 `device_us` 是 `kernel_details.csv` 中 device kernel duration 的总和除以采样次数，单位为微秒；
`warmup` 和 `repeat` 记录本次运行使用的轮数。
默认保留每组原始 profile；只需要汇总时增加 `--discard-profile`。

`status=error` 表示当前环境不支持该参数组合或 profile 失败，原因记录在 `error` 列，不会中断其余组合。
