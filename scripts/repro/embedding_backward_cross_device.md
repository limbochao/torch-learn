# Embedding backward cross-device repro

`embedding_backward_cross_device.py` 是从用户提供的 NPU AOT/Inductor graph 还原出的标准 PyTorch 版本。
`eager_forward` 保留原始 ATen 算子顺序，仅将设备从输入 tensor 推导，以便同一份脚本运行在 NPU 或 CUDA：

- `source.reshape(B, 1, 2280)[:, :, 1064:1192]` 与两个 `[B, 1, 128]` 梯度相加。
- `active=False` 或 `indices == -1` 时贡献为零。
- 负索引先加 `98166`，然后对 `[98166, 128]` 输出做累加。

实现使用 `index_put_(accumulate=True)`，不导入、不调用 Triton，因此同一个脚本可以比较 NPU 或 CUDA 上的 eager 与 Inductor。

```bash
# GPU
python scripts/repro/embedding_backward_cross_device.py --device cuda --device-id 3

# NPU（需要 torch_npu 和 register_inductor_npu 环境）
python scripts/repro/embedding_backward_cross_device.py --device npu --device-id 3
```

脚本固定使用 AOT graph 的原始输入 shape，不提供 shape、dynamic 或 check 参数。
`--warmup` 和 `--repeat` 仅控制 profiler 的预热和采集轮数。profile 结果默认保存到
`prof_log/embedding_backward_cross_device/<timestamp>/{eager,inductor}/`，也可通过
`--profile-dir` 指定根目录。输出的
`*_device_us` 是每轮 device kernel duration 之和：CUDA 从 Chrome trace 的
`kernel` event 读取，NPU 从 `kernel_details.csv` 读取；不使用主机 wall-clock 时间。
通过 `--device` 和 `--device-id` 传入设备类型及物理卡 ID，脚本会自动设置
`CUDA_VISIBLE_DEVICES`（CUDA）或 `ASCEND_RT_VISIBLE_DEVICES`（NPU）。选中的物理卡会映射为
runtime 的 `cuda:0`/`npu:0`；例如 `--device npu --device-id 3` 实际使用物理卡 3。
