---
title: TorchBench 2.13 环境准备与模型测试
---

# TorchBench 2.13 环境准备与模型测试

这套脚本用于准备指定的 50 个 TorchBench / HuggingFace 模型，并调用 Ascend 的 TorchBench runner 测试。

- 安装脚本直接执行，自动选择源码存放位置；不安装 `torch`、`torch_npu`、`triton`、`triton_ascend`。
- 测试脚本自动下载缺少的模型数据，batch size 默认采用各模型配置。
- 模型测试支持选择模型、开关 Inductor、开关精度检查；默认使用 Inductor + Triton 检查精度。

脚本入口、配套依赖、数据清单、参数语义和执行命令统一见
[TorchBench 脚本使用说明](../../../../scripts/torchbench.md)。
