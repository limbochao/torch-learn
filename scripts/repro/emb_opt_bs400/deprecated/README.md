# Deprecated CUDA-derived runnable

本目录中的 `fx_graph_runable_npu.py` 和 `fx_graph_custom_kernels_npu.py` 是此前基于 CUDA runnable 改造的
NPU 版本。其图结构、自定义 Triton kernel 和 NPU 原始 runnable 不一致，仅保留用于历史问题对照，不再维护，
也不应作为 static/dynamic/group 性能测试入口。
