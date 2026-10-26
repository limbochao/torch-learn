# repro

本目录存放最小复现脚本。

建议脚本文件名包含问题关键词，例如：

- `inductor_dynamic_shape_guard.py`
- `dynamo_graph_break_case.py`

每个脚本尽量在文件头部说明复现目标、依赖环境和运行命令。

已归档脚本：

- [cicd/dinov2_simt_reduction_20261026/](cicd/dinov2_simt_reduction_20261026/README.md)：DINOv2 最终 LayerNorm 参数梯度的原始完整 kernel、简化归约和固定配置对照；原 32×64 SIMT 配置漏算贡献。

- [cicd/load_order_fixed_20261023/](cicd/load_order_fixed_20261023/README.md)：04 load 重排的固定 tiling
  DSL 对照，复制单个 `run.py` 即可比较集中 load 和就近 load 的 UB 编译结果，打印明确的 tiling，
  可选直接启动并检查 eager 数值。
- `max_pool2d_kernel_1_eager_repro.py`：从用户提供的 `_relu_1` kernel 抽取单个 MaxPool2d eager 子图，
  固定 seed 构造原始 `[16,64,112,112]` 连续 NCHW 输入，默认在 NPU 执行 Inductor 编译和 eager 对比。
- `emb_opt_bs400/`：embedding 模型 `bs=400` 的 NPU 动态 shape/group 与静态编译复现，详细用法见子目录 README。
- `simt_template_fused_reduction_precision_repro.py`：融合 add、ReLU、exp、cos 和 sum 的 SIMT template reduction 精度问题复现。
- `rms_norm_simd_multi_reduction_repro.py`：RMSNorm weight grad 的 SIMD 多 reduction 轴 codegen 复现脚本，默认使用原始大 shape。
- `rms_norm_simd_multi_reduction_pass_case.py`：同一表达式的可通过 case，默认使用 `BATCH=2 SEQ=128 HEADS=8 HEAD_DIM=128 CHECK=1`。
- `rms_norm_simd_multi_reduction_manual_tiling.py`：同一表达式的手动 tiling case，默认使用 `R2BLOCK_SUB=7`，覆盖非整除 reduction tile。
- `embedding_backward_cross_device.py`：从 NPU AOT graph 原样还原输入 shape 和 `eager_forward` 算子顺序的标准 PyTorch 片段，支持 CUDA/NPU 的 eager 与 Inductor 性能对比，不依赖 Triton。
- `dlrm_embedding_select_perf.py`：从 DLRM 整网还原一个 `[128,16]` `select + embedding` ATen 片段，支持 CUDA/NPU 的 eager 与 Inductor 对比，不依赖 Triton。
- `dlrm_embedding_dense_backward_perf.py`：复现 DLRM `add + slice + embedding_dense_backward`，拆分报告完整 eager、split eager 和 Inductor split。
- `dlrm_embedding_dense_phases_cross_device.py`：从 0914 DLRM graph 抽取真实 `full` 和融合 atomic scatter
  phase，分别对比 CUDA/NPU 的 Eager 与 Inductor，不依赖 Triton 源码。
- `dlrm_embedding_dense_full_26_cross_device.py`：从 0914 DLRM graph 抽取 26 个真实 full kernel 的
  `[rows, 16]` ATen `full` case，支持按单个 kernel 或全部 26 个 case 对比 CUDA/NPU Eager 与 Inductor。
- `symbolic_mask_perf/`：对比无 `x2` mask、`x2 < 200` 和显式 full-shape 恒真 mask 的三份 SIMT kernel。
- `yolov3_maxpool_channels_last_repro.py`：从 YOLOv3 缓存提取的单个 `max_pool2d`，使用 `torch.compile`
  复现 NPU channels-last 输出 stride 断言失败；默认保留原始 `[8,512,12,16]` 输入。
- `resnet50_batchnorm_channels_last_repro.py`：从 resnet50 训练缓存提取的单个 `batch_norm`，使用
  `torch.compile` 复现 channels-last 输出 stride 断言失败；保留 TorchBench 移除 BN decomposition 的配置，
  默认输入 `[32,64,112,112]`，可用 `SHAPE=1,2,3,3` 缩小输入。
- `tf_efficientnet_b0_pad_channels_last_repro.py`：从 tf_efficientnet_b0 推理缓存提取的单个 `F.pad`，
  使用 `torch.compile` 复现 `constant_pad_nd` 的 channels-last 输出 stride 断言失败；默认输入
  `[128,96,112,112]`，可用 `SHAPE=1,2,3,3` 缩小输入，无需修改 decomposition 或 fallback 配置。
