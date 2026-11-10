# 2.10 原生 Combo/foreach 用例执行结果

执行时间：2026-11-10 12:07:17 至 2026-11-10 13:39:02（北京时间）。

使用用户指定的 `v2.10.0/test_inductor` 测试目录，保留原始参数、配置、数值及结构断言。
仅为 Combo 文件补充与 foreach 一致的 NPU 初始化；测试插件只负责设备初始化、选择和记录。

测试源码提交：`f998b3ddbbed76955e0df794099285d495df5849`。

安装包：`torch_npu 2.10.0.post7.dev20261110`；`torch 2.10.0+cpu`；`triton 3.6.0`（实际导入版本）。

本轮没有叠加生产代码修复。安装源码运行前后 SHA256 一致。

54 个源方法：失败 31，通过 23。
参数化实例共 402 个：`{'failed': 369, 'passed': 33}`。

方法只有在其全部实例通过时才计为 PASS；跳过和执行异常单独记录。
用户预期 51 个方法 PASS，实际为 23 个，差值 -28。旧 2.13 适配版曾删除部分结构断言，结果不直接等价。

失败方法按本轮实际触发的检查分类：数值比较 1 个，kernel 数量断言 29 个，provenance 映射 1 个。
kernel 数量断言失败表示生成数量不符合原版预期；本表仍按原测试判为失败，不能直接归为算子功能不支持或数值错误。

2.10 原版 `test_mean` 不主动开启 Combo，其通过仅说明原始配置下通过。

[完整 CSV 表](inductor-combo-tests-native-v2.10.csv)；[本轮证据](../../../../artifacts/combo-native210-daily20261110/summary.json)。

2026-11-10 定向定位补充：27 个 foreach 方法的失败有共同的入口条件问题。
`NPUTritonScheduling.backend_features` 没有 `BackendFeature.FOREACH`；社区 2.10 的
`make_foreach_pointwise` / foreach_map lowering 只有在此能力为真时才注册 operation list，
供 scheduler 创建 foreach 节点。全局 `combo_kernels=True` 是另一条入口，因此显式开启 Combo 的
测试通过，并不能证明默认 foreach 路径已经启用。

在相同的 `torch._foreach_abs` 输入上做了三个对照，全部通过 eager 数值比较：

| 诊断配置 | 混合尺寸输入 | 相同尺寸输入 |
| --- | --- | --- |
| 当前安装包，默认原生 foreach 路径 | 2 个普通 kernel | 2 个普通 kernel |
| 仅在诊断进程中补 FOREACH 能力 | 1 个 Combo kernel | 1 个 Combo kernel |
| 当前安装包，显式设置 combo_kernels=True | 1 个 Combo kernel | 本轮未测 |

默认混合尺寸 wrapper 确实有两次普通 kernel 调用，显式 Combo 则只有一次融合调用，
因此不是单纯计数器误报。对照仅证明最小用例，尚不能承诺补上能力后全部 27 个方法通过。
生产安装文件未修改，196 个源码文件的 SHA256 与全量测试前一致。

NPU 的 `NPUComboKernel.horizontal_partition` 另外要求子 kernel 的类型、reduction 模式及
轴的 block 参数布局一致，以便共用 launcher/autotuner。这确实是额外的兼容性限制，
但上述 abs 输入满足该条件，启用能力后被放入同一个分区。代码历史只能证明 NPU 显式能力列表
缺少 FOREACH，尚无证据确认维护者是否有意关闭该能力。

两个原生 2D benchmark 在保留断言的定向复测中都得到计数 6，未满足原版要求的 7–8。
追踪确认：benchmark 候选将三个节点放入一个 Combo 分区；正式 codegen 则分成一个 1D 单节点
和一个 2D 双节点分区，Combo 检查成功，没有触发独立 kernel 回退。
`generated_kernel_count` 会累计 benchmark 生成的候选，不能等同于最终启动次数；
本轮追踪中正式 Combo codegen 进入时为 4，完成后为 6。
社区原有分区也要求相同的 1D/2D blocking strategy，不能仅凭这里的分区将原因归于 NPU 新增限制。
本轮未在 CUDA 上实测 7–8 的生成过程，不对其候选数量来源作推断。

[诊断脚本](../../../../scripts/tests/combo_kernel_210/diagnose_native_fusion.py)；
[原包对照](../../../../artifacts/combo-native210-daily20261110/fusion_diagnosis/baseline/results.json)；
[补能力对照](../../../../artifacts/combo-native210-daily20261110/fusion_diagnosis/foreach_enabled/results.json)；
[2D benchmark 追踪](../../../../artifacts/combo-native210-daily20261110/fusion_diagnosis/benchmarks/events.json)。

| 测试文件 | 函数名 | 最新结果 | 当前分析 |
| --- | --- | --- | --- |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 失败（FAIL 2；共 2） | 数值比较失败；AssertionError: Tensor-likes are not close! / Mismatched elements: 288 / 288 (100.0%) / Greatest absolute difference: 5.596200466156006 at index (1, 2, 0, 2) (up to 1e-05 allowed) |
| test_torchinductor.py | test_mean | 通过（PASS 1；共 1） | — |
| test_torchinductor.py | test_rand_like_deterministic | 通过（PASS 2；共 2） | — |
| test_combo_kernels.py | test_activation_functions | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_reduce_functions | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_mutated_args | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_reduce_split | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_2d_blocking_partitioning | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_activation_benchmark | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_reduce_benchmark | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_mutated_benchmark | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_round_robin_dispatch | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_2d_blocking_benchmark | 失败（FAIL 1；共 1） | 定向复测 generated_kernel_count=6，原断言要求 7–8；Combo 已生成且无回退。计数包含 benchmark 候选，最终分区为 1D 单节点及 2D 双节点；不能按该计数直接判定融合未生效 |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 通过（PASS 1；共 1） | — |
| test_foreach.py | test_single_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_single_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_single_scalar_tensor | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_scheduler_fusion_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_scheduler_fusion_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_broadcasting | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_type_promotion | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_kernel_split_arg_limit_list | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 124. / Absolute difference: 122；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_fusion_duplicate_buffer_list | 失败（PASS 9、FAIL 11；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 1. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_non_foreach_producer_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_non_foreach_producer_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_non_foreach_consumer_producer_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_non_foreach_consumer_producer_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_enable_dynamic_shapes_python_wrapper | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_decomp | 失败（FAIL 2；共 2） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_fuse_concat | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 6. / Absolute difference: 4；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_2d_blocking | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_2d_blocking_partitioning_elems | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 3. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_2d_blocking_partitioning_mixed_sizes | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 3. / Absolute difference: 2；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_reinplacing | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_reinplacing_mut_before | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_reinplacing_mut_after | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_aliasing | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 4 but got 3. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_2d_block_no_mixed_sizes_no_mask | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 3. / Absolute difference: 1；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_2d_block_mixed_sizes_with_mask | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 3. / Absolute difference: 2；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_foreach_map_backward_binary | 失败（FAIL 10；共 10） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 12. / Absolute difference: 7；kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 9. / Absolute difference: 4；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_foreach.py | test_foreach_map_backward_unary | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 9. / Absolute difference: 4；后端未声明 FOREACH，原生 foreach 分组入口未启用；补能力后的本方法结果尚未复测 |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_activations | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 失败（FAIL 1；共 1） | 定向复测 generated_kernel_count=6，原断言要求 7–8；Combo 已生成且无回退。计数包含 benchmark 候选，最终分区为 1D 单节点及 2D 双节点；不能按该计数直接判定融合未生效 |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_mutated | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_activations_no_autotune | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking_round_robin | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 通过（PASS 1；共 1） | — |
| test_torchinductor.py | test_combo_kernel_filter_cpu | 通过（PASS 1；共 1） | — |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（FAIL 1；共 1） | provenance 映射检查失败；AssertionError: The length of the sequences mismatch: 0 != 1 |
| test_combo_kernels.py | test_helper_fn_defined | 通过（PASS 1；共 1） | — |
