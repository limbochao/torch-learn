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
| test_combo_kernels.py | test_2d_blocking_benchmark | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: False is not true |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 通过（PASS 1；共 1） | — |
| test_foreach.py | test_single_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_single_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_single_scalar_tensor | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_scheduler_fusion_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_scheduler_fusion_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_broadcasting | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_type_promotion | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_kernel_split_arg_limit_list | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 124. / Absolute difference: 122 |
| test_foreach.py | test_fusion_duplicate_buffer_list | 失败（PASS 9、FAIL 11；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 1. / Absolute difference: 1 |
| test_foreach.py | test_non_foreach_producer_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_non_foreach_producer_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_non_foreach_consumer_producer_list | 失败（FAIL 32；共 32） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_non_foreach_consumer_producer_scalar | 失败（FAIL 16；共 16） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_enable_dynamic_shapes_python_wrapper | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_decomp | 失败（FAIL 2；共 2） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_fuse_concat | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 6. / Absolute difference: 4 |
| test_foreach.py | test_2d_blocking | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_2d_blocking_partitioning_elems | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 3. / Absolute difference: 1 |
| test_foreach.py | test_2d_blocking_partitioning_mixed_sizes | 失败（FAIL 20；共 20） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 3. / Absolute difference: 2 |
| test_foreach.py | test_reinplacing | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_reinplacing_mut_before | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_reinplacing_mut_after | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 2. / Absolute difference: 1 |
| test_foreach.py | test_aliasing | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 4 but got 3. / Absolute difference: 1 |
| test_foreach.py | test_2d_block_no_mixed_sizes_no_mask | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 2 but got 3. / Absolute difference: 1 |
| test_foreach.py | test_2d_block_mixed_sizes_with_mask | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 1 but got 3. / Absolute difference: 2 |
| test_foreach.py | test_foreach_map_backward_binary | 失败（FAIL 10；共 10） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 12. / Absolute difference: 7；kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 9. / Absolute difference: 4 |
| test_foreach.py | test_foreach_map_backward_unary | 失败（FAIL 4；共 4） | kernel 数量断言未满足；AssertionError: Scalars are not equal! / Expected 5 but got 9. / Absolute difference: 4 |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_activations | 通过（PASS 1；共 1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 失败（FAIL 1；共 1） | kernel 数量断言未满足；AssertionError: False is not true |
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
