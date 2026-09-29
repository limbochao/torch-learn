# NPU A5 PTA 2.13 Inductor 用例验证

验证日期：2026-09-29。目标环境为 NPU_A5、conda `pta_213`、Ascend950PR；用例目录已泛化为 `<workspace>/pytorch/test/inductor`。通过 `transfer_to_npu` 后将 `GPU_TYPE` 和 Triton backend 设置为 NPU，并用独立进程运行每个文件。

结果统计：75 通过，3 跳过，20 失败；其中 41 项因本次 driver 修复从失败变为通过。参数化测试按原始 case 名称折叠。

本次本地修复：

- `pytorch_new/torch_npu/contrib/transfer_to_npu.py`：`has_triton` 使用 NPU 检测，在 transfer 初始化时选择 Triton Ascend driver，并将上游 `set_driver_to_gpu` 重定向到 NPU；子进程通过 `TORCH_TRANSFER_TO_NPU=1` 继承该适配。
- `pytorch_new/test/contrib/test_transfer_to_npu.py`：增加 Triton 检测、driver 和 Inductor backend 回归断言。

测试文件本身未修改，因此没有需要复制到本仓库的“修改后完整测试文件”；执行器和清单见 [`run_cases.py`](../../../scripts/tests/npu_inductor_213/run_cases.py) 与 [`cases.tsv`](../../../scripts/tests/npu_inductor_213/cases.tsv)。

| 文件名 | case 名 | 最终结果 | 是否修复 | 修复方式/失败原因 |
|---|---|---|---|---|
| `test_unbacked_symints.py` | `test_expand` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_expand_ok_with_runtime_assert` | 失败 | 否 | NPU aten.nonzero fake/meta kernel 返回 stride=(1,128)，运行时 stride=(2,1)，触发 expand 输出检查。 |
| `test_unbacked_symints.py` | `test_broadcast_tensors` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_autotuning` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_split_with_sizes` | 失败 | 否 | NPU compiled reduction 与 eager 结果数值不一致，最大误差约 1.52。 |
| `test_unbacked_symints.py` | `test_view_of_slice` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_kernel_grid` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_nonzero_in_inference_mode` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_equivalent_backed_unbacked` | 失败 | 否 | Inductor backend assertion，NPU lowering/codegen 未满足该 backed/unbacked 等价性用例。 |
| `test_unbacked_symints.py` | `test_vertical_pointwise_reduction_fusion` | 失败 | 否 | NPU 生成 kernel 数量为 2，用例期望为 1，融合策略与 CUDA 假设不同。 |
| `test_unbacked_symints.py` | `test_mm_and_friends` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_unbacked_range_tree_divisor` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_masked_scatter` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_repeat` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_repeat_interleave_with_unbacked_scalar` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_slice_on_subclass` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_issue_143498` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_einsum` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_softmax` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_sdpfa` | 失败 | 否 | aten._scaled_dot_product_flash_attention 在 NPU 上 fallback 到 CPU，但 CPU backend 没有该算子。 |
| `test_unbacked_symints.py` | `test_sdfpa_unbacked_strides` | 失败 | 否 | 同上；NPU FlashAttention fallback 到 CPU 后算子未实现。 |
| `test_unbacked_symints.py` | `test_unbacked_linear_layer_norm_input` | 失败 | 否 | Inductor backend assertion，NPU layer_norm/unbacked input lowering 失败。 |
| `test_unbacked_symints.py` | `test_to_int_with_unbacked_size` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_combo_kernel_size_hint_failure` | 失败 | 否 | NPU ComboKernel.horizontal_partition API 不接受 kernel_map 参数，torch/torch_npu 接口不匹配。 |
| `test_unbacked_symints.py` | `test_triton_kernel_with_unbacked_symint_fallback` | 失败 | 否 | 无法从 data-dependent expression u0 - 5 提取 specialized integer，NPU fallback 路径未处理该 unbacked SymInt。 |
| `test_unbacked_symints.py` | `test_autotune_with_unbacked_stride` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_fmod_with_out_arg` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_type_mismatch` | 失败 | 否 | NPU Triton/MLIR 编译 f64 lift_fresh kernel 失败。 |
| `test_unbacked_symints.py` | `test_triton_trunc_large_float_scalar_tensor` | 失败 | 否 | NPU 不支持 double，fake tensor 阶段无法将大整数转换为 float64 tensor。 |
| `test_unbacked_symints.py` | `test_triton_trunc_float_scalar_tensor_preserves_positive_zero` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_symbolic_int_exponent` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_symbolic_negative_int_exponent` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_slice_unbacked_bindings_with_later_constraint` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_standalone_compile_reuses_fallback_unbacked_binding` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_compiled` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_compiled_tolist` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_multiple_items` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_cat_sympy_channels_last_contiguous` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_constant_fold_uniform_value_dynamic` | 失败 | 否 | 用例读取 generated source_codes 时得到空列表，NPU wrapper/codegen 路径与 CUDA 断言不兼容。 |
| `test_torchinductor_dynamic_shapes.py` | `test_constant_fold_uniform_value_self_referential_shape` | 失败 | 否 | NPU A5 Triton-Ascend MLIR plan-memory-regbase 编译失败。 |
| `test_torchinductor_dynamic_shapes.py` | `test_arange_dynamic` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_shape_as_constant_reciprocal_float_exp` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_bool_mask_nobreak` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_adaptive_max_pool3d_with_indices` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unwrap_storage_didnt_work_repro` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sym_sum_unbacked` | 失败 | 否 | torch.tensor(y) 需要 materialize 多个 unbacked SymInt，NPU Dynamo 无法 specialized。 |
| `test_torchinductor_dynamic_shapes.py` | `test_nonzero_size_factory_nobreak` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_nonzero_no_realloc` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_bool_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_noops_tensor_repropagate` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_zeros_nobreak` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_return` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_float_item_inf` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_float_item_neginf` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_to_inputs_kernel_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_float_item_return` | 跳过 | 否 | 测试源码使用 unittest.skipUnless(IS_FBCODE)，当前环境不是 FBCODE。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_index_select` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_return_unbacked_view_split` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_matmul` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_save_for_backwards` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_reduction` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_cat_unbacked_duplicate_size` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_cat_backwards` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unbacked_cat_backwards_save_data_dependent` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_embedding_backward_dynamic_shapes_large_grid` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_dynamic_stride_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_multi_output_unbacked_custom_op` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_meta_dynamic_shapes` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_floor` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_pad_dynamic` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_slice_scatter` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_slice_index_changing_sign` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sym_stride_lowering` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_materialize` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_abs` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_float_is_integer` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_magic_method_lowerings_with_symbolic_scalars` | 失败 | 否 | NPU Triton/MLIR 对 float64 pow 组合编译失败。 |
| `test_torchinductor_dynamic_shapes.py` | `test_arithmetic_constant_folding` | 跳过 | 否 | 测试源码使用 @onlyCPU，按设计不在 NPU 执行。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sub_constant_folding` | 跳过 | 否 | 测试源码使用 @onlyCPU，按设计不在 NPU 执行。 |
| `test_torchinductor_dynamic_shapes.py` | `test_full_symbolic_value` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_interpolate_ceil_eq` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_full_recompiles` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_math_ops` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_wrapper_codegen_statically_known_int_or_none` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_unbacked_stride_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_symint_sum_list` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_mark_unbacked_slice` | 失败 | 否 | 用例路径调用 CUDA 专属 torch._C._cuda_cudaCachingAllocator_is_enabled，NPU 不提供该属性。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_operations` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_fallback_specialization` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_softshrink` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_dynamic_rblock_bounds` | 失败 | 否 | FileCheck 查找 CUDA Triton 文本 R0_BLOCK，但 NPU wrapper 生成形态不同。 |
| `test_torchinductor_dynamic_shapes.py` | `test_non_persistent_dynamic_rblock` | 失败 | 否 | NPU 生成 persistent_reduction，和用例要求的非 persistent 断言不一致。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_dynamic` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_fallback_symint_specialization` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sort_dynamic_shape_with_check` | 失败 | 否 | NPU metrics.generated_kernel_count 为 0，而用例按 GPU kernel 计数期望 1。 |
| `test_torchinductor_dynamic_shapes.py` | `test_coalescing_analysis_sympy_is_constant` | 失败 | 否 | 缺少 npu.npu_fusion_attention.default lowering。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sympy_infinity_bounds_in_persistent_reduction` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |

验证证据保存在本地临时目录 `tmp/npu-a5-213-validation`，包含 final JSON 和日志；该目录不纳入归档。
