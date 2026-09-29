# NPU A5 PTA 2.13 Inductor 用例验证

验证日期：2026-09-29。目标环境为 NPU_A5、conda `pta_213`、Ascend950PR；用例目录已泛化为 `<workspace>/pytorch/test/inductor`。通过 `transfer_to_npu` 后将 `GPU_TYPE` 和 Triton backend 设置为 NPU，通过执行器运行清单中的 NPU 参数化实例。

结果统计（2026-09-29 修正代码全量复跑）：75 通过，3 跳过，20 失败；其中 41 项因本次 driver 修复从失败变为通过。全量执行了 115 个 NPU 参数化实例，按清单折叠为 98 个逻辑 case。

复核结论：执行器只把对应 NPU 参数化实例的 `call=passed` 计为通过；skip、无 call、setup/teardown 失败均不会计入通过。98 个逻辑 case 全部成功收集，75 个通过项均有 `_npu` nodeid 且所有 call 阶段通过。`test_unbacked_symints.py::test_expand` 实际 nodeid 为 `TestUnbackedSymintsPRIVATEUSE1::test_expand_npu`，`call=passed`，不是 skip；相邻的 `test_expand_ok_with_runtime_assert_npu` 为失败。3 个跳过项为 `test_arithmetic_constant_folding`、`test_float_item_return` 和 `test_sub_constant_folding`，均为测试自身的 CPU/FBCODE 条件。

原会话本地修复：

- `pytorch_new/torch_npu/contrib/transfer_to_npu.py`：`has_triton` 使用 NPU 检测，在 transfer 初始化时选择 Triton Ascend driver，并将上游 `set_driver_to_gpu` 重定向到 NPU；子进程通过 `TORCH_TRANSFER_TO_NPU=1` 继承该适配。
- `pytorch_new/test/contrib/test_transfer_to_npu.py`：增加 Triton 检测、driver 和 Inductor backend 回归断言。

测试文件本身未修改，因此没有需要复制到本仓库的“修改后完整测试文件”；执行器和清单见 [`run_cases.py`](../../../scripts/tests/npu_inductor_213/run_cases.py) 与 [`cases.tsv`](../../../scripts/tests/npu_inductor_213/cases.tsv)。

失败归因补充：以下“否”表示该项尚未修复，不表示已经证明不能修复。原会话完成了 driver 适配和全量复跑，
没有为剩余每项提供修复后的复测证据。本次对照最终 traceback、当时的测试源码及目标环境安装源码补充分类，
没有重新执行 NPU 测试，也没有把修改断言、降低 dtype 或关闭被测功能视为原用例修复成功。

20 个失败按当前处置方向分为：11 项待修复 bug（其中部分责任层仍待定位）、4 项当前路径的功能/类型支持缺口、
5 项测试预期或代码采集待适配。NPU 机制差异会在对应项单独说明；它不自动意味着硬件无法实现。
3 个 skip 仅表示原测试条件不满足，不能证明 NPU 通过或不支持。详细证据和未修复原因见表后的分类说明。

| 文件名 | case 名 | 最终结果 | 是否修复 | 修复方式/失败原因 |
|---|---|---|---|---|
| `test_unbacked_symints.py` | `test_expand` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_expand_ok_with_runtime_assert` | 失败 | 否（待修复） | 【meta/运行时布局契约 bug】nonzero 的 fake stride=(1,128)，NPU 实际=(2,1)。NPU 输出布局差异必须由 meta/运行时契约处理；需修 torch_npu/算子集成并复测，不能关掉 stride 检查冒充修复。见 [F1](#f1)。 |
| `test_unbacked_symints.py` | `test_broadcast_tensors` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_autotuning` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_split_with_sizes` | 失败 | 否（待定位） | 【正确性 bug，根因待定位】同一输入的 compiled/eager 标量和相差 1.51525，远超 1e-5 容差。需定位 split/slice/reduction 生成代码；尚无错误指令或有效修复证据，不能归因于正常 NPU 精度差异。见 [F2](#f2)。 |
| `test_unbacked_symints.py` | `test_view_of_slice` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_kernel_grid` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_nonzero_in_inference_mode` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_equivalent_backed_unbacked` | 失败 | 否（待修复） | 【transfer 适配 bug】pad_mm benchmark 的默认设备枚举把 CUDA 别名和 NPU 重复计入，触发 len(avail_gpus)<=1；并非 backed/unbacked lowering 不支持。可修 transfer 的设备选择，原会话未修复复测。见 [F3](#f3)。 |
| `test_unbacked_symints.py` | `test_vertical_pointwise_reduction_fusion` | 失败 | 否（待适配） | 【融合策略差异/测试待适配】当前输入数值比较已通过，失败仅为 kernel 数 2!=1。不能据此认定 NPU 无法融合；需先确认 NPU 融合目标，再适配计数或修 scheduler，尚未验证。见 [F4](#f4)。 |
| `test_unbacked_symints.py` | `test_mm_and_friends` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_unbacked_range_tree_divisor` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_masked_scatter` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_repeat` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_repeat_interleave_with_unbacked_scalar` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_unbacked_slice_on_subclass` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_issue_143498` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_einsum` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_softmax` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_sdpfa` | 失败 | 否（当前接口不支持） | 【功能不支持：指定 ATen 接口】aten._scaled_dot_product_flash_attention 没有 NPU 专用实现，CPU fallback 也无此算子。需实现/桥接该接口及其返回契约；换高层 SDPA 不等于原接口测试通过。见 [F5](#f5)。 |
| `test_unbacked_symints.py` | `test_sdfpa_unbacked_strides` | 失败 | 否（当前接口不支持） | 【功能不支持：指定 ATen 接口】同 [F5](#f5)；还需保留 unbacked stride 语义。当前缺少该 ATen 接口实现，不能用换算子或取消动态 stride 作为原用例修复。 |
| `test_unbacked_symints.py` | `test_unbacked_linear_layer_norm_input` | 失败 | 否（待修复） | 【transfer 适配 bug】与 [F3](#f3) 相同，失败在 pad_mm 的 benchmark 设备枚举断言；没有到达证明 layer_norm lowering 失败的位置。可修设备去重/选择，尚未复测。 |
| `test_unbacked_symints.py` | `test_to_int_with_unbacked_size` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_combo_kernel_size_hint_failure` | 失败 | 否（待修复） | 【torch/torch_npu 接口兼容 bug】NPU 仍传 kernel_map，目标 torch 的 horizontal_partition 已改为 NodeInfo 接口。需同步 NPU ComboKernel 调用及元数据，超出 transfer/测试适配范围；关闭 combo 会取消被测功能。见 [F6](#f6)。 |
| `test_unbacked_symints.py` | `test_triton_kernel_with_unbacked_symint_fallback` | 失败 | 否（待定位） | 【动态图/transfer 集成 bug，责任层待定位】失败在 torch.tensor([1,u0-5]) 的 Dynamo 捕获，尚未进入被测 Triton fallback。需区分包装函数与原生 SymInt 构造路径；不能据用例名判定 NPU fallback 不支持。见 [F7](#f7)。 |
| `test_unbacked_symints.py` | `test_autotune_with_unbacked_stride` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_fmod_with_out_arg` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_type_mismatch` | 失败 | 否（当前类型不支持） | 【功能不支持：当前 A5 编译路径的 FP64 broadcast/store】原测试明确要求 float64；lift_fresh 的 f64 vbrc 被编译器类型校验拒绝。改成 float32 会丢失原测试覆盖，需后端补支持或另列 NPU 用例。见 [F8](#f8)。 |
| `test_unbacked_symints.py` | `test_triton_trunc_large_float_scalar_tensor` | 失败 | 否（待定位） | 【大整数构造 bug 待定位，另有 FP64 限制】当前首错为 fake tensor 无法将 Python int 转到 C++；输入产生 2**70。不能仅凭此错误断言由 NPU double 限制导致；需 eager/compile 和包装前后对照。见 [F9](#f9)。 |
| `test_unbacked_symints.py` | `test_triton_trunc_float_scalar_tensor_preserves_positive_zero` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_symbolic_int_exponent` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_triton_pow_symbolic_negative_int_exponent` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_slice_unbacked_bindings_with_later_constraint` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_standalone_compile_reuses_fallback_unbacked_binding` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_compiled` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_compiled_tolist` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_unbacked_symints.py` | `test_override_optimization_hint_multiple_items` | 通过 | 否 | 通过。 |
| `test_unbacked_symints.py` | `test_cat_sympy_channels_last_contiguous` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_constant_fold_uniform_value_dynamic` | 失败 | 否（待定位/适配） | 【测试代码采集问题待定位】run_and_get_code 返回空列表，访问 source_codes[0] 越界；尚未执行后面的数值断言。需确认空图消除或采集遗漏，再适配检查；不能直接认定常量折叠不支持或数值已通过。见 [F10](#f10)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_constant_fold_uniform_value_self_referential_shape` | 失败 | 否（待修复） | 【NPU UB 约束触发的编译配置/分块 bug】fill kernel 在 plan-memory-regbase 阶段要求 6815744 bits，预算仅 1769472 bits。需修 tiling/内存规划并验证有效配置；硬件片上容量有限不等于 fill 功能不支持。见 [F11](#f11)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_arange_dynamic` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_shape_as_constant_reciprocal_float_exp` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_bool_mask_nobreak` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_adaptive_max_pool3d_with_indices` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unwrap_storage_didnt_work_repro` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sym_sum_unbacked` | 失败 | 否（待定位） | 【动态图/transfer 集成 bug，责任层待定位】sum(tolist()) 的 unbacked SymInt 经 torch.tensor(y) 构造时报取常量错误。尚未做包装前后对照，不能认定 NPU 不支持动态求和；需修捕获/构造路径。见 [F7](#f7)。 |
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
| `test_torchinductor_dynamic_shapes.py` | `test_float_item_return` | 跳过 | 否（未执行） | 【测试环境条件】@unittest.skipUnless(IS_FBCODE) 在当前开源环境跳过；不代表 NPU 不支持 item。需确认条件原因后构造等价开源测试，尚未验证。见 [S2](#s2)。 |
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
| `test_torchinductor_dynamic_shapes.py` | `test_magic_method_lowerings_with_symbolic_scalars` | 失败 | 否（待修复） | 【codegen bug】float32 输入的符号整数幂被生成为 libdevice.pow(float64,float64)，而当前 CANN libdevice 无此签名。需修符号表达式生成并验证数值范围，不能泛称用例要求 FP64 或魔术方法不支持。见 [F12](#f12)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_arithmetic_constant_folding` | 跳过 | 否（未执行） | 【测试范围：CPU-only】@onlyCPU 明确排除 NPU，不是 NPU 失败或通过。NPU 版本可另行设计并实测；仅去掉装饰器不能证明支持。见 [S1](#s1)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sub_constant_folding` | 跳过 | 否（未执行） | 【测试范围：CPU-only】@onlyCPU 明确排除 NPU。原测试未执行，尚无 NPU 修复/验证结果；需保留常量折叠覆盖设计 NPU 测试。见 [S1](#s1)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_full_symbolic_value` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_interpolate_ceil_eq` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_full_recompiles` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_math_ops` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_wrapper_codegen_statically_known_int_or_none` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_item_unbacked_stride_nobreak` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_symint_sum_list` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_mark_unbacked_slice` | 失败 | 否（待修复） | 【transfer/CUDA Graph 适配 bug】reduce-overhead 进入 CUDA allocator 检查，误调用 NPU 环境缺失的私有 CUDA C 接口。可适配 NPU graph/allocator 检查；关闭该模式会改变原覆盖，尚未修复验证。见 [F13](#f13)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_operations` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_fallback_specialization` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_softshrink` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_dynamic_rblock_bounds` | 失败 | 否（待适配） | 【NPU codegen 机制差异/测试待适配】首组数值比较已通过，失败在固定匹配 R0_BLOCK: tl.constexpr = 64。需检查 NPU 实际 reduction block 和动态边界，不能只改字符串判通过；后续分支未执行。见 [F14](#f14)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_non_persistent_dynamic_rblock` | 失败 | 否（待适配） | 【NPU reduction 策略差异/测试待适配】NPU 选择 persistent_reduction，与测试强制非 persistent 的文本预期冲突。失败发生在 allclose 之前；需确认 NPU 预期并补数值/边界验证，不能称已通过。见 [F14](#f14)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_dynamic` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_unspecialized_float_fallback_symint_specialization` | 通过 | 否 | 通过。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sort_dynamic_shape_with_check` | 失败 | 否（待适配） | 【NPU 算子执行路径差异/测试待适配】首组排序结果正确；NPU sort.stable 走算子 fallback，生成 kernel 计数为 0，测试预期 1。需以 NPU 路径验证动态排序/复用，后续分支尚未执行。见 [F15](#f15)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_coalescing_analysis_sympy_is_constant` | 失败 | 否（待补齐支持） | 【编译功能缺口：待补 lowering】eager SDPA 已执行，compiled 图中的 npu_fusion_attention 无 lowering/decomposition。需补 torch_npu Inductor 注册及动态 shape 支持；不能说 NPU 没有 attention 算子。见 [F16](#f16)。 |
| `test_torchinductor_dynamic_shapes.py` | `test_sympy_infinity_bounds_in_persistent_reduction` | 通过 | 是 | transfer_to_npu 的 NPU Triton driver 选择修复后通过。 |

## 归因证据与修复边界

[`failure-evidence.json`](../../../artifacts/npu-inductor-213-validation-20260929/failure-evidence.json)
保留全部 98 项的最终状态，以及 20 个失败、3 个跳过的 nodeid、失败阶段、去敏 traceback 和原测试函数。
[`source-evidence.json`](../../../artifacts/npu-inductor-213-validation-20260929/source-evidence.json)
保留归因使用的函数和注册片段，包括来源、原文件行号和文件 SHA-256。
Python 实现取自目标 `pta_213` 环境的安装源码；用于机制对照的本地 C++ 片段已单独标注。
安装源码在本次复核时只读采集，不假定本地开发分支与当时的 wheel 完全一致。

下文的“可修复”表示有明确修复方向，尚不表示方案已验证有效。“待修复 bug”是问题处置状态，
不表示已经创建上游 Issue。责任层尚未确定的项会明确标注，不能把尚未定位写成 NPU 固有限制。

几个术语：fake/meta kernel 只推导张量形状、dtype 和 stride，不实际计算；lowering 将算子转为后端执行形式；
unbacked SymInt 是依赖运行时张量数据、编译时没有具体值的符号整数；fallback 表示改由已有算子实现执行。
fallback 到 NPU 算子与 fallback 到 CPU 是不同路径，后者也要求 CPU 存在相应实现。

<a id="f1"></a>
### F1：nonzero 布局差异引出的 meta 契约 bug

`test_expand_ok_with_runtime_assert` 的首错来自 `aten.nonzero.default` 的输出 stride 检查，
不是 expand 算术失败。目标环境 `torch/_subclasses/fake_impls.py::nonzero` 返回
`new_empty_strided((nnz, arg.dim()), (1, nnz), dtype=torch.int64)`；当 `nnz=128`、输入 rank=2 时，
预测 stride 为 `(1,128)`。最终日志中的实际 stride 为 `(2,1)`。
本地 `NonzeroKernelNpuOpApi.cpp::nonzero` 使用常规输出分配，也与这种行连续布局一致。

这是 NPU 运行时布局与编译期描述不一致的集成 bug。可以通过设备感知的 fake/meta 实现或一致的输出布局契约修复；
具体应修改哪一层仍需 opcheck、动态 shape 和非连续输入验证。原会话未完成该修复，
不能用关闭检查或在测试中加 contiguous 掩盖框架契约错误。

<a id="f2"></a>
### F2：split 后求和的正确性 bug，具体生成错误待定位

`test_split_with_sizes` 用同一组输入运行 eager 和 compiled：`x.shape=(32,)`，split sizes 为 `(7,16,9)`，
返回第一个 split 的和及 size 之和。最终日志中第一个返回值为 `-6.655652046203613`，
eager 为 `-5.14039945602417`，绝对差 `1.5152525901794434`，允许值为 `1e-5`。

现有证据足以确认原测试的正确性检查失败，但尚未保存能说明错误索引、mask 或 reduction 的完整生成 kernel。
因此列为待定位的正确性 bug，不能写成正常的 NPU 浮点误差，也不能声称已经定位到 reduction 指令。
需固定随机输入、保存生成 kernel、逐项比较 split 边界和归约结果后修后端；提高容差不构成修复。

<a id="f3"></a>
### F3：benchmark 重复识别 CUDA 别名与 NPU，可修复的 transfer bug

`test_equivalent_backed_unbacked` 和 `test_unbacked_linear_layer_norm_input` 均在
`pad_mm._should_pad → benchmark_gpu → _get_default_gpu_device_type` 失败。
目标 `benchmarking.py:32` 枚举可用设备并断言 `len(avail_gpus) <= 1`。
`torch_npu/_inductor/__init__.py` 将 `GPU_BENCHMARK_DEVICE_TYPES` 扩展为 `("cuda", "xpu", "mtia", "npu")`，
而 `transfer_to_npu._patch_cuda` 把 `torch.cuda` 替换为 NPU 模块。
在 NPU 可用时，`cuda` 和 `npu` 两个名字因此都返回可用，违反了枚举逻辑的单设备假设。

修复方向是在 transfer 模式中为 benchmark 明确选择 NPU，或去掉指向同一后端的设备别名，
同时保持正常 NPU compile 的设备注册。原会话只修复了 Triton driver，没有修复此处并复测。
这两项在原允许的适配范围内，不应解释成“无法修复”或 layer_norm/unbacked 功能不支持；
当前失败也不足以判断消除此阻塞后是否还有后续问题。

<a id="f4"></a>
### F4：融合数量不同，需要区分测试适配与优化缺口

`test_vertical_pointwise_reduction_fusion` 的 `assert_close(actual, expected)` 在 kernel 数检查之前，
最终失败位置是 `generated_kernel_count` 的 `2 != 1`，因此该输入的数值检查已经通过。
它说明当前 NPU 编译路径没有得到原测试要求的单 kernel；尚无 scheduler 拒绝原因或硬件约束证据，
不能说 NPU 必须生成两个 kernel。

如果 NPU 的验收要求允许当前分解方式，可添加有依据的后端计数断言并保留数值检查；
如果该测试仍要求验证单 kernel 融合，就应跟进 scheduler 优化缺口。原会话未确认这项预期并复测，
所以保留失败、标为待适配，不能直接把期望值改为 2 后认定原融合目标已实现。

<a id="f5"></a>
### F5：指定 FlashAttention ATen 接口未实现

`test_sdpfa` 和 `test_sdfpa_unbacked_strides` 直接调用低层
`aten._scaled_dot_product_flash_attention.default`，并不只是调用高层 SDPA。
日志的 PrivateUse1 注册项为 `VariableFallbackKernel.cpp` 中的 backend fallback，
随后报 CPU 没有该算子。本地 `npu_cpu_fallback` 调用 `at::native::cpu_fallback`，说明设备名替换
不能补出缺失的算子实现。

限制是当前构建没有这个 ATen 接口的 NPU 实现，不能扩大为“NPU 不支持 FlashAttention”。
原样通过需要桥接/实现该接口的输出、附加返回值和动态 shape/stride 契约，属于后端功能开发。
换成高层 SDPA 或 NPU 专用 attention API 可以另建适配用例，但改变了被测接口；
后一项还必须保留动态 stride 覆盖，不能据替代路径通过宣称原用例修复。

<a id="f6"></a>
### F6：ComboKernel 接口版本不匹配，待修复兼容 bug

目标 `torch_npu/_inductor/codegen/scheduling.py::generate_combo_kernel_code` 仍传递
`kernel_map=subkernel_map`，但安装的 `torch/_inductor/codegen/triton_combo_kernel.py::horizontal_partition`
已不接受该参数，改用 `node_info_map: dict[BaseSchedulerNode, NodeInfo]`。日志正是
`unexpected keyword argument 'kernel_map'`，尚未进入原用例关注的 size hint 处理。

需要同步 NPU 调用方与新 `NodeInfo` 数据结构，不能只删除一个关键字就假定兼容。
这是 torch/torch_npu 版本集成 bug，有明确修复方向；它超出原会话允许的 transfer/测试文件适配范围，
因此未修复。关闭 `combo_kernels` 或 benchmark 会取消测试目标，不能算通过。

<a id="f7"></a>
### F7：构造含 unbacked SymInt 的 tensor 失败，责任层待定位

`test_triton_kernel_with_unbacked_symint_fallback` 在 `torch.tensor([1, u0 - 5], device=device)` 失败；
`test_sym_sum_unbacked` 在 `torch.tensor(sum(a.tolist()))` 失败。两者均为 Dynamo 捕获阶段的
`Could not extract specialized integer`，栈包含 `transfer_to_npu.py:194` 的 `decorated → fn(*args, **kwargs)`。
transfer 的白名单确实包装了 `torch.tensor`，但出现包装栈本身不证明包装器就是根因。

两项列为待修复的动态图/transfer 集成 bug，尚需用同一构造表达式比较包装前后、CPU/NPU 捕获路径，
判断是 Dynamo 对包装函数的识别问题，还是后端的符号标量构造支持缺口。
原会话没有完成此对照，不能写成“NPU 天生不支持 unbacked SymInt”，也不能把前一项归因为尚未执行到的
Triton fallback。固定 `u0`、把数据改成 Python 常量或允许 graph break 都会改变原覆盖。

<a id="f8"></a>
### F8：原测试要求 FP64，当前 A5 编译路径缺少对应类型支持

`test_triton_pow_type_mismatch` 明确返回 `dtype=torch.float64`。
最终首个失败 kernel 为 `triton_poi_fused_lift_fresh_0`；其 IR 用 `f64` 常量广播到 `tensor<1xf64>`，
`hivm.hir.vbrc` 的类型校验明确排除了 f64。失败发生于当前 A5 编译路径的 FP64 广播/写出，
并非已经重现原测试描述中的混合类型 pow 错误。

保持原 dtype 和回归目标需要后端提供该类型路径；transfer 的设备转换不能补齐编译器能力。
改用 float32 会改变原测试对 float32/float64 的覆盖，只能作为另一个 NPU 用例。
此结论限定于当前版本和失败路径，不据此宣称所有 NPU 产品、所有执行方式都永久无法处理 float64。

<a id="f9"></a>
### F9：大整数到 tensor 的转换首错，不能直接归因于 FP64

`test_triton_trunc_large_float_scalar_tensor` 在输入长度 4 时计算
`math.trunc(math.sqrt(4) ** 70) = 2**70`，随后构造 float64 tensor。
日志首错是 fake tensor 调用无法把 Python `int` 转为 C++ 类型，还没有进入设备 kernel 编译。
原报告直接写“NPU 不支持 double”无法解释这个具体阶段，应撤回该单一根因结论。

当前列为大整数构造 bug 待定位：需检查 eager 是否也失败、包装前后是否一致，以及标量转换在哪层发生。
FP64 仍是这个用例后续需要处理的独立约束，F8 只证明另一个 kernel 的类型限制，不能替代本项根因证据。
原会话未做这些对照，因此尚未修复；把指数缩小或把 Python int 预先转为 float 会避开原回归目标。

<a id="f10"></a>
### F10：生成代码采集为空，尚不能判断是哪一种后端差异

`test_constant_fold_uniform_value_dynamic` 在 `FileCheck(...).run(source_codes[0])` 越界。
目标 `run_and_get_code` 通过拦截 `GraphLowering.save_output_code` 收集字符串；空列表仅说明此次没有
捕获到该回调的输出。原测试的常量折叠会把函数化简为直接返回输入，空图消除或代码采集路径差异都需核对，
不能仅凭空列表就确认某一种原因。

可修复方向是确认实际图和调用路径，再对无 kernel 的情况增加准确的检查，并保留所有输出断言。
原测试在数值比较之前失败，不能写数值已通过。原会话未定位采集原因、未适配复测，故保留失败。

<a id="f11"></a>
### F11：UB 容量约束触发编译失败，分块/配置问题待修复

`test_constant_fold_uniform_value_self_referential_shape` 在编译 `triton_poi_fused_fill_1` 时失败。
最终日志不仅有 `PlanMemoryRegBase`，还明确写出：

```text
ub overflow, requires 6815744 bits while 1769472 bits available!
```

UB 是 NPU 向量计算使用的片上缓冲区；此数值是该编译配置的需求和可用预算，不是对整卡内存的描述。
IR 中存在 `memref<65536xi64, ...ub>` 等大块临时数据。片上容量约束是真实机制，但普通 fill 可以通过
分块处理，不能由某组配置超预算推出这个算子或动态 shape 功能无法支持。

当前列为待修复的编译配置/分块 bug。需要检查所有候选 tile 的临时数据存活和内存规划，
确认为什么没有可编译配置，再修 torch_npu tiling 或 Triton/BiSheng 后端。
现有日志尚不能把责任精确分配给其中一层；原适配范围也不包含这些后端修改。
缩小输入或忽略失败配置只能帮助定位，未经原尺寸正确性和生成代码验证不能算修复。

<a id="f12"></a>
### F12：符号整数幂误生成 FP64，待修复 codegen bug

`test_magic_method_lowerings_with_symbolic_scalars` 的失败子函数是
`x + 2 ** floor(log2(x.shape[0]) + 1)`，输入由 `torch.randn(7,5)` 创建，并没有显式要求 float64。
日志中的生成表达式却调用 `libdevice.pow(tl.full([], 2.0, tl.float64), ...to(tl.float64))`，
随后报 `KeyError: (triton.language.float64, triton.language.float64)`。
目标 `triton/language/extra/cann/libdevice.py::pow` 没有 FP64 签名，与该失败直接对应。

因此这里应跟进 NPU 符号表达式 codegen 对后端支持类型的适配，而不能照搬 F8 的“原用例要求 FP64”结论。
需要选择保持整数幂语义的生成方式并验证动态范围、溢出和舍入；简单把所有 float64 替换为 float32
还不能证明语义正确。原会话没有修复 codegen 并复测，故标为待修复 bug。

<a id="f13"></a>
### F13：reduce-overhead 误入 CUDA allocator 私有接口

`test_mark_unbacked_slice` 使用 `torch.compile(..., mode="reduce-overhead", fullgraph=True)`。
目标 `check_caching_allocator_for_cudagraphs` 先检查 `torch.cuda.is_available()`，再直接调用
`torch._C._cuda_cudaCachingAllocator_is_enabled()`。transfer 让前一个检查代表 NPU 可用性，
但它不会给 CPU 构建的 `torch._C` 添加 CUDA 专有 C 接口，因此发生 AttributeError。

这是 transfer 与上游 CUDA Graph 检查之间的适配 bug，不能推出 NPU 不支持 graph capture 或 unbacked slice。
可以在 transfer 模式中衔接 NPU 的 graph/allocator 检查，并验证 capture/replay 语义；
原会话尚未修复此路径。换成默认 compile mode 仅能验证另一条路径，不能宣称原用例修复。

<a id="f14"></a>
### F14：reduction block 与 persistent 策略预期待适配

`test_dynamic_rblock_bounds` 首组输入的 eager/compiled 比较已经通过，失败于精确搜索
`R0_BLOCK: tl.constexpr = 64`。日志只展示了被搜索代码的开头，不能由此确认实际 block 的具体数值；
需取得完整 kernel 检查 NPU 的轴、block 和动态边界。后续分支尚未执行。

`test_non_persistent_dynamic_rblock` 的生成代码明确包含 `persistent_reduction`。
目标 NPU `should_use_persistent_reduction` 使用自己的 reduction hint 和阈值；
对 A5 的 DEFAULT/INNER 阈值为 4096，而原测试的 reduction dim 受限于 6..64，
这与当前选择 persistent 的结果一致。persistent 指归约维在单个 tile 中处理，不能简单套用其他后端的选择预期。

这两项需要保留动态范围和数值检查，按 NPU 的设计目标适配 kernel 结构断言；若仍要测试非 persistent 路径，
应通过 NPU 支持的策略控制触发该路径再验证。第二项在 `allclose` 之前失败，不能称其数值通过。
原会话未完成这些适配，两个 case 均保持失败；仅删除 FileCheck 会丢失原先的结构覆盖。

<a id="f15"></a>
### F15：sort 使用已有 NPU 算子，kernel 计数预期待适配

`test_sort_dynamic_shape_with_check` 第一组升序排序的数值/索引比较已通过，随后 `check_count(1)` 得到 0。
目标 NPU `lowering_fallback_list.py` 注册了 `aten.sort.stable`，这与通过已有算子执行、没有生成
Inductor Triton kernel 的计数结果一致。生成 kernel 数为 0 不能解释为排序没有执行或排序功能不支持。

可以针对 NPU 的算子执行路径适配计数，并继续验证升降序、动态尺寸和编译复用。
如果要求的是生成并复用 persistent sort kernel，则还涉及后端优化能力，而不是简单更改断言。
原会话未适配；首次计数检查之后的分支没有执行，不能把整项视为已验证通过。

<a id="f16"></a>
### F16：attention 的 Inductor lowering 支持缺口

`test_coalescing_analysis_sympy_is_constant` 先执行 eager，再执行 compiled；失败发生于后者的
`MissingOperatorWithoutDecomp: npu.npu_fusion_attention.default`，输入为 float16、shape `[29,50,32,5]`。
这证明当前编译图缺少该算子的 lowering/decomposition，不能写“NPU 没有 attention 算子”，
也不能说已经触发测试名称描述的 SymPy coalescing 问题。

原样通过需要补齐 torch_npu Inductor 的算子注册/执行路径及动态 shape 支持，属于后端功能缺口，待补齐。
原会话允许的 transfer/测试适配不足以完成这项开发；切换 attention 实现或关掉相关融合后通过，
仍需单独说明覆盖变化，不能把原图的 lowering 缺口标为修复。

<a id="s1"></a>
### S1：两个 CPU-only 跳过项

`test_arithmetic_constant_folding` 与 `test_sub_constant_folding` 的原源码均有 `@onlyCPU`，
最终报告为 `Skipped: Only runs on cpu`。这是测试定义的范围，既不是 NPU bug 证据，也不是不支持证据。
如果需要 NPU 覆盖，应构造保留常量折叠断言的 NPU 变体并实际执行；原会话没有提供这种验证，保持跳过。

<a id="s2"></a>
### S2：FBCODE 条件跳过项

`test_float_item_return` 使用 `@unittest.skipUnless(IS_FBCODE, "")`，当前环境不满足条件。
FBCODE 是测试中的内部构建环境标记；本次没有据此推导 NPU 的 item 支持情况。
应先核对该条件背后的依赖，再决定是否增加开源环境中的等价测试。仅伪造环境标记或删除装饰器不能替代复测。

## 验证范围

本次补充核对了 98 项表格与最终 JSON、23 项非通过结果及其测试代码，并保存了可复查的去敏证据；
没有实施新的功能修复或 NPU 复测，统计仍为 75 通过、3 跳过、20 失败。
本地检查包括 JSON 解析、表格/清单/证据逐项一致性、文档链接、敏感路径扫描和 `git diff --check`。
需要重新验证单项时，在原目标环境的 Docker 工作目录激活 `pta_213` 后使用：

```bash
TORCH_NPU_DEVICE_CAPABILITY=8.0 python <torch-learn>/scripts/tests/npu_inductor_213/run_cases.py \
  --case-dir <workspace>/pytorch/test/inductor \
  --manifest <torch-learn>/scripts/tests/npu_inductor_213/cases.tsv \
  --file test_unbacked_symints.py --case test_expand_ok_with_runtime_assert \
  --output <workspace>/case-result.json
```

`--file`、`--case` 替换为表中目标项；只有对应 NPU 实例实际执行且检查通过，才能更新最终结果。
