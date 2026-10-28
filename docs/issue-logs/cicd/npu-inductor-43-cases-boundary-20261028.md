---
title: NPU Inductor 2.13：43 项上游用例复现与定界
published: false
---

# NPU Inductor 2.13：43 项上游用例复现与定界

本记录按任务输入的 43 项顺序保留基线、适配/候选复测和组件结论。**调查仍在进行**，
`PENDING`、`RUNNING` 和超时均不算通过；同一方法的全部参数实例完成后才更新终态。
历史问题说明只用于确定测试范围，当前结论以本次记录为准。

当前26个方法完整通过（含适配后的明确负向检查）；其余17项保留实际失败或能力/规格结论。
第18、21项的后续低精度错误及第34项的事件/上下文错误尚未完成根因闭环，不能将本报告视为全部调查完成。

## 环境与执行范围

2026-10-28，Docker 内独立 conda 环境，Ascend950PR_9579，torch `2.13.0+cpu`、
torch_npu `2.13.0.dev20261028`、Triton `3.6.0`，CANN 9.1.0、driver 25.7.rc1。
测试原始提交 `d70307f9ceeedee7156135bb2c198d72095f8353`。

初始基线只暴露一个设备，编译线程和 NPU precompile worker 均设为 2；后续长用例使用16个预编译worker和4个编译线程对照，不改变候选、输入或断言。每个方法独立解释器、
独立 Inductor cache。适配树与源码候选均隔离运行，共享安装包未修改。
当前基线由 [run_boundary_cases.py](../../../scripts/tests/npu_inductor_213/run_boundary_cases.py)
和 [run_batch.py](../../../scripts/tests/npu_inductor_213/run_batch.py) 记录 setup/call/teardown。

测试收集前调用 `_dynamo.patch_has_triton()`，使 NPU 的可用性判断先于 `inductor_utils` 缓存，
避免 harness 提前导入造成假跳过。该初始化对基线和适配复测一致。
`test_conv_bn_eval` 是嵌套 helper，实际执行完整外层 `test_basic_npu`；不是只选择一个不存在的 pytest 节点。
多设备用例已在物理卡1、3上完成原方法复测；其中两项通过，一项在候选预运行的事件计时失败。

以下“适配复测”列也包含修复候选和延长时限对照，具体改动见结论列和对应证据。
整包 Python overlay 的首次探索遇到 worker context 问题，因此已用原安装布局的对照与
逐模块 source loader 复核；这些探索失败保留为失败尝试，不当作原用例的新终态。
候选源码和基线的逐文件 SHA 已核对；子解释器继承同一 source loader。

## 按输入顺序的结果

| 序号 | 文件 / 用例 | 基线 | 适配复测 | 组件 / 分类 | 原因 | 修复尝试与结论 |
|---:|---|---|---|---|---|---|
| 1 | `test_alignment.py`<br>`test_no_align_for_custom_op_2d` | 1 FAILED | 1 PASSED | 测试 custom-op 注册 / 用例适配，已验证 | define_custom_op_for_test 未给同一 fn 注册 PrivateUse1；基线输出 stride 1024 与 Meta 1040 不一致，错误 Meta 用例先被该断言截住。 | 32609ed 补注册 PrivateUse1，63565c7 适配树无 overlay 对照通过；保留 stride/offset 和错误 Meta 的原始断言。初次整包 overlay 复测的 worker context 错误不作为用例终态。 |
| 2 | `test_alignment.py`<br>`test_incorrect_meta_for_custom_op_2d` | 1 FAILED | 1 PASSED | 测试 custom-op 注册 / 用例适配，已验证 | define_custom_op_for_test 未给同一 fn 注册 PrivateUse1；基线输出 stride 1024 与 Meta 1040 不一致，错误 Meta 用例先被该断言截住。 | 32609ed 补注册 PrivateUse1，63565c7 适配树无 overlay 对照通过；保留 stride/offset 和错误 Meta 的原始断言。初次整包 overlay 复测的 worker context 错误不作为用例终态。 |
| 3 | `test_alignment.py`<br>`test_Q4_K_dequantization` | 1 FAILED | — | op-plugin eager __rshift__ / 其他组件实现 bug | eager 的 uint8 广播右移应分配 [221184,4,2,32]，实际输出 [221184,4,1,32]。本地 __Rshift__KernelNpuOpApi.cpp 对 tensor/tensor 分配 apply_tensor(self)，忽略广播 shape。 | 原用例已复现且尚未进入 Inductor；需修复 op-plugin 输出分配并重编译，不能通过改 dtype 或 broadcast 输入记为原项通过。 |
| 4 | `test_smoke.py`<br>`test_compile_decorator` | 1 FAILED | 1 PASSED | Inductor cudagraph_utils 与 transfer_to_npu / 实现 bug | CPU 构建 torch 不提供 CUDA allocator C 符号；transfer 将 torch.cuda.is_available 别名到 NPU，而新增检查未判断原生 CUDA/HIP 构建。NPU graph 的设备检查允许 npu，故先检查设备仍会调用缺失符号。 | 顺序调整失败；第二版增加 native CUDA/HIP build guard，candidate_v2 原用例通过。保留 reduce-overhead，不关闭 NPU graph。原生 CUDA/HIP allocator 关闭拒绝逻辑的控制流对照通过。 |
| 5 | `test_benchmarking.py`<br>`test_benchmark_smoke` | 3 PASSED, 1 XFAIL | 4 PASSED | benchmark 用例 / 预期负向检查 | Benchmarker.benchmark_gpu 基类明确抛 NotImplementedError；TritonBenchmarker 正向测时和 counter 检查通过。 | 32609ed 将基类宽泛 xfail 改为 assertRaises(NotImplementedError)，保留子类原 timing/counter 断言；适配复测全部实例通过。 |
| 6 | `test_benchmarking.py`<br>`test_benchmark_gpu_smoke` | 1 PASSED, 1 XFAIL | 2 PASSED | benchmark 用例 / 预期负向检查 | Benchmarker.benchmark_gpu 基类明确抛 NotImplementedError；TritonBenchmarker 正向测时和 counter 检查通过。 | 32609ed 将基类宽泛 xfail 改为 assertRaises(NotImplementedError)，保留子类原 timing/counter 断言；适配复测全部实例通过。 |
| 7 | `test_benchmarking.py`<br>`test_benchmark_safely_infers_device_no_devices` | 2 XFAIL | 2 PASSED | benchmark 用例 / 预期负向检查 | 没有 tensor 输入时 infer_device 按契约抛 ValueError，属于预期拒绝。 | 32609ed 去掉宽泛 xfail，明确断言 ValueError 及设备推断错误文本；适配复测两个实例通过。 |
| 8 | `test_benchmarking.py`<br>`test_benchmark_safely_infers_device_many_devices` | 2 XFAIL | 2 PASSED | benchmark 用例 / 上游测试实现 bug | 原用例调用不存在的 make_sum，AttributeError 被 expectedFailure 掩盖，未执行多设备输入检查。 | 32609ed 改为已有 make_params，并明确断言真正的多设备 ValueError；适配复测两个实例通过。 |
| 9 | `test_efficient_conv_bn_eval.py`<br>`test_conv_bn_eval` | 1 FAILED | 1 FAILED | NPU精度选项测试适配 / eager ConvTranspose2d运行时 / 用例适配后暴露其他组件实现bug | 原CUDA精度装饰器没有控制NPU HF32。a365d01改为实际NPU开关后越过原Conv1d精度失败；完整方法在MultiUserConvOp/ConvTranspose2d/use_bias=True处发生硬件scalar buffer越界。独立单组合在调用torch.compile之前的eager forward同步即复现507015。 | 完整适配方法已重新执行，1639.1秒以硬件错误失败；保留全部组合与forward/backward断言。单组合INPUT_READY之后、EAGER_FORWARD_PASS之前失败，当前阻塞属于eager/CANN路径，不能记为Inductor修复通过。 |
| 10 | `test_inplace_padding.py`<br>`test_pad_non_zero` | 1 FAILED | — | NPU Inductor padding lowering/layout / 未接入上游优化能力 | NPU lowering_fallback_list 明确将 constant_pad_nd 设为 fallback，__init__ 禁用 comprehensive_padding；无法进入社区 inplace_constant_pad_nd 优化。基线分别在 padded buffer 文本、compile-time autotune buffer 文本、inplace_padding=0 断言失败。 | 功能执行已完成后在原布局/优化计数断言失败。保留断言和失败记录；需要实现 NPU inplace padding 优化，不能通过删除 FileCheck 或接受计数0冒充支持。 |
| 11 | `test_inplace_padding.py`<br>`test_pad_non_zero_cpp_wrapper` | 1 FAILED | — | NPU Inductor padding lowering/layout / 未接入上游优化能力 | NPU lowering_fallback_list 明确将 constant_pad_nd 设为 fallback，__init__ 禁用 comprehensive_padding；无法进入社区 inplace_constant_pad_nd 优化。基线分别在 padded buffer 文本、compile-time autotune buffer 文本、inplace_padding=0 断言失败。 | 功能执行已完成后在原布局/优化计数断言失败。保留断言和失败记录；需要实现 NPU inplace padding 优化，不能通过删除 FileCheck 或接受计数0冒充支持。 |
| 12 | `test_inplace_padding.py`<br>`test_mutating_padding_input` | 1 FAILED | — | NPU Inductor padding lowering/layout / 未接入上游优化能力 | NPU lowering_fallback_list 明确将 constant_pad_nd 设为 fallback，__init__ 禁用 comprehensive_padding；无法进入社区 inplace_constant_pad_nd 优化。基线分别在 padded buffer 文本、compile-time autotune buffer 文本、inplace_padding=0 断言失败。 | 功能执行已完成后在原布局/优化计数断言失败。保留断言和失败记录；需要实现 NPU inplace padding 优化，不能通过删除 FileCheck 或接受计数0冒充支持。 |
| 13 | `test_inplace_padding.py`<br>`test_linear_and_cel` | 1 FAILED | — | NPU Inductor padding lowering/layout / 未接入上游优化能力 | NPU lowering_fallback_list 明确将 constant_pad_nd 设为 fallback，__init__ 禁用 comprehensive_padding；无法进入社区 inplace_constant_pad_nd 优化。基线分别在 padded buffer 文本、compile-time autotune buffer 文本、inplace_padding=0 断言失败。 | 数值和梯度比较先通过，再因 inplace_padding=0 而失败；原始优化计数断言保留。 |
| 14 | `test_inplace_padding.py`<br>`test_linear_and_cel_max_autotune` | TIMEOUT | 1 FAILED | NPU Inductor padding lowering/layout / 未接入上游优化能力 | NPU lowering_fallback_list 明确将 constant_pad_nd 设为 fallback，__init__ 禁用 comprehensive_padding；无法进入社区 inplace_constant_pad_nd 优化。基线分别在 padded buffer 文本、compile-time autotune buffer 文本、inplace_padding=0 断言失败。 | 16/4并发下完整方法119.9秒结束；数值及梯度比较先通过，随后原 inplace_padding 计数断言0!=1失败，与同族未实现优化一致。 |
| 15 | `test_triton_extension_backend.py`<br>`test_codegen_with_custom_heuristics_module` | 1 FAILED | 1 FAILED | extension 测试隔离 / NPU autotuner 扩展接口 / 用例适配后暴露未支持接口 | 原 BaseExtensionBackendTests 将测试用 PrivateUse1 allocator 加载到真实 NPU，首先导致 D2D copy 失败。改用普通 TestCase 后，cached_autotune 拒绝 caching_autotuner_cls 关键字。 | 26077d3 隔离测试 allocator 后完整方法已复测，首错消失但自定义 autotuner class API 仍不支持；保持原代码断言，不记为通过。 |
| 16 | `test_triton_extension_backend.py`<br>`test_codegen_with_custom_heuristics_module_udtk` | 1 FAILED | 1 PASSED | extension 测试 PrivateUse1 allocator 隔离 / 用例适配，已验证 | 真实 NPU 已占用 PrivateUse1；测试基类加载另一个 fake allocator 导致注意力 pattern 初始化中的 NPU tensor copy 失败。 | 26077d3 在真实 NPU 上使用 TestCase，保留 custom heuristics import 与 UDTK 代码断言；完整方法通过。不能归纳为所有 custom heuristics 均不支持。 |
| 17 | `test_compile_subprocess.py`<br>`test_async` | 2 FAILED | 2 FAILED | Inductor subprocess lowering 序列化 / 实现 bug | NPU 注册表包含 HigherOrderOperator fallback；子进程还原 fallback 时调用 force_fallback，其类型断言只允许 OpOverload。 | HigherOrderOperator 类型断言候选已越过首错；随后子进程恢复 config 时缺 npu_backend，表明 worker 未完成 NPU backend/config 注册，整项仍失败，不能记为修复通过。 |
| 18 | `test_max_autotune.py`<br>`test_deferred_layout_constraint_cat_fusion` | TIMEOUT | 2 FAILED, 2 PASSED | 延迟布局测试的 CUDA padding 假设 / NPU模板布局 / 用例适配后暴露epilogue精度bug | NPU 禁用 comprehensive_padding，实际 B 缓冲 (32,512,1490)、stride (762880,1490,1) 未补到1536。epilogue=False 原断言通过；True 选合法未补齐 Triton BMM，却被 CUDA 专属 check_not(triton_tem) 拒绝。 | 33da2f4的真实stride检查通过；结合metadata候选仍有两项epilogue=True失败，BF16主输出转FP32未保留BF16舍入，24377643/24412160元素不符。两项False通过。候选尚未修复此精度错误。 |
| 19 | `test_max_autotune.py`<br>`test_mm_layout_freezing_behavior` | 4 FAILED | 4 PASSED | Inductor native CUDA allocator 检查 / 实现 bug，候选修复通过 | CPU 构建 torch 不提供 CUDA allocator C 符号；transfer 将 torch.cuda.is_available 别名到 NPU，而新增检查未判断原生 CUDA/HIP 构建。NPU graph 的设备检查允许 npu，故先检查设备仍会调用缺失符号。 | 原方法全部所选参数在 candidate_v2 通过；仅 build guard，未关闭 cudagraphs、冻结或 max_autotune。 |
| 20 | `test_max_autotune.py`<br>`test_deffered_layout_constraint_mm_cat_mm` | 2 FAILED | 2 PASSED | Triton Ascend batched-matmul lowering / NPU模板选择 / 实现 bug，关闭宏的候选通过 | 单个 triton_npu_triton_bmm_batched 配置编译成功但同步报507015，LOAD2D写L0B越界；独立导出保留原配置/输入布局复现。相同 DSL 仅关闭 enable_hivm_batch_matmul 后启动成功，去除 multibuffer 仍失败。 | 72932e5 所含 BMM 候选仅关闭 enable_hivm_batch_matmul，完整原方法通过（193.4秒）。此为编译选项规避，尚未修复编译器宏实现，需评估性能影响；原 shape/dtype/layout 和数值断言保留。 |
| 21 | `test_max_autotune.py`<br>`test_triton_gemm_epilogue_fusion_truncates_accumulator` | TIMEOUT | 8 FAILED, 4 PASSED | NPU Inductor template fusion metadata / 生成名称适配 / 实现 bug，候选未完全修复 | NPU codegen_template 在 render/store_output 前缺少上游 _compute_fusion_metadata，导致累加器截断指令不生成。纯 mm 路径的融合实际名称为 matmul，另有 CUDA 名称断言不适配。 | 53baa39名称适配和metadata候选完成全部12实例复测：4个fp32通过，8个fp16/bf16在融合/未融合精确比较失败。原代码检查已通过，说明首个缺失cast已修复；低精度截断仍需继续定位，候选未完全修复。 |
| 22 | `test_max_autotune.py`<br>`test_max_autotune_prune_choices` | 1 FAILED | 1 PASSED | Inductor native CUDA allocator 检查 / 实现 bug，候选修复通过 | CPU 构建 torch 不提供 CUDA allocator C 符号；transfer 将 torch.cuda.is_available 别名到 NPU，而新增检查未判断原生 CUDA/HIP 构建。NPU graph 的设备检查允许 npu，故先检查设备仍会调用缺失符号。 | 原方法全部所选参数在 candidate_v2 通过；仅 build guard，未关闭 cudagraphs、冻结或 max_autotune。 |
| 23 | `test_torchinductor.py`<br>`test_linspace4` | 1 SKIPPED | 1 PASSED | 多设备测试环境 / 单卡可见性限制；双卡原方法通过 | 初始 ASCEND_RT_VISIBLE_DEVICES 仅暴露一个设备，被 requires_multigpu 合法跳过。双卡暴露后执行真实多设备路径。 | multi_device_v2 在物理卡1、3完整通过；未删除 requires_multigpu 或修改目标设备索引。 |
| 24 | `test_torchinductor.py`<br>`test_views1` | TIMEOUT | 1 PASSED | 测试执行时限/编译耗时 / 编译耗时，完整原方法通过 | 28次 common 子调用及多候选编译超出初始420秒。低并发2/2时2400秒仍未完成，堆栈持续等待编译future，子调用在推进。 | 保持全部输入、shape/stride/dtype和断言，16个NPU预编译worker、4个Inductor编译线程下683.1秒完整通过；不是一开始死锁。 |
| 25 | `test_torchinductor.py`<br>`test_views2` | TIMEOUT | 1 PASSED | 测试执行时限/编译耗时 / 编译耗时，完整原方法通过 | 6次 common 子调用及多候选编译超出初始420秒，延长时限后有完整终态。 | 原并发2/2、原输入和断言，1991.4秒完整通过；无需测试语义或实现补丁。 |
| 26 | `test_torchinductor.py`<br>`test_builtins_round` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 当前 torch_npu 已有 Tensor 类型检查，未再对 SymFloat 无条件读取 dtype；原始方法完整通过。 | 本轮无需补丁；不将环境已有改动记为本轮修复。 |
| 27 | `test_torchinductor.py`<br>`test_builtins_round_float_ndigits_pos` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 当前 torch_npu 已有 Tensor 类型检查，未再对 SymFloat 无条件读取 dtype；原始方法完整通过。 | 本轮无需补丁；不将环境已有改动记为本轮修复。 |
| 28 | `test_torchinductor.py`<br>`test_builtins_round_float_ndigits_zero` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 当前 torch_npu 已有 Tensor 类型检查，未再对 SymFloat 无条件读取 dtype；原始方法完整通过。 | 本轮无需补丁；不将环境已有改动记为本轮修复。 |
| 29 | `test_torchinductor.py`<br>`test_builtins_round_float_ndigits_neg` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 当前 torch_npu 已有 Tensor 类型检查，未再对 SymFloat 无条件读取 dtype；原始方法完整通过。 | 本轮无需补丁；不将环境已有改动记为本轮修复。 |
| 30 | `test_torchinductor.py`<br>`test_builtins_round_float_ndigits_neg_uses_value_expr` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 当前 torch_npu 已有 Tensor 类型检查，未再对 SymFloat 无条件读取 dtype；原始方法完整通过。 | 本轮无需补丁；不将环境已有改动记为本轮修复。 |
| 31 | `test_torchinductor.py`<br>`test_floordiv_div_by_zero_int` | 1 FAILED | — | 整数除零跨后端规格 / 上游测试假设 / 规格边界 | 原测试源码承认整数除零结果 undefined，仍固定断言 CUDA 路径为0。NPU eager int32 返回 ±2147450751、int64 返回 INT64_MAX，Triton 返回0。 | 已用5组被除数和两种原始整数 dtype 对照 eager/compiled。未找到 NPU 将零除结果保证为0的规格；该 undefined 输入不能作为精度正确性判据。保留原始失败记录，不改变数据或删除零项记为通过。 |
| 32 | `test_torchinductor.py`<br>`test_gather3` | 1 FAILED | 1 PASSED | op-plugin npu_gather_backward Meta / 其他组件实现 bug，候选修复通过 | common_utils.TestCase.setUp 开启 sparse invariants；Meta 内 torch.sparse_coo_tensor 继承全局数据检查，试图将 meta size tensor 复制到 CPU。_to_copy/device_put 分解相互重入后表现为 RecursionError；未进入 kernel lowering。 | CPU Meta 四组合对照通过；局部给 Meta 内三个 sparse_coo_tensor 构造传 check_invariants=False，原 NPU 方法完整通过（158.5秒），保留真实输入的全局 sparse invariants 检查。 |
| 33 | `test_torchinductor.py`<br>`test_to_dtype` | 1 FAILED | — | NPU Triton/Inductor dtype 能力 / float64 kernel 未支持 | 原方法有 float64 输入及 b+1，DSL/MLIR 需要 f64 load；HIVM load verifier 的合法 element type 不含 f64。eager 参考路径先成功不代表 Triton arithmetic 支持。 | 原失败记录含完整生成 DSL 与 f64 MLIR verifier 文本；保留 float64 输入失败，未替换 dtype 或添加通用 fallback 冒充原项通过。 |
| 34 | `test_torchinductor.py`<br>`test_multi_gpu_device` | 1 SKIPPED | 1 FAILED | NPU Event运行时 / Inductor autotune计时 / 双卡运行时错误，定位中 | 双卡原方法在triton_poi_fused__to_copy_div_0候选预运行计时失败，aclrtEventElapsedTime报507000/event recorder null。生成wrapper含独立device0/1 guard和跨卡copy。事件探针记录start/end都为npu:0，尚无证据支持跨设备event假设。 | 已在物理卡1+3及2+3各复现一次；正在核对纯Event控制与真实CANN上下文，不能把跳过改为通过。 |
| 35 | `test_torchinductor.py`<br>`test_multi_gpu_recompile_on_index` | 1 SKIPPED | 1 PASSED | 多设备测试环境 / 单卡可见性限制；双卡原方法通过 | 初始 ASCEND_RT_VISIBLE_DEVICES 仅暴露一个设备，被 requires_multigpu 合法跳过。双卡暴露后执行真实多设备路径。 | multi_device_v2 在物理卡1、3完整通过；未删除 requires_multigpu 或修改目标设备索引。 |
| 36 | `test_unbacked_symints.py`<br>`test_split_with_sizes` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 原方法和原 dtype/动态参数完整通过；测试初始化前补齐 patch_has_triton，避免测试收集错误地缓存 CUDA-only 可用性。 | 不修改测试语义，保留当前通过记录。 |
| 37 | `test_unbacked_symints.py`<br>`test_triton_kernel_grid` | 1 PASSED | — | 当前环境未复现历史失败 / 基线通过 | 原方法和原 dtype/动态参数完整通过；测试初始化前补齐 patch_has_triton，避免测试收集错误地缓存 CUDA-only 可用性。 | 不修改测试语义，保留当前通过记录。 |
| 38 | `test_unbacked_symints.py`<br>`test_triton_kernel_with_unbacked_symint_fallback` | 1 FAILED | — | transfer_to_npu / Dynamo 符号 tensor 构造 / 其他组件实现 bug | 在构造 torch.tensor([1, unbacked_symint], device=npu) 时 Dynamo 请求符号取实值而报 GuardOnDataDependentSymNode，尚未调用 Inductor。transfer 将 torch.tensor 替换为 Python 包装，影响基于函数身份的特殊处理。 | 保留原输入与动态符号；同类 CPU backend=eager 对照中原生 tensor 构造通过、启用 transfer 后失败，支持边界在迁移包装与 Dynamo 集成；不是 Triton codegen 错误。 |
| 39 | `test_torchinductor_dynamic_shapes.py`<br>`test_constant_fold_uniform_value_dynamic` | 1 FAILED | 1 PASSED | 动态常量折叠测试代码 / 用例适配，已验证 | 完全折叠的图可返回输入而不生成 kernel，source_codes 为空；原检查无条件访问 source_codes[0]。 | 32609ed 将代码检查拼接全部 source_codes；保留实际输出比较、多输入尺寸和动态选项。control_no_overlay 完整方法通过。 |
| 40 | `test_torchinductor_dynamic_shapes.py`<br>`test_sym_sum_unbacked` | 1 FAILED | — | transfer_to_npu / Dynamo 符号 tensor 构造 / 其他组件实现 bug | torch.tensor(sum(a.tolist())) 在 Dynamo 阶段尝试取 unbacked SymInt 的值；未到 Inductor。torch.tensor 的 Dynamo 特殊 handler 按函数对象注册，而 transfer 全局替换该函数。 | CPU + backend=eager/fullgraph 对照：未启用 transfer 通过，启用 transfer 后复现同类数据依赖符号 guard。无需 NPU kernel 即可触发；保留原失败记录。 |
| 41 | `test_torchinductor_dynamic_shapes.py`<br>`test_mark_unbacked_slice` | 1 FAILED | 1 PASSED | Inductor Triton driver 选择 / transfer 别名 / 实现 bug，候选修复通过 | triton_helpers._is_backend_active 将 NPU 的 CUDA availability 别名误当原生 CUDA，选择 NVIDIA driver 并查找 libcuda.so。41 在 allocator guard 修复后暴露，42 原始基线直接暴露。 | candidate_v3 同时加载 native allocator 与 NVIDIA driver build guard，完整 reduce-overhead 方法通过（204秒）；未关闭 NPU graph。 |
| 42 | `test_torchinductor_dynamic_shapes.py`<br>`test_sort_dynamic_shape_with_check` | 1 FAILED | 1 PASSED | NPU Inductor codegen fallback计数 / Triton driver / 实现 bug，候选修复通过 | native driver guard 消除 libcuda 首错。Kernel 构造栈证实 Index codegen 先创建 NPUIndexTritonKernel，TTIR检查失败再生成 NPUTritonKernel；失败尝试未回滚 generated_kernel_count，实际只运行一个稳定排序内核。 | 72932e5 在 codegen_node 失败回退前恢复 generated_kernel_count；完整方法通过（95.3秒），保留原数值、稳定排序、两个方向及动态重用计数断言。没有修改预期内核数。 |
| 43 | `test_torchinductor_dynamic_shapes.py`<br>`test_coalescing_analysis_sympy_is_constant` | 1 FAILED | — | NPU Inductor attention 接入 / 未支持 lowering/decomposition | scaled_dot_product_attention 在 tracing 中替换为 npu_fusion_attention.default，当前 lowering 表与 decomp 表缺少该节点；在 lowering 阶段失败。 | 原始 fp16 输入和 dynamic=True/fullgraph=True 保留；未用关闭 attention 或通用 fallback 冒充能力实现。 |

## 证据与复现入口

- [机器可读结果与参数级记录](../../../artifacts/npu-inductor-boundary-20261028/results.csv)
- [方法和关键实现的源码证据及 SHA](../../../artifacts/npu-inductor-boundary-20261028/source-evidence.json)
- [当前进度统计](../../../artifacts/npu-inductor-boundary-20261028/progress.json)
- [CUDA allocator 检查候选](../../../artifacts/npu-inductor-boundary-20261028/cudagraph-device-check.patch)
- [Triton driver 原生构建检查候选](../../../artifacts/npu-inductor-boundary-20261028/triton-native-backend-check.patch)
- [子进程 HigherOrderOperator fallback 候选](../../../artifacts/npu-inductor-boundary-20261028/subprocess-hop-fallback.patch)
- [GEMM epilogue metadata 候选，原代码检查通过但精度未完全修复](../../../artifacts/npu-inductor-boundary-20261028/template-fusion-metadata.patch)

测试适配分支为 `Community-Use-Case-Testing:codex/inductor-boundary-20261028`；
调试脚本和候选存于 `torch-learn:codex/npu-inductor-boundary-20261028`。
本地先提交、推送，再传递对应已推送提交的 Git bundle/归档供目标容器读取，
以绕过目标容器直接访问代码托管站点失败的问题。环境中的实际 Python 包未替换。

本地控制流探针不分配设备 tensor，只验证构建判定和 CUDA/HIP 原有 allocator 拒绝语义，
不替代 NPU 原用例验证；源码推断和动态实测分别标注。


## 已确认的实现边界

`views1` 和 `views2` 的初始420秒超时不能当作死锁。原方法分别有28次和6次 `common` 子调用。
`views1` 在2/2编译并发下2400秒仍未完成，但持续有子调用返回；16/4并发下683.1秒全部通过。
`views2` 保留2/2并发，1991.4秒完整通过。逐调用记录与堆栈支持耗时来自编译等待。

Conv/BN 的 NPU HF32 开关适配后，完整方法在 `MultiUserConvOp`、`ConvTranspose2d`、
`use_bias=True`、非SyncBN且不预分解的组合处发生507015。单组合探针在调用 `torch.compile` 之前，
于 eager forward 的显式同步即复现 scalar buffer越界，因此这个后续硬件阻塞属于 eager/CANN 路径。
这项仍保留失败终态；前面组合成功和原精度首错消失均不能代表完整方法通过。

### 固定 BMM 配置的独立证据

[完整生成DSL](../../../artifacts/npu-inductor-boundary-20261028/bmm-fixed-config/generated-bmm.py)
对应 `triton_npu_triton_bmm_batched`，num_warps=4、num_stages=4，
BLOCK_Q/M/N/K=2/128/256/144，grid=(28,1,1)，输入和输出均为FP16。
A为(3072,128,144)，stride=(18432,144,1)；B为(3072,144,160)，
捕获布局stride=(38400,160,1)；输出为(3072,128,160)，stride=(20480,160,1)。
metadata模式保留布局，使用seed=0合成数值，没有保存用户tensor数据。

[独立直接启动脚本](../../../artifacts/npu-inductor-boundary-20261028/bmm-fixed-config/bmm_batched_launch.py)
不运行Inductor autotuner，仍在同步时报507015，硬件明示LOAD2D写L0B越界。
删除multibuffer标注仍失败；同一DSL仅关闭 `enable_hivm_batch_matmul` 后启动成功。
[宏关闭候选](../../../artifacts/npu-inductor-boundary-20261028/bmm-disable-batch-macro.patch)
带回原完整方法后，两个测试类实例均通过，共193.4秒。
这证明故障收敛到batched-matmul宏的后端编译/执行路径；关闭宏是规避，尚未修复编译器宏本身，
也没有验证性能。生成DSL与输入metadata分别保留，不能由启动成功推定任意布局的数学正确性。

### 稳定排序的计数恢复

内核构造栈记录显示，NPU先尝试 `NPUIndexTritonKernel`，TTIR检查失败后回退为
`NPUTritonKernel`。失败尝试先增加了 `generated_kernel_count`，回退没有恢复，
导致一次实际稳定排序被计为两个内核。这里不是两个输出各运行一个内核。
[候选补丁](../../../artifacts/npu-inductor-boundary-20261028/fallback-kernel-metric-rollback.patch)
在失败回退前恢复计数；完整动态排序方法95.3秒通过，原稳定排序、两个方向、
后续动态复用及1/2内核计数断言全部保留。

### 精度与布局候选的剩余失败

上游 `TritonTemplateKernel.store_output` 根据fusion metadata决定是否在epilogue之前
将FP32累加器截断到原输出dtype。NPU的独立 `codegen_template` 没有准备这份metadata，
因此最初缺少截断语句。补齐metadata后，第21项全部代码检查通过，12个实例中4个FP32通过，
8个FP16/BF16仍未通过融合与未融合的精确比较。当前候选没有完全修复精度问题。

已保存同一失败实例的[融合代码](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-fused.py)
与[未融合代码](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-unfused.py)。
两者的GEMM均使用128×128×128 tile；融合代码在bias相加后先转BF16、再转FP32，随后执行relu/sub；
未融合代码先将GEMM结果写入BF16缓冲，再由pointwise内核读取并执行relu/sub。
因此，首个metadata缺陷修复后仍发生的差异，不能再仅用“生成代码缺少cast”解释。

第18项的NPU生成代码保留宽度1490及行步长1490；CUDA测试设想的1536填充布局没有出现。
测试适配检查真实步长并增加输出比较后，两个epilogue=False分支通过，
两个True分支在BF16输出转FP32的结果上失败：主BF16结果比较通过，但FP32结果未保留BF16舍入。
该新增精度证据保留为失败，不能仅凭布局检查通过就认定修复完成。

该次运行保留的[BMM候选DSL](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case18-bmm-candidate.py)
同样包含 `acc.to(tl.bfloat16)` 再转FP32，并分别写入BF16和FP32输出；
该文件尚无selected winner记录，不能直接认定它就是产生最终错误的配置。
[源文件校验清单](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/manifest.json)
区分原文件和去敏副本SHA。

[固定模板精度对照脚本](../../../scripts/repro/probe_template_roundtrip_precision.py)
保留DSL函数原文，直接调用Triton编译和launch，输出编译IR及误差JSON。
第18项比较同次BMM的FP32结果与其BF16结果在CPU上扩展后的值；
第21项比较融合GEMM与同输入、同tile的未融合GEMM加CPU epilogue。
该脚本是待验证的定界实验，不能代替原方法的完整复测，也尚未证明下层编译器存在缺陷。

2026-10-29北京时间00:05后，目标SSH在密钥交换前持续关闭，导致剩余设备实验无法启动。
已完成的本地语法/提取结构检查不计为NPU验证。恢复连接后需执行上述固定模板对照，
补取第9项eager探针和第34项事件探针的日志，并验证实际CANN上下文及binary loader上下文恢复候选。

稀疏gather的递归来自Meta构造继承了测试全局开启的sparse invariants数据检查，
并在meta tensor与CPU复制分解间重入。仅给Meta内三个 `sparse_coo_tensor` 构造传入
`check_invariants=False` 后，完整原NPU方法158.5秒通过；真实输入的全局检查继续启用。
