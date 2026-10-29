---
title: NPU Inductor 2.13：43 项上游用例复现与定界
published: false
---

# NPU Inductor 2.13：43 项上游用例复现与定界

本记录按任务输入的43项顺序保留基线、适配/候选复测、组件原因与修复尝试。
当前26个方法完整通过（含明确的预期负向检查），其余17项保留实际失败及实现、能力或规格结论。
“定界完成”表示已确定问题边界并记录修复尝试，不表示43项全部通过。
方法的全部参数实例完成后才更新终态，超时、跳过和局部成功不算完整方法通过。

## 环境与执行范围

2026-10-28至29日，Docker 内独立 conda 环境，Ascend950PR_9579，torch `2.13.0+cpu`、
torch_npu `2.13.0.dev20261028`、Triton `3.6.0`，CANN 9.1.0、driver 25.7.rc1。
测试原始提交 `d70307f9ceeedee7156135bb2c198d72095f8353`。

初始基线只暴露一个设备，编译线程和 NPU precompile worker 均设为 2；后续长用例使用16个预编译worker和4个编译线程对照，不改变候选、输入或断言。每个方法独立解释器、
独立 Inductor cache。适配树与源码候选均隔离运行，共享安装包未修改。
当前基线由 [run_boundary_cases.py](../../../scripts/tests/npu_inductor_213/run_boundary_cases.py)
和 [run_batch.py](../../../scripts/tests/npu_inductor_213/run_batch.py) 记录 setup/call/teardown。

测试收集前调用 `_dynamo.patch_has_triton()`，使 NPU 的可用性判断先于 `inductor_utils` 缓存，
避免 harness 提前导入造成假跳过。该初始化对基线和适配复测一致。
`test_conv_bn_eval` 是嵌套 helper，实际执行完整外层 `test_basic_npu`；不是只选择一个不存在的 pytest 节点。
多设备方法中第23、35项通过；第34项的异步Event计时错误经同步对照收敛到eager跨卡复制失败。

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
| 18 | `test_max_autotune.py`<br>`test_deferred_layout_constraint_cat_fusion` | TIMEOUT | 2 FAILED, 2 PASSED | 延迟布局测试适配 / CANN BiShengIR 精度语义 / 用例适配已验证；后续设备编译器实现 bug | NPU 禁用 comprehensive_padding，B 实际行步长1490而非CUDA假设1536。适配后实际所选BMM已生成FP32→BF16→FP32，但BiShengIR将该舍入边界消去，FP32输出直接使用累加结果。TTIR/TTadapter保留truncf/extf，下层NPU IR已丢失。 | 33da2f4保留真实stride与所有输出比较；完整4实例2通过、2精度失败。补取失败实例实际所选L1_COPIES=2模板，绕过Inductor直接启动仍有24411084/24412160元素不符、max_abs=0.25。初始L1_COPIES=0对照关闭enable_fp_fusion仍失败；metadata候选未能修复下层错误，整项保留失败。 |
| 19 | `test_max_autotune.py`<br>`test_mm_layout_freezing_behavior` | 4 FAILED | 4 PASSED | Inductor native CUDA allocator 检查 / 实现 bug，候选修复通过 | CPU 构建 torch 不提供 CUDA allocator C 符号；transfer 将 torch.cuda.is_available 别名到 NPU，而新增检查未判断原生 CUDA/HIP 构建。NPU graph 的设备检查允许 npu，故先检查设备仍会调用缺失符号。 | 原方法全部所选参数在 candidate_v2 通过；仅 build guard，未关闭 cudagraphs、冻结或 max_autotune。 |
| 20 | `test_max_autotune.py`<br>`test_deffered_layout_constraint_mm_cat_mm` | 2 FAILED | 2 PASSED | Triton Ascend batched-matmul lowering / NPU模板选择 / 实现 bug，关闭宏的候选通过 | 单个 triton_npu_triton_bmm_batched 配置编译成功但同步报507015，LOAD2D写L0B越界；独立导出保留原配置/输入布局复现。相同 DSL 仅关闭 enable_hivm_batch_matmul 后启动成功，去除 multibuffer 仍失败。 | 72932e5 所含 BMM 候选仅关闭 enable_hivm_batch_matmul，完整原方法通过（193.4秒）。此为编译选项规避，尚未修复编译器宏实现，需评估性能影响；原 shape/dtype/layout 和数值断言保留。 |
| 21 | `test_max_autotune.py`<br>`test_triton_gemm_epilogue_fusion_truncates_accumulator` | TIMEOUT | 8 FAILED, 4 PASSED | NPU Inductor fusion metadata / CANN BiShengIR 精度语义 / Inductor实现 bug 已尝试修复；后续设备编译器实现 bug | NPU codegen_template未调用_compute_fusion_metadata，最初缺少累加器截断。补齐后全部代码检查通过；FP16/BF16融合模板的truncf/extf仍被下层编译器消去，relu/sub使用未舍入FP32累加结果。未融合GEMM先写低精度，保留舍入边界。 | 53baa39适配真实matmul名称；metadata候选完成12实例复测，4个FP32通过、8个FP16/BF16数值失败。固定同tile、同输入BF16融合/未融合模板绕过Inductor仍有1537/16384元素不符、max_abs=0.125。实际所选FP16模板直接重放1499/16384元素不符、max_abs=0.015625；两种dtype的TTadapter与下层IR证明舍入语义丢失。关闭enable_fp_fusion无效，不能把首个Inductor缺陷修复等同整项通过。 |
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
| 34 | `test_torchinductor.py`<br>`test_multi_gpu_device` | 1 SKIPPED | 1 FAILED | torch_npu eager跨卡复制 / CANN runtime与driver / 其他组件跨卡复制实现/能力边界 | 异步运行首先暴露EventElapsedTime 507000，但Torch与CANN当前设备均为0，start/end同在npu:0。关闭任务队列的完整原方法在torch.compile之前的eager r.to(npu:1)即报507899，底层MemcpyAsync返回0x7020010并提示不支持该操作。 | 原4元素FP32 eager表达式独立复现16字节kind=3跨卡复制失败，默认队列及TASK_QUEUE_ENABLE=0均失败；不依赖Inductor。实际CANN设备探针排除已提出的跨设备event解释，binary-loader上下文恢复候选无效。保留双卡原方法失败；关闭队列仅用于暴露首错，没有通过关闭计时或CPU中转冒充修复。 |
| 35 | `test_torchinductor.py`<br>`test_multi_gpu_recompile_on_index` | 1 SKIPPED | 1 PASSED | 多设备测试环境 / 单卡可见性限制；双卡原方法通过 | 初始 ASCEND_RT_VISIBLE_DEVICES 仅暴露一个设备，被 requires_multigpu 合法跳过。双卡暴露后执行真实多设备路径。 | multi_device_v2 在物理卡1、3完整通过；未删除 requires_multigpu 或修改目标设备索引。 |
| 36 | `test_unbacked_symints.py`<br>`test_split_with_sizes` | 1 PASSED | — | NPU Inductor persistent reduction mask / 历史实现 bug，PR !48448 已修复并验证 | split sizes来自tolist，首段有效归约长度为7。旧codegen虽生成r0_mask，却从persistent reduction的load/归约中删掉该条件，尾部无效元素参与sum；不是普通浮点舍入误差。 | dd35816d26（!48448，2026-10-23）保留未证明恒真的归约mask，并在求和前用单位元屏蔽无效lane。同一当前环境仅替换该PR前后codegen/triton.py：父版本精确复现历史-5.140399→-6.655652错误，PR版本及当前安装版本均完整通过。 |
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
于 eager forward 的显式同步即复现 scalar buffer越界，见[独立eager日志](../../../artifacts/npu-inductor-boundary-20261028/conv-eager/probe.log)，因此这个后续硬件阻塞属于 eager/CANN 路径。
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

### GEMM 舍入边界：Inductor 首错与设备编译器后续错误

上游 `TritonTemplateKernel.store_output` 根据fusion metadata决定是否在epilogue之前
将FP32累加器截断到原输出dtype。NPU独立的 `codegen_template` 没有准备这份metadata，
因此最初缺少截断语句。[metadata候选](../../../artifacts/npu-inductor-boundary-20261028/template-fusion-metadata.patch)
补齐这一步后，第21项12个实例的代码检查全部通过；4个FP32实例数值通过，
8个FP16/BF16实例仍在融合与未融合的精确比较处失败。首个Inductor缺陷已越过，完整方法仍失败。

“舍入边界”指先把FP32累加值写成低精度，再扩展回FP32所造成的数值变化。
例如FP32的1.001转BF16后为1.0，再转FP32仍为1.0，不能直接用原来的1.001替代。
融合可以去掉中间缓冲，但必须保留这次舍入。

[BF16融合模板](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-fused.py)
与[未融合模板](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-unfused.py)
均使用128×128×128 GEMM tile。固定相同输入，直接调用Triton编译和launch，
绕过Inductor及autotuner，融合结果与“未融合GEMM的BF16输出加CPU relu/sub”仍有
1537/16384个元素不同，最大绝对误差0.125。
[默认配置结果](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-default/result.json)
和[关闭enable_fp_fusion结果](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-fp_off/result.json)
相同。debug模式重放也复现，排除仅由调试选项引起的差异。

[融合TTadapter](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-debug/fused.ttadapter)
在bias加法后仍保留 `arith.truncf` 到BF16、`arith.extf` 回FP32，之后才执行relu/sub。
[下层优化IR](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-debug/fused.hivm.opt.mlir)
却将FP32累加值直接送入relu/sub，仅在最终输出时转换成BF16；输入bias的BF16→FP32转换与
最终输出转换仍存在，但不能替代中间舍入。
[未融合IR](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-debug/unfused.hivm.opt.mlir)
通过float→bfloat16的fixpipe写出GEMM结果，保留了物化缓冲的舍入。
因此已证实的后续BF16缺陷位于TTadapter之后的BiShengIR编译流程，不能继续归因于Inductor少生成cast。
当前证据定位到语义被消去的编译阶段，未指定内部具体pass名称。
[实际安装的Triton源码](../../../artifacts/npu-inductor-boundary-20261028/compiler-source-evidence.json)
确认TTadapter文本作为输入交给BiShengIR，debug输出在其graph-sync-solver之后取得。

FP16分支另取原失败实例实际所选的[融合wrapper](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-fp16-fused.py)
及[未融合wrapper](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case21-fp16-unfused.py)，
两者均为128×128×128、num_stages=2。相同输入直接重放有1499/16384个元素不同，
最大绝对误差0.015625，见[独立结果](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-fp16-debug/result.json)。
[FP16 TTadapter](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-fp16-debug/fused.ttadapter)
保留f32→f16→f32；[下层IR](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/21-fp16-debug/fused.hivm.opt.mlir)
的relu/sub同样直接读取FP32累加值，只在最终输出时转FP16。因此FP16也有独立实测证据，
结论不依赖从BF16外推；这里仍未逐一重放12个实例的全部autotune配置。

第18项先是CUDA布局假设不适配：NPU禁用comprehensive_padding，B的实际shape为
(32,512,1490)、stride为(762880,1490,1)，没有CUDA测试预设的1536行步长。
33da2f4检查真实stride，并比较全部输出。完整4实例中两个epilogue=False通过，两个True精度失败。
补取一次失败True实例的完整wrapper，已确认实际所选为
`triton_tem_fused__to_copy_bmm_2`，BLOCK_M/N/K=128/256/256，L1_COPIES=2，
不是仅根据未获选候选猜测。
[完整所选wrapper](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/case18-selected.py)
与[原实例失败结果](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/18-selected-original/result.json)
保留对应证据。

直接重放该所选BMM，比较同次计算的FP32输出和其BF16输出在CPU上扩展的结果，
24411084/24412160个元素不同，最大绝对误差0.25；
[结果与编译选项](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/18-selected-debug/result.json)
记录seed、源文件SHA及实际编译配置。
[TTadapter](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/18-selected-debug/bmm.ttadapter)
保留 `truncf/extf`，但[下层IR](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/18-selected-debug/module.hivm.opt.mlir)
仅为BF16输出做FP32→BF16转换，FP32输出直接写原累加缓冲，缺少BF16→FP32的舍入后扩展。
这给出了与第21项一致的、独立于Inductor的语义丢失证据。
早期L1_COPIES=0候选也复现；该候选关闭enable_fp_fusion仍失败，
不能将该选项记为修复，也不把它与后续实际所选L1_COPIES=2模板混为一次实验。

[固定模板重放脚本](../../../scripts/repro/probe_template_roundtrip_precision.py)
原样提取已保存DSL，仅移除Inductor autotuner装饰器；使用seed=1234合成输入，
不保存用户tensor。第18项为BF16 A(32,512,512)、B(32,512,1490)，第21项为BF16 bias(128)、x(128,128)。
原完整方法与固定kernel对照属于不同验证范围；本次未修改设备编译器内部实现。

```bash
PYTHONPATH="$PWD/scripts/tests/npu_inductor_213/bootstrap${PYTHONPATH:+:$PYTHONPATH}" \
BOUNDARY_CANDIDATE_DIR="$PWD/artifacts/npu-inductor-boundary-20261028/candidate" \
BOUNDARY_NATIVE_DRIVER=1 python scripts/repro/probe_template_roundtrip_precision.py \
  --case 21 --compiler-debug --output <test-workspace>/precision-roundtrip-21
```

用 `--case 21 --dtype float16`运行已捕获的FP16对照，`--case 18`运行实际所选BMM；`--fp-fusion-off`是额外对照，不能替代原方法验证。
[源文件清单](../../../artifacts/npu-inductor-boundary-20261028/precision-roundtrip/manifest.json)
与[后续产物校验清单](../../../artifacts/npu-inductor-boundary-20261028/followup-artifact-manifest.json)
区分原始和去敏副本SHA；IR只替换环境路径等敏感内容，保留原始格式。

### 双卡复制：eager 首错被异步 Event 错误掩盖

第34项原表达式为4元素FP32的 `(x / y).to('npu:1') * 2`。
异步完整方法在候选预运行中报 `aclrtEventElapsedTime` 507000 / event recorder null。
[实际CANN设备探针](../../../artifacts/npu-inductor-boundary-20261028/multi-device/multi_event_raw/event-devices.jsonl)
确认Torch与CANN当前device均为0，start/end均为npu:0，设备同步后两个event仍未完成，
直接CANN elapsed查询已报507000。因此没有证据支持最初提出的“两个event在不同设备”解释。
生成wrapper虽有独立device guard，也不能由此推定下层跨卡复制可用。

[恢复binary-loader上下文的候选](../../../scripts/tests/npu_inductor_213/boundary_binary_context.py)
带回完整方法后仍失败；[失败结果](../../../artifacts/npu-inductor-boundary-20261028/multi-device/multi_event_restore/case-34.json)
保留该被否定的修复尝试，不将loader设置设备这一源码事实认定为根因。

关闭任务队列仅作同步诊断，完整原方法在 `check_model` 的 `correct = ref_model(...)` 处失败，
此时尚未调用 `torch.compile`。首错为eager `r.to('npu:1')` 的 `aclrtMemcpyAsync` 507899，
底层 `MemcpyAsyncTask` 返回0x7020010并提示operation unsupported。
[完整方法结果](../../../artifacts/npu-inductor-boundary-20261028/multi-device/multi_event_queue_off/case-34.json)
保留该调用链。它是运行时/驱动复制路径的边界，不能定为Inductor codegen错误或显存不足。

[纯eager脚本](../../../scripts/repro/probe_multi_device_copy.py)
保留相同shape、dtype和表达式，不导入Inductor。
[默认队列结果](../../../artifacts/npu-inductor-boundary-20261028/multi-device/exact-eager-copy-default.json)
与[关闭队列结果](../../../artifacts/npu-inductor-boundary-20261028/multi-device/exact-eager-copy-off.json)
均在16字节、kind=3跨卡复制失败。较早1024元素探针只作辅助，结论以4元素对照及完整方法为准。
这里确认当前设备组合与软件栈的复制失败，不推定所有NPU或所有设备组合均不支持跨卡复制。

```bash
ASCEND_RT_VISIBLE_DEVICES=<idle-allowed-physical-pair> \
python scripts/repro/probe_multi_device_copy.py <test-workspace>/eager-copy-default.json
TASK_QUEUE_ENABLE=0 ASCEND_RT_VISIBLE_DEVICES=<idle-allowed-physical-pair> \
python scripts/repro/probe_multi_device_copy.py <test-workspace>/eager-copy-off.json
```

第34项保留失败，没有通过关闭计时、修改设备、CPU中转或跳过原方法来宣称通过。
此前SSH连接中断已恢复；固定模板、实际CANN设备、上下文候选及eager复制实验均已取得终态。

稀疏gather的递归来自Meta构造继承了测试全局开启的sparse invariants数据检查，
并在meta tensor与CPU复制分解间重入。仅给Meta内三个 `sparse_coo_tensor` 构造传入
`check_invariants=False` 后，完整原NPU方法158.5秒通过；真实输入的全局检查继续启用。

### 第36项历史精度错误的修复PR确认

`test_split_with_sizes` 由 [PR !48448](https://gitcode.com/Ascend/pytorch/merge_requests/48448)
中的persistent reduction mask修复恢复，合入提交为 `dd35816d26c7cb8721521ce9e6860dba8858f743`，
合入时间2026-10-23 20:37（北京时间）。PR号与合入时间依据本地Ascend/pytorch合入提交记录。
此前只记录10月28日基线通过；本次补上修复归因对照，未改写历史失败或通过记录。

2026-10-29在同一NPU环境、同一测试源码和同一输入配置下，只以source loader替换
`torch_npu._inductor.codegen.triton`；其余安装模块和native-build guards保持一致。
不是在不同历史wheel之间进行完整版本回归。

| 加载模块版本 | 完整原方法结果 |
|---|---|
| PR父提交版本 | FAIL，期望-5.14039945602417，实际-6.655652046203613，绝对差1.5152525901794434，与历史失败一致 |
| PR合入版本 | PASS |
| 当前安装版本 | PASS |

[修复前完整wrapper](../../../artifacts/npu-inductor-boundary-20261028/pr48448-split-with-sizes/before/output_code.py)
虽然生成 `r0_mask = r0 < r0_numel`，却执行无mask的 `tl.load(in_ptr0 + r0, None)`，
随后直接 `tl.sum(tmp0, 0)`，将split首段边界之外的tile尾部纳入求和。
[修复后完整wrapper](../../../artifacts/npu-inductor-boundary-20261028/pr48448-split-with-sizes/after/output_code.py)
改为带 `r0_mask`、`other=0.0` 的load，并在归约前通过 `tl.where(mask, value, 0)` 再次屏蔽无效lane。
因此该问题是归约边界mask缺失导致的计算错误，不是正常浮点误差，也不是测试初始化修复了数值。

[对照汇总及范围](../../../artifacts/npu-inductor-boundary-20261028/pr48448-split-with-sizes/summary.json)、
[前后源码commit与SHA](../../../artifacts/npu-inductor-boundary-20261028/pr48448-split-with-sizes/versions.json)
及各次 `loaded-sources.txt` 保存实际加载版本；对应目录保存完整pytest结果和去敏生成代码。
