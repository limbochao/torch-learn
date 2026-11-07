---
title: liulm v2.10.0-combokernel-dev 的 Combo/Foreach 功能复测
---

# liulm v2.10.0-combokernel-dev 的 Combo/Foreach 功能复测

验证日期：2026-11-07。**本轮全量已完成：62 个方法中 60 个通过、2 个失败；409 个实例中 407 个通过、2 个失败，0 跳过、0 执行异常。**

本表按上游对照总表的行与顺序记录 `a8dcca84bf` 本轮完整全量结果；历史定向记录保留在后文。
可下载 [CSV 表格](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv)；
[上游版本对照总表](inductor-combo-tests-v2.13-vs-v2.10.md) 保留最初的版本对照与旧修复历史。

## 五个修复的逐行评审文档

[五篇文档总览](inductor-combo-fixes-v2.10-review.md)分别记录 F10/F12、F11、F8、F4、F5，
包含完整提交 diff、逐行解释、两版社区源码对照、正确性与性能风险，以及已有验证证据。

## 最新全量复测

完成时间：2026-11-07 01:31:59（北京时间）。11 项安装源码运行前后校验一致；六个已修复问题涉及的 8 个方法、10 个实例均通过。

启动源码提交 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，[查看 PR](https://gitcode.com/liujm2001/pytorch/merge_requests/7)。
本轮按 62 个方法、409 个实例启动，实际记录 409 个实例；实例通过 407、失败 2、跳过 0。数值断言和测试适配未变更，没有加载函数修复或 F3 overlay。当前逐项结果均来自本轮，历史定向记录保留在后文。

## F4 优化与历史定向复测

F4 已整理为单个提交 `53f8af09f144aaaec685562acd4c3a3ffaffb71a`，开发分支最新提交为 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，已推送 origin。
在原有整除项提取的基础上，利用循环索引范围估计各项余数上界；仅当**全部余数上界之和小于正除数**时，
才将剩余加法进一步拆成独立除法。证明不了时保留合并除法，不新增运行时判断或 shape guard。
例如 `0≤x≤1、0≤y≤5` 时，`(10*x+y)//8` 可以安全拆分；y 上界增至 6 时拒绝。
`(8*i+10)//3` 在 `i=0..5` 有进位，仍保留合并除法。

本次在 NPU_A5 物理卡 2 上仅复测 `test_adaptive_avg_pool2d1` 的 Combo False/True 两个实例，**2/2 PASS**，
覆盖原有三组输入及低精度检查。使用已推送提交的完整安装源码，11 项源码运行前后校验一致，未加载函数修复 overlay。
另对 500 组随机范围、13,982 个取值点检查等价性，覆盖余数恰好进位、多项共同进位、负数项和未知范围。
生成 DSL 中的 pooling 语义 mask 和可能进位的合并边界表达式仍保留。

原 F4 `a2f7b18da9` 与本次优化合并为 `53f8af09f1`；F5 仅因重放从 `bc9d3c4ee9` 变为 `a8dcca84bf`，
其补丁内容完全一致。该历史定向轮次仅复测 F4，F5 当时沿用重放前验证；本轮全量已重新覆盖 F5，表中提交更新为 `a8dcca84bf`。本轮未进行性能测量。
证据见 [F4 优化与复测记录](../../../../artifacts/combo-f4-nocarry-20261106/README.md)。

## 六问题定向修复验证

本轮基线为 `85dec1f1a10ee42ebce87f1a1f8d2095909c0af6`，F3 已合入，未再加载旧 F3 overlay。
开发分支为 `fix/combo-f10-f12-f11-f8-f4-f5-v2.10`，前一轮 10 实例验证的安装源码为 `bc9d3c4ee973dde621ebf61600967ffbd66d0571`。
前一轮仅运行指定六个问题涉及的 **8 个失败方法、10 个参数实例**：原新包 0/10 通过，修复后 **10/10 通过**。
该历史定向轮次未重跑全量；随后本轮在统一安装源码 `a8dcca84bf` 上完整运行了 62 个方法、409 个实例，见上方全量结果。

环境为 Ascend950PR、torch `2.10.0+cpu`、torch_npu `2.10.0.post7+git85dec1f`、Triton `3.6.0`。
先以源码函数 overlay 逐项定位和验证，再从已推送提交提取四个完整 Python 文件覆盖安装目录后，重新执行全部 10 个目标实例。
最终一轮不加载函数修复 overlay；11 项安装源码运行前后校验一致，四个修改文件的 SHA256 与提交完全一致。
测试输入、stride、dtype、数值/梯度/突变断言和精度阈值沿用原适配，未新增正式测试、未关闭 benchmark。
物理卡 4、5、6 按本次用户授权使用；共享卡的时间数据不用于性能结论。

| 问题 | 修复提交 | 目标实例结果 | 修复内容 |
| --- | --- | --- | --- |
| F10 | `64a3e7e8b9` | 1/1 PASS | Combo 最终渲染使用子 kernel 的 inplace 删除状态，防止 sum 覆盖 mean |
| F12 | `64a3e7e8b9` | 1/1 PASS | 已确认与 F10 同源，共用同一个修复提交 |
| F11 | `9b1a894d65` | 3/3 PASS | 汇总 removed_buffers，删除已移出签名的中间 buffer store |
| F8 | `fdf73f229e` | 1/1 PASS | benchmark 尺寸收集不提前向 wrapper 写入未绑定的 u0 |
| F4 | `53f8af09f1` | 2/2 PASS（最新复测） | 保留安全拆分机会，修正常量处理、加法进位及 pooling 语义 mask 丢失 |
| F5 | `a8dcca84bf`（原 `bc9d3c4ee9`） | 2/2 PASS（本轮全量） | 普通 Index 与 Combo 都注册值为 1 的 constexpr 参数；旧 dtype 问题未复现，无需修改 |

F10/F12 共用根因，六个问题对应五个有效修复提交。F4 方法覆盖 16×16、3×3、6×6 三组输入及原有低精度检查。
F5 的缺失常量导致 seed offset 在 Triton IR 中成为 `None`，不代表随机数数值算法有误。
当前通过项的“当前分析”统一为 `—`；修复提交写在“最新结果”。

完整证据和具体原因见[修复验证记录](../../../../artifacts/combo-fixes-85dec1f-20261106/README.md)。

## 历史版本与验证口径（0fd5a10）

- 新能力基线为 `liulm/v2.10.0-combokernel-dev`，提交 `0fd5a10da2a3c33cab8c200f5778d711e490ff6b`。
  使用用户提供的 `torch_npu-2.10.0.post7+git0fd5a10-cp311-cp311-linux_x86_64.whl`，通过 pip 强制重装且不变更依赖。
- **前置修正**：原 wheel 与该提交的 `scheduling.py` 有一个多余缩进空格，原样加载会出现 `IndentationError`。
  当时仅在安装目录删除这一空格后测试；该历史轮次的“新包”均指 **0fd5a10 + 这一处语法修正**，不是未改动 wheel 的运行结果。
  原文件、语法失败和精确补丁保存在证据目录；当时未形成独立修复提交。
- wheel 的 METADATA 仍标为 `a369f6c`，但模块版本和实际源码为 `0fd5a10`。10 项 torch_npu 源码与本地提交校验一致，
  torch 侧第 11 项源码与上轮一致；修正后的 11 项校验值在每轮测试前后均一致。不能仅凭 pip 安装成功提示判断运行版本。
- 环境为 NPU_A5 Docker、Ascend950PR，torch `2.10.0+cpu`、torch_npu `2.10.0.post7+git0fd5a10`、Triton `3.6.0`、CANN `9.1.0`。
  用户授权共用物理卡 1、2、3，各卡内部串行、卡间并行；本轮只比较功能，不据此比较性能。
- 已有 F1 的 `kernel_benchmark_extra_args()` 实现已包含在新基线中，未重复覆盖。已有 F3 来自 `e578ec524d592475fd52702b975ce03d1a95bbf2`，
  通过进程内 overlay 仅替换 `horizontal_partition()`；不覆盖整个旧文件，保留新包的回退与 helper 修复。每个 pytest 进程记录实际加载的修复及源码哈希。
- 执行顺序：新包单独复测原失败项 → 新包+F3 复测原失败项 → 更新推送表格 → 新包+F3 全量复测。
  定向范围是上轮 13 个失败方法、15 个实例；全量范围为此前核定的 62 个方法、409 个实例。
- 测试来自 `Community-Use-Case-Testing` 提交 `be6049a0059677ca408da1360653b694b6d1afb5` 的 v2.13.0 NPU 目录，
  五个执行文件与上一轮最终适配逐字节一致；保留输入、stride、dtype、输出/梯度/突变检查和精度阈值。
  原有依赖适配及 CUDA 专属 kernel 数、源码结构、profiler/grid 检查的处理保持一致。未新增测试方法，未关闭 benchmark。
- 功能通过不保证实际采用 Combo，也不代表达到 CUDA 的 grid、融合数量或性能要求。观测记录明确区分 Combo 成功和失败后回退。
  “v2.10.0存在性/实现差异”仍是上游版本对照，不代表原始 2.13 文件可不经适配直接运行。

## 各轮结果

| 版本 / 范围 | 方法全部通过 / 总数 | 实例通过 / 总数 | 说明 |
| --- | ---: | ---: | --- |
| a369f6c 全量 | 49 / 62 | 394 / 409 | 上一轮完整执行，未加载旧 F3 |
| 0fd5a10 定向 | 1 / 13 | 1 / 15 | 原失败项，仅 F2 用例依靠回退通过 |
| 0fd5a10 + F3 定向 | 7 / 13 | 7 / 15 | 6 个方法仍失败，含新增 F10 |
| 0fd5a10 + F3 全量 | 52 / 62 | 397 / 409 | 本轮全量原始记录；后续复测见下表 |
| 85dec1f 原包定向 | 0 / 8 | 0 / 10 | 六个指定问题对应失败方法，F3 已合入 |
| 85dec1f + 原五个修复定向 | 8 / 8 | 10 / 10 | 安装源码 bc9d3c4ee9；仅目标失败项，非全量 |
| F4 余数不进位优化 | 1 / 1 | 2 / 2 | 安装源码 a8dcca84bf；只重跑 F4 |
| a8dcca84bf 完整全量 | 60 / 62 | 407 / 409 | 统一安装源码的本轮全量 |

“原 e32+F1/F3 结果”是旧 `e32cdebf77` 基线合并 F1 `6c6454567f`、F3 `e578ec524d` 定向复测的历史记录，
共 55/62 个方法、400/409 个实例通过，既不是当前新包，也不是一次统一安装版本的全量执行。

## 历史定向复测（0fd5a10）

| 测试文件 | 方法 | 仅新包 | 新包+F3 | 历史说明 |
| --- | --- | --- | --- | --- |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 失败（0/1） | 失败（0/1） | F6：NPU eager 不支持 channels_last contiguous，尚未进入编译 |
| test_combo_kernels.py | test_reduce_benchmark | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 失败（0/1） | 失败（0/1） | — |
| test_combo_kernels.py | test_helper_fn_defined | 失败（0/1） | 通过（1/1） | — |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（0/1） | 失败（0/1） | F7：provenance 映射为空，Inductor 追溯检查失败，AOTI 未执行 |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 失败（0/2） | 失败（0/2） | — |
| test_torchinductor.py | test_rand_like_deterministic | 失败（0/2） | 失败（0/2） | — |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 失败（0/1） | 失败（0/1） | — |

## 失败项复测

85dec1f + 五个修复提交（a8dcca84bf）完整全量，F3 已合入，独立目录 `full_a8dcca84bf_20261106`，复测 62 个方法、409 个实例：通过 407，失败 2。数值断言与测试适配均未变更。

| 测试文件 | 函数名 | 最新结果 | 当前分析 |
| --- | --- | --- | --- |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 通过（2/2）；修复 53f8af09f1 | — |
| test_torchinductor.py | test_mean | 通过（1/1）；修复 64a3e7e8b9 | — |
| test_torchinductor.py | test_rand_like_deterministic | 通过（2/2）；修复 a8dcca84bf | — |
| test_combo_kernels.py | test_activation_functions | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce_functions | 通过（1/1） | — |
| test_combo_kernels.py | test_mutated_args | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce_split | 通过（1/1） | — |
| test_combo_kernels.py | test_2d_blocking_partitioning | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_per | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red_per | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 失败（0/1） | F6 仍存在：NPU eager 的 contiguous 不支持 channels_last memory format，失败发生在 torch.compile 之前。 |
| test_combo_kernels.py | test_combo_kernel_max_num_nodes | 通过（1/1） | — |
| test_combo_kernels.py | test_activation_benchmark | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce_benchmark | 通过（1/1） | — |
| test_combo_kernels.py | test_mutated_benchmark | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_round_robin_dispatch | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_2d_blocking_benchmark | 通过（1/1） | — |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 通过（1/1） | — |
| test_foreach.py | test_single_list | 通过（32/32） | — |
| test_foreach.py | test_single_scalar | 通过（16/16） | — |
| test_foreach.py | test_single_scalar_tensor | 通过（16/16） | — |
| test_foreach.py | test_scheduler_fusion_list | 通过（32/32） | — |
| test_foreach.py | test_scheduler_fusion_scalar | 通过（16/16） | — |
| test_foreach.py | test_broadcasting | 通过（16/16） | — |
| test_foreach.py | test_type_promotion | 通过（20/20） | — |
| test_foreach.py | test_kernel_split_arg_limit_list | 通过（16/16） | — |
| test_foreach.py | test_fusion_duplicate_buffer_list | 通过（20/20） | — |
| test_foreach.py | test_non_foreach_producer_list | 通过（32/32） | — |
| test_foreach.py | test_non_foreach_producer_scalar | 通过（16/16） | — |
| test_foreach.py | test_non_foreach_consumer_producer_list | 通过（32/32） | — |
| test_foreach.py | test_non_foreach_consumer_producer_scalar | 通过（16/16） | — |
| test_foreach.py | test_enable_dynamic_shapes_python_wrapper | 通过（1/1） | — |
| test_foreach.py | test_decomp | 通过（1/1） | — |
| test_foreach.py | test_fuse_concat | 通过（1/1） | — |
| test_foreach.py | test_2d_blocking | 通过（20/20） | — |
| test_foreach.py | test_2d_blocking_partitioning_elems | 通过（20/20） | — |
| test_foreach.py | test_2d_blocking_partitioning_mixed_sizes | 通过（20/20） | — |
| test_foreach.py | test_reinplacing | 通过（4/4） | — |
| test_foreach.py | test_reinplacing_mut_before | 通过（4/4） | — |
| test_foreach.py | test_reinplacing_mut_after | 通过（4/4） | — |
| test_foreach.py | test_aliasing | 通过（1/1） | — |
| test_foreach.py | test_2d_block_no_mixed_sizes_no_mask | 通过（1/1） | — |
| test_foreach.py | test_2d_block_mixed_sizes_with_mask | 通过（1/1） | — |
| test_foreach.py | test_foreach_map_backward_binary | 通过（10/10） | — |
| test_foreach.py | test_foreach_map_backward_unary | 通过（4/4） | — |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 通过（1/1）；修复 fdf73f229e | — |
| test_combo_kernels.py | test_combo_kernel_dynamic_shapes_grid_changes | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_activations | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_mutated | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_dynamic_shapes_activations_no_autotune | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking_round_robin | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 通过（1/1）；修复 64a3e7e8b9 | — |
| test_torchinductor.py | test_combo_kernel_filter_cpu | 通过（1/1） | — |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（0/1） | F7 仍存在：provenance 的 cppCodeToPost 映射为空（预期 1 项），Inductor 追溯断言失败；AOTI 未执行。 |
| test_combo_kernels.py | test_helper_fn_defined | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_inductor_meta_no_kernel_num_gb_without_profile | 通过（1/1） | — |

## 当前仍存在的问题

本轮剩余失败为 F6、F7，已依据本轮日志及实际执行的用例源码核对失败阶段；没有新增失败项。F7 映射生成或登记为空的具体原因仍待源码分析。

| 测试文件 | 函数名 | 最新结果 | 当前分析 |
| --- | --- | --- | --- |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 失败（0/1） | F6 仍存在：NPU eager 的 contiguous 不支持 channels_last memory format，失败发生在 torch.compile 之前。 |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（0/1） | F7 仍存在：provenance 的 cppCodeToPost 映射为空（预期 1 项），Inductor 追溯断言失败；AOTI 未执行。 |

## F10 的历史生成代码证据（已修复）

`test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda` 使用 `x=(16,128)`、`y=(32,128)`、`z=(32,256)`，
返回 `x.sum(1), y.mean(1), z.max(1)`；三个输入的首维标为动态。实际生成的同一 Combo 分支中：

```python
            tmp10 = tl.sum(tmp8, 1).reshape(X0BLOCK_SUB, 1)
            tmp11 = 128.0
            tmp12 = (tmp10 / tmp11)
            tl.store(in_out_ptr0 + (x0), tmp12, x0_mask)
            tl.store(in_out_ptr0 + (x0 ), tmp10, x0_mask)
```

第二次 store 对相同地址写回 sum，覆盖了第一次正确的 mean。实际断言显示第二个输出 32/32 个元素不一致，
最大绝对误差约 69.52、相对误差约 0.99218756，与 sum/mean 相差 128 倍的行为一致。
本例观测到 Combo 成功生成且没有回退；原格式完整 wrapper 和候选源码保存在新证据目录中。
本轮已定位为子 kernel 删除状态未传给最终 DeferredLine 渲染；修复 `64a3e7e8b9` 后，F10 和同源 F12 的目标路径均通过。

## 逐项结果

与上游对照总表保持 116 条记录及原始顺序，包含 62 个独立执行方法、43 个未运行方法、11 条嵌套函数记录；嵌套记录不重复计数。未运行原因与原表逐字一致。

最新结果来源：`full_a8dcca84bf_20261106` 的完整全量报告；62 个独立执行方法均使用本轮结果，43 个未运行方法及 11 条嵌套记录的处理保持原表一致。历史轮次保存在证据及上方汇总中。

| 测试文件 | 函数名 | v2.10.0存在性/实现差异 | 执行范围 | 原 e32+F1/F3 结果 | 最新结果 | 当前分析 |
| --- | --- | --- | --- | --- | --- | --- |
| test_triton_kernels.py | test_combo_fp_self_sub_rewrite | 未找到同名定义（缺修复：浮点自减的局部数值改写） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 一致 | 已有方法 | 失败（0/2） | 通过（2/2）；修复 53f8af09f1 | — |
| test_torchinductor.py | test_mean | 存在（差异：开启 Combo） | 已有方法 | 通过（1/1） | 通过（1/1）；修复 64a3e7e8b9 | — |
| test_torchinductor.py | test_rand_like_deterministic | 存在（差异：增加 random.seed） | 已有方法 | 失败（0/2） | 通过（2/2）；修复 a8dcca84bf | — |
| test_combo_kernels.py | test_activation_functions | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_functions | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_mutated_args | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_split | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_2d_blocking_partitioning | 存在（差异：增加 C++ wrapper grid 检查） | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_persistent_reduction_size_hint | 未找到同名定义（缺修复：按最大归约维选择 size hint） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_sort_in_combo_kernel_forces_persistent_reduction | 未找到同名定义（缺修复：Combo sort 强制持久归约） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_fuse_mix_order_reductions_combo_kernels | 未找到同名定义（缺修复：混合顺序归约及消费者排除） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_scalar_store_broadcast | 未找到同名定义（缺修复：标量 store 指针广播） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_single_combo_kernels | 未找到同名定义（缺能力：单节点分区改用普通 kernel） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_per | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red_per | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_r0_r1 | 未找到同名定义（缺能力：二维归约排除 Combo 策略） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi_3d | 未找到同名定义（缺能力：3D pointwise 排除策略） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 未找到同名定义（部分支持：共享块已有，独立块及扁平 grid 缺失） | 共享块 / 旧版分支 | 失败（0/1） | 失败（0/1） | F6 仍存在：NPU eager 的 contiguous 不支持 channels_last memory format，失败发生在 torch.compile 之前。 |
| test_combo_kernels.py | test_combo_kernels_pointwise_only | 未找到同名定义（缺能力：仅融合 pointwise 的开关） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_max_num_nodes | 未找到同名定义（部分支持：节点上限固定 8，不可配置） | 固定 8 节点分支 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_yz_overflow | 未找到同名定义（缺修复：grid Y 溢出转移至 Z） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_stitched_config | 未找到同名定义（缺能力：免 benchmark 拼接独立块配置） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_persistent_reduction | 未找到同名定义（缺能力：免 benchmark 持久归约配置拼接） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_carve_out | 未找到同名定义（缺能力：免 benchmark 不适配节点拆出） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_activation_benchmark | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_benchmark | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_mutated_benchmark | 存在（差异：kernel 数 6/9→5～10） | 已有方法 | 通过（1/1） | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_round_robin_dispatch | 一致 | 已有方法 | 通过（1/1） | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_2d_blocking_benchmark | 存在（差异：kernel 数 7～8→6） | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_combo_kernel_per_subkernel_reduction_hint | 未找到同名定义（缺能力：逐子 kernel 归约方向提示） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_rms_norm_forwards_add_persistent_rblock | 未找到同名定义（缺能力：转发持久归约候选标记） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_foreach.py | test_single_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | — |
| test_foreach.py | test_single_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_single_scalar_tensor | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_scheduler_fusion_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | — |
| test_foreach.py | test_scheduler_fusion_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_broadcasting | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_type_promotion | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | — |
| test_foreach.py | test_kernel_split_arg_limit_list | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_fusion_duplicate_buffer_list | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | — |
| test_foreach.py | test_non_foreach_producer_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | — |
| test_foreach.py | test_non_foreach_producer_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_non_foreach_consumer_producer_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | — |
| test_foreach.py | test_non_foreach_consumer_producer_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | — |
| test_foreach.py | test_enable_dynamic_shapes_python_wrapper | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_decomp | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_fuse_concat | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_2d_blocking | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | — |
| test_foreach.py | test_2d_blocking_partitioning_elems | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | — |
| test_foreach.py | test_2d_blocking_partitioning_mixed_sizes | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | — |
| test_foreach.py | test_reinplacing | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | — |
| test_foreach.py | test_reinplacing_mut_before | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | — |
| test_foreach.py | test_reinplacing_mut_after | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | — |
| test_foreach.py | test_aliasing | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_2d_block_no_mixed_sizes_no_mask | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_2d_block_mixed_sizes_with_mask | 存在（差异：启用子 kernel 独立块并检查生成代码） | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_foreach.py | test_foreach_map_backward_binary | 一致 | 已有方法 | 通过（10/10） | 通过（10/10） | — |
| test_foreach.py | test_foreach_map_backward_unary | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | — |
| test_snode_runtime.py | test_foreach_basic | 未找到同名定义（缺能力：调度节点输入距离指标） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_snode_runtime.py | test_foreach_after_extern | 未找到同名定义（缺能力：外部算子后的输入距离指标） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_torchinductor.py | test_combo_kernel_store_mask | 未找到同名定义（缺修复：动态 Combo 常量索引 store mask） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 一致 | 已有方法 | 修复后仍失败（0/1） | 通过（1/1）；修复 fdf73f229e | — |
| test_combo_kernels.py | test_combo_kernel_dynamic_shapes_grid_changes | 未找到同名定义（部分支持：动态 grid 已有，独立块扁平 grid 缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_activations | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 存在（差异：kernel 数 7～8→6） | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_mutated | 存在（差异：增加 XPU 跳过条件） | 已有方法 | 修复后通过（1/1） | 通过（1/1）；修复 9b1a894d65 | — |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_activations_no_autotune | 存在（差异：kernel 数=5 改为源码编译调用数=1） | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking_round_robin | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 存在（差异：装饰器仅限 CUDA） | 已有方法 | 修复后通过（1/1） | 通过（1/1）；修复 64a3e7e8b9 | — |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu | 未找到同名定义（待验证：已有同体 GPU 测试，XPU 待验证） | 未运行 | 未运行 | 未运行 | XPU 专项，不在 NPU 验证范围 |
| test_torchinductor.py | test_combo_kernel_filter_cpu | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | — |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 一致 | 已有方法 | 失败（0/1） | 失败（0/1） | F7 仍存在：provenance 的 cppCodeToPost 映射为空（预期 1 项），Inductor 追溯断言失败；AOTI 未执行。 |
| test_user_streams.py | test_combo_kernel_fusion_within_same_stream | 未找到同名定义（缺能力：按 stream 分组及传递 stream） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_tma_descriptor_block_name | 未找到同名定义（缺修复：Combo TMA 前导代码及块名替换） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_amd_special_config_args | 未找到同名定义（缺修复：独立块路径的 HIP 编译参数传递） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_helper_fn_defined | 一致 | 已有方法 | 修复后仍失败（0/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_pdl_codegen_in_combo_kernel | 未找到同名定义（缺能力：Combo PDL 代码生成） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_pdl_combo_kernel_pointwise | 未找到同名定义（缺能力：Combo pointwise PDL 接入） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_pdl_combo_kernel_reduction | 未找到同名定义（缺能力：Combo reduction PDL 接入） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_max_autotune | 未找到同名定义（缺能力：独立块逐子 kernel 最大调优） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_max_autotune_with_reduction | 未找到同名定义（缺能力：独立块归约最大调优） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_autotune_many_subkernels | 未找到同名定义（缺能力：逐子 kernel 串行调优策略） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_autotune_grouping | 未找到同名定义（缺能力：相同子 kernel 分组调优） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_autotune_grouping_by_metadata_fingerprint | 未找到同名定义（缺能力：按完整元数据指纹分组） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_max_persistent_rblock_gated_on_per_subkernel | 未找到同名定义（缺能力：持久归约块上限及分模式控制） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_coordesc_tunes_largest_subkernel_first | 未找到同名定义（缺能力：Combo 坐标下降及大子 kernel 优先） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_inductor_meta_has_optimize_mem | 未找到同名定义（缺能力：Combo 内存优化标记） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_inductor_meta_optimize_mem_false_in_training_forward | 未找到同名定义（缺能力：训练前向内存优化标记） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_triton_meta_has_disable_ftz | 未找到同名定义（缺能力：disable_ftz 配置及传递） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields | 未找到同名定义（缺能力：子 kernel pointwise 元数据） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_reduction_combo_grid_meta_has_per_subkernel_fields | 未找到同名定义（缺能力：子 kernel reduction 元数据） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel | 未找到同名定义（缺能力：子 kernel 连续归约维元数据） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_per_kernel_inductor_meta_matches_standalone | 未找到同名定义（缺能力：与普通 kernel 一致的子级元数据） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_num_gb_and_flop_match_standalone_sum | 未找到同名定义（缺能力：Combo 字节量及 FLOP 汇总） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth | 未找到同名定义（缺能力：带宽分析的 Combo 字节量） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_inductor_meta_no_kernel_num_gb_without_profile | 未找到同名定义（已有行为：默认不输出统计字段；新增覆盖） | 旧版默认行为 | 通过（1/1） | 通过（1/1） | — |
| test_combo_kernels.py | test_benchmark_combo_kernel_emits_real_num_gb | 未找到同名定义（缺能力：benchmark 使用真实字节量） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_peak_memory_wide_resnet | 未找到同名定义（缺能力：按峰值内存限制融合） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |

## 证据与复现

[0fd5a10 安装、定向及全量证据](../../../../artifacts/combo-liulm-0fd5a10-20261106/README.md) 记录实际版本、前置语法修正、
逐实例结果、三阶段报告、回退路径、F3 overlay 和 F9/F10 的生成代码。
[a369f6c 完整历史证据](../../../../artifacts/combo-liulm-a369f6c-20261106/README.md) 保持独立，未覆盖。
表格不放源码链接、行号或完整限定名；完整 pytest node ID 只保留在执行证据。
