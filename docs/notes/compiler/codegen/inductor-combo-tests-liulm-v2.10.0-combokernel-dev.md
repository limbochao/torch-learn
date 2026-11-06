---
title: liulm v2.10.0-combokernel-dev 的 Combo/Foreach 功能复测
---

# liulm v2.10.0-combokernel-dev 的 Combo/Foreach 功能复测

验证日期：2026-11-06。**最新结果已刷新：全量 55/62 个方法结束，其中 46 个通过、8 个失败、1 个执行异常；其余状态按实际记录展示。**

本表按上游对照总表的行与顺序，记录 `0fd5a10` 新包叠加 F3 的最新结果；历史轮次保留在证据中。
可下载 [CSV 表格](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv)；
[上游版本对照总表](inductor-combo-tests-v2.13-vs-v2.10.md) 保留最初的版本对照与旧修复历史。

## 本轮版本与验证口径

- 新能力基线为 `liulm/v2.10.0-combokernel-dev`，提交 `0fd5a10da2a3c33cab8c200f5778d711e490ff6b`。
  使用用户提供的 `torch_npu-2.10.0.post7+git0fd5a10-cp311-cp311-linux_x86_64.whl`，通过 pip 强制重装且不变更依赖。
- **前置修正**：原 wheel 与该提交的 `scheduling.py` 有一个多余缩进空格，原样加载会出现 `IndentationError`。
  本轮仅在安装目录删除这一空格后测试；因此下文“新包”均指 **0fd5a10 + 这一处语法修正**，不是未改动 wheel 的运行结果。
  原文件、语法失败和精确补丁保存在证据目录；该修正尚未作为分支提交。
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
| 0fd5a10 + F3 全量 | 46 / 62 | 391 / 409 | 已结束 55/62 个方法，1 个执行异常 |

“原 e32+F1/F3 结果”是旧 `e32cdebf77` 基线合并 F1 `6c6454567f`、F3 `e578ec524d` 定向复测的历史记录，
共 55/62 个方法、400/409 个实例通过，既不是当前新包，也不是一次统一安装版本的全量执行。

## 原失败方法的最新复测

| 测试文件 | 方法 | 仅新包 | 新包+F3 | 当前分析 |
| --- | --- | --- | --- | --- |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red | 通过（1/1） | 通过（1/1） | F2 未修复：编译错误被新回退屏蔽，普通 kernel 路径通过 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 失败（0/1） | 失败（0/1） | F6：NPU eager 不支持 channels_last contiguous，尚未进入编译 |
| test_combo_kernels.py | test_reduce_benchmark | 失败（0/1） | 通过（1/1） | 仅新包仍为 F3；加既有 e578ec524d 修复后定向通过 |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 失败（0/1） | 通过（1/1） | 仅新包仍为 F3；加既有 e578ec524d 修复后定向通过 |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 失败（0/1） | 通过（1/1） | 仅新包仍为 F3；加既有 e578ec524d 修复后定向通过 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 失败（0/1） | 通过（1/1） | 仅新包仍为 F3；加既有 e578ec524d 修复后定向通过 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 失败（0/1） | 通过（1/1） | 仅新包仍为 F3；加既有 e578ec524d 修复后定向通过 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 失败（0/1） | 失败（0/1） | 仅新包 F3 阻塞；加 F3 后新增 F10：sum 覆盖 mean 输出，精度失败 |
| test_combo_kernels.py | test_helper_fn_defined | 失败（0/1） | 通过（1/1） | 新包单独仍被 F3 阻塞；加 F3 后通过，F9 helper 定义已验证修复 |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（0/1） | 失败（0/1） | F7：provenance 映射为空，Inductor 追溯检查失败，AOTI 未执行 |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 失败（0/2） | 失败（0/2） | F4 仍在：Combo 索引化简错误后回退，最终两个开关分支均被 shape 检查阻塞 |
| test_torchinductor.py | test_rand_like_deterministic | 失败（0/2） | 失败（0/2） | F5 仍在：Combo 索引 dtype 错误后回退，最终两个开关分支均被 shape 检查阻塞 |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 失败（0/1） | 失败（0/1） | F8：wrapper 在 nonzero 绑定 u0 前使用 u0 |

## 当前仍存在的问题

以下结论只针对新包叠加 F3 的实际执行。F1、F3、F9 在对应已完成的复测中未再出现，移出待分析范围。
F3 已同步为新基线修复提交 `113692bb66ee5879ba9b64656e4207ec8894910c`，见 [PR !4](https://gitcode.com/liujm2001/pytorch/merge_requests/4)；本轮仍使用与该方法逻辑一致的 overlay。

| 问题 | 当前状态 | 后续处理 |
| --- | --- | --- |
| F10 | 混合动态归约的 mean 被后续 sum store 覆盖，定向和最新全量均精度失败 | 最优先；成功执行后返回错误结果 |
| F8 | wrapper 在 nonzero 绑定 u0 前使用 u0，定向和最新全量均失败 | 优先；尺寸收集的 wrapper 副作用仍在 |
| F11 | 全量新增：test_mutated_benchmark、test_dynamic_shapes_mutated 的 store 使用未声明 out_ptr1 | 应定位；编译错误已确认，是否与 F10 属于同一状态传播问题尚未确认 |
| F4 | Combo 索引化简错误仍被记录，回退后两个开关分支均被 output_shape 断言阻塞 | 保留；区分线性化缺陷与 2.10 shape 检查兼容性 |
| F5 | Combo 索引 dtype 与检查不一致仍被记录，回退后同样触发 shape 断言 | 保留；尚不能认定随机数数值算法错误 |
| F2 | Combo store 类型错误仍在，用例依靠普通 kernel 回退通过 | 暂不优先分析，不能记为 Combo 缺陷已修复 |
| F7 | provenance 映射为空，AOTI 未执行 | 后置 |
| F6 | eager 不支持 channels_last，未进入编译 | 不列入当前 Combo 修复重点 |
| 环境执行异常 | test_round_robin_dispatch 采集时 NPU stream 资源不足，错误码 207005，未进入用例 | 单列待复测，不归为代码缺陷，不计通过或功能失败 |

本轮新增失败的 traceback、执行阶段与退出码保存在[全量执行快照](../../../../artifacts/combo-liulm-0fd5a10-20261106/execution/full_snapshot.json)，记录快照时间及独立校验值。
原包的缩进和版本元数据问题仍需在正式打包源码中纠正；安装目录前置修正不等于已提交修复。

## F10 的生成代码证据

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
已确认直接错误语句；具体是哪项调度/写回状态传播导致重复 store，尚未通过源码定位和提交二分确认，不能直接归因于 F3 补丁。

## 逐项结果

与上游对照总表保持 116 条记录及原始顺序，包含 62 个独立执行方法、43 个未运行方法、11 条嵌套函数记录；嵌套记录不重复计数。未运行原因与原表逐字一致。

最新全量快照：2026-11-06T16:20:25.590371+08:00。优先展示本轮全量结果，尚未结束时单独注明已有定向结果；执行异常不记为功能失败或通过。历史轮次保存在证据及上方汇总中。

| 测试文件 | 函数名 | v2.10.0存在性/实现差异 | 执行范围 | 原 e32+F1/F3 结果 | 最新结果 | 当前分析 |
| --- | --- | --- | --- | --- | --- | --- |
| test_triton_kernels.py | test_combo_fp_self_sub_rewrite | 未找到同名定义（缺修复：浮点自减的局部数值改写） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 一致 | 已有方法 | 失败（0/2） | 失败（0/2） | F4 仍存在：Combo 索引化简解包失败后回退；两个开关分支最终均被 output_shape 断言阻塞 |
| test_torchinductor.py | test_mean | 存在（差异：开启 Combo） | 已有方法 | 通过（1/1） | 待执行 | 全量尚未结束，当前无新增失败证据 |
| test_torchinductor.py | test_rand_like_deterministic | 存在（差异：增加 random.seed） | 已有方法 | 失败（0/2） | 失败（0/2） | F5 仍存在：Combo 索引 int64 与 int32 检查不一致后回退；两个开关分支最终均被 output_shape 断言阻塞 |
| test_combo_kernels.py | test_activation_functions | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_functions | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_mutated_args | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_split | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_2d_blocking_partitioning | 存在（差异：增加 C++ wrapper grid 检查） | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_persistent_reduction_size_hint | 未找到同名定义（缺修复：按最大归约维选择 size hint） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_sort_in_combo_kernel_forces_persistent_reduction | 未找到同名定义（缺修复：Combo sort 强制持久归约） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_fuse_mix_order_reductions_combo_kernels | 未找到同名定义（缺修复：混合顺序归约及消费者排除） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_scalar_store_broadcast | 未找到同名定义（缺修复：标量 store 指针广播） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_single_combo_kernels | 未找到同名定义（缺能力：单节点分区改用普通 kernel） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_per | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red_per | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red | 未找到同名定义（部分支持：共享块已有，独立块缺失） | 共享块 / 旧版分支 | 失败（0/1） | 通过（1/1） | F2 仍存在：Combo store 编译失败后回退普通 kernel，功能通过；暂不优先分析 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_r0_r1 | 未找到同名定义（缺能力：二维归约排除 Combo 策略） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi_3d | 未找到同名定义（缺能力：3D pointwise 排除策略） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size | 未找到同名定义（部分支持：共享块已有，独立块及扁平 grid 缺失） | 共享块 / 旧版分支 | 失败（0/1） | 失败（0/1） | F6 仍存在：eager 不支持 channels_last contiguous，未进入编译；不列入 Combo 修复重点 |
| test_combo_kernels.py | test_combo_kernels_pointwise_only | 未找到同名定义（缺能力：仅融合 pointwise 的开关） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_max_num_nodes | 未找到同名定义（部分支持：节点上限固定 8，不可配置） | 固定 8 节点分支 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_combo_kernel_yz_overflow | 未找到同名定义（缺修复：grid Y 溢出转移至 Z） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_stitched_config | 未找到同名定义（缺能力：免 benchmark 拼接独立块配置） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_persistent_reduction | 未找到同名定义（缺能力：免 benchmark 持久归约配置拼接） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_no_bench_carve_out | 未找到同名定义（缺能力：免 benchmark 不适配节点拆出） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_activation_benchmark | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_reduce_benchmark | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | F3 修复后通过，NumelList 分区错误未再出现，不再分析 |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_mutated_benchmark | 存在（差异：kernel 数 6/9→5～10） | 已有方法 | 通过（1/1） | 失败（0/1） | F11 全量新增：Combo 生成 store 使用未声明的 out_ptr1，Triton 编译失败；是否与 F10 写回状态有关尚待确认 |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_round_robin_dispatch | 一致 | 已有方法 | 通过（1/1） | 执行异常（有效结果 0/1） | 环境执行异常：采集用例时 NPU stream 资源不足（207005），未进入用例；不能判为代码缺陷，需复测 |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_2d_blocking_benchmark | 存在（差异：kernel 数 7～8→6） | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_persistent_reduction_no_x_dim | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | F3 修复后通过，NumelList 分区错误未再出现，不再分析 |
| test_combo_kernels.py | test_combo_kernel_per_subkernel_reduction_hint | 未找到同名定义（缺能力：逐子 kernel 归约方向提示） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_rms_norm_forwards_add_persistent_rblock | 未找到同名定义（缺能力：转发持久归约候选标记） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_foreach.py | test_single_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | 通过，当前无失败待分析 |
| test_foreach.py | test_single_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_single_scalar_tensor | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_scheduler_fusion_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | 通过，当前无失败待分析 |
| test_foreach.py | test_scheduler_fusion_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_broadcasting | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_type_promotion | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | 通过，当前无失败待分析 |
| test_foreach.py | test_kernel_split_arg_limit_list | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_fusion_duplicate_buffer_list | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | 通过，当前无失败待分析 |
| test_foreach.py | test_non_foreach_producer_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | 通过，当前无失败待分析 |
| test_foreach.py | test_non_foreach_producer_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_non_foreach_consumer_producer_list | 一致 | 已有方法 | 通过（32/32） | 通过（32/32） | 通过，当前无失败待分析 |
| test_foreach.py | test_non_foreach_consumer_producer_scalar | 一致 | 已有方法 | 通过（16/16） | 通过（16/16） | 通过，当前无失败待分析 |
| test_foreach.py | test_enable_dynamic_shapes_python_wrapper | 一致 | 已有方法 | 通过（1/1） | 待执行 | 全量尚未结束，当前无新增失败证据 |
| test_foreach.py | test_decomp | 一致 | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_foreach.py | test_fuse_concat | 一致 | 已有方法 | 通过（1/1） | 待执行 | 全量尚未结束，当前无新增失败证据 |
| test_foreach.py | test_2d_blocking | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | 通过，当前无失败待分析 |
| test_foreach.py | test_2d_blocking_partitioning_elems | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | 通过，当前无失败待分析 |
| test_foreach.py | test_2d_blocking_partitioning_mixed_sizes | 一致 | 已有方法 | 通过（20/20） | 通过（20/20） | 通过，当前无失败待分析 |
| test_foreach.py | test_reinplacing | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | 通过，当前无失败待分析 |
| test_foreach.py | test_reinplacing_mut_before | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | 通过，当前无失败待分析 |
| test_foreach.py | test_reinplacing_mut_after | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | 通过，当前无失败待分析 |
| test_foreach.py | test_aliasing | 一致 | 已有方法 | 通过（1/1） | 执行中（已完成 0/1） | 全量尚未结束，当前无新增失败证据 |
| test_foreach.py | test_2d_block_no_mixed_sizes_no_mask | 一致 | 已有方法 | 通过（1/1） | 待执行 | 全量尚未结束，当前无新增失败证据 |
| test_foreach.py | test_2d_block_mixed_sizes_with_mask | 存在（差异：启用子 kernel 独立块并检查生成代码） | 已有方法 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_foreach.py | test_foreach_map_backward_binary | 一致 | 已有方法 | 通过（10/10） | 通过（10/10） | 通过，当前无失败待分析 |
| test_foreach.py | test_foreach_map_backward_unary | 一致 | 已有方法 | 通过（4/4） | 通过（4/4） | 通过，当前无失败待分析 |
| test_snode_runtime.py | test_foreach_basic | 未找到同名定义（缺能力：调度节点输入距离指标） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_snode_runtime.py | test_foreach_after_extern | 未找到同名定义（缺能力：外部算子后的输入距离指标） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_torchinductor.py | test_combo_kernel_store_mask | 未找到同名定义（缺修复：动态 Combo 常量索引 store mask） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_unbacked_symints.py | test_combo_kernel_size_hint_failure | 一致 | 已有方法 | 修复后仍失败（0/1） | 失败（0/1） | F8 仍存在：wrapper 在 nonzero 绑定 u0 前使用 u0，优先分析 |
| test_combo_kernels.py | test_combo_kernel_dynamic_shapes_grid_changes | 未找到同名定义（部分支持：动态 grid 已有，独立块扁平 grid 缺失） | 共享块 / 旧版分支 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_dynamic_shapes_activations | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 存在（差异：kernel 数 7～8→6） | 已有方法 | 修复后通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_dynamic_shapes_reduce | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | F3 修复后通过，NumelList 分区错误未再出现，不再分析 |
| test_combo_kernels.py | test_reduce | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_mutated | 存在（差异：增加 XPU 跳过条件） | 已有方法 | 修复后通过（1/1） | 失败（0/1） | F11 全量新增：Combo 生成 store 使用未声明的 out_ptr1，Triton 编译失败；是否与 F10 写回状态有关尚待确认 |
| test_combo_kernels.py | test_mutated | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_activations_no_autotune | 存在（差异：kernel 数=5 改为源码编译调用数=1） | 已有方法 | 修复后通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_activations | 一致 | 随外层方法验证 | 随外层方法验证 | 随外层方法验证 | 嵌套函数，不独立计数 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | F3 修复后通过，NumelList 分区错误未再出现，不再分析 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_no_x_dim_2 | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | F3 修复后通过，NumelList 分区错误未再出现，不再分析 |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking_round_robin | 一致 | 已有方法 | 修复后通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda | 存在（差异：装饰器仅限 CUDA） | 已有方法 | 修复后通过（1/1） | 失败（0/1） | F10 仍存在：mean 输出被后续 sum store 覆盖，精度失败；最优先分析 |
| test_combo_kernels.py | test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu | 未找到同名定义（待验证：已有同体 GPU 测试，XPU 待验证） | 未运行 | 未运行 | 未运行 | XPU 专项，不在 NPU 验证范围 |
| test_torchinductor.py | test_combo_kernel_filter_cpu | 一致 | 已有方法 | 通过（1/1） | 待执行 | 全量尚未结束，当前无新增失败证据 |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 一致 | 已有方法 | 失败（0/1） | 失败（0/1） | F7 仍存在：provenance 映射为空，AOTI 未执行；后置 |
| test_user_streams.py | test_combo_kernel_fusion_within_same_stream | 未找到同名定义（缺能力：按 stream 分组及传递 stream） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_tma_descriptor_block_name | 未找到同名定义（缺修复：Combo TMA 前导代码及块名替换） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_amd_special_config_args | 未找到同名定义（缺修复：独立块路径的 HIP 编译参数传递） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_helper_fn_defined | 一致 | 已有方法 | 修复后仍失败（0/1） | 执行中（已完成 0/1）；定向：通过（1/1） | F9 已修复：helper 用例实际 Combo 路径通过，不再分析；F3 错误未再出现（全量待完成） |
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
| test_combo_kernels.py | test_combo_inductor_meta_no_kernel_num_gb_without_profile | 未找到同名定义（已有行为：默认不输出统计字段；新增覆盖） | 旧版默认行为 | 通过（1/1） | 通过（1/1） | 通过，当前无失败待分析 |
| test_combo_kernels.py | test_benchmark_combo_kernel_emits_real_num_gb | 未找到同名定义（缺能力：benchmark 使用真实字节量） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |
| test_combo_kernels.py | test_combo_kernel_peak_memory_wide_resnet | 未找到同名定义（缺能力：按峰值内存限制融合） | 未运行 | 未运行 | 未运行 | 缺新增能力或场景修复，按范围排除 |

## 证据与复现

[0fd5a10 安装、定向及全量证据](../../../../artifacts/combo-liulm-0fd5a10-20261106/README.md) 记录实际版本、前置语法修正、
逐实例结果、三阶段报告、回退路径、F3 overlay 和 F9/F10 的生成代码。
[a369f6c 完整历史证据](../../../../artifacts/combo-liulm-a369f6c-20261106/README.md) 保持独立，未覆盖。
表格不放源码链接、行号或完整限定名；完整 pytest node ID 只保留在执行证据。
