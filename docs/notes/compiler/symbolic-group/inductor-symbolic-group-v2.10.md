---
title: torch_npu v2.10.0 Symbolic Group 实现
---

# torch_npu v2.10.0 Symbolic Group 实现

## 1. 文档边界

本文只依据 `pytorch_new` 仓库 `v2.10.0` tag（commit `cd0e5c85624308422b5a3ae2ce5e0dd4c4b6a20c`）中的实现，说明已经进入该版本的 symbolic grouped autotune 行为。

本文明确**不包含** range-aware、memory-aware、10/20 GiB 限制或按有效符号范围动态生成档位等后续设计。相关未实现设想单独记录在 [Range/Memory-Aware 分组设计草案](../../../requirements-design/symbolic-group/inductor-symbolic-group-range-memory-aware.md)。

## 2. 开关

`torch_npu/_inductor/config.py` 在模块导入时读取：

```bash
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1
```

可选 rollout 开关 `INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES` 默认允许：

```text
pointwise,reduction,persistent_reduction
```

由于配置在 import 阶段加载，环境变量必须在导入 `torch`/`torch_npu` 前设置。只在 Python 进程中后置修改环境变量，不会重新初始化已经加载的 config。

## 3. Codegen 准入

入口是 `SplitTiling.select_split_tiling_axis()`，完成普通 split/tiling axis 选择后调用 `apply_grouped_rewrite_if_needed()`。

主要步骤：

1. 检查全局开关和 template allowlist。
2. 从 dynamic split axes 中选 primary group axis；secondary dynamic split axes 会从 split axis 降级。
3. 若没有 dynamic split axis，只对受支持的单个 dynamic tiling axis 场景尝试分组。
4. 构造 `GroupedKernelMeta`；无法构造可靠 feature 或 representative 时，禁用本 kernel 的 group 路径。

pointwise 还会按 scheduler dependency 和 LoopBody primitive 保守识别 `group_workload=elementwise`。识别失败不会阻止普通 pointwise group feature，也不会改变数学 DSL。

## 4. v2.10.0 固定档位

`SplitTiling._build_group_features()` 在 v2.10.0 中使用固定经验边界：

| 场景 | feature source | buckets |
| --- | --- | --- |
| reduction outer | `outer_product` | `(256,)` |
| reduction axis | `reduction_product` | `(8192,)` |
| elementwise pointwise | `outer_product` | `(num_vector_core * 4096,)` |
| generic pointwise | `outer_product` | `(num_vector_core * 4096,)` |
| broadcast / transpose+broadcast | `axis` | `(16, 64, 256, 1024, 4096)` |
| transpose | `axis` | `(64, 128, 256, 512)` |

这里没有读取 `mark_dynamic` 的 min/max 来裁剪 buckets，也没有根据 tensor 内存占用反推符号范围。

## 5. Group representative

`runtime/symbolic_grouping.py::build_group_representatives()` 为所有 bucket 组合构造代表值：

- closed bucket 使用上界作为代表 feature value。
- open tail 使用最后边界的两倍。
- product feature 会除去静态 axis 因子，反算唯一动态 axis 的代表值。
- 同一个 feature 内出现多个独立 symbolic axes 时，当前 representative builder 不支持。
- 无法落入 bucket 的组合标记为 unreachable；所有 group 都不可达时拒绝 grouped plan。

输出包含 `reachable_group_ids`、每组 `benchmark_feature_inputs` 和 `benchmark_axis_values`。这些值用于构造 autotune 输入，不等同于某次真实 runtime shape。

## 6. Candidate plan

`_triton_config_npu_index_grouped()` 对每个 reachable group 执行以下操作：

1. 用代表 axis values 覆盖普通 size hints。
2. 调用 legacy NPU config generator 得到该组候选。
3. 将 runtime block 从 compile variant 中剥离，去重得到 `variants`。
4. 为每个 group/variant 构造 launch policy，包括 static blocks、runtime block rules 和 grid target。
5. 保存 `group_to_candidates`、`policies`、axis/feature arg indices 及 benchmark arg specs。

最终 plan 写入 generated kernel 的 `inductor_meta['grouped_candidate_plan']`。同时可见的通用 metadata 包括：

```text
group_enabled
group_template
group_workload
primary_group_axis
static_split_axes
secondary_runtime_symbolic_axes
group_features
runtime_block_arg_names
```

## 7. Runtime autotune 与 dispatch

`NPUSymbolicGroupedAutotuner` 负责运行时行为。第一次进入该 kernel 时：

1. 预编译 plan 中去重后的 variants。
2. 按每个 reachable group 的 representative materialize benchmark tensors 和 size args。
3. benchmark 每组候选，并为每个 group 保存独立 best candidate/launcher。

因此不是“runtime shape 落到哪个 group 就只 tune 哪个 group”；首次 grouped autotune 会确保全部 reachable groups 已完成选择。

正式 dispatch 时从真实 runtime args 计算 feature product/axis value，按 buckets 解析 `group_id`，再取该组 winner。launch 前根据 policy 和真实 axis numel materialize runtime blocks，最后执行对应 launcher。

日志 `grouped dispatch group` 可直接观察：

```text
kernel, feature_inputs, group_id, selected_config, runtime_blocks
```

## 8. 回退边界

group metadata 或 benchmark arg spec 无法安全建立时，codegen 会把 `group_enabled` 置为 `False`。对于仍为动态 shape 的 kernel，v2.10.0 会在该 fallback 路径设置 `enable_auto_blockify`，随后走普通 NPU dynamic autotuner。

线上问题定位时应先检查 generated `inductor_meta`，不能只根据外部环境变量判断某个 kernel 已进入 group 路径。
