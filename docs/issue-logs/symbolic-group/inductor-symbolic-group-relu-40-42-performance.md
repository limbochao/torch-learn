---
title: relu_40 / relu_42 Static 与 Group 性能差异记录
---

# relu_40 / relu_42 Static 与 Group 性能差异记录

> **证据边界：本文只记录当前产物和测试现象。分析产物来自非线上代码，未基于 `v2.10.0` 线上实现完成复现，因此不下 root cause 结论。**

## 1. 现象

测试符号 `s0` 的运行值为 `4、8、16`。同批 structurally similar 的 ReLU pointwise kernels 中，
只有 group 产物 `triton_poi_fused_relu_42` 相对对应 static kernel 出现超过 20% 的性能劣化，
其余 kernel 未出现同量级回退。

kernel 名称后缀不用于直接配对。当前根据 graph fragment、固定轴长度和 load/store indexing，将 static `relu_40` 与 group `relu_42` 视为同一计算片段的候选对照。

## 2. 已知 kernel 信息

| 项目 | Static | Dynamic group |
| --- | --- | --- |
| kernel | `triton_poi_fused_relu_40` | `triton_poi_fused_relu_42` |
| 主要二维迭代域 | `y0=8, x1=1596` | `y0=s0, x1=1596` |
| group metadata | `group_enabled=False` | `group_enabled=True` |
| kernel type | `simt_template` | `simt_template` |
| split / tiling axis | `[0, 1] / [0, 1]` | `[0, 1] / [0, 1]` |
| low dims | `{1}` | `{1}` |
| primary / static axis | 不适用 | `primary=y0`, `static=(x1,)` |
| group feature | 不适用 | `elementwise_numel=outer_product(y0,x1)` |
| bucket | 不适用 | `(229376,)` |
| 测试运行值 | 固定 shape 分别编译 | `s0=4, 8, 16` 复用 group kernel |
| DSL 主体 | ReLU load/compare/store | 与 static 主体结构近似 |

两侧 `size_hints` 都记录 `y0=8, x1=1596`；group 侧的 `y0=8` 是首次编译 hint，
不代表后续 runtime 仍为 8。两边主体均从 `x1 + 1596*y0` 读取，执行 ReLU 后写回同一地址。

当前材料表明两侧的 elementwise 数学表达式、kernel type 和主要 indexing 近似，差异集中在符号轴、
group metadata、代表 shape 下生成的 compile variant，以及运行时的 launch config。

## 3. 原始产物范围

本次对比来自用户提供的两份 generated kernel 片段：一份 static 产物，一份 dynamic group 产物。
原始片段没有保存在仓库中，因此下面直接保留重新定位所需的信息。

为避免环境信息干扰，以下内容只移除了：

- `/tmp/torchinductor_root/...` kernel path。
- 完整 `backend_hash`。
- `TRACED_GRAPH_HASH` 和 `TRACED_GRAPH_DIR` 占位字段。
- 两侧完全相同、且不参与本次判断的 cache/debug flags。

Graph fragment、shape/stride、scheduler node、Triton signature、影响本次判断的 codegen metadata 和
完整 kernel body 均保留。

### 3.1 同批 ReLU kernel 映射

同批产物共有六组相同 graph 结构、不同 `x1` 长度的 ReLU kernel。配对依据是 graph fragment 和
`x1`，不是 kernel 后缀：

| Graph tensor | `x1` | Static kernel | Group kernel |
| --- | ---: | --- | --- |
| `mm_49 -> relu_14` | 12288 | `relu_36` | `relu_38` |
| `mm_53 -> relu_18` | 6144 | `relu_37` | `relu_39` |
| `mm_57 -> relu_22` | 3072 | `relu_38` | `relu_40` |
| `mm_61 -> relu_26` | 1536 | `relu_39` | `relu_41` |
| `mm_63 -> relu_28` | 1596 | `relu_40` | `relu_42` |
| `mm_65 -> relu_30` | 768 | `relu_41` | `relu_43` |

只有 `x1=1596` 的 group kernel 在当前测试中出现超过 20% 的性能劣化。其他五组的存在很重要：
它们说明“使用符号 `y0`”本身不足以解释为什么只有 `relu_42` 明显退化。

## 4. Static `triton_poi_fused_relu_40`

### 4.1 Graph fragment

```text
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_30 => relu_28
# Graph fragment:
#   %mm_63 : Tensor "f32[8, 1596][1596, 1]npu:0" = PlaceHolder[target=mm_63]
#   %relu_28 : Tensor "f32[8, 1596][1596, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm_63,), kwargs = {})
#   return %relu_28
# SchedulerNodes: [SchedulerNode(name='op289')]
```

### 4.2 Generated kernel

以下片段省略公共 import、`async_compile.triton(...)` 外层和去敏字段，其余内容保持
`output_code.py` 的原始排版。

```python
@triton_heuristics.pointwise(
    size_hints={'y0': 8, 'x1': 1596}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_poi_fused_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 8), ('x1', 1596)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_40(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_out_ptr0 + (x1 + 1596*y0), x1_mask & y0_mask)
            tmp1 = tl.full([1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x1 + 1596*y0), tmp2, x1_mask & y0_mask)
```

## 5. Group `triton_poi_fused_relu_42`

### 5.1 Graph fragment

```text
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_30 => relu_28
# Graph fragment:
#   %mm_63 : Tensor "f32[s0, 1596][1596, 1]npu:0" = PlaceHolder[target=mm_63]
#   %relu_28 : Tensor "f32[s0, 1596][1596, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm_63,), kwargs = {})
#   return %relu_28
# SchedulerNodes: [SchedulerNode(name='op317')]
```

### 5.2 Generated kernel

以下片段使用与 static 相同的省略规则，并保持 `output_code.py` 的原始排版。

```python
@triton_heuristics.pointwise(
    size_hints={'y0': 8, 'x1': 1596}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_poi_fused_relu_42', 'mutated_arg_names': ['in_out_ptr0'], 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('x1', 1596),), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'group_enabled': True, 'group_template': 'pointwise', 'group_workload': 'elementwise', 'primary_group_axis': 'y0', 'static_split_axes': ('x1',), 'secondary_runtime_symbolic_axes': (), 'group_features': ({'name': 'elementwise_numel', 'source': 'outer_product', 'axis_names': ('y0', 'x1'), 'buckets': (229376,)},), 'ordered_arg_specs': ({'kind': 'tensor', 'source': 'buffer', 'name': 'in_out_ptr0', 'dtype': 'torch.float32', 'device': 'npu:0', 'size_exprs': ({'axis_name': 'y0'}, {'axis_name': 'x1'}), 'stride_exprs': ({'axis_name': 'x1'}, {'const': 1})}, {'kind': 'size', 'source': 'axis_expr', 'name': 'y0_numel', 'expr': {'axis_name': 'y0'}}, {'kind': 'size', 'source': 'axis_expr', 'name': 'x1_numel', 'expr': {'axis_name': 'x1'}}), 'workspace_arg_specs': (), 'extra_launcher_arg_specs': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_42(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_out_ptr0 + (x1 + 1596*y0), x1_mask & y0_mask)
            tmp1 = tl.full([1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x1 + 1596*y0), tmp2, x1_mask & y0_mask)
```

## 6. 两份实际 kernel 能确认什么

可以直接确认：

- 两侧 graph 都是同一个 `mm_63 -> aten.relu` 片段。
- shape 只在第一维从常量 `8` 变为符号 `s0`，stride 都是 `[1596, 1]`。
- 两侧 Triton signature、`split_axis`、`tiling_axis`、`low_dims` 和 `npu_kernel_type` 相同。
- 两侧 DSL body 按 `output_code.py` 原始排版逐语句一致，地址公式都是 `x1 + 1596*y0`。
- group 侧将 `y0` 作为 `primary_group_axis`，`x1` 作为 static split axis，并按
  `y0*x1` 的 `elementwise_numel` 进入 bucket `(229376,)`。

还可以确认 metadata schema 不完全相同：group 侧包含 `inductor_ascend_linear_mode="linear"` 和
benchmark arg specs；static 侧原始片段中还包含 `incremental_autotune`、`batch_invariant`、
`mix_order_reduction_allow_multi_stages`、`dynamic_disable_pipelining` 等字段，而 group 片段未展示这些字段。
这些字段差异本身不能证明两侧来自不同 revision，但说明后续不能用线上 `v2.10.0` 的 metadata schema
替当前产物补全缺失字段。

两份 generated kernel **不能**确认最终 tiling。附件中没有以下运行期结果：

- group representative 的实际 axis values。
- static/group 最终 `selected_config`。
- group runtime `feature_inputs` 和 `group_id`。
- 最终 `runtime_blocks` 和实际 launch grid。

因此后续定位不能从 `size_hints={"y0": 8, "x1": 1596}` 反推出最终 winner，也不能把
bucket 上界 `229376` 直接当成实际 representative axis values。

## 7. 尚不能确认的解释

以下只是后续排查方向，不是本文结论：

- group representative 与实际 `s0` 差异导致 selected config 不适合 `x1=1596` 的尾块结构。
- runtime blocks 或 grid policy 改变并行度，造成 core 利用率下降。
- group variant 的 `X/YBLOCK_SUB`、`num_warps`、`num_stages` 或 compile mode 与 static 不同。
- 两个看似相同的 DSL 实际落到了不同 `npu_kernel_type` 或 compiler optimization path。

已有分析没有在干净的 `v2.10.0` 线上代码上逐项验证上述变量，因此不能把其中任何一项写成原因。

## 8. 后续所需证据

要形成根因结论，至少需要在同一线上版本和同一设备上收集：

1. static/group 完整 generated DSL 和 graph fragment。
2. group 的 `group_features`、runtime `feature_inputs`、`group_id` 和 representative axis values。
3. 两侧最终 `selected_config`、`runtime_blocks` 和实际 grid。
4. 单 kernel 多次 device timing，排除整网调度和 profile 噪声。
5. 固定 group tiling 后的对照结果，用于确认回退是否由 config 而非 DSL body 引起。

在这些证据补齐前，本文状态保持“现象记录”。
