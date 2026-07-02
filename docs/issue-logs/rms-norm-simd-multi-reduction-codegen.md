---
title: RMSNorm SIMD 多 reduction 轴 codegen 问题复盘
---

# RMSNorm SIMD 多 reduction 轴 codegen 问题复盘

## 问题现象

RMSNorm weight grad 在 `torch.compile` 后走 NPU Inductor codegen 路径，生成的 SIMD reduction kernel 结果和 eager 不一致。问题不在 eager 算子语义，而在 generated DSL 对连续多 reduction 轴的处理：accumulator 的生命周期、value/mask flatten 前的轴顺序、最终 `tl.sum` 的 reduction dim、以及 `tl.store` 的位置没有按同一套 layout 规则生成。

A2 这类环境无法依赖 SIMT fallback 绕过 SIMD kernel，因此错误 DSL 会直接编译并运行，表现为 compiled output 精度错误。相关术语见：[Inductor Codegen 术语说明](../notes/compiler/inductor-codegen-terms.md)。

## 复现方式

复现脚本在仓库内：[rms_norm_simd_multi_reduction_repro.py](../../scripts/repro/rms_norm_simd_multi_reduction_repro.py)。

核心 eager 表达式如下：

```python
def rms_norm_weight_grad(grad_out_base, q, q_square_sum):
    grad_out = grad_out_base.permute(0, 2, 1, 3)
    inv_rms = torch.rsqrt(q_square_sum.unsqueeze(-1) / q.shape[-1] + 1e-6)
    grad_weight = (grad_out * q.float() * inv_rms).sum(dim=(0, 1, 2))
    return grad_weight.to(torch.bfloat16)
```

默认输入是 `batch=2, seq=4096, heads=64, head_dim=128`。`grad_out_base` 的物理 shape 是 `[B, H, S, D]`，进入函数后 `permute(0, 2, 1, 3)`，逻辑 shape 变成 `[B, S, H, D]`，stride 是 `(33554432, 128, 524288, 1)`。

这组 stride 会让 generated DSL 中 `grad_out` 的 load 地址变成：

```python
in_ptr0 + (x0 + 128*r2 + 524288*r1 + 33554432*r3)
```

其中 `x0` 是保留轴，对应 `head_dim`；`r3/r2/r1` 是 reduction 轴，分别对应 `batch/seq/heads`。数学语义是：

```python
out[x0] = sum_{r3, r2, r1}(
    grad_out[r3, r2, r1, x0]
    * q[r3, r2, r1, x0]
    * rsqrt(q_square_sum[r3, r2, r1] / 128 + 1e-6)
)
```

运行方式：

```bash
CHECK=1 python scripts/repro/rms_norm_simd_multi_reduction_repro.py
```

`CHECK=1` 会同时执行 eager 版本并用 `torch.testing.assert_close` 校验 compiled output。

## 错误 DSL

错误版本生成的 DSL 如下。为避免写入环境相关信息，`backend_hash`、trace 目录等 metadata 已省略，函数体和关键 meta 保留实际生成风格。

```python
@triton_heuristics.reduction(
    size_hints={'x0': 128, 'r3': 2, 'r2': 4096, 'r1': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r2_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4'), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_rsqrt_sum_0', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'axis_names': ['x0', 'r3', 'r2', 'r1'], 'axis_static_values': (('x0', 128), ('r3', 2), ('r2', 4096), ('r1', 64)), 'low_dims': {0}, 'numof_reduction_axis': 3, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simd', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',)}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_rsqrt_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, x0_numel, r3_numel, r2_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,None,None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        for loop_r3 in range(loops_r3):
            r3 = (loop_r3 * R3BLOCK_SUB) + base_r3[:,None,None,None]
            r3_mask = r3 < r3_numel
            for loop_r2 in range(loops_r2):
                r2_1 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None,None]
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,None,:,None]
                r2_mask = r2 < r2_numel
                r2_1_mask = r2_1 < r2_numel
                _tmp13 = tl.full([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB], 0, tl.float32)
                for loop_r1 in range(loops_r1):
                    r1_2 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,None,:,None]
                    r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:,None,None]
                    r1_mask = r1 < r1_numel
                    r1_2_mask = r1_2 < r1_numel
                    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 524288*r1 + 33554432*r3), r3_mask & r2_mask & x0_mask & r1_mask, other=0.0)
                    tmp1 = tl.load(in_ptr1 + (x0 + 128*r1_2 + 8192*r2_1 + 33554432*r3), r1_2_mask & r3_mask & r2_1_mask & x0_mask, other=0.0).to(tl.float32)
                    tmp2 = tmp1.permute([0, 2, 1, 3])
                    tmp5 = tl.load(in_ptr2 + (r1 + 64*r2 + 262144*r3), r3_mask & r2_mask & r1_mask, other=0.0)
                    tmp3 = tmp2.to(tl.float32)
                    tmp4 = tmp0 * tmp3
                    tmp6 = 0.0078125
                    tmp7 = tmp5 * tmp6
                    tmp8 = 1e-06
                    tmp9 = tmp7 + tmp8
                    tmp10 = tl.rsqrt(tmp9)
                    tmp11 = tmp4 * tmp10
                    tmp12 = tl.reshape(tmp11, [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB])
                    tmp14 = _tmp13 + tmp12
                    _tmp13 = tl.where((r1_mask & r2_mask & r3_mask & x0_mask).reshape([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]), tmp14, _tmp13)
                tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
                tmp15 = tmp13.to(tl.float32)
                tl.store(out_ptr1 + (x0 + tl.arange(0,1) ), tmp15, x0_mask)
```

这段 DSL 有三个关键问题。

`_tmp13` 初始化在 `loop_r2` 内部。它只能覆盖当前 `r2` tile 及其内部的 `r1` loop，无法覆盖完整 `r3/r2/r1` reduction 范围；每次进入新的 `r2` tile 都会重新清零 accumulator。

`tmp11` 直接 reshape 成 `[X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]`。但 `tmp11` 当时的真实轴顺序不是 `[x0, r3, r2, r1]`，直接 flatten 会把保留轴和 reduction 轴按错误顺序解释。

`tl.sum` 和 `tl.store` 仍在 `loop_r2` 内部。最终写回发生在 partial reduction 之后，而不是完整遍历 `r3/r2/r1` 之后。

## 正确 DSL 结构

正确 DSL 需要满足同一套 layout 语义：保留轴 `x0` 在前，所有 reduction 轴 flatten 到后面；accumulator 覆盖完整 reduction loop；`tl.sum` 和 `tl.store` 等到完整 reduction 结束后执行。

修复后的核心 DSL 结构如下：

```python
@triton_heuristics.reduction(
    size_hints={'x0': 128, 'r3': 2, 'r2': 4096, 'r1': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r2_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4'), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_rsqrt_sum_0', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'axis_names': ['x0', 'r3', 'r2', 'r1'], 'axis_static_values': (('x0', 128), ('r3', 2), ('r2', 4096), ('r1', 64)), 'low_dims': {0}, 'numof_reduction_axis': 3, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simd', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',)}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_rsqrt_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, x0_numel, r3_numel, r2_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,None,None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp13 = tl.full([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB], 0, tl.float32)
        for loop_r3 in range(loops_r3):
            r3 = (loop_r3 * R3BLOCK_SUB) + base_r3[:,None,None,None]
            r3_mask = r3 < r3_numel
            for loop_r2 in range(loops_r2):
                r2_1 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None,None]
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,None,:,None]
                r2_mask = r2 < r2_numel
                r2_1_mask = r2_1 < r2_numel
                for loop_r1 in range(loops_r1):
                    r1_2 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,None,:,None]
                    r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:,None,None]
                    r1_mask = r1 < r1_numel
                    r1_2_mask = r1_2 < r1_numel
                    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 524288*r1 + 33554432*r3), r3_mask & r2_mask & x0_mask & r1_mask, other=0.0)
                    tmp1 = tl.load(in_ptr1 + (x0 + 128*r1_2 + 8192*r2_1 + 33554432*r3), r1_2_mask & r3_mask & r2_1_mask & x0_mask, other=0.0).to(tl.float32)
                    tmp2 = tmp1.permute([0, 2, 1, 3])
                    tmp5 = tl.load(in_ptr2 + (r1 + 64*r2 + 262144*r3), r3_mask & r2_mask & r1_mask, other=0.0)
                    tmp3 = tmp2.to(tl.float32)
                    tmp4 = tmp0 * tmp3
                    tmp6 = 0.0078125
                    tmp7 = tmp5 * tmp6
                    tmp8 = 1e-06
                    tmp9 = tmp7 + tmp8
                    tmp10 = tl.rsqrt(tmp9)
                    tmp11 = tmp4 * tmp10
                    tmp12 = tl.reshape(tmp11.permute([3, 0, 1, 2]), [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB])
                    tmp14 = _tmp13 + tmp12
                    _tmp13 = tl.where((r1_mask & r2_mask & r3_mask & x0_mask).permute([3, 0, 1, 2]).reshape([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]), tmp14, _tmp13)
        tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
        tmp15 = tmp13.to(tl.float32)
        tl.store(out_ptr1 + (x0 + tl.arange(0,1) ), tmp15, x0_mask)
```

这份结构里，`_tmp13` 在 `loop_r3/r2/r1` 之前初始化，完整 reduction loop 只更新同一个 accumulator。`tmp11` 和 mask 在 reshape 前使用同一个 `permute([3, 0, 1, 2])`，保证 flatten 后的 value lane 和 mask lane 指向同一个逻辑坐标。`tl.sum(_tmp13, 1)` 和 `tl.store` 在完整 reduction loop 外执行，只写一次完整结果。

## 修复思路

修复目标是让 codegen 按通用 reduction 规则生成 DSL，而不是针对 RMSNorm 的 shape 或 kernel name 做 special case。

store 位置由 `write_pointwise(allow_stores=None)` 控制。单 reduction 轴及以下保持原有行为，多 reduction 轴默认延后 store；只有后续 fused node 已经不依赖 reduction 轴时，才显式传入 `allow_stores=True`。

loop emission 侧增加 `is_first_reduction_tiling` 和 `use_outer_reduction_post_loop`。前者识别完整 reduction 轴组的入口，后者判断是否已有 reduction accumulator 需要在外层做 post-loop combine/store。这样 prefix 初始化、`tl.sum`、`tl.store` 都能落到完整 reduction loop 的正确作用域。

value 和 mask flatten 前共用同一套 axis order。contiguous multi-reduction 场景下，先从 `golden_var_list` 得到当前 value order，再构造目标 order：非 reduction 轴在前，reduction 轴在后。只有 `permute_order` 不是 identity 时才生成 `permute`，避免在顺序已经正确的 case 里引入不必要的 DSL。

reduction 结果 reshape 时，连续多 reduction 轴已经被 reduce 掉，所有 `r` 轴都应恢复成广播维度 `1`。因此 `reduction_resize()` 会把 dense shape 中的 reduction 轴位置改成 `1`。

reduction dim 由 `ReductionAnalysis.analyze_reduction_dim()` 重新计算。连续多 reduction 轴 flatten 后的结构是：

```text
[non-reduction axes, flattened reduction axes]
```

所以 `tl.sum` 的 dim 等于非 reduction 轴数量。当前 RMSNorm 只有一个保留轴 `x0`，对应 dim 为 `1`；如果以后是 `[x0, x1, flat_r]`，dim 就是 `2`。

## 影响范围

修复范围集中在 `numof_reduction_axis() > 1` 的 reduction codegen，重点是 contiguous multi-reduction 下的 store 作用域、value/mask flatten 顺序、reduction resize 和 reduction dim。单 reduction 路径不改变语义；非 contiguous multi-reduction 仍按原路径处理。

SIMD 和 SIMT template 只影响后端 kernel 选择及执行模板，不改变上层 DSL 必须满足的 tensor semantics。这个问题的根因是 DSL 生成规则不一致，不是通过 fallback 或过滤 tiling config 规避。

## 验证结果

RMSNorm repro 在目标环境运行：

```bash
CHECK=1 python scripts/repro/rms_norm_simd_multi_reduction_repro.py
```

期望输出包含：

```text
check=passed
```

同时补充覆盖了 multi-reduction 相关的 `var_mean` 场景，确认后续 fused node 和多 reduction 轴路径没有出现提前 store 或 reduction dim 选择错误。
