---
title: RMSNorm SIMD 多 reduction 轴 codegen 精度问题定位
---

# RMSNorm SIMD 多 reduction 轴 codegen 精度问题定位

## 背景

问题发生在 RMSNorm weight grad 的 `torch.compile` codegen 阶段。A2 环境只能走 SIMD kernel，不能通过 SIMT fallback 绕开问题；在触发 SIMD 路径后，kernel 可以生成并运行，但打开 `CHECK=1` 后，编译结果和 eager 结果不一致，输出的 128 个元素全部 mismatch。

DSL 指 Inductor 生成的类 Triton Python kernel，不是用户手写 eager 代码。它由 FX/IR、张量 stride 和 tiling 方案共同决定。阅读时主要看三类信息：`tl.load` 从哪些地址读数据，`for loop_r*` 覆盖哪些 reduction 维度，以及 `tl.reshape/tl.sum/tl.store` 在什么维度、什么循环层级上发生。

## 最小复现

复现脚本已放到本仓库内：[rms_norm_simd_multi_reduction_repro.py](../../scripts/repro/rms_norm_simd_multi_reduction_repro.py)。脚本只保留触发问题的 RMSNorm weight grad 片段：

```python
def rms_norm_weight_grad(grad_out_base, q, q_square_sum):
    grad_out = grad_out_base.permute(0, 2, 1, 3)
    inv_rms = torch.rsqrt(q_square_sum.unsqueeze(-1) / q.shape[-1] + 1e-6)
    grad_weight = (grad_out * q.float() * inv_rms).sum(dim=(0, 1, 2))
    return grad_weight.to(torch.bfloat16)
```

默认输入 shape 是 `batch=2, seq=4096, heads=64, head_dim=128`。`grad_out_base` 的物理 shape 是 `[B, H, S, D]`，dtype 为 `float32`；进入函数后执行 `permute(0, 2, 1, 3)`，逻辑 shape 变为 `[B, S, H, D]`，stride 变成 `(33554432, 128, 524288, 1)`。这个非连续 stride 是复现问题的关键，因为它对应 DSL 中 `in_ptr0 + (x0 + 128*r2 + 524288*r1 + 33554432*r3)` 这种读地址。

`q` 是 `[B, S, H, D]` 的 bfloat16 contiguous 输入，stride 为 `(33554432, 8192, 128, 1)`；`q_square_sum` 是 `[B, S, H]` 的 float32 输入，stride 为 `(262144, 64, 1)`。脚本默认打印输入 shape、stride 和编译输出；设置 `CHECK=1` 后，会再跑一遍 eager 版本，并用 `torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)` 校验。`DYNAMIC=1` 可以让 `torch.compile(..., dynamic=True)` 走动态 shape 编译，默认 `DYNAMIC=0`。

运行方式：

```bash
CHECK=1 python scripts/repro/rms_norm_simd_multi_reduction_repro.py
```

从数学语义看，这个 case 要对 `B/S/H` 三个维度求和，只保留 `D=128`：

```python
out[x0] = sum_{r3, r2, r1}(
    grad_out[r3, r2, r1, x0]
    * q[r3, r2, r1, x0]
    * rsqrt(q_square_sum[r3, r2, r1] / 128 + 1e-6)
)
```

后面的 DSL 中，`x0` 是保留下来的 `head_dim` 维；`r3` 对应 `batch`，范围是 2；`r2` 对应 `seq`，范围是 4096；`r1` 对应 `heads`，范围是 64。`X0BLOCK_SUB/R3BLOCK_SUB/R2BLOCK_SUB/R1BLOCK_SUB` 是每个 program 内的 tile 大小，`loops_*` 是把完整维度拆成多少个 tile 循环，`*_mask` 用来避免 tile 超出真实 shape 后读写越界。

## 错误 DSL

错误版本生成的完整 DSL 如下。为保证文档无个人信息，`backend_hash`、trace 目录、进程号等环境相关字段未写入；函数体和关键 meta 保持实际生成风格。

```python
@triton_heuristics.reduction(
    size_hints={'x0': 128, 'r3': 2, 'r2': 4096, 'r1': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r2_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4'), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_rsqrt_su_0', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'axis_names': ['x0', 'r3', 'r2', 'r1'], 'axis_static_values': (('x0', 128), ('r3', 2), ('r2', 4096), ('r1', 64)), 'low_dims': {0}, 'numof_reduction_axis': 3, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simd', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',)}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_rsqrt_su_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, x0_numel, r3_numel, r2_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
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

## 错误行为

这段 DSL 有三个问题，三个问题叠在一起导致精度完全不对。

第一个问题是 `_tmp13` 的生命周期错了。`_tmp13` 是 reduction 累加 buffer，负责保存当前 `x0` tile 上的部分和。这个 kernel 的目标是对 `r3/r2/r1` 全部求和，所以 `_tmp13` 应该在进入 `r3/r2/r1` reduction loop 之前初始化一次，然后在所有 reduction tile 上不断累加。错误 DSL 把 `_tmp13 = tl.full(...)` 放在 `loop_r2` 内部，导致每进入一个新的 `r2` tile 就清零一次，前面 `r2` tile 已经累加的结果会丢掉。

第二个问题是 `tmp11` flatten 前没有调整维度顺序。根据 DSL 中的下标形状：

```python
r3 = base_r3[:,None,None,None]
r1 = base_r1[None,:,None,None]
r2 = base_r2[None,None,:,None]
x0 = base_x0[None,None,None,:]
```

`tmp11` 的真实布局是 `[R3, R1, R2, X0]`。错误 DSL 直接执行 `tl.reshape(tmp11, [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB])`，相当于把第 0 维当成 `X0`，把后续维度当成 reduction 维。但第 0 维实际是 `R3`，最后一维才是 `X0`。后面的 `tl.sum(_tmp13, 1)` 实际在错误的二维视图上求和，把不同 `x0` 的数据混在一起。

第三个问题是 `tl.sum` 和 `tl.store` 的位置错了。它们位于 `loop_r2` 内部，`r2` 尚未遍历完就已经把 partial result 写到 `out_ptr1[x0]`。同一个输出地址会被多个 `r2` tile 反复写入，保留下来的不是完整 `sum(r3, r2, r1)`，而是某个不完整 reduction 的结果。

## 理论正确 DSL

理论上正确的 DSL 需要满足三件事：累加 buffer 覆盖完整 reduction loop；把真实布局 `[R3, R1, R2, X0]` 调整成 `[X0, R3, R1, R2]` 后再 flatten；最终 `tl.sum/tl.store` 必须等所有 `r3/r2/r1` loop 结束后再执行。完整形态如下：

```python
@triton_heuristics.reduction(
    size_hints={'x0': 128, 'r3': 2, 'r2': 4096, 'r1': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r2_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4'), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_rsqrt_su_0', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'axis_names': ['x0', 'r3', 'r2', 'r1'], 'axis_static_values': (('x0', 128), ('r3', 2), ('r2', 4096), ('r1', 64)), 'low_dims': {0}, 'numof_reduction_axis': 3, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simd', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',)}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_rsqrt_su_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, x0_numel, r3_numel, r2_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
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
                    _tmp13 = tl.where((r1_mask & r2_mask & r3_mask & x0_mask).reshape([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]), tmp14, _tmp13)
        tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
        tmp15 = tmp13.to(tl.float32)
        tl.store(out_ptr1 + (x0 + tl.arange(0,1) ), tmp15, x0_mask)
```

## 修复思路和影响

修复目标是让 codegen 生成上面这种结构。第一处修改在 `NPUIndexTritonKernel.codegen_body`：对 `numof_reduction_axis() > 1` 的 kernel 识别第一个 reduction tiling 轴。

```python
is_first_reduction_tiling = (
    self.numof_reduction_axis() > 1
    and range_val.is_tiling_axis
    and range_val.prefix == "r"
    and not any(ax.prefix == "r" for ax in self.sorted_axis[:index])
)
```

识别出第一个 reduction tiling 轴后，`prefix` 对应 `_tmp13 = tl.full(...)` 这样的初始化代码，放在第一个 reduction loop 外侧；`post_loop_combine/post_loop_store/stores` 放在第一个 reduction loop 退出后。`tl.sum/tl.store` 不再插到内层 `r2/r1` loop 中。

第二处修改在 `NPUIndexTritonKernel.reduction`。对于 contiguous multi-reduction，flatten 前先根据 `golden_var_list` 推出当前 value 的维度顺序，再把非 reduction 轴放前面、reduction 轴放后面。

```python
value_order = list(reversed(self.golden_var_list))
target_order = [x for x in value_order if x.name[0] != "r"] + [
    x for x in value_order if x.name[0] == "r"
]
permute_order = [value_order.index(x) for x in target_order]
tl.reshape(v.permute(permute_order), dense_size_str)
```

这个 case 中 `value_order` 对应 `[r3, r1, r2, x0]`，目标顺序是 `[x0, r3, r1, r2]`，所以实际生成 `permute([3, 0, 1, 2])`。

第三处修改在 `ReductionAnalysis.analyze_reduction_dim`。当 contiguous multi-reduction 被调整成 `[X0, R...]` 后，`tl.sum` 应该沿 dim 1 求和；所以 reduced dim 不再取第一个 `r` 在原始布局中的位置，而是取非 reduction 轴数量。这个 case 只有一个非 reduction 轴 `x0`，因此 dim 是 1。

还有一个和 tiling 稳定性相关的补充修改在 `TileGenerator.descend_split_tiling`。对于 `dual_reduction and npu_kernel_type == SIMD`，如果存在 reduction 轴能被 sub-block 整除的配置，就只保留这些配置。这样可以避免 flatten 后 reduction lane 和尾块 mask 组合出不稳定的配置；如果没有整除配置，则不强行丢弃所有 config。

修改影响主要限定在 `numof_reduction_axis() > 1 and is_contiguous_reduction()` 的 SIMD 多 reduction codegen。单 reduction 轴仍走原有 reshape/reduction_dim 逻辑；非 contiguous 多 reduction 仍保持原来的 `reduced_dim=0` 处理；tiling config 过滤也只在 SIMD dual reduction 且存在整除配置时收窄候选配置。潜在风险是其它 contiguous multi-reduction kernel 的 DSL 形态会随之变化，但这种变化是把错误的“内层 partial store”和错误的 flatten 顺序改成符合 reduction 数学语义的 `[x, r...]` 求和，因此不属于扩大功能面。

## 实际修复后的 DSL

A2 上 `CHECK=1` 已验证通过，生成的 DSL 和理论正确形态一致。完整 DSL 如下：

```python
@triton_heuristics.reduction(
    size_hints={'x0': 128, 'r3': 2, 'r2': 4096, 'r1': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r2_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B4'), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_rsqrt_su_0', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'axis_names': ['x0', 'r3', 'r2', 'r1'], 'axis_static_values': (('x0', 128), ('r3', 2), ('r2', 4096), ('r1', 64)), 'low_dims': {0}, 'numof_reduction_axis': 3, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simd', 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',)}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_rsqrt_su_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, x0_numel, r3_numel, r2_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
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
                    _tmp13 = tl.where((r1_mask & r2_mask & r3_mask & x0_mask).reshape([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]), tmp14, _tmp13)
        tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
        tmp15 = tmp13.to(tl.float32)
        tl.store(out_ptr1 + (x0 + tl.arange(0,1) ), tmp15, x0_mask)
```

正确 DSL 可以按“读、乘、累加、写”来读。每个 program 负责一段 `x0`，也就是一段 `head_dim`；在这个 `x0` 范围内，kernel 遍历全部 `r3/r2/r1` tile。`tmp0/tmp1/tmp5` 分别读 `grad_out/q/q_square_sum`，`tmp11` 计算单个 reduction lane 上的贡献值。`tmp11.permute([3, 0, 1, 2])` 把 `x0` 放到第 0 维，后面三个维度全部是 reduction lane；reshape 后的 `_tmp13` 是 `[当前 x0 tile, 当前所有 reduction lane]` 的二维累加视图。所有 reduction loop 完成后，`tl.sum(_tmp13, 1)` 对 dim 1 求和，得到每个 `x0` 的完整 weight grad，最后只 store 一次。

最终验证输出包含 `check=passed`，说明编译结果和 eager 在当前容差下对齐。
