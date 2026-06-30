---
title: RMSNorm SIMD 多 reduction 轴 codegen 问题复盘
---

# RMSNorm SIMD 多 reduction 轴 codegen 问题复盘

## 问题现象

这个问题出在 `torch.compile` 的 codegen 阶段，不是 eager 算子本身的问题。A2 环境在这条路径上只能走 SIMD kernel，不能靠 SIMT fallback 绕开，所以一旦生成到错误 DSL，编译出来的 kernel 就会直接带着错误语义运行。

相关术语说明见：[Inductor Codegen 术语说明](../notes/compiler/inductor-codegen-terms.md)。
本次修复的逐块代码改动说明见：[RMSNorm SIMD 多 reduction 轴 codegen 修改详解](rms-norm-simd-multi-reduction-codegen-change-detail.md)。

## 复现

复现脚本放在仓库内：[rms_norm_simd_multi_reduction_repro.py](../../scripts/repro/rms_norm_simd_multi_reduction_repro.py)。

复现脚本里的核心算子如下：

```python
def rms_norm_weight_grad(grad_out_base, q, q_square_sum):
    grad_out = grad_out_base.permute(0, 2, 1, 3)
    inv_rms = torch.rsqrt(q_square_sum.unsqueeze(-1) / q.shape[-1] + 1e-6)
    grad_weight = (grad_out * q.float() * inv_rms).sum(dim=(0, 1, 2))
    return grad_weight.to(torch.bfloat16)
```

默认输入是 `batch=2, seq=4096, heads=64, head_dim=128`。`grad_out_base` 的物理形状是 `[B, H, S, D]`，进入函数后先 `permute(0, 2, 1, 3)`，逻辑形状变成 `[B, S, H, D]`，stride 是 `(33554432, 128, 524288, 1)`。这个非连续 stride 是关键，因为它会把生成的 load 地址变成类似：

```python
in_ptr0 + (x0 + 128*r2 + 524288*r1 + 33554432*r3)
```

其中 `x0` 是保留下来的 `head_dim`，`r3/r2/r1` 分别对应 `batch/seq/heads`。

脚本默认打印输入 shape、stride 和 compiled 输出；设置 `CHECK=1` 时，会再跑 eager 版本并做 `assert_close`。运行方式是：

```bash
CHECK=1 python scripts/repro/rms_norm_simd_multi_reduction_repro.py
```

从数学上看，这个 case 要对 `B/S/H` 三个维度求和，只保留 `D=128`：

```python
out[x0] = sum_{r3, r2, r1}(
    grad_out[r3, r2, r1, x0]
    * q[r3, r2, r1, x0]
    * rsqrt(q_square_sum[r3, r2, r1] / 128 + 1e-6)
)
```

## 错误 DSL

错误 DSL 如下，`_tmp13` 的累加范围、`tl.sum` 的位置和 `tl.store` 的位置都不对。

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

这里最早的根因是 `_tmp13` 被放进了 `loop_r2`，所以每个 `r2` tile 都会把累加结果清零；同时 `tmp11` 被直接 flatten 成 `[X0BLOCK_SUB, ...]`，但它真实布局并不是这个顺序；最后 `tl.sum` 和 `tl.store` 都太早了，只做了局部 tile 的 reduction。

## 正确 DSL 和修复方向

正确的目标不是“让某一行不报错”，而是让 reduction 的布局和 store 位置同时正确：

- `_tmp13` 必须在进入 `r3/r2/r1` reduction loop 前初始化，并覆盖完整 reduction loop。
- flatten 前要把保留轴 `x0` 排到前面，把所有 reduction 轴放到后面。
- `tl.sum` 必须沿 reduction 维求和，而不是沿错误的 flatten 维度求和。
- `tl.store` 必须在完整 reduction 结束后执行，不能在 `r2` 或 `r1` 的 partial tile 内提前写回。

修复后的核心结构如下。这里不再展示完整 import 和 meta，只保留和语义相关的函数体：

```python
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
                    _tmp13 = tl.where((r1_mask & r2_mask & r3_mask & x0_mask).reshape([X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]), tmp14, _tmp13)
        tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
        tmp15 = tmp13.to(tl.float32)
        tl.store(out_ptr1 + (x0 + tl.arange(0,1) ), tmp15, x0_mask)
```

这几块改动分别对应下面的作用：

1. `is_first_reduction_tiling` / `use_outer_reduction_post_loop`：识别第一个 reduction tiling 轴，把 reduction 累加 buffer 的初始化和后续合并放到正确的作用域里，避免 `_tmp13` 在中间 tile 里被重新清零。
2. `value.permute([3, 0, 1, 2])` 再 `reshape`：把真实布局从 `[r3, r1, r2, x0]` 调整成 `[x0, r3, r1, r2]`，这样 flatten 以后 `x0` 仍然是保留轴，不会被卷进 reduction 维。
3. `ReductionAnalysis.analyze_reduction_dim()`：把 multi-reduction 的 `tl.sum` 维度改成正确的 reduction 维，保证最后求和沿的是 reduction lane，而不是保留轴。
4. `TileGenerator` 对 SIMD / SIMT template 多 reduction 轴的候选 config 做收窄：如果存在 reduction 轴整除 sub-block 的配置，就只保留这类配置，避免尾块 mask 和 flatten 后的 reduction lane 对不齐。

第 4 点可以用一个小例子理解。假设只看一个 reduction 轴 `r2`，真实长度是 `r2_numel=10`，但某个候选 config 选了 `R2BLOCK_SUB=6`。那么 `r2` 会被拆成两个 tile：

```text
loop_r2 = 0: r2 = [0, 1, 2, 3, 4, 5], mask = [T, T, T, T, T, T]
loop_r2 = 1: r2 = [6, 7, 8, 9, 10, 11], mask = [T, T, T, T, F, F]
```

进入 `_tmp13` 前，当前 tile 的贡献值会先被 flatten 到固定 lane：

```text
lane:         0   1   2   3   4   5
loop_r2 = 0: a0  a1  a2  a3  a4  a5
loop_r2 = 1: b6  b7  b8  b9  xx  xx
```

`tl.where(mask, tmp14, _tmp13)` 的意思是：mask 为真时写入新的累加值，mask 为假时保留 `_tmp13` 原来的值。所以第二个 tile 里 lane 4 和 lane 5 不会清零，而是保留上一个 tile 留下来的 `a4/a5`：

```text
处理 loop_r2 = 0 后: _tmp13 = [a0, a1, a2, a3, a4, a5]
处理 loop_r2 = 1 后: _tmp13 = [a0+b6, a1+b7, a2+b8, a3+b9, a4, a5]
最终 tl.sum 后:      a0+a1+a2+a3+a4+a5+b6+b7+b8+b9
```

如果只看单个 `r2` 轴，这个结果刚好还是对的；但当前 kernel 是 `r3/r2/r1` 三个 reduction 轴一起 flatten，`loop_r3/loop_r2/loop_r1` 会反复复用同一个 `[X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]` 累加平面。尾块里 mask 为假的 lane 保留的是“上一组 tile 在同一 flatten lane 上的旧值”，而不是当前 `r3/r2/r1` 组合应该有的 0。后续再执行 `tl.sum(_tmp13, 1)` 时，这些旧值会被当成当前 reduction 平面的有效值一起求和，表现为部分有效值漏加、部分旧值被错加。选择能整除 reduction 轴的 sub-block 后，每个 reduction tile 都没有这种尾部无效 lane，当前累加平面和最终 `tl.sum` 的 lane 语义能保持一致。

修复后，RMSNorm 对应的生成 DSL 已经回到正确结构。

术语说明见：[Inductor Codegen 术语说明](../notes/compiler/inductor-codegen-terms.md)。

## 影响范围

这次修复只针对 `numof_reduction_axis() > 1` 的 reduction codegen，重点是 contiguous multi-reduction 和其对应的 store 位置、flatten 顺序、reduction dim 选择。单 reduction 的路径没有改语义；非 contiguous multi-reduction 仍然走原有处理。

## 验证结果

在 A2 上验证过的命令包括：

```bash
CHECK=1 python3 scripts/repro/rms_norm_simd_multi_reduction_repro.py
```

```bash
python3 - <<'PY'
import torch, torch_npu
...
out = torch.compile(op_calc, backend="inductor", dynamic=False)(x)
...
PY
```

结果：

- RMSNorm weight grad：通过，`check=passed`
- `var_mean((0,2))`：通过，误差约 `3.6e-07 / 1.49e-08`
- `var_mean` 其它 dim：通过
- `batch_norm`：通过
