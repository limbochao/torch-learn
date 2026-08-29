"""
SIMT template fused-reduction precision repro.

Run:

    python scripts/repro/simt_template_fused_reduction_precision_repro.py
"""

import torch
import torch_npu
from torch import empty_strided
from torch._dynamo.testing import rand_strided
from torch._inductor.async_compile import AsyncCompile
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream

torch_npu.npu._initialized = torch_npu.npu.is_initialized()
async_compile = AsyncCompile()


def forward(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    expand = torch.ops.aten.expand.default(arg1_1, [100, 520]); arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, expand); arg0_1 = expand = None
    relu = torch.ops.aten.relu.default(add); add = None
    view = torch.ops.aten.reshape.default(relu, [100, -1, 10]); relu = None
    view_1 = torch.ops.aten.reshape.default(arg2_1, [100, 10, 52]); arg2_1 = None
    expand_1 = torch.ops.aten.expand.default(arg3_1, [100, 10, 52]); arg3_1 = None
    add_1 = torch.ops.aten.add.Tensor(view_1, expand_1); view_1 = expand_1 = None
    exp = torch.ops.aten.exp.default(add_1); add_1 = None
    permute = torch.ops.aten.permute.default(exp, [0, 2, 1]); exp = None
    view_2 = torch.ops.aten.reshape.default(arg4_1, [100, 10, 52]); arg4_1 = None
    expand_2 = torch.ops.aten.expand.default(arg5_1, [100, 10, 52]); arg5_1 = None
    add_2 = torch.ops.aten.add.Tensor(view_2, expand_2); view_2 = expand_2 = None
    permute_1 = torch.ops.aten.permute.default(add_2, [0, 2, 1]); add_2 = None
    cos = torch.ops.aten.cos.default(permute_1); permute_1 = None
    mul = torch.ops.aten.mul.Tensor(permute, cos); permute = cos = None
    add_3 = torch.ops.aten.add.Tensor(view, mul); view = mul = None
    sum_1 = torch.ops.aten.sum.dim_IntList(add_3, [2], True); add_3 = None
    return sum_1


triton_per_fused_add_cos_mul_native_layer_norm_27 = async_compile.triton(
    'triton_per_fused_add_cos_mul_native_layer_norm_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.persistent_reduction(
    size_hints={'y0': 100, 'x1': 52, 'r2': 10},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cos_mul_native_layer_norm_27', 'mutated_arg_names': [], 'backend_hash': 'B0FB9F7305F57BE665DC57954C547C6FB198CECD703C4A921D02D58CE26FF172', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 100), ('x1', 52), ('r2', 10)), 'low_dims': {1, 2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'cg5fubx4aqepcceiycgqwm3p3q6pk36zustghpnnxyb6f36bez3o', 'traced_graph_dir': '/tmp/torchinductor_root/traced_fx_graph_cache', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'group_enabled': False, 'group_template': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_cos_mul_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, Y0BLOCK_SUB: tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 10
    x1_numel = 52
    X1BLOCK_SUB: tl.constexpr = 52
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    base_r2 = tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        x1 = base_x1[None, :, None]
        r2 = base_r2[None, None, :]
        r2_mask = r2 < r2_numel
        tmp0 = tl.load(in_ptr0 + (r2 + 10*x1 + 520*y0), y0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + 10*x1), None)
        tmp5 = tl.load(in_ptr2 + (x1 + 52*r2 + 520*y0), y0_mask, other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + 52*r2), None)
        tmp9 = tl.load(in_ptr4 + (x1 + 52*r2 + 520*y0), y0_mask, other=0.0)
        tmp10 = tl.load(in_ptr5 + (x1 + 52*r2), None)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1, 1], 0, tl.int32)
        tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
        tmp7 = tmp5 + tmp6
        tmp8 = tl_math.exp(tmp7)
        tmp11 = tmp9 + tmp10
        tmp12 = tl_math.cos(tmp11)
        tmp13 = tmp8 * tmp12
        tmp14 = tmp4 + tmp13
        tmp16 = tl.sum(tmp14, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x1 + 52*y0), tmp16, y0_mask)
''', device_str='npu')

async_compile.wait(globals())
del async_compile


def main():
    torch.manual_seed(0)
    buf45 = rand_strided((100, 520), (520, 1), device='npu', dtype=torch.float32)
    arg33_1 = rand_strided((520,), (1,), device='npu', dtype=torch.float32)
    buf48 = rand_strided((1000, 52), (52, 1), device='npu', dtype=torch.float32)
    arg30_1 = rand_strided((1, 10, 52), (520, 52, 1), device='npu', dtype=torch.float32)
    buf52 = rand_strided((1000, 52), (52, 1), device='npu', dtype=torch.float32)
    arg31_1 = rand_strided((1, 10, 52), (520, 52, 1), device='npu', dtype=torch.float32)
    buf53 = empty_strided((100, 52, 1), (52, 1, 5200), device='npu', dtype=torch.float32)

    stream0 = get_raw_stream(0)
    triton_per_fused_add_cos_mul_native_layer_norm_27.run(buf45, arg33_1, buf48, arg30_1, buf52, arg31_1, buf53, 100, 52, 10, stream=stream0)
    torch_npu.npu.synchronize()
    res = forward(buf45, arg33_1, buf48, arg30_1, buf52, arg31_1)
    torch_npu.npu.synchronize()

    diff = (res - buf53).abs()
    print(f'max_abs_error={diff.max().item()}')
    print(f'max_rel_error={(diff / res.abs().clamp_min(1e-12)).max().item()}')
    torch.testing.assert_close(res, buf53)


if __name__ == '__main__':
    main()
