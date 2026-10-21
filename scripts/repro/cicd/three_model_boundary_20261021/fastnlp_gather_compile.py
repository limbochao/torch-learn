"""Fixed failing fastNLP gather candidate, extracted with compile-debug skill.
Only ASTSource/GPUTarget/triton.compile are required; no model or autotuner.
"""
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

@triton.jit
def _standalone_triton_promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)

@triton.jit
def triton_red_fused_eq_index_masked_fill_max_ne_s_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, r0_numel, R0BLOCK_SUB: tl.constexpr
):
    base_r0 = tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    _tmp13 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    _tmp17 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    for loop_r0 in range(loops_r0):
        r0 = loop_r0 * R0BLOCK_SUB + base_r0
        r0_mask = r0 < r0_numel
        tmp0 = tl.load(in_ptr0 + r0, r0_mask, other=0.0)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1], False, tl.int1)
        tmp4 = tmp2 == tmp3
        tmp5 = tl.full([R0BLOCK_SUB], 2869, tl.int32)
        tmp6 = tmp0 + tmp5
        tmp7 = tmp0 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp0)
        tl.device_assert((0 <= tmp8) & (tmp8 < 2869) | ~r0_mask, 'index out of bounds: 0 <= tmp8 < 2869')
        tmp10 = tl.load(in_ptr1 + tmp8, r0_mask)
        tmp11 = tl.where(tmp4, tmp1, tmp10)
        tmp12 = tl.reshape(tmp11, [R0BLOCK_SUB])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(tl.reshape(r0_mask, [R0BLOCK_SUB]), tmp14, _tmp13)
        tmp15 = tmp2.to(tl.int64)
        tmp16 = tl.reshape(tmp15, [R0BLOCK_SUB])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(tl.reshape(r0_mask, [R0BLOCK_SUB]), tmp18, _tmp17)
        tl.store(out_ptr0 + r0, tmp2, r0_mask)
        tl.store(out_ptr1 + r0, tmp11, r0_mask)
    tmp13 = _standalone_triton_promote_to_tensor(tl.sum(_tmp13, 0))
    tmp17 = _standalone_triton_promote_to_tensor(tl.sum(_tmp17, 0))
    tl.store(out_ptr2 + tl.arange(0, 1), tmp13, None)
    tl.store(out_ptr3 + tl.arange(0, 1), tmp17, None)

torch.npu.set_device(0)
signature = {'in_out_ptr0': '*i64',
 'in_ptr0': '*i64',
 'in_ptr1': '*i64',
 'out_ptr0': '*i1',
 'out_ptr1': '*i64',
 'out_ptr2': '*i64',
 'out_ptr3': '*i64',
 'r0_numel': 'i32'}
constants = {'R0BLOCK_SUB': 1}
attrs = {(0,): [['tt.divisibility', 16]],
 (1,): [['tt.divisibility', 16]],
 (2,): [['tt.divisibility', 16]],
 (3,): [['tt.divisibility', 16]],
 (4,): [['tt.divisibility', 16]],
 (5,): [['tt.divisibility', 16]],
 (6,): [['tt.divisibility', 16]]}
options = {'num_warps': 1,
 'num_stages': 2,
 'debug': False,
 'compile_mode': 'simd_simt_template',
 'multibuffer': True}
source = ASTSource(triton_red_fused_eq_index_masked_fill_max_ne_s_0,
                   signature=signature, constexprs=constants, attrs=attrs)
triton.compile(source, target=GPUTarget("npu", "Ascend950PR_9579", 32), options=options)
print("COMPILE_PASS")
