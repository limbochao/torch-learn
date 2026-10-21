"""ConvNeXt add/permute candidate: fixed BiShengIR PlanMemory failure."""
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_7(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    ks0,
    ks1,
    t0_numel,
    z1_numel,
    y2_numel,
    x3_numel,
    T0BLOCK,
    Y2BLOCK,
    T0BLOCK_SUB: tl.constexpr,
    Z1BLOCK_SUB: tl.constexpr,
    Y2BLOCK_SUB: tl.constexpr,
    X3BLOCK_SUB: tl.constexpr,
):
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0 = tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    base_z1 = tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (z1_numel + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    y2_offset = tl.program_id(1) * Y2BLOCK
    base_y2 = tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (Y2BLOCK + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3 = tl.arange(0, X3BLOCK_SUB)
    loops_x3 = (x3_numel + X3BLOCK_SUB - 1) // X3BLOCK_SUB
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + loop_t0 * T0BLOCK_SUB + base_t0[:, None, None, None]
        t0_mask = t0 < min(T0BLOCK + t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = loop_z1 * Z1BLOCK_SUB + base_z1[None, :, None, None]
            z1_mask = z1 < z1_numel
            for loop_y2 in range(loops_y2):
                y2 = y2_offset + loop_y2 * Y2BLOCK_SUB + base_y2[None, None, :, None]
                y2_mask = y2 < min(Y2BLOCK + y2_offset, y2_numel)
                for loop_x3 in range(loops_x3):
                    x3 = loop_x3 * X3BLOCK_SUB + base_x3[None, None, None, :]
                    x3_mask = x3 < x3_numel
                    tmp0 = tl.load(in_ptr0 +
                        (x3 +
                        y2 * (1 * (1 >= ks0) +
                        ks0 * (ks0 > 1)) +
                        z1 * (1 * (1 >= ks0) +
                        ks0 * (ks0 > 1)) * (1 * (1 >= ks0) +
                        ks0 * (ks0 > 1)) +
                        80 * t0 * (1 * (1 >= ks0) +
                        ks0 * (ks0 > 1)) * (1 * (1 >= ks0) +
                        ks0 * (ks0 > 1))), t0_mask & x3_mask & y2_mask & z1_mask)
                    tmp1 = tl.load(in_ptr1 +
                        (z1 +
                        80 * x3 +
                        80 * ks0 * y2 +
                        80 * ks1 * t0), t0_mask & x3_mask & y2_mask & z1_mask)
                    tmp2 = tmp0 + tmp1
                    tl.store(out_ptr0 +
                        (x3 +
                        ks0 * y2 +
                        ks1 * z1 +
                        80 * ks1 * t0), tmp2, t0_mask & x3_mask & y2_mask & z1_mask)


torch.npu.set_device(0)
signature = {'in_ptr0': '*fp32',
 'in_ptr1': '*fp32',
 'out_ptr0': '*fp32',
 'ks0': 'i64',
 'ks1': 'i64',
 't0_numel': 'i32',
 'z1_numel': 'i32',
 'y2_numel': 'i32',
 'x3_numel': 'i32',
 'T0BLOCK': 'i32',
 'Y2BLOCK': 'i32'}
constants = {'T0BLOCK_SUB': 1, 'Z1BLOCK_SUB': 80, 'Y2BLOCK_SUB': 16, 'X3BLOCK_SUB': 6}
options = {'num_warps': 1,
 'num_stages': 2,
 'debug': False,
 'compile_mode': 'simd_simt_template',
 'multibuffer': True}
source = ASTSource(
    triton_poi_fused_add_native_layer_norm_permute_7, signature=signature, constexprs=constants, attrs={},
)
triton.compile(source, target=GPUTarget("npu", "Ascend950PR_9579", 32), options=options)
print("COMPILE_PASS")
