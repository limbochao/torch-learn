import triton
import triton.language as tl

# Copied from the installed torch/_inductor/runtime/triton_helpers.py.
@triton.jit
def div_floor_integer(a, b):
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)

@triton.jit
def triton_poi_fused_avg_pool2d_15(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks0)
    x2 = xindex // ks1
    tmp0 = tl.load(in_ptr0 + (x2 + 2*x0 + 2*x1 + x2*(div_floor_integer((-1) + ks2,  4))*(div_floor_integer((-1) + ks2,  4)) + 2*x1*(div_floor_integer((-1) + ks2,  4)) + 2*x2*(div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(div_floor_integer((-1) + ks2,  4))*(div_floor_integer((-1) + ks2,  4)) + 2*x1*(div_floor_integer((-1) + ks2,  4)) + 2*x2*(div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(div_floor_integer((-1) + ks2,  4))*(div_floor_integer((-1) + ks2,  4)) + 2*x1*(div_floor_integer((-1) + ks2,  4)) + 2*x2*(div_floor_integer((-1) + ks2,  4)) + (div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + x2 + 2*x0 + 2*x1 + x2*(div_floor_integer((-1) + ks2,  4))*(div_floor_integer((-1) + ks2,  4)) + 2*x1*(div_floor_integer((-1) + ks2,  4)) + 2*x2*(div_floor_integer((-1) + ks2,  4)) + (div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tl.store(out_ptr0 + (x0 + x1 + x2 + x1*(div_floor_integer((-1) + ks2,  8)) + x2*(div_floor_integer((-1) + ks2,  8))*(div_floor_integer((-1) + ks2,  8)) + 2*x2*(div_floor_integer((-1) + ks2,  8))), tmp8, xmask)

if __name__ == "__main__":
    import torch
    import torch_npu
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    signature = {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64',
                 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}
    attrs = {(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]],
             (5,): [['tt.divisibility', 16]]}
    options = {'num_warps': 1, 'num_stages': 2, 'debug': False,
               'compile_mode': 'simd_simt_template', 'multibuffer': False, 'enable_auto_blockify': True}
    source = ASTSource(triton_poi_fused_avg_pool2d_15,
                       signature, {(6,): 2048}, attrs)
    binary = triton.compile(source, target=GPUTarget('npu', 'Ascend950PR_9579', 32), options=options)
    torch.manual_seed(1337)
    data = torch.randn(2, 256, 56, 56, device='npu')
    expected = torch.nn.functional.avg_pool2d(data, 2)
    out = torch.empty_like(expected)
    binary[(triton.cdiv(out.numel(), 2048), 1, 1)](data, out, 28, 784, 224, out.numel(), 2048)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    print('allclose=True, max_abs=', (out - expected).abs().max().item())
