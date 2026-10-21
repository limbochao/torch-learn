"""ConvNeXt LayerNorm stack overflow: one fixed compile and one direct launch.

Default NUM_WARPS=64 reproduces; NUM_WARPS=8 runs the diagnostic control.
No model, Inductor, autotuner, capture files, or weights are required.
"""
import os
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
import torch
import torch_npu


@triton.jit
def triton_red_fused_add_native_layer_norm_permute_8(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel,
    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr,
):
    r0_numel = 80
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_2 + 80*ks0*x1),
                       r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.full([1, 1], 80.0, tl.float32)
    tmp5 = (tmp2 / tmp4)
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp6 = tl.load(in_ptr0 + (x0 + ks0*r0_2 + 80*ks0*x1),
                       r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 - tmp5
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    x4 = (xindex % ks1)
    x5 = ((xindex // ks1) % ks1)
    tmp12 = tl.full([1, 1], 80.0, tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tl.full([1, 1], 1e-06, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tl.store(out_ptr1 + (x4
                      + x1*((1) * ((1) >= (ks1)) + (ks1) * ((ks1) > (1)))*((1) * ((1) >= (ks1)) + (ks1) * ((ks1) > (1)))
                        + x5*((1) * ((1) >= (ks1)) + (ks1) * ((ks1) > (1)))), tmp16, xmask)


torch.npu.set_device(0)
torch.manual_seed(0)
num_warps = int(os.environ.get("NUM_WARPS", "64"))
source = ASTSource(
    triton_red_fused_add_native_layer_norm_permute_8,
    signature={
        "in_out_ptr0": "*fp32", "in_ptr0": "*fp32", "out_ptr0": "*fp32", "out_ptr1": "*fp32",
        "ks0": "i64", "ks1": "i64", "xnumel": "i32", "r0_numel": "i32",
        "XBLOCK": "constexpr", "R0_BLOCK": "constexpr",
    },
    constexprs={"XBLOCK": 8, "R0_BLOCK": 128},
    attrs={(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3, 7)},
)
binary = triton.compile(
    source, target=GPUTarget("npu", "Ascend950PR_9579", 32),
    options={"num_warps": num_warps, "num_stages": 2, "debug": False,
             "compile_mode": "simt_only", "enable_auto_blockify": True},
)
print("COMPILED", binary.metadata, flush=True)
x = torch.randn((128, 80, 56, 56), device="npu", dtype=torch.float32)
mean = torch.empty_strided((128, 56, 56, 1), (3136, 56, 1, 401408), device="npu")
var = torch.empty_strided((128, 56, 56, 1), (3136, 56, 1, 401408), device="npu")
rstd = torch.empty_strided((128, 56, 56, 1), (3136, 56, 1, 1), device="npu")
# The Triton runner consumes constexpr positions too; they are not binary ABI parameters.
binary[(50176, 1, 1)](mean, x, var, rstd, 3136, 56, 401408, 80, 8, 128)
torch.npu.synchronize()
print("LAUNCH_PASS", flush=True)

expected_mean = x.mean(1, keepdim=False).unsqueeze(-1)
expected_var = (x - expected_mean.permute(0, 3, 1, 2)).square().sum(1).unsqueeze(-1)
for actual, expected in ((mean, expected_mean), (var, expected_var),
                         (rstd, torch.rsqrt(expected_var / 80 + 1e-6))):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
print("THREE_OUTPUTS_PASS", flush=True)
