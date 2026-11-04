"""Direct Triton replay of the two accuracy patterns, without Inductor."""
import json
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def two_adds(A, B, C, O, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + index, index < N)
    b = tl.load(B + index, index < N)
    c = tl.load(C + index, index < N)
    ab = a + b
    tl.static_assert(ab.dtype == tl.bfloat16)
    result = ab + c
    tl.store(O + index, result, index < N)


@triton.jit
def add_sigmoid_mul(A, B, O, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + index, index < N)
    b = tl.load(B + index, index < N)
    ab = a + b
    tl.static_assert(ab.dtype == tl.bfloat16)
    sigmoid = tl.sigmoid(ab.to(tl.float32)).to(tl.bfloat16)
    result = ab * sigmoid
    tl.store(O + index, result, index < N)


def main():
    torch.npu.set_device('npu:0')
    values = torch.arange(129, dtype=torch.float32) / 256 + 0.5
    a = values[:, None].expand(129, 129).contiguous().bfloat16().flatten().npu()
    b = values[None, :].expand(129, 129).contiguous().bfloat16().flatten().npu()
    c = torch.full_like(a, 0.50390625)
    reports = []
    for name, fn in [('two_adds', two_adds), ('add_sigmoid_mul', add_sigmoid_mul)]:
        actual = torch.empty_like(a)
        if name == 'two_adds':
            fn[(triton.cdiv(a.numel(), 256),)](a, b, c, actual, a.numel(), 256)
            expected = (a + b) + c
            promoted = (a.float() + b.float() + c.float()).bfloat16()
        else:
            fn[(triton.cdiv(a.numel(), 256),)](a, b, actual, a.numel(), 256)
            expected = (a + b) * torch.sigmoid(a + b)
            promoted = ((a.float() + b.float())
                        * torch.sigmoid(a.float() + b.float()).bfloat16().float()).bfloat16()
        torch.npu.synchronize()
        close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
        row = {'pattern': name, 'numel': a.numel(), 'mismatched': int((~close).sum()),
               'max_abs': float((actual.float() - expected.float()).abs().max()),
               'matches_unrounded_add_model': torch.equal(actual, promoted)}
        bad = (~close).nonzero()
        if bad.numel():
            i = int(bad[0])
            row['example'] = {'a': float(a[i]), 'b': float(b[i]), 'c': float(c[i]),
                              'eager': float(expected[i]), 'actual': float(actual[i]),
                              'unrounded_add_model': float(promoted[i])}
        reports.append(row)
        print(json.dumps(row), flush=True)
    Path('direct_triton_probe.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
