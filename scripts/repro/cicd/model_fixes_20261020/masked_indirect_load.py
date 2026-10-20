"""Independent TA regression: no torch.compile, Inductor, or reduction."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def masked_gather(ids, table, out, rows, width, XBLOCK, B: tl.constexpr, OTHER: tl.constexpr):
    row0 = tl.program_id(0) * XBLOCK
    lane = tl.arange(0, B)[None, :]
    for step in range(XBLOCK):
        row = row0 + step + tl.arange(0, 1)[:, None]
        active = row < rows
        token = tl.load(ids + row, active, other=0)
        value = tl.load(table + token * 768 + lane, active & (lane < width), other=OTHER)
        tl.store(out + row * B + lane, value, active)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expect-fixed', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = torch.zeros(32, device='npu', dtype=torch.int64)
    table = torch.full((4096,), 9.0, device='npu')
    table[:768] = 3
    out = torch.empty((32, 1024), device='npu')
    records = []
    for mode in ('simd_simt_template', 'simt_only'):
        for other in (0.0, -7.0):
            expected = torch.full_like(out, other)
            expected[:, :768] = 3
            binary = masked_gather[(32,)](
                ids, table, out, 32, 768, 1, B=1024, OTHER=other,
                num_warps=1, num_stages=2, compile_mode=mode, multibuffer=False)
            torch.npu.synchronize()
            records.append(dict(mode=mode, other=other, allclose=torch.equal(out, expected),
                                valid_correct=bool((out[:, :768] == 3).all()),
                                tail_min=out[:, 768:].min().item(), tail_max=out[:, 768:].max().item()))
            for kind in ('ttir', 'ttadapter'):
                if kind in binary.asm:
                    (args.output_dir / f'{mode}_{other}.{kind}.txt').write_text(binary.asm[kind])
            print(json.dumps(records[-1]), flush=True)
    (args.output_dir / 'results.json').write_text(json.dumps(records, indent=2))
    if args.expect_fixed:
        assert all(record['allclose'] for record in records), records
