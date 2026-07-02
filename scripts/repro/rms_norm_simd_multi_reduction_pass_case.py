"""
Passing RMSNorm weight-gradient case for SIMD multi-reduction codegen.

This case keeps the same eager expression and non-contiguous grad_out layout as
the larger repro, but uses a smaller shape that has been checked against eager
on NPU_A2.

Run:

    python scripts/repro/rms_norm_simd_multi_reduction_pass_case.py
"""

import os

from rms_norm_simd_multi_reduction_repro import main


os.environ.setdefault("BATCH", "2")
os.environ.setdefault("SEQ", "128")
os.environ.setdefault("HEADS", "8")
os.environ.setdefault("HEAD_DIM", "128")
os.environ.setdefault("CHECK", "1")
os.environ.setdefault("DYNAMIC", "0")


if __name__ == "__main__":
    main()
