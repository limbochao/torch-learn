"""Run the temporary test copy with full-precision NPU convolution.

Accepts run_batch.py arguments. Set TORCHNPU_PRECOMPILE_THREADS before startup,
because this wrapper imports torch_npu before run_batch parses its arguments.
"""

import sys

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from run_batch import main
from skip_validation_support import record_observation


if __name__ == "__main__":
    before = torch.npu.conv.allow_hf32
    record_observation({"kind": "conv_precision_before", "allow_hf32": before})
    try:
        with torch_npu.npu.aclnn.flags(allow_hf32=False):
            record_observation({"kind": "conv_precision_active", "allow_hf32": torch.npu.conv.allow_hf32})
            result = main()
    finally:
        after = torch.npu.conv.allow_hf32
        record_observation({"kind": "conv_precision_after", "allow_hf32": after, "restored": after == before})
    sys.exit(result)
