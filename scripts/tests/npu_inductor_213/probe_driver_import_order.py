"""Reproduce a stale helper alias when migration is enabled after NPU helper import.

Run with TORCH_TRANSFER_TO_NPU=0 on the PR plus restored-driver experiment.
The observed failure is a gap in that patch, not a new regression versus the PR.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('TORCH_TRANSFER_TO_NPU') == '0'

    import torch_npu  # noqa: F401
    import triton
    from torch._inductor.runtime import triton_helpers as upstream
    from torch_npu._inductor.runtime import triton_helpers as npu_helpers

    old = upstream.set_driver_to_gpu
    assert npu_helpers.set_driver_to_gpu is old
    from torch_npu.contrib import transfer_to_npu

    result = {
        'upstream_replaced': upstream.set_driver_to_gpu is transfer_to_npu._patch_triton_driver,
        'npu_alias_is_old': npu_helpers.set_driver_to_gpu is old,
        'same_helper': npu_helpers.set_driver_to_gpu is upstream.set_driver_to_gpu,
    }
    for label, helper in [('upstream', upstream), ('previously_imported_npu', npu_helpers)]:
        try:
            helper.set_driver_to_gpu()
            result[label] = {'result': 'passed', 'driver': type(triton.runtime.driver.active).__name__}
        except Exception as error:
            result[label] = {'result': 'failed', 'error': type(error).__name__ + ': ' + str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
