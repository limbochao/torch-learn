"""Validate the installed migration helper, a compile worker, and the arange1 graph.

Run inside the target container with TORCH_TRANSFER_TO_NPU=1 and a fresh cache.
This script observes the installed implementation; it does not install a patch.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch_npu
    import triton
    from torch._inductor.compile_worker.subproc_pool import SubprocPool
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.utils import run_and_get_code

    torch.npu.set_device(args.device_index)
    assert triton_helpers.set_driver_to_gpu.__name__ == '_patch_triton_driver'
    pool = SubprocPool(1)
    expression = """(lambda helper, triton: {
        'helper_name': helper.set_driver_to_gpu.__name__,
        'helper_module': helper.set_driver_to_gpu.__module__,
        'call_result': helper.set_driver_to_gpu(),
        'driver_class': type(triton.runtime.driver.active).__name__,
    })(__import__('torch._inductor.runtime.triton_helpers', fromlist=['set_driver_to_gpu']),
       __import__('triton'))"""
    try:
        worker = pool.submit(eval, expression).result(timeout=150)
    finally:
        pool.shutdown()
    assert worker['helper_name'] == '_patch_triton_driver'
    assert worker['driver_class'] == 'NPUDriver'
    (args.output_dir / 'worker-probe.json').write_text(json.dumps(worker, indent=2) + '\n')

    def fn(x):
        rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
        rng2 = torch.arange(10, 18, device=x.device)
        tmp = x * rng1
        return tmp, tmp + rng2

    scenarios = []
    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(0)
        x = torch.randn(8, 8, device='npu', dtype=dtype)
        expected = fn(x)
        compiled = torch.compile(fn, fullgraph=True)
        actual, modules = run_and_get_code(compiled, x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(compiled(x), expected)
        scenarios.append({
            'check': 'passed', 'shape': list(x.shape), 'dtype': str(dtype),
            'helper_calls_in_generated_code': sum(m.count('triton_helpers.set_driver_to_gpu()') for m in modules),
            'modules': modules,
        })
    result = {
        'check': 'passed', 'scenarios': scenarios,
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'triton': triton.__version__,
        'driver': type(triton.runtime.driver.active).__name__,
    }
    (args.output_dir / 'arange-artifact.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ARANGE_CHECK=' + json.dumps([
        {k: v for k, v in scenario.items() if k != 'modules'} for scenario in scenarios
    ]), flush=True)


if __name__ == '__main__':
    main()
