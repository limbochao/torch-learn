"""PR !48475: NFNet pooling with a recorded real UB retry.
Keep the original float32[128,256,72,72] input and dynamic=True.
The paired DSL is the pointwise kernel actually used in baseline/retry replay.
See ../../README.md and ../../historical_triggers.json for evidence and trigger conditions.
"""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    mul_tensor_12,
):
    avg_pool2d = torch.ops.aten.avg_pool2d.default(mul_tensor_12, [2, 2], [2, 2], [0, 0], True, False)
    return avg_pool2d


DEFAULT_DYNAMIC = True
INPUT_NAMES = (
    'mul_tensor_12',
)


def make_inputs(device, batch=128, height=288):
    s21 = batch
    s65 = height
    specs = (
        (
            (s21, 256, (s65 - 1) // 4 + 1, (s65 - 1) // 4 + 1),
            (256 * ((s65 - 1) // 4 + 1) ** 2, ((s65 - 1) // 4 + 1) ** 2, (s65 - 1) // 4 + 1, 1),
            torch.float32,
        ),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    return tuple(inputs)


def replay_ub(value, expected):
    """Replay the archived ordinary autotuner that actually exposed the missing small tiles."""
    import importlib.util
    import sys
    from pathlib import Path
    from torch_npu._inductor.runtime import triton_heuristics as runtime

    cached_autotune = runtime.cached_autotune

    def uncached(*values, **kwargs):
        kwargs['inductor_meta']['force_disable_caches'] = True
        return cached_autotune(*values, **kwargs)

    # The saved DSL is the second file of this case; only this explicit replay mode uses it.
    runtime.cached_autotune = uncached
    try:
        source = Path(__file__).with_name('problem_dsl.py')
        spec = importlib.util.spec_from_file_location('archived_pooling', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        runtime.cached_autotune = cached_autotune
    tuner = next(item for item in vars(module).values() if isinstance(item, runtime.NPUCachingAutotuner))
    tiles = sorted({cfg.kwargs['XBLOCK'] for cfg in tuner.configs})
    print(f'REPLAY original_candidates={len(tuner.configs)} original_tiles={tiles}', flush=True)
    # Fail loudly if a different installed generator already has the needed small candidates.
    if not tiles or min(tiles) < 4096:
        raise RuntimeError('Original all-UB condition absent: generator already includes tiles below 4096')
    tuner.precompile(warm_cache_only=True)
    retried = getattr(tuner, '_ub_retry_attempted', False)
    print(f'REPLAY retry_attempted={retried} final_candidates={len(tuner.configs)}', flush=True)
    if not retried:
        raise RuntimeError('UB retry was not exercised; inspect compiler version and original candidates')
    tuner._make_launchers()
    stream = tuner.get_device_interface().get_raw_stream(value.device.index or 0)
    output = torch.empty_like(expected)
    height, width = value.shape[-2:]
    for entry in tuner.compiled_candidate_entries:
        blocks = tuple(block for _, block in entry['candidate']['runtime_blocks'])
        # ks2 is original model height=288, rather than pooling input height=72.
        entry['launcher'](value, output, width // 2, (height // 2) * (width // 2), 288,
                          output.numel(), *blocks, stream=stream)
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-4)
        print(f'PASS replay candidate={entry["launcher"].config.kwargs}', flush=True)


def main():
    import argparse
    from torch.utils import _pytree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile the extracted ATen graph with Inductor")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--replay-ub", action="store_true", help="replay the actual archived all-UB autotuner")
    args = parser.parse_args()
    if args.replay_ub and (not args.device.startswith('npu') or not args.compile):
        parser.error('--replay-ub requires --device npu[:index] --compile')
    if args.replay_ub and (args.batch != 128 or args.height != 288):
        parser.error('--replay-ub preserves the recorded batch=128 height=288')
    if args.replay_ub and args.device not in ('npu', 'npu:0'):
        parser.error('--replay-ub uses the recorded logical device npu:0')
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu
        print(f"torch_npu={torch_npu.__version__}", flush=True)
    torch.manual_seed(0)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if torch.device(args.device).type == "cuda":
        import triton
        print(f"gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    inputs = make_inputs(args.device, args.batch, args.height)
    for name, value in zip(INPUT_NAMES, inputs):
        print(
            f"input={name} shape={tuple(value.shape)} stride={value.stride()} "
            f"dtype={value.dtype} device={value.device} offset={value.storage_offset()}"
        )
    expected = eager_forward(*inputs)
    leaves = _pytree.tree_leaves(expected)
    for value in leaves:
        assert torch.isfinite(value).all().item(), "non-finite eager result"
    print(f"PASS eager outputs={[tuple(value.shape) for value in leaves]}", flush=True)
    if args.compile:
        if args.replay_ub:
            replay_ub(inputs[0], expected)
            return
        compiled = torch.compile(eager_forward, backend="inductor", fullgraph=True, dynamic=args.dynamic)
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print("PASS compiled_vs_eager", flush=True)


if __name__ == "__main__":
    main()
