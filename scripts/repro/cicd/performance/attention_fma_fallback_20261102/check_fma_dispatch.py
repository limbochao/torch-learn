"""Count CPU operations executed by one prims.fma; no device timing or compiler patch."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from torch._inductor import inductor_prims  # noqa: F401 registers prims.fma
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    a = torch.rand(2, 4, 32, 32)
    b = torch.rand(2, 4, 32, 1)
    c = torch.rand_like(a)
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        result = torch.ops.prims.fma.default(a, b, c)
    torch.testing.assert_close(result, (a * b) + c)
    counts = {
        event.key: event.count for event in profiler.key_averages()
        if event.key in ("prims::fma", "aten::mul", "aten::add")
    }
    observed_duplicate = counts == {"prims::fma": 1, "aten::mul": 2, "aten::add": 2}
    record = {
        "torch": torch.__version__, "device": "cpu", "dtype": str(a.dtype),
        "input_shape": list(a.shape), "input_stride": list(a.stride()),
        "broadcast_shape": list(b.shape), "broadcast_stride": list(b.stride()),
        "counts": counts, "output_check": "passed", "observed_duplicate": observed_duplicate,
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if observed_duplicate else 1


if __name__ == "__main__":
    raise SystemExit(main())
