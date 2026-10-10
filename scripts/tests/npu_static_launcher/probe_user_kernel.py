"""Compare user-defined Triton mutation with static launching enabled/disabled."""

import argparse
import json
import os
import traceback
from pathlib import Path
from unittest import mock


os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch_npu
import triton
import triton.language as tl
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch_npu._inductor.triton_experimental.static_launcher import (
    NPUStaticallyLaunchedTritonKernel,
    NPUStaticTritonCompileResult,
)


@triton.jit
def add_scalar_kernel(arg_0, arg_1):
    x = tl.load(arg_0)
    tl.store(arg_0, x + arg_1)


def function(x):
    add_scalar_kernel[(1,)](x, 5)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=int, choices=(0, 1), required=True)
    parser.add_argument("--allow-user", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    triton.runtime.driver.active.get_current_target()
    torch.manual_seed(123)
    eager = torch.zeros(1, device="npu")
    function(eager)
    torch.npu.synchronize()
    result = {
        "static": bool(args.static),
        "allow_user": bool(args.allow_user),
        "eager": eager.cpu().tolist(),
        "compiled": [],
        "error": None,
    }
    original_make = NPUStaticTritonCompileResult.make_launcher
    original_run = NPUStaticallyLaunchedTritonKernel.run
    with (
        config.patch({
            "compile_threads": 1,
            "cpp_wrapper": False,
            "triton.store_cubin": False,
            "use_static_triton_launcher": bool(args.static),
            "strict_static_triton_launcher": True,
            "static_launch_user_defined_triton_kernels": bool(args.allow_user),
            "force_disable_caches": True,
        }),
        mock.patch.object(NPUStaticTritonCompileResult, "make_launcher",
                          autospec=True, side_effect=original_make) as made,
        mock.patch.object(NPUStaticallyLaunchedTritonKernel, "run",
                          autospec=True, side_effect=original_run) as launched,
    ):
        try:
            compiled = torch.compile(function, fullgraph=True,
                                     options={"npu_backend": "triton_experimental"})
            x = torch.zeros(1, device="npu")
            actual, code = run_and_get_code(compiled, x)
            torch.npu.synchronize()
            result["compiled"].append(actual.cpu().tolist())
            for index, source in enumerate(code):
                (args.output / f"output_code_{index}.py").write_text(source)
            x = torch.zeros(1, device="npu")
            actual = compiled(x)
            torch.npu.synchronize()
            result["compiled"].append(actual.cpu().tolist())
        except Exception:
            result["error"] = traceback.format_exc()
        result["npu_static_make_calls"] = made.call_count
        result["npu_static_run_calls"] = launched.call_count
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
