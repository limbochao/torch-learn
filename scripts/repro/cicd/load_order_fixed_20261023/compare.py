"""Compile an extracted Albert kernel directly, changing only its load order.

Run --order front/near with identical options. No Inductor codegen or autotuner
is called. The default is compile-only; --check additionally launches the
compiled binary on deterministic synthetic inputs and compares eager outputs.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import traceback

import torch
import torch_npu
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


ROOT = Path(__file__).resolve().parent


def reference(inputs):
    weighted = [(inputs[0] * ((inputs[1] - inputs[2][:, None]) * inputs[3][:, None])).sum(0)]
    summed = [inputs[0].sum(0)]
    for start in range(4, 81, 7):
        dy = inputs[start] + inputs[start + 1] + inputs[start + 2] + inputs[start + 3]
        normalized = (inputs[start + 4] - inputs[start + 5][:, None]) * inputs[start + 6][:, None]
        weighted.append((dy * normalized).sum(0))
        summed.append(dy.sum(0))
    out0, out1 = weighted[0], summed[0]
    for index in [1, 2, 3, 4, 5, 7, 9, 11, 6, 8, 10]:
        out0 = out0 + weighted[index]
        out1 = out1 + summed[index]
    return out0, out1


def check(binary, output):
    torch.npu.set_device(0)
    torch.manual_seed(0)
    scalar_inputs = {2, 3}
    for start in range(4, 81, 7):
        scalar_inputs.update((start + 5, start + 6))
    inputs = [
        torch.randn((2048,) if index in scalar_inputs else (2048, 4096), device="npu")
        for index in range(81)
    ]
    expected = reference(inputs)
    actual = [torch.empty(4096, device="npu") for _ in range(2)]
    args = [*actual, *inputs, 4096, 2048, 64]
    stream = torch.npu.current_stream().npu_stream
    binary[(64, 1, 1)](*args, stream=stream)
    torch.npu.synchronize()
    errors = [(left - right).abs().max().item() for left, right in zip(actual, expected)]
    output["max_abs"] = errors
    output["atol"], output["rtol"] = 2e-3, 1e-3
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=2e-3, rtol=1e-3)
    output["numeric_status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("front", "near"), required=True)
    parser.add_argument("--multibuffer", choices=("false", "true"), default="false")
    parser.add_argument("--vf-fusion", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kernel_path = ROOT / f"kernel_{args.order}.py"
    spec = importlib.util.spec_from_file_location("extracted_albert", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.triton_per_fused_add_native_layer_norm_backwar_14
    signature = {"in_out_ptr0": "*fp32", "in_out_ptr1": "*fp32"}
    signature.update({f"in_ptr{index}": "*fp32" for index in range(81)})
    signature.update({"x0_numel": "i32", "r1_numel": "i32", "X0BLOCK": "i32"})
    constants = {"X0BLOCK_SUB": 1}
    options = {
        "num_warps": 1,
        "num_stages": 2,
        "debug": False,
        "compile_mode": "simd_simt_template",
        "multibuffer": args.multibuffer == "true",
    }
    if args.vf_fusion:
        options["enable_vf_fusion"] = True
    result = {
        "order": args.order,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": getattr(torch_npu.version, "git_version", None),
        "torch_npu_file": torch_npu.__file__,
        "triton": triton.__version__,
        "triton_file": triton.__file__,
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "target": {"backend": "npu", "arch": "Ascend950PR_9579", "warp_size": 32},
        "constants": constants,
        "runtime": {"x0_numel": 4096, "r1_numel": 2048, "X0BLOCK": 64, "grid": [64, 1, 1]},
        "options": options,
        "cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "inductor_codegen_called": False,
        "compile_status": "NOT_RUN",
        "numeric_status": "NOT_RUN",
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    start = time.monotonic()
    try:
        binary = triton.compile(
            ASTSource(kernel, signature, constants),
            target=GPUTarget("npu", "Ascend950PR_9579", 32),
            options=options,
        )
        result["compile_status"] = "PASS"
        result["compile_seconds"] = time.monotonic() - start
        result["compiled_metadata"] = dict(binary.metadata._asdict())
        for name, text in binary.asm.items():
            if isinstance(text, str):
                (args.output / f"kernel.{name}").write_text(text)
        if args.check:
            result["numeric_status"] = "RUNNING"
            (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
            print("Compile PASS; starting numeric check", flush=True)
            check(binary, result)
    except Exception as error:
        result["exception_type"] = type(error).__name__
        result["error"] = str(error)
        if result["compile_status"] == "NOT_RUN":
            result["compile_status"] = "FAIL"
            result["compile_seconds"] = time.monotonic() - start
        else:
            result["numeric_status"] = "FAIL" if args.check else "NOT_RUN"
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        match = re.search(r"requires (\d+) bits while (\d+) bits available", str(error))
        if match:
            result["ub_required_bits"], result["ub_available_bits"] = map(int, match.groups())
    (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
