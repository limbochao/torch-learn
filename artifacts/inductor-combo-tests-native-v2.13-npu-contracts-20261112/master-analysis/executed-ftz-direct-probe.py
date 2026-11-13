import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor-cache"))

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo
_dynamo.use_jit_script = True
_dynamo.patch_has_triton()
import torch_npu._inductor
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

torch.npu.set_device(0)
result = {"started": time.time(), "torch_git": torch.version.git_version,
          "torch_npu": torch_npu.__version__, "triton": triton.__version__,
          "physical_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
          "master_device_retest": False, "records": []}


@triton.jit
def raw_numerics(A, E, O_copy, O_add, O_half, O_exp2, O_exp, N: tl.constexpr):
    x = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = x < N
    a = tl.load(A + x, valid, other=0)
    e = tl.load(E + x, valid, other=0)
    tl.store(O_copy + x, a, valid)
    tl.store(O_add + x, a + a, valid)
    tl.store(O_half + x, a * 0.5, valid)
    tl.store(O_exp2 + x, libdevice.exp2(e), valid)
    tl.store(O_exp + x, libdevice.exp(e * 0.6931471805599453), valid)


def bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int32).tolist()


def save():
    result["finished"] = time.time()
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


a = torch.tensor([1, 0x00400000, 0x007fffff, 0x00800000, 0x01000000,
                  0x3f800000, -2147483647, -2143289344, 0], dtype=torch.int32).view(torch.float32).repeat(128)
e = torch.tensor([-127.0, -128.0, -126.0, -125.0, 0.0, 1.0, -149.0, -150.0, -136.0]).repeat(128)
result["cpu_expected_bits"] = [bits(x)[:9] for x in [a, a + a, a * 0.5, torch.exp2(e),
                                                    torch.exp(e * 0.6931471805599453)]]
a_device, e_device = a.to("npu"), e.to("npu")
result["eager_npu_bits"] = [bits(x)[:9] for x in [a_device, a_device + a_device, a_device * 0.5,
                                                 torch.exp2(e_device),
                                                 torch.exp(e_device * 0.6931471805599453)]]

for mode in ["simd", "simd_simt_template", "simt_only"]:
    record = {"mode": mode}
    try:
        outputs = [torch.empty_like(a_device) for _ in range(5)]
        kernel = raw_numerics[(triton.cdiv(a.numel(), 128),)](
            a_device, e_device, *outputs, a.numel(), compile_mode=mode)
        torch.npu.synchronize()
        record["output_bits"] = [bits(x)[:9] for x in outputs]
        record["compiled_hash"] = kernel.hash
        record["compiler_mode"] = kernel.metadata.compile_mode
        record["is_pure_simt"] = kernel.metadata.is_pure_simt
        record["parallel_mode"] = kernel.metadata.parallel_mode
        record["asm"] = []
        for kind in ["ttir", "ttadapter", "bcmlir"]:
            source = kernel.asm.get(kind)
            if isinstance(source, str):
                p = root / f"{mode}-{kind}.mlir"
                p.write_text(source)
                record["asm"].append({"kind": kind, "path": p.name,
                                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        record["status"] = "OBSERVED"
    except Exception:
        record["status"] = "PROBE_ERROR"
        record["error"] = traceback.format_exc()
    result["records"].append(record)
    save()
    print("DIRECT_FTZ_RECORD=" + json.dumps(record, ensure_ascii=False), flush=True)


def independent_exponentials(x, y):
    return torch.exp2(x), torch.exp(y * 0.6931471805599453)


from torch._inductor import config
from torch._inductor.utils import run_and_get_code
for flag in [False, True]:
    record = {"mode": "independent_combo", "disable_ftz": flag}
    try:
        torch._dynamo.reset()
        with config.patch({"combo_kernels": True, "benchmark_combo_kernel": False,
                           "combo_kernel_per_subkernel_blocks": False,
                           "combo_kernel_max_distance": -1,
                           "combo_kernel_peak_memory_increase_gb": None,
                           "combo_kernel_peak_memory_pct_threshold": None,
                           "eager_numerics.disable_ftz": flag}):
            actual, sources = run_and_get_code(torch.compile(independent_exponentials),
                                               e_device, e_device.clone())
            torch.npu.synchronize()
        record["output_bits"] = [bits(x)[:9] for x in actual]
        record["combo_count"] = sum(s.count("@triton_heuristics.combo_kernel") for s in sources)
        record["metadata_has_disable_ftz"] = any("disable_ftz" in s for s in sources)
        for index, source in enumerate(sources):
            (root / f"independent-{flag}-{index}.py").write_text(source)
        record["status"] = "OBSERVED"
    except Exception:
        record["status"] = "PROBE_ERROR"
        record["error"] = traceback.format_exc()
    result["records"].append(record)
    save()
    print("COMBO_FTZ_RECORD=" + json.dumps(record, ensure_ascii=False), flush=True)
