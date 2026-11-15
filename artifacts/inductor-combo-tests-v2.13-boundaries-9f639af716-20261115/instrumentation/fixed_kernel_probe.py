"""Compile/replay fixed reduction variants for the archived accuracy inputs."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


parser = argparse.ArgumentParser()
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--tile-sweep", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root / ("tile-sweep" if args.tile_sweep else "fixed-kernel")
out.mkdir(exist_ok=True)
original = (root / "original-kernel.py").read_text()
module_ast = ast.parse(original)
fn = next(n for n in module_ast.body if isinstance(n, ast.FunctionDef) and n.name.startswith("triton_red"))
meta_node = next(n.value for n in ast.walk(module_ast) if isinstance(n, ast.keyword) and n.arg == "triton_meta")
attrs_node = next(v for k, v in zip(meta_node.keys, meta_node.values) if ast.literal_eval(k) == "configs")
attrs = ast.literal_eval(attrs_node)[0]
removed = set()
for decorator in fn.decorator_list:
    if isinstance(decorator, ast.Call):
        removed.update(range(decorator.lineno, decorator.end_lineno + 1))
direct = "\n".join(line for number, line in enumerate(original.splitlines(), 1) if number not in removed) + "\n"
variants = {
    "original": direct,
    "r1_then_r0": direct.replace("tl.sum(tmp4, 1)", "tl.sum(tl.sum(_tmp2, 2), 1)"),
    "r0_then_r1": direct.replace("tl.sum(tmp4, 1)", "tl.sum(tl.sum(_tmp2, 1), 1)"),
}
if args.tile_sweep:
    variants = {"original": direct, "r0_block16": direct, "r1_block32": direct, "r0_block1": direct}
if args.compile_only:
    assert not torch_npu.npu.is_initialized()

    def forbid_device_initialization(*pos, **kw):
        raise RuntimeError("Offline compilation must not initialize an NPU device")

    torch_npu.npu._lazy_init = forbid_device_initialization

compiled = []
for name, source in variants.items():
    path = out / (name + ".py")
    if path.exists():
        assert path.read_text() == source
    else:
        path.write_text(source)
    spec = importlib.util.spec_from_file_location("fixed_sum_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, fn.name)
    signature = {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32", "r0_numel": "i32",
                 "r1_numel": "i32", "XBLOCK": "constexpr", "R0_BLOCK": "constexpr", "R1_BLOCK": "constexpr"}
    constants = {"XBLOCK": 2, "R0_BLOCK": 32, "R1_BLOCK": 64}
    if name == "r0_block16":
        constants["R0_BLOCK"] = 16
    if name == "r1_block32":
        constants["R1_BLOCK"] = 32
    if name == "r0_block1":
        constants["R0_BLOCK"] = 1
    options = {"compile_mode": "simd_simt_template", "multibuffer": True, "num_warps": 1, "num_stages": 2}
    binary = triton.compile(ASTSource(kernel, signature, constants, attrs=attrs),
                            target=GPUTarget("npu", "Ascend950PR_9579", 32), options=options)
    metadata = binary.metadata._asdict()
    metadata["target"] = str(metadata["target"])
    record = {"variant": name, "metadata": metadata, "asm_keys": list(binary.asm),
              "constants": constants, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for extension, text in binary.asm.items():
        if isinstance(text, str) and extension in {"ttir", "ttgir", "ttadapter", "llir", "npulir"}:
            (out / f"{name}.{extension}").write_text(text)
    compiled.append((record, kernel, constants, options))

result = {"compile_only": args.compile_only, "variants": [item[0] for item in compiled]}
if args.compile_only:
    assert not torch_npu.npu.is_initialized()
    result["device_remained_uninitialized"] = True
else:
    inputs = torch.load(root / "diagnostic-inputs.pt", map_location="cpu", weights_only=True)
    old = torch.load(root / "diagnostic-outputs.pt", map_location="cpu", weights_only=True)
    result["inputs"] = []
    for index, cpu in enumerate(inputs):
        tensor = cpu.npu()
        assert tensor.stride() == (64, 1024, 1)
        eager = tensor.sum((1, 2)).cpu()
        reference = cpu.double().sum((1, 2))
        row = {"index": index, "eager": eager.tolist(), "reference": reference.tolist(), "variants": {}}
        for record, kernel, constants, options in compiled:
            actual = torch.empty(16, device="npu", dtype=torch.float32)
            kernel[(8,)](tensor, actual, 16, 32, 64, **constants, **options)
            torch.npu.synchronize()
            actual = actual.cpu()
            row["variants"][record["variant"]] = {
                "output": actual.tolist(),
                "exact_archived_compiled_matches": int((actual == old["compiled"][index]).sum()),
                "max_reference_error": (actual.double() - reference).abs().max().item(),
                "mismatch_eager_rows": (~torch.isclose(actual, eager, rtol=1.3e-6, atol=1e-5)).nonzero().flatten().tolist(),
            }
        result["inputs"].append(row)
(root / ("tile-sweep.json" if args.tile_sweep else "offline-target-compile.json" if args.compile_only else "fixed-kernel-replay.json")).write_text(
    json.dumps(result, indent=2, default=str) + "\n"
)
print(json.dumps({"compile_only": args.compile_only, "variants": [r[0]["variant"] for r in compiled],
                  "device_initialized": torch_npu.npu.is_initialized()}, default=str))
