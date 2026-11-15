"""Compare fixed diagnostic inputs with explicit FP32 summation orders."""

import hashlib
import json
import math
from pathlib import Path

import torch


torch.set_num_threads(1)
root = Path(__file__).resolve().parent
inputs = torch.load(root / "diagnostic-inputs.pt", map_location="cpu", weights_only=True)
outputs = torch.load(root / "diagnostic-outputs.pt", map_location="cpu", weights_only=True)


def tree_sum(values, adjacent):
    values = values.clone()
    while values.shape[-1] > 1:
        half = values.shape[-1] // 2
        values = values[..., ::2] + values[..., 1::2] if adjacent else values[..., :half] + values[..., half:]
    return values.squeeze(-1)


def error(value, reference):
    difference = value.double() - reference
    return {"max_abs_error": difference.abs().max().item(), "signed_errors": difference.tolist()}


results = []
for index, tensor in enumerate(inputs):
    assert tensor.shape == (16, 32, 64) and tensor.stride() == (64, 1024, 1)
    addresses = (torch.arange(16)[:, None, None] * 64
                 + torch.arange(32)[None, :, None] * 1024
                 + torch.arange(64)[None, None, :])
    storage = tensor.as_strided((tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,))
    assert torch.equal(storage[addresses], tensor)
    flat = tensor.reshape(16, 2048)
    reference = torch.tensor([math.fsum(row.tolist()) for row in flat], dtype=torch.float64)
    assert torch.equal(reference, flat.double().sum(1))
    eager = outputs["eager"][index]
    compiled = outputs["compiled"][index]
    variants = {
        "cpu_sum": flat.sum(1),
        "sum_r0_then_r1": tensor.sum(1).sum(1),
        "sum_r1_then_r0": tensor.sum(2).sum(1),
        "balanced_adjacent": tree_sum(flat, True),
        "balanced_halves": tree_sum(flat, False),
        "r0_tree_then_r1_tree": tree_sum(tree_sum(tensor.transpose(1, 2), False), False),
        "r1_tree_then_r0_tree": tree_sum(tree_sum(tensor, False), False),
        "correctly_rounded_fp32": reference.float(),
    }
    mismatch = ~torch.isclose(compiled, eager, rtol=1.3e-6, atol=1e-5)
    result = {
        "input_index": index, "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "dsl_address_map_matches_all_input_elements": True,
        "reference_float64_matches_math_fsum": True,
        "reference": reference.tolist(), "eager": eager.tolist(), "compiled": compiled.tolist(),
        "eager_error": error(eager, reference), "compiled_error": error(compiled, reference),
        "original_comparison_mismatch_rows": mismatch.nonzero().flatten().tolist(),
        "variants": {},
    }
    for name, value in variants.items():
        result["variants"][name] = {
            **error(value, reference), "output": value.tolist(),
            "exact_compiled_matches": int((value == compiled).sum()),
            "exact_eager_matches": int((value == eager).sum()),
            "mismatch_eager_rows": (~torch.isclose(value, eager, rtol=1.3e-6, atol=1e-5)).nonzero().flatten().tolist(),
        }
    results.append(result)

summary = {
    "torch": torch.__version__, "torch_git": torch.version.git_version,
    "input_sha256": hashlib.sha256((root / "diagnostic-inputs.pt").read_bytes()).hexdigest(),
    "input_provenance": "Fixed supplementary diagnostic inputs from accuracy-v2; not original full-run inputs.",
    "device_execution": "CPU only; archived NPU outputs compared without replacing input data.",
    "inputs": results,
}
(root / "cpu-summation-orders.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in results:
    print("input", row["input_index"], "original mismatches", row["original_comparison_mismatch_rows"])
    for name, value in row["variants"].items():
        print(name, "compiled exact", value["exact_compiled_matches"], "eager exact", value["exact_eager_matches"],
              "max error", value["max_abs_error"], "mismatches vs eager", value["mismatch_eager_rows"])
