"""Experiment matrices and forwards. Planning imports no torch/NPU packages."""

from __future__ import annotations

import hashlib
import json
import math

from policy import DEFAULTS, FLOAT_DTYPES, INTEGER_DTYPES, split_boundaries


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "bool": 1, "int32": 4, "int64": 8}
PRIMITIVES = tuple((op, dtype) for op in ("sum", "amax", "amin", "prod")
                   for dtype in (*FLOAT_DTYPES, *INTEGER_DTYPES)) + (
                       ("any", "bool"), ("all", "bool"), ("xor_sum", "int32"), ("xor_sum", "int64"))
FUSIONS = ("mean", "sum_fp32", "add_sum", "exp_sum", "rmsnorm", "softmax", "sum_amax")
CONTROLS = ("argmax", "argmin", "var_mean", "cumsum")
OPS = {op for op, _ in PRIMITIVES} | set(FUSIONS) | set(CONTROLS)
LAYOUTS = ("contiguous", "stride2", "transpose", "multi_axis", "broadcast")
PRESETS = ("smoke", "coarse", "boundary", "coverage", "holdout", "dynamic", "controls")
ANCHORS = tuple(2 ** exponent for exponent in range(8, 23))


def x_values(cores):
    return sorted({1, 2, 4, max(1, cores // 4), max(1, cores // 2), max(1, cores - 1), cores, cores + 1, 2 * cores})


def make_case(op, x, r, dtype="float32", layout="contiguous", role="calibration", **extra):
    spec = dict(op=op, X=x, R=r, dtype=dtype, layout=layout, role=role, **extra)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    spec["id"] = f"{op}-{dtype}-{layout}-x{x}-r{r}-{fingerprint}"
    return spec


def from_spec(spec, **changes):
    values = {key: value for key, value in spec.items() if key != "id"}
    values.update(changes)
    return make_case(**{{"X": "x", "R": "r"}.get(key, key): value for key, value in values.items()})


def context_spec(case):
    return {key: value for key, value in case.items()
            if key not in ("id", "X", "R", "role", "dynamic_axes", "runtime_shapes")}


def context_id(case):
    return json.dumps(context_spec(case), sort_keys=True, separators=(",", ":"))


def matrix(preset, cores, around=(), params=None):
    params = params or DEFAULTS
    result = []

    def add(*args, **kwargs):
        result.append(make_case(*args, **kwargs))

    if preset == "smoke":
        for op, dtype in PRIMITIVES:
            for layout in LAYOUTS:
                add(op, max(1, min(2, cores - 1)), 4096 if layout == "multi_axis" else 65537,
                    dtype, layout, role="smoke")
        for r in (256, 4096, 262144):
            add("sum", 1, r, role="smoke")
        add("sum", cores, 65536, role="smoke")
        add("sum", 7, 65536, layout="transpose", role="smoke")
        for dtype in FLOAT_DTYPES:
            for op in FUSIONS:
                add(op, 2, 65537, dtype, role="smoke")
            for layout in ("stride2", "transpose"):
                for op in ("add_sum", "rmsnorm"):
                    add(op, 2, 65537, dtype, layout, role="smoke")
    elif preset in ("coarse", "holdout", "boundary"):
        for op, dtype in PRIMITIVES:
            for x in x_values(cores):
                if preset == "coarse":
                    sizes = ANCHORS
                elif preset == "holdout":
                    sizes = [math.isqrt(a * b) + 3 for a, b in zip(ANCHORS, ANCHORS[1:])]
                else:
                    changes = split_boundaries(x, cores, params)
                    # First/last S transition up front; intermediate transitions
                    # are added automatically to regions with measured wins.
                    centers = set(around) | set(changes[:1] + changes[-1:])
                    sizes = sorted({max(2, center + delta) for center in centers for delta in (-1, 0, 1)})
                for r in sizes:
                    add(op, x, r, dtype, role="holdout" if preset == "holdout" else "calibration")
    elif preset == "coverage":
        for x in sorted({1, max(2, cores // 4), cores}):
            for r in (4096, 65537, 262144, 1048576):
                for dtype in FLOAT_DTYPES:
                    for op in FUSIONS:
                        add(op, x, r, dtype)
                for layout in ("stride2", "transpose", "multi_axis", "broadcast"):
                    for op, dtype in PRIMITIVES:
                        add(op, x, r, dtype, layout)
                for layout in ("stride2", "transpose"):
                    for dtype in FLOAT_DTYPES:
                        for op in ("add_sum", "rmsnorm"):
                            add(op, x, r, dtype, layout)
    elif preset == "dynamic":
        for op, dtype in (*PRIMITIVES, ("rmsnorm", "float32")):
            for seed_r in (4096, 262144, 1048576):
                add(op, 2, seed_r, dtype, role="dynamic", dynamic_axes=[1],
                    runtime_shapes=[[2, r] for r in (4096, 65537, 262144, 1048576, 4096)])
            for seed_x in (2, max(3, cores // 2)):
                add(op, seed_x, 262144, dtype, role="dynamic", dynamic_axes=[0],
                    runtime_shapes=[[x, 262144] for x in (2, max(3, cores // 2), cores + 1, 2)])
            add(op, 2, 65537, dtype, role="dynamic", dynamic_axes=[0, 1],
                runtime_shapes=[[2, 4096], [max(3, cores // 2), 262144], [cores + 1, 65537], [2, 65537]])
    elif preset == "controls":
        for op in CONTROLS:
            for x in (1, 2):
                add(op, x, 65537, role="control")
    else:
        raise ValueError(preset)
    return list({case["id"]: case for case in result}.values())


def shapes(case):
    return [tuple(shape) for shape in case.get("runtime_shapes", [[case["X"], case["R"]]])]


def input_bytes(case, x, r):
    elements = x * r * (2 if case["layout"] == "stride2" else 1)
    if case["layout"] == "broadcast":
        return (x + r) * DTYPE_BYTES[case["dtype"]]
    return 2 * elements * DTYPE_BYTES[case["dtype"]]


def estimated_working_bytes(case, x, r, settings):
    buffers = settings["rotate_buffers"] if "rotate" in settings["cache_modes"] else 1
    output_size = x * r if case["op"] in ("rmsnorm", "softmax", "cumsum") else x * 3
    return (input_bytes(case, x, r) * buffers + x * r * 32
            + output_size * 8 * len(settings["modes"]))


def input_spec(case, x, r):
    if case["layout"] == "multi_axis":
        inner = next((d for d in (64, 16, 4, 2) if r % d == 0), 1)
        return (x, r // inner, inner), None
    strides = {"contiguous": (r, 1), "transpose": (1, x), "stride2": (2 * r, 2)}
    return (x, r), strides.get(case["layout"])


def forward_function(case):
    import torch

    op = case["op"]
    # prims.xor_sum requires nonnegative axes in its FakeTensor/meta path.
    dims = (1, 2) if case["layout"] == "multi_axis" else (1,)
    broadcast = case["layout"] == "broadcast"
    logical = case["dtype"] == "bool"

    def forward(a, b):
        z = (torch.logical_or(a, b) if logical else a * b if op == "prod" else
             torch.bitwise_xor(a, b) if op == "xor_sum" else a + b) if broadcast else a
        if op == "sum":
            return z.sum(dims)
        if op == "sum_fp32":
            return z.sum(dims, dtype=torch.float32)
        if op == "mean":
            return z.mean(dims)
        if op == "amax":
            return z.amax(dims)
        if op == "amin":
            return z.amin(dims)
        if op == "prod":
            return (z.flatten(1) if len(dims) > 1 else z).prod(-1)
        if op == "any":
            return z.any(dims)
        if op == "all":
            return z.all(dims)
        if op == "xor_sum":
            return torch.ops.prims.xor_sum.default(z, list(dims))
        if op == "add_sum":
            return (a.float() + b.float()).sum(dims)
        if op == "exp_sum":
            return z.float().exp().sum(dims)
        if op == "sum_amax":
            z = a.float() + b.float()
            return z.sum(dims), z.amax(dims)
        if op == "argmax":
            return z.argmax(-1)
        if op == "argmin":
            return z.argmin(-1)
        if op == "var_mean":
            return torch.var_mean(z, dim=dims, correction=0)
        if op == "cumsum":
            return z.cumsum(-1)
        z = z.float()
        if op == "rmsnorm":
            return z * torch.rsqrt((z * z).mean(dims, keepdim=True) + 1e-5)
        if op == "softmax":
            return torch.softmax(z, -1)
        raise ValueError(op)

    return forward


def reference(case, pair):
    """CPU oracle; xor_sum has no eager implementation in PyTorch."""
    import torch

    cpu = tuple(t.detach().cpu() for t in pair)
    if case["op"] != "xor_sum":
        return forward_function(case)(*cpu)
    z = torch.bitwise_xor(*cpu) if case["layout"] == "broadcast" else cpu[0]
    z = z.reshape(z.shape[0], -1)
    while z.shape[1] > 1:
        pairs = z.shape[1] // 2
        combined = torch.bitwise_xor(z[:, :2 * pairs:2], z[:, 1:2 * pairs:2])
        z = torch.cat((combined, z[:, -1:]), dim=1) if z.shape[1] % 2 else combined
    return z[:, 0]


def cpu_values(case, dims, generator, pattern="normal", positions=()):
    """Finite, nontrivial data, plus untimed identity/tail/partition probes."""
    import torch

    dtype = getattr(torch, case["dtype"])
    op = case["op"]
    if op == "prod":
        values = torch.ones(dims, dtype=dtype)
        flat = values.reshape(dims[0], -1)
        n = flat.shape[1]
        if dtype.is_floating_point:
            # At most eight power-of-two pairs: every possible partial product
            # stays finite even when a SIMD lane accumulates only even indices.
            count = min(8, (n - 1) // 2)
            for k in range(count):
                index = (k * (n - 1) // count) if count else 0
                flat[:, index] = 2
                flat[:, index + 1] = 0.5
            flat[:, -1] = 1.125
        else:
            flat[:, :2 * ((n - 1) // 2)] = -1
            flat[:, -1] = 3
        signs = torch.randint(0, 2, (dims[0],), generator=generator) * 2 - 1
        flat[:, -1] *= signs.to(dtype)
    elif dtype == torch.bool:
        values = torch.rand(dims, generator=generator) < 0.05
    elif not dtype.is_floating_point:
        values = torch.randint(-127, 128, dims, generator=generator, dtype=dtype)
    else:
        values = (torch.rand(dims, generator=generator) - 0.5).to(dtype)
    if pattern != "normal":
        identity = 1 if op in ("prod", "all") else 0
        values.fill_(identity)
        if pattern != "identity":
            flat = values.reshape(dims[0], -1)
            indices = [flat.shape[1] - 1] if pattern in ("tail", "zero_tail") else list(positions)
            for index in indices:
                if 0 <= index < flat.shape[1]:
                    flat[:, index] = 0 if op == "all" or pattern == "zero_tail" else -1 if op == "prod" else 1
    return values
