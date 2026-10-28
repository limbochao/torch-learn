# Standalone single-kernel compile and launch reproducer.
# Extracted by repro_compile.py for triton_npu_triton_bmm_batched.
# It calls Triton's compiler/runtime directly with one fixed candidate config.
# It does not instantiate Inductor's autotuner or rerun Dynamo/AOTAutograd.

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
tl_math = tl.math
import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
import triton.language.extra.cann.libdevice as libdevice
import triton.language.extra.cann.extension as extension
tl_math = tl.math

@triton.jit
def triton_npu_triton_bmm_batched(arg_A, arg_B, out_ptr0):
    BLOCK_Q: tl.constexpr = 2
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 256
    BLOCK_K: tl.constexpr = 144
    GROUPS_PER_PROGRAM: tl.constexpr = 55
    RAGGED_GROUPS: tl.constexpr = True
    N_GROUP: tl.constexpr = 28
    L1_COPIES: tl.constexpr = 2
    MASKED_M: tl.constexpr = False
    MASKED_N: tl.constexpr = True
    MASKED_K: tl.constexpr = False
    ALLOW_HF32: tl.constexpr = False
    ACC_TYPE: tl.constexpr = tl.float32
    ALLOW_TF32: tl.constexpr = False
    INDEX_DTYPE: tl.constexpr = tl.int32
    A = arg_A
    B = arg_B
    M = 128
    N = 160
    K = 144
    BATCH = 3072
    stride_aq = 18432
    stride_am = 144
    stride_ak = 1
    stride_bq = 23040
    stride_bk = 160
    stride_bn = 1
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)
    first_q = tl.program_id(0) * (GROUPS_PER_PROGRAM * BLOCK_Q)
    n_groups = tl.maximum(0, tl.minimum(GROUPS_PER_PROGRAM, (BATCH - first_q) // BLOCK_Q))
    for g in range(n_groups):
        q = first_q + g * BLOCK_Q + offs_q
        idx_q = q[:, None, None].to(INDEX_DTYPE)
        a_ptrs = A + idx_q * stride_aq + offs_m[None, :, None] * stride_am + offs_k[None, None, :] * stride_ak
        a = tl.load(a_ptrs)
        extension.multibuffer(a, L1_COPIES)
        b_ptrs = B + idx_q * stride_bq + offs_k[None, :, None] * stride_bk + offs_n[None, None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_k[None, :, None] < K) & (offs_n[None, None, :] < N), other=0.0)
        extension.multibuffer(b, L1_COPIES)
        acc = tl.zeros((BLOCK_Q, BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        acc = tl.dot(a, b, acc=acc, out_dtype=ACC_TYPE)
        idx_m = offs_m[None, :, None]
        idx_n = offs_n[None, None, :]
        mask = idx_n < N
        xindex = idx_n + 160 * idx_m + 20480 * idx_q
        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_Q, BLOCK_M, BLOCK_N]), acc, mask)

# ---- direct standalone reproducer helpers ----
_STANDALONE_FN_NAME = 'triton_npu_triton_bmm_batched'
_STANDALONE_WANTED_CONFIG = {'num_warps': 4, 'num_stages': 4}
_STANDALONE_SIGNATURE = {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}
_STANDALONE_CONSTANTS = {}
_STANDALONE_ATTRS = {(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}
_STANDALONE_TARGET = {'backend': 'npu', 'arch': 'Ascend950PR_9579', 'warp_size': 32}
_STANDALONE_DEVICE_INDEX = 0
_STANDALONE_OPTIONS = {'num_warps': 4, 'num_stages': 4, 'debug': False, 'compile_mode': 'simd_simt_template', 'enable_hivm_batch_matmul': True, 'unit_flag': False}
_STANDALONE_INDUCTOR_META = {'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2']}
_STANDALONE_LAUNCH_ARG_NAMES = ['arg_A', 'arg_B', 'out_ptr0']
_STANDALONE_RUNTIME_BLOCK_NAMES = ()
_STANDALONE_BASE_LAUNCH_ARG_NAMES = ['arg_A', 'arg_B', 'out_ptr0']
_STANDALONE_EXTRA_LAUNCH_ARG_NAMES = ['_grid_0', '_grid_1', '_grid_2']


def _standalone_repro_compile():
    import triton
    try:
        import torch
        if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            torch.npu.set_device(_STANDALONE_DEVICE_INDEX)
    except (ImportError, RuntimeError):
        # Triton can still select its active device in environments where the
        # torch_npu convenience namespace is unavailable (for example during
        # compile-only diagnostics with a mocked driver).
        pass
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    kernel_fn = globals().get(_STANDALONE_FN_NAME)
    if kernel_fn is None:
        raise RuntimeError(
            f"Kernel function {_STANDALONE_FN_NAME!r} is missing from export"
        )
    target_info = _STANDALONE_TARGET
    if not target_info.get("arch"):
        raise RuntimeError(
            "Captured device architecture is unavailable; recapture with "
            "trace_compile.py so direct compile metadata is journaled"
        )
    target = GPUTarget(
        target_info.get("backend", "npu"),
        target_info["arch"],
        target_info.get("warp_size") or 32,
    )
    attrs = _STANDALONE_ATTRS or None
    if isinstance(attrs, dict):
        # Triton-Ascend 3.2 serializes an AttrsDescriptor as
        # ``{'arg_properties': ..., 'cls': ...}``; older releases use keyword
        # fields such as ``divisible_by_16`` and newer releases consume the
        # raw tuple-key mapping directly.  Rebuild only the descriptor shapes
        # that identify themselves, leaving the newest raw mapping untouched.
        descriptor_wrapper = attrs.get("__attrs_descriptor__")
        if isinstance(descriptor_wrapper, dict):
            attrs = descriptor_wrapper
        descriptor_data = attrs
        try:
            from triton.compiler.compiler import AttrsDescriptor
        except ImportError:
            try:
                from triton.backends.compiler import AttrsDescriptor
            except ImportError:
                AttrsDescriptor = None
        if AttrsDescriptor is not None:
            if {
                "arg_properties",
                "cls",
            }.issubset(descriptor_data) and hasattr(
                AttrsDescriptor,
                "from_dict",
            ):
                attrs = AttrsDescriptor.from_dict(descriptor_data)
            elif any(
                key in descriptor_data
                for key in (
                    "divisible_by_16",
                    "equal_to_1",
                    "pointer_range_32",
                )
            ):
                try:
                    attrs = AttrsDescriptor(**descriptor_data)
                except (TypeError, ValueError):
                    # A backend-specific descriptor may require a different
                    # constructor; passing its mapping through is preferable
                    # to turning a useful capture into an export-time crash.
                    attrs = descriptor_data
    if attrs is None:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
        )
    else:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
            attrs,
        )
    return triton.compile(
        source,
        target=target,
        options=dict(_STANDALONE_OPTIONS),
    )


def _standalone_repro_current_stream():
    """Return the backend's raw current stream without an Inductor import."""
    import torch

    npu = getattr(torch, "npu", None)
    current_stream = getattr(npu, "current_stream", None)
    if not callable(current_stream):
        return None
    stream = current_stream()
    # torch_npu streams expose the integer ABI handle as ``npu_stream``;
    # keeping the object as a fallback also works with lightweight test
    # doubles and Triton versions that accept a stream wrapper directly.
    return getattr(stream, "npu_stream", stream)


def _standalone_repro_runtime_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_RUNTIME_BLOCK_NAMES
        or meta.get("runtime_block_append_order")
        or meta.get("runtime_block_arg_names")
        or ()
    )


def _standalone_repro_extra_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_EXTRA_LAUNCH_ARG_NAMES
        or meta.get("extra_launcher_args")
        or ()
    )


def _standalone_repro_legacy_prefix(args, legacy_launch_args):
    """Strip candidate-specific runtime blocks from an old payload tuple."""
    args = tuple(args)
    if legacy_launch_args is None:
        return args
    legacy = tuple(legacy_launch_args)
    runtime_count = len(_STANDALONE_RUNTIME_BLOCKS)
    if runtime_count and len(legacy) >= len(args) + runtime_count:
        prefix = legacy[:-runtime_count]
        if len(prefix) >= len(args):
            return prefix
    return legacy if len(legacy) >= len(args) else args


def _standalone_repro_split_launch_args(raw_launch_args):
    """Split Inductor's ``kernel args + extra grid args + runtime blocks``."""
    raw_launch_args = tuple(raw_launch_args)
    launch_names = tuple(_STANDALONE_LAUNCH_ARG_NAMES)
    runtime_names = _standalone_repro_runtime_names()
    base_names = tuple(
        _STANDALONE_BASE_LAUNCH_ARG_NAMES
        or (name for name in launch_names if name not in runtime_names)
    )
    extra_names = _standalone_repro_extra_names()
    raw_names = (*base_names, *extra_names, *runtime_names)

    if len(raw_launch_args) == len(raw_names):
        raw_values = dict(zip(raw_names, raw_launch_args))
        kernel_args = tuple(raw_values[name] for name in launch_names)
        return kernel_args, raw_values
    if len(raw_launch_args) == len(launch_names):
        # Newer capture records may already have removed extra launcher args.
        return raw_launch_args, dict(zip(launch_names, raw_launch_args))

    raise RuntimeError(
        "Captured launch argument count does not match the direct signature: "
        f"got {len(raw_launch_args)}, expected {len(raw_names)} (raw) or "
        f"{len(launch_names)} (kernel)"
    )


def _standalone_repro_value(values, name):
    if name in values:
        return values[name]
    lowered = str(name).lower()
    for key, value in values.items():
        if str(key).lower() == lowered:
            return value
    return None


def _standalone_repro_numel(values, axis):
    candidates = (
        f"{axis}_numel",
        f"{axis}numel",
        f"{str(axis).upper()}_NUMEL",
        f"{str(axis).upper()}NUMEL",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No numel argument found for axis {axis!r}")


def _standalone_repro_block(values, axis):
    candidates = (
        f"{str(axis).upper()}BLOCK",
        f"{axis}block",
        f"{axis}_block",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    # A non-runtime block is normally a constexpr/config value.
    for name in candidates:
        value = _standalone_repro_value(_STANDALONE_CONSTANTS, name)
        if value is None:
            value = _standalone_repro_value(_STANDALONE_WANTED_CONFIG, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No block value found for axis {axis!r}")


def _standalone_repro_grid_value(expression, values):
    if isinstance(expression, bool):
        return int(expression)
    if isinstance(expression, (int, float)):
        return int(expression)
    if not isinstance(expression, str):
        raise RuntimeError(f"Unsupported captured grid expression: {expression!r}")
    direct = _standalone_repro_value(values, expression)
    if direct is not None:
        return int(direct)
    try:
        return int(expression)
    except ValueError:
        pass
    # Fixed/precomputed grids emitted by Inductor generally contain only
    # symbols and arithmetic.  Evaluate that narrow expression against the
    # captured values, with Python builtins disabled.
    try:
        return int(eval(expression, {"__builtins__": {}}, dict(values)))
    except Exception as error:
        raise RuntimeError(
            f"Cannot evaluate captured grid expression {expression!r}: {error}"
        ) from error


def _standalone_repro_grid(raw_launch_args):
    _, values = _standalone_repro_split_launch_args(raw_launch_args)
    meta = _STANDALONE_INDUCTOR_META or {}

    fixed_grid = meta.get("fixed_grid")
    if fixed_grid is not None:
        if not isinstance(fixed_grid, (tuple, list)):
            raise RuntimeError(f"Unsupported captured fixed_grid: {fixed_grid!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in fixed_grid
        )
        return (result + (1, 1, 1))[:3]

    precomputed = meta.get("precomputed_grids")
    if precomputed:
        selected = None
        for candidate in precomputed:
            candidate_config = candidate.get("config", {})
            if all(
                _STANDALONE_WANTED_CONFIG.get(name) == value
                for name, value in candidate_config.items()
            ):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                "No precomputed grid matches the captured fixed config: "
                f"{_STANDALONE_WANTED_CONFIG!r}"
            )
        expressions = selected.get("python") or selected.get("python_slow")
        if expressions is None:
            raise RuntimeError(f"Precomputed grid has no Python expression: {selected!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in expressions
        )
        return (result + (1, 1, 1))[:3]

    axis_names = tuple(meta.get("axis_names") or ())
    explicit_runtime_names = _standalone_repro_runtime_names()
    runtime_names = explicit_runtime_names
    grid_type = str(meta.get("grid_type") or "")
    # If an older record has only ``split_blocks`` metadata, defer the
    # runtime-name inference below.  Inferring ``Z0BLOCK``/etc. too early
    # would make the split-block fallback unreachable and incorrectly demand
    # values that were never appended to the launch ABI.
    if not runtime_names and not (
        grid_type == "GridNpu" and meta.get("split_blocks") is not None
    ):
        # Older records did not preserve runtime-block names.  GridNpu uses
        # <AXIS>BLOCK names in split-axis order; infer those conservatively.
        split_axis = tuple(meta.get("split_axis") or range(min(3, len(axis_names))))
        runtime_names = tuple(
            f"{str(axis_names[index]).upper()}BLOCK"
            for index in split_axis[:3]
            if index < len(axis_names)
        )

    grids = []
    if runtime_names and axis_names:
        for block_name in runtime_names[:3]:
            axis = str(block_name).removesuffix("BLOCK").lower()
            if axis not in axis_names:
                raise RuntimeError(
                    f"Cannot map runtime block {block_name!r} to captured axes "
                    f"{axis_names!r}"
                )
            block = _standalone_repro_value(values, block_name)
            if block is None:
                raise RuntimeError(f"No runtime value found for {block_name!r}")
            block = int(block)
            if block <= 0:
                raise RuntimeError(f"Invalid runtime block {block_name}={block}")
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # A few older NPU captures did not expose runtime block argument names;
    # their GridNpu metadata instead records the split blocks directly.  Keep
    # this path independent of Inductor's GridExpr implementation.
    if grid_type == "GridNpu" and axis_names:
        split_axis = tuple(meta.get("split_axis") or range(len(axis_names)))
        split_blocks = meta.get("split_blocks")
        for position, axis_index in enumerate(split_axis[:3]):
            if not isinstance(axis_index, int) or axis_index >= len(axis_names):
                raise RuntimeError(
                    f"Invalid captured split axis {axis_index!r} for {axis_names!r}"
                )
            axis = str(axis_names[axis_index])
            numel = _standalone_repro_numel(values, axis)
            if isinstance(split_blocks, (tuple, list)) and position < len(split_blocks):
                block_expr = split_blocks[position]
                block = (
                    1
                    if block_expr is None
                    else _standalone_repro_grid_value(block_expr, values)
                )
            else:
                block = _standalone_repro_block(values, axis)
            if block <= 0:
                raise RuntimeError(f"Invalid split block for {axis}: {block}")
            grids.append(numel if block == 1 else (numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # Common upstream grid classes use X/Y/ZBLOCK constexprs and xnumel /
    # ynumel / znumel arguments.  This also makes old non-NPU Grid1D/2D/3D
    # captures useful without importing Inductor's grid classes.
    dimension_axes = {
        "Grid1D": ("x",),
        "Grid2D": ("x", "y"),
        "Grid3D": ("x", "y", "z"),
        "Grid2DWithYZOverflow": ("x", "y"),
    }.get(grid_type)
    if dimension_axes:
        for axis in dimension_axes:
            block = _standalone_repro_block(values, axis)
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        if grid_type == "Grid2DWithYZOverflow":
            grids.append(1)
        return tuple((grids + [1, 1, 1])[:3])

    return (1, 1, 1)


_STANDALONE_RUNTIME_BLOCKS = ()
_STANDALONE_INPUT_CAPTURE_MODE = 'metadata'
_STANDALONE_INPUT_METADATA = {'format_version': 3,
 'capture_mode': 'metadata',
 'args': {'kind': 'tuple',
          'items': [{'kind': 'tensor',
                     'dtype': 'torch.float16',
                     'device': 'npu:0',
                     'shape': (3072, 128, 144),
                     'stride': (18432, 144, 1),
                     'storage_offset': 0,
                     'requires_grad': False},
                    {'kind': 'tensor',
                     'dtype': 'torch.float16',
                     'device': 'npu:0',
                     'shape': (3072, 144, 160),
                     'stride': (38400, 160, 1),
                     'storage_offset': 0,
                     'requires_grad': False},
                    {'kind': 'tensor',
                     'dtype': 'torch.float16',
                     'device': 'npu:0',
                     'shape': (3072, 128, 160),
                     'stride': (20480, 160, 1),
                     'storage_offset': 0,
                     'requires_grad': False},
                    {'kind': 'literal', 'value': 28},
                    {'kind': 'literal', 'value': 1},
                    {'kind': 'literal', 'value': 1}]},
 'kwargs': {'kind': 'dict', 'items': []}}

def _standalone_repro_load_payload():
    return _STANDALONE_INPUT_METADATA




def _standalone_repro_resolve_dtype(dtype_name):
    import torch

    name = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"Cannot reconstruct dtype {dtype_name!r}")
    return dtype


def _standalone_repro_required_storage_size(shape, stride, storage_offset):
    if len(shape) != len(stride):
        raise RuntimeError(
            f"Captured shape/stride rank mismatch: {shape} vs {stride}"
        )
    if storage_offset < 0 or any(step < 0 for step in stride):
        raise RuntimeError(
            "Metadata replay does not support negative storage offsets or strides"
        )
    if any(size < 0 for size in shape):
        raise RuntimeError(f"Invalid captured shape: {shape}")
    if any(size == 0 for size in shape):
        return max(storage_offset + 1, 1)
    return storage_offset + sum(
        (size - 1) * step for size, step in zip(shape, stride)
    ) + 1


def _standalone_repro_random_storage(size, dtype, device):
    import torch

    try:
        if dtype == torch.bool:
            return torch.randint(0, 2, (size,), device=device, dtype=dtype)
        if dtype.is_floating_point or dtype.is_complex:
            return torch.randn((size,), device=device, dtype=dtype)
        return torch.randint(0, 2, (size,), device=device, dtype=dtype)
    except (RuntimeError, TypeError):
        return torch.zeros((size,), device=device, dtype=dtype)


def _standalone_repro_materialize(spec, map_location=None):
    import torch

    kind = spec.get("kind")
    if kind == "literal":
        return spec.get("value")
    if kind == "tuple":
        return tuple(
            _standalone_repro_materialize(item, map_location)
            for item in spec["items"]
        )
    if kind == "list":
        return [
            _standalone_repro_materialize(item, map_location)
            for item in spec["items"]
        ]
    if kind == "dict":
        return {
            _standalone_repro_materialize(key, map_location):
            _standalone_repro_materialize(value, map_location)
            for key, value in spec["items"]
        }
    if kind != "tensor":
        raise RuntimeError(f"Unknown captured input metadata kind: {kind!r}")

    dtype = _standalone_repro_resolve_dtype(spec["dtype"])
    device = map_location or spec["device"]
    shape = tuple(spec["shape"])
    stride = tuple(spec["stride"])
    storage_offset = int(spec["storage_offset"])
    storage_size = _standalone_repro_required_storage_size(
        shape,
        stride,
        storage_offset,
    )
    storage = _standalone_repro_random_storage(storage_size, dtype, device)
    tensor = torch.as_strided(storage, shape, stride, storage_offset)
    if spec.get("requires_grad") and (
        dtype.is_floating_point or dtype.is_complex
    ):
        tensor.requires_grad_(True)
    return tensor



def _standalone_repro_prepare_inputs():
    import os
    import torch

    payload = _standalone_repro_load_payload()
    capture_mode = payload.get("capture_mode", "legacy-values")
    map_location = os.environ.get("TRITON_REPRO_MAP_LOCATION") or None
    synthetic_seed = None
    if capture_mode == "metadata":
        synthetic_seed = int(os.environ.get("TRITON_REPRO_SEED", "0"))
        torch.manual_seed(synthetic_seed)
        args = tuple(_standalone_repro_materialize(payload["args"], map_location))
        kwargs = dict(_standalone_repro_materialize(payload["kwargs"], map_location))
        encoded_prefix = payload.get("launch_prefix")
        if encoded_prefix is not None:
            launch_prefix = tuple(
                _standalone_repro_materialize(encoded_prefix, map_location)
            )
        else:
            encoded_launch_args = payload.get("launch_args")
            if encoded_launch_args is not None:
                legacy_launch_args = tuple(
                    _standalone_repro_materialize(
                        encoded_launch_args,
                        map_location,
                    )
                )
                launch_prefix = _standalone_repro_legacy_prefix(
                    args,
                    legacy_launch_args,
                )
            else:
                launch_prefix = args
    else:
        args = tuple(payload["args"])
        kwargs = dict(payload["kwargs"])
        launch_prefix = payload.get("launch_prefix")
        if launch_prefix is None:
            legacy_launch_args = payload.get("launch_args")
            launch_prefix = _standalone_repro_legacy_prefix(
                args,
                tuple(legacy_launch_args) if legacy_launch_args is not None else None,
            )
        else:
            launch_prefix = tuple(launch_prefix)
    launch_args = (*tuple(launch_prefix), *_STANDALONE_RUNTIME_BLOCKS)
    return args, launch_args, kwargs, capture_mode, synthetic_seed


def _standalone_repro_main():
    import time
    import torch

    if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(_STANDALONE_DEVICE_INDEX)
    _args, raw_launch_args, kwargs, capture_mode, synthetic_seed = (
        _standalone_repro_prepare_inputs()
    )
    print(f"Compiling fixed config and launching once: {_STANDALONE_FN_NAME}")
    print(f"Config: {_STANDALONE_WANTED_CONFIG}")
    print(f"Runtime blocks: {_STANDALONE_RUNTIME_BLOCKS}")
    print(f"Input capture mode: {capture_mode}")
    if synthetic_seed is not None:
        print(f"Synthetic input seed: {synthetic_seed}")

    binary = _standalone_repro_compile()
    kernel_args, _launch_values = _standalone_repro_split_launch_args(
        raw_launch_args
    )
    grid = _standalone_repro_grid(raw_launch_args)
    print(f"Grid: {grid}")
    torch.npu.synchronize()
    start = time.perf_counter()
    # ``CompiledKernel.__getitem__`` accepts positional kernel arguments and
    # an optional raw ``stream``.  Inductor's generated launcher normally has
    # already consumed any other kwargs; fail explicitly if an old capture
    # contains one instead of silently passing it to the wrong ABI.
    launch_kwargs = dict(kwargs)
    stream = launch_kwargs.pop("stream", None)
    if stream is None:
        stream = _standalone_repro_current_stream()
    if launch_kwargs:
        raise RuntimeError(
            "Direct Triton runner cannot consume captured launcher kwargs: "
            f"{sorted(launch_kwargs)}"
        )
    binary[grid](*kernel_args, stream=stream)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"Kernel prerun succeeded in {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    _standalone_repro_main()
