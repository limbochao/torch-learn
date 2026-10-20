# Standalone single-kernel compile reproducer.
# Extracted by repro_compile.py for triton_per_fused__to_copy_add_embedding_mul_na_1.
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
def triton_per_fused__to_copy_add_embedding_mul_na_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr4, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 2048
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0 = tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = r0 < min(R0BLOCK + r0_offset, r0_numel)
    tmp0 = tl.load(in_ptr0 + tl.arange(0, 1) + 63)
    tmp1 = tl.reshape(tmp0, [1])
    tmp4 = tl.load(in_ptr2 + tl.arange(0, 1) + 63)
    tmp5 = tl.reshape(tmp4, [1])
    tmp7 = tl.load(in_ptr3 + tl.arange(0, 1) + 63)
    tmp8 = tl.reshape(tmp7, [1])
    tmp29 = tl.load(in_ptr5 + r0, r0_mask, other=0.0)
    tmp31 = tl.load(in_ptr6 + r0, r0_mask, other=0.0)
    tl.device_assert((0 <= tmp1) & (tmp1 < 2048), 'index out of bounds: 0 <= tmp1 < 2048')
    tmp3 = tl.load(in_ptr1 + (r0 + 1536 * tmp1), r0_mask, other=0.0)
    tmp6 = tmp5.to(tl.int32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 2047, tl.int64)
    tmp12 = tmp10 + tmp11
    tl.device_assert((0 <= tmp12) & (tmp12 < 4096), 'index out of bounds: 0 <= tmp12 < 4096')
    tmp14 = tl.load(in_ptr4 + (r0 + 1536 * tmp12), r0_mask, other=0.0)
    tmp15 = tmp3 + tmp14
    tmp17 = _standalone_triton_promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.full([1], 1536.0, tl.float32)
    tmp19 = tmp17 / tmp18
    tmp20 = tmp15 - tmp19
    tmp21 = tmp20 * tmp20
    tmp23 = _standalone_triton_promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp23 / tmp18
    tmp25 = tl.full([1], 1e-05, tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp20 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + r0, tmp15, r0_mask)
    tl.store(out_ptr4 + r0, tmp32, r0_mask)


# ---- Triton helper shims (copied into the standalone file) ----
@triton.jit
def _standalone_triton_promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def _standalone_triton_is_floating(x):
    return _standalone_triton_promote_to_tensor(x).dtype.is_floating()


@triton.jit
def _standalone_triton_minimum(a, b):
    mask = a < b
    if _standalone_triton_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def _standalone_triton_maximum(a, b):
    mask = a > b
    if _standalone_triton_is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def _standalone_triton_min2(a, dim):
    return tl.reduce(a, dim, _standalone_triton_minimum)


@triton.jit
def _standalone_triton_max2(a, dim):
    return tl.reduce(a, dim, _standalone_triton_maximum)


@triton.jit
def _standalone_triton_minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if _standalone_triton_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _standalone_triton_maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if _standalone_triton_is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def _standalone_triton_restore_reduced_dim(
    reduced, dim: tl.constexpr, ndim: tl.constexpr
):
    if ndim == 1:
        return _standalone_triton_promote_to_tensor(reduced)
    return tl.expand_dims(reduced, dim)


@triton.jit
def _standalone_triton_extremum(value, dim: tl.constexpr, want_max: tl.constexpr):
    if value.dtype.is_floating():
        if want_max:
            return tl.max(value, dim)
        return tl.min(value, dim)
    if want_max:
        return tl.reduce(value, dim, _standalone_triton_maximum)
    return tl.reduce(value, dim, _standalone_triton_minimum)


@triton.jit
def _standalone_triton_max_with_index(value, index, dim: tl.constexpr):
    peak = _standalone_triton_extremum(value, dim, True)
    peak_bcast = _standalone_triton_restore_reduced_dim(
        peak, dim, len(value.shape)
    )
    filler = _standalone_triton_restore_reduced_dim(
        tl.max(index, dim), dim, len(value.shape)
    )
    return peak, tl.min(
        tl.where(value == peak_bcast, index, filler),
        dim,
    )


@triton.jit
def _standalone_triton_min_with_index(value, index, dim: tl.constexpr):
    valley = _standalone_triton_extremum(value, dim, False)
    valley_bcast = _standalone_triton_restore_reduced_dim(
        valley, dim, len(value.shape)
    )
    filler = _standalone_triton_restore_reduced_dim(
        tl.max(index, dim), dim, len(value.shape)
    )
    return valley, tl.min(
        tl.where(value == valley_bcast, index, filler),
        dim,
    )

# ---- direct standalone reproducer helpers ----
_STANDALONE_FN_NAME = 'triton_per_fused__to_copy_add_embedding_mul_na_1'
_STANDALONE_WANTED_CONFIG = {'R0BLOCK_SUB': 1536, 'split_axis': [0], 'num_warps': 1, 'num_stages': 2}
_STANDALONE_SIGNATURE = {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr4': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}
_STANDALONE_CONSTANTS = {}
_STANDALONE_ATTRS = {(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}
_STANDALONE_TARGET = {'backend': 'npu', 'arch': 'Ascend950PR_9579', 'warp_size': 32}
_STANDALONE_DEVICE_INDEX = 0
_STANDALONE_OPTIONS = {'num_warps': 1, 'num_stages': 2, 'debug': False, 'compile_mode': 'simd_simt_template'}
_STANDALONE_INDUCTOR_META = {'grid_type': 'GridNpu', 'split_axis': [0], 'axis_names': ['r0'], 'runtime_block_arg_names': ['R0BLOCK'], 'group_enabled': False}
_STANDALONE_LAUNCH_ARG_NAMES = ['in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'in_ptr5', 'in_ptr6', 'out_ptr0', 'out_ptr4', 'r0_numel', 'R0BLOCK']
_STANDALONE_RUNTIME_BLOCK_NAMES = ['R0BLOCK']
_STANDALONE_BASE_LAUNCH_ARG_NAMES = ['in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'in_ptr5', 'in_ptr6', 'out_ptr0', 'out_ptr4', 'r0_numel']
_STANDALONE_EXTRA_LAUNCH_ARG_NAMES = ()


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



def _standalone_repro_main():
    print(f"Compiling fixed config only: {_STANDALONE_FN_NAME}")
    print(f"Config: {_STANDALONE_WANTED_CONFIG}")
    _standalone_repro_compile()
    print("Kernel compiled successfully; the original compiler failure did not reproduce.")


if __name__ == "__main__":
    _standalone_repro_main()
