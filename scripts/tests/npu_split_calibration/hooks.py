"""Process-local IR and runtime observation; no installed files are modified."""

from __future__ import annotations

import functools
import inspect
import math
import sys
from contextlib import contextmanager

from common import recorder_path
from policy import SUPPORTED, admit, candidate


def install_ordinary_lowerings():
    """Test prerequisite: bypass explicit fallback entries, never installed files.

    This exposes the existing ordinary IR/codegen for num_splits calibration.
    It is a distinct experiment profile, not the installed production behavior.
    """
    import torch
    from torch._inductor import lowering

    restored = []
    for packet, function in ((torch.ops.aten.prod, lowering.prod), (torch.ops.aten.any, lowering.reduce_any)):
        for name in packet.overloads():
            if name in ("default", "dim_int", "dim", "dims"):
                op = getattr(packet, name)
                lowering.lowerings[op] = function
                lowering.needs_realized_inputs.discard(op)
                restored.append(str(op))
    # NPU backend activation can replace the lowerings dictionary. The default
    # argument of register_lowering can still refer to the old dictionary.
    xor_lowering = lowering.register_lowering(torch.ops.prims.xor_sum.default, lowering_dict=lowering.lowerings)(
        lowering.make_reduction("xor_sum"))
    for op in (torch.ops.prims.xor_sum, torch.ops.prims.xor_sum.default):
        lowering.lowerings[op] = xor_lowering
        lowering.needs_realized_inputs.discard(op)
    restored.append(str(torch.ops.prims.xor_sum.default))
    return sorted(restored)


def reduction_meta(bound):
    import sympy
    from torch._inductor import dependencies, ir
    from torch._inductor.virtualized import V

    ranges, rranges = bound["ranges"], bound["reduction_ranges"]
    xexpr, rexpr = ir.sympy_product(ranges), bound["reduction_numel"]
    sizevars = V.graph.sizevars
    unbacked = getattr(sizevars, "all_unbacked_explicitly_hinted", None)
    if unbacked is None:
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
        hinted = not free_unbacked_symbols([xexpr, rexpr])
    else:
        hinted = unbacked([xexpr, rexpr])
    meta = dict(op=bound["reduction_type"], src_dtype=str(bound["src_dtype"]).replace("torch.", ""),
                dst_dtype=str(bound["dst_dtype"]).replace("torch.", ""),
                X=int(sizevars.optimization_hint(xexpr)) if hinted else 0,
                R=int(sizevars.optimization_hint(rexpr)) if hinted else 0,
                x_expr=str(xexpr), r_expr=str(rexpr), hinted=hinted,
                dynamic=any(not isinstance(d, (int, sympy.Integer)) for d in (*ranges, *rranges)),
                layout="unknown", reads=[])
    if meta["op"] not in SUPPORTED or not hinted:
        return meta
    try:
        reduction = ir.Reduction(device=bound["device"], dtype=bound["dst_dtype"],
                                 inner_fn=bound["inner_fn"], ranges=ranges, reduction_ranges=rranges,
                                 reduction_type=meta["op"], src_dtype=bound["src_dtype"],
                                 reduction_hint=ir.ReductionHint.DEFAULT)
        buffer = ir.ComputedBuffer(name=None, layout=ir.FlexibleLayout(
            device=bound["device"], dtype=bound["dst_dtype"], size=ranges), data=reduction)
        rw = buffer.get_read_writes()
        (xvars, rvars), var_ranges = dependencies.index_vars_squeeze(ranges, rranges)
        xvars = [v for v in xvars if isinstance(v, sympy.Symbol)]
        rvars = [v for v in rvars if isinstance(v, sympy.Symbol)]
        kinds = []
        for dep in sorted(rw.reads, key=lambda d: d.name):
            if not hasattr(dep, "index"):
                continue
            index = sizevars.simplify_with_ranges(dep.index, var_ranges)
            rs = [int(s) for s in sizevars.stride_hints(index, rvars, list(var_ranges))]
            xs = [int(s) for s in sizevars.stride_hints(index, xvars, list(var_ranges))]
            meta["reads"].append(dict(name=dep.name, index=str(index), x_strides=xs, r_strides=rs))
            if not any(rs):
                continue  # broadcast loads alone cannot prove a contiguous reduction
            kinds.append("inner" if 1 in rs else "outer" if 1 in xs else "strided")
        if kinds:
            meta["layout"] = kinds[0] if len(set(kinds)) == 1 else "mixed"
    except Exception as exc:
        meta["read_analysis_error"] = f"{type(exc).__name__}: {exc}"
    return meta


class SplitController:
    def __init__(self, mode, settings, cores, table=None):
        self.mode, self.settings, self.cores = mode, settings, cores
        self.table = table or {}
        self.decisions, self.splits = [], []
        self.depth = 0
        self.compile_id = 0

    @contextmanager
    def install(self):
        from torch._inductor import ir
        from torch._inductor.codegen.common import BackendFeature
        from torch._inductor.virtualized import V

        original_descriptor = inspect.getattr_static(ir.Reduction, "num_splits")
        original = ir.Reduction.num_splits
        signature = inspect.signature(original)
        saved = []

        def choose(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if values["device"].type != "npu":
                return original(*args, **kwargs)
            # Installed mode is observational: do not execute speculative read
            # analysis before its original selector (layout analysis can realize IR).
            if self.mode == "policy" and self.settings["policy"] == "installed":
                hint, split = original(*args, **kwargs)
                meta = dict(op=values["reduction_type"], ranges=str(values["ranges"]),
                            reduction_ranges=str(values["reduction_ranges"]))
                reason = "installed"
            else:
                meta = reduction_meta(values)
                hint = ir.ReductionHint.INNER if meta["layout"] == "inner" else (
                    ir.ReductionHint.OUTER if meta["layout"] == "outer" else ir.ReductionHint.DEFAULT)
                # plain and candidate have the SAME hint classifier.
                split, reason = candidate(meta, self.cores, self.settings["selector"])
                if self.mode == "plain":
                    split, reason = 1, "forced_plain"
                elif self.depth:
                    split, reason = 1, "nested_split_disabled"
                elif V.graph.has_feature(values["device"], BackendFeature.REDUCE_TO_SINGLE_ELEMENT):
                    split, reason = 1, "backend_reduce_to_single_element"
                elif self.mode == "policy" and not admit(meta, split, self.table):
                    split, reason = 1, "outside_admission_table"
            self.decisions.append(dict(compile_id=self.compile_id, depth=self.depth,
                                       meta=meta, S=str(split), hint=str(hint), reason=reason))
            return hint, split

        def observe(function, cls_name, method):
            sig = inspect.signature(function)

            @functools.wraps(function)
            def wrapper(cls, *args, **kwargs):
                values = sig.bind(cls, *args, **kwargs).arguments
                event = dict(compile_id=self.compile_id, depth=self.depth, ir_class=cls_name, method=method,
                             arguments={k: str(v) for k, v in values.items() if k in (
                                 "ranges", "reduction_ranges", "original_ranges", "new_ranges",
                                 "new_reduction_ranges", "split", "reduction_type", "dst_dtype", "src_dtype")})
                if "split" in values and "ranges" in values:
                    from torch._inductor.virtualized import V
                    count = ir.sympy_product(values["ranges"]) * values["split"]
                    dtype = values.get("dst_dtype")
                    partial = "float32" if str(dtype) in ("torch.float16", "torch.bfloat16") else str(dtype)
                    event.update(partial_numel_expr=str(count), partial_dtype_expected=partial,
                                 partial_numel_hint=int(V.graph.sizevars.optimization_hint(count)))
                self.splits.append(event)
                self.depth += 1
                try:
                    return function(cls, *args, **kwargs)
                finally:
                    self.depth -= 1
            return classmethod(wrapper)

        ir.Reduction.num_splits = staticmethod(choose)
        try:
            for name in ("Reduction", "WelfordReduction", "OnlineSoftmaxReduction"):
                cls = getattr(ir, name, None)
                if cls is None:
                    continue
                for method in ("create_multilayer", "create_multilayer_existing_ranges"):
                    descriptor = cls.__dict__.get(method)
                    if isinstance(descriptor, classmethod):
                        saved.append((cls, method, descriptor))
                        setattr(cls, method, observe(descriptor.__func__, name, method))
            yield
        finally:
            ir.Reduction.num_splits = original_descriptor
            for cls, method, descriptor in reversed(saved):
                setattr(cls, method, descriptor)


def make_recorder():
    """Reuse the repository's selected-tiling recorder; observe actual grid too."""
    sys.path.insert(0, str(recorder_path().parent))
    from autotune_tiling import BestTilingRecorder, _config_payload

    class Recorder(BestTilingRecorder):
        def __init__(self):
            super().__init__("npu")
            self.launches = []

        def install(self):
            # Older torch_npu releases do not have grouped autotuning.
            import torch_npu._inductor.runtime.triton_heuristics as runtime
            for name in ("NPUCachingAutotuner", "NPUSymbolicGroupedAutotuner"):
                cls = getattr(runtime, name, None)
                if cls is not None:
                    self._patch_class(cls, grouped=name == "NPUSymbolicGroupedAutotuner")

        def _patch_class(self, cls, grouped):
            original = cls.run
            recorder = self

            @functools.wraps(original)
            def run(autotuner, *args, **kwargs):
                result = original(autotuner, *args, **kwargs)
                if recorder._capturing:
                    if grouped:
                        recorder._record_grouped(autotuner, args)
                    else:
                        recorder._record_standard(autotuner)
                    recorder.record_launch(autotuner, args, grouped)
                return result
            cls.run = run
            self._patches.append((cls, original))

        def record_launch(self, tuner, args, grouped):
            record = dict(kernel_name=(getattr(tuner, "inductor_meta", {}) or {}).get("kernel_name", ""))
            launcher = getattr(tuner, "best_launcher", None)
            config = getattr(tuner, "best_candidate_config", None)
            blocks = getattr(tuner, "best_runtime_blocks", ())
            try:
                if grouped:
                    group = tuner._resolve_group_id(tuner._runtime_feature_inputs(args))
                    item = tuner.best_candidate_map[group]
                    launcher = tuner.best_launcher_map[group]
                    blocks = tuner._materialize_runtime_blocks(item, args)
                    from torch_npu._inductor.runtime.triton_heuristics import config_from_dict
                    config = config_from_dict(item["full_config"])
                    record["group_id"] = group
                if launcher is None:
                    launchers = getattr(tuner, "launchers", ())
                    launcher = launchers[0] if len(launchers) == 1 else None
                config = config or getattr(launcher, "config", None)
                record["selected_config"] = _config_payload(config)
                record["runtime_blocks"] = self._runtime_blocks_payload(tuner, blocks)
                interpreter = tuner._interpret_args_grid
                if "runtime_blocks" in inspect.signature(interpreter).parameters:
                    _, grid = interpreter(args, config, blocks)
                else:
                    _, grid = interpreter(args, config)
                record["grid"] = [int(value) for value in grid]
                record["programs"] = math.prod(record["grid"])
            except Exception as exc:
                record["grid_error"] = f"{type(exc).__name__}: {exc}"
            record["fallback"] = bool(getattr(launcher, "fallback", False))
            self.launches.append(record)

    return Recorder()
