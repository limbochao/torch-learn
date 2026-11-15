"""Observe one existing Combo test without changing installed code or its assertions."""

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    return str(value)


def node_info(node):
    return {
        "name": node.get_name(),
        "type": type(node).__name__,
        "group": str(getattr(node, "group", None)),
        "min_order": node.min_order,
        "max_order": node.max_order,
        "unmet_dependencies": [str(d) for d in node.unmet_dependencies],
        "origins": [
            {"name": origin.name, "target": str(origin.target)}
            for leaf in node.get_nodes()
            if leaf.node is not None
            for origin in leaf.node.origins
        ],
    }


def kernel_info(kernel):
    return {
        "type": type(kernel).__name__,
        "npu_kernel_type": str(kernel.npu_kernel_type),
        "inside_reduction": kernel.inside_reduction,
        "persistent_reduction": kernel.persistent_reduction,
        "axes": [
            {"name": a.name, "length": str(a.length), "split": a.is_split_axis,
             "tiling": a.is_tiling_axis, "no_loop": a.is_no_loop_axis, "reduction": a.is_reduction}
            for a in kernel.sorted_axis
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recorder", type=Path)
    parser.add_argument("--open-software-gate", action="store_true")
    parser.add_argument("--pytest", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.test_root)
    assert Path("/.dockerenv").is_file()
    sys.path.insert(0, str(args.test_root.resolve()))
    filename, classname, name = args.node.split("::")
    source_class, source_name = classname, name
    module = importlib.import_module(filename.removesuffix(".py"))
    if classname == "CommonTemplate":
        classname = "GPUTests"
        name += "_" + module.GPU_TYPE
    import torch
    import torch_npu
    import triton
    import torch_npu._inductor
    from torch._inductor import config, metrics
    from torch._inductor.scheduler import Scheduler, ForeachKernelSchedulerNode
    from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
    from torch_npu._inductor.codegen.triton_combo_kernel import NPUComboKernel

    assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
    assert torch_npu.version.git_version == "efc05dd90b5fe31717d5f13e51fb3e49035bea30"
    result = {
        "node": args.node, "diagnostic_only": True, "changes_original_results": False,
        "opened_software_gate": args.open_software_gate,
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "torch_npu": torch_npu.__version__, "torch_npu_git": torch_npu.version.git_version,
        "triton": triton.__version__, "events": [],
    }
    expected = json.loads(Path(__file__).with_name("expected.json").read_text())
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected["test_source_sha256"]
    for relative, digest in expected["overlay_sha256"].items():
        installed_path = Path(torch_npu.__file__).parent.parent / relative
        assert hashlib.sha256(installed_path.read_bytes()).hexdigest() == digest
    result["expected"] = expected
    result["physical_card"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    installed = list(Path(torch_npu.__file__).parent.joinpath("_inductor").rglob("*.py"))
    installed += list(Path(torch.__file__).parent.joinpath("_inductor").rglob("*.py"))
    installed_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in installed}

    def event(kind, **values):
        result["events"].append({"kind": kind, **json_value(values)})

    def observe_method(stack, cls, method_name, kind):
        original = getattr(cls, method_name)

        @functools.wraps(original)
        def observed(self, *pos, **kw):
            try:
                return original(self, *pos, **kw)
            except Exception as exc:
                values = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                if kind == "ttir-check-error":
                    number = sum(e["kind"] == kind for e in result["events"])
                    path = args.output / f"failed-ttir-{number}.py"
                    path.write_text(pos[0])
                    values["source"] = path.name
                event(kind, **values)
                raise

        stack.enter_context(patch.object(cls, method_name, observed))

    with contextlib.ExitStack() as stack:
        original_topo = Scheduler._topological_sort_nodes

        def topological(scheduler):
            groups = original_topo(scheduler)
            event("topological-waves", waves=[[node_info(n) for n in g] for g in groups])
            return groups

        stack.enter_context(patch.object(Scheduler, "_topological_sort_nodes", topological))
        original_combinable = ForeachKernelSchedulerNode.combinable_nodes.__func__

        def combinable(cls, nodes):
            members = original_combinable(cls, nodes)
            event("combinable-filter", before=[node_info(n) for n in nodes],
                  after=[n.get_name() for n in members],
                  pointwise_only=config.combo_kernels_pointwise_only)
            return members

        stack.enter_context(patch.object(ForeachKernelSchedulerNode, "combinable_nodes", classmethod(combinable)))
        original_windows = Scheduler._distance_windows

        def windows(*pos, **kw):
            for group in original_windows(*pos, **kw):
                event("distance-window", members=[n.get_name() for n in pos[0]],
                      window=[n.get_name() for n in group])
                yield group

        stack.enter_context(patch.object(Scheduler, "_distance_windows", staticmethod(windows)))
        if "max_persistent_rblock" in name:
            from torch_npu._inductor.runtime.combo_tile_generator import ComboTileGenerator
            from torch_npu._inductor.runtime.triton_heuristics import NPUCachingAutotuner
            from torch_npu._inductor import config as npu_config
            original_tiles = ComboTileGenerator.descend_split_tiling

            def tiles(generator):
                candidates = original_tiles(generator)
                event("combo-resource-candidates", subkernels=generator.subkernels,
                      kernel_type=generator.npu_kernel_type,
                      ub_size=npu_config.ub_size, dtype_bytes=generator.dtype_bytes,
                      input_ptr_num=generator.input_ptr_num,
                      max_numel_threshold=generator.max_numel_threshold,
                      max_programs=generator.max_programs,
                      candidates=[{"kwargs": c.kwargs,
                                   "resident_numel": generator.calculate_config_numel(c.kwargs)} for c in candidates])
                return candidates

            stack.enter_context(patch.object(ComboTileGenerator, "descend_split_tiling", tiles))
            original_best = NPUCachingAutotuner._set_best_candidate

            def best(autotuner, *pos, **kw):
                value = original_best(autotuner, *pos, **kw)
                launcher = pos[1]
                binary = getattr(launcher, "bin", None)
                if binary is None:
                    binary = getattr(launcher, "__globals__", {}).get("bin")
                metadata = getattr(binary, "metadata", None)
                event("resource-selected-candidate", candidate=pos[0],
                      metadata=metadata._asdict() if hasattr(metadata, "_asdict") else str(metadata),
                      launcher_attributes=sorted(vars(launcher)),
                      launcher_globals=sorted(getattr(launcher, "__globals__", {})))
                return value

            stack.enter_context(patch.object(NPUCachingAutotuner, "_set_best_candidate", best))

        original_peak = torch.npu.max_memory_allocated

        def measured_peak(*pos, **kw):
            value = original_peak(*pos, **kw)
            event("runtime-peak", bytes=value)
            return value

        stack.enter_context(patch.object(torch.npu, "max_memory_allocated", measured_peak))
        original_create = Scheduler.create_combo_kernel_nodes

        def create(scheduler, *pos, **kw):
            event("schedule-before-combo", nodes=[node_info(n) for n in scheduler.nodes])
            value = original_create(scheduler, *pos, **kw)
            event("schedule-after-combo", nodes=[node_info(n) for n in scheduler.nodes])
            return value

        stack.enter_context(patch.object(Scheduler, "create_combo_kernel_nodes", create))
        for benchmark_method in ("speedup_by_combo_kernel", "benchmark_fused_nodes", "benchmark_combo_kernel"):
            original_benchmark = getattr(Scheduler, benchmark_method)

            def benchmark(scheduler, *pos, _original=original_benchmark, _name=benchmark_method, **kw):
                value = _original(scheduler, *pos, **kw)
                event(_name, value=value)
                return value

            stack.enter_context(patch.object(Scheduler, benchmark_method, benchmark))
        original_gate = Scheduler._try_combo_with_memory_check
        from torch._inductor import memory
        original_region_peak = memory.estimate_region_peak_memory
        region_peaks = []

        def region_peak(*pos, **kw):
            value = original_region_peak(*pos, **kw)
            region_peaks.append(value)
            return value

        stack.enter_context(patch.object(memory, "estimate_region_peak_memory", region_peak))

        def gate(scheduler, group, ctx, *pos, **kw):
            previous_peak = ctx.running_peak
            value = original_gate(scheduler, group, ctx, *pos, **kw)
            event("memory-gate", nodes=[node_info(n) for n in group], accepted=value[0] is not None,
                  baseline=ctx.baseline_peak, running=ctx.running_peak,
                  region_peak=region_peaks[-1], delta=max(previous_peak, region_peaks[-1]) - ctx.baseline_peak,
                  abs_threshold=config.combo_kernel_peak_memory_increase_gb,
                  pct_threshold=config.combo_kernel_peak_memory_pct_threshold)
            return value

        stack.enter_context(patch.object(Scheduler, "_try_combo_with_memory_check", gate))
        original_partition = NPUComboKernel.horizontal_partition

        def partition(nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm=False):
            event("partition-input", members=[{"node": node_info(n), "kernel": kernel_info(kernel_map[n])}
                                               for n in nodes])
            value = original_partition(nodes, triton_scheduling, kernel_map, node_info_map, custom_algorithm)
            event("partition-output", groups=[[n.get_name() for n in group] for group in value])
            return value

        stack.enter_context(patch.object(NPUComboKernel, "horizontal_partition", staticmethod(partition)))
        observe_method(stack, NPUTritonScheduling, "generate_combo_kernel_code", "combo-codegen-error")
        observe_method(stack, NPUTritonScheduling, "make_ttir_for_check", "ttir-check-error")
        observe_method(stack, NPUTritonScheduling, "codegen_node", "index-node-codegen-error")
        original_define = NPUTritonScheduling.define_kernel

        def define(scheduling, source, schedule, kernel, *pos, **kw):
            value = original_define(scheduling, source, schedule, kernel, *pos, **kw)
            number = sum(e["kind"] == "defined-kernel" for e in result["events"])
            path = args.output / f"defined-{number}.py"
            path.write_text(source)
            event("defined-kernel", name=value[0], source=path.name, nodes=[node_info(n) for n in schedule],
                  kernel=kernel_info(kernel) if hasattr(kernel, "sorted_axis") else type(kernel).__name__)
            return value

        stack.enter_context(patch.object(NPUTritonScheduling, "define_kernel", define))
        if hasattr(module, "run_and_get_code"):
            original_get_code = module.run_and_get_code

            def get_code(*pos, **kw):
                value, sources = original_get_code(*pos, **kw)
                for number, source in enumerate(sources):
                    (args.output / f"wrapper-{number}.py").write_text(source)
                return value, sources

            stack.enter_context(patch.object(module, "run_and_get_code", get_code))

        if name == "test_combo_kernel_per_config_subkernel_r0_r1":
            original_compile = torch.compile

            def compile_fn(fn, *pos, **kw):
                compiled = original_compile(fn, *pos, **kw)

                def call(*inputs, **kwargs):
                    cpu_inputs = [t.detach().cpu() for t in inputs]
                    torch.save(cpu_inputs, args.output / "diagnostic-inputs.pt")
                    eager = fn(*inputs, **kwargs)
                    actual = compiled(*inputs, **kwargs)
                    records = []
                    for inp, expected, output in zip(cpu_inputs, eager, actual):
                        expected, output = expected.cpu(), output.cpu()
                        reference = inp.double().sum((1, 2))
                        records.append({
                            "shape": list(inp.shape), "stride": list(inp.stride()), "dtype": str(inp.dtype),
                            "eager": expected.tolist(), "compiled": output.tolist(),
                            "reference_float64": reference.tolist(),
                            "max_eager_reference_error": (expected.double() - reference).abs().max().item(),
                            "max_compiled_reference_error": (output.double() - reference).abs().max().item(),
                            "max_eager_compiled_error": (output - expected).abs().max().item(),
                        })
                    result["accuracy"] = records
                    torch.save({"eager": [t.cpu() for t in eager], "compiled": [t.cpu() for t in actual]},
                               args.output / "diagnostic-outputs.pt")
                    return actual

                return call

            stack.enter_context(patch.object(torch, "compile", compile_fn))

        recorder = None
        if args.recorder:
            spec = importlib.util.spec_from_file_location("autotune_tiling", args.recorder)
            recorder_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(recorder_module)
            recorder = recorder_module.BestTilingRecorder("npu")
            recorder.install()
            stack.callback(recorder.uninstall)
            recorder.start_capture()
        case_class = getattr(module, classname)
        if args.open_software_gate:
            assert source_name in {"test_foreach_basic", "test_foreach_after_extern", "test_combo_kernel_store_mask",
                                   "test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu"}
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8-sig"))
            source_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source_class)
            fn = next(n for n in source_cls.body if isinstance(n, ast.FunctionDef) and n.name == source_name)
            kept_decorators = []
            for decorator in fn.decorator_list:
                text = ast.unparse(decorator)
                if text == "requires_xpu_and_triton" or text.startswith("unittest.skip("):
                    continue
                kept_decorators.append(decorator)
            fn.decorator_list = kept_decorators
            compiled_fn = compile(ast.Module(body=[fn], type_ignores=[]), module.__file__, "exec")
            namespace = {}
            exec(compiled_fn, module.__dict__, namespace)
            stack.enter_context(patch.object(case_class, name, namespace[source_name]))
        case = case_class(name)
        started = time.monotonic()
        if args.pytest:
            import pytest

            def snapshot():
                result["generated_kernel_count"] = metrics.generated_kernel_count
                if hasattr(module, "metrics"):
                    result["input_distances"] = [
                        {"node": node_info(s), "min": s.min_input_distance, "max": s.max_input_distance}
                        for s, _ in module.metrics.nodes_num_elem
                    ]

            original_teardown = case_class.tearDown

            def teardown(instance):
                snapshot()
                return original_teardown(instance)

            stack.enter_context(patch.object(case_class, "tearDown", teardown))
            actual_node = f"{filename}::{classname}::{name}"
            result["pytest_node"] = actual_node

            class PytestObserver:
                def pytest_collection_modifyitems(self, items):
                    for item in items:
                        actual_class = item.cls
                        actual_teardown = actual_class.tearDown

                        def collected_teardown(instance, _original=actual_teardown):
                            actual_module = sys.modules[type(instance).__module__]
                            result["generated_kernel_count"] = metrics.generated_kernel_count
                            if hasattr(actual_module, "metrics"):
                                result["input_distances"] = [
                                    {"node": node_info(s), "min": s.min_input_distance, "max": s.max_input_distance}
                                    for s, _ in actual_module.metrics.nodes_num_elem
                                ]
                            return _original(instance)

                        stack.enter_context(patch.object(actual_class, "tearDown", collected_teardown))
                        if args.open_software_gate:
                            actual_module = sys.modules[actual_class.__module__]
                            local_namespace = {}
                            exec(compiled_fn, actual_module.__dict__, local_namespace)
                            function = local_namespace[source_name]
                            stack.enter_context(patch.object(actual_class, name, function))
                            bound = function.__get__(item.instance, actual_class)
                            stack.enter_context(patch.object(item.instance, name, bound))
                            item._obj = bound

            rc = pytest.main(["-vs", "-ra", "--color=no", "--tb=short", actual_node,
                              "--junitxml=" + str(args.output / "diagnostic.xml")], plugins=[PytestObserver()])
            result["outcome"] = "completed" if rc == 0 else "pytest-nonzero"
            result["pytest_returncode"] = int(rc)
            result.setdefault("generated_kernel_count", metrics.generated_kernel_count)
            if recorder:
                result["selected_tiling"] = recorder.stop_capture()
            result["seconds"] = round(time.monotonic() - started, 2)
            result["installed_source_unchanged"] = all(
                hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in installed_before.items()
            )
            result["installed_source_files"] = len(installed_before)
            (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return
        case.setUp()
        try:
            method = getattr(case, name)
            method()
            result["outcome"] = "completed"
        except Exception as exc:
            result.update(outcome="exception", error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
        finally:
            result["generated_kernel_count"] = metrics.generated_kernel_count
            if hasattr(module, "metrics"):
                result["input_distances"] = [
                    {"node": node_info(s), "min": s.min_input_distance, "max": s.max_input_distance}
                    for s, _ in module.metrics.nodes_num_elem
                ]
            if recorder:
                result["selected_tiling"] = recorder.stop_capture()
            case.tearDown()
        result["seconds"] = round(time.monotonic() - started, 2)
    result["installed_source_unchanged"] = all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in installed_before.items()
    )
    result["installed_source_files"] = len(installed_before)
    inputs = args.output / "diagnostic-inputs.pt"
    if inputs.is_file():
        result["inputs_sha256"] = hashlib.sha256(inputs.read_bytes()).hexdigest()
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["node", "outcome", "generated_kernel_count", "seconds"]}))


if __name__ == "__main__":
    main()
