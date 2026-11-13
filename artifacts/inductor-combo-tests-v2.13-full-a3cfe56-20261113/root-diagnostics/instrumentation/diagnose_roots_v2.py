"""Observe one existing Combo test without changing installed code or its assertions."""

import argparse
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
        "group": str(node.group),
        "min_order": node.min_order,
        "max_order": node.max_order,
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.test_root)
    assert Path("/.dockerenv").is_file()
    sys.path.insert(0, str(args.test_root.resolve()))
    filename, classname, name = args.node.split("::")
    module = importlib.import_module(filename.removesuffix(".py"))
    if classname == "CommonTemplate":
        classname = "GPUTests"
        name += "_" + module.GPU_TYPE
    import torch
    import torch_npu
    import triton
    import torch_npu._inductor
    from torch._inductor import config, metrics
    from torch._inductor.scheduler import Scheduler
    from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
    from torch_npu._inductor.codegen.triton_combo_kernel import NPUComboKernel

    assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
    assert torch_npu.version.git_version == "44555141711c0c825c5a201e303dcadad1f5fcb4"
    result = {
        "node": args.node, "diagnostic_only": True, "changes_original_results": False,
        "opened_software_gate": args.open_software_gate,
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "torch_npu": torch_npu.__version__, "torch_npu_git": torch_npu.version.git_version,
        "triton": triton.__version__, "events": [],
    }

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
        original_create = Scheduler.create_combo_kernel_nodes

        def create(scheduler, *pos, **kw):
            event("schedule-before-combo", nodes=[node_info(n) for n in scheduler.nodes])
            value = original_create(scheduler, *pos, **kw)
            event("schedule-after-combo", nodes=[node_info(n) for n in scheduler.nodes])
            return value

        stack.enter_context(patch.object(Scheduler, "create_combo_kernel_nodes", create))
        original_gate = Scheduler._try_combo_with_memory_check

        def gate(scheduler, group, ctx, *pos, **kw):
            value = original_gate(scheduler, group, ctx, *pos, **kw)
            event("memory-gate", nodes=[node_info(n) for n in group], accepted=value[0] is not None,
                  baseline=ctx.baseline_peak, running=ctx.running_peak,
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
        case = getattr(module, classname)(name)
        started = time.monotonic()
        case.setUp()
        try:
            method = getattr(case, name)
            if args.open_software_gate:
                assert getattr(method, "__unittest_skip__", False)
                method.__wrapped__(case)
            else:
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
    inputs = args.output / "diagnostic-inputs.pt"
    if inputs.is_file():
        result["inputs_sha256"] = hashlib.sha256(inputs.read_bytes()).hexdigest()
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["node", "outcome", "generated_kernel_count", "seconds"]}))


if __name__ == "__main__":
    main()
