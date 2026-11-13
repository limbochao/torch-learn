import argparse
import contextlib
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from unittest.mock import patch


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["foreach", "ftz", "wrn"])
parser.add_argument("--cases", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "inductor-cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(output / "triton-cache"))
sys.path.insert(0, args.cases)

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()
torch.cuda.get_device_capability = lambda: (10, 0)
import torch_npu.testing
import torch_npu._inductor
import triton
from torch._inductor import config, metrics
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V

torch.npu.set_device(0)
torch.manual_seed(11213)
torch.npu.manual_seed_all(11213)
result = {
    "started": time.time(),
    "mode": args.mode,
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "torch_npu": torch_npu.__version__,
    "triton": triton.__version__,
    "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    "logical_device": 0,
    "master_device_retest": False,
    "source_hashes": {},
    "records": [],
}
for module_name in [
    "torch._inductor.scheduler", "torch._inductor.memory",
    "torch_npu._inductor.codegen.scheduling", "torch_npu._inductor.codegen.triton_combo_kernel",
    "torch_npu._inductor.runtime.triton_heuristics", "triton.backends.ascend.compiler",
]:
    module = importlib.import_module(module_name)
    file = Path(inspect.getsourcefile(module))
    result["source_hashes"][module_name] = hashlib.sha256(file.read_bytes()).hexdigest()


def write_result():
    result["finished"] = time.time()
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def snode_record(node):
    leaves = node.get_nodes()
    return {
        "name": node.get_name(),
        "type": type(node).__name__,
        "min_distance": node.min_input_distance,
        "max_distance": node.max_input_distance,
        "unmet_dependencies": [str(d) for d in node.unmet_dependencies],
        "leaves": [
            {"name": n.get_name(), "type": type(n).__name__,
             "ir": type(n.node).__name__, "min": n.min_input_distance,
             "max": n.max_input_distance} for n in leaves
        ],
    }


def run_foreach():
    tests = importlib.import_module("test_snode_runtime")
    original = tests.InputDistanceTests._get_snodes
    for name in ["test_foreach_basic", "test_foreach_after_extern"]:
        record = {"test": name, "bypassed_skip": True, "observations": []}

        def collect(owner, function, *inputs):
            expected = function(*inputs)
            nodes = original(owner, function, *inputs)
            assert nodes, "Input-distance test requires a nonempty scheduler observation"
            record["observations"] = [snode_record(node) for node in nodes]
            actual = torch.compile(function, backend=tests.compile_but_use_eager)(*inputs)
            torch.testing.assert_close(actual, expected)
            torch.npu.synchronize()
            record["eager_return_correctness"] = True
            return nodes

        test = tests.InputDistanceTests(methodName=name)
        try:
            test.setUp()
            with patch.object(tests.InputDistanceTests, "_get_snodes", collect):
                method = getattr(test, name)
                assert getattr(method, "__unittest_skip__", False)
                inspect.unwrap(getattr(tests.InputDistanceTests, name))(test)
            record["status"] = "PASS_DIAGNOSTIC"
        except Exception:
            record["status"] = "FAIL_DIAGNOSTIC"
            record["error"] = traceback.format_exc()
        finally:
            test.tearDown()
        result["records"].append(record)
        write_result()
        print("FOREACH_RECORD=" + json.dumps(record, ensure_ascii=False), flush=True)


def fp32_bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int32).tolist()


def run_ftz():
    import triton.language as tl
    from torch._inductor.utils import run_and_get_code
    from triton.backends.ascend.compiler import AscendBackend
    from triton.backends.compiler import GPUTarget
    backend = AscendBackend(triton.runtime.driver.active.get_current_target())
    for options in [{}, {"disable_ftz": True}, {"enable_reflect_ftz": False}]:
        parsed = backend.parse_options(options.copy())
        result.setdefault("compiler_options", []).append({
            "requested": options,
            "registered_ftz_fields": [name for name in parsed.__dataclass_fields__
                                      if any(word in name for word in ["ftz", "denorm", "subnormal"])],
            "parsed_hash": parsed.hash(),
        })

    # Construct bits on CPU, then copy unchanged. Values cover input subnormals,
    # normal inputs producing a subnormal, both signs, zero, and a normal control.
    bits = torch.tensor([1, 0x00400000, 0x007fffff, 0x00800000, 0x01000000,
                         0x3f800000, -2147483647, -2143289344, 0], dtype=torch.int32)
    x_cpu = bits.view(torch.float32).repeat(128)
    x = x_cpu.to("npu")
    small_normal = torch.full_like(x, 2.0 ** -126)

    def pointwise(a, b):
        return a + a, a * 0.5, a * 2.0, b * 0.5

    def reductions(a):
        return a.reshape(128, 9).sum(-1), a.reshape(128, 9).sum(0)

    def exponentials(a):
        return torch.exp2(a), torch.exp(a * 0.6931471805599453)

    exponents = torch.tensor([-127.0, -128.0, -126.0, -125.0, 0.0, 1.0,
                              -149.0, -150.0, -136.0], device="npu").repeat(128)

    result["input_copy_bits_preserved"] = fp32_bits(x) == fp32_bits(x_cpu)
    for flag in [False, True]:
        torch._dynamo.reset()
        metrics.reset()
        with config.patch({"combo_kernels": True, "benchmark_combo_kernel": False,
                           "combo_kernel_per_subkernel_blocks": False,
                           "combo_kernel_max_distance": -1,
                           "combo_kernel_peak_memory_increase_gb": None,
                           "combo_kernel_peak_memory_pct_threshold": None,
                           "eager_numerics.disable_ftz": flag}):
            for label, function, inputs in [("pointwise", pointwise, (x, small_normal)),
                                             ("reductions", reductions, (x,)),
                                             ("exponentials", exponentials, (exponents,))]:
                record = {"disable_ftz": flag, "graph": label}
                expected = function(*inputs)
                cpu_expected = function(*(a.cpu() for a in inputs))
                actual, sources = run_and_get_code(torch.compile(function), *inputs)
                torch.npu.synchronize()
                record.update({
                    "eager_bits": [fp32_bits(a)[:18] for a in expected],
                    "compiled_bits": [fp32_bits(a)[:18] for a in actual],
                    "cpu_bits": [fp32_bits(a)[:18] for a in cpu_expected],
                    "bitwise_eager_compiled": [torch.equal(a.cpu().view(torch.int32), b.cpu().view(torch.int32))
                                               for a, b in zip(expected, actual)],
                    "bitwise_cpu_compiled": [torch.equal(a.cpu().view(torch.int32), b.cpu().view(torch.int32))
                                             for a, b in zip(cpu_expected, actual)],
                    "metadata_has_disable_ftz": any("disable_ftz" in s for s in sources),
                    "combo_decorators": sum(s.count("@triton_heuristics.combo_kernel") for s in sources),
                })
                for index, source in enumerate(sources):
                    path = output / f"ftz-{flag}-{label}-{index}.py"
                    path.write_text(source)
                result["records"].append(record)
                write_result()
                print("FTZ_RECORD=" + json.dumps(record, ensure_ascii=False), flush=True)


def run_wrn():
    tests = importlib.import_module("test_combo_kernels")
    from torch._inductor import memory as memory_module
    from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
    original_create = Scheduler.create_combo_kernel_nodes
    original_init = Scheduler._init_peak_memory_context
    original_try = Scheduler._try_combo_with_memory_check
    original_region = memory_module.estimate_region_peak_memory
    original_emit = NPUTritonScheduling.codegen_combo_kernel
    active = {}

    def summarize(node):
        return {"name": node.get_name(), "type": type(node).__name__,
                "members": [n.get_name() for n in node.get_nodes()]}

    def create(owner, *positional, **keywords):
        before = [summarize(n) for n in owner.nodes]
        answer = original_create(owner, *positional, **keywords)
        active["schedules"].append({"before": before, "after": [summarize(n) for n in owner.nodes]})
        return answer

    def init(owner):
        context = original_init(owner)
        active["baseline_context"].append({"baseline_peak": context.baseline_peak,
                                             "live_before": list(context.baseline_live_before)})
        return context

    def region(*positional, **keywords):
        value = original_region(*positional, **keywords)
        active["last_region_peak"] = value
        return value

    def check(owner, nodes, context, enable_autotune):
        previous = context.running_peak
        active.pop("last_region_peak", None)
        answer = original_try(owner, nodes, context, enable_autotune)
        peak = active.pop("last_region_peak")
        delta = max(previous, peak) - context.baseline_peak
        active["candidates"].append({
            "members": [n.get_name() for n in nodes], "region_peak": peak,
            "baseline_peak": context.baseline_peak, "running_peak_before": previous,
            "delta": delta, "accepted": answer[0] is not None,
            "threshold_bytes": 1024 * 1024,
        })
        return answer

    def emit(owner, node):
        entry = {"combo": summarize(node)}
        before = dict(V.graph.wrapper_code.src_to_kernel)
        answer = original_emit(owner, node)
        generated = [name for source, name in V.graph.wrapper_code.src_to_kernel.items()
                     if source not in before]
        entry["generated_kernels"] = generated
        active["emitted"].append(entry)
        return answer

    model = tests.ComboKernelPeakMemoryTests._make_wide_resnet_like().to("npu").eval()
    x = torch.randn(1, 3, 224, 224, device="npu")
    result["model_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    result["buffer_bytes"] = sum(p.numel() * p.element_size() for p in model.buffers())
    for label, threshold in [("disabled", None), ("tight", 1.0 / 1024)]:
        torch._dynamo.reset()
        metrics.reset()
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        active = {"label": label, "candidates": [], "emitted": [], "schedules": [],
                  "baseline_context": [], "start_allocated": torch.npu.memory_allocated()}
        record = active
        memory = torch.npu.memory
        memory._record_memory_history(enabled="all", context="alloc", stacks="python", max_entries=100000)
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(tests.fresh_cache(dir=str(output), delete=False))
                stack.enter_context(config.patch({
                    "combo_kernels": True, "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                    "combo_kernel_max_distance": 32,
                    "combo_kernel_peak_memory_increase_gb": threshold,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }))
                for owner, name, replacement in [
                    (Scheduler, "create_combo_kernel_nodes", create),
                    (Scheduler, "_init_peak_memory_context", init),
                    (Scheduler, "_try_combo_with_memory_check", check),
                    (memory_module, "estimate_region_peak_memory", region),
                    (NPUTritonScheduling, "codegen_combo_kernel", emit),
                ]:
                    stack.enter_context(patch.object(owner, name, replacement))
                with torch.no_grad():
                    compiled = torch.compile(model)
                    y = compiled(x)
                torch.npu.synchronize()
                record["first_call_peak"] = torch.npu.max_memory_allocated()
                record["first_call_current"] = torch.npu.memory_allocated()
                snapshot = memory._snapshot()
                import pickle
                (output / f"{label}-first-snapshot.pkl").write_bytes(pickle.dumps(snapshot))
                memory._record_memory_history(enabled=None)
                del y
                gc.collect()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                memory._record_memory_history(enabled="all", context="alloc", stacks="python", max_entries=100000)
                with torch.no_grad():
                    y = compiled(x)
                torch.npu.synchronize()
                record["warm_call_peak"] = torch.npu.max_memory_allocated()
                record["warm_call_current"] = torch.npu.memory_allocated()
                snapshot = memory._snapshot()
                (output / f"{label}-warm-snapshot.pkl").write_bytes(pickle.dumps(snapshot))
                record["generated_kernel_count"] = metrics.generated_kernel_count
                record["cache_dir"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
                record["status"] = "OBSERVED"
                del y
        except Exception:
            record["status"] = "PROBE_ERROR"
            record["error"] = traceback.format_exc()
        finally:
            memory._record_memory_history(enabled=None)
        result["records"].append(record)
        write_result()
        print("WRN_RECORD_SUMMARY=" + json.dumps({key: value for key, value in record.items()
                                                  if key not in ["schedules", "candidates", "emitted"]},
                                                 ensure_ascii=False), flush=True)
        del compiled
        gc.collect()


try:
    {"foreach": run_foreach, "ftz": run_ftz, "wrn": run_wrn}[args.mode]()
except Exception:
    result["fatal_error"] = traceback.format_exc()
    print(result["fatal_error"], flush=True)
    raise
finally:
    write_result()
