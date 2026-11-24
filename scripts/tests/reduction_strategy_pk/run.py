"""Run paired profiler-only strategy experiments, one serial queue per device."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types

from common import (cases, compare, cpu_inputs, device_inputs, eager_function, profile_candidates,
                    reference, sha256, write_json)

HERE = Path(__file__).resolve().parent
STOP = threading.Event()


def checkpoint(out, stage, **extra):
    write_json(out / "active.json", dict(stage=stage, time=time.time(), **extra))
    print(stage, json.dumps(extra), flush=True)


@contextmanager
def capture_splits(records):
    from torch._inductor import ir

    saved = []

    def decorate(function):
        def wrapped(cls, *args, **kwargs):
            bound = inspect.signature(function).bind(cls, *args, **kwargs).arguments
            records.append({"ir_class": cls.__name__, **{key: str(value) for key, value in bound.items()
                            if key in ("ranges", "reduction_ranges", "split", "reduction_type", "dst_dtype")}})
            return function(cls, *args, **kwargs)
        return classmethod(wrapped)

    for cls in (ir.Reduction, ir.OnlineSoftmaxReduction, ir.WelfordReduction):
        descriptor = cls.__dict__["create_multilayer"]
        saved.append((cls, descriptor))
        cls.create_multilayer = decorate(descriptor.__func__)
    try:
        yield
    finally:
        for cls, descriptor in reversed(saved):
            cls.create_multilayer = descriptor


def save_kernel(kernel, out, name):
    out.mkdir(parents=True, exist_ok=True)
    meta = kernel.metadata
    meta = meta._asdict() if hasattr(meta, "_asdict") else vars(meta)
    write_json(out / (name + ".metadata.json"), meta)
    for stage in ("ttir", "ttgir", "ttadapter", "llir", "ptx"):
        value = kernel.asm.get(stage)
        if isinstance(value, str):
            (out / (name + "." + stage)).write_text(value)
    return meta


def compile_strategy(case, strategy, inputs, backend, out, expected):
    import torch
    from torch._inductor import metrics
    from torch._inductor.utils import run_and_get_code

    tools = HERE if (HERE / "autotune_tiling.py").exists() else HERE.parents[1] / "tools"
    sys.path.insert(0, str(tools))
    from autotune_tiling import BestTilingRecorder, _config_payload

    class Recorder(BestTilingRecorder):
        def __init__(self):
            super().__init__(backend)
            self.saved = set()
            self.artifacts = []

        def _record_standard(self, autotuner):
            super()._record_standard(autotuner)
            self.save(autotuner)

        def _record_grouped(self, autotuner, args):
            super()._record_grouped(autotuner, args)
            self.save(autotuner)

        def save(self, autotuner):
            if id(autotuner) in self.saved:
                return
            launchers = getattr(autotuner, "launchers", ())
            launcher = getattr(autotuner, "best_launcher", None)
            if launcher is None and len(launchers) == 1:
                launcher = launchers[0]
            if launcher is None:
                return
            kernel = getattr(launcher, "bin", None)
            selected = _config_payload(getattr(launcher, "config", None))
            results = getattr(autotuner, "compile_results", ()) or ()
            if kernel is None:
                kernel = next((item.kernel for item in results if _config_payload(item.config) == selected), None)
            if kernel is None:
                return
            name = "kernel-" + str(len(self.saved))
            if hasattr(kernel, "metadata"):
                metadata = save_kernel(kernel, out / "ir", name)
            else:
                from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key

                cache = Path(os.environ["TRITON_CACHE_DIR"]) / triton_hash_to_path_key(kernel.hash)
                metadata = json.loads((cache / (kernel.name + ".json")).read_text())
                (out / "ir").mkdir(exist_ok=True)
                write_json(out / "ir" / (name + ".metadata.json"), metadata)
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    path = cache / (kernel.name + "." + stage)
                    if path.exists():
                        (out / "ir" / (name + "." + stage)).write_bytes(path.read_bytes())
            self.saved.add(id(autotuner))
            self.artifacts.append(dict(name=name, selected=selected, metadata=metadata,
                                       candidates=[_config_payload(item.config) for item in results]))

    out.mkdir(parents=True, exist_ok=True)
    options = {"split_reductions": strategy in ("split", "natural"),
               "triton.cooperative_reductions": strategy in ("cooperative", "natural"),
               "triton.force_cooperative_reductions": strategy == "cooperative",
               "triton.multi_kernel": 0, "max_autotune": True, "triton.cudagraphs": False,
               "fx_graph_cache": False, "autotune_local_cache": False}
    record = dict(strategy=strategy, entry="inductor", config=options, status="RUNNING", splits=[])
    recorder = Recorder()
    recorder.install()
    recorder.start_capture()
    try:
        fn = eager_function(case["op"])
        # Distinct Dynamo code objects prevent strategy caches sharing a config.
        fn = types.FunctionType(fn.__code__.replace(co_name="forward_" + strategy), fn.__globals__,
                                "forward_" + strategy, fn.__defaults__, fn.__closure__)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False, options=options)
        metrics.reset()
        with capture_splits(record["splits"]), torch.no_grad():
            first, sources = run_and_get_code(compiled, *inputs)
        getattr(torch, backend).synchronize()
        record["selected_configs"] = recorder.stop_capture()
        record["kernels"] = recorder.artifacts
        record["generated_kernel_count"] = metrics.generated_kernel_count
        for index, text in enumerate(sources):
            (out / f"output_code_{index}.py").write_text(text)
        text = "\n".join(sources)
        record["source_summary"] = dict(cooperative_decorators=text.count("@triton_heuristics.cooperative_reduction("),
                                         barrier_calls=text.count("triton_helpers.x_grid_barrier("))
        rsplit = [int(item["selected_config"]["RSPLIT"]) for item in record["selected_configs"]
                  if "RSPLIT" in item["selected_config"]]
        record["actual_path"] = ("cooperative" if record["source_summary"]["cooperative_decorators"] and
                                  rsplit and max(rsplit) > 1 else "split" if record["splits"] else "ordinary")
        record["RS"] = rsplit
        record["accuracy"] = [compare(first, expected, case["op"])]
        record["status"] = "PASS" if record["accuracy"][0]["passed"] else "WRONG"
        write_json(out / "result.json", record)
        return record, lambda: compiled(*inputs)
    finally:
        recorder.uninstall()


def make_dsl(case, config, inputs, buffers, out):
    import triton
    import npu_kernels as kernels

    x, r, s, block = case["X"], case["R"], config["S"], config["B"]
    op = ("sum", "amax", "rmsnorm", "softmax", "sum_amax").index(case["op"])
    a, b = inputs
    workspace, output, maxout = buffers
    shared = dict(R=r, S=s, X=x, BLOCK=block, SX=a.stride(0), SR=a.stride(1), OP=op)
    options = dict(compile_mode="simd", num_warps=4, launch_cooperative_grid=config["strategy"] == "cooperative")
    saved = []

    def invoke():
        if config["strategy"] == "split":
            first = kernels.partial[(s, x)](a, b, workspace, **shared, INTERLEAVED=config["I"], **options)
            second = kernels.finish[(config["E"], x)](
                a, b, workspace, output, maxout, **shared, P=triton.next_power_of_2(s), E=config["E"], **options)
            binaries = (first, second)
        else:
            binaries = (kernels.cooperative[(s, x)](
                a, b, workspace, output, maxout, **shared, P=triton.next_power_of_2(s), **options),)
        if not saved:
            for index, binary in enumerate(binaries):
                saved.append(save_kernel(binary, out / "ir", "kernel-" + str(index)))
        return (output, maxout) if case["op"] == "sum_amax" else output

    return invoke, saved


def config_id(config):
    return f"{config['strategy']}-s{config['S']}-b{config['B']}-i{int(config['I'])}-e{config['E']}"


def initial_configs(case, smoke, vector_cores=56):
    sizes = (2, 8) if smoke else sorted({2, 4, 8, 14, max(2, vector_cores // 2),
                                       vector_cores, 2 * vector_cores})
    blocks = (1024,) if smoke else (64, 256, 512) if case["R"] <= 4096 else (512, 2048)
    rows = case["X"]
    result = []
    for strategy in ("split", "cooperative"):
        for s in sizes:
            if s > 2 * vector_cores or strategy == "cooperative" and (s > vector_cores or rows * s > vector_cores):
                continue
            for block in blocks:
                if case["R"] <= 4096 and s > max(2, (case["R"] + block - 1) // block):
                    continue
                e = s if case["op"] in ("rmsnorm", "softmax") else 1
                result.append(dict(strategy=strategy, S=s, B=block, I=False, E=e))
    return result


def integer_exchange(case, splits, out):
    import torch
    import triton
    import npu_kernels as kernels

    rows = case["X"]
    workspace = torch.full((rows * splits,), -1, dtype=torch.int32, device="npu")
    output = torch.full((4, rows, splits), -1, dtype=torch.int32, device="npu")
    for epoch in (1, 7):
        binary = kernels.exchange[(splits, rows)](
            workspace, output, epoch, S=splits, P=triton.next_power_of_2(splits), X=rows,
            compile_mode="simd", num_warps=4, launch_cooperative_grid=True)
        torch.npu.synchronize()
        expected = torch.empty((4, rows, splits), dtype=torch.int32)
        for phase in range(4):
            for row in range(rows):
                expected[phase, row, :] = splits * (splits + 1) // 2 + splits * (row * 100 + phase * 10000 + epoch * 1000)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    save_kernel(binary, out / "exchange" / f"s{splits}", "exchange")
    return dict(passed=True, epochs=[1, 7], phases=4, values=8 * rows * splits)


def validate_final(case, inputs, selected, expected, out):
    import torch

    api = getattr(torch, inputs[0].device.type)
    for label, (record, invoke) in selected.items():
        checks = record.setdefault("accuracy", [])
        for _ in range(3):
            value = invoke()
        api.synchronize()
        checks.append(compare(value, expected, case["op"]))
    for epoch in (1, 2):
        values = cpu_inputs(case, 213 + epoch)
        wanted = reference(case, values)
        for tensor, value in zip(inputs, values):
            tensor.copy_(value)
        for label, (record, invoke) in selected.items():
            for _ in range(3):
                actual = invoke()
            api.synchronize()
            record["accuracy"].append(compare(actual, wanted, case["op"]))
            record["status"] = "PASS" if all(check["passed"] for check in record["accuracy"]) else "WRONG"
    write_json(out / "final_checks.json", {label: record["accuracy"] for label, (record, _) in selected.items()})


def environment(backend, device, vector_cores=56):
    import torch
    import triton

    prop = getattr(torch, backend).get_device_properties(0)
    record = dict(torch=torch.__version__, torch_commit=torch.version.git_version, triton=triton.__version__,
                  device=str(prop), requested_device=device, source_sha256={})
    modules = ["torch._inductor.choices", "torch._inductor.ir", "torch._inductor.codegen.triton"]
    if backend == "npu":
        import torch_npu
        from triton.backends.ascend.driver import NPUUtils
        import ctypes

        physical = ctypes.c_int32(-1)
        status = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId(0, ctypes.byref(physical))
        if status or str(physical.value) != device:
            raise RuntimeError(f"Wrong physical NPU: {physical.value} != {device}")
        record.update(torch_npu=torch_npu.__version__, physical_device=physical.value,
                      effective_vector_cores=NPUUtils().get_device_properties(0)["num_vectorcore"],
                      device_limit=torch.npu.get_device_limit(0))
        if record["effective_vector_cores"] != vector_cores:
            raise RuntimeError(f"Declared Vector budget {vector_cores} differs from effective "
                               f"budget {record['effective_vector_cores']}")
        modules += ["torch_npu._inductor.codegen.triton", "torch_npu._inductor.ir"]
    else:
        record.update(uuid=str(prop.uuid), sm_count=prop.multi_processor_count, cuda=torch.version.cuda)
        if "GPU-" + str(prop.uuid).removeprefix("GPU-") != device:
            raise RuntimeError("Wrong physical CUDA UUID")
    for name in modules:
        record["source_sha256"][name] = sha256(inspect.getfile(importlib.import_module(name)))
    return record


def worker(job, out):
    import torch

    backend, case = job["backend"], job["case"]
    if backend == "npu":
        import torch_npu  # noqa: F401

    api = getattr(torch, backend)
    api.set_device(0)
    torch.set_num_threads(4)
    policy = job.get("execution_policy", {})
    vector_cores = policy.get("npu_vector_cores", 56)
    max_splits = 2 * vector_cores
    if backend == "npu":
        api.set_op_timeout_ms(15000)
    else:
        fraction = policy.get("small_memory_fraction", .035) if job["small_card"] else policy.get("gpu_memory_fraction", .20)
        api.set_per_process_memory_fraction(fraction, 0)
    record = dict(case=case, backend=backend, environment=environment(backend, job["device"], vector_cores), candidates=[], selected=[],
                  submission=backend.upper() + " Graph replay" if os.getenv("PK_PROFILE_GRAPH") == "1" else "direct callable")
    if policy:
        record["execution_policy"] = policy
    write_json(out / "environment.json", record["environment"])
    values = cpu_inputs(case, 213)
    inputs = device_inputs(case, values, backend)
    expected = reference(case, values)
    fn = eager_function(case["op"])
    record["eager_check"] = compare(fn(*inputs), expected, case["op"])
    record["input_stride"] = list(inputs[0].stride())
    del values
    checkpoint(out, "compile")
    ready = {}

    def measure(items, directory, rounds, batch):
        def validation(label, actual):
            check = compare(actual, expected, case["op"])
            ready[label][0].setdefault("graph_accuracy", []).append(check)
            return check

        return profile_candidates(backend, items, directory, rounds=rounds, batch=batch, validation=validation)

    confirmation_batch = 3 if case["X"] * case["R"] > 33554432 else 5
    record["confirmation_batch"] = confirmation_batch
    for strategy in (("plain", "split", "cooperative", "natural") if backend == "cuda" else ("plain",)):
        checkpoint(out, "compile", strategy=strategy)
        try:
            result, invoke = compile_strategy(case, strategy, inputs, backend, out / strategy, expected)
            if result["status"] == "PASS":
                ready[strategy] = (result, invoke)
        except Exception:
            result = dict(strategy=strategy, entry="inductor", status="ERROR", error=traceback.format_exc())
            print(result["error"], flush=True)
        record["candidates"].append(result)
        write_json(out / "progress.json", record)
    if backend == "cuda":
        if ready:
            checkpoint(out, "profile")
            for round_index in range(3):
                measured = measure({k: v[1] for k, v in ready.items()},
                                   out / f"confirm-{round_index}", rounds=3, batch=confirmation_batch)
                for name, measurement in measured.items():
                    ready[name][0].setdefault("confirmation", []).append(measurement)
            validate_final(case, inputs, ready, expected, out)
        for name, (result, _) in ready.items():
            result["timing"] = merge_timing(result["confirmation"])
            record["selected"].append(result)
        record["selected"].extend(r for r in record["candidates"] if r["strategy"] not in ready)
    else:
        shape = (case["X"], case["R"]) if case["op"] in ("rmsnorm", "softmax") else (case["X"],)
        buffers = (torch.full((2 * case["X"] * max_splits,), float("nan"), device="npu"),
                   torch.full(shape, float("nan"), device="npu"),
                   torch.full((case["X"],), float("nan"), device="npu"))
        exchange_results = {}

        def prepare(configs):
            added = {}
            for config in configs:
                label = config_id(config)
                if label in ready or any(item.get("id") == label for item in record["candidates"]):
                    continue
                checkpoint(out, "dsl_compile", candidate=label)
                result = dict(id=label, strategy=config["strategy"], entry="dsl_prototype", config=config,
                              actual_path=config["strategy"], grid=[config["S"], case["X"]], status="RUNNING")
                try:
                    invoke, metadata = make_dsl(case, config, inputs, buffers, out / label)
                    actual = invoke()
                    api.synchronize()
                    result["accuracy"] = [compare(actual, expected, case["op"])]
                    result["metadata"] = metadata
                    if config["strategy"] == "cooperative":
                        if config["S"] not in exchange_results:
                            exchange_results[config["S"]] = integer_exchange(case, config["S"], out)
                        result["exchange"] = exchange_results[config["S"]]
                    result["status"] = "PASS" if result["accuracy"][0]["passed"] else "WRONG"
                    if result["status"] == "PASS":
                        ready[label] = (result, invoke)
                        added[label] = invoke
                except Exception:
                    result.update(status="ERROR", error=traceback.format_exc())
                    print(result["error"], flush=True)
                record["candidates"].append(result)
                write_json(out / "progress.json", record)
            return added

        tuning_record = job.get("tuning_record")
        if tuning_record:
            previous = json.loads(Path(tuning_record).read_text())
            if previous["case"] != case or previous["backend"] != backend:
                raise ValueError("Prior tuning case differs")
            configs = [item["config"] for item in previous.get("candidates", [])
                       if item.get("entry") == "dsl_prototype" and item.get("status") == "PASS"
                       and item.get("confirmation")]
            if not configs:
                raise ValueError("No previously confirmed DSL configurations")
            record["tuning_source"] = dict(path=tuning_record, sha256=sha256(tuning_record),
                                          method="remeasure previously confirmed finalists")
            added = prepare(configs)
        else:
            added = prepare(initial_configs(case, job["smoke"], vector_cores))
        if added and not tuning_record:
            checkpoint(out, "profile_search")
            search = measure(added, out / "search", rounds=3, batch=3)
            for name, measurement in search.items():
                ready[name][0]["search_timing"] = measurement
        if not job["smoke"] and not tuning_record:
            extras = []
            for strategy in ("split", "cooperative"):
                ranked = sorted((r for r, _ in ready.values() if r["entry"] == "dsl_prototype"
                                 and r["strategy"] == strategy and "search_timing" in r),
                                key=lambda r: r["search_timing"]["median_us"])[:2]
                for result in ranked:
                    config = result["config"]
                    for block in (256, 1024, 4096):
                        extras.append({**config, "B": block})
                    if strategy == "split":
                        extras.append({**config, "I": True})
                        if case["op"] in ("rmsnorm", "softmax"):
                            for emit in (max(1, vector_cores // case["X"]), min(max_splits, 2 * config["S"])):
                                extras.append({**config, "E": emit})
            added = prepare(extras)
            if added:
                checkpoint(out, "profile_refine")
                for name, measurement in measure(added, out / "refine", rounds=3, batch=3).items():
                    ready[name][0]["search_timing"] = measurement
        shortlist = {"plain": ready["plain"]} if "plain" in ready else {}
        for strategy in ("split", "cooperative"):
            eligible = [(name, pair) for name, pair in ready.items() if pair[0]["entry"] == "dsl_prototype"
                        and pair[0]["strategy"] == strategy and (tuning_record or "search_timing" in pair[0])]
            ranked = eligible if tuning_record else sorted(eligible,
                       key=lambda pair: pair[1][0]["search_timing"]["median_us"])[:2]
            shortlist.update(ranked)
        if shortlist:
            for round_index in range(3):
                checkpoint(out, "profile_confirm", repeat=round_index)
                for name, measurement in measure({k: v[1] for k, v in shortlist.items()},
                                                out / f"confirm-{round_index}", rounds=3, batch=confirmation_batch).items():
                    shortlist[name][0].setdefault("confirmation", []).append(measurement)
            validate_final(case, inputs, shortlist, expected, out)
            for result, _ in shortlist.values():
                result["timing"] = merge_timing(result["confirmation"])
        for strategy in ("plain", "split", "cooperative"):
            finalists = [r for r, _ in shortlist.values() if r["strategy"] == strategy and r["status"] == "PASS"]
            if finalists:
                record["selected"].append(min(finalists, key=lambda r: r["timing"]["median_us"]))
            else:
                record["selected"].append(dict(strategy=strategy, status="NOT_APPLICABLE" if
                                              strategy == "cooperative" and case["X"] * 2 > vector_cores else "NO_VALID_CONFIG"))
    record["status"] = "PASS" if all(r["status"] in ("PASS", "NOT_APPLICABLE") for r in record["selected"]) and record["selected"] else "INCOMPLETE"
    record["eager_check_passed"] = record["eager_check"]["passed"]
    if not record["eager_check_passed"]:
        record["status"] = "WRONG"
    write_json(out / "result.json", record)
    checkpoint(out, "complete", status=record["status"])
    return record


def merge_timing(values):
    medians = [item["median_us"] for item in values]
    return dict(median_us=statistics.median(medians), min_us=min(item["min_us"] for item in values),
                max_us=max(item["max_us"] for item in values), round_medians_us=medians,
                samples_us=[v for item in values for v in item["samples_us"]],
                events_per_call=values[0]["events_per_call"], submission=values[0].get("submission", "direct callable"),
                method="profiler device durations; median of 3 session medians")


def launch_case(args, case, device, small_card):
    out = args.out / "cases" / case["id"]
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    if args.backend == "npu":
        waiting_since = time.monotonic()
        while True:
            if STOP.is_set():
                return dict(case=case, backend=args.backend, status="INTERRUPTED")
            try:
                usage = subprocess.run(["npu-smi", "info", "-i", device, "-t", "usages"],
                                       capture_output=True, text=True, timeout=20)
                rates = [int(line.split(":", 1)[1].strip()) for line in usage.stdout.splitlines()
                         if "HBM Usage Rate(%)" in line]
            except (subprocess.TimeoutExpired, OSError):
                rates = []
            if rates and rates[0] <= getattr(args, "max_hbm_percent", 95):
                break
            write_json(out / "resource_wait.json", dict(device=device, hbm_usage_percent=rates, time=time.time()))
            deadline = getattr(args, "resource_wait_timeout", 0)
            if deadline and time.monotonic() - waiting_since >= deadline:
                result = dict(case=case, backend=args.backend, status="RESOURCE_TIMEOUT", device=device,
                              error="HBM occupancy stayed above the limit, or npu-smi did not return a valid reading")
                write_json(out / "result.json", result)
                return result
            print("waiting for free HBM", device, rates, flush=True)
            STOP.wait(min(30, max(1, deadline - (time.monotonic() - waiting_since))) if deadline else 30)
    job = dict(backend=args.backend, case=case, device=device, smoke=args.smoke, small_card=small_card)
    if getattr(args, "execution_policy", None):
        job["execution_policy"] = args.execution_policy
    if getattr(args, "tuning_source", None):
        if args.backend != "npu":
            raise ValueError("Saved DSL finalists apply only to NPU")
        job["tuning_record"] = str(args.tuning_source.resolve() / "cases" / case["id"] / "result.json")
    write_json(out / "job.json", job)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES" if args.backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"] = device
    env["TORCHINDUCTOR_CACHE_DIR"] = str(args.out / "cache" / device / "inductor")
    env["TRITON_CACHE_DIR"] = str(args.out / "cache" / device / "triton")
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MAX_JOBS"] = "4"
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", str(HERE / "run.py"), "--job", str(out / "job.json")],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            while process.poll() is None:
                if STOP.is_set():
                    break
                active = out / "active.json"
                modified = active.stat().st_mtime if active.exists() else (out / "job.json").stat().st_mtime
                if time.time() - modified > args.timeout:
                    timed_out = True
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
    else:
        result = dict(case=case, backend=args.backend, status="INTERRUPTED" if STOP.is_set() else
                      "TIMEOUT" if timed_out else "PROCESS_ERROR",
                      returncode=process.returncode)
        write_json(out / "result.json", result)
    print(device, case["id"], result["status"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"))
    parser.add_argument("--devices")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--emit-manifest", type=Path)
    parser.add_argument("--tuning-source", type=Path,
                        help="NPU only: independently remeasure prior confirmed finalists; no new search")
    parser.add_argument("--small-devices", default=None,
                        help="CUDA UUIDs restricted to X*R<=1048576; default retains the historical secondary-card policy")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.20)
    parser.add_argument("--small-memory-fraction", type=float, default=.035)
    parser.add_argument("--npu-vector-cores", type=int, default=56)
    parser.add_argument("--max-hbm-percent", type=int, default=95)
    parser.add_argument("--resource-wait-timeout", type=float, default=0,
                        help="Maximum NPU HBM wait in seconds; zero waits until interrupted")
    parser.add_argument("--uniform-no-grad", action="store_true",
                        help="Use no_grad throughout compilation, warmup, capture and validation")
    args = parser.parse_args()
    if args.emit_manifest:
        write_json(args.emit_manifest, cases())
        return 0
    if args.job:
        job = json.loads(args.job.read_text())
        out = args.job.parent
        checkpoint(out, "initialize")
        try:
            if job.get("execution_policy", {}).get("uniform_no_grad"):
                import torch
                with torch.no_grad():
                    result = worker(job, out)
            else:
                result = worker(job, out)
        except Exception:
            result = dict(case=job["case"], backend=job["backend"], status="ERROR", error=traceback.format_exc())
            print(result["error"], flush=True)
            write_json(out / "result.json", result)
        return 0 if result["status"] == "PASS" else 2
    if not args.backend or not args.devices or args.out is None:
        parser.error("--backend, --devices and --out are required")
    if not (0 < args.gpu_memory_fraction <= 1 and 0 < args.small_memory_fraction <= 1):
        parser.error("Allocator fractions must be in (0,1]")
    if args.npu_vector_cores < 2 or args.timeout <= 0 or args.resource_wait_timeout < 0:
        parser.error("Vector cores must be >=2, timeout positive and resource wait nonnegative")
    if not 0 < args.max_hbm_percent <= 100:
        parser.error("--max-hbm-percent must be in 1..100")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    devices = args.devices.split(",")
    if any(not device for device in devices) or len(set(devices)) != len(devices):
        parser.error("--devices contains empty or duplicate selectors")
    small_devices = devices[1:] if args.backend == "cuda" and args.small_devices is None else (
        args.small_devices.split(",") if args.small_devices else [])
    if not set(small_devices) <= set(devices) or (args.backend == "npu" and small_devices):
        parser.error("--small-devices must be selected CUDA devices")
    selected = cases()
    if args.case_ids:
        ids = args.case_ids.split(",")
        selected = [case for case in selected if case["id"] in ids]
        if len(selected) != len(ids):
            raise ValueError("Unknown or duplicate case ID")
    elif args.smoke:
        selected = [case for case in selected if case["X"] == 7 and case["R"] == 32768 and case["group"] == "core"]
    hashes = {path.name: sha256(path) for path in HERE.glob("*.py")}
    args.execution_policy = dict(npu_vector_cores=args.npu_vector_cores,
                                 gpu_memory_fraction=args.gpu_memory_fraction,
                                 small_memory_fraction=args.small_memory_fraction,
                                 small_devices=small_devices, uniform_no_grad=args.uniform_no_grad,
                                 max_hbm_percent=args.max_hbm_percent,
                                 resource_wait_timeout=args.resource_wait_timeout)
    manifest = dict(backend=args.backend, devices=devices, cases=selected, source_sha256=hashes,
                    smoke=args.smoke, profile_graph=os.getenv("PK_PROFILE_GRAPH", "0"),
                    tuning_source=str(args.tuning_source.resolve()) if args.tuning_source else None,
                    date=datetime.now(timezone.utc).isoformat(), execution_policy=args.execution_policy)
    existing = args.out / "run.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if not args.resume or any(old.get(k) != manifest.get(k) for k in ("backend", "devices", "cases", "source_sha256", "smoke", "profile_graph", "tuning_source", "execution_policy")):
            raise ValueError("Existing output configuration/source differs; select a new directory")
    else:
        write_json(existing, manifest)
    queues = [[] for _ in devices]
    for case in selected:
        eligible = list(range(len(devices)))
        if args.backend == "cuda" and case["X"] * case["R"] > 1048576:
            eligible = [index for index, device in enumerate(devices) if device not in small_devices]
        if not eligible:
            raise ValueError("At least one regular CUDA device is required for the selected large cases")
        target = min(eligible, key=lambda index: len(queues[index]))
        queues[target].append(case)

    def queue(index):
        results = []
        for case in queues[index]:
            if STOP.is_set():
                break
            results.append(launch_case(args, case, devices[index], devices[index] in small_devices))
        return results

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: STOP.set())
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        batches = list(executor.map(queue, range(len(devices))))
    results = [record for batch in batches for record in batch]
    write_json(args.out / "results.json", results)
    return 130 if STOP.is_set() else 0 if len(results) == len(selected) and all(
        record["status"] == "PASS" for record in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
