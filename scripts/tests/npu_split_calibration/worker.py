"""One process per case, paired strategies on the same tensors and device."""

from __future__ import annotations

import argparse
import gc
import importlib
import math
import os
from pathlib import Path
import platform
import subprocess
import traceback
import types
from unittest.mock import patch

from cases import cpu_values, estimated_working_bytes, forward_function, input_bytes, input_spec, reference, shapes
from common import file_hash, read_json, require_profiler_settings, source_hashes, timing_contract, write_json
from profiling import benchmark
from hooks import SplitController, install_ordinary_lowerings, make_recorder
from policy import digest, split_count, validate_table


def initialize(device, lowering_profile="installed"):
    import torch
    import torch_npu

    if os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default") != "default":
        raise ValueError("this suite currently targets TORCHINDUCTOR_NPU_BACKEND=default")
    torch.npu.set_device(device)
    import torch_npu._inductor  # activate the backend BEFORE installing hooks
    from torch_npu._inductor import config as npu_config
    import triton

    restored = install_ordinary_lowerings() if lowering_profile == "ordinary_ir" else []
    props = torch.npu.get_device_properties(device)
    fingerprint = dict(device_name=torch.npu.get_device_name(device),
                       vector_cores=int(npu_config.num_vector_core),
                       physical_vector_cores=int(props.vector_core_num),
                       torch=str(torch.__version__), torch_git=getattr(torch.version, "git_version", None),
                       torch_npu=str(torch_npu.__version__), triton=str(triton.__version__),
                       backend=os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default"),
                       device_limit=os.getenv("NPU_DEVICE_LIMIT", ""),
                       lowering_profile=lowering_profile, restored_lowerings=restored)
    modules = ("torch._inductor.ir", "torch._inductor.choices", "torch_npu._inductor.ir",
               "torch_npu._inductor.config", "torch_npu._inductor.codegen.triton",
               "torch_npu._inductor.runtime.tile_generator", "torch_npu._inductor.runtime.triton_heuristics",
               "torch._inductor.lowering", "torch_npu._inductor.lowering_fallback_list")
    fingerprint["source_hashes"] = {name: file_hash(importlib.import_module(name).__file__) for name in modules}
    fingerprint["policy_environment"] = {
        key: value for key, value in os.environ.items()
        if key.startswith(("TORCHINDUCTOR_", "TORCHNPU_", "TRITON_", "ENABLE_", "NPU_"))
        and key not in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    }
    for name in ("CANN_VERSION", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        fingerprint[name] = os.getenv(name, "")
    version_paths = [Path("/usr/local/Ascend/driver/version.info")]
    if os.getenv("ASCEND_HOME_PATH"):
        version_paths += [Path(os.environ["ASCEND_HOME_PATH"]) / name for name in (
            "version.cfg", "version.info", "arm64-linux/ascend_toolkit_install.info",
            "aarch64-linux/ascend_toolkit_install.info", "x86_64-linux/ascend_toolkit_install.info")]
    fingerprint["runtime_version_files"] = {str(path): path.read_text(errors="replace") for path in version_paths if path.is_file()}
    info = dict(fingerprint=fingerprint, python=platform.python_version(), platform=platform.platform(),
                device_assignment=dict(hostname=platform.node(), logical_device=device,
                                       visible_devices=os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")),
                properties=str(props), env={k: v for k, v in os.environ.items() if k.startswith((
                    "TORCH", "TRITON", "ASCEND", "NPU_", "ENABLE_"))})
    try:
        result = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=15)
        info["npu_smi"] = result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["npu_smi_error"] = str(exc)
    return info


def make_inputs(case, x, r, buffers, device, seed, pattern="normal", positions=()):
    import torch

    generator = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape, strides = input_spec(case, x, r)
    result = []
    for _ in range(buffers):
        pair = []
        for index in range(2):
            dims = ((x, 1) if index == 0 else (1, r)) if case["layout"] == "broadcast" else shape
            cpu = cpu_values(case, dims, generator, pattern, positions)
            # A broadcast tail probe must not become true for every column
            # through the [X,1] input; inject the sentinel into [1,R] only.
            if case["layout"] == "broadcast" and index == 0 and pattern != "normal":
                cpu.fill_(1 if case["op"] == "prod" else 0)
            elif case["layout"] == "broadcast" and index == 0 and case["op"] == "prod":
                cpu = (torch.randint(0, 2, dims, generator=generator) * 2 - 1).to(dtype)
            tensor = (torch.empty_strided(dims, strides, dtype=dtype, device=device) if strides
                      else torch.empty(dims, dtype=dtype, device=device))
            tensor.copy_(cpu)
            pair.append(tensor)
        result.append(tuple(pair))
    return result


def accuracy(actual, expected, dtype):
    import torch

    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError("output structure differs")
        return dict(passed=True, outputs=[accuracy(a, e, dtype) for a, e in zip(actual, expected)])
    tolerance = {torch.float32: (2e-4, 5e-4), torch.float16: (3e-3, 3e-3), torch.bfloat16: (2e-2, 2e-2)}
    # Bool/integer/index outputs are exact; FP32 outputs of low-precision inputs
    # must not inherit loose BF16 output tolerances.
    rtol, atol = tolerance.get(expected.dtype, (0, 0))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    error = (actual.double() - expected.double()).abs()
    return dict(passed=True, rtol=rtol, atol=atol, output_dtype=str(expected.dtype),
                max_abs_error=float(error.max()) if error.numel() else 0,
                exact=not expected.is_floating_point())


def to_cpu(value):
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    return value.detach().cpu()


class Strategy:
    def __init__(self, mode, settings, case, cores, table, directory):
        import torch
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx

        self.mode, self.directory = mode, directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.controller = SplitController(mode, settings, cores, table)
        self.compiles, self.sources, self.active_compile_id = 0, 0, None
        self.inspections = 0
        self.options = {"split_reductions": True, "min_num_split": 0,
                        "triton.cooperative_reductions": False, "triton.force_cooperative_reductions": False,
                        "triton.multi_kernel": 0, "max_autotune": settings["max_autotune"],
                        "triton.cudagraphs": False, "fx_graph_cache": False,
                        "autotune_local_cache": False, "compile_threads": 1,
                        "triton.mix_order_reduction": False}
        # Do not silently ignore missing essential controls on another torch release.
        for key in self.options:
            obj = config
            for part in key.split("."):
                obj = getattr(obj, part)

        def backend(gm, example_inputs):
            self.compiles += 1
            compile_id = self.compiles
            self.controller.compile_id = compile_id
            (directory / f"fx_{self.compiles}.py").write_text(gm.code, encoding="utf-8")
            with config.patch(self.options), self.controller.install():
                compiled_graph = compile_fx(gm, example_inputs)

            def invoke(*args, **kwargs):
                self.active_compile_id = compile_id
                return compiled_graph(*args, **kwargs)

            if hasattr(compiled_graph, "_boxed_call"):
                invoke._boxed_call = compiled_graph._boxed_call
            return invoke

        fn = forward_function(case)
        # Avoid Dynamo code-cache reuse between the strategies.
        fn = types.FunctionType(fn.__code__.replace(co_name=f"forward_{mode}"), fn.__globals__,
                                f"forward_{mode}", fn.__defaults__, fn.__closure__)
        self.compiled = torch.compile(fn, backend=backend, fullgraph=True,
                                      dynamic=None if case.get("dynamic_axes") else False)

    def inspect_call(self, inputs, expected, dtype):
        import torch
        from torch._inductor.graph import GraphLowering

        artifacts = f"launches_{self.inspections:04d}"
        self.inspections += 1
        recorder = make_recorder(self.directory / artifacts)
        recorder.install()
        try:
            sources = []
            # run_and_get_code() resets Dynamo and would invalidate both the
            # paired baseline cache and the dynamic reuse experiment.
            with patch.object(GraphLowering, "save_output_code", sources.append):
                output = self.compiled(*inputs)
            torch.npu.synchronize()
            for text in sources:
                (self.directory / f"output_code_{self.sources}.py").write_text(text, encoding="utf-8")
                self.sources += 1
            accuracy(to_cpu(output), expected, dtype)
            # Capture ONE steady launch after compilation/autotune, not their trial launches.
            recorder.start_capture()
            output = self.compiled(*inputs)
            torch.npu.synchronize()
            selected = recorder.stop_capture()
            check = accuracy(to_cpu(output), expected, dtype)
        finally:
            recorder.uninstall()  # no Python observation overhead in timed calls
            write_json(self.directory / "lowering.json", dict(options=self.options,
                       decisions=self.controller.decisions, splits=self.controller.splits))
        record = dict(compiles=self.compiles, active_compile_id=self.active_compile_id,
                      selected=selected, launches=recorder.launches, accuracy=check, launch_artifacts=artifacts)
        return record


def checked_reference(case, pair):
    expected = reference(case, pair)
    # prims.xor_sum deliberately has no eager CPU/NPU implementation. Its
    # independent bitwise tree oracle is used for both compiled strategies.
    if case["op"] != "xor_sum":
        accuracy(to_cpu(forward_function(case)(*pair)), expected, case["dtype"])
    return expected


def check_special_patterns(case, x, r, strategies, device, settings, cores):
    """Reuse compiled code for untimed identity/tail/partition correctness."""
    if case["op"] not in ("prod", "any", "all", "xor_sum"):
        return []
    s = split_count(x, r, cores, settings["selector"])
    chunk = math.ceil(r / s)
    positions = sorted({p for i in range(1, s) for p in (i * chunk - 1, i * chunk) if p < r})
    checks = []
    patterns = ("identity", "tail", "partition_edges", "zero_tail") if case["op"] == "prod" else ("identity", "tail", "partition_edges")
    for pattern in patterns:
        inputs = make_inputs(case, x, r, 1, device, settings["seed"], pattern, positions)
        expected = checked_reference(case, inputs[0])
        result = {}
        for mode, strategy in strategies.items():
            output = strategy.compiled(*inputs[0])
            result[mode] = accuracy(to_cpu(output), expected, case["dtype"])
        checks.append(dict(pattern=pattern, positions=positions if pattern == "partition_edges" else [], results=result))
    return checks


def run_case(job, out):
    import torch

    if job["source_hashes"] != source_hashes():
        raise RuntimeError("suite source changed after manifest creation")
    settings, case = job["settings"], job["case"]
    require_profiler_settings(settings)
    environment = initialize(settings["device"], settings.get("lowering_profile", "installed"))
    fingerprint = environment["fingerprint"]
    if fingerprint != job["environment"]:
        raise RuntimeError("worker environment differs from preflight")
    write_json(out / "environment.json", environment)
    cores = fingerprint["vector_cores"]
    table = job.get("table", {})
    if settings["policy"] == "table" and "policy" in settings["modes"]:
        validate_table(table, fingerprint, settings["selector"])
        if table.get("suite_source_hashes") != source_hashes():
            raise ValueError("table suite code differs from current code; recalibrate")
        if table.get("max_autotune") != settings["max_autotune"]:
            raise ValueError("table max_autotune setting differs")
        if table.get("timing_contract") != timing_contract(settings):
            raise ValueError("table profiler timing contract differs; recalibrate")
        if table.get("domain", {}).get("contexts") and case["op"] not in table["domain"]["contexts"]:
            raise ValueError("forward context was not covered by table calibration")
        for key, current in (("executions", settings["executions"]), ("cache_modes", settings["cache_modes"])):
            if set(current) - set(table["domain"][key]):
                raise ValueError(f"table did not cover requested {key}")
    device = f"npu:{settings['device']}"
    torch.manual_seed(settings["seed"])
    torch._dynamo.config.suppress_errors = False
    strategies = {mode: Strategy(mode, settings, case, cores, table, out / mode) for mode in settings["modes"]}
    record = dict(schema="npu-split-result-v3", case=case, settings=settings, environment=fingerprint, table_digest=digest(table),
                  device_assignment=environment["device_assignment"],
                  source_hashes=job["source_hashes"], status="RUNNING", probes=[])
    write_json(out / "result.json", record)
    if input_bytes(case, case["X"], case["R"]) + case["X"] * case["R"] * 32 > settings["max_working_mib"] * 1024**2:
        record["status"] = "SKIP_MEMORY"
        write_json(out / "result.json", record)
        return
    seed_inputs = make_inputs(case, case["X"], case["R"], 1, device, settings["seed"])
    if case.get("dynamic_axes"):
        for tensor in seed_inputs[0]:
            for axis in case["dynamic_axes"]:
                torch._dynamo.mark_dynamic(tensor, axis)
    with torch.no_grad():
        seed_expected = checked_reference(case, seed_inputs[0])
        record["seed"] = {mode: strategy.inspect_call(seed_inputs[0], seed_expected, case["dtype"])
                          for mode, strategy in strategies.items()}
        del seed_inputs, seed_expected
        for probe_id, (x, r) in enumerate(shapes(case)):
            for cache in settings["cache_modes"]:
                buffers = 1 if cache == "hot" else settings["rotate_buffers"]
                estimate = estimated_working_bytes(case, x, r, settings)
                if estimate > settings["max_working_mib"] * 1024**2:
                    record["probes"].append(dict(probe_id=probe_id, X=x, R=r, cache=cache, status="SKIP_MEMORY",
                                                 estimated_bytes=estimate))
                    continue
                inputs = make_inputs(case, x, r, buffers, device, settings["seed"] + probe_id)
                expected = [checked_reference(case, pair) for pair in inputs]
                metadata = {mode: strategy.inspect_call(inputs[0], expected[0], case["dtype"])
                            for mode, strategy in strategies.items()}
                for mode, strategy in strategies.items():
                    for pair, reference in zip(inputs[1:], expected[1:]):
                        accuracy(to_cpu(strategy.compiled(*pair)), reference, case["dtype"])
                for execution in settings["executions"]:
                    pattern_checks = check_special_patterns(case, x, r, strategies, device, settings, cores)
                    before = {mode: s.compiles for mode, s in strategies.items()}
                    active_before = {mode: s.active_compile_id for mode, s in strategies.items()}
                    samples, profiles = benchmark(strategies, inputs, metadata, settings, settings["seed"] + probe_id,
                        out / "profiles" / f"probe-{probe_id:03d}-{cache}", out)
                    for strategy in strategies.values():
                        for pair, target in zip(inputs, expected):
                            accuracy(to_cpu(strategy.compiled(*pair)), target, case["dtype"])
                    after = {mode: s.compiles for mode, s in strategies.items()}
                    stable = before == after
                    inspected = all(before[mode] == info["compiles"] and active_before[mode] == info["active_compile_id"]
                                    for mode, info in metadata.items())
                    probe = dict(probe_id=probe_id, X=x, R=r, cache=cache, execution=execution,
                                 status=("RECOMPILED_AFTER_INSPECTION" if not inspected else
                                         "PASS" if stable else "RECOMPILED_DURING_TIMING"),
                                 buffers=buffers, input_allocated_bytes=sum(t.untyped_storage().nbytes() for pair in inputs for t in pair),
                                 input_working_set_bytes=sum(t.untyped_storage().nbytes() for pair in inputs for t in
                                     (pair if case["op"] in ("add_sum", "sum_amax") or case["layout"] == "broadcast" else pair[:1])),
                                 inputs=[dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype)) for t in inputs[0]],
                                 metadata=metadata, samples=samples, profiles=profiles, correctness_patterns=pattern_checks,
                                 reference_kind="independent_cpu_xor_tree" if case["op"] == "xor_sum" else "cpu_and_npu_eager",
                                 compiles_before=before, compiles_after=after,
                                 active_before=active_before,
                                 reused_seed_graph={mode: info["active_compile_id"] == record["seed"][mode]["active_compile_id"]
                                                    for mode, info in metadata.items()})
                    record["probes"].append(probe)
                    record["strategies"] = {mode: dict(compiles=s.compiles, sources=s.sources,
                        decisions=s.controller.decisions, splits=s.controller.splits) for mode, s in strategies.items()}
                    write_json(out / "result.json", record)
                del inputs, expected, metadata
                gc.collect()
                torch.npu.empty_cache()
        record["strategies"] = {mode: dict(compiles=s.compiles, sources=s.sources,
            decisions=s.controller.decisions, splits=s.controller.splits) for mode, s in strategies.items()}
    record["status"] = "PASS" if record["probes"] and all(p["status"] == "PASS" for p in record["probes"]) else "INCOMPLETE"
    write_json(out / "result.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lowering-profile", choices=("installed", "ordinary_ir"), default="installed")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.probe:
            write_json(args.out / "environment.json", initialize(args.device, args.lowering_profile))
        else:
            run_case(read_json(args.job), args.out)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        write_json(args.out / "error.json", dict(status="ERROR", traceback=error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
