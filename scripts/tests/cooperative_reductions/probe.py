"""Focused A5 cooperative-reduction research; no installed files are patched."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback


SOURCES = {
    "torch/_inductor/choices.py": ["def should_use_cooperative_reduction"],
    "torch/_inductor/codegen/triton.py": ["def init_cooperative_reduction", "def get_backend_features"],
    "torch_npu/_inductor/ir.py": ["def num_splits"],
    "torch_npu/_inductor/codegen/scheduling.py": ["class NPUTritonScheduling", "def create_kernel_choices"],
    "torch_npu/_inductor/codegen/npu_combined_scheduling.py": ["def codegen_node", "def benchmark_fused_nodes"],
    "torch_npu/_inductor/compile_options.py": ["def register_npu_backend"],
    "torch_npu/_inductor/codegen/triton.py": ["def _get_heuristic", "def codegen_kernel"],
    "torch_npu/_inductor/codegen/split_tiling.py": ["def select_split_axis"],
    "torch_npu/_inductor/runtime/triton_heuristics.py": ["class GridNpu", "def reduction("],
    "torch_npu/_inductor/runtime/triton_helpers.py": ["def welford"],
    "triton/backends/ascend/compiler.py": ["class NPUOptions", "def add_stages", "def parse_options"],
    "triton/backends/ascend/driver.py": ["def get_current_target", "syncBlockLock", "launch_cooperative_grid"],
    "triton/language/extra/cann/extension/core.py": ["def sync_block_all"],
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def inspect_stack(out):
    site = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    manifest = {"python": sys.version, "packages": {}, "source_sha256": {}, "version_py": {}}
    for name in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            manifest["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"][name] = None
    for name in ("torch/version.py", "torch_npu/version.py"):
        values = {}
        for node in ast.parse((site / name).read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("__version__", "git_version"):
                        values[target.id] = node.value.value
        manifest["version_py"][name] = values
    excerpts = []
    for name, anchors in SOURCES.items():
        path = site / name
        if not path.exists():
            manifest["source_sha256"][name] = None
            continue
        content = path.read_bytes()
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
        lines = content.decode("utf-8-sig").splitlines()
        selected = set()
        for index, line in enumerate(lines):
            if any(anchor in line for anchor in anchors):
                selected.update(range(max(0, index - 2), min(len(lines), index + 100)))
        excerpts.append(f"\n===== {name} =====\n")
        previous = -2
        for index in sorted(selected):
            if index != previous + 1:
                excerpts.append("...\n")
            excerpts.append(f"{index + 1}: {lines[index]}\n")
            previous = index
    for tool in ("bishengir-compile", "triton-adapter-opt"):
        binary = shutil.which(tool)
        if binary:
            result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
            manifest[tool] = {"binary": binary, "version": result.stdout + result.stderr}
    manifest["ASCEND_HOME_PATH"] = os.getenv("ASCEND_HOME_PATH")
    write_json(out / "stack.json", manifest)
    (out / "installed-source-excerpts.txt").write_text("".join(excerpts))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def define_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cann import extension as ext
    from torch._inductor.runtime.triton_helpers import x_grid_barrier

    # JIT resolves globals, so expose these imports in the module dictionary.
    globals().update(triton=triton, tl=tl, ext=ext, x_grid_barrier=x_grid_barrier)

    @triton.jit
    def partial_sum(A, W, R: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1)
        chunk = tl.cdiv(R, S * B) * B
        acc = tl.full((B,), 0, tl.float32)
        for start in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), B):
            offsets = start + tl.arange(0, B)
            acc += tl.load(A + row * R + offsets, offsets < R, other=0).to(tl.float32)
        tl.store(W + row * S + shard, tl.sum(acc, 0))

    @triton.jit
    def finish_sum(W, Y, S: tl.constexpr, P: tl.constexpr):
        row = tl.program_id(0)
        peers = tl.arange(0, P)
        values = tl.load(W + row * S + peers, peers < S, other=0)
        tl.store(Y + row, tl.sum(values, 0))

    @triton.jit
    def cooperative_sum(A, W, Y, R: tl.constexpr, S: tl.constexpr, B: tl.constexpr, P: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1)
        chunk = tl.cdiv(R, S * B) * B
        acc = tl.full((B,), 0, tl.float32)
        for start in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), B):
            offsets = start + tl.arange(0, B)
            acc += tl.load(A + row * R + offsets, offsets < R, other=0).to(tl.float32)
        tl.store(W + row * S + shard, tl.sum(acc, 0))
        ext.sync_block_all("all_vector", 0)
        peers = tl.arange(0, P)
        values = tl.load(W + row * S + peers, peers < S, other=0)
        tl.store(Y + row, tl.sum(values, 0), shard == 0)

    @triton.jit
    def native_visibility(W, Y, P: tl.constexpr, REPEATS: tl.constexpr):
        pid = tl.program_id(0)
        peers = tl.arange(0, P)
        for phase in range(REPEATS):
            tl.store(W + pid, pid + 1 + phase)
            ext.sync_block_all("all_vector", 0)
            values = tl.load(W + peers)
            tl.store(Y + phase * P + pid, tl.sum(values, 0))
            # All readers must finish before the next phase overwrites W.
            ext.sync_block_all("all_vector", 1)

    @triton.jit
    def cuda_barrier(W, Sem, Y):
        pid = tl.program_id(0)
        tl.store(W + pid, pid + 1)
        x_grid_barrier(Sem)
        tl.store(Y + pid, tl.load(W + pid))

    return partial_sum, finish_sum, cooperative_sum, native_visibility, cuda_barrier


def compile_probes(args, out):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.backends.ascend.utils import _get_npucompiler_path

    os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
    compiler, compiler_env = _get_npucompiler_path()
    version = subprocess.run([compiler, "--version"], env=compiler_env,
                             capture_output=True, text=True, timeout=30)
    write_json(out / "actual-npuir-compiler.json", {"compiler": compiler,
                                                   "version": version.stdout + version.stderr})
    _, _, cooperative_sum, native_visibility, cuda_barrier = define_kernels()
    probes = [
        (cooperative_sum, {"A": "*fp32", "W": "*fp32", "Y": "*fp32"},
         {"R": 1048576, "S": 8, "B": 1024, "P": 8}),
        (native_visibility, {"W": "*fp32", "Y": "*fp32"}, {"P": 8, "REPEATS": 8}),
        (cuda_barrier, {"W": "*fp32", "Sem": "*u32", "Y": "*fp32"}, {}),
    ]
    results = []
    for mode in args.compile_modes.split(","):
        for kernel, signature, constants in probes:
            case = out / f"{kernel.__name__}-{mode}"
            case.mkdir(exist_ok=True)
            record = {"kernel": kernel.__name__, "target": args.arch, "compile_mode": mode,
                      "signature": signature, "constexprs": constants, "launched": False}
            try:
                compiled = triton.compile(
                    ASTSource(kernel, signature=signature, constexprs=constants),
                    target=GPUTarget("npu", args.arch, 0), options={"compile_mode": mode, "debug": True},
                )
                for name, value in compiled.asm.items():
                    if isinstance(value, str):
                        (case / f"kernel.{name}").write_text(value)
                record.update(status="PASS", metadata=compiled.metadata._asdict())
            except Exception:
                record.update(status="FAIL", error=traceback.format_exc())
                (case / "error.txt").write_text(record["error"])
            results.append(record)
            write_json(out / "compile.json", results)
            print(json.dumps(record, default=str), flush=True)


def init_device(args, out):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    prop = torch.npu.get_device_properties(args.device)
    write_json(out / "device.json", {"logical_device": args.device,
                                     "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                                     "properties": str(prop), "name": torch.npu.get_device_name(args.device),
                                     "cube_core_num": prop.cube_core_num,
                                     "vector_core_num": prop.vector_core_num})
    torch.manual_seed(1234)
    return torch, torch_npu


def event_time(torch, fn, *, graph=False):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    batch = 32
    if graph:
        graph_obj = torch.npu.NPUGraph()
        with torch.npu.graph(graph_obj):
            for _ in range(batch):
                fn()
        fn = graph_obj.replay
        denominator = batch * 10
        launches = 10
        for _ in range(3):
            fn()
        torch.npu.synchronize()
    else:
        denominator = batch
        launches = batch
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(launches):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / denominator)
    return {"median_us": statistics.median(samples), "samples_us": samples,
            "method": "NPU graph + events" if graph else "events + Python enqueue"}


def parse_shapes(text):
    return [tuple(map(int, shape.split("x"))) for shape in text.split(",")]


def bench_probes(args, out):
    torch, _ = init_device(args, out)
    import triton

    partial_sum, finish_sum, cooperative_sum, native_visibility, _ = define_kernels()
    cores = torch.npu.get_device_properties(args.device).vector_core_num
    # Native global barriers are only launched within the physical core count.
    visibility = []
    for programs in (1, 2, 8):
        if programs > cores:
            continue
        workspace = torch.empty(programs, device="npu")
        output = torch.empty((8, programs), device="npu")
        native_visibility[(programs,)](workspace, output, P=programs, REPEATS=8, compile_mode="simd")
        expected = torch.tensor([programs * (programs + 1) / 2 + phase * programs for phase in range(8)])
        torch.testing.assert_close(output.cpu(), expected[:, None].expand(8, programs))
        visibility.append({"programs": programs, "phases": 8, "status": "PASS"})
        write_json(out / "native-visibility.json", visibility)
        print("VISIBILITY", programs, "PASS", flush=True)
    records = []
    for rows, reduction in parse_shapes(args.shapes):
        ratio = max(1, cores // rows)
        splits = min(64, 1 << (ratio.bit_length() - 1))
        if args.splits:
            splits = args.splits
        if rows * splits > cores:
            raise ValueError("Native-barrier experiment requires rows * splits <= vector_core_num")
        power = triton.next_power_of_2(splits)
        data = torch.randn((rows, reduction), device="npu", dtype=getattr(torch, args.dtype))
        workspace = torch.empty((rows, splits), device="npu", dtype=torch.float32)
        result = torch.empty(rows, device="npu", dtype=torch.float32)
        reference = data.float().sum(-1).cpu()

        def single():
            partial_sum[(1, rows)](data, result, R=reduction, S=1, B=1024, compile_mode="simd")

        def split():
            partial_sum[(splits, rows)](data, workspace, R=reduction, S=splits, B=1024, compile_mode="simd")
            finish_sum[(rows,)](workspace, result, S=splits, P=power, compile_mode="simd")

        def coop():
            cooperative_sum[(splits, rows)](
                data, workspace, result, R=reduction, S=splits, B=1024, P=power, compile_mode="simd")

        for name, fn in (("single", single), ("split", split), ("native_cooperative", coop)):
            fn()
            torch.npu.synchronize()
            torch.testing.assert_close(result.cpu(), reference, atol=2e-2, rtol=2e-4)
            record = {"strategy": name, "rows": rows, "R": reduction, "splits": splits,
                      "dtype": args.dtype, "B": 1024, "compile_mode": "simd", "correct": True,
                      **event_time(torch, fn, graph=args.graph)}
            records.append(record)
            write_json(out / "bench.json", records)
            print(json.dumps(record), flush=True)


def inductor_probes(args, out):
    torch, _ = init_device(args, out)
    from torch._inductor import config, metrics
    from torch._inductor.utils import run_and_get_code
    from torch._inductor.graph import GraphLowering
    from unittest import mock

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from autotune_tiling import BestTilingRecorder

    records = []
    modes = {"baseline": (False, False, True), "unsplit": (False, False, False),
             "enabled": (True, False, True), "forced": (True, True, True)}
    for mode in args.inductor_modes.split(","):
        coop, force, split = modes[mode]
        for rows, reduction in parse_shapes(args.shapes):
            case = out / f"{mode}-{rows}x{reduction}-dynamic{int(args.dynamic)}"
            case.mkdir(exist_ok=True)
            torch._dynamo.reset()
            metrics.reset()
            data = torch.randn((rows, reduction), device="npu", dtype=getattr(torch, args.dtype))
            other = torch.randn_like(data)
            reference = (data + other).sum(-1).cpu()
            record = {"mode": mode, "rows": rows, "R": reduction, "dynamic": args.dynamic,
                      "dtype": args.dtype, "operation": "(a + b).sum(-1)",
                      "cooperative_reductions": coop, "force_cooperative_reductions": force,
                      "split_reductions": split}
            recorder = BestTilingRecorder("npu")
            recorder.install()
            recorder.start_capture()
            try:
                with config.patch({"triton.cooperative_reductions": coop,
                                   "triton.force_cooperative_reductions": force, "split_reductions": split}):
                    compiled = torch.compile(lambda a, b: (a + b).sum(-1), fullgraph=True, dynamic=args.dynamic)
                    result, sources = run_and_get_code(compiled, data, other)
                    torch.npu.synchronize()
                    torch.testing.assert_close(result.cpu(), reference, atol=2e-2, rtol=2e-4)
                    for index, source in enumerate(sources):
                        (case / f"output_code_{index}.py").write_text(source)
                    if args.dynamic and args.reuse_r:
                        record["dynamic_reuse"] = []
                        for next_r in map(int, args.reuse_r.split(",")):
                            next_data = torch.randn((rows, next_r), device="npu", dtype=data.dtype)
                            next_other = torch.randn_like(next_data)
                            before = metrics.generated_kernel_count
                            next_sources = []
                            # run_and_get_code resets Dynamo; reuse must call the original callable directly.
                            with mock.patch.object(GraphLowering, "save_output_code", next_sources.append):
                                actual = compiled(next_data, next_other)
                            torch.testing.assert_close(actual.cpu(), (next_data + next_other).sum(-1).cpu(),
                                                       atol=2e-2, rtol=2e-4)
                            for index, source in enumerate(next_sources):
                                (case / f"reuse_r{next_r}_code_{index}.py").write_text(source)
                            record["dynamic_reuse"].append({"R": next_r, "correct": True,
                                                           "new_kernels": metrics.generated_kernel_count - before})
                    record.update(status="PASS", generated_kernel_count=metrics.generated_kernel_count,
                                  **event_time(torch, lambda: compiled(data, other), graph=args.graph))
            except Exception:
                record.update(status="FAIL", error=traceback.format_exc())
                (case / "error.txt").write_text(record["error"])
            finally:
                record["selected_configs"] = recorder.stop_capture()
                recorder.uninstall()
            records.append(record)
            write_json(out / "inductor.json", records)
            print(json.dumps(record, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inspect", "compile", "bench", "inductor"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="Ascend950")
    parser.add_argument("--compile-modes", default="simd,simd_simt_template,simt_only")
    parser.add_argument("--inductor-modes", default="baseline,enabled,forced")
    parser.add_argument("--shapes", default="1x1048576")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--splits", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--reuse-r", default="", help="Additional R values for the same dynamic compiled callable")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.out / "inductor_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out / "triton_cache"))
    os.environ.setdefault("TRITON_DUMP_DIR", str(args.out / "triton_dump"))
    write_json(args.out / "command.json", {"argv": sys.argv, "cwd": os.getcwd()})
    phases = {"inspect": inspect_stack, "compile": lambda out: compile_probes(args, out),
              "bench": lambda out: bench_probes(args, out), "inductor": lambda out: inductor_probes(args, out)}
    phases[args.phase](args.out)


if __name__ == "__main__":
    main()
