"""Preflight, freeze sources, run paired CUDA/NPU tests, plot and audit the evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True

from common import cases, write_json
from pack import source_files


HERE = Path(__file__).resolve().parent
SCHEMA = "reduction-strategy-pk-one-click-v1"
GIB = 1024 ** 3


def split_devices(value):
    items = [item.strip() for item in value.split(",")] if value else []
    if any(not item for item in items) or len(set(items)) != len(items):
        raise ValueError("Device lists must not contain empty or duplicate selectors")
    return items


def smoke_cases(args, vector_cores=56):
    x = 1 if args.backend == "npu" and vector_cores < 14 else 7
    r = 32768
    if args.max_elements and x * r > args.max_elements:
        x = 1
        allowed = [case["R"] for case in cases() if case["group"] == "core" and case["X"] == 1
                   and case["R"] <= min(r, args.max_elements)]
        if not allowed:
            return []
        r = max(allowed)
    return [case for case in cases() if case["group"] == "core" and case["X"] == x and case["R"] == r]


def selected_cases(args, vector_cores=56):
    selected = cases()
    if args.case_ids:
        ids = split_devices(args.case_ids)
        selected = [case for case in selected if case["id"] in ids]
        if len(selected) != len(ids):
            raise ValueError("Unknown case ID; see run.py --emit-manifest")
    elif args.smoke:
        selected = smoke_cases(args, vector_cores)
    if args.max_elements:
        selected = [case for case in selected if case["X"] * case["R"] <= args.max_elements]
    if not selected:
        raise ValueError("The requested case selection is empty")
    return selected


def inventory_worker(job):
    import torch
    import triton

    backend = job["backend"]
    if backend == "npu":
        import torch_npu
    from torch._inductor import config, ir
    from torch._inductor.utils import run_and_get_code

    required = ("split_reductions", "triton.cooperative_reductions", "triton.force_cooperative_reductions",
                "triton.multi_kernel", "max_autotune", "triton.cudagraphs", "fx_graph_cache", "autotune_local_cache")
    for name in required:
        getattr(config, name)
    for name in ("Reduction", "OnlineSoftmaxReduction", "WelfordReduction"):
        if "create_multilayer" not in getattr(ir, name).__dict__:
            raise RuntimeError(f"Unsupported Inductor API: {name}.create_multilayer")
    if not callable(run_and_get_code):
        raise RuntimeError("Missing torch._inductor.utils.run_and_get_code")
    sys.path.insert(0, str(source_files()["autotune_tiling.py"].parent))
    from autotune_tiling import BestTilingRecorder
    recorder = BestTilingRecorder(backend)
    recorder.install()
    recorder.uninstall()
    if not job["no_plot"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
        import numpy  # noqa: F401
    api = getattr(torch, backend)
    if not api.is_available():
        raise RuntimeError(f"{backend} is unavailable in this Python environment")
    if not hasattr(api, "NPUGraph" if backend == "npu" else "CUDAGraph") or not hasattr(api, "graph"):
        raise RuntimeError("This backend does not expose the required Graph capture/replay API")
    versions = dict(python=sys.version.split()[0], torch=torch.__version__, torch_commit=torch.version.git_version,
                    triton=triton.__version__, cuda=torch.version.cuda)
    modules = ["torch._inductor.choices", "torch._inductor.ir", "torch._inductor.codegen.triton"]
    devices = []
    if backend == "cuda":
        visible = []
        for index in range(api.device_count()):
            prop = api.get_device_properties(index)
            uuid = "GPU-" + str(prop.uuid).removeprefix("GPU-")
            visible.append(dict(selector=uuid, logical_device=index, name=prop.name,
                                sm_count=prop.multi_processor_count, capability=[prop.major, prop.minor]))
        selectors = job["selectors"] or [str(index) for index in range(len(visible))]
        for selector in selectors:
            if selector.isdecimal():
                if int(selector) >= len(visible):
                    raise ValueError(f"CUDA logical index {selector} is outside the currently visible devices")
                device = dict(visible[int(selector)])
            else:
                matches = [item for item in visible if item["selector"] == selector]
                if not matches:
                    raise ValueError(f"CUDA UUID is not currently visible: {selector}")
                device = dict(matches[0])
            api.set_device(device["logical_device"])
            device["free_bytes"], device["total_bytes"] = api.mem_get_info()
            device["requested_selector"] = selector
            devices.append(device)
    else:
        from triton.backends.ascend.driver import NPUUtils
        from triton.language.extra.cann import extension
        if not hasattr(extension, "sync_block_all"):
            raise RuntimeError("This Ascend Triton build lacks extension.sync_block_all")
        if not hasattr(torch_npu.profiler.ProfilerLevel, "Level0") or not hasattr(api, "get_device_limit"):
            raise RuntimeError("Missing torch_npu Level0 profiler or device-limit API")
        api.set_device(0)
        physical = ctypes.c_int32(-1)
        query = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
        query.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        query.restype = ctypes.c_int32
        status = query(0, ctypes.byref(physical))
        if status or str(physical.value) != job["selectors"][0]:
            raise RuntimeError(f"NPU physical mapping mismatch: status={status}, physical={physical.value}")
        prop = api.get_device_properties(0)
        free, total = api.mem_get_info()
        devices.append(dict(selector=str(physical.value), physical_device=physical.value,
                            requested_selector=job["selectors"][0], name=prop.name,
                            free_bytes=free, total_bytes=total, vector_cores=prop.vector_core_num,
                            effective_vector_cores=NPUUtils().get_device_properties(0)["num_vectorcore"],
                            device_limit=api.get_device_limit(0)))
        versions["torch_npu"] = torch_npu.__version__
        modules += ["torch_npu._inductor.codegen.triton", "torch_npu._inductor.ir"]
    installed = []
    for name in modules:
        path = Path(inspect.getfile(importlib.import_module(name)))
        installed.append(dict(module_path=name.replace(".", "/") + ".py",
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len({device["selector"] for device in devices}) != len(devices):
        raise ValueError("Duplicate physical devices after resolving selectors")
    return dict(status="PASS", backend=backend, versions=versions, devices=devices,
                installed_source=installed, performance="No performance data collected by preflight")


def run_inventory(out, backend, selectors, env, no_plot, timeout):
    job = out.with_suffix(".job.json")
    log_path = out.with_suffix(".log")
    write_json(job, dict(backend=backend, selectors=selectors, no_plot=no_plot))
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(HERE / "one_click.py"), "--inventory-job", str(job),
                                 "--inventory-out", str(out)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout)
    if not out.exists():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:])
        raise RuntimeError(f"Preflight exited with code {result.returncode} without a result; "
                           f"see {log_path}\n{tail[-2000:]}")
    record = json.loads(out.read_text(encoding="utf-8"))
    if result.returncode or record["status"] != "PASS":
        raise RuntimeError(f"Preflight failed: {record.get('error', 'unknown error')}; "
                           f"see {out.with_suffix('.log')}")
    return record


def discover_npu_devices():
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        return split_devices(visible)
    result = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, timeout=20, check=True)
    devices = re.findall(r"NPU ID\s*:\s*(\d+)", result.stdout)
    if not devices:
        raise RuntimeError("Cannot parse npu-smi physical IDs; provide --devices for inventory")
    return list(dict.fromkeys(devices))


def preflight(args, out):
    environment = dict(os.environ)
    environment["PK_PROFILE_GRAPH"] = "1"
    if args.backend == "cuda":
        if args.npu_vector_cores is not None:
            raise ValueError("--npu-vector-cores applies only to NPU")
        record = run_inventory(out / "inventory.json", "cuda", args.devices, environment,
                               args.no_plot or args.list_devices, args.preflight_timeout)
        selectors = [device["selector"] for device in record["devices"]]
        requested = {device["requested_selector"]: device["selector"] for device in record["devices"]}
        small = []
        for selector in args.small_devices:
            resolved = requested.get(selector, selector)
            if resolved not in selectors:
                raise ValueError(f"--small-devices is not a selected CUDA device: {selector}")
            small.append(resolved)
        if len(set(small)) != len(small):
            raise ValueError("Duplicate physical small devices")
        record["small_devices"] = small
        return record, environment
    if args.small_devices:
        raise ValueError("--small-devices applies only to CUDA")
    if not shutil.which("npu-smi"):
        raise RuntimeError("npu-smi is required for bounded HBM occupancy checks")
    selectors = args.devices or discover_npu_devices()
    visible = split_devices(os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""))
    if visible and not set(selectors) <= set(visible):
        raise ValueError("NPU selectors are physical IDs and must be in ASCEND_RT_VISIBLE_DEVICES if it is set")
    records = []
    for selector in selectors:
        if not selector.isdecimal():
            raise ValueError("NPU --devices must contain physical integer IDs")
        env = {**environment, "ASCEND_RT_VISIBLE_DEVICES": selector}
        records.append(run_inventory(out / f"inventory-{selector}.json", "npu", [selector], env,
                                     args.no_plot or args.list_devices, args.preflight_timeout))
    budget = args.npu_vector_cores or min(record["devices"][0]["effective_vector_cores"] for record in records)
    if budget < 2 or any(record["devices"][0]["effective_vector_cores"] < budget for record in records):
        raise ValueError("Declared Vector budget must be >=2 and must not exceed any selected device's effective budget")
    if budget % 2:
        raise ValueError("NPU Vector budget must be even: torch_npu requires Vector = 2 * Cube in NPU_DEVICE_LIMIT")
    if any(record["devices"][0]["effective_vector_cores"] != budget for record in records):
        cube = budget // 2
        if any(record["devices"][0]["device_limit"]["cube_core_num"] < cube for record in records):
            raise ValueError("The paired Cube budget exceeds a selected device's effective Cube limit")
        environment["NPU_DEVICE_LIMIT"] = f"{cube},{budget}"
        records = [run_inventory(out / f"limited-inventory-{selector}.json", "npu", [selector],
                   {**environment, "ASCEND_RT_VISIBLE_DEVICES": selector}, args.no_plot or args.list_devices,
                   args.preflight_timeout) for selector in selectors]
    if any(record["devices"][0]["effective_vector_cores"] != budget for record in records):
        raise RuntimeError("NPU_DEVICE_LIMIT did not produce the declared Vector budget")
    first = records[0]
    if any(record["versions"] != first["versions"] or record["installed_source"] != first["installed_source"]
           for record in records):
        raise RuntimeError("Selected NPU environments differ")
    return {**first, "devices": [record["devices"][0] for record in records],
            "npu_vector_cores": budget, "small_devices": []}, environment


def check_memory(args, inventory, selected):
    small = set(inventory["small_devices"])
    devices = inventory["devices"]
    for case in selected:
        if (args.backend == "cuda" and case["X"] * case["R"] > 1048576 and
                all(device["selector"] in small for device in devices)):
            raise ValueError("At least one regular GPU is required for the selected large cases")
    for device in devices:
        # Workers take eligible cases dynamically; each device must be checked
        # against the largest case it could receive, rather than a predicted queue.
        queue = [case for case in selected if args.backend == "npu" or case["X"] * case["R"] <= 1048576
                 or device["selector"] not in small]
        estimate = 0
        for case in queue:
            n = case["X"] * case["R"]
            dtype_bytes = 4 if case["dtype"] == "float32" else 2
            input_bytes = 2 * n * dtype_bytes * (2 if case["layout"] == "stride2" else 1)
            output_bytes = 4 * n if case["op"] in ("rmsnorm", "softmax") else 8 * case["X"]
            batch = 3 if n > 33554432 else 5
            retained = (4 * batch if args.backend == "cuda" else batch + 2) * output_bytes
            estimate = max(estimate, input_bytes + retained + 8 * n + 256 * 1024 ** 2)
        device["estimated_peak_bytes"] = estimate
        device["eligible_cases"] = len(queue)
        available = device["free_bytes"] - 256 * 1024 ** 2
        if args.backend == "cuda":
            fraction = args.small_memory_fraction if device["selector"] in small else args.gpu_memory_fraction
            device["allocator_fraction"] = fraction
            available = min(available, int(device["total_bytes"] * fraction))
        if queue and estimate > available:
            raise ValueError(f"Device {device['selector']} estimated peak {estimate / GIB:.2f} GiB exceeds "
                             f"available/allocator budget {available / GIB:.2f} GiB; use a freer device, "
                             "adjust --gpu-memory-fraction, or explicitly limit --max-elements/--case-ids")
    inventory["memory_estimate_scope"] = "Conservative input, retained Graph outputs and scratch estimate; not a memory reservation or exact peak guarantee"


def freeze_source(out, resume):
    source = out / "source"
    manifest_path = out / "source-sha256.json"
    if resume:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in expected}
        if actual != expected or {path.name for path in source.iterdir() if path.is_file()} != set(expected):
            raise RuntimeError("Frozen source changed; do not resume this directory")
    else:
        source.mkdir()
        for name, path in source_files().items():
            shutil.copyfile(path, source / name)
        write_json(manifest_path, {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(source.iterdir())})
    return source


def execute(command, log_path, environment):
    print("$ " + shlex.join(str(item) for item in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(item) for item in command], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=20)
            raise
        finally:
            process.stdout.close()


def recover_interrupted(root, ids, source):
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            tokens = [part.decode(errors="replace") for part in path.read_bytes().split(b"\0") if part]
        except (OSError, ProcessLookupError):
            continue
        if str(source / "run.py") in tokens:
            raise RuntimeError(f"A runner/worker using this frozen source is still active (PID {path.parent.name})")
    history = root / "resume-interruptions.json"
    records = json.loads(history.read_text(encoding="utf-8")) if history.exists() else []
    for case_id in ids:
        directory = root / "cases" / case_id
        if not directory.exists():
            continue
        result = directory / "result.json"
        if result.exists() and json.loads(result.read_text(encoding="utf-8"))["status"] != "INTERRUPTED":
            continue
        destination = root / "interruptions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") / case_id
        if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Interrupted case path is outside this run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(destination)
        records.append(dict(case=case_id, previous_artifacts=str(destination.relative_to(root)),
                            reason="Resume an interrupted or unfinished case; previous artifacts preserved"))
    if records:
        write_json(history, records)


def run_command(args, source, root, inventory, ids, smoke, environment):
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        recover_interrupted(root, ids, source)
    installed = root / "installed-source.json"
    if not installed.exists():
        write_json(installed, dict(files=inventory["installed_source"]))
    elif json.loads(installed.read_text(encoding="utf-8"))["files"] != inventory["installed_source"]:
        raise RuntimeError("Installed compiler modules changed before resume")
    command = [sys.executable, "-u", source / "run.py", "--backend", args.backend,
               "--devices", ",".join(device["selector"] for device in inventory["devices"]), "--out", root,
               "--case-ids", ",".join(ids), "--small-devices", ",".join(inventory["small_devices"]),
               "--gpu-memory-fraction", str(args.gpu_memory_fraction),
               "--small-memory-fraction", str(args.small_memory_fraction),
               "--npu-vector-cores", str(inventory.get("npu_vector_cores", 56)),
               "--max-hbm-percent", str(args.max_hbm_percent),
               "--resource-wait-timeout", str(args.resource_wait_timeout), "--timeout", str(args.timeout)]
    if smoke:
        command.append("--smoke")
    if args.resume:
        command.append("--resume")
    if not args.legacy_grad_context:
        command.append("--uniform-no-grad")
    return execute(command, root.parent / (root.name + ".log"), environment)


def extract_evidence(archive, destination):
    destination.mkdir()
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not member.isfile():
                raise ValueError(f"Unsafe evidence archive member: {member.name}")
        package.extractall(root, members=members)


def write_report(out, inventory, root, summary, status):
    lines = ["# Split / cooperative 测试结果", "", f"状态：**{status}**；数据目录：`{root.name}`。",
             "", "性能为 profiler 记录的一次完整 forward 的必要 device operation duration 之和；"
             "包含初始化，排除 CPU duration 和 kernel 间空隙。", "",
             "| 实现 | 用例 | 严格配对 | cooperative 胜 | split 胜 | 接近/方向变化 | noisy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for backend, data in summary.items():
        outcomes = data["outcomes"]
        lines.append(f"| {backend} | {data['cases']} | {data['strict_pairs']} | {outcomes.get('cooperative', 0)} | "
                     f"{outcomes.get('split', 0)} | {outcomes.get('close_or_variable', 0)} | {data['noisy']} |")
    lines += ["", "胜负采用超过 5% 且三个会话方向一致的规则；会话中位数 max/min>1.15 单列 noisy。",
              "", "| 设备 | 物理选择器 | 有效 Vector 核 / SM | 已记录用例 |",
              "| --- | --- | ---: | ---: |"]
    recorded = Counter()
    for path in (root / "cases").glob("*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        recorded[str(result.get("environment", {}).get("requested_device", ""))] += 1
    for device in inventory["devices"]:
        lines.append(f"| {device['name']} | `{device['selector']}` | "
                     f"{device.get('effective_vector_cores', device.get('sm_count'))} | {recorded[device['selector']]} |")
    lines += ["", "- [逐用例 CSV](figures/results.csv)", "- [完整 JSON 与配置](figures/results.json)",
              "- [统计摘要](figures/summary.json)", "- [环境与实际设备](preflight/inventory-final.json)",
              "- [冻结源码摘要](source-sha256.json)", "- [运行参数](invocation.json)"]
    for path, label in (("evidence-verification.json", "原始 profiler 与来源审计"),
                        ("cuda-launch-verification.json", "CUDA 实际启动核验"),
                        ("input-fingerprints.json", "三个校验种子的 CPU 输入字节摘要"),
                        ("evidence.tar.gz", "可回收原始证据包")):
        if (out / path).exists():
            lines.append(f"- [{label}]({path})")
    for name, label in (("comparison.svg", "策略对照"), (inventory["backend"] + "_r_sweep.svg", "R 扫描")):
        if (out / "figures" / name).exists():
            lines += ["", f"![{label}](figures/{name})"]
    lines += ["", "编译配置标注为 initial_compile；NPU 候选参数标注为 dsl_launch。"
              "默认全程 no_grad，编译/预热/Graph 使用一致的 grad context。"
              "若使用 --legacy-grad-context，需考虑初始编译与后续预热的再编译差异。"]
    if inventory["backend"] == "npu":
        lines += ["", f"NPU Vector 预算为 {inventory['npu_vector_cores']}，cooperative 的 S 和 X*S 均不得超过该值。"
                  "split 可使用至 2 倍预算的规约分片与独立输出并行度。"
                  "cooperative 使用指定的 debug_barrier / all_vector / debug_barrier；"
                  "结果属于 SIMD DSL 原型，硬件/编译模式的生产同步契约需另外确认。"]
    lines += ["", "不同机器比较前对齐用例、源码摘要、有效核数和运行参数。"
              "显存估计与 allocator 上限不构成资源预留；共享设备的持续干扰仍可能影响计时。", ""]
    (out / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def pipeline(args):
    if args.plan_only:
        selected = selected_cases(args, args.npu_vector_cores or 56)
        print(json.dumps(dict(backend=args.backend, requested_devices=args.devices,
                              output_directory=str(args.out), cases=len(selected),
                              groups=dict(Counter(case["group"] for case in selected)),
                              profiler="Graph replay; complete device operation durations only",
                              automatic_smoke=not args.smoke and not args.skip_smoke,
                              compatibility_smoke_case_ids=[case["id"] for case in smoke_cases(
                                  args, args.npu_vector_cores or 56)] if not args.smoke and not args.skip_smoke else [],
                              case_ids=[case["id"] for case in selected]), indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Device execution requires Linux with compatible PyTorch/Triton; packaging and --plan-only work offline")
    if not args.devices and not args.list_devices:
        raise ValueError("--devices is required; physical devices are never chosen implicitly")
    if args.backend == "npu" and any(not device.isdecimal() for device in args.devices):
        raise ValueError("NPU selectors must be physical integer IDs")
    out = args.out.resolve()
    invocation_path = out / "invocation.json"
    if not args.list_devices and not args.check_only:
        if args.resume != invocation_path.exists():
            raise ValueError("Use a new --out, or supply --resume for an existing invocation")
        if not args.resume and out.exists() and any(out.iterdir()):
            raise ValueError("New --out must be empty; existing files are never overwritten")
    elif invocation_path.exists():
        raise ValueError("Inventory/check-only output must be separate from an existing experiment")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "one-click.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another one-click process is already using this output directory")
        return device_pipeline(args, out)


def device_pipeline(args, out):
    invocation_path = out / "invocation.json"
    preflight_dir = out / ("preflight-resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                          if args.resume else "preflight")
    preflight_dir.mkdir(exist_ok=True)
    inventory, environment = preflight(args, preflight_dir)
    for device in inventory["devices"]:
        print(f"Device {device['selector']}: {device['name']}; free {device['free_bytes'] / GIB:.2f} GiB", flush=True)
    if args.list_devices:
        write_json(preflight_dir / "inventory-final.json", inventory)
        return 0
    selected = selected_cases(args, inventory.get("npu_vector_cores", 56))
    compatibility_smoke = smoke_cases(args, inventory.get("npu_vector_cores", 56)) if not args.smoke and not args.skip_smoke else []
    check_memory(args, inventory, selected + compatibility_smoke)
    write_json(preflight_dir / "inventory-final.json", inventory)
    if args.check_only:
        print(f"Preflight PASS; {len(selected)} cases selected. No performance tests executed.", flush=True)
        return 0
    config = dict(schema=SCHEMA, backend=args.backend,
                  devices=[device["selector"] for device in inventory["devices"]],
                  small_devices=inventory["small_devices"], case_ids=[case["id"] for case in selected],
                  smoke=args.smoke, skip_smoke=args.skip_smoke, uniform_no_grad=not args.legacy_grad_context,
                  gpu_memory_fraction=args.gpu_memory_fraction, small_memory_fraction=args.small_memory_fraction,
                  npu_vector_cores=inventory.get("npu_vector_cores", 56),
                  max_hbm_percent=args.max_hbm_percent, resource_wait_timeout=args.resource_wait_timeout,
                  timeout=args.timeout, profile_graph="1", versions=inventory["versions"],
                  installed_source=inventory["installed_source"])
    if args.resume:
        if json.loads(invocation_path.read_text(encoding="utf-8"))["config"] != config:
            raise RuntimeError("Resume configuration, device identity or installed environment changed")
    else:
        write_json(invocation_path, dict(config=config, started_utc=datetime.now(timezone.utc).isoformat()))
    source = freeze_source(out, args.resume)
    root = out / "run"
    returncode = 0
    if not args.smoke and not args.skip_smoke:
        returncode = run_command(args, source, out / "smoke-check", inventory,
                                 [case["id"] for case in compatibility_smoke], True, environment)
        if returncode:
            root = out / "smoke-check"
            print("Compatibility smoke failed; the requested full matrix was not started.", flush=True)
    if not returncode:
        returncode = run_command(args, source, root, inventory, [case["id"] for case in selected], args.smoke, environment)
    summarize_command = [sys.executable, source / "summarize.py", "--input", root, "--out", out / "figures"]
    if args.no_plot:
        summarize_command.append("--no-plot")
    summary_rc = execute(summarize_command, out / "summarize.log", environment)
    if summary_rc:
        raise RuntimeError(f"Summary failed; see {out / 'summarize.log'}")
    profiled_cases = json.loads((root / "run.json").read_text(encoding="utf-8"))["cases"]
    fingerprint_rc = execute([sys.executable, source / "input_fingerprints.py", "--source", source,
                              "--case-ids", ",".join(case["id"] for case in profiled_cases),
                              "--out", out / "input-fingerprints.json"], out / "fingerprints.log", environment)
    archive = out / "evidence.tar.gz"
    archive_rc = execute([sys.executable, source / "collect.py", "--input", root, "--source", source,
                          "--output", archive], out / "collect.log", environment)
    audit_rc = 1
    if not archive_rc:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        destination = out / ("evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        extract_evidence(archive, destination)
        audit_rc = execute([sys.executable, source / "verify_evidence.py", "--input", destination,
                            "--output", out / "evidence-verification.json"], out / "audit.log", environment)
    launch_rc = 0
    if args.backend == "cuda":
        launch_rc = execute([sys.executable, source / "verify_cuda_launch.py", "--input", root,
                             "--output", out / "cuda-launch-verification.json"], out / "cuda-launch.log", environment)
    summary = json.loads((out / "figures/summary.json").read_text(encoding="utf-8"))
    all_pass = (summary[args.backend]["cases"] == len(selected) and
                summary[args.backend]["status"] == {"PASS": len(selected)} and root.name == "run")
    status = "PASS" if all_pass and not any((returncode, fingerprint_rc, archive_rc, audit_rc, launch_rc)) else "INCOMPLETE"
    write_report(out, inventory, root, summary, status)
    write_json(out / "completion.json", dict(status=status, data_directory=root.name, expected_cases=len(selected),
               observed_cases=summary[args.backend]["cases"], runner_returncode=returncode,
               input_fingerprints_returncode=fingerprint_rc, archive_returncode=archive_rc,
               audit_returncode=audit_rc, cuda_launch_returncode=launch_rc,
               completed_utc=datetime.now(timezone.utc).isoformat()))
    print(f"{status}: {out / 'RUN_SUMMARY.md'}", flush=True)
    return 0 if status == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--devices", default="", help="CUDA: current logical indices/full UUIDs; NPU: physical IDs")
    parser.add_argument("--small-devices", default="", help="Selected CUDA devices restricted to X*R<=1048576")
    parser.add_argument("--out", type=Path, help="External result directory; must be outside torch-learn")
    parser.add_argument("--smoke", action="store_true", help="Only four representative forwards; reduced NPU search")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the automatic four-case compatibility run")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--max-elements", type=int, help="Explicitly limit X*R; resulting coverage is recorded")
    parser.add_argument("--resume", action="store_true", help="Continue missing cases from unchanged frozen sources")
    parser.add_argument("--no-plot", action="store_true", help="Still export CSV/JSON; render figures elsewhere")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.50)
    parser.add_argument("--small-memory-fraction", type=float, default=.035)
    parser.add_argument("--npu-vector-cores", type=int, help="Cap the effective Vector budget; default detects it")
    parser.add_argument("--max-hbm-percent", type=int, default=95)
    parser.add_argument("--resource-wait-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=900, help="Per-case no-progress deadline, not total runtime")
    parser.add_argument("--preflight-timeout", type=float, default=180)
    parser.add_argument("--legacy-grad-context", action="store_true", help="Reproduce the original mixed grad-context protocol")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check APIs, devices and memory; do not execute test forwards")
    parser.add_argument("--plan-only", action="store_true", help="Offline case selection; no device/PyTorch initialization")
    parser.add_argument("--inventory-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inventory-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inventory_job:
        try:
            record = inventory_worker(json.loads(args.inventory_job.read_text(encoding="utf-8")))
        except Exception as error:
            import traceback
            traceback.print_exc()
            record = dict(status="ERROR", error=f"{type(error).__name__}: {error}")
        write_json(args.inventory_out, record)
        return 0 if record["status"] == "PASS" else 2
    try:
        args.devices, args.small_devices = split_devices(args.devices), split_devices(args.small_devices)
        if not (0 < args.gpu_memory_fraction <= 1 and 0 < args.small_memory_fraction <= 1):
            raise ValueError("Allocator fractions must be in (0,1]")
        if args.npu_vector_cores is not None and (args.npu_vector_cores < 2 or args.npu_vector_cores % 2):
            raise ValueError("--npu-vector-cores must be an even integer >=2 (Vector = 2 * Cube)")
        if args.max_elements is not None and args.max_elements <= 0:
            raise ValueError("--max-elements must be positive")
        if args.timeout <= 0 or args.preflight_timeout <= 0 or args.resource_wait_timeout < 0 or not 0 < args.max_hbm_percent <= 100:
            raise ValueError("Invalid timeout or HBM threshold")
        if args.resume and args.out is None:
            raise ValueError("--resume requires an explicit --out")
        if args.out is None:
            parent = Path.home() / "torch-learn-runs" / "reduction-strategy-pk"
            args.out = parent / (args.backend + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.out = args.out.expanduser().resolve()
        repository = next((parent for parent in HERE.parents
                           if (parent / "AGENTS.md").is_file() and
                           (parent / "scripts").is_dir() and (parent / "docs").is_dir()), None)
        if repository is not None and args.out.is_relative_to(repository):
            raise ValueError("--out must be outside torch-learn; keep complete experiment data outside the project")
        return pipeline(args)
    except KeyboardInterrupt:
        print("Interrupted; completed cases are retained. Resume with the same arguments and --resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
