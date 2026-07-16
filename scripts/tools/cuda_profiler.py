"""Extract device-side CUDA kernel durations from PyTorch profiler traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class CudaProfilerConfig:
    """Configuration for collecting PyTorch CUDA profiler traces."""

    output_dir: str | Path = "prof_log"
    wait: int = 0
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    skip_first: int = 0
    activities: tuple[str, ...] = ("CPU", "CUDA")
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False


class TorchCudaProfiler:
    """Collect CUDA Chrome Trace files with a scheduled PyTorch profiler."""

    def __init__(self, output_dir: str | Path = "prof_log", **overrides: object) -> None:
        self.config = replace(CudaProfilerConfig(output_dir=output_dir), **overrides)
        self.trace_paths: list[Path] = []

    def profile(self):
        """Create a scheduled `torch.profiler.profile` context manager."""

        torch = self._import_torch()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def export_trace(profiler) -> None:
            trace_path = output_dir / f"trace_{len(self.trace_paths):03d}.json"
            profiler.export_chrome_trace(str(trace_path))
            self.trace_paths.append(trace_path)

        return torch.profiler.profile(
            activities=[
                getattr(torch.profiler.ProfilerActivity, name)
                for name in self.config.activities
            ],
            schedule=torch.profiler.schedule(
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
                skip_first=self.config.skip_first,
            ),
            on_trace_ready=export_trace,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        )

    def run_steps(
        self,
        step_fn: Callable[[], object],
        steps: int | None = None,
        synchronize: bool = True,
    ) -> None:
        """Run `step_fn` under profiler and call `prof.step()` each iteration."""

        torch = self._import_torch()
        self.trace_paths.clear()
        total_steps = steps if steps is not None else self.default_total_steps()
        with self.profile() as prof:
            for _ in range(total_steps):
                step_fn()
                if synchronize:
                    torch.cuda.synchronize()
                prof.step()

    def default_total_steps(self, redundant_steps: int = 0) -> int:
        """Return the number of loop iterations needed by the schedule."""

        schedule_steps = self.config.wait + self.config.warmup + self.config.active
        return self.config.skip_first + self.config.repeat * schedule_steps + redundant_steps

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchCudaProfiler requires torch to run profiling.") from exc
        return torch


@dataclass(frozen=True)
class CudaKernelRecord:
    """One complete CUDA kernel event from a Chrome Trace JSON file."""

    kernel_name: str
    duration: float
    grid: str
    block: str


class CudaProfileParser:
    """Parse device-side kernel events from a PyTorch CUDA profiler trace."""

    CSV_COLUMNS = ("kernel_name", "duration", "grid", "block")

    def __init__(self, trace_path: str | Path) -> None:
        self.trace_path = Path(trace_path)

    def kernel_records(self) -> list[CudaKernelRecord]:
        """Return all complete events whose phase is `X` and category is `kernel`."""

        records = []
        for event in self._trace_events():
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue

            kernel_name = event.get("name")
            duration = event.get("dur")
            if not isinstance(kernel_name, str) or not kernel_name or not isinstance(duration, (int, float)):
                continue

            args = event.get("args")
            if not isinstance(args, Mapping):
                args = {}
            records.append(
                CudaKernelRecord(
                    kernel_name=kernel_name,
                    duration=float(duration),
                    grid=self._format_launch_dim(args.get("grid")),
                    block=self._format_launch_dim(args.get("block")),
                )
            )
        return records

    def export_kernel_csv(self, output_path: str | Path) -> Path:
        """Write device-side kernel records to CSV and return the output path."""

        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for record in self.kernel_records():
                writer.writerow(
                    {
                        "kernel_name": record.kernel_name,
                        "duration": record.duration,
                        "grid": record.grid,
                        "block": record.block,
                    }
                )
        return csv_path

    def _trace_events(self) -> list[Mapping[str, Any]]:
        with self.trace_path.open(encoding="utf-8") as trace_file:
            trace = json.load(trace_file)

        if isinstance(trace, Mapping):
            events = trace.get("traceEvents")
        elif isinstance(trace, list):
            events = trace
        else:
            events = None
        if not isinstance(events, list):
            raise ValueError(f"{self.trace_path} does not contain a traceEvents list")
        return [event for event in events if isinstance(event, Mapping)]

    @staticmethod
    def _format_launch_dim(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)


def cuda_kernel_label(kernel_name: str) -> str:
    """Return a concise summary label while leaving profiler artifacts intact."""

    if kernel_name.startswith("triton_"):
        return kernel_name

    patterns = (
        (r"::([A-Za-z0-9_]+)_kernel_cuda(?:\(|<)", lambda match: match.group(1)),
        (r"CUDAFunctorOnSelf_([A-Za-z0-9_]+)", lambda match: f"{match.group(1)}_self"),
        (r"CUDAFunctor_([A-Za-z0-9_]+)", lambda match: match.group(1)),
        (r"binary_internal::([A-Za-z0-9_]+)Functor", lambda match: match.group(1).lower()),
    )
    for pattern, formatter in patterns:
        match = re.search(pattern, kernel_name)
        if match:
            return formatter(match)

    concise_name = kernel_name.removeprefix("void ")
    if len(concise_name) <= 96:
        return concise_name
    return f"{concise_name[:93]}..."


def default_output_path(trace_path: Path) -> Path:
    return trace_path.with_name(f"{trace_path.stem}_kernels.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="PyTorch CUDA profiler Chrome Trace JSON")
    parser.add_argument("-o", "--output", type=Path, help="output CSV path")
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.trace)
    CudaProfileParser(args.trace).export_kernel_csv(output_path)
    print(f"exported CUDA kernel records to {output_path}")


if __name__ == "__main__":
    main()
