"""Small helpers for collecting and summarizing torch_npu profiler results.

The collection helper follows the common torch_npu profiler pattern:

    profiler = TorchNpuProfiler("./prof_log")
    with profiler.profile() as prof:
        for _ in range(10):
            run_model()
            torch.npu.synchronize()
            prof.step()

The parser only depends on the Python standard library, so existing profile
outputs can be inspected on a local machine without NPU runtime installed.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


Number = int | float
ShapeKey = str | Callable[[Mapping[str, str]], str | None] | None


@dataclass(frozen=True)
class NpuProfilerConfig:
    """Default torch_npu profiler configuration.

    `profiler_level` defaults to 1 and `activities` defaults to CPU + NPU,
    matching the common debugging profile used for compiled Torch workloads.
    """

    output_dir: str | Path = "prof_log"
    wait: int = 2
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    skip_first: int = 0
    profiler_level: int = 1
    activities: tuple[str, ...] = ("CPU", "NPU")
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = True


@dataclass(frozen=True)
class DurationSummary:
    """Aggregated duration statistics in microseconds."""

    key: str
    count: int
    total_us: float
    mean_us: float
    min_us: float
    max_us: float
    median_us: float


@dataclass(frozen=True)
class StepDurationSummary(DurationSummary):
    """Aggregated step duration statistics for one device."""

    device_id: str
    time_column: str


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _summary(key: str, values: Sequence[Number]) -> DurationSummary:
    durations = [float(value) for value in values]
    if not durations:
        return DurationSummary(key, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return DurationSummary(
        key=key,
        count=len(durations),
        total_us=sum(durations),
        mean_us=statistics.mean(durations),
        min_us=min(durations),
        max_us=max(durations),
        median_us=statistics.median(durations),
    )


class TorchNpuProfiler:
    """Factory for torch_npu profiler contexts with reusable defaults."""

    def __init__(self, output_dir: str | Path = "prof_log", **overrides: object) -> None:
        self.config = replace(NpuProfilerConfig(output_dir=output_dir), **overrides)

    def profile(self):
        """Create a `torch_npu.profiler.profile` context manager."""

        torch_npu = self._import_torch_npu()
        profiler_level = self._enum_value(torch_npu.profiler.ProfilerLevel, "Level", self.config.profiler_level)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=profiler_level,
        )
        return torch_npu.profiler.profile(
            activities=[self._activity(torch_npu, name) for name in self.config.activities],
            schedule=torch_npu.profiler.schedule(
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
                skip_first=self.config.skip_first,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.config.output_dir)),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            experimental_config=experimental_config,
        )

    def run_steps(
        self,
        step_fn: Callable[[], object],
        steps: int | None = None,
        synchronize: bool = True,
    ) -> None:
        """Run `step_fn` under profiler and call `prof.step()` each iteration."""

        torch = self._import_torch()
        total_steps = steps if steps is not None else self.default_total_steps()
        with self.profile() as prof:
            for _ in range(total_steps):
                step_fn()
                if synchronize:
                    torch.npu.synchronize()
                prof.step()

    def default_total_steps(self, redundant_steps: int = 0) -> int:
        """Return the number of loop iterations needed by the current schedule."""

        schedule_steps = self.config.wait + self.config.warmup + self.config.active
        return self.config.skip_first + self.config.repeat * schedule_steps + redundant_steps

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchNpuProfiler requires torch to run profiling.") from exc
        return torch

    @staticmethod
    def _import_torch_npu():
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("TorchNpuProfiler requires torch_npu to run profiling.") from exc
        return torch_npu

    @staticmethod
    def _enum_value(enum_obj: object, prefix: str, value: int | str) -> object:
        enum_name = str(value) if isinstance(value, str) else f"{prefix}{value}"
        return getattr(enum_obj, enum_name)

    @staticmethod
    def _activity(torch_npu: object, name: str) -> object:
        return getattr(torch_npu.profiler.ProfilerActivity, name.upper())


class ProfileResultParser:
    """Parse torch_npu profiler outputs generated by tensorboard trace handler."""

    KERNEL_FILE = "kernel_details.csv"
    STEP_FILE = "step_trace_time.csv"
    KERNEL_NAME_COLUMNS = ("Name", "Op Name")
    KERNEL_DURATION_COLUMNS = ("Duration(us)", "Task Duration(us)")
    SHAPE_COLUMNS = ("Input Shapes", "Input Shape", "Input shapes", "Shapes", "Shape")

    def __init__(self, profile_root: str | Path) -> None:
        self.profile_root = Path(profile_root)

    def kernel_rows(self) -> list[dict[str, str]]:
        """Return rows from every `kernel_details.csv` below `profile_root`."""

        return self._read_named_csv(self.KERNEL_FILE)

    def step_rows(self) -> list[dict[str, str]]:
        """Return rows from every `step_trace_time.csv` below `profile_root`."""

        return self._read_named_csv(self.STEP_FILE)

    def kernel_time_by_name(self, name_prefix: str | None = None) -> list[DurationSummary]:
        """Summarize kernel duration by kernel name."""

        groups: dict[str, list[float]] = defaultdict(list)
        for row in self.kernel_rows():
            name = self._first_value(row, self.KERNEL_NAME_COLUMNS)
            if not name or (name_prefix and not name.startswith(name_prefix)):
                continue
            groups[name].append(_to_float(self._first_value(row, self.KERNEL_DURATION_COLUMNS)))
        return self._sorted_summaries(groups)

    def kernel_time_by_shape(
        self,
        shape_key: ShapeKey = None,
        name_prefix: str | None = None,
    ) -> list[DurationSummary]:
        """Summarize kernel duration by `(kernel name, shape)`.

        `shape_key` can be:

        - a CSV column name, for example `"Input Shapes"`;
        - a callback that receives a row and returns a shape label;
        - `None`, which auto-detects common shape columns and then falls back
          to the first directory component under `profile_root`.
        """

        groups: dict[str, list[float]] = defaultdict(list)
        for row in self.kernel_rows():
            name = self._first_value(row, self.KERNEL_NAME_COLUMNS)
            if not name or (name_prefix and not name.startswith(name_prefix)):
                continue
            shape = self._resolve_shape(row, shape_key)
            groups[f"{name} | shape={shape}"].append(
                _to_float(self._first_value(row, self.KERNEL_DURATION_COLUMNS))
            )
        return self._sorted_summaries(groups)

    def step_time_summary(
        self,
        time_column: str = "Stage",
        device_id: str | None = None,
    ) -> list[StepDurationSummary]:
        """Summarize step duration by device."""

        groups: dict[str, list[float]] = defaultdict(list)
        for row in self.step_rows():
            row_device_id = row.get("Device_id", "")
            if device_id is not None and row_device_id != str(device_id):
                continue
            groups[row_device_id].append(_to_float(row.get(time_column)))

        summaries = []
        for current_device_id, values in groups.items():
            base = _summary(current_device_id, values)
            summaries.append(StepDurationSummary(**base.__dict__, device_id=current_device_id, time_column=time_column))
        summaries.sort(key=lambda item: item.mean_us, reverse=True)
        return summaries

    def average_step_time_us(self, time_column: str = "Stage", device_id: str | None = None) -> float:
        """Return the average step time in microseconds."""

        summaries = self.step_time_summary(time_column=time_column, device_id=device_id)
        if not summaries:
            return 0.0
        values = [summary.mean_us for summary in summaries]
        return statistics.mean(values)

    def _read_named_csv(self, file_name: str) -> list[dict[str, str]]:
        rows = []
        for csv_path in self._find_files(file_name):
            with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row["__csv_path__"] = str(csv_path)
                    row["__profile_label__"] = self._profile_label(csv_path)
                    rows.append(row)
        return rows

    def _find_files(self, file_name: str) -> Iterable[Path]:
        if self.profile_root.is_file() and self.profile_root.name == file_name:
            yield self.profile_root
            return
        yield from sorted(self.profile_root.rglob(file_name))

    def _profile_label(self, csv_path: Path) -> str:
        try:
            relative_parts = csv_path.relative_to(self.profile_root).parts
        except ValueError:
            return "unknown"
        if len(relative_parts) <= 1:
            return self.profile_root.name
        return relative_parts[0]

    @staticmethod
    def _first_value(row: Mapping[str, str], columns: Sequence[str]) -> str:
        for column in columns:
            value = row.get(column)
            if value:
                return value
        return ""

    def _resolve_shape(self, row: Mapping[str, str], shape_key: ShapeKey) -> str:
        if callable(shape_key):
            return shape_key(row) or "unknown"
        if isinstance(shape_key, str):
            return row.get(shape_key) or "unknown"
        shape = self._first_value(row, self.SHAPE_COLUMNS)
        if shape:
            return shape
        return row.get("__profile_label__", "unknown")

    @staticmethod
    def _sorted_summaries(groups: Mapping[str, Sequence[Number]]) -> list[DurationSummary]:
        summaries = [_summary(key, values) for key, values in groups.items()]
        summaries.sort(key=lambda item: item.mean_us, reverse=True)
        return summaries
