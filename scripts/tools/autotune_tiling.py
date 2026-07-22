"""Capture the best CUDA/NPU Inductor autotune tiling configs."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from typing import Any


LOGGER = logging.getLogger(__name__)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def _config_payload(config: object) -> dict[str, object]:
    if isinstance(config, Mapping):
        return {str(key): _json_value(value) for key, value in config.items()}
    payload = dict(getattr(config, "kwargs", {}) or {})
    for field in (
        "num_warps",
        "num_stages",
        "num_ctas",
        "num_consumer_groups",
        "num_buffers_warp_spec",
    ):
        value = getattr(config, field, None)
        if value is not None:
            payload[field] = value
    return {str(key): _json_value(value) for key, value in payload.items()}


class BestTilingRecorder:
    """Hook Inductor autotuners and capture their selected launch configs."""

    def __init__(self, device: str) -> None:
        if device not in ("cuda", "npu"):
            raise ValueError("device must be 'cuda' or 'npu'")
        self.device = device
        self._capturing = False
        self._records: list[dict[str, object]] = []
        self._seen: set[str] = set()
        self._patches: list[tuple[type, Any]] = []
        self._warnings: set[str] = set()
        self._lock = threading.Lock()

    def install(self) -> None:
        """Install runtime hooks after the target device backend is imported."""

        if self._patches:
            return
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        self._patch_class(CachingAutotuner, grouped=False)
        if self.device == "npu":
            from torch_npu._inductor.runtime.triton_heuristics import (
                NPUCachingAutotuner,
                NPUSymbolicGroupedAutotuner,
            )

            self._patch_class(NPUCachingAutotuner, grouped=False)
            self._patch_class(NPUSymbolicGroupedAutotuner, grouped=True)

    def uninstall(self) -> None:
        """Restore all original autotuner methods."""

        while self._patches:
            cls, original_run = self._patches.pop()
            cls.run = original_run

    def start_capture(self) -> None:
        """Clear old records and begin capturing autotuner launches."""

        with self._lock:
            self._records = []
            self._seen = set()
            self._capturing = True

    def stop_capture(self) -> list[dict[str, object]]:
        """Stop capturing and return a detached copy of unique records."""

        with self._lock:
            self._capturing = False
            return [dict(record) for record in self._records]

    def _patch_class(self, cls: type, grouped: bool) -> None:
        original_run = cls.run
        recorder = self

        def wrapped_run(autotuner, *args, **kwargs):
            result = original_run(autotuner, *args, **kwargs)
            if grouped:
                recorder._record_grouped(autotuner, args)
            else:
                recorder._record_standard(autotuner)
            return result

        cls.run = wrapped_run
        self._patches.append((cls, original_run))

    def _record_standard(self, autotuner: object) -> None:
        launcher = getattr(autotuner, "best_launcher", None)
        if launcher is None:
            launchers = getattr(autotuner, "launchers", ())
            launcher = launchers[0] if len(launchers) == 1 else None
        if launcher is None:
            return

        config = getattr(autotuner, "best_candidate_config", None)
        if config is None:
            config = getattr(launcher, "config", None)
        self._append_record(
            autotuner,
            selected_config=_config_payload(config),
            runtime_blocks=self._runtime_blocks_payload(
                autotuner,
                getattr(autotuner, "best_runtime_blocks", ()),
            ),
        )

    def _record_grouped(self, autotuner: object, args: tuple[object, ...]) -> None:
        try:
            feature_inputs = autotuner._runtime_feature_inputs(args)
            group_id = autotuner._resolve_group_id(feature_inputs)
            candidate = autotuner.best_candidate_map[group_id]
            runtime_block_values = autotuner._materialize_runtime_blocks(candidate, args)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            warning = f"failed to capture grouped best tiling: {type(exc).__name__}: {exc}"
            if warning not in self._warnings:
                LOGGER.warning(warning)
                self._warnings.add(warning)
            return

        selected_config = candidate.get("full_config")
        if selected_config is None:
            variant_id = candidate["variant_id"]
            selected_config = autotuner.candidate_plan["variants"][variant_id]["config"]
        self._append_record(
            autotuner,
            selected_config=_config_payload(selected_config),
            runtime_blocks=self._runtime_blocks_payload(
                autotuner,
                runtime_block_values,
            ),
            group_id=group_id,
            feature_inputs=feature_inputs,
        )

    @staticmethod
    def _runtime_blocks_payload(
        autotuner: object,
        values: object,
    ) -> dict[str, object]:
        candidate_plan = getattr(autotuner, "candidate_plan", {}) or {}
        names = tuple(
            getattr(autotuner, "runtime_block_arg_names", ())
            or candidate_plan.get("runtime_block_append_order", ())
        )
        return {
            name: _json_value(value)
            for name, value in zip(names, tuple(values or ()))
        }

    def _append_record(
        self,
        autotuner: object,
        selected_config: dict[str, object],
        runtime_blocks: dict[str, object],
        group_id: int | None = None,
        feature_inputs: object = (),
    ) -> None:
        with self._lock:
            if not self._capturing:
                return
            inductor_meta = getattr(autotuner, "inductor_meta", {}) or {}
            fn = getattr(autotuner, "fn", None)
            record = {
                "device": self.device,
                "kernel_name": inductor_meta.get(
                    "kernel_name", getattr(fn, "__name__", "")
                ),
                "group_id": group_id,
                "feature_inputs": _json_value(feature_inputs),
                "selected_config": selected_config,
                "runtime_blocks": runtime_blocks,
            }
            serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if serialized in self._seen:
                return
            self._records.append(record)
            self._seen.add(serialized)
