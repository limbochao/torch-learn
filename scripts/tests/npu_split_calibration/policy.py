"""Deterministic experimental S selector. This file does not import torch."""

from __future__ import annotations

import hashlib
import json
import math


VERSION = "npu-fill-v2"
DEFAULTS = {"outputs_per_program": 1, "target_waves": 1, "min_chunk": 256, "max_s": 0}
KEY_FIELDS = ("op", "src_dtype", "dst_dtype", "layout", "X")
SUPPORTED = {"sum", "min", "max", "prod", "any", "xor_sum"}
FLOAT_DTYPES = ("float16", "bfloat16", "float32")
INTEGER_DTYPES = ("int32", "int64")


def supported_dtype(op, dtype):
    if op == "any":
        return dtype == "bool"
    if op == "xor_sum":
        return dtype in INTEGER_DTYPES
    return op in SUPPORTED and dtype in (*FLOAT_DTYPES, *INTEGER_DTYPES)


def split_count(x, r, cores, params):
    p0_estimate = math.ceil(x / params["outputs_per_program"])
    fill = math.ceil(cores * params["target_waves"] / p0_estimate)
    cap = params["max_s"] or cores * params["target_waves"]
    return max(1, min(fill, cap, r - 1, r // params["min_chunk"]))


def split_boundaries(x, cores, params):
    """Exact integer R where the FIXED selector changes S, never an S sweep."""
    cap = min(math.ceil(cores * params["target_waves"] / math.ceil(x / params["outputs_per_program"])),
              params["max_s"] or cores * params["target_waves"])
    return [max(s + 1, s * params["min_chunk"]) for s in range(2, cap + 1)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family(meta):
    return {key: meta[key] for key in KEY_FIELDS}


def family_id(meta):
    return json.dumps(family(meta), sort_keys=True, separators=(",", ":"))


def candidate(meta, cores, params):
    """A starting hypothesis, not a calibrated production policy or an S search.

    outputs_per_program is explicitly an estimate, not measured grid occupancy.
    A partial may contain multiple local R tiles; min_chunk is NOT a UB limit.
    """
    if cores < 1 or any(params[k] < 1 for k in ("outputs_per_program", "target_waves", "min_chunk")):
        raise ValueError("cores and selector parameters must be positive")
    if params["max_s"] < 0:
        raise ValueError("max_s must be >= 0")
    if meta["op"] not in SUPPORTED or not meta.get("hinted", False):
        return 1, "unsupported_op_or_unhinted"
    if meta["layout"] == "unknown" or meta["X"] < 1 or meta["R"] < 2:
        return 1, "unknown_reads_or_empty"
    if not supported_dtype(meta["op"], meta["src_dtype"]):
        return 1, "unsupported_dtype"
    split = split_count(meta["X"], meta["R"], cores, params)
    return max(1, split), "fill_estimate" if split > 1 else "insufficient_work_or_parallelism"


def admit(meta, split, table):
    # Hints do not define a guarded runtime domain. Dynamic tables are deliberately
    # disabled until a real guard/dispatch mechanism exists in the backend.
    if meta.get("dynamic"):
        return False
    return any(
        rule["key"] == family(meta)
        and rule["r_min"] <= meta["R"] <= rule["r_max"]
        for rule in table.get("rules", [])
    ) and split > 1


def validate_table(table, fingerprint, params):
    if table.get("schema") != "npu-split-table-v2":
        raise ValueError("unsupported admission table schema")
    if table.get("status") not in ("DRAFT_REQUIRES_HOLDOUT", "VALIDATED_MEASURED_DOMAIN"):
        raise ValueError("table is incomplete or still requires refinement")
    if table.get("environment") != fingerprint:
        raise ValueError("table hardware/software fingerprint differs from this machine")
    if table.get("selector") != {"version": VERSION, "params": params}:
        raise ValueError("table was calibrated with a different S selector")
    for rule in table.get("rules", []):
        if set(rule["key"]) != set(KEY_FIELDS) or not 1 < rule["r_min"] <= rule["r_max"]:
            raise ValueError("invalid admission rule")
