"""Compare barriers when output-axis tasks underfill or fill the physical cores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import barrier_suite as suite


def build_plain_kernel():
    @triton.jit
    def plain(A, B, Out, R: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0, tl.float32)
        for begin in range(0, R, BLOCK):
            index = begin + offsets
            a = tl.load(A + row * R + index, index < R, other=0).to(tl.float32)
            b = tl.load(B + row * R + index, index < R, other=0).to(tl.float32)
            acc += a + b
        tl.store(Out + row, tl.sum(acc, 0))

    return plain


def compare(actual, expected, exact=False):
    actual, expected = actual.cpu(), expected.cpu()
    matching = actual == expected if exact else torch.isclose(actual.double(), expected, rtol=2e-4, atol=2e-2)
    bad = (~matching).nonzero()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual.double() - expected.double()).abs()

    def value(number):
        item = number.item()
        return item if math.isfinite(item) else str(item)

    return {"checked_values": actual.numel(), "mismatched_values": len(bad),
            "nonfinite_values": int((~torch.isfinite(actual)).sum()),
            "max_finite_abs_error": difference[finite].max().item() if bool(finite.any()) else None,
            "examples": [{"index": index.tolist(), "actual": value(actual[tuple(index.tolist())]),
                          "expected": value(expected[tuple(index.tolist())])} for index in bad[:8]],
            "rtol": 0 if exact else 2e-4, "atol": 0 if exact else 2e-2}


def mapping_metadata(compiled):
    metadata = suite.metadata_dict(compiled)
    fields = ("auto_blockify_enabled", "auto_blockify_size", "program_grid_mapping_applied",
              "program_grid_transforms", "launch_cooperative_grid", "compile_mode", "mix_mode")
    return {key: metadata.get(key) for key in fields}


def run(args, api, info):
    exchange, reduce = suite.build_kernels()
    plain = build_plain_kernel()
    records = []
    for rows in args.rows:
        for count in args.rnumels:
            _, output, expected, initial = suite.make_reduction(args, reduce, rows, count, 1, "original")
            left, right = initial[:2]
            for splits in args.splits:
                workspace = torch.full((rows, splits), float("nan"), device=args.backend)
                sem = torch.zeros(rows, dtype=torch.int32, device=args.backend)
                prepared = (output, expected, (left, right, workspace, sem))
                kinds = (["plain"] if splits == 1 else []) + args.kinds
                shift = args.splits.index(splits) % len(kinds)
                for kind in kinds[shift:] + kinds[:shift]:
                    record = {"backend": args.backend, "device_name": info.get("device_name", "NPU"),
                              "X": rows, "R": count, "S": splits, "kind": kind,
                              "output_axis_full": rows == info["core_count"],
                              "program_count": rows * splits, "core_count": info["core_count"],
                              "diagnostic_over_core": rows * splits > info["core_count"],
                              "block": args.block, "dtype": "float32"}
                    if kind == "original" and record["diagnostic_over_core"]:
                        record.update(status="SKIP_CORE_BUDGET",
                                      reason="Upstream cooperative candidates bound the grid by the SM budget")
                    else:
                        workspace.fill_(float("nan"))
                        output.fill_(float("nan"))
                        sem.zero_()
                        api.synchronize()
                        if kind == "plain":
                            def invoke():
                                return plain[(rows,)](left, right, output, R=count, BLOCK=args.block,
                                                       **args.options)
                        else:
                            invoke, _, _, _ = suite.make_reduction(args, reduce, rows, count, splits, kind,
                                                                  prepared=prepared)
                        compiled = invoke()
                        api.synchronize()
                        record["accuracy"] = compare(output, expected)
                        record["metadata"] = mapping_metadata(compiled)
                        if kind != "plain":
                            call, exchange_out, exchange_sem, _ = suite.make_exchange(
                                args, exchange, splits, rows, 1, kind, phases=args.phases, epoch=11, skew=True)
                            exchange_compiled = call()
                            api.synchronize()
                            reference = suite.exchange_expected(splits, rows, args.phases, 11, True)
                            record["exchange"] = compare(exchange_out, reference, exact=True)
                            record["exchange"]["phases"] = args.phases
                            if kind != "original":
                                assert bool((exchange_sem.cpu() == 123).all())
                            if count == args.rnumels[0] and (rows, splits, kind) in (
                                (7, 8, "native"), (7, 16, "native"), (56, 2, "native"),
                                (56, 16, "native"), (56, 16, "debug"), (56, 1, "original")):
                                stem = f"x{rows}-s{splits}-{kind}"
                                suite.save_kernel(args, compiled, "sum-" + stem)
                                suite.save_kernel(args, exchange_compiled, "exchange-" + stem)
                        failed = record["accuracy"]["mismatched_values"] or record.get("exchange", {}).get(
                            "mismatched_values", 0)
                        record["status"] = "FAIL_WRONG_RESULT" if failed else "PASS"
                        if not failed:
                            record.update(suite.device_time(args, api, invoke))
                            record["post_accuracy"] = compare(output, expected)
                            assert record["post_accuracy"]["mismatched_values"] == 0
                        record["buffer_addresses"] = [t.data_ptr() for t in prepared[2]] + [output.data_ptr()]
                    records.append(record)
                    suite.write_json(args.out / "output-axis.json", records)
                    print(json.dumps({key: record[key] for key in
                                      ("X", "R", "S", "kind", "status")}
                                     | {"median_us": record.get("median_us")}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("npu", "cuda"), default="cuda")
    parser.add_argument("--physical-device", type=int, help="required for NPU; optional CUDA physical-ID annotation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="auto", help="comma-separated X; auto uses 7,56,SMs on CUDA, 7,cores on NPU")
    parser.add_argument("--cuda-barrier", choices=("snapshot", "installed"), default="snapshot",
                        help="CUDA helper: fixed v2.13.0 snapshot or installed Inductor helper")
    parser.add_argument("--rnumels", default="32768,1048593")
    parser.add_argument("--splits", default="1,2,4,8,16,32,48,56")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--op-timeout-ms", type=int, default=10000)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "npu" and args.physical_device is None:
        parser.error("--physical-device is required for NPU mapping verification")
    for field in ("rnumels", "splits") + (() if args.rows == "auto" else ("rows",)):
        try:
            setattr(args, field, list(dict.fromkeys(map(int, getattr(args, field).split(",")))))
        except ValueError:
            parser.error(f"--{field} requires comma-separated positive integers")
        if not getattr(args, field) or min(getattr(args, field)) < 1:
            parser.error(f"--{field} requires positive integers")
    for field in ("block", "phases", "batch", "replays", "samples"):
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be positive")
    args.kinds = ["original", "native", "debug"] if args.backend == "npu" else ["original"]
    args.mapping, args.phase = "auto", "output-axis"
    torch_module, triton_module, api, args.options = suite.load_stack(args)
    globals().update(torch=torch_module, triton=triton_module, tl=suite.tl)
    info = suite.manifest(args, api)
    if args.rows == "auto":
        args.rows = sorted({7, info["core_count"]} | ({56} if args.backend == "cuda" else set()))
    info["rows"] = args.rows
    info["rnumels"], info["splits"] = args.rnumels, args.splits
    info["block"], info["seed"], info["phases"] = args.block, args.seed, args.phases
    info["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info["driver"] = Path(__file__).name
    info["scope"] = "Output-axis occupancy; over-core native/debug points are mapping diagnostics"
    bundle = Path(__file__).with_name("bundle_manifest.json")
    if bundle.exists():
        info["bundle_source_commit"] = json.loads(bundle.read_text())["source_commit"]
        info["bundle_manifest_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    suite.write_json(args.out / "manifest.json", info)
    print(json.dumps({key: info.get(key) for key in
                      ("backend", "device_name", "device_uuid", "core_count", "rows", "cuda_barrier")}), flush=True)
    records = run(args, api, info)
    print(json.dumps({"counts": {status: sum(r["status"] == status for r in records)
                                 for status in sorted({r["status"] for r in records})}}), flush=True)
    if args.backend == "cuda" and any(r["status"] != "PASS" and not r["status"].startswith("SKIP_") for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
