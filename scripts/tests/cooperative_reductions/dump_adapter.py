"""Recompile an archived adapter with its metadata and save final compiler outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_DUMP_DIR"] = str(args.out.resolve())

    from triton.backends.ascend.compiler import linalg_to_bin_enable_npu_compile_910_95
    from triton.backends.compiler import GPUTarget

    adapter = args.adapter.read_text()
    metadata = json.loads(args.metadata.read_text())
    metadata["target"] = GPUTarget("npu", metadata["target_arch"], 0)
    metadata["hash"] = hashlib.sha256((adapter + str(args.out)).encode()).hexdigest()
    options = SimpleNamespace(debug=True, is_pure_simt=metadata["is_pure_simt"])
    # Compile only. debug saves intermediate files and the generated object;
    # it does not add device instrumentation to the existing adapter.
    binary = linalg_to_bin_enable_npu_compile_910_95(adapter, metadata, options)
    (args.out / "kernel.o").write_bytes(binary)
    manifest = {"adapter": str(args.adapter), "metadata": str(args.metadata),
                "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "final_metadata": metadata, "launched": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
