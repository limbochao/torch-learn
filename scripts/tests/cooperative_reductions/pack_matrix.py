"""Build a standalone compile/DSL matrix archive with source hashes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    paths = {name: here / name for name in ("run_matrix.sh", "run_matrix.py", "plot_compile_matrix.py",
             "reduction_matrix.py", "barrier_suite.py", "npu_triton_helpers.py", "cuda_triton_helpers.py",
             "LICENSE-pytorch.txt")}
    paths["autotune_tiling.py"] = here.parents[1] / "tools" / "autotune_tiling.py"
    files = {name: path.read_bytes() for name, path in paths.items()}
    files["README.md"] = (here / "COMPILE_MATRIX_README.md").read_bytes()
    files["requirements.txt"] = b"matplotlib>=3.8\n"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    manifest = {"schema": "cooperative-compile-matrix-v1", "source_commit": commit,
                "files": {name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
                          for name, body in files.items()}}
    files["bundle_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name, body in sorted(files.items()):
            info = tarfile.TarInfo("cooperative-compile-matrix/" + name)
            info.size = len(body)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(body))
    temporary.replace(args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"{args.out}: {args.out.stat().st_size} bytes, {len(files)} files, sha256={digest}")


if __name__ == "__main__":
    main()
