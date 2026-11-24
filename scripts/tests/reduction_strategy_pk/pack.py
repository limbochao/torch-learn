"""Build a standalone reduction-strategy test bundle with exact source hashes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent


def source_files():
    paths = {path.name: path for path in HERE.glob("*.py")}
    recorder = HERE / "autotune_tiling.py"
    if not recorder.exists():
        recorder = HERE.parents[1] / "tools" / "autotune_tiling.py"
    if not recorder.exists():
        raise FileNotFoundError("Missing scripts/tools/autotune_tiling.py")
    paths["autotune_tiling.py"] = recorder
    for name in ("one_click.sh", "requirements-plot.txt"):
        paths[name] = HERE / name
    return paths


def build(output):
    files = {name: path.read_bytes() for name, path in source_files().items()}
    readme = HERE / "ONE_CLICK_README.md"
    files["README.md"] = (readme if readme.exists() else HERE / "README.md").read_bytes()
    license_path = HERE / "LICENSE-pytorch.txt"
    if not license_path.exists():
        license_path = HERE.parent / "cooperative_reductions" / "LICENSE-pytorch.txt"
    if license_path.exists():
        files["LICENSE-pytorch.txt"] = license_path.read_bytes()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(schema="reduction-strategy-pk-bundle-v1", source_commit=commit,
                    source_identity="File hashes describe the actual working files, including uncommitted edits",
                    files={name: dict(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
                           for name, body in sorted(files.items())})
    files["bundle_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, body in sorted(files.items()):
                info = tarfile.TarInfo("reduction-strategy-pk/" + name)
                info.size, info.mtime = len(body), 0
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(info, io.BytesIO(body))
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(dict(archive=str(output), files=len(files), bytes=output.stat().st_size, sha256=digest), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
