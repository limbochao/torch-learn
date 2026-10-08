#!/usr/bin/env python3
"""Install TorchBench companion packages into the active Python environment."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


PROTECTED_PACKAGES = {"torch", "torch-npu", "triton", "triton-ascend"}

# Source revisions required by the Ascend TorchBench 2.13 runner.
SOURCES = {
    "benchmark": (
        "pytorch/benchmark", "fb14629994956c16d27c7a9940c5f081afbf68dc",
        "6caca9b6b742a6a6795f01fa2abffaf79ca9a6b27ac3cfd3a1b88f825a85598d",
    ),
    "pyg": (
        "pyg-team/pytorch_geometric", "cabcd4097442ba60aa1efa11e1619dd9bb8fb527",
        "14a01bdbc25fce4a0c7fe0f794c7a4f861910b30c92156dfc69ee93b7eb167f2",
    ),
    "multimodal": (
        "facebookresearch/multimodal", "e54f602d1cb33f8a5deb3540955271c4223f7c9c",
        "bf1813e83c4f11be095e60c8c6edae0da5bf734a9157c862c3bae1f376de8bca",
    ),
}
REQUIREMENTS = """
numpy==1.26.4
pandas
scipy
PyYAML
psutil
tabulate
tqdm
requests
Pillow
torchvision==0.28.0
transformers==4.36.0
gym==0.26.2
opencv-python-headless==4.11.0.86
onnx
ml_dtypes==0.5.4
kaldi_io
tensorboardX
future
pydot
scikit-learn
scikit-image
kornia
sentencepiece
gymnasium
pygame
ftfy
iopath==0.1.9
regex
""".split()


def default_install_dir():
    return Path(sys.prefix) / "share" / "torchbench-213"


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_sha256(destination).startswith(sha256):
        print(f"Cached: {destination}", flush=True)
        return destination
    partial = destination.with_name(destination.name + ".part")
    print(f"Download: {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if not file_sha256(partial).startswith(sha256):
        raise RuntimeError(f"SHA-256 mismatch: {partial}")
    partial.replace(destination)
    return destination


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")


def prepare_source(name, work_dir):
    repository, revision, checksum = SOURCES[name]
    destination = work_dir / "sources" / name
    marker = destination / ".torchbench-source-commit"
    if marker.is_file() and marker.read_text().strip() == revision:
        return destination
    if destination.exists():
        raise RuntimeError(f"Source directory already exists without the expected revision marker: {destination}")
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    archive = download(url, work_dir / "downloads" / f"{name}-{revision}.tar.gz", checksum)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        extract(archive, Path(temporary))
        roots = list(Path(temporary).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(f"Unexpected source archive layout: {archive}")
        roots[0].rename(destination)
    marker.write_text(revision + "\n")
    return destination


def install_software(install_dir):
    installed = {normalize_name(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}
    missing = {"torch", "torch-npu"} - installed.keys()
    if missing:
        raise RuntimeError(f"Install the base stack separately first: {', '.join(sorted(missing))}")
    install_dir.mkdir(parents=True, exist_ok=True)
    protected = {name: installed[name] for name in PROTECTED_PACKAGES if name in installed}
    constraints = dict(protected)
    # Keep dependencies already supplied by the base stack, including Triton-Ascend's attrs pin.
    for name in ("attrs", "torchvision", "torchaudio"):
        if name in installed:
            constraints[name] = installed[name]
    constraint_file = install_dir / "constraints.txt"
    constraint_file.write_text("".join(f"{name}=={version}\n" for name, version in sorted(constraints.items())))
    for name in SOURCES:
        prepare_source(name, install_dir)
    requirements = install_dir / "requirements.txt"
    pyg_requirement = "torch_geometric @ " + (install_dir / "sources/pyg").as_uri()
    requirements.write_text("\n".join(REQUIREMENTS + [pyg_requirement]) + "\n")
    plan_path = install_dir / "install-plan.json"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(plan_path),
        "-c", str(constraint_file), "-r", str(requirements),
    ], check=True)
    plan = json.loads(plan_path.read_text())["install"]
    forbidden = {normalize_name(item["metadata"]["name"]) for item in plan} & PROTECTED_PACKAGES
    if forbidden:
        raise RuntimeError(f"Refusing to install or replace excluded packages: {', '.join(sorted(forbidden))}")
    # Install exactly the resolved artifacts without allowing a second dependency resolution.
    artifacts = [item["download_info"]["url"] for item in plan]
    if artifacts:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps", "--report",
            str(install_dir / "install-report.json"), *artifacts,
        ], check=True)
    for name, version in protected.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"Excluded package unexpectedly changed: {name}")
    (install_dir / "packages.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True,
    ))
    print("Companion packages and pinned sources prepared; the base Torch/NPU/Triton stack was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("install",), default="install")
    parser.parse_args()
    try:
        install_dir = default_install_dir()
        install_software(install_dir)
        print(f"Companion sources: {install_dir / 'sources'}")
    except (OSError, ValueError, RuntimeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
