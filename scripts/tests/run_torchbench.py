#!/usr/bin/env python3
"""Prepare missing model data and run TorchBench/HuggingFace with Inductor + Triton accuracy checks by default."""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from torchbench_setup import default_install_dir, download, extract, file_sha256  # noqa: E402


HF_MODELS = """
AlbertForMaskedLM BartForCausalLM BertForMaskedLM BlenderbotForCausalLM DebertaV2ForMaskedLM
DistilBertForMaskedLM ElectraForCausalLM GPT2ForSequenceClassification LayoutLMForMaskedLM MBartForCausalLM
MT5ForConditionalGeneration MegatronBertForCausalLM MobileBertForMaskedLM OPTForCausalLM PLBartForCausalLM
PegasusForCausalLM RobertaForCausalLM T5ForConditionalGeneration T5Small TrOCRForCausalLM XGLMForCausalLM
""".split()
TORCHBENCH_MODELS = """
BERT_pytorch LearningToPaint Super_SloMo alexnet basic_gnn_edgecnn basic_gnn_gcn basic_gnn_sage dcgan
densenet121 dlrm drq lennard_jones llama mobilenet_v2 nanogpt nvidia_deeprecommender phlippe_densenet
phlippe_resnet pytorch_stargan pytorch_unet resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0
soft_actor_critic speech_transformer squeezenet1_1 torch_multimodal_clip vgg16
""".split()
MODEL_SUITES = {name: "huggingface.py" for name in HF_MODELS}
MODEL_SUITES.update({name: "torchbench.py" for name in TORCHBENCH_MODELS})
WEIGHTS = {
    "alexnet": "alexnet-owt-7be5be79.pth",
    "densenet121": "densenet121-a639ec97.pth",
    "mobilenet_v2": "mobilenet_v2-b0353104.pth",
    "resnet18": "resnet18-f37072fd.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "resnext50_32x4d": "resnext50_32x4d-7cdf4587.pth",
    "shufflenet_v2_x1_0": "shufflenetv2_x1-5666bf0f80.pth",
    "squeezenet1_1": "squeezenet1_1-b8a52dc0.pth",
    "vgg16": "vgg16-397923af.pth",
    "Super_SloMo": "vgg16-397923af.pth",
}
DATA_ARCHIVES = {
    "Super_SloMo": "df8c078664e2121de32287d67992c83c78cd2b91630a47a0768be43edae97581",
    "pytorch_stargan": "9bbe4bb583d1c6e570c4e23e052abe11f98527149ecdeaf84a0942624e9d66e8",
    "speech_transformer": "a86e85e2fbcf05011111d9096a344ec2455dd9d1e246458093abcbc0c87ac376",
}


def select_models(names=None, model_file=None):
    selected = list(names or [])
    if model_file:
        for line in Path(model_file).read_text().splitlines():
            selected.extend(line.split("#", 1)[0].replace(",", " ").split())
    if names is None and model_file is None:
        return list(MODEL_SUITES)
    selected = [part for name in selected for part in name.split(",") if part]
    if not selected:
        raise ValueError("The selected model list is empty")
    unknown = sorted(set(selected) - MODEL_SUITES.keys())
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def environment(work_dir):
    sources = default_install_dir() / "sources"
    return {
        "TORCHBENCH_DATA_PATH": str(work_dir / "data"),
        "TORCH_HOME": str(work_dir / "cache/torch"),
        "HF_HOME": str(work_dir / "cache/huggingface"),
        "FVCORE_CACHE": str(work_dir / "cache/iopath"),
        "TORCHINDUCTOR_CACHE_DIR": str(work_dir / "cache/inductor"),
        "TRITON_CACHE_DIR": str(work_dir / "cache/triton"),
        "PYTHONPATH": os.pathsep.join(map(str, [sources, sources / "benchmark", sources / "multimodal"])),
    }


def archive_inputs_ready(work_dir, model):
    data_dir = work_dir / "data"
    archive = work_dir / "downloads" / f"{model}_inputs.tar.gz"
    if archive.is_file() and file_sha256(archive) == DATA_ARCHIVES[model]:
        with tarfile.open(archive) as stream:
            files = [member for member in stream.getmembers() if member.isfile()]
        return bool(files) and all(
            (data_dir / member.name).is_file() and (data_dir / member.name).stat().st_size == member.size
            for member in files
        )
    # Also accept inputs prepared separately, without keeping their original archive.
    root = data_dir / f"{model}_inputs"
    if model == "Super_SloMo":
        return any((root / "dataset/train").glob("*/*.jpg"))
    if model == "pytorch_stargan":
        return (root / "data/celeba/list_attr_celeba.txt").is_file() and any(
            (root / "data/celeba/images").glob("*.jpg")
        )
    files = [f"input_data/{split}/{name}" for split in ("train", "dev", "test")
             for name in ("data.json", "feats.1.ark")]
    files.append("input_data/lang_1char/train_chars.txt")
    return all((root / name).is_file() for name in files)


def prepare_data(work_dir, model):
    assets = []
    if model in DATA_ARCHIVES and not archive_inputs_ready(work_dir, model):
        filename = f"{model}_inputs.tar.gz"
        assets.append((
            f"https://ossci-datasets.s3.amazonaws.com/torchbench/data/{filename}",
            work_dir / "downloads" / filename, DATA_ARCHIVES[model], True,
        ))
    if model == "drq":
        assets.append((
            "https://ossci-datasets.s3.amazonaws.com/torchbench/models/drq/obs.pkl",
            work_dir / "data/obs.pkl", "3c70db82eba7eba0fbc5a52ed44bdce6c6fc77b69f29158489d98f145b023627", False,
        ))
    if model == "torch_multimodal_clip":
        assets.extend([
            (
                "https://ossci-assets.s3.amazonaws.com/2880px-Pizza-3007395.jpg",
                work_dir / "data/pizza.jpg",
                "580d3bb6381b0a249f9d90fb5e8b206c4a481308db78c493a74fd6c93984e0a0", False,
            ),
            (
                "https://download.pytorch.org/models/text/clip_merges.bpe",
                work_dir / "cache/iopath/models/text/clip_merges.bpe",
                "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85", False,
            ),
        ])
    if model in WEIGHTS:
        filename = WEIGHTS[model]
        assets.append((
            f"https://download.pytorch.org/models/{filename}",
            work_dir / "cache/torch/hub/checkpoints" / filename,
            filename.rsplit("-", 1)[1].split(".")[0], False,
        ))
    for url, path, checksum, unpack in assets:
        download(url, path, checksum)
        if unpack:
            extract(path, work_dir / "data")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_command(args, model, output):
    command = [
        sys.executable, "-P", str(args.runner_dir / MODEL_SUITES[model]),
        "--only", model, "--device", "npu",
        "--execution-mode", "both" if args.inductor else "eager",
        "--accuracy" if args.accuracy else "--performance",
        "--inference" if args.inference else "--training",
        "--float32", "--output", str(output),
    ]
    if args.inductor:
        command.extend(["--backend", "inductor", "--npu-backend", "triton"])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.accuracy:
        command.extend(["--iterations", str(args.iterations)])
    else:
        command.extend(["--repeat", str(args.repeat), "--iterations-per-run", "1"])
    return command


def check_result(path, model, args):
    if not path.is_file():
        return False, "missing result CSV"
    try:
        with path.open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("name") == model]
    except (OSError, csv.Error) as error:
        return False, f"unreadable result CSV: {error}"
    if not rows:
        return False, "model missing from result CSV"
    row = rows[-1]
    if args.accuracy:
        expected = "pass_accuracy" if args.inductor else "pass_eager"
        return row.get("accuracy") == expected, row.get("accuracy", "missing accuracy status")
    try:
        latency = float(row["abs_latency"])
        success = math.isfinite(latency) and latency > 0
        if args.inductor:
            speedup = float(row["speedup"])
            success = success and math.isfinite(speedup) and speedup > 0
        else:
            success = success and row.get("execution_mode") == "eager"
    except (KeyError, ValueError):
        success = False
    return success, f"latency={row.get('abs_latency', 'missing')} ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", type=Path, required=True, help="pytorch_new/benchmarks/torchbench")
    parser.add_argument(
        "--work-dir", type=Path, help="Data, caches and results; defaults to RUNNER_DIR/torchbench_workspace",
    )
    parser.add_argument("--models", nargs="+", help="Model names, separated by spaces or commas; default: all 50")
    parser.add_argument("--models-file", type=Path, help="Additional model names, one per line")
    parser.add_argument("--inductor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accuracy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference", action="store_true", help="Use inference instead of the default training mode")
    parser.add_argument("--batch-size", type=positive_int, help="Default: use each model's runner configuration")
    parser.add_argument("--iterations", type=positive_int, default=2, help="Iterations for accuracy checks")
    parser.add_argument("--repeat", type=positive_int, default=3, help="Timing repetitions when accuracy is disabled")
    parser.add_argument("--output-dir", type=Path, help="New output directory for logs and CSV files")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without downloading data, importing Torch or running models",
    )
    args = parser.parse_args()
    try:
        models = select_models(args.models, args.models_file)
        args.runner_dir = args.runner_dir.expanduser().resolve()
        for suite in {MODEL_SUITES[model] for model in models}:
            if not (args.runner_dir / suite).is_file():
                raise ValueError(f"Runner not found: {args.runner_dir / suite}")
        work_dir = (args.work_dir or args.runner_dir / "torchbench_workspace").expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = (args.output_dir or work_dir / "results" / stamp).expanduser().resolve()
        if not args.dry_run:
            for source in ("benchmark/torchbenchmark", "multimodal/torchmultimodal"):
                if not (default_install_dir() / "sources" / source).is_dir():
                    raise ValueError(f"Missing source {source}; run torchbench_setup.py first")
            output_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        prepared = environment(work_dir)
        prepared["PYTHONPATH"] += os.pathsep + str(args.runner_dir)
        prepared["PYTHONPATH"] += os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        env.update(prepared)
        env.setdefault("OMP_NUM_THREADS", "8")
        env.setdefault("MKL_NUM_THREADS", "8")
        env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
        if args.inductor:
            # In torch_npu, "default" selects Triton. Clear competing backend switches.
            env["TORCHINDUCTOR_NPU_BACKEND"] = "default"
            env.pop("TORCHINDUCTOR_USE_AKG", None)
            env.pop("TORCHINDUCTOR_ENABLE_MFUSION", None)
        failed = []
        for model in models:
            result = output_dir / f"{model}.csv"
            command = build_command(args, model, result)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            log = output_dir / f"{model}.log"
            with log.open("w") as stream:
                stream.write(shlex.join(command) + "\n")
                stream.flush()
                try:
                    prepare_data(work_dir, model)
                except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
                    detail = f"{model}: FAIL (data preparation: {error})"
                    stream.write(detail + "\n")
                    print(detail, file=sys.stderr)
                    failed.append(model)
                    continue
                process = subprocess.run(
                    command, cwd=args.runner_dir, env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
            success, detail = check_result(result, model, args)
            success = success and process.returncode == 0
            print(f"{model}: {'PASS' if success else 'FAIL'} ({detail}, exit={process.returncode}); log: {log}")
            if not success:
                failed.append(model)
        if failed:
            print(f"Failed models: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
