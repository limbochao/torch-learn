#!/usr/bin/env bash
# 按模型 CSV 串行运行 TorchBench 测试；只依赖 Bash、Python 标准库及目标测试环境。
set -euo pipefail

usage() {
    cat <<'EOF'
用法：bash run_torchbench_matrix.sh --models CSV --output-dir DIR [选项]

  --models CSV        模型清单，表头为 model_name,suit
  --shape VALUE       all（默认）、static 或 dynamic
  --mode VALUE        all（默认）、inference 或 training
  --metric VALUE      all（默认）、performance 或 accuracy
  --output-dir DIR    产物根目录，每次运行创建独立子目录
  --bench-dir DIR     torchbench runner 目录，也可设置 TORCHBENCH_DIR
  --data-path DIR     dataset 目录，也可设置 TORCHBENCH_DATA_PATH
  --python PATH       Python 可执行文件，默认使用当前环境的 python
  -h, --help          查看帮助

默认路径使用 NPU_A5 的 pta_213_cicd 专用环境。运行前激活目标环境并加载 CANN。
HF_ENDPOINT 默认 https://hf-mirror.com，也可通过环境变量覆盖。
退出码：0=所有已选测试通过；1=存在失败；2=参数或环境错误。
EOF
}

die() { echo "错误：$*" >&2; exit 2; }

models=''
output_dir=''
shape=all
mode=all
metric=all
bench_dir="${TORCHBENCH_DIR:-$PWD/pytorch_new/benchmarks/torchbench}"
data_path="${TORCHBENCH_DATA_PATH:-$PWD/pta_213_cicd/data}"
python_bin=python

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --models|--output-dir|--shape|--mode|--metric|--bench-dir|--data-path|--python)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数值"
            case "$1" in
                --models) models=$2 ;;
                --output-dir) output_dir=$2 ;;
                --shape) shape=$2 ;;
                --mode) mode=$2 ;;
                --metric) metric=$2 ;;
                --bench-dir) bench_dir=$2 ;;
                --data-path) data_path=$2 ;;
                --python) python_bin=$2 ;;
            esac
            shift 2 ;;
        *) die "未知参数：$1" ;;
    esac
done

[[ -n "$models" && -n "$output_dir" ]] || die '必须指定 --models 和 --output-dir'
case "$shape" in all|static|dynamic) ;; *) die '--shape 仅支持 all/static/dynamic' ;; esac
case "$mode" in all|inference|training) ;; *) die '--mode 仅支持 all/inference/training' ;; esac
case "$metric" in all|performance|accuracy) ;; *) die '--metric 仅支持 all/performance/accuracy' ;; esac
[[ -f "$models" ]] || die "模型清单不存在：$models"
[[ -d "$bench_dir" ]] || die "测试目录不存在：$bench_dir"
python_bin=$(command -v "$python_bin") || die '找不到 Python 可执行文件'
# 在切换测试目录之前将外部路径全部转换为绝对路径。
python_bin=$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$python_bin")
bench_dir=$(cd -- "$bench_dir" && pwd -P)
data_path=$("$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$data_path")
export TORCHBENCH_DATA_PATH="$data_path"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# CSV 读写和日志解析使用标准库，避免 shell 按逗号切分时破坏带引号的模型名。
csv_helper() {
    "$python_bin" - "$@" <<'PY'
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "model", "s_infer_perf", "s_infer_acc", "s_train_perf", "s_train_acc",
    "d_infer_perf", "d_infer_acc", "d_train_perf", "d_train_acc", "log_dir",
]


def write_results(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    action, root, *args = sys.argv[1:]
    root = Path(root)
    summary = root / "results_running.csv"
    if action == "prepare":
        models_path, bench_path = map(Path, args)
        with models_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            if [cell.strip() for cell in header] != ["model_name", "suit"]:
                raise ValueError("模型 CSV 表头必须为 model_name,suit")
            models = []
            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) != 2:
                    raise ValueError(f"CSV 第 {reader.line_num} 行必须包含两列")
                model, suite = (cell.strip() for cell in row)
                if not model or any(char in model for char in "\r\n\0"):
                    raise ValueError(f"CSV 第 {reader.line_num} 行模型名为空或包含换行/NUL")
                if suite not in {"torchbench", "timm_models", "huggingface"}:
                    raise ValueError(f"CSV 第 {reader.line_num} 行 suit 不支持：{suite}")
                if not (bench_path / f"{suite}.py").is_file():
                    raise ValueError(f"找不到 runner：{bench_path / (suite + '.py')}")
                models.append((model, suite))
        if not models:
            raise ValueError("模型 CSV 没有待测模型")
        rows = []
        with (root / "models.nul").open("wb") as manifest:
            for index, (model, suite) in enumerate(models):
                slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model)[:80]
                log_dir = root / f"{index + 1:04d}_{suite}_{slug}"
                log_dir.mkdir()
                for value in (str(index), model, suite, str(log_dir)):
                    manifest.write(value.encode("utf-8") + b"\0")
                rows.append(dict.fromkeys(COLUMNS, "-"))
                rows[-1].update(model=model, log_dir=str(log_dir))
        with (root / "models.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_name", "suit"])
            writer.writerows(models)
        write_results(summary, rows)
    elif action == "record":
        index, column, metric, exit_code, log_path = args
        result = "FAIL"
        if int(exit_code) == 0:
            # 清除终端颜色后匹配独立结果行，不把 traceback 或模型名当作结果。
            ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
            pattern = r"\s*([0-9]+(?:\.[0-9]+)?x)\s*" if metric == "performance" else r"\s*pass_accuracy\s*"
            with Path(log_path).open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = re.fullmatch(pattern, ansi.sub("", line))
                    if match:
                        result = match.group(1) if metric == "performance" else "PASS"
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[int(index)][column] = result
        write_results(summary, rows)
        print(result)
    elif action == "finish":
        completed = root / f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
        summary.replace(completed)
        print(completed)
    else:
        raise ValueError(f"未知操作：{action}")


try:
    main()
except (OSError, ValueError, csv.Error) as error:
    print(f"错误：{error}", file=sys.stderr)
    sys.exit(2)
PY
}

mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
run_dir=$(mktemp -d "$output_dir/torchbench_$(date +%Y%m%d_%H%M%S)_XXXXXX")
csv_helper prepare "$run_dir" "$models" "$bench_dir"
shapes=(static dynamic)
modes=(inference training)
metrics=(performance accuracy)
[[ "$shape" == all ]] || shapes=("$shape")
[[ "$mode" == all ]] || modes=("$mode")
[[ "$metric" == all ]] || metrics=("$metric")

echo "产物目录：$run_dir"
echo "Python：$python_bin"
echo "测试目录：$bench_dir"
trap 'echo "测试已中断；已完成结果保存在 $run_dir/results_running.csv" >&2; exit 130' INT
trap 'echo "测试已终止；已完成结果保存在 $run_dir/results_running.csv" >&2; exit 143' TERM
failures=0
total=0
while IFS= read -r -d '' index && IFS= read -r -d '' model_name && \
      IFS= read -r -d '' suite && IFS= read -r -d '' model_dir; do
    for current_shape in "${shapes[@]}"; do
        for current_mode in "${modes[@]}"; do
            for current_metric in "${metrics[@]}"; do
                shape_key=s
                mode_key=infer
                metric_key=perf
                [[ "$current_shape" != dynamic ]] || shape_key=d
                [[ "$current_mode" != training ]] || mode_key=train
                [[ "$current_metric" != accuracy ]] || metric_key=acc
                column="${shape_key}_${mode_key}_${metric_key}"
                case_dir="$model_dir/$column"
                mkdir -- "$case_dir"
                log_path="$case_dir/run.log"
                command=("$python_bin" "$bench_dir/$suite.py" "--$current_metric")
                if [[ "$current_metric" == performance ]]; then
                    command+=(--repeat 1 --iterations 1)
                else
                    command+=(--iterations 5)
                fi
                command+=("--$current_mode" --float32 --backend inductor --npu-backend triton)
                command+=(--disable-aclgraph --only "$model_name" --output "$case_dir/benchmark.csv")
                [[ "$current_shape" != dynamic ]] || command+=(--dynamic-shapes)
                {
                    printf '# cwd: %s\n' "$bench_dir"
                    printf '# TORCHBENCH_DATA_PATH=%q\n' "$TORCHBENCH_DATA_PATH"
                    printf '# HF_ENDPOINT=%q\n' "$HF_ENDPOINT"
                    printf '# command:'
                    printf ' %q' "${command[@]}"
                    printf '\n# started: %s\n' "$(date -Is)"
                } > "$log_path"
                echo "[$suite/$model_name] $column"
                set +e
                (cd -- "$bench_dir" && PYTHONUNBUFFERED=1 "${command[@]}" < /dev/null) 2>&1 | tee -a "$log_path"
                pipeline_status=("${PIPESTATUS[@]}")
                set -e
                exit_code=${pipeline_status[0]}
                [[ "${pipeline_status[1]}" == 0 ]] || die "日志写入失败：$log_path"
                printf '\n# exit_code: %s\n# finished: %s\n' "$exit_code" "$(date -Is)" >> "$log_path"
                result=$(csv_helper record "$run_dir" "$index" "$column" "$current_metric" "$exit_code" "$log_path")
                echo "[$suite/$model_name] $column = $result (exit_code=$exit_code)"
                total=$((total + 1))
                [[ "$result" != FAIL ]] || failures=$((failures + 1))
            done
        done
    done
done < "$run_dir/models.nul"
results_csv=$(csv_helper finish "$run_dir")
echo "汇总：$results_csv"
echo "完成 $total 项，失败 $failures 项"
[[ "$failures" == 0 ]]
