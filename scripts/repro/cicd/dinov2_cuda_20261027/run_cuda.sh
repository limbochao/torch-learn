#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-trajectory}"
iterations="${2:-50}"
batch_size="${3:-4}"
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "batch_size must be a positive integer" >&2
  exit 2
fi
case_name="${mode}_${iterations}_bs${batch_size}_$(date +%Y%m%d_%H%M%S)"
out="$PWD/results/$case_name"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCHINDUCTOR_CACHE_DIR="$out/inductor_cache"
export TRITON_CACHE_DIR="$out/triton_cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$out/debug"
extra=()
if [[ "$mode" == trajectory ]]; then
  extra+=(--trajectory --backend inductor)
elif [[ "$mode" == benchmark ]]; then
  extra+=(--backend inductor)
elif [[ "$mode" == eager ]]; then
  extra+=(--execution-mode eager)
else
  echo "Usage: bash run_cuda.sh [trajectory|benchmark|eager] [iterations] [batch_size=4]" >&2
  exit 2
fi
nvidia-smi > "$out/nvidia-smi.txt"
python -m pip freeze > "$out/pip-freeze.txt"
python -u run_cuda.py "${extra[@]}" \
  --device cuda --accuracy --training --float32 --disable-aclgraph \
  --only vit_base_patch14_dinov2.lvd142m --dynamic-shapes --batch-size "$batch_size" \
  --iterations "$iterations" --accu-summary --output "$out/benchmark.csv" \
  2>&1 | tee "$out/run.log"
echo "Results: $out"
