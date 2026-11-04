#!/usr/bin/env bash
# Run from the dedicated TorchBench workspace after activating pta_213_cicd.
set -uo pipefail
job=$(realpath "$1")
group=$2
bundle=$job/bundle
export TORCHINDUCTOR_NPU_BACKEND=triton
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHNPU_PRECOMPILE_THREADS=8
export TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCH_COMPILE_DEBUG=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
kinds=(performance accuracy)
if [ "$group" = small ]; then
  export ASCEND_RT_VISIBLE_DEVICES=3
  specs=('basic_gnn_sage inference baseline' 'basic_gnn_sage inference scatter'
         'basic_gnn_sage training baseline' 'basic_gnn_sage training scatter'
         'lennard_jones training baseline' 'lennard_jones training split')
elif [ "$group" = pool ]; then
  export ASCEND_RT_VISIBLE_DEVICES=2
  specs=('nfnet_l0 training baseline' 'nfnet_l0 training pool-fallback')
elif [ "$group" = small-recheck ]; then
  export ASCEND_RT_VISIBLE_DEVICES=3
  specs=('basic_gnn_sage training baseline' 'basic_gnn_sage training scatter'
         'basic_gnn_sage training baseline' 'lennard_jones training baseline'
         'lennard_jones training split' 'lennard_jones training baseline')
  kinds=(performance)
else
  exit 2
fi
if [ -n "${3:-}" ]; then
  case "$3" in 1|2|3) export ASCEND_RT_VISIBLE_DEVICES=$3 ;; *) exit 2 ;; esac
fi
index=0
for kind in "${kinds[@]}"; do
  for spec in "${specs[@]}"; do
    read -r model mode variant <<< "$spec"
    run=$job/${model}_${mode}_${variant}
    cache=$run/cache
    if [ "$group" = small-recheck ]; then
      index=$((index + 1))
      run=${run}_recheck$index
    fi
    if [ "$kind" = accuracy ]; then run=${run}_accuracy; fi
    mkdir -p "$run"
    cache_group=$group
    if [ "$group" = small-recheck ]; then cache_group=small; fi
    export TORCHINDUCTOR_CACHE_DIR=$cache TRITON_CACHE_DIR=$job/triton_cache_$cache_group
    export TORCH_COMPILE_DEBUG_DIR=$run/debug
    echo "${model}_${mode}_${variant}_${kind}" > "$job/${group}_current"
    extra=()
    batch=1
    runner=torchbench.py
    if [ "$model" = lennard_jones ]; then batch=4096; extra+=(--split-min-reduction 4096); fi
    if [ "$model" = nfnet_l0 ]; then batch=128; runner=timm_models.py; fi
    if [ "$kind" = performance ]; then
      measure=(--performance --iterations 20 --repeat 3 --enable-profiler 0
               --prof-output-path "$run/profile" --output "$run/benchmark.csv")
    else
      measure=(--accuracy --iterations 2 --accu-summary --output "$run/accuracy.csv")
    fi
    timeout 2700 python "$bundle/scripts/repro/cicd/performance/gnn_validation/model_run.py" \
      --runner "$PWD/pytorch_new/benchmarks/torchbench/$runner" --output "$run/metadata" \
      --variant "$variant" "${extra[@]}" -- --only "$model" --device npu "--$mode" \
      --float32 --batch-size "$batch" --backend inductor --npu-backend triton \
      --disable-aclgraph "${measure[@]}" > "$run/run.log" 2>&1
    echo "$?" > "$run/exitcode"
  done
done
date -Is > "$job/${group}_finished_at"
