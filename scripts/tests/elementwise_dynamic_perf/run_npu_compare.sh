#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

RUN_ID=${RUN_ID:-npu_only_001}
PROFILE_ROOT=${PROFILE_ROOT:-prof_log/elementwise_dynamic_perf}
COMPILE_SHAPES=${COMPILE_SHAPES:-"128 8192 1048576"}
SYMBOLIC_DIMS=${SYMBOLIC_DIMS:-0}
CASES=${CASES:-memory_add,exp_log}
PYTHON_BIN=${PYTHON_BIN:-python}

CASE_SCRIPT=scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
COMPARE_SCRIPT=scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_compare.py
COMMON_ENV=("RUN_ID=$RUN_ID" "PROFILE_ROOT=$PROFILE_ROOT" "DEVICE=npu" "CASES=$CASES")

run_case() {
    local execution=$1
    shift
    env "${COMMON_ENV[@]}" "EXECUTION=$execution" "$@" "$PYTHON_BIN" "$CASE_SCRIPT"
}

# Eager does not compile, so no compile cache cleanup is needed.
run_case eager

# Static compiles one specialized kernel for every runtime shape.
rm -rf /tmp/torchinductor_root/*
run_case static

# Each dynamic run compiles once with its own first shape, then reuses that kernel.
read -r -a compile_shapes <<< "$COMPILE_SHAPES"
for compile_shape in "${compile_shapes[@]}"; do
    rm -rf /tmp/torchinductor_root/*
    run_case dynamic "COMPILE_SHAPE=$compile_shape" "SYMBOLIC_DIMS=$SYMBOLIC_DIMS"
done

# Condense all executions and first shapes into CSV and XLSX reports.
"$PYTHON_BIN" "$COMPARE_SCRIPT" "$PROFILE_ROOT/$RUN_ID/summary.csv"
