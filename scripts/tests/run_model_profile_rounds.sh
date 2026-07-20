#!/usr/bin/env bash
PTA="${PTA:-xxx}"
TA="${TA:-xxx}"
CANN="${CANN:-xxx}"
test_command='source /root/.bashrc;export REPORT_ID=ca909532-1499-4ab4-a92c-2189a1db0479;export SAVE_PROFILING=TRUE;bash /data/autotest/20260716115918983/model_test/script/common_run_cmd.sh /data/autotest/20260716115918983/model_test/data/RECSDK_REPO/rec_model_zoo_pytorch/behaviour_and_multi_task mmoe 6 inductor True 128'
rounds="${1:-3}"
report_id="$(sed -n 's/.*export REPORT_ID=\([^;[:space:]]*\).*/\1/p' <<< "${test_command}")"
autotest_root="$(grep -o '/data/autotest/[^/[:space:];]*' <<< "${test_command}" | head -1)"
[[ -n "${report_id}" && -n "${autotest_root}" ]] || { echo 'Failed to parse REPORT_ID or autotest path' >&2; exit 1; }
compile_root="${autotest_root}/model_test/data/dump/${report_id}/torch_compile_debug"
result_dir="${RESULT_DIR:-profile_results}"
mkdir -p "${result_dir}/${PTA}_${TA}_${CANN}"
results_csv="${result_dir}/total_results.csv"
[[ -f "${results_csv}" ]] || \
    echo 'PTA,TA,CANN,round,time,kernel_column_11_total,profile_path,compile_path' > "${results_csv}"
for ((i = 1; i <= rounds; i++)); do
    rm -rf /tmp/torchinductor_root/*
    time="$(date +%Y%m%d_%H%M%S_%N)"
    run_dir="${result_dir}/${PTA}_${TA}_${CANN}/${time}"; mkdir -p "${run_dir}"; touch "${run_dir}/start"
    bash -c "${test_command}" > "${run_dir}/run.log" 2>&1
    profile_path="$(sed -n 's/.*profile_path:[[:space:]]*//p' "${run_dir}/run.log" | tail -1 | tr -d '\r')"
    compile_path="$(find "${compile_root}" -mindepth 1 -maxdepth 1 -type d -newer "${run_dir}/start" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    [[ -n "${profile_path}" && -n "${compile_path}" ]] || {
        echo "Failed to find profile or compile path: ${run_dir}/run.log" >&2; exit 1;
    }
    total="$(awk -F, 'NR>1 {sum+=$11} END {print sum}' "${profile_path}/ASCEND_PROFILER_OUTPUT/kernel_details.csv")"
    profile_copy_path="${run_dir}/profile"; compile_copy_path="${run_dir}/compile"
    cp -r "${profile_path}" "${profile_copy_path}"; cp -r "${compile_path}" "${compile_copy_path}"
    echo "${PTA},${TA},${CANN},${i},${time},${total},${profile_copy_path},${compile_copy_path}" >> "${results_csv}"
done
