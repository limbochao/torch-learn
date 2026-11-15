set -euo pipefail
source <workspace>
conda activate pta_213_case
combo_root=<workspace>
cd "$combo_root"
python - <<'PY'
import json
from pathlib import Path
root = Path('<workspace>')
state = json.loads((root / 'pass-regression/status.json').read_text())
context = json.loads((root / 'pass-validation-context.json').read_text())
assert state['stage'] == 'complete' and state['installed_source_unchanged']
assert context['test_source_unchanged']
assert (root / 'pass-evidence.tar.gz').exists(), 'Freeze the full-run evidence before changing test sources'
PY
git -C Community-Use-Case-Testing pull --ff-only origin fix/combo_case
test "$(git -C Community-Use-Case-Testing rev-parse HEAD)" = 023fd814784896f3e2a11186787fc40b2ebc60c7
export TMPDIR="$combo_root/tmp"
export ASCEND_RT_VISIBLE_DEVICES=1
export TORCH_LOGS=output_code,inductor_metrics
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$combo_root/followup-triton-cache"
python - <<'PY'
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
import torch_npu

root = Path('<workspace>')
repo = root / 'Community-Use-Case-Testing'
base = repo / 'v2.13.0/inductor_npu'
name = 'v2.13.0/inductor_npu/test_combo_kernels.py'
original = subprocess.check_output(['git', 'show', '7332db7:' + name], cwd=repo, text=True)
current = (repo / name).read_text()
allowed = {'test_combo_kernel_per_config_subkernel_poi', 'test_combo_kernel_per_config_subkernel_r0_r1'}
def normalized(source):
    tree = ast.parse(source)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'ComboKernelTests':
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in allowed:
                    node.body = [ast.Pass()]
    return ast.dump(tree)
assert normalized(original) == normalized(current)
snode = 'v2.13.0/inductor_npu/test_snode_runtime.py'
assert subprocess.check_output(['git', 'show', '7332db7:' + snode], cwd=repo) == (repo / snode).read_bytes()
sources = list(Path(torch_npu.__file__).parent.joinpath('_inductor').rglob('*.py'))
sources += list(Path(torch.__file__).parent.joinpath('_inductor').rglob('*.py'))
fingerprint = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
assert fingerprint == json.loads((root / 'pass-regression/installed_sha256.json').read_text())
context = {'test_commit': '023fd814784896f3e2a11186787fc40b2ebc60c7',
           'npu_fix_commit': '5db866a16403644c265f3f43d747a856124de8b1',
           'TORCH_LOGS': os.environ['TORCH_LOGS'], 'installed_source_files': len(fingerprint),
           'all_installed_match_full_regression': True,
           'only_two_method_bodies_changed': sorted(allowed), 'all_other_test_AST_unchanged': True,
           'SNode_test_source_unchanged': True,
           'selected_methods': 4, 'selected_instances': 4,
           'reason': 'Two NPU structural assertion adaptations; restore metrics collection for both SNode cases.',
           'basic_rerun_reason': 'The full-run output_code-only environment suppressed set_logs and could pass an empty loop.',
           'test_source_sha256_before': {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                                        for name in ('test_combo_kernels.py', 'test_snode_runtime.py')}}
(root / 'followup-validation-context.json').write_text(json.dumps(context, indent=2) + '\n')
PY
python run_plain_pytest.py \
  --test-root "$combo_root/Community-Use-Case-Testing/v2.13.0/inductor_npu" \
  --manifest followup-manifest.json --layout table_rows.json \
  --output "$combo_root/case-followup" --cache-root "$combo_root/case-followup-cache" \
  --test-commit 023fd814784896f3e2a11186787fc40b2ebc60c7 \
  --npu-fix-commit 5db866a16403644c265f3f43d747a856124de8b1 \
  --devices 1,2,3 --foreach-functional > case-followup-run.log 2>&1
cd "$combo_root/Community-Use-Case-Testing/v2.13.0/inductor_npu"
PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}" \
TORCHINDUCTOR_CACHE_DIR="$combo_root/snode-proof-cache" \
python "$combo_root/probe_snode_metrics.py" > "$combo_root/snode-metrics-proof.log" 2>&1
cd "$combo_root"
python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('<workspace>')
state = json.loads((root / 'case-followup/status.json').read_text())
assert state['stage'] == 'complete' and state['installed_source_unchanged']
assert state['method_results'] == state['instance_results'] == {'PASS': 4}, state
context = json.loads((root / 'followup-validation-context.json').read_text())
base = root / 'Community-Use-Case-Testing/v2.13.0/inductor_npu'
after = {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
         for name in context['test_source_sha256_before']}
assert after == context['test_source_sha256_before']
context.update(test_source_sha256_after=after, test_source_unchanged=True,
               all_expected_instances_reported=True, all_four_methods_passed=True)
(root / 'followup-validation-context.json').write_text(json.dumps(context, indent=2) + '\n')
print(json.dumps(state))
PY
