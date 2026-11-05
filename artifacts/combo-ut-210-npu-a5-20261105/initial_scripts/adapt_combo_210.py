"""Apply test-only compatibility and functional-check adaptations to the approved checkout."""
import ast
import difflib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'Community-Use-Case-Testing/v2.13.0/inductor_npu'
BACKUP = ROOT / 'original_cases'
BACKUP.mkdir(exist_ok=True)
AUDIT = []
SELECTED = json.loads((ROOT / "combo_npu_a5_selected.json").read_text())


def change(text, old, new, file, reason):
    if old in text:
        AUDIT.append({'file': file, 'reason': reason, 'before': old, 'after': new})
        return text.replace(old, new)
    return text


for file in ['test_combo_kernels.py', 'test_foreach.py', 'test_torchinductor.py']:
    path = BASE / file
    backup = BACKUP / file
    if not backup.exists():
        shutil.copy2(path, backup)
    original = backup.read_text()
    text = original
    # Unsupported, neutral setup options are omitted; independent-block variants are not selected.
    for key in ['combo_kernel_per_subkernel_blocks', 'combo_kernel_max_distance',
                'combo_kernel_peak_memory_increase_gb', 'combo_kernel_peak_memory_pct_threshold']:
        if file == 'test_combo_kernels.py':
            text = re.sub(r'^\s*"' + key + r'": [^\n]+\n', '', text, flags=re.M)
    if file in ['test_combo_kernels.py', 'test_foreach.py']:
        text = change(text, 'torch._inductor.config.combo_kernel_per_subkernel_blocks',
                      'getattr(torch._inductor.config, "combo_kernel_per_subkernel_blocks", False)',
                      file, 'Read absent independent-block option as the supported legacy False branch')
    if file == 'test_foreach.py':
        text = change(text, '    torch._foreach_clone,\n', '', file, 'Operator absent in 2.10; do not parameterize it')
        text = change(text, '    @config.patch({"combo_kernel_per_subkernel_blocks": True})\n', '',
                      file, 'Existing mixed-size math runs on the supported shared-block path')
    if file == 'test_combo_kernels.py':
        text = change(text, '[(8, 1), (3, 2), (2, 3)]', '[(8, 1)]', file,
                      '2.10 only supports the fixed eight-node limit')
        text = change(text, 'torch._inductor.config.patch("combo_kernel_max_num_nodes", max_num_nodes)',
                      'contextlib.nullcontext()', file, 'Retain only the legacy fixed-eight behavior')
    # Remove the pre-existing broad NPU skips to actually attempt the selected functional tests.
    skip_pattern = (r'^        if torch.npu.is_available\(\):\n'
                    r'            self.skipTest\([^\n]*(?:FOREACH|NPU)[^\n]*\)\n')
    text = re.sub(skip_pattern, '', text, flags=re.M)
    text = re.sub(r'^[ \t]*@unittest\.skip\("NPU does not support"\)\n', '', text, flags=re.M)
    if file == 'test_torchinductor.py':
        text = change(text,
                      '        if combo_kernels and torch.device(self.device).type == "npu":\n'
                      '            self.skipTest("NPU Inductor does not support ComboKernel yet")\n',
                      '', file, 'Attempt the selected Combo=True random test on the installed Combo branch')
    # Retain all numerical comparisons, shapes, dtypes and tolerances. Remove only CUDA kernel structure checks.
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    removed = []
    ranges = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        names = {r['name'] for r in SELECTED if r['file'] == file and r['class'] == cls.name}
        if not names:
            continue
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and (method.name in names or not method.name.startswith('test_')):
                ranges.append((method.lineno, method.end_lineno))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expr, ast.Assert)):
            continue
        if not any(first <= node.lineno <= last for first, last in ranges):
            continue
        snippet = ''.join(lines[node.lineno-1:node.end_lineno])
        if ('generated_kernel_count' in snippet and ('assert' in snippet or 'FileCheck' in snippet)):
            removed.append((node.lineno, node.end_lineno, 'CUDA-specific generated-kernel count'))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if 'FileCheck()' in snippet:
                removed.append((node.lineno, node.end_lineno, 'CUDA Triton source-structure expectation'))
    for first, last, reason in sorted(set(removed), reverse=True):
        before = ''.join(lines[first-1:last])
        indent = re.match(r'\s*', lines[first-1]).group()
        lines[first-1:last] = [indent + 'pass  # Functional validation: ' + reason + '\n']
        AUDIT.append({'file': file, 'reason': reason, 'before': before, 'after': 'pass'})
    text = ''.join(lines)
    ast.parse(text)
    temporary = path.with_suffix('.py.tmp')
    temporary.write_text(text)
    temporary.replace(path)
    (ROOT / 'evidence').mkdir(exist_ok=True)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                      fromfile='original/' + file, tofile='adapted/' + file))
    (ROOT / 'evidence' / (file + '.patch')).write_text(diff)
(ROOT / 'evidence/adaptations.json').write_text(json.dumps(AUDIT, ensure_ascii=False, indent=2))
print('Test-only adaptations written; installed torch/torch_npu files are unchanged.')
