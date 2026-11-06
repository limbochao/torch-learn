"""Capture the environment and generated Python sources without changing the installed packages."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
site = Path('/usr/local/python3.11.15/lib/python3.11/site-packages')
source_paths = [
    'torch_npu/_inductor/codegen/triton.py',
    'torch_npu/_inductor/codegen/npu_kernel_features.py',
    'torch_npu/_inductor/codegen/ir.py',
    'torch_npu/_inductor/codegen/triton_combo_kernel.py',
    'torch_npu/_inductor/codegen/scheduling.py',
    'torch_npu/_inductor/runtime/triton_heuristics.py',
    'torch_npu/_inductor/config.py',
    'torch/_inductor/codegen/triton_combo_kernel.py',
]
source_hashes = {}
for relative in source_paths:
    source = site / relative
    target = OUT / 'installed_source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
(OUT / 'installed_source_sha256.json').write_text(json.dumps(source_hashes, indent=2))

test_hashes = {}
selected = json.loads((ROOT / 'combo_npu_a5_selected.json').read_text())
for filename in sorted({method['file'] for method in selected}):
    executed = ROOT / 'Community-Use-Case-Testing/v2.13.0/inductor_npu' / filename
    original = ROOT / 'original_cases' / filename
    for kind, source in [('executed_tests', executed), ('original_tests', original if original.exists() else executed)]:
        target = OUT / kind / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        test_hashes[str(target.relative_to(OUT))] = hashlib.sha256(source.read_bytes()).hexdigest()
(OUT / 'test_source_sha256.json').write_text(json.dumps(test_hashes, indent=2))
for source in ROOT.glob('*.py'):
    target = OUT / 'scripts' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
shutil.copy2(ROOT / 'combo_npu_a5_selected.json', OUT / 'scripts/combo_npu_a5_selected.json')
if (ROOT / 'clone.log').exists():
    shutil.copy2(ROOT / 'clone.log', OUT / 'clone.log')

# Capture generated Python kernels/wrappers. Large compiler caches and binaries stay on the server.
manifest = []
for path in sorted((ROOT / 'cache').rglob('*.py')):
    data = path.read_bytes()
    if b'async_compile' not in data and b'def triton_' not in data:
        continue
    relative = path.relative_to(ROOT / 'cache')
    target = OUT / 'generated_python' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    manifest.append({'cache_file': str(path), 'artifact': str(target.relative_to(OUT)),
                     'sha256': hashlib.sha256(data).hexdigest()})
(OUT / 'generated_python_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Captured installed source files:', len(source_hashes), 'generated Python files:', len(manifest))
