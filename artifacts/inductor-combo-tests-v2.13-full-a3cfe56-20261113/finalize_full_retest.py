"""Freeze source hashes and package only a completed native pytest run."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


root = Path(__file__).resolve().parent
assert root.parent == Path('/home/l30023782/PTA_213') and Path('/.dockerenv').is_file()
shards = json.loads((root / 'shards.json').read_text())
states = [json.loads((root / 'results' / item['name'] / 'status.json').read_text()) for item in shards]
assert all(state['stage'] == 'complete' for state in states), 'Workers still running or failed'
assert all(state['installed_source_unchanged'] for state in states), 'Installation changed'

expected = json.loads((root / 'test-source-before.json').read_text())
after = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
assert after == expected == json.loads((root / 'validation-source-sha256.json').read_text())
(root / 'test-source-after.json').write_text(json.dumps(after, indent=2) + '\n')
check = {'files': len(expected), 'unchanged': True, 'changed_files': [],
         'test_commit': states[0]['test_commit'], 'hardware_skip_patch': 'hardware-skip.patch',
         'test_method_bodies_changed_during_execution': False}
(root / 'test-source-validation.json').write_text(json.dumps(check, indent=2) + '\n')

installed_before = json.loads((root / 'full-collection/installed_sha256.json').read_text())
source_roots = set()
for name in installed_before:
    marker = '/torch_npu/_inductor/' if '/torch_npu/_inductor/' in name else '/torch/_inductor/'
    source_roots.add(Path(name.split(marker)[0] + marker.rstrip('/')))
installed_after = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in source_roots for path in directory.rglob('*.py')
}
assert installed_before == installed_after, 'Installed sources changed, added, or removed'
assert all(json.loads((root / 'results' / item['name'] / 'installed_sha256.json').read_text())
           == installed_before for item in shards)
(root / 'installed-source-after.json').write_text(json.dumps(installed_after, indent=2) + '\n')
triton_dir = root / 'installed-triton-source'
triton_manifest = json.loads((triton_dir / 'manifest.json').read_text())
for record in triton_manifest['source_files']:
    assert hashlib.sha256(Path(record['installed_path']).read_bytes()).hexdigest() == record['sha256']
    assert hashlib.sha256((triton_dir / record['copy']).read_bytes()).hexdigest() == record['sha256']
    if record['module'] == 'triton.backends.ascend.compiler':
        compiler_ast = ast.parse((triton_dir / record['copy']).read_text())
        options = next(node for node in compiler_ast.body
                       if isinstance(node, ast.ClassDef) and node.name == 'NPUOptions')
        record['option_class'] = options.name
        record['option_fields'] = [node.target.id for node in options.body
                                   if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
triton_manifest['option_lookup_corrected'] = {
    'previous_lookup': 'Options', 'actual_class': 'NPUOptions',
    'source_code_changed': False,
    'disable_ftz_registered': 'disable_ftz' in next(
        record['option_fields'] for record in triton_manifest['source_files']
        if record['module'] == 'triton.backends.ascend.compiler'),
}
(triton_dir / 'manifest.json').write_text(json.dumps(triton_manifest, indent=2) + '\n')
compiler_record = next(record for record in triton_manifest['source_files']
                       if record['module'] == 'triton.backends.ascend.compiler')
option_check = {
    'source_manifest': 'manifest.json',
    'classes_and_annotated_fields': {'NPUOptions': compiler_record['option_fields']},
    'disable_ftz_registered': triton_manifest['option_lookup_corrected']['disable_ftz_registered'],
}
(triton_dir / 'options-ast-check.json').write_text(json.dumps(option_check, indent=2) + '\n')
with (root / 'npu-smi-after.log').open('w') as stream:
    subprocess.run(['npu-smi', 'info'], stdout=stream, stderr=subprocess.STDOUT, check=True)
receipt = {'completed': True, 'checked_at': time.time(), 'workers': len(states),
           'test_files_unchanged': len(expected), 'installed_files_unchanged': len(installed_after),
           'triton_sources_unchanged': len(triton_manifest['source_files'])}
(root / 'completion-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

names = [
    'results', 'full-collection', 'environment.json', 'card-smoke.json',
    'npu-smi-before.log', 'npu-smi-after.log', 'collection-scope.json',
    'launch-stage.json', 'launch.json', 'shards.json', 'test-source-before.json',
    'test-source-after.json', 'test-source-validation.json', 'installed-source-before.json',
    'installed-source-after.json', 'installed-triton-source', 'completion-receipt.json',
    'launch-full.log', 'full-collection.run.log',
]
names.extend(item['name'] + '.run.log' for item in shards)
with tarfile.open(root / 'completed-results.tar.gz', 'w:gz') as archive:
    for name in names:
        archive.add(root / name, arcname=name)
print(json.dumps(receipt, ensure_ascii=False))
