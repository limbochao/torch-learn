"""Archive completed liulm Combo phases, retaining source hashes and sanitizing logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def clean(text):
    text = re.sub(r'/home/[^\s"\'<>]+', '<workspace>', text)
    text = re.sub(r"(['\"])backend_hash\1\s*:\s*(['\"])[0-9A-Fa-f]+\2",
                  "'backend_hash': '<removed>'", text)
    return re.sub(r'\bPID[:= ]+[0-9]+', 'PID=<removed>', text)


def clean_value(value):
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, list):
        return [clean_value(x) for x in value]
    if isinstance(value, dict):
        return {key: '<removed>' if key == 'backend_hash' else clean_value(x) for key, x in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.raw_root.resolve()
    target = args.artifact_root.resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = []

    def copy(path, relative):
        raw = path.read_bytes()
        text = original = raw.decode()
        if path.suffix == '.json':
            value = json.loads(text)
            if path.name == 'installation.json' and 'wheel_metadata' in value:
                value['wheel_metadata'] = {
                    name: {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                           for line in metadata.splitlines()
                           if line.startswith(('Name:', 'Version:', 'Metadata-Version:'))}
                    for name, metadata in value['wheel_metadata'].items()
                }
            updated = clean_value(value)
            if updated != json.loads(text):
                text = json.dumps(updated, ensure_ascii=False, indent=2) + '\n'
        elif path.suffix == '.jsonl':
            text = '\n'.join(json.dumps(clean_value(json.loads(line)), ensure_ascii=False)
                             for line in text.splitlines()) + '\n'
        else:
            text = clean(text)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        manifest.append({'artifact': str(relative), 'original_sha256': hashlib.sha256(raw).hexdigest(),
                         'archived_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'sanitized': text != original})

    for path in (source / 'install').iterdir():
        if path.name != 'raw_import.log':
            copy(path, Path('install') / path.name)
    metadata = ['resource_authorization.json', 'expected_raw_source_sha256.json',
                'expected_source_sha256.json', 'run_focused_then_wait.py', 'run_full_after_docs.py',
                'docs_push_receipt.json', 'pipeline_status.json', 'full_snapshot.json']
    for name in metadata:
        if (source / name).exists():
            copy(source / name, Path('execution') / name)
    summary = {}
    for phase in ['failed_first', 'with_fixes', 'full']:
        evidence = source / phase / 'evidence'
        if not (evidence / 'integrity_checks.json').exists():
            continue
        result = json.loads((evidence / 'functional_results.json').read_text())
        check = json.loads((evidence / 'integrity_checks.json').read_text())
        assert check['all_expected_instances_reported'] and check['installed_source_unchanged']
        summary[phase] = {
            'methods': len(result['methods']),
            'passed_methods': sum(m['returncode'] == 0 and m['counts'].get('passed', 0) == m['expected_instances']
                                  for m in result['methods']),
            'instances': result['expected_instances'], 'counts': result['instance_counts'],
        }
        failed = {m['file'].removesuffix('.py') + '.' + m['name']
                  for m in result['methods'] if m['returncode'] != 0}
        files = {'functional_results.json', 'integrity_checks.json', 'environment.json', 'collection.json',
                 'collection_equivalence.json', 'test_equivalence.json', 'run_summary.json', 'jobs.jsonl',
                 'test_source_sha256.json'}
        for path in evidence.iterdir():
            if not path.is_file():
                continue
            reports = path.name.startswith(('test_', 'audit.')) and path.suffix == '.jsonl'
            failed_log = path.suffix == '.log' and path.stem in failed
            if path.name in files or reports or failed_log:
                copy(path, Path(phase) / path.name)
        if phase == 'failed_first':
            for path in (evidence / 'installed_source').rglob('*.py'):
                copy(path, Path('installed_source') / path.relative_to(evidence / 'installed_source'))
            for path in (evidence / 'scripts').iterdir():
                if path.suffix in {'.py', '.json'}:
                    copy(path, Path('scripts') / path.name)
        for path in (evidence / 'combo_sources').rglob('*.py'):
            if phase != 'full' or any(name in str(path) for name in
                                      ['mixed_x_dim_cuda', 'helper_fn_defined', 'subkernel_red']):
                copy(path, Path(phase) / path.relative_to(evidence))
        known = {'test_adaptive_avg_pool2d1', 'test_rand_like_deterministic',
                 'test_combo_kernel_per_config_subkernel_block_size',
                 'test_triton_kernel_to_post_grad_tracing_combo_kernel'}
        for path in (evidence / 'generated_on_failure').rglob('*.py'):
            preserve = ('persistent_reduction_mixed_x_dim_cuda' in str(path)
                        or 'u0 = buf0.size(0)' in path.read_text())
            unknown = phase == 'full' and not any(name in str(path) for name in known)
            if preserve or unknown:
                copy(path, Path(phase) / path.relative_to(evidence))
    (target / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = {
        'baseline': json.loads((source / 'install/installation.json').read_text())['commit'],
        'prerequisite_fix': 'one indentation space in scheduling.py',
        'existing_f3_commit': 'e578ec524d592475fd52702b975ce03d1a95bbf2',
        'phase_results': summary,
        'execution_order': ['failed_first', 'with_fixes', 'docs_update_and_push', 'full_with_f3'],
        'full_complete': 'full' in summary,
    }
    (target / 'validation_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'archived_files': len(manifest), 'phase_results': summary}))


if __name__ == '__main__':
    main()
