"""Export the focused GNN evidence, with personal path prefixes sanitized."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def export(source, destination):
        if not source.is_file():
            return
        raw = source.read_bytes()
        text = raw.decode()
        usernames = set(re.findall(r'/home/([^/\s"\'<>]+)', text))
        text = text.replace(str(args.job.resolve()), '<job>')
        text = re.sub(r'/home/[^/\s"\'<>]+', '<home>', text)
        for username in usernames:
            if len(username) >= 6:
                text = text.replace(username, '<user>')
        text = re.sub(r'<user>_\d+_20\d+_ascend_pt', '<profile-trace>', text)
        text = re.sub(r'(\[(?:INFO|WARNING|ERROR|DEBUG)\]\s+)\[\d+\]', r'\1[pid]', text)
        text = re.sub(r'(?m)^# kernel path:.*$', '# kernel path: <generated-cache>', text)
        text = re.sub(r"('backend_hash': )'[^']+'", r"\1'<build-hash>'", text)
        target = args.output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        manifest[str(destination)] = {'original_sha256': hashlib.sha256(raw).hexdigest(),
                                      'sanitized_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

    export(args.job / 'summary.json', Path('summary.json'))
    export(args.job / 'selected_configs.json', Path('selected_configs.json'))
    export(args.job / 'runner_identity.json', Path('runner_identity.json'))
    for case in sorted(args.job.iterdir()):
        if not case.is_dir() or not case.name.startswith(('micro_', 'basic_gnn_', 'lennard_jones_', 'nfnet_l0_')):
            continue
        # The initial micro run used a stale raw stream during capture. Retain only
        # the corrected run; failed graph-output checks must never become timing evidence.
        if case.name.startswith('micro_') and not case.name.endswith('_verified'):
            continue
        for name in ['result.json', 'exitcode', 'benchmark.csv', 'accuracy.csv', 'run.log']:
            export(case / name, Path(case.name) / name)
        export(case / 'metadata/environment.json', Path(case.name) / 'environment.json')
        for profile in case.glob('profile/**/kernel_details.csv'):
            phase = 'dynamo' if 'dynamo' in profile.parts else 'eager'
            export(profile, Path(case.name) / 'profiles' / (phase + '.csv'))
        # Accuracy CSV/logs establish the result; duplicate accuracy graphs add no evidence.
        generated_graphs = [] if case.name.endswith('_accuracy') else case.glob('debug/**/output_code.py')
        for generated in generated_graphs:
            graph = generated.parent.name
            for name in ['output_code.py', 'fx_graph_readable.py', 'ir_pre_fusion.txt', 'ir_post_fusion.txt']:
                export(generated.parent / name, Path(case.name) / 'graphs' / graph / name)
        for binary in (case / 'ir').glob('**/*'):
            export(binary, Path(case.name) / binary.relative_to(case))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest)} files to {args.output}')


if __name__ == '__main__':
    main()
