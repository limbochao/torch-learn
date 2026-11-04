"""Read saved final autotune winners using the installed PyTorch cache-key implementation."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    from torch._inductor.runtime.autotune_cache import AutotuneCache

    references = {}
    for graph in args.job.glob('*/debug/**/output_code.py'):
        case_name = graph.relative_to(args.job).parts[0]
        if case_name.endswith('_accuracy'):
            continue
        source_path = None
        for line in graph.read_text().splitlines():
            if line.startswith('# kernel path: '):
                source_path = line.removeprefix('# kernel path: ').strip()
            elif ' = async_compile.triton(' in line and source_path:
                reference = f'{case_name}/graphs/{graph.parent.name}/output_code.py'
                references.setdefault(source_path, []).append(reference)
                source_path = None

    result = {}
    for case in sorted(args.job.iterdir()):
        if not (case / 'metadata/environment.json').exists():
            continue
        records = []
        for source in (case / 'cache').rglob('*.py'):
            text = source.read_text()
            names = re.findall(r"'kernel_name': '([^']+)'", text)
            if len(set(names)) != 1:
                continue
            key = AutotuneCache._prepare_key(str(source))
            best = Path(AutotuneCache._make_local_cache_key(str(source.parent), key))
            if best.exists():
                records.append({'kernel_name': names[0], 'source_file': source.name,
                                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                'used_by_graphs': references.get(str(source), []),
                                'best': json.loads(best.read_text())})
        result[case.name] = records
    (args.job / 'selected_configs.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Collected {sum(map(len, result.values()))} saved final configs')


if __name__ == '__main__':
    main()
