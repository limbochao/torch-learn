"""Summarize the four model experiments from exported NPU kernel_details.csv files."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median


def summarize_profile(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    steps = defaultdict(list)
    for row in rows:
        step = row.get('Step Id', '')
        if step.isdigit():
            steps[int(step)].append(row)
    if not steps:
        raise ValueError(f'No numbered profiler steps in {path}')
    totals = {}
    groups = defaultdict(lambda: {'ms': 0.0, 'calls': 0})
    windows = []
    for step, entries in sorted(steps.items()):
        totals[step] = sum(float(r['Duration(us)']) for r in entries) / 1000
        for index, row in enumerate(entries):
            name = row['Name']
            groups[name]['ms'] += float(row['Duration(us)']) / 1000 / len(steps)
            groups[name]['calls'] += 1
            if 'softmax' in name.lower() and 'backward' in name.lower():
                part = entries[index + 1:index + 5]
                if len(part) == 4 and all(
                    part[j]['Name'].startswith('aclnnMul_' if j % 2 == 0 else 'aclnnAdd_')
                    for j in range(4)
                ):
                    windows.append({'step': step, 'row_after_header': rows.index(row) + 1,
                                    'preceding_kernel': name, 'names': [r['Name'] for r in part],
                                    'first_pair_ms': sum(float(r['Duration(us)']) for r in part[:2]) / 1000,
                                    'all_four_ms': sum(float(r['Duration(us)']) for r in part) / 1000})
    for group in groups.values():
        group['calls_per_step'] = group.pop('calls') / len(steps)
    return {'profile_steps': len(steps), 'step_ms': totals, 'mean_ms': mean(totals.values()),
            'median_ms': median(totals.values()), 'min_ms': min(totals.values()), 'max_ms': max(totals.values()),
            'duplicate_windows_per_step': len(windows) / len(steps),
            'duplicate_first_pair_ms': sum(w['first_pair_ms'] for w in windows) / len(steps),
            'four_op_window_ms': sum(w['all_four_ms'] for w in windows) / len(steps),
            'groups': dict(sorted(groups.items(), key=lambda item: -item[1]['ms'])), 'windows': windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for case in sorted(args.job_root.iterdir()):
        if not case.is_dir() or not (case / 'metadata/environment.json').exists():
            continue
        row = {'exitcode': (case / 'exitcode').read_text().strip() if (case / 'exitcode').exists() else None,
               'profiles': {}}
        for profile in case.glob('profile/**/kernel_details.csv'):
            phase = 'dynamo' if 'dynamo' in profile.parts else 'eager' if 'eager' in profile.parts else 'unknown'
            if phase in row['profiles']:
                raise ValueError(f'Multiple {phase} profiles in {case}; do not silently merge separate runs')
            row['profiles'][phase] = summarize_profile(profile)
        result[case.name] = row
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for case, row in result.items():
        print(case, 'exitcode', row['exitcode'])
        for phase, stats in row['profiles'].items():
            print(' ', phase, f"{stats['mean_ms']:.6f} ms/step", 'FMA windows',
                  stats['duplicate_windows_per_step'], 'first pair', stats['duplicate_first_pair_ms'])


if __name__ == '__main__':
    main()
