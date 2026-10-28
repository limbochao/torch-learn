"""Preserve ordered case results and sanitize pytest evidence for the boundary investigation."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r'/home/[^/\s]+/miniforge3/envs/[^/\s]+/lib/python3\.11/site-packages',
                   '<site-packages>', value)
    value = re.sub(r'/home/[^/\s]+/PTA_213/Community-Use-Case-Testing', '<test-repo>', value)
    value = re.sub(r'/home/[^/\s]+', '<home>', value)
    value = re.sub(r'miniforge3/envs/[^/\s]+', '<python-env>', value)
    value = re.sub(r'PID[:= ]+\d+', 'PID:<redacted>', value)
    value = re.sub(r'\bpid[:= ]+\d+', 'pid:<redacted>', value)
    value = re.sub(r'backend_hash[\'\"]?\s*[:=]\s*[\'\"][^\'\"]+[\'\"]',
                   'backend_hash=<redacted>', value)
    return value


def result_status(data, invocation):
    if invocation.get('timeout'):
        return 'TIMEOUT'
    if data.get('case_results'):
        reports = data['reports']
        calls = [r for r in reports if r['when'] == 'call']
        counts = Counter('XFAIL' if r.get('wasxfail') else r['outcome'].upper() for r in calls)
        for report in reports:
            if report['when'] != 'call' and report['outcome'] != 'passed':
                counts['XFAIL' if report.get('wasxfail') else report['outcome'].upper()] += 1
        return ', '.join(f'{n} {name}' for name, n in sorted(counts.items())) or 'NO_CALL'
    if invocation:
        return f'INCOMPLETE (exit={invocation.get("returncode")})'
    return 'RUNNING' if data else 'PENDING'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--triage', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = Path(__file__).with_name('cases_20261028.tsv')
    cases = [line.split('\t')[:2] for line in manifest.read_text().splitlines()]
    triage = json.loads(args.triage.read_text()) if args.triage else {}
    rows = []
    experiments = {
        **{n: 'control_no_overlay' for n in [1, 2, 39]},
        **{n: 'adaptation' for n in [5, 6, 7, 8]},
        **{n: 'candidate_v2' for n in [4, 19, 22]},
        **{n: 'candidate_v3' for n in [17, 41, 42]},
        **{n: 'adapted_extension' for n in [15, 16]},
        **{n: 'long_budget' for n in [24, 25]},
        **{n: 'multi_device' for n in [23, 34, 35]},
        9: 'conv_precision', 14: 'padding_long', 18: 'layout_long', 21: 'fusion_metadata',
    }
    for index, (filename, case) in enumerate(cases, 1):
        baseline_folder = 'baseline_v2' if index <= 10 else 'baseline_main' if index <= 23 else 'baseline_tail'
        row = {'序号': index, '文件': filename, '用例': case}
        for label, folder in [('基线', baseline_folder), ('适配复测', experiments.get(index))]:
            if folder is None:
                row[label] = '—'
                continue
            name = f'case-{index:02}.json'
            source = args.raw / folder / name
            invocation_file = source.with_name(f'case-{index:02}-invocation.json')
            data = json.loads(source.read_text()) if source.exists() else {}
            invocation = json.loads(invocation_file.read_text()) if invocation_file.exists() else {}
            row[label] = result_status(data, invocation)
            if data:
                destination = args.output / folder / name
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2) + '\n')
            if invocation:
                destination = args.output / folder / invocation_file.name
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(json.dumps(sanitize(invocation), ensure_ascii=False, indent=2) + '\n')
        details = triage.get(str(index), {})
        row.update({key: details.get(key, '待完成') for key in ['组件', '分类', '原因', '修复尝试与结论']})
        row['证据'] = f'{baseline_folder}/case-{index:02}.json'
        rows.append(row)
    with (args.output / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'progress.json').write_text(json.dumps({
        'rows': len(rows), 'baseline': dict(Counter(r['基线'] for r in rows)),
        'adapted': dict(Counter(r['适配复测'] for r in rows)),
        'triaged': sum(r['组件'] != '待完成' for r in rows),
    }, ensure_ascii=False, indent=2) + '\n')
    table = ['| 序号 | 文件 / 用例 | 基线 | 适配复测 | 组件 / 分类 | 原因 | 修复尝试与结论 |',
             '|---:|---|---|---|---|---|---|']
    for row in rows:
        cells = [str(row['序号']), f"`{row['文件']}`<br>`{row['用例']}`", row['基线'], row['适配复测'],
                 row['组件'] + ' / ' + row['分类'], row['原因'], row['修复尝试与结论']]
        table.append('| ' + ' | '.join(c.replace('|', '\\|').replace('\n', '<br>') for c in cells) + ' |')
    (args.output / 'table.md').write_text('\n'.join(table) + '\n')
    for folder in args.raw.iterdir():
        if not folder.is_dir():
            continue
        for source in folder.glob('*.json'):
            destination = args.output / folder.name / source.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps(sanitize(json.loads(source.read_text())),
                                             ensure_ascii=False, indent=2) + '\n')
    print((args.output / 'progress.json').read_text())


if __name__ == '__main__':
    main()
