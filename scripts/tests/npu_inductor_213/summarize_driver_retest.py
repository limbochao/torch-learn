"""Compare the restored-driver experiment with the archived, unmodified PR run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from summarize_batches import node_status, sanitize, write_csv


LABELS = {'passed': '通过', 'failed': '失败', 'skipped': '跳过', 'xfail_or_xpass': '预期失败'}
DRIVER_CHANGE = (
    '在完整 PR #47384（16437be0）上加回 transfer_to_npu.py::_patch_triton_driver() 及 _init() 调用；'
    '显式选择 Ascend driver，并替换 triton_helpers.set_driver_to_gpu。'
)
PR_CHANGE = (
    '采用 PR #47384（16437be0）的完整 transfer_to_npu.py；其中将 '
    'benchmarking._get_default_gpu_device_type 替换为返回 npu 的函数，消除 CUDA/NPU 重复设备判断。'
)
OVERLAY_EVIDENCE = '../npu-inductor-213-batches-20261008/skip-validation/manifest.json'


def pass_change(old, state):
    if state != '通过':
        return '尚未全部通过', '原用例尚未全部通过，不能将已应用的修改记为修复成功。', 'restore-driver.patch'
    if old['PR状态'] != '通过':
        return '加回 driver patch 后恢复', DRIVER_CHANGE, 'restore-driver.patch'
    if old['基线状态'] != '通过':
        return '采用 PR 后通过', PR_CHANGE, '../npu-inductor-pr47384-20261008/pr-transfer.patch'
    return '基线已通过', '基线环境已通过；采用 PR、再加回 driver patch 后均保持通过，无本轮新增修复。', ''


def overlay_change(index, method, manifest):
    changes = [f"{item['before']} → {item['after']}" for item in manifest['changes'] if item['method'] == method]
    description = '；'.join(changes) if changes else '保留原测试及真实 TMA/CUTLASS 条件，未伪造能力或强制解除 skip。'
    if index in ('163', '165', '166', '167'):
        description += '；使用 NPU 真实可用内存检查，保留原 shape/stride/offset。'
    if index in ('96', '197'):
        description += '；按 NPU 可用性、has_triton 和 TRITON_TEMPLATES 实际能力决定执行。'
    if index == '201':
        description += '；使用 torch_npu.npu.aclnn.flags(allow_hf32=False) 局部关闭卷积 HF32，结束恢复。'
    return '测试副本设置 PYTORCH_TEST_WITH_SLOW=1；' + description


def sanitize_evidence(value):
    if isinstance(value, list):
        return [sanitize_evidence(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_evidence(item) for key, item in value.items()}
    if isinstance(value, str):
        # Pytest may shorten a home path before it reaches the normal sanitizer.
        value = re.sub(r'''\.\.\.[^\s"'\\]*/PTA_213/pytorch/''', '...<pytorch>/', value)
    return sanitize(value)


def read_result(raw, filename, requested, expected_hash):
    path = raw / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if 'summary' not in data:
        return None
    assert not data['collection_errors'], filename
    assert data['environment']['transfer_sha256'] == expected_hash, filename
    result = data['case_results'][requested]
    nodes = result['nodeids']
    reports = [r for r in data['reports'] if r['nodeid'] in nodes]
    counts = Counter(node_status([r for r in reports if r['nodeid'] == n]) for n in nodes)
    assert nodes and not counts['no_call'], (filename, requested)
    state = LABELS[result['status']]
    if counts['passed'] and result['status'] == 'xfail_or_xpass':
        state = '部分通过/含预期失败'
    if counts['xpass']:
        state = '意外通过'
    if counts['passed'] and state == '跳过':
        state = '部分通过/部分跳过'
    errors = [r['longrepr'] for r in reports if r['outcome'] == 'failed']
    return {'state': state, 'counts': dict(counts), 'errors': errors, 'data': data, 'nodes': nodes}


def error_tail(errors):
    lines = []
    for error in errors:
        error = sanitize_evidence(error)
        significant = [s.strip() for s in error.splitlines() if s.startswith('E ') or s.startswith('E\t')]
        significant = [s for s in significant if s != 'E' and not any(marker in s for marker in (
            'To execute this test', 'python test/', 'This message can be suppressed',
        ))]
        chosen = significant[:8] if significant else error.splitlines()[-5:]
        lines.append('\n'.join(s[:350] + (' ...' if len(s) > 350 else '') for s in chosen))
    return '\n\n'.join(lines)


def change(before, after):
    if after == '待测试':
        return '待测试'
    if before == after:
        return '状态不变'
    if before == '通过':
        return '回归'
    if after == '通过':
        return '转为通过'
    return before + '→' + after


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    previous = out.parent / 'npu-inductor-pr47384-20261008'
    installation = json.loads((raw / 'installation.json').read_text())
    expected = installation['new_sha256']
    targeted_rows = {x['row'] for x in json.loads((raw / 'regressions.json').read_text())}
    old_rows = list(csv.DictReader((previous / 'results.csv').open(encoding='utf-8-sig')))
    manifest = json.loads((previous.parent / 'npu-inductor-213-batches-20261008/skip-validation/manifest.json')
                          .read_text())
    rows, instances, evidence = [], Counter(), {}
    plan = json.loads((raw / 'plan.json').read_text())
    for batch in plan:
        for index, requested in zip(batch['rows'], batch['cases'], strict=True):
            old = old_rows[index - 1]
            filename = (f'case-{index}-isolated-results.json' if index in (93, 94, 95)
                        else f"batch-{batch['batch']}-results.json")
            if index in targeted_rows:
                filename = 'targeted-results.json'
            result = read_result(raw, filename, requested, expected)
            state = result['state'] if result else '待测试'
            pass_kind, pass_description, pass_evidence = pass_change(old, state)
            row = {k: old[k] for k in ('序号', '批次', '文件', '测试类', '用例', '测试目的', 'NPU覆盖建议', '基线状态')}
            row.update({'PR状态': old['PR状态'], '补丁状态': state, '相对PR': change(old['PR状态'], state),
                        '本轮验证修改': DRIVER_CHANGE, '通过变化类型': pass_kind,
                        '通过对应修改': pass_description, '通过修改证据': pass_evidence,
                        '参数化统计': json.dumps(result['counts'] if result else {}, ensure_ascii=False),
                        '驱动错误仍存在': str(any('libcuda.so cannot found!' in e for e in result['errors']))
                        if result else '', '失败摘要': error_tail(result['errors']) if result else '',
                        '原PR结论': old['当前结论'],
                        '日志证据': filename + '#' + requested if result else '',
                        '目的源码证据': old['目的源码证据']})
            if result:
                instances.update(result['counts'])
                evidence[filename] = result['data']
            rows.append(row)
    assert [int(r['序号']) for r in rows] == list(range(1, 248))
    status = {'total': 247, 'status': dict(Counter(r['补丁状态'] for r in rows)),
              'changes': dict(Counter(r['相对PR'] for r in rows)), 'instances': dict(instances),
              'regressions': [r['序号'] for r in rows if r['相对PR'] == '回归'],
              'remaining_driver_errors': [r['序号'] for r in rows if r['驱动错误仍存在'] == 'True']}
    old_skips = list(csv.DictReader((previous / 'skip-validation.csv').open(encoding='utf-8-sig')))
    skip_rows, skip_instances = [], Counter()
    for old in old_skips:
        index = old['原序号']
        requested = old['文件'] + '::' + old['用例']
        filename = 'skip/case-' + ('201-hf32off' if index == '201' else index) + '.json'
        result = read_result(raw, filename, requested, expected)
        state = result['state'] if result else '待测试'
        adaptation = overlay_change(index, old['用例'], manifest)
        if state != '通过':
            pass_description = '适配复测仍未通过；下列适配条件不代表修复成功。'
        elif old['PR最终配置结果'] != '通过':
            pass_description = DRIVER_CHANGE + '在相同测试适配基础上，本轮消除 libcuda 驱动错误后恢复通过。'
        elif index == '201':
            pass_description = '解除 NPU skip 并局部关闭卷积 HF32 后通过；默认 HF32 配置仍失败，非 driver patch 改善。'
        elif old['基线适配结果'] != '通过':
            pass_description = PR_CHANGE + '该项在相同测试适配基础上已通过，本轮 driver patch 保持通过。'
        else:
            pass_description = '采用下列测试适配后，基线即已通过；PR 与本轮 driver patch 均保持通过。'
        row = {k: old[k] for k in ('原序号', '文件', '用例', '测试目的', 'NPU覆盖建议')}
        row.update({'PR适配结果': old['PR最终配置结果'], '补丁适配结果': state,
                    '相对PR': change(old['PR最终配置结果'], state),
                    '本轮验证修改': DRIVER_CHANGE, '测试适配修改': adaptation,
                    '通过对应修改': pass_description, '修改证据': OVERLAY_EVIDENCE + '；restore-driver.patch',
                    '参数化统计': json.dumps(result['counts'] if result else {}, ensure_ascii=False),
                    '失败摘要': error_tail(result['errors']) if result else '',
                    '日志证据': filename + '#' + requested if result else '',
                    '适配源码证据': old['适配源码证据']})
        skip_rows.append(row)
        if result:
            skip_instances.update(result['counts'])
            evidence[filename] = result['data']
    write_csv(out / 'skip-validation.csv', skip_rows, list(skip_rows[0]))
    overlays = {r['原序号']: r for r in skip_rows}
    for row in rows:
        overlay = overlays.get(row['序号'])
        row.update({
            '适配复测结果': overlay['补丁适配结果'] if overlay else '未另行适配复测',
            '适配测试修改': overlay['测试适配修改'] if overlay else '',
            '适配通过对应修改': overlay['通过对应修改'] if overlay else '',
            '适配修改证据': overlay['修改证据'] if overlay else '',
            '适配日志证据': overlay['日志证据'] if overlay else '',
        })
    write_csv(out / 'results.csv', rows, list(rows[0]))
    status['skip'] = {'total': 14, 'status': dict(Counter(r['补丁适配结果'] for r in skip_rows)),
                      'changes': dict(Counter(r['相对PR'] for r in skip_rows)), 'instances': dict(skip_instances)}
    for filename in ['targeted-results.json', 'driver-probe.json', 'installation.json', 'preflight.json',
                     'postflight.json', 'plan.json', 'regressions.json', 'skip-plan.json',
                     'execution-plan.json', 'failure-comparison.json', 'validation.json', 'review.json',
                     'import-order-probe.json',
                     'skip/case-201.json', 'arange-artifact.json', 'worker-probe.json']:
        path = raw / filename
        if path.exists():
            evidence[filename] = json.loads(path.read_text())
    for filename, data in evidence.items():
        target = out / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(sanitize_evidence(data), ensure_ascii=False, indent=2) + '\n')
    (out / 'restore-driver.patch').write_bytes((raw / 'restore-driver.patch').read_bytes())
    (out / 'progress.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
