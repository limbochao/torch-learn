"""Archive repair evidence and merge it with all 247 original rows without losing history."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from summarize_batches import node_status, write_csv
from summarize_driver_retest import error_tail, sanitize_evidence


PREVIOUS = 'npu-inductor-pr47384-restored-driver-20261009'
DRIVER_HASH = '49a970650f02ffdb71ceff01d0f681ea88eb5f70af1f5eff01ea55cced9acf3a'
PASS_CHANGES = {
    **dict.fromkeys(range(51, 56), '用例：将 const=torch.tensor(0.0) 建立在 GPU_TYPE/npu，与输入同设备。'),
    **dict.fromkeys((68, 69), '用例：custom-op 同一 fn 新增 PrivateUse1 注册，保留 stride/对齐断言。'),
    **dict.fromkeys(range(132, 137), 'Inductor：dtype_optimal_pass 仅对 Tensor.to 的 Tensor 输入读取 dtype。'),
    **dict.fromkeys((188, 189), 'Inductor：NPU tuned_bmm 接入浮点小点积 mul+sum，保留静态 K=32/33 和动态范围检查。'),
    76: '用例：基类 NPU benchmark 显式检查 NotImplementedError，其他参数仍验证 timing/counter。',
    77: '用例：基类 benchmark_gpu 显式检查 NotImplementedError，TritonBenchmarker 保留原正向断言。',
    78: '用例：将 expectedFailure 改为精确检查无设备输入的 ValueError。',
    79: '用例：make_sum 改为已有 make_params，并精确检查多设备输入的 ValueError。',
    83: '测试环境：局部关闭卷积 HF32，保留原外层参数化 test_basic、容差和优化计数断言。',
    90: 'Inductor：仅内建 POINTWISE autotuner 添加 is_vetted_benchmarking 标记，确定性开关保持启用。',
    96: '用例：按真实 NPU 模板能力解除 SM 门槛；Inductor：force_fallback 支持 HigherOrderOperator 恢复。',
    97: 'Inductor：force_fallback 接受单个 HigherOrderOperator，保留 handler 恢复及异步编译流程。',
    190: '用例：原 int64 BMM 输入分别检查 eager/compiled 的 NPU RuntimeError 和 DT_INT64 错误。',
    195: '用例：原 mixed-dtype MM 输入检查 NPU matB/DT_INT64 错误。',
    196: '用例：分别匹配 NPU eager matA/DT_INT64 和 compiled scalar-type 错误，保留 autograd 负向断言。',
    243: '用例：对 join(source_codes) 执行原 check_not，允许零代码，保留数值/shape 检查。',
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_evidence(value), ensure_ascii=False, indent=2) + '\n')


def historical_ref(value):
    return '；'.join(part if not part or part.startswith('../') else '../' + PREVIOUS + '/' + part
                     for part in value.split('；'))


def archive_round(raw, name, out):
    changes = json.loads((raw / 'changes.json').read_text())
    target = out / name
    target.mkdir(exist_ok=True)
    for change in changes:
        if change.get('prerequisite'):
            change['patch'] = '../../' + PREVIOUS + '/restore-driver.patch'
            continue
        source = raw / change['patch']
        destination = target / change['patch']
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    save_json(target / 'changes.json', changes)
    save_json(target / 'preflight.json', json.loads((raw / 'preflight.json').read_text()))
    results = {}
    launch = json.loads((raw / 'launch.json').read_text())
    expected = {row for lane in launch.values() for row in lane['rows']}
    for index in sorted(expected):
        filename = f'case-{index}.json'
        data = json.loads((raw / name / filename).read_text())
        invocation = json.loads((raw / name / f'case-{index}-invocation.json').read_text())
        assert data.get('summary') and not data.get('collection_errors'), (name, index)
        assert not invocation.get('timeout'), (name, index)
        assert data['environment']['transfer_sha256'] == DRIVER_HASH
        nodes = data['selected']
        counts = Counter(node_status([r for r in data['reports'] if r['nodeid'] == node]) for node in nodes)
        assert nodes and not counts['no_call'] and not counts['xpass'], (name, index, counts)
        hashes = data['environment']['test_sha256']
        for change in changes:
            if change['domain'] == 'test' and change['file'] in hashes:
                assert hashes[change['file']] == change['after_sha256'], (name, index, change['file'])
        state = '失败' if counts['failed'] else ('跳过' if counts['skipped'] else (
            '含预期失败' if counts['xfail'] else '通过'))
        errors = [r['longrepr'] for r in data['reports'] if r['outcome'] == 'failed']
        evidence = name + '/' + filename
        save_json(out / evidence, data)
        save_json(target / f'case-{index}-invocation.json', invocation)
        results[index] = {'state': state, 'counts': dict(counts), 'errors': error_tail(errors),
                          'evidence': evidence, 'changes': name + '/changes.json'}
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--first-dir', type=Path, required=True)
    parser.add_argument('--second-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    review = json.loads((out / 'review.json').read_text())
    previous = list(csv.DictReader((out.parent / PREVIOUS / 'results.csv').open(encoding='utf-8-sig')))
    required = {row['序号'] for row in previous if row['补丁状态'] != '通过'}
    assert set(review) == required and len(required) == 100
    first = archive_round(args.first_dir, 'first', out)
    second = archive_round(args.second_dir, 'second', out)
    latest = first | second
    rows, triage = [], []
    for old in previous:
        row = dict(old)
        for key in ('通过修改证据', '日志证据', '目的源码证据', '适配修改证据', '适配日志证据'):
            row[key] = historical_ref(row[key])
        index = int(row['序号'])
        result = latest.get(index)
        assessment = review.get(str(index), {})
        state = result['state'] if result else row['补丁状态']
        row.update({
            '首轮修复结果': first.get(index, {}).get('state', '未另行复测'),
            '第二轮修复结果': second.get(index, {}).get('state', '未另行复测'),
            '累计最新结果': state,
            '本轮参数化统计': json.dumps(result['counts'], ensure_ascii=False) if result else '',
            '本轮日志证据': result['evidence'] if result else '',
            '本轮修改清单': result['changes'] if result else '',
            '本轮失败摘要': result['errors'] if result else '',
            '修复组': assessment.get('修复组', ''),
            '是否必要及理由': assessment.get('是否必要及理由', '上轮已通过；本轮仅选择受改动影响的相关回归。'),
            '修复方向': assessment.get('修复方向', ''),
            '负责模块及位置': assessment.get('负责模块及位置', ''),
            '评审处置': assessment.get('评审处置', '保持既有结论。'),
            '方案及验收': assessment.get('方案及验收', ''),
            '风险或不修原因': assessment.get('风险或不修原因', ''),
            '当前阻塞证据': assessment.get('当前阻塞证据', ''),
            '本轮通过对应修改': '',
        })
        if result and state == '通过':
            description = PASS_CHANGES.get(index)
            if description is None and old['补丁状态'] == '跳过':
                description = old['适配测试修改'] + '；本轮相同适配下复测保持通过。'
            if description is None:
                assert old['补丁状态'] == '通过', index
                description = '已有通过项的回归验证；沿用 PR+driver 前提及本轮组合补丁，未增加本用例专属修复。'
            row['本轮通过对应修改'] = description
        elif result:
            row['本轮通过对应修改'] = '本轮未完全通过；实际应用补丁不等于已修复，详见最新阻塞与日志。'
        else:
            row['本轮通过对应修改'] = '未另行复测，沿用此前结果及通过对应修改。'
        rows.append(row)
        if str(index) in required:
            triage.append(row)
    assert len(rows) == 247 and len(triage) == 100
    regressions = [r['序号'] for r in rows if r['补丁状态'] == '通过' and r['累计最新结果'] != '通过']
    assert not regressions, regressions
    write_csv(out / 'results.csv', rows, list(rows[0]))
    write_csv(out / 'remaining-triage.csv', triage, list(triage[0]))
    summary = {
        'total_rows': len(rows), 'reviewed_remaining': len(triage),
        'previous_states': dict(Counter(r['补丁状态'] for r in rows)),
        'latest_states': dict(Counter(r['累计最新结果'] for r in rows)),
        'first_round_rows': len(first), 'second_round_rows': len(second),
        'unique_retested_rows': len(latest),
        'retested_remaining_rows': sum(int(n) in latest for n in required),
        'newly_passed_rows': [r['序号'] for r in triage if r['累计最新结果'] == '通过'],
        'regression_rows': regressions,
        'latest_retest_instances': dict(sum((Counter(r['counts']) for r in latest.values()), Counter())),
        'test_scope': '累计结果合并本轮测试适配/Inductor 候选与此前结果；不是 247 项在单一配置下全量重跑。',
        'shared_prerequisite': '完整 PR #47384 16437be0 + 恢复 Ascend driver/helper；capability=8.0 为测试兼容值。',
    }
    save_json(out / 'progress.json', summary)
    save_json(out / 'validation.json', {
        'review_rows_complete': True, 'original_row_order_preserved': True,
        'all_selected_nodes_completed': True, 'test_hashes_match_manifests': True,
        'driver_hash_verified': True, 'no_regression_in_retested_prior_passes': True,
        'result_csv_sha256': hashlib.sha256((out / 'results.csv').read_bytes()).hexdigest(),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
