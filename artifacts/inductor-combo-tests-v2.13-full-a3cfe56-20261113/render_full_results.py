"""Render the verified full run into the stable CSV and the maintained 2.13 note."""

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import textwrap
from zoneinfo import ZoneInfo

from priority_candidates import CANDIDATES, HARDWARE


REPO = Path('/home/qc/Repos/Ascend/torch-learn')
ROOT = Path(__file__).resolve().parent
ART = REPO / 'artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113'
TABLE = REPO / 'docs/notes/compiler/codegen/inductor-combo-tests-native-v2.13-sync-64dc85c.csv'
DOC = REPO / 'docs/notes/compiler/codegen/inductor-combo-tests-v2.13.md'
LIST210 = REPO / 'scripts/tests/combo_kernel_210/combo_native_210_passed_nodeids.txt'
LIST213 = REPO / 'scripts/tests/combo_kernel_210/combo_native_213_passed_nodeids.txt'
SHA210 = 'ae33f60f8586fd74d3f9f37159b69c4151b9cde8d59e82d896fc2cffd7a4b509'


def load(path):
    return json.loads(path.read_text())


def counts(job):
    return dict(Counter(report['status'] for report in job['reports'].values()))


def key(job):
    return f"{job['file']}::{job['class']}::{job['name']}"


def paragraph(text):
    return textwrap.fill(text, width=100, break_long_words=False, break_on_hyphens=False) + '\n\n'


def brief(detail):
    lines = detail.splitlines()
    important = [line.strip() for line in lines if re.match(
        r'^(?:E\s+)?(?:AssertionError|RuntimeError|TypeError|AttributeError|KeyError|NotImplementedError)',
        line.strip())]
    return (important[0] if important else next((line.strip() for line in lines if line.strip()), ''))[:240]


def main():
    state, audit, environment = (load(ART / name) for name in
                                  ('status.json', 'junit-cross-check.json', 'environment.json'))
    assert state['stage'] == 'complete' and state['full_coverage'] and audit['verified']
    assert not state['sampling'] and not state['reusing_old_results']
    assert state['test_commit'] == 'a3cfe564cd60cb83f4c81bab5dcd388305992ff2'
    assert hashlib.sha256(LIST210.read_bytes()).hexdigest() == SHA210
    jobs, layout = load(ART / 'jobs.json'), load(ART / 'table_rows.json')
    overlap = {(row['file'], row['class'], row['name']) for row in load(ART / 'overlap60-jobs.json')}
    by_key = {(job['file'], job['class'], job['name']): job for job in jobs}
    assert len(by_key) == 105 and sum(len(job['nodes']) for job in jobs) == 464
    override_file = ROOT / 'manual-analysis-overrides.json'
    overrides = load(override_file) if override_file.exists() else {}
    records, hardware, overlap_nonpass = [], [], []
    for job in jobs:
        if job['status'] == 'PASS':
            continue
        record = {'method': key(job), 'file': job['file'], 'class': job['class'], 'name': job['name'],
                  'status': job['status'], 'instance_results': counts(job), 'nodes': job['nodes'],
                  'nonpass_nodes': [node for node, report in job['reports'].items()
                                    if report['status'] != 'PASS'],
                  'observed': brief(next(report['detail'] for report in job['reports'].values()
                                         if report['status'] != 'PASS'))}
        is_overlap = (job['file'], job['class'], job['name']) in overlap
        if job['name'] in HARDWARE:
            assert job['status'] == 'SKIP', ('Hardware exclusion did not skip', key(job))
            record.update(reason=HARDWARE[job['name']], keep_skip=True)
            hardware.append(record)
        elif not is_overlap:
            record.update(CANDIDATES[job['name']])
            record.update(overrides.get(key(job), overrides.get(job['name'], {})))
            records.append(record)
        else:
            record.update(overrides.get(key(job), overrides.get(job['name'], {})))
            record.setdefault('current_analysis', '当前原断言未通过：' + record['observed'])
            overlap_nonpass.append(record)
    records.sort(key=lambda record: (record['priority'], record['order'], record['method']))
    for index, record in enumerate(records, 1):
        record['sequence'] = index
        record['anchor'] = f'repair-{index:02d}'
    nonpass_outside = {key(job) for job in jobs if job['status'] != 'PASS'
                       and (job['file'], job['class'], job['name']) not in overlap}
    assert {r['method'] for r in records + hardware} == nonpass_outside
    assert not {r['method'] for r in records} & {r['method'] for r in hardware}
    current = {r['method']: r for r in records + hardware + overlap_nonpass}
    analyses = {}
    for method, record in current.items():
        if record.get('keep_skip'):
            text = '硬件／平台协议差异，保留 SKIP：' + record['reason']
        elif 'current_analysis' in record:
            text = record['current_analysis']
        else:
            prefix = f"{record['priority']}，顺序 {record['sequence']:02d}；"
            status = '当前测试提前 SKIP，功能目标尚未执行。' if record['status'] == 'SKIP' else (
                '本轮首个失败点：' + record.get('observed_human', record['observed']) + '。')
            text = (prefix + status + '目标：' + record['purpose'] + '缺口／适配：'
                    + record['source_assessment'] + '先修：' + record['fix_entry']
                    + '验收：' + record['acceptance'])
        analyses[method] = text
    (ART / 'repair-priority.json').write_text(json.dumps({
        'test_commit': state['test_commit'], 'torch_npu_git': state['torch_npu_git'],
        'cohort': 'outside the established 60-method 2.10 overlap',
        'nonpass_outside_methods': len(nonpass_outside),
        'ranked_methods': len(records), 'hardware_skip_methods': len(hardware),
        'priority_counts': dict(Counter(r['priority'] for r in records)),
        'complete_coverage': True, 'records': records, 'hardware_exclusions': hardware,
    }, ensure_ascii=False, indent=2) + '\n')
    (ART / 'current-analysis.json').write_text(json.dumps(analyses, ensure_ascii=False, indent=2) + '\n')
    with TABLE.open(encoding='utf-8-sig', newline='') as stream:
        old_rows = list(csv.DictReader(stream))
    assert len(layout) == len(old_rows) == 116
    columns = ['测试文件', '函数名', '最新结果', '实例结果', '当前分析']
    rendered = []
    for mapping, old_row in zip(layout, old_rows):
        assert (mapping['file'], mapping['name']) == (old_row['测试文件'], old_row['函数名'])
        source = mapping.get('source_method', mapping['name'])
        job = by_key[mapping['file'], mapping['class'], source]
        instances = '; '.join(f'{status} {number}' for status, number in sorted(counts(job).items()))
        result = job['status']
        if mapping.get('nested'):
            result = '随外层验证/' + result
            instances = f'随 {source} 验证'
        analysis = '—' if job['status'] == 'PASS' else analyses[key(job)]
        rendered.append([mapping['file'], mapping['name'], result, instances, analysis])
    with TABLE.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rendered)

    overlap_pass = sorted(node for job in jobs if (job['file'], job['class'], job['name']) in overlap
                          for node, report in job['reports'].items() if report['status'] == 'PASS')
    LIST213.write_text('\n'.join(overlap_pass) + '\n')
    (ART / 'overlap-passed-nodeids.txt').write_text(LIST213.read_text())
    source_url = '../../../../artifacts/' + ART.name
    start = datetime.fromtimestamp(state['started'], ZoneInfo('Asia/Shanghai'))
    end = datetime.fromtimestamp(state['finished'], ZoneInfo('Asia/Shanghai'))
    document = ['---\npublished: false\n---\n\n# 2.13 Combo/foreach 最新结果与问题分析\n\n',
                '<a id="validation213"></a>\n\n<!-- combo-section:validation213:start -->\n',
                '## 最新执行结果\n\n']
    document.append(paragraph(
        f"{end:%Y-%m-%d} 在 NPU_A5 Docker 的 `pta_213_case` 环境，按固定 CSV 全量重新收集和执行 "
        f"105 个源方法、464 个参数实例。CSV 的 116 行包含 11 个随外层方法验证的嵌套函数。"
        f"实例结果为 **{state['instance_results'].get('PASS', 0)} PASS、"
        f"{state['instance_results'].get('FAIL', 0)} FAIL、"
        f"{state['instance_results'].get('SKIP', 0)} SKIP**；没有抽样或复用旧执行结果。"))
    document.append('| 范围 | PASS | FAIL | SKIP |\n| --- | ---: | ---: | ---: |\n')
    for label in ('overlap', 'outside'):
        cohort = audit['cohorts'][label]
        name = '2.10 重合范围' if label == 'overlap' else '重合范围外'
        for unit, result_key in (('方法', 'method_results'), ('实例', 'instance_results')):
            total = cohort['methods' if unit == '方法' else 'instances']
            c = cohort[result_key]
            document.append(f"| {name} {total} 个{unit} | {c.get('PASS', 0)} | "
                            f"{c.get('FAIL', 0)} | {c.get('SKIP', 0)} |\n")
    for total, unit, result_key in ((105, '方法', 'method_results'), (464, '实例', 'instance_results')):
        c = state[result_key]
        document.append(f"| 全量 {total} 个{unit} | {c.get('PASS', 0)} | "
                        f"{c.get('FAIL', 0)} | {c.get('SKIP', 0)} |\n")
    document.append('\n### 验证版本与范围\n\n')
    for text in (
        f"torch_npu `{environment['torch_npu']}`，提交 `{environment['torch_npu_git']}`。",
        f"PyTorch `{environment['torch']}`，提交 `{environment['torch_git']}`；Triton `{environment['triton']}`。",
        '物理卡 1、2、3，每卡两个进程，Inductor/Triton 缓存和日志独立；没有生产代码 overlay。',
        f"测试分支 `fix/combo-213-210-overlap`，提交 `{state['test_commit']}`，main 基线 "
        '`91435bdf732e772c3240329797a9a29166ce533a`。',
        f'执行时间：北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。',
    ):
        document.append(paragraph(text))
    document.append(paragraph(
        'Community 分支只包含与既定 60 个重合方法相关的适配，修改 3 个文件、30 个方法；'
        '范围外测试修改未迁入该分支。仅本轮验证副本加上 TMA 和三个 PDL 方法的硬件 SKIP，'
        '方法 body 和断言保持原内容。foreach 按已确认口径保留数值、mutation、布局和梯度检查，'
        'NPU 不要求 CUDA 单 kernel／FOREACH 专属结构。'))
    document.append(paragraph(
        '执行前后 301 个测试文件、588 个 Torch/NPU Inductor Python 文件哈希一致；'
        '其中 194 个 NPU Inductor Python 文件与实际包提交逐项一致。'
        '三卡基础计算均通过。全部分片 JUnit 已独立重新解析，464 个实例无遗漏或跨分片重复。'))
    document.append(paragraph(
        f"[2.13 PASS 清单][pass213]保留 2.10 重合范围口径，当前为 {len(overlap_pass)} 个实际 PASS 实例，"
        '在 2.13 NPU 测试目录执行；[全范围 PASS][all-pass]另存在本轮证据目录。'
        '2.10 原生 397 条清单保持不变。既定 60 方法适配范围与 2.10 原生 54 方法清单的口径不同，'
        '本轮用 [60 方法范围清单][overlap]精确划分。'))
    document.append(paragraph(
        '`test_combo_autotune_grouping_by_metadata_fingerprint` 直接调用社区 helper 和模拟 metadata；'
        '其 PASS 证明该 helper 的分组规则，本轮未验收真实 NPU 成员 metadata 到该分组的连接。'))
    document.append(paragraph(
        '[当前 CSV][csv]保持原文件名、116 行顺序及五列，所有行均来自本轮结果。'
        '[方法与逐实例报告][jobs]、[完整环境和分片证据][evidence]、'
        '[JUnit 交叉核对][junit]及[Community 范围审计][scope]可追溯当前结论。'))
    document.append('<!-- combo-section:validation213:end -->\n\n<a id="analysis213"></a>\n\n'
                    '<!-- combo-section:analysis213:start -->\n## 当前问题与修复顺序\n\n')
    document.append(paragraph(
        f"重合范围外共有 {len(nonpass_outside)} 个非 PASS 方法：{len(records)} 个非硬件 FAIL／SKIP "
        f"全部进入下表，{len(hardware)} 个硬件／平台协议方法保留 SKIP。"
        f"P0 {sum(r['priority'] == 'P0' for r in records)} 项先定位本轮多轴归约精度失败；"
        f"P1 {sum(r['priority'] == 'P1' for r in records)} 项恢复有效测试入口、eager 布局、metadata、统计和追踪；"
        f"P2 {sum(r['priority'] == 'P2' for r in records)} 项处理独立分块、成员调优、融合目标和 FTZ 数学路径；"
        f"P3 {sum(r['priority'] == 'P3' for r in records)} 项补齐峰值门控的有效验收。"))
    document.append(paragraph(
        '以下是修复顺序与验收要求，本轮未实现生产修复。原测试提前 SKIP 是实际执行状态，'
        '不代表功能通过，也不作为后续规避目标的理由；非硬件旧 skip 应移除，未完成的目标保持 FAIL。'
        '当前 PASS 不保留旧失败分析。源码分析固定在实际安装的 NPU/PyTorch 提交，'
        '不把其他开发分支或最新 master 当作本轮运行代码。'))
    document.append('### metadata 和统计分别缺什么\n\n')
    document.append(paragraph(
        '`optimize_mem` 两个方法检查同一个字段的两个图阶段；'
        '七个统计方法检查多种计数、布局信息和成本出口，不能写成“缺七种统计”。'
        '`combo_subkernel_meta` 已有真实成员轴、尺寸和模式，'
        '它不等于社区承载下表统计的 `inductor_meta_i` 字典。'))
    fields = [
        ('`optimize_mem`', '推理／反向 True，训练前向 False，约束 autotune 输入备份和恢复。'),
        ('`reduction_hint_i`', '逐成员 INNER/OUTER 访问方向，用于候选分类和观察。'),
        ('`num_load`', '去除 CSE 缓存复用后实际 load 数，描述访存工作量。'),
        ('`num_store`', '保留的实际写回数；NPU store 路径目前未统一递增此计数。'),
        ('`num_reduction`', '归约 primitive 数，用于候选与工作量统计。'),
        ('`autotune_hints`', '如 ONE_ELEMENT_PER_THREAD 的候选提示，须来自真实成员。'),
        ('`atomic_add_found`', '是否存在原子加，约束候选及写回行为。'),
        ('`has_loadstore_with_contiguous_rdim`', '实际归约维是否连续，参与归约候选过滤；不是确定性支持开关。'),
        ('`no_x_dim`', '成员是否缺独立 X 维；需按 NPU 真实 split 轴和成员布局转译。'),
        ('`tiling_scores`', '各轴 tiling 的评价信息；需与 native 候选语义对应。'),
        ('`kernel_num_gb`', '实际访存字节的 GB 估算，供成本和带宽报告使用。'),
        ('`kernel_flop`', '按统一规则估算计算量，Combo 聚合成员值。'),
        ('benchmark `num_gb`', '独立 benchmark 带宽分子；当前 NPU Combo 模板收到硬编码 0。'),
        ('`add_persistent_rblock`', '成员增加大 R 候选的资格；字段缺失不证明 NPU 没有 persistent 归约。'),
        ('`disable_ftz`', '控制非正规数保留，需要编译器与数学路径消费，不能只导出文字。'),
    ]
    document.append('| 字段／出口 | 具体作用 |\n| --- | --- |\n')
    document.extend(f'| {field} | {role} |\n' for field, role in fields)
    document.append('\n')
    document.append(paragraph(
        '计数与布局字段先由真实 load/store/reduction 生成，再由 standalone 与 Combo 统一导出；'
        '访存／FLOP 先形成有效的 standalone 估算基准，再聚合 Combo；'
        '成员 fingerprint 分组依赖完整 metadata。'
        '成员独立 block ABI／dispatch 接通后，才有独立默认候选、顺序调优和最大成员优先的 coordesc。'
        'NPU 已有消费真实成员轴约束的共享搜索，缺这些导出字段不证明整个归约／调优能力缺失。'
        '共享候选不能替代独立成员目标，补 0、空字典或日志字符串也不算完成。'))
    document.append(paragraph(
        '有效 store 计数的缺口位于 Native Index 的 `CSEProxy.store/store_reduction`；'
        'no-linear 普通 kernel 继承社区 CSE，已有的计数逻辑与 NPU metadata 出口需分别核对。'
        '`kernel_num_gb` 是逻辑读写量的 GB 估算，不是硬件 profiler 测得的传输量。'))
    document.append(paragraph(
        '当前安装的 Triton Ascend 使用 `NPUOptions`，其 [真实字段列表][triton-options] 未声明 '
        '`disable_ftz`；[原始编译器源码][triton-compiler] 已保存。'
        '这项验收还需核对通用编译选项及下层数学路径的实际消费，不能从 Python 字段缺失直接推定硬件能力。'))
    if any(record['priority'] == 'P0' for record in records):
        document.append(paragraph(
            '本轮多轴归约的 P0 精度失败走 no-linear 普通 kernel；'
            '其 [原始生成 wrapper][accuracy-wrapper]和[按执行时间／输入布局匹配的来源][accuracy-source]'
            '已保存。该路径继承社区多轴 reduction DSL，再由 NPU heuristic／编译器执行；'
            '不能按 Native Combo 的统计出口缺失来定界这项精度问题。'))
    document.append('### 范围外逐用例修复顺序\n\n'
                    '| 顺序 | 优先级 | 用例 | 当前结果 | 具体待修内容 |\n'
                    '| ---: | --- | --- | --- | --- |\n')
    for record in records:
        document.append(f"| {record['sequence']:02d} | {record['priority']} | "
                        f"[{record['name']}](#{record['anchor']}) | {record['status']} | "
                        f"{record.get('summary', record['kind'])} |\n")
    document.append('\n')
    for record in records:
        document.append(f"<a id=\"{record['anchor']}\"></a>\n\n"
                        f"#### {record['sequence']:02d} {record['priority']} `{record['name']}`\n\n")
        document.append(paragraph(f"位置：`{record['file']}::{record['class']}`；本轮 "
                                  + '; '.join(f'{s} {n}' for s, n in record['instance_results'].items()) + '。'))
        document.append(paragraph('功能目标：' + record['purpose']))
        document.append(paragraph('当前失败／跳过：' + record.get('observed_human', record['observed'])))
        document.append(paragraph('具体缺口／适配：' + record['source_assessment']))
        document.append(paragraph('先修入口：' + record['fix_entry'] + ' 验收：' + record['acceptance']))
        document.append(paragraph('证据：' + '、'.join(f'[{ref}][{ref}]' for ref in record['evidence'])
                                  + '；[原方法与断言][test-index]；[本轮报告][jobs]。'))
    document.append('### 保留 SKIP 的硬件／平台协议\n\n')
    for record in hardware:
        document.append(paragraph(f"`{record['name']}`：{record['reason']}"
                                  + ' 本轮 SKIP，未计为 PASS，也不进入 NPU 同协议实现队列。'))
    document.append(paragraph(
        'XPU-only 的动态归约方法没有 XPU 专属 intrinsic，故已进入修复顺序；'
        '它的设备门控不能与 TMA／PDL／HIP 协议一并归为硬件差异。'))
    if overlap_nonpass:
        document.append('### 2.10 重合范围的当前非 PASS\n\n')
        document.append(paragraph(
            '这些方法不进入上面的“范围外”排序，但仍保留本轮实际结果；'
            '非硬件 SKIP 不作为已完成支持。'))
        for record in overlap_nonpass:
            document.append(paragraph(f"`{record['file']}::{record['name']}`（{record['status']}）："
                                      + analyses[record['method']]))
        document.append(paragraph('以上原方法与断言见 [测试源码索引][test-index]，'
                                  '实际失败阶段见 [逐实例报告][jobs]，'
                                  'NPU 成员分区和共享 block ABI 见 [E1][E1]。'))
    document.append('### 水平融合、垂直融合与 foreach 的关系\n\n')
    document.append(paragraph(
        '水平融合把没有 producer/consumer 依赖的独立成员打包进一个 Combo；'
        '垂直融合沿依赖链把生产与消费算子融进同一成员。'
        '`torch._foreach_*` 是对 tensor 列表逐项计算的前端 API；编译器的 FOREACH 分组融合路径'
        '把各项形成成员、吸收生产者／消费者链，再将独立成员水平打包。'
        '当前 NPU 未注册该 FOREACH 后端能力，普通算子的垂直融合入口仍可使用。'))
    document.append('```text\n成员 0：add(a0, a0) → foreach 第 0 项 → mul(d0, a0)\n'
                    '成员 1：mul(a1, a1) → foreach 第 1 项 → mul(d1, a1)\n```\n\n')
    document.append(paragraph(
        '`test_non_foreach_consumer_producer_list` 体现这两条链；'
        '`test_scheduler_fusion_list` 体现 foreach→foreach 的成员配对融合，'
        '`test_single_list` 体现列表成员水平打包。社区原目标是单 kernel；'
        '本轮 foreach 按既定功能口径执行，所以数值 PASS 不证明完整 FOREACH 分组融合完成。'))
    document.append(paragraph(
        '社区 2.13 支持上述水平／垂直组合，基础机制在 2.10 已存在，见版本源码 [E15][E15]。'
        'NPU 的普通 can_fuse_vertical 入口已有，backend features 当前未注册 FOREACH；'
        '应将完整列表分组与依赖链吸收作为 NPU 待适配功能，不能从 FOREACH 入口未启用推断'
        '普通垂直融合缺失，也不能把缺 mixed-order 联合归约或成员独立调优笼统写成融合策略差异。'))
    document.append('<!-- combo-section:analysis213:end -->\n\n')
    links = {
        'pass213': '../../../../scripts/tests/combo_kernel_210/combo_native_213_passed_nodeids.txt',
        'csv': TABLE.name, 'jobs': source_url + '/jobs.json', 'evidence': source_url + '/status.json',
        'junit': source_url + '/junit-cross-check.json', 'scope': source_url + '/community-branch-scope.json',
        'overlap': source_url + '/overlap60-jobs.json', 'all-pass': source_url + '/passed-nodeids.txt',
        'test-index': source_url + '/source/test-method-index.json',
        'accuracy-wrapper': source_url + '/generated/r0-r1-accuracy-1.py',
        'accuracy-source': source_url + '/generated/r0-r1-accuracy-source.json',
        'triton-options': source_url + '/installed-triton-source/options-ast-check.json',
        'triton-compiler': source_url + '/installed-triton-source/compiler.py',
    }
    for fragment in load(ART / 'source/source-index.json')['fragments']:
        links[fragment['id']] = source_url + '/source/' + fragment['path']
    for label, target in links.items():
        document.append(f'[{label}]:\n  {target}\n')
    DOC.write_text(''.join(document))
    assert hashlib.sha256(LIST210.read_bytes()).hexdigest() == SHA210
    receipt = {'csv_rows': len(rendered), 'nested_rows': sum(bool(r.get('nested')) for r in layout),
               'all_pass_analyses_cleared': all(r[-1] == '—' for r in rendered if r[2].endswith('PASS')),
               'list213_scope': 'established 2.10 overlap', 'list213_nodes': len(overlap_pass),
               'list210_unchanged': True, 'ranked_outside_methods': len(records),
               'hardware_skip_methods': len(hardware), 'current_source': ART.name}
    (ART / 'publication-audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
