"""Independently reconcile JUnit cases, shard reports and the full collected node set."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


root = Path(sys.argv[1]).resolve()
collection = json.loads((root / 'full-collection/collection.json').read_text())
expected = {node for job in collection for node in job['nodes']}
assert sum(len(job['nodes']) for job in collection) == len(expected)
merged = json.loads((root / 'jobs.json').read_text())
merged_reports = {node: report for job in merged for node, report in job['reports'].items()}
assert len(merged_reports) == sum(len(job['reports']) for job in merged)
assert set(merged_reports) == expected
observed = {}
xml_files = []
case_records = 0
same_node_records = 0
for shard in json.loads((root / 'shards.json').read_text()):
    directory = root / 'results' / shard['name']
    jobs = json.loads((directory / 'jobs.json').read_text())
    for job in jobs:
        xml_path = directory / f"{job['index']:02d}-{job['name']}.xml"
        tree = ET.parse(xml_path)
        lookup = {
            (Path(node.split('::')[0]).stem, *node.split('::')[1:]): node
            for node in job['nodes']
        }
        current = {}
        for case in tree.getroot().iter('testcase'):
            case_records += 1
            module, cls = case.attrib['classname'].rsplit('.', 1)
            key = module.rsplit('.', 1)[-1], cls, case.attrib['name']
            assert key in lookup, (xml_path, case.attrib)
            node = lookup[key]
            tags = {child.tag for child in case}
            assert 'error' not in tags, (node, ET.tostring(case, encoding='unicode'))
            status = 'FAIL' if 'failure' in tags else 'SKIP' if 'skipped' in tags else 'PASS'
            if node in current:
                same_node_records += 1
                if current[node] == 'FAIL' or status == 'FAIL':
                    status = 'FAIL'
                elif current[node] == 'SKIP' or status == 'SKIP':
                    status = 'SKIP'
            current[node] = status
        assert set(current) == set(job['nodes']), (xml_path, 'Missing JUnit case')
        assert set(job['reports']) == set(current)
        assert job['returncode'] in (0, 1)
        for node, status in current.items():
            assert node not in observed, ('Duplicate shard instance', node)
            assert status == job['reports'][node]['status'] == merged_reports[node]['status'], node
            observed[node] = status
        xml_files.append({'path': str(xml_path.relative_to(root)),
                          'sha256': hashlib.sha256(xml_path.read_bytes()).hexdigest(),
                          'instances': len(current)})
assert set(observed) == expected
state = json.loads((root / 'status.json').read_text())
counts = dict(Counter(observed.values()))
assert counts == state['instance_results']
methods = Counter(job['status'] for job in merged)
assert dict(methods) == state['method_results']
overlap = {(row['file'], row['class'], row['name'])
           for row in json.loads((root / 'overlap60-jobs.json').read_text())}
cohorts = {}
for label, included in [('overlap', True), ('outside', False)]:
    rows = [job for job in merged if ((job['file'], job['class'], job['name']) in overlap) == included]
    cohorts[label] = {
        'methods': len(rows), 'method_results': dict(Counter(job['status'] for job in rows)),
        'instances': sum(len(job['nodes']) for job in rows),
        'instance_results': dict(Counter(r['status'] for job in rows for r in job['reports'].values())),
    }
receipt = {'verified': True, 'methods': len(merged), 'instances': len(observed),
           'instance_results': counts, 'method_results': dict(methods), 'cohorts': cohorts,
           'junit_files': len(xml_files), 'junit_testcase_records': case_records,
           'additional_same_node_subtest_records': same_node_records,
           'missing_instances': [], 'duplicate_shard_instances': [], 'execution_errors': [],
           'files': xml_files}
(root / 'junit-cross-check.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, ensure_ascii=False, indent=2))
