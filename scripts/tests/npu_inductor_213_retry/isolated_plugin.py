import retry_gates
from retry_gates import pytest_configure, pytest_collection_modifyitems
import json
import os
from pathlib import Path

REPORTS = {}
COLLECTED = []
COLLECTION_ERRORS = []
OUT = Path(os.environ['RECHECK_JSON'])


def save(exitstatus=None):
    data = {'pid': os.getpid(), 'exitstatus': exitstatus, 'reports': REPORTS,
            'collected': COLLECTED, 'collection_errors': COLLECTION_ERRORS,
            'retry_gate_events': retry_gates.EVENTS}
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(OUT)


def pytest_collection_finish(session):
    COLLECTED.extend(item.nodeid for item in session.items)
    save()


def pytest_collectreport(report):
    if report.failed:
        COLLECTION_ERRORS.append(str(report.longrepr))
        save()


def pytest_runtest_logreport(report):
    rec = REPORTS.setdefault(report.nodeid, {'nodeid': report.nodeid})
    rec[report.when] = {'outcome': report.outcome, 'duration': report.duration,
                        'longrepr': str(report.longrepr) if report.longrepr else '',
                        'wasxfail': getattr(report, 'wasxfail', None)}
    if report.when == 'call' or report.outcome != 'passed':
        rec['outcome'] = report.outcome
    save()


def pytest_sessionfinish(session, exitstatus):
    save(int(exitstatus))
