"""Run independent numerical cases and keep each exit status, including failed cases."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
out = Path(os.environ.get('REGRESSION_DIR', root / 'regressions'))
out.mkdir(exist_ok=True)
records = []
for group in ('mask', 'axes', 'negative'):
    flags = ('none', 'dump', 'check') if group == 'negative' else ('none',)
    for dynamic in (False, True):
        for flag in flags:
            key = f'{group}_{"dynamic" if dynamic else "static"}_{flag}'
            env = os.environ.copy()
            env.pop('INDUCTOR_ASCEND_DUMP_FX_GRAPH', None)
            env.pop('INDUCTOR_ASCEND_CHECK_ACCURACY', None)
            if flag != 'none':
                name = 'DUMP_FX_GRAPH' if flag == 'dump' else 'CHECK_ACCURACY'
                env[f'INDUCTOR_ASCEND_{name}'] = '1'
            command = [sys.executable, str(root / 'validate_codegen.py'), '--group', group,
                       '--output', str(out / f'{key}.json')]
            if dynamic:
                command.append('--dynamic')
            with (out / f'{key}.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(case=key, exit_code=result.returncode, command=command))
            (out / 'status.json').write_text(json.dumps(records, indent=2))
            print(key, result.returncode, flush=True)
sys.exit(any(record['exit_code'] != 0 for record in records))
