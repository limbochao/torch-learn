"""Bridge the kernel tracer's unittest selector normalization to the pytest recorder."""

import os
from pathlib import Path
import runpy
import sys

output = sys.argv[1]
sys.argv = [sys.argv[0], '--case-names', '--precompile-workers', os.environ.get('TORCHNPU_PRECOMPILE_THREADS', '2'), '--output', output,
            'test_max_autotune.py::test_deffered_layout_constraint_mm_cat_mm']
runpy.run_path(str(Path(__file__).with_name('boundary_pytest.py')), run_name='__main__')
