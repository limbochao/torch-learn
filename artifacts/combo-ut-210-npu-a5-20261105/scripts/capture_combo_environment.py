import json
import os
import platform
import sys
from pathlib import Path
import torch
import torch_npu
import triton
root = Path(__file__).resolve().parent
info = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'torch': torch.__version__, 'torch_path': torch.__file__, 'torch_git': torch.version.git_version,
        'torch_npu': torch_npu.__version__, 'torch_npu_path': torch_npu.__file__,
        'torch_npu_git': torch_npu.version.git_version, 'triton': triton.__version__,
        'working_directory': str(root), 'allowed_physical_devices': [1,2,3],
        'cann_home': os.environ.get('ASCEND_HOME_PATH'),
        'case_source_commit': 'be6049a0059677ca408da1360653b694b6d1afb5',
        'case_source_path': 'Community-Use-Case-Testing/v2.13.0/inductor_npu',
        'source_transfer': 'local git archive after remote clone timed out',
        'test_scope': '54 existing + 7 supported legacy branches + 1 existing behavior',
        'selected_methods': 62, 'validated_parameterized_instances': 409, 'preflight_prefix_collisions_removed': 21}
(root/'evidence/environment.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
