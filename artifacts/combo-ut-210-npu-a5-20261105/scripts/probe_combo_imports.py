import ast
import importlib
import json
import os
from pathlib import Path
import combo_validation_plugin
root = Path('/home/l30023782/PTA_210/combo_kernel_ut/Community-Use-Case-Testing/v2.13.0/inductor_npu')
missing = []
for file in ['test_torchinductor.py','test_combo_kernels.py','test_foreach.py','test_unbacked_symints.py','test_provenance_tracing.py']:
    tree = ast.parse((root/file).read_text())
    for node in tree.body:
        if not isinstance(node,ast.ImportFrom) or node.level or not node.module.startswith('torch'):
            continue
        try:
            module = importlib.import_module(node.module)
            names = [n.name for n in node.names if not hasattr(module,n.name)]
            if names: missing.append({'file':file,'module':node.module,'missing':names})
        except Exception as e:
            missing.append({'file':file,'module':node.module,'error':repr(e)})
print(json.dumps(missing, indent=2))
import torch.testing._internal.common_device_type as cd
print('DEVICE_BASES', [(b.__name__,b.device_type) for b in cd.device_type_test_bases])
