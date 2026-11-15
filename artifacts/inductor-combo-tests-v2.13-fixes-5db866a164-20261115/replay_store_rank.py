import importlib.util
import json
from pathlib import Path
import torch
import torch_npu
import triton

root = Path.cwd()
source = (root / 'failed-combo-store.py').read_text()
body = source[source.index('@triton.jit'):].replace('def Placeholder.KERNEL_NAME(', 'def combo_store(')
header = '''import triton
import triton.language as tl
from torch_npu._inductor.runtime import triton_helpers
'''
torch.manual_seed(23)
a, b = [torch.randn(32, 64, 128, device='npu') for _ in range(2)]
expected = [x.sum((0, 2)).cpu() for x in (a, b)]
results = []
for variant in ['original', 'normalized']:
    code = body
    if variant == 'normalized':
        code = code.replace(' + (x0 )', ' + (x0 + tl.arange(0,1) )')
        assert code != body
    path = root / ('replay_' + variant + '.py')
    path.write_text(header + code)
    spec = importlib.util.spec_from_file_location('replay_' + variant, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {'variant': variant}
    outputs = [torch.empty(64, device='npu') for _ in range(2)]
    try:
        module.combo_store[(2 * triton.cdiv(64, 7),)](a, b, *outputs, 7, compile_mode='simd_simt_template')
        torch.npu.synchronize()
        for target, actual in zip(expected, outputs):
            torch.testing.assert_close(target, actual.cpu(), atol=1e-4, rtol=1e-4)
        row['status'] = 'PASS'
    except Exception as error:
        row['status'] = 'FAIL'
        row['error'] = str(error)
    results.append(row)
(root / 'store-rank-replay.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results))
assert results[0]['status'] == 'FAIL' and 'block' in results[0]['error']
assert results[1]['status'] == 'PASS'
