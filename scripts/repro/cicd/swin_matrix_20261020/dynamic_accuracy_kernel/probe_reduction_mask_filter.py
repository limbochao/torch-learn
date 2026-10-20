"""Exercise the installed mask filter on a synthetic persistent reduction axis."""
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

from torch._inductor.virtualized import V
from torch_npu._inductor.codegen import triton as codegen

method = codegen.NPUIndexTritonKernel.filter_masks
source = textwrap.dedent(inspect.getsource(method))
needle = '            or is_persistent_reduction_axis\n'
assert source.count(needle) == 1
scope = dict(method.__globals__)
exec(source.replace(needle, ''), scope)
node = SimpleNamespace(
    name='r3', is_vectorized_split=False, is_reduction=True,
    is_tiling_axis=True, is_no_loop_axis=False,
)
kernel = SimpleNamespace(
    sorted_axis=[node], persistent_reduction=True, current_subblock_axis=set(),
    is_unified_simt_kernel=lambda: False,
)
results = {}
with V.set_kernel_handler(kernel):
    for name, function in [('installed', method), ('diagnostic_remove_skip', scope['filter_masks'])]:
        masks = {'r3_mask'}
        function(kernel, masks)
        results[name] = sorted(masks)
assert results == {'installed': [], 'diagnostic_remove_skip': ['r3_mask']}
print(json.dumps({
    'scenario': 'static extent 49, padded persistent reduction block 64',
    'input_masks': ['r3_mask'], 'results': results,
    'source_sha256': hashlib.sha256(Path(codegen.__file__).read_bytes()).hexdigest(),
    'note': 'Synthetic object only; no installed file was changed.',
}), flush=True)
