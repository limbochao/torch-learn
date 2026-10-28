"""Check sparse Meta construction under the invariants enabled by TestCase.setUp.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0; no accelerator is required. This
isolates metadata construction and does not validate the NPU gather kernel.
"""

import json

import torch
from torch._decomp.decompositions import _to_copy
from torch.fx.experimental.proxy_tensor import make_fx


indices = torch.empty((4, 200), dtype=torch.int64, device='meta')
values = torch.empty(200, device='meta')
size = (4, 5, 10, 6)

for enabled in (False, True):
    for explicit_no_checks in (False, True):
        def construct(index, value):
            kwargs = {'check_invariants': False} if explicit_no_checks else {}
            return torch.sparse_coo_tensor(index, value, size, device='meta', **kwargs)

        record = {'global_invariants': enabled, 'meta_check_invariants_false': explicit_no_checks,
                  'torch': torch.__version__}
        try:
            with torch.sparse.check_sparse_tensor_invariants(enabled):
                graph = make_fx(construct, decomposition_table={torch.ops.aten._to_copy.default: _to_copy})(
                    indices, values
                )
                output = graph(indices, values)
            record.update(status='passed', shape=list(output.shape), nnz=output._nnz(), layout=str(output.layout))
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
        print(json.dumps(record), flush=True)
