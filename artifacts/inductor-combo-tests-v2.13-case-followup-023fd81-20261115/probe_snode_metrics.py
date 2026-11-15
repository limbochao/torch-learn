"""Check that both original SNode cases examine real scheduler nodes."""

import json
import os
from pathlib import Path

from torch._inductor import metrics
from test_snode_runtime import InputDistanceTests

root = Path('<workspace>')
proof = {'TORCH_LOGS': os.environ.get('TORCH_LOGS'), 'methods': []}
for method in ('test_foreach_basic', 'test_foreach_after_extern'):
    case = InputDistanceTests(method)
    case.setUp()
    try:
        getattr(case, method)()
        nodes = [{'kind': type(node).__name__, 'min_input_distance': node.min_input_distance,
                  'max_input_distance': node.max_input_distance, 'numel': str(numel)}
                 for node, numel in metrics.nodes_num_elem]
        assert nodes, f'{method} collected no scheduler nodes'
        proof['methods'].append({'name': method, 'nonempty': True, 'nodes': nodes})
    finally:
        case.tearDown()
        case.doCleanups()
(root / 'snode-metrics-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
