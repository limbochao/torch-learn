"""Check device rejection before CUDA allocator access without allocating a device tensor."""

import ast
import __future__
from pathlib import Path
import sys
from unittest.mock import Mock


def check_source(path):
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'check_lowering_disable_cudagraph')
    namespace = {}
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    fn = namespace[node.name]
    failures = 0
    for device in ['npu', 'cpu', 'cuda', 'mixed']:
        reason = None if device == 'cuda' else f'unsupported {device}'
        namespace['check_multiple_devices_or_any_cpu_nodes'] = Mock(return_value=reason)
        allocator = namespace['check_caching_allocator_for_cudagraphs'] = Mock(return_value=None)
        if reason:
            allocator.side_effect = AssertionError('CUDA allocator was accessed for a rejected graph')
        try:
            result = fn({device: object()})
        except AssertionError as error:
            print(device, 'FAIL', str(error))
            failures += 1
            continue
        assert result == reason
        assert allocator.call_count == int(reason is None)
        print(device, 'PASS', 'allocator_calls', allocator.call_count)
    for allocator_reason in [None, 'uncached CUDA allocator']:
        namespace['check_multiple_devices_or_any_cpu_nodes'] = Mock(return_value=None)
        namespace['check_caching_allocator_for_cudagraphs'] = Mock(return_value=allocator_reason)
        assert fn({'cuda': object()}) == allocator_reason
    print('CUDA allocator rejection semantics preserved')
    return int(failures != 0)


if __name__ == '__main__':
    sys.exit(check_source(sys.argv[1]))
