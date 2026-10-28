"""Probe native build checks with transfer-like CUDA aliases and no device allocation."""

import ast
import __future__
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


def extract(path, name, namespace):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def main(root):
    root = Path(root)
    failures = 0
    for cuda, hip, allocator_enabled in [(None, None, None), ('12.9', None, True),
                                         ('12.9', None, False), (None, '7.0', False)]:
        allocator = Mock(return_value=allocator_enabled)
        torch = SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip),
                                cuda=SimpleNamespace(is_available=lambda: True),
                                _C=SimpleNamespace(_cuda_cudaCachingAllocator_is_enabled=allocator))
        fn = extract(root / 'cudagraph_utils.py', 'check_caching_allocator_for_cudagraphs',
                     {'torch': torch, 'format_default_skip_message': lambda value: value})
        actual = fn()
        expected_rejection = (cuda is not None or hip is not None) and not allocator_enabled
        ok = bool(actual) == expected_rejection and allocator.call_count == int(cuda is not None or hip is not None)
        print('allocator', cuda, hip, allocator_enabled, 'PASS' if ok else 'FAIL',
              'calls', allocator.call_count, 'result', actual)
        failures += not ok
    for cuda in [None, '12.9']:
        torch = SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=None),
                                cuda=SimpleNamespace(is_available=lambda: True))
        active = Mock(return_value=True)
        backend = SimpleNamespace(driver=SimpleNamespace(is_active=active))
        fn = extract(root / 'runtime/triton_helpers.py', '_is_backend_active', {})
        with patch.dict(sys.modules, {'torch': torch}):
            actual = fn('nvidia', backend)
            ascend = fn('ascend', backend)
        ok = actual == (cuda is not None) and ascend is True
        print('driver', cuda, 'PASS' if ok else 'FAIL', 'nvidia', actual, 'ascend', ascend)
        failures += not ok
    return int(failures != 0)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
