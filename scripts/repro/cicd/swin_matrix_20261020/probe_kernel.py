"""Compare fixed tiles without changing the generated kernel body or compiler options."""

import importlib.util
import json
import re
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'static_inference_kernel/compile_repro.py'
    spec = importlib.util.spec_from_file_location('swin_repro', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for y1, y3 in [(2, 8), (1, 4), (1, 2), (1, 1)]:
        module._STANDALONE_CONSTANTS.update(Y1BLOCK_SUB=y1, Y3BLOCK_SUB=y3)
        item = {'Y1BLOCK_SUB': y1, 'Y3BLOCK_SUB': y3}
        try:
            module._standalone_repro_compile()
            item['compile'] = 'PASS'
        except Exception as exc:
            item['compile'] = 'FAIL'
            item['errors'] = re.findall(r'ub overflow[^!]*!|LLVM ERROR:[^\n]*', str(exc))[:3]
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
