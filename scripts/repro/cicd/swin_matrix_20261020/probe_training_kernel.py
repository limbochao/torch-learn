"""Compare fixed tiles without changing the generated kernel body or compiler options."""

import importlib.util
import json
import re
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent / 'training_accuracy_kernel/compile_repro.py'
    spec = importlib.util.spec_from_file_location('swin_repro', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for tile in [512, 16, 8, 4, 2, 1]:
        module._STANDALONE_CONSTANTS.update(X0BLOCK_SUB=tile)
        item = {'X0BLOCK_SUB': tile}
        try:
            module._standalone_repro_compile()
            item['compile'] = 'PASS'
        except Exception as exc:
            item['compile'] = 'FAIL'
            item['errors'] = re.findall(r'ub overflow[^!]*!|LLVM ERROR:[^\n]*', str(exc))[:3]
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
