"""Preserve run_and_get_code outputs while executing the original pytest assertions."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path.cwd()))
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from torch._inductor import utils
    from run_batch import main as run_batch

    original = utils.run_and_get_code
    records = []

    def capture(*a, **kw):
        result = original(*a, **kw)
        names = []
        for index, source in enumerate(result[1]):
            name = f'call-{len(records) + 1}-source-{index + 1}.py'
            (args.artifact_dir / name).write_text(source)
            names.append(name)
        records.append({'files': names})
        (args.artifact_dir / 'index.json').write_text(json.dumps(records, indent=2) + '\n')
        return result

    sys.argv = [sys.argv[0], *remaining]
    with patch.object(utils, 'run_and_get_code', capture):
        return run_batch()


if __name__ == '__main__':
    sys.exit(main())
