"""Create an isolated Python package overlay from exact candidate source files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    files = list(args.candidate.rglob('*.py'))
    assert files, 'No candidate source files'
    directories = set()
    for path in files:
        relative = path.relative_to(args.candidate)
        assert (args.site / relative).is_file(), relative
        directories.update(relative.parents)
    args.output.mkdir(parents=True)
    for directory in sorted(directories, key=lambda p: len(p.parts)):
        if directory == Path('.'):
            continue
        target = args.output / directory
        target.mkdir(exist_ok=True)
        for path in (args.site / directory).iterdir():
            relative = path.relative_to(args.site)
            if relative in directories or (args.candidate / relative).is_file():
                continue
            (args.output / relative).symlink_to(path, target_is_directory=path.is_dir())
    records = []
    for path in files:
        relative = path.relative_to(args.candidate)
        compile(path.read_text(), str(relative), 'exec')
        shutil.copy2(path, args.output / relative)
        records.append({'file': str(relative),
                        'before_sha256': hashlib.sha256((args.site / relative).read_bytes()).hexdigest(),
                        'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output / 'overlay.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
