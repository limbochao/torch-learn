"""Retain generated code visible in a failing pytest traceback without changing execution."""

import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.hookimpl(trylast=True)
def pytest_runtest_makereport(item, call):
    if call.excinfo is None or not os.environ.get('BOUNDARY_ARTIFACT_DIR'):
        return
    output = Path(os.environ['BOUNDARY_ARTIFACT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    frame = call.excinfo.value.__traceback__
    while frame is not None:
        for name in ['code', 'code_unfused', 'source_codes', 'source_code']:
            value = frame.tb_frame.f_locals.get(name)
            values = [value] if isinstance(value, str) else value if isinstance(value, (tuple, list)) else []
            for index, source in enumerate(values):
                if not isinstance(source, str) or ('import triton' not in source and '# AOT ID:' not in source):
                    continue
                digest = hashlib.sha256(source.encode()).hexdigest()
                filename = f'{digest[:16]}.py'
                (output / filename).write_text(source)
                records.append({'nodeid': item.nodeid, 'local': name, 'index': index,
                                'file': filename, 'sha256': digest})
        frame = frame.tb_next
    if records:
        manifest = output / 'manifest.json'
        previous = json.loads(manifest.read_text()) if manifest.exists() else []
        manifest.write_text(json.dumps(previous + records, indent=2) + '\n')


def pytest_configure():
    from torch._inductor.codecache import PyCodeCache

    original = PyCodeCache.load.__func__

    @classmethod
    def load(cls, source_code, *args, **kwargs):
        directory = os.environ.get('BOUNDARY_ARTIFACT_DIR')
        if directory and isinstance(source_code, str) and (
            'import triton' in source_code or '# AOT ID:' in source_code
        ):
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(source_code.encode()).hexdigest()
            (output / f'{digest[:16]}.py').write_text(source_code)
        return original(cls, source_code, *args, **kwargs)

    PyCodeCache.load = load

    original_by_key = PyCodeCache.load_by_key_path.__func__

    @classmethod
    def load_by_key_path(cls, key, path, *args, **kwargs):
        directory = os.environ.get('BOUNDARY_ARTIFACT_DIR')
        source_path = Path(path)
        if directory and source_path.is_file():
            source = source_path.read_text()
            if 'async_compile.wait' in source:
                output = Path(directory)
                output.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(source.encode()).hexdigest()
                (output / f'wrapper-{digest[:16]}.py').write_text(source)
        return original_by_key(cls, key, path, *args, **kwargs)

    PyCodeCache.load_by_key_path = load_by_key_path
