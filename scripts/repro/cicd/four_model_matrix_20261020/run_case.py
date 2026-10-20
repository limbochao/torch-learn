"""Run a benchmark, preserving failed direct Triton compilations and enforcing a candidate timeout."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time

import psutil
import triton

capture = Path(os.environ['CASE_DIR']) / 'compile_capture'
capture.mkdir(exist_ok=True)
original = triton.compile


def guarded_compile(src, *args, **kwargs):
    start = time.time()
    fn = getattr(src, 'fn', None)
    body = getattr(fn, 'src', '')
    name = getattr(fn, '__name__', 'kernel')
    info = {'kernel': name, 'started': start, 'signature': repr(getattr(src, 'signature', None)),
            'constants': repr(getattr(src, 'constants', None)), 'attrs': repr(getattr(src, 'attrs', None)),
            'target': repr(kwargs.get('target')), 'options': repr(kwargs.get('options'))}
    identity = body + info['signature'] + info['constants'] + info['attrs'] + repr(kwargs)
    key = f'{name}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}'

    def preserve(event, error=None):
        out = capture / key
        out.mkdir(exist_ok=True)
        (out / 'kernel_body.py').write_text(body)
        record = dict(info, event=event, elapsed=time.time()-start, error=str(error) if error else None)
        (out / 'record.json').write_text(json.dumps(record, indent=2))
        return out

    def expired():
        import shutil
        out = preserve('CANDIDATE_TIMEOUT_3600')
        commands = []
        for child in psutil.Process().children(recursive=True):
            try:
                command = child.cmdline()
                if any('bisheng' in value for value in command):
                    commands.append(command)
                    for value in command:
                        path = Path(value)
                        if path.is_file() and path.suffix in ('.ll', '.cpp', '.mlir', '.ttir'):
                            shutil.copy2(path, out / path.name)
                            for sibling in path.parent.iterdir():
                                if sibling.is_file() and sibling.suffix in ('.ll', '.cpp', '.mlir', '.ttir'):
                                    shutil.copy2(sibling, out / sibling.name)
            except (OSError, psutil.Error):
                pass
        (out / 'compiler_commands.json').write_text(json.dumps(commands, indent=2))
        print(f'CANDIDATE_TIMEOUT_3600: {name}; evidence={out}', flush=True)
        os.killpg(os.getpgrp(), signal.SIGTERM)

    timer = threading.Timer(3600, expired)
    timer.daemon = True
    timer.start()
    try:
        return original(src, *args, **kwargs)
    except Exception as error:
        preserve('COMPILE_EXCEPTION', error)
        raise
    finally:
        timer.cancel()


triton.compile = guarded_compile
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
