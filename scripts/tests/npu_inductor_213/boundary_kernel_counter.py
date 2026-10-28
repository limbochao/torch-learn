"""Record kernel construction sites without changing metric values."""

import json
import os
from pathlib import Path
import traceback


def pytest_configure():
    from torch._inductor.codegen.common import Kernel
    original = Kernel.__init__

    def initialize(self, *args, **kwargs):
        directory = os.environ.get("BOUNDARY_ARTIFACT_DIR")
        if directory:
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "kernel-construction.jsonl").open("a") as stream:
                stream.write(json.dumps({"type": type(self).__name__,
                                         "stack": traceback.format_stack(limit=18),
                                         "kwargs": repr(kwargs)}) + "\n")
        return original(self, *args, **kwargs)

    Kernel.__init__ = initialize
