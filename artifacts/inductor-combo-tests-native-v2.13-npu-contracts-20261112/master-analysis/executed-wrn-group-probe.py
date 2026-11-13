import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--cases", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor-cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton-cache"))
sys.path.insert(0, args.cases)

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo
_dynamo.use_jit_script = True
_dynamo.patch_has_triton()
torch.cuda.get_device_capability = lambda: (10, 0)
import torch_npu.testing
import torch_npu._inductor
import test_combo_kernels as cases
from torch._inductor import config
from torch._inductor.scheduler import Scheduler, ForeachKernelSchedulerNode
from torch._inductor.utils import run_and_get_code

torch.npu.set_device(0)
result = {"started": time.time(), "master_device_retest": False,
          "torch_git": torch.version.git_version, "torch_npu": torch_npu.__version__,
          "physical_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "groups": []}
original = Scheduler.create_combo_kernel_nodes


def node_record(node):
    return {"name": node.get_name(), "type": type(node).__name__,
            "members": [n.get_name() for n in node.get_nodes()],
            "unmet_dependencies": [str(dep) for dep in node.unmet_dependencies]}


def create(owner, *args, **kwargs):
    groups = ForeachKernelSchedulerNode.group_nodes_for_combo_kernels(owner)
    levels = owner._topological_sort_nodes()
    result["level_sizes"] = [len(group) for group in levels]
    result["groups"] = [
        {"members": [node_record(n) for n in group],
         "combinable": [node_record(n) for n in ForeachKernelSchedulerNode.combinable_nodes(group)]}
        for group in groups
    ]
    result["before"] = [node_record(n) for n in owner.nodes]
    answer = original(owner, *args, **kwargs)
    result["after"] = [node_record(n) for n in owner.nodes]
    return answer


model = cases.ComboKernelPeakMemoryTests._make_wide_resnet_like().to("npu").eval()
x = torch.randn(1, 3, 224, 224, device="npu")
with config.patch({"combo_kernels": True, "benchmark_combo_kernel": False,
                   "combo_kernel_per_subkernel_blocks": True,
                   "combo_kernel_max_distance": 32,
                   "combo_kernel_peak_memory_increase_gb": 1.0 / 1024,
                   "combo_kernel_peak_memory_pct_threshold": None}), \
        patch.object(Scheduler, "create_combo_kernel_nodes", create), torch.no_grad():
    actual, sources = run_and_get_code(torch.compile(model), x)
    torch.npu.synchronize()
for i, source in enumerate(sources):
    (root / f"output-{i}.py").write_text(source)
result["finished"] = time.time()
result["multi_combinable_groups"] = sum(len(g["combinable"]) >= 2 for g in result["groups"])
result["combo_count"] = sum(s.count("'grid_type': 'SequentialComboKernelGrid'") for s in sources)
(root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("WRN_GROUP_SUMMARY=" + json.dumps({k: v for k, v in result.items()
                                        if k not in ["groups", "before", "after"]}), flush=True)
