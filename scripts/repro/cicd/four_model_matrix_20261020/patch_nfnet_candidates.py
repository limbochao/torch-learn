"""Diagnosis only: add a small pooling tile while preserving all original candidates."""
import copy
import runpy
import sys

import torch_npu
from torch_npu._inductor.runtime import triton_heuristics

original = triton_heuristics.triton_config_npu_index


def add_small_pool_tile(size_hints, inductor_meta, *args, **kwargs):
    configs = original(size_hints, inductor_meta, *args, **kwargs)
    if ('avg_pool2d' in inductor_meta.get('kernel_name', '')
            and inductor_meta.get('requires_no_linear_block_remap') and configs):
        candidate = copy.deepcopy(configs[0])
        candidate.kwargs.update(XBLOCK=2048, split_blocks=(2048,), multibuffer=False)
        candidate.kwargs.pop('enable_vf_fusion', None)
        configs.append(candidate)
        print('DIAGNOSTIC_POOL_TILE', inductor_meta['kernel_name'], candidate.kwargs, flush=True)
    return configs


triton_heuristics.triton_config_npu_index = add_small_pool_tile
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
