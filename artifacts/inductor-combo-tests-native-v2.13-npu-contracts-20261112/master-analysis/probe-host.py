from pathlib import Path
import json
payload = json.loads(Path(__file__).with_name('host-probe-payload.json').read_text())
import ast
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch._inductor.config as config
from torch._inductor import debug
from torch._inductor.scheduler import Scheduler

assert 'torch_npu' not in sys.modules
result = {'device_execution': False, 'torch': torch.__version__, 'torch_git': torch.version.git_version}

class FakeKernel:
    def __init__(self):
        self.num_load = self.num_store = self.num_reduction = 0
        self.emitted_stores = 0
        self.cse = SimpleNamespace(invalidated_stores=set(), store_cache={})
        self.store_buffer_names = set()
        self.current_node = None
    def load(self, name, index):
        return SimpleNamespace(use_count=1)
    def store(self, name, index, value, mode=None):
        self.emitted_stores += 1
    def reduction(self, *args):
        return 'value'

def run_proxy(label):
    kernel = FakeKernel()
    proxy = SimpleNamespace(kernel=kernel, _update_store_cache=lambda *args: None)
    ns = {'self': kernel, 'CSEProxy': SimpleNamespace(_update_store_cache=lambda *args: None),
          'V': SimpleNamespace(graph=SimpleNamespace(removed_buffers=set())),
          'free_symbol_is_type': lambda *args: False,
          'NPUSymT': SimpleNamespace(TMP=None), 'SymT': SimpleNamespace(TMP=None)}
    for method in ['load', 'store', 'reduction']:
        exec('from __future__ import annotations\n' + payload[label + '_' + method], ns)
        args = {'load': ['input', 0], 'store': ['output', 0, 'value'],
                'reduction': [torch.float32, torch.float32, 'sum', 'value']}[method]
        if label == 'common':
            args.insert(0, proxy)
        ns[method](*args)
    return {k: getattr(kernel, k) for k in ['num_load', 'num_store', 'num_reduction', 'emitted_stores']}

result['cse_source_replay'] = {label: run_proxy(label) for label in ['npu', 'common']}

def bench_replay(flag):
    calls = {'clone': 0, 'cpu_copy': 0, 'restore': 0, 'launch': 0}
    def clone(*args, **kwargs):
        calls['clone'] += 1
        return args, kwargs
    def cpu_copy(*args, **kwargs):
        calls['cpu_copy'] += 1
        return {}
    def restore(*args, **kwargs):
        calls['restore'] += 1
    def launch(*args, **kwargs):
        calls['launch'] += 1
    interface = SimpleNamespace(current_device=lambda: 0, get_raw_stream=lambda device: None)
    owner = SimpleNamespace(optimize_mem=flag, inductor_meta={}, clone_args=clone,
                            copy_args_to_cpu_if_needed=cpu_copy, restore_args_from_cpu=restore,
                            get_device_interface=lambda: interface, reset_to_zero_args=lambda *a, **k: None)
    def benchmark(fn, **kwargs):
        fn()
        return 1.0
    ns = {'benchmarker': SimpleNamespace(benchmark_gpu=benchmark)}
    exec('from __future__ import annotations\n' + payload['npu_bench'], ns)
    ns['_bench_with_launch_args'](owner, launch, (1, 2), (1, 2))
    return calls

result['npu_bench_optimize_mem_source_replay'] = {str(flag): bench_replay(flag) for flag in [False, True]}

def provenance(nodes):
    debug._inductor_triton_kernel_to_post_grad_node_info = {}
    debug._inductor_kernel_stack_trace = {}
    debug._inductor_kernel_provenance_debug_handle = 0
    with config.patch({'trace.provenance_tracking_level': 2}):
        handle = debug.set_kernel_post_grad_provenance_tracing(nodes, 'combo_kernel')
    return {'handle': handle, 'mapping': debug._inductor_triton_kernel_to_post_grad_node_info}

leaves = [SimpleNamespace(node=SimpleNamespace(origins=[SimpleNamespace(name=name)],
                                              get_stack_traces=lambda: []))
          for name in ['relu', 'sigmoid', 'tanh']]
result['provenance_source_replay'] = {
    'combo_shell': provenance([SimpleNamespace(node=None)]), 'partition_leaves': provenance(leaves)}

InductorTestCase = unittest.TestCase
exec('from __future__ import annotations\n' + payload['gate_helpers'])
test = ComboKernelPeakMemoryTests()
test.test_threshold_gating()
test.test_region_carry_in_uses_post_free_boundary()
result['gate_existing_host_tests'] = {
    'positive_2MB_delta_rejected_under_1MB_limit': True,
    'zero_delta_accepted_under_zero_limit': True,
    'source': 'existing test_threshold_gating and test_region_carry_in_uses_post_free_boundary'}

class DistanceNode:
    def __init__(self, name, deps):
        self.name = name
        self.unmet_dependencies = [SimpleNamespace(name=n) for n in deps]
    def get_name(self):
        return self.name

def distances(after_extern):
    nodes = [DistanceNode('input_foreach', [])] if not after_extern else [
        DistanceNode('mm', []), DistanceNode('foreach_add', ['mm_output'])]
    owner = SimpleNamespace(nodes=nodes, name_to_buf={
        'mm_output': SimpleNamespace(defining_op_name=lambda: 'mm')})
    Scheduler.compute_input_distances(owner)
    return [{'name': n.name, 'min': n.min_input_distance, 'max': n.max_input_distance} for n in nodes]

result['scheduler_distance_source_replay'] = {
    'direct_input': distances(False), 'after_extern': distances(True),
    'scope': 'dependency algorithm only, not the skipped NPU foreach tests'}
with torch.device('meta'):
    model = ComboKernelPeakMemoryTests._make_wide_resnet_like()
result['wide_resnet_meta_model'] = {
    'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
    'buffer_bytes': sum(p.numel() * p.element_size() for p in model.buffers()),
    'scope': 'meta-device model footprint only; no attribution of the NPU runtime peak'}
result['npu_backend_loaded'] = 'torch_npu' in sys.modules
assert not result['npu_backend_loaded']
print('HOST_PROBE_JSON=' + json.dumps(result, ensure_ascii=False))
