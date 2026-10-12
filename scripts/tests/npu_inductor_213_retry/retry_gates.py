"""Process-local test gates for the explicitly requested NPU retry."""
import os
import unittest

EVENTS = []
REASONS = {
    'requires gpu and triton', 'requires cuda and triton', 'CUDA not available',
    'requires CPU and GPU', 'requires GPU', 'Requires Triton backend.',
    'torch.compile for gpu requires triton',
    'Skipping triton backend only since not big GPU (not enough SM)',
}


def note(kind, target, **details):
    EVENTS.append(dict(kind=kind, target=target, **details))


def pytest_configure(config):
    if os.environ.get('RETRY_UNSKIP') != '1':
        return
    import torch
    from torch.testing._internal import common_device_type as cdt
    from torch.testing._internal import inductor_utils as iu
    from torch.testing._internal import triton_utils as tu

    if not torch.npu.is_available():
        raise RuntimeError('NPU is unavailable; cannot enable NPU retry gates')
    original_skip = unittest.skip

    def skip(reason):
        if reason == "" and os.environ.get('RETRY_FBCODE_FLOAT_ITEM') == '1':
            def fbcode_gate(obj):
                if obj.__qualname__ == 'TestInductorDynamic.test_float_item_return':
                    note('fbcode_decorator', obj.__qualname__, reason='IS_FBCODE-only marker')
                    return obj
                return original_skip(reason)(obj)
            return fbcode_gate
        if reason not in REASONS:
            return original_skip(reason)

        def decorate(obj):
            note('skip_decorator', obj.__qualname__, reason=reason)
            return obj

        return decorate

    unittest.skip = skip
    unittest.case.skip = skip
    for name, reason in [
        ('requires_gpu_and_triton', 'requires gpu and triton'),
        ('requires_cuda_and_triton', 'requires cuda and triton'),
    ]:
        setattr(tu, name, skip(reason))
    original_call = cdt.skipIf.__call__

    def device_skip(decorator, fn):
        if decorator.reason in REASONS and decorator.dep is True:
            note('device_skip_decorator', fn.__qualname__, reason=decorator.reason)
            return fn
        return original_call(decorator, fn)

    cdt.skipIf.__call__ = device_skip
    if os.environ.get('RETRY_EXTENSION_GATE') == '1':
        note('test_capability', 'inductor_utils.HAS_GPU_AND_TRITON',
             before=bool(iu.HAS_GPU_AND_TRITON), after=True)
        iu.HAS_GPU_AND_TRITON = True
    if os.environ.get('RETRY_NPU_MEMORY') == '1':
        original_memory_check = cdt._has_sufficient_memory

        def sufficient_memory(device, size):
            dev = torch.device(device)
            if dev.type != 'npu':
                return original_memory_check(device, size)
            torch.npu.empty_cache()
            free, total = torch.npu.mem_get_info(dev)
            note('memory_check', str(device), required_bytes=size, free_bytes=free, total_bytes=total)
            return free >= size

        cdt._has_sufficient_memory = sufficient_memory


def pytest_collection_modifyitems(session, config, items):
    if os.environ.get('RETRY_DYNAMIC_GATE') == '1':
        for module in {item.module for item in items}:
            if module.__name__.endswith('test_torchinductor_dynamic_shapes'):
                note('test_setup_gate', module.__name__ + '.HAS_GPU', before=bool(module.HAS_GPU), after=True)
                module.HAS_GPU = True
