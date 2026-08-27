"""
Compare three equivalent x2 mask forms in the same generated SIMT kernel.

Run on an NPU environment:

    python scripts/repro/symbolic_mask_perf/compare_mask_kernels.py

Optional environment variables: S0, WARMUP, REPEAT, CHECK, PRINT_KERNEL_SOURCE.
"""

import hashlib
import os
import time

import torch
import torch_npu
from torch import empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream

torch_npu.npu._initialized = torch_npu.npu.is_initialized()
async_compile = AsyncCompile()


KERNEL_TEMPLATE = r'''
import triton
import triton.language as tl

from torch._inductor.runtime.hints import DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics

@triton_heuristics.pointwise(
    size_hints={'z0': 32, 'y1': 256, 'x2': 200},
    filename=__file__,
    triton_meta={
        'signature': {
            'in_ptr0': '*fp16',
            'in_ptr1': '*fp16',
            'out_ptr0': '*fp16',
            'ks0': 'i64',
            'z0_numel': 'i32',
            'y1_numel': 'i32',
            'x2_numel': 'i32',
            'Z0BLOCK': 'i32',
            'Y1BLOCK': 'i32',
        },
        'device': DeviceProperties(
            type='npu',
            index=0,
            multi_processor_count=56,
            cc='Ascend950PR_9579',
            major=None,
            regs_per_multiprocessor=None,
            max_threads_per_multi_processor=None,
            max_threads_per_block=None,
            warp_size=None,
        ),
        'constants': {},
        'mix_mode': 'aiv',
    },
    inductor_meta={
        'grid_type': 'GridNpu',
        'autotune_hints': set(),
        'kernel_name': '__KERNEL_NAME__',
        'mutated_arg_names': [],
        'backend_hash': '__BACKEND_HASH__',
        'split_axis': [0, 1],
        'tiling_axis': [0, 1, 2],
        'no_loop_axis': [2],
        'axis_names': ['z0', 'y1', 'x2'],
        'axis_static_values': (('z0', 32), ('x2', 200)),
        'low_dims': {2},
        'numof_reduction_axis': 0,
        'split_axis_dtype': torch.float16,
        'dual_reduction': False,
        'npu_kernel_type': 'simt_template',
        'traced_graph_hash': 'TRACED_GRAPH_HASH',
        'traced_graph_dir': 'TRACED_GRAPH_DIR',
        'are_deterministic_algorithms_enabled': False,
        'inductor_ascend_linear_mode': 'linear',
        'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'),
        'assert_indirect_indexing': True,
        'autotune_local_cache': True,
        'autotune_pointwise': True,
        'autotune_remote_cache': None,
        'force_disable_caches': False,
        'dynamic_scale_rblock': True,
        'max_autotune': False,
        'max_autotune_pointwise': False,
        'min_split_scan_rblock': 256,
        'spill_threshold': 16,
        'store_cubin': False,
        'deterministic': False,
        'force_filter_reduction_configs': False,
        'group_enabled': False,
        'group_template': None,
        'group_workload': None,
        'primary_group_axis': None,
        'static_split_axes': (),
        'secondary_runtime_symbolic_axes': (),
        'group_features': (),
    },
    min_elem_per_thread=0,
)
@triton.jit
def __KERNEL_NAME__(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    ks0,
    z0_numel,
    y1_numel,
    x2_numel,
    Z0BLOCK,
    Y1BLOCK,
    Z0BLOCK_SUB: tl.constexpr,
    Y1BLOCK_SUB: tl.constexpr,
):
    x2_numel = 200
    X2BLOCK_SUB: tl.constexpr = 200
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1 = tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:, None, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None, :, None]
            y1_mask = y1 < min(Y1BLOCK + y1_offset, y1_numel)
            x2 = base_x2[None, None, :]
            __X2_MASK_SETUP__
            tmp0 = tl.load(
                in_ptr0 + (x2 + 1160 * y1 + 1160 * ks0 * z0),
                __INPUT0_MASK__,
            ).to(tl.float32)
            tmp1 = tl.load(
                in_ptr1 + (x2 + 1160 * z0),
                __INPUT1_MASK__,
            ).to(tl.float32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2 * tmp2
            tmp4 = tmp3 * tmp2
            tmp5 = 0.044715
            tmp6 = tmp4 * tmp5
            tmp7 = tmp2 + tmp6
            tmp8 = 1.5957691216057308
            tmp9 = tmp7 * tmp8
            tmp10 = tl.sigmoid(tmp9)
            tmp11 = tmp2 * tmp10
            tl.store(
                out_ptr0 + (x2 + 200 * y1 + 200 * ks0 * z0),
                tmp11,
                __OUTPUT_MASK__,
            )
'''


MASK_VARIANTS = {
    'a_no_x2_mask': {
        'setup': '',
        'input0': 'y1_mask & z0_mask',
        'input1': 'z0_mask',
        'output': 'y1_mask & z0_mask',
    },
    'b_range_x2_mask': {
        'setup': 'x2_mask = x2 < 200',
        'input0': 'x2_mask & y1_mask & z0_mask',
        'input1': 'x2_mask & z0_mask',
        'output': 'x2_mask & y1_mask & z0_mask',
    },
    'c_full_true_x2_mask': {
        'setup': 'x2_mask = tl.full((1, 1, 200), True, tl.int1)',
        'input0': 'x2_mask & y1_mask & z0_mask',
        'input1': 'x2_mask & z0_mask',
        'output': 'x2_mask & y1_mask & z0_mask',
    },
}


def make_kernel_source(kernel_name, variant):
    source = KERNEL_TEMPLATE
    replacements = {
        '__KERNEL_NAME__': kernel_name,
        '__X2_MASK_SETUP__': variant['setup'],
        '__INPUT0_MASK__': variant['input0'],
        '__INPUT1_MASK__': variant['input1'],
        '__OUTPUT_MASK__': variant['output'],
    }
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    backend_hash = hashlib.sha256(source.encode()).hexdigest().upper()
    return source.replace('__BACKEND_HASH__', backend_hash)


KERNEL_SOURCES = {
    name: make_kernel_source(f'triton_mask_perf_{name}', variant)
    for name, variant in MASK_VARIANTS.items()
}

kernel_a_no_x2_mask = async_compile.triton(
    'triton_mask_perf_a_no_x2_mask',
    KERNEL_SOURCES['a_no_x2_mask'],
    device_str='npu',
)
kernel_b_range_x2_mask = async_compile.triton(
    'triton_mask_perf_b_range_x2_mask',
    KERNEL_SOURCES['b_range_x2_mask'],
    device_str='npu',
)
kernel_c_full_true_x2_mask = async_compile.triton(
    'triton_mask_perf_c_full_true_x2_mask',
    KERNEL_SOURCES['c_full_true_x2_mask'],
    device_str='npu',
)

async_compile.wait(globals())
del async_compile

KERNELS = {
    'A/no_x2_mask': kernel_a_no_x2_mask,
    'B/x2_lt_200': kernel_b_range_x2_mask,
    'C/full_true_x2_mask': kernel_c_full_true_x2_mask,
}


def launch(kernel, in_ptr0, in_ptr1, out_ptr0, s0, stream):
    kernel.run(
        in_ptr0,
        in_ptr1,
        out_ptr0,
        s0,
        32,
        s0,
        200,
        stream=stream,
    )


def benchmark(kernel, args, warmup, repeat):
    for _ in range(warmup):
        launch(kernel, *args)
    torch_npu.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        launch(kernel, *args)
    torch_npu.npu.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / repeat


def main():
    s0 = int(os.getenv('S0', '256'))
    warmup = int(os.getenv('WARMUP', '20'))
    repeat = int(os.getenv('REPEAT', '100'))
    check = os.getenv('CHECK', '1') == '1'

    if os.getenv('PRINT_KERNEL_SOURCE', '0') == '1':
        for name, source in KERNEL_SOURCES.items():
            print(f'===== {name} =====')
            print(source)

    torch.manual_seed(0)
    in_ptr0 = torch.randn((32, s0, 1160), device='npu', dtype=torch.float16)
    in_ptr1 = torch.randn((32, 1, 1160), device='npu', dtype=torch.float16)
    outputs = {
        name: empty_strided(
            (32, s0, 200),
            (200 * s0, 200, 1),
            device='npu',
            dtype=torch.float16,
        )
        for name in KERNELS
    }
    stream = get_raw_stream(0)

    for name, kernel in KERNELS.items():
        launch(kernel, in_ptr0, in_ptr1, outputs[name], s0, stream)
    torch_npu.npu.synchronize()

    if check:
        reference = outputs['A/no_x2_mask']
        for name in ('B/x2_lt_200', 'C/full_true_x2_mask'):
            torch.testing.assert_close(outputs[name], reference)
        print('accuracy: A, B and C outputs match')

    print(f's0={s0}, warmup={warmup}, repeat={repeat}')
    for name, kernel in KERNELS.items():
        args = (in_ptr0, in_ptr1, outputs[name], s0, stream)
        latency_us = benchmark(kernel, args, warmup, repeat)
        print(f'{name}: {latency_us:.3f} us')


if __name__ == '__main__':
    main()
