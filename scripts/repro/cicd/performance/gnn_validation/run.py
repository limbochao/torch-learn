"""Same-boundary GNN kernel experiments; no installed-source changes."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--case', choices=['edgecnn', 'gcn', 'scatter'], required=True)
    args = parser.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == str(args.device)
    import torch
    import torch_npu
    import triton
    from kernels import partial, finish, scatter_message

    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root / 'scripts/tools'))
    from autotune_tiling import BestTilingRecorder

    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(20261104)
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.trace.enabled = True
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'case': args.case, 'physical_device': args.device,
              'versions': {m.__name__: m.__version__ for m in (torch, torch_npu, triton)}, 'cases': {}}

    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')

    def bench(name, fn, expected):
        try:
            for _ in range(5):
                outputs = fn()
            torch.npu.synchronize()
            errors = []
            for actual, reference in zip(outputs, expected):
                actual = actual.cpu().double()
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-3)
                errors.append(float((actual - reference).abs().max()))
            count = 16
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(count):
                    outputs = fn()
            torch.npu.synchronize()
            for tensor in outputs:
                tensor.fill_(float('nan'))
            graph.replay()
            torch.npu.synchronize()
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual.cpu().double(), reference, rtol=1e-4, atol=2e-3)
            samples = []
            for _ in range(9):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / count)
            result['cases'][name] = {'us': statistics.median(samples), 'samples_us': samples,
                                      'max_abs_error_fp64': errors, 'graph_output_check': 'pass'}
            print(name, result['cases'][name], flush=True)
        except Exception:
            result['cases'][name] = {'error': traceback.format_exc()}
            print(name, result['cases'][name], flush=True)
        save()

    if args.case == 'scatter':
        edges = 209978
        x_cpu = torch.randn(10000, 64)
        edge_cpu = torch.randint(0, 10000, (2, edges))
        weight_cpu = torch.rand(edges)
        ref = torch.zeros(10000, 64, dtype=torch.float64)
        ref.index_add_(0, edge_cpu[1], x_cpu.double()[edge_cpu[0]] * weight_cpu.double()[:, None])
        x, edge, weight = x_cpu.npu(), edge_cpu.npu(), weight_cpu.npu()

        def eager(x, edge, weight):
            message = weight[:, None] * x[edge[0]]
            return torch.zeros_like(x).scatter_add(0, edge[1, :, None].expand(-1, 64), message)

        bench('eager_same_boundary', lambda: (eager(x, edge, weight),), (ref,))
        compiled = torch.compile(eager, fullgraph=True)
        bench('compiled_fallback', lambda: (compiled(x, edge, weight),), (ref,))
        out = torch.empty_like(x)
        for block in [256, 1024, 2048]:
            def fused():
                out.zero_()
                scatter_message[(triton.cdiv(edges * 64, block),)](x, edge, weight, out, edges, block)
                return (out,)
            bench(f'fused_atomic_b{block}', fused, (ref,))
        return

    relu = args.case == 'edgecnn'
    rows = 200000 if relu else 10000
    a_cpu, g_cpu = torch.randn(rows, 64), torch.randn(rows, 64)
    a, g = a_cpu.npu(), g_cpu.npu()
    value_cpu = torch.where(a_cpu <= 0, 0., g_cpu) if relu else a_cpu
    reference = (value_cpu.double(), value_cpu.double().sum(0, keepdim=True)) if relu else (
        value_cpu.double().sum(0, keepdim=True),)
    output = torch.empty_like(a)
    reduced = torch.empty((1, 64), device='npu')
    path = root / f'artifacts/cicd/performance/gnn-validation/original/{args.case}.py'
    spec = importlib.util.spec_from_file_location('original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = 'triton_red_fused_sum_threshold_backward_3' if relu else 'triton_red_fused_sum_0'
    kernel = getattr(module, name)
    recorder = BestTilingRecorder('npu')
    recorder.install()
    recorder.start_capture()
    # Original ReLU kernel mutates its activation argument. Always copy it afresh,
    # including timed iterations, to preserve the original activation predicate.
    def original():
        stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
        if relu:
            output.copy_(a)
            kernel.run(output, g, reduced, 64, rows, stream=stream)
            return output, reduced
        kernel.run(a, reduced, 64, rows, stream=stream)
        return (reduced,)

    original()
    result['selected'] = recorder.stop_capture()
    recorder.uninstall()
    result['copy_included_for_inplace_baseline'] = relu
    bench('original_plus_restore' if relu else 'original', original, reference)
    selected = result['selected'][0]['selected_config']
    options = {k: selected[k] for k in ['num_warps', 'num_stages', 'compile_mode', 'multibuffer']}

    def eager(a, g):
        if relu:
            v = torch.where(a <= 0, 0., g)
            return v, v.sum(0, keepdim=True)
        return (a.sum(0, keepdim=True),)

    bench('eager_same_boundary', lambda: eager(a, g), reference)
    compiled = torch.compile(eager, fullgraph=True)
    bench('regenerated', lambda: compiled(a, g), reference)
    for xb, rb in [(1, 256), (4, 256), (16, 256), (32, 128), (64, 64)]:
        def fixed():
            if relu:
                output.copy_(a)
                kernel.fn[(triton.cdiv(64, xb),)](
                    output, g, reduced, 64, rows, xb, X0BLOCK_SUB=xb, R1BLOCK_SUB=rb, **options)
                return output, reduced
            kernel.fn[(triton.cdiv(64, xb),)](
                a, reduced, 64, rows, xb, X0BLOCK_SUB=xb, R1BLOCK_SUB=rb, **options)
            return (reduced,)
        bench(f'same_dsl_x{xb}_r{rb}', fixed, reference)
    for chunk in [128, 512, 2048]:
        parts = triton.cdiv(rows, chunk)
        temp = torch.empty((parts, 64), device='npu')
        def split():
            partial[(2, parts)](a, g, output, temp, rows, relu, chunk, 32, 128, **options)
            finish[(2,)](temp, reduced, parts, 32, triton.next_power_of_2(parts), **options)
            return (output, reduced) if relu else (reduced,)
        bench(f'two_stage_chunk{chunk}', split, reference)


if __name__ == '__main__':
    main()
