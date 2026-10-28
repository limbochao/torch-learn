"""Replay one original Conv/BN combination with explicit stage synchronization."""

import argparse
import ast
import copy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--test-file', type=Path, required=True)
    parser.add_argument('--hf32', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    import torch
    import torch_npu
    from torch import nn
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from torch_npu.utils import _dynamo

    _dynamo.patch_has_triton()
    _dynamo.use_jit_script = True
    torch.npu.conv.allow_hf32 = bool(args.hf32)
    torch.manual_seed(1234)
    namespace = {'torch': torch, 'nn': nn}
    tree = ast.parse(args.test_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {'VanillaConvOp', 'MultiUserConvOp'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-test-model>', 'exec'), namespace)
    model = namespace['MultiUserConvOp'](
        nn.ConvTranspose2d, nn.BatchNorm2d, True, 3, 32, 'npu', kernel_size=3, stride=2
    ).eval()
    optimized = copy.deepcopy(model)
    inp = torch.rand(4, 3, 4, 4, device='npu')
    torch.npu.synchronize()
    print('INPUT_READY hf32=', args.hf32, flush=True)
    eager = model(inp)
    torch.npu.synchronize()
    print('EAGER_FORWARD_PASS', flush=True)
    eager.mean().backward()
    torch.npu.synchronize()
    print('EAGER_BACKWARD_PASS', flush=True)
    compiled = torch.compile(optimized)
    actual = compiled(inp)
    torch.npu.synchronize()
    print('COMPILED_FORWARD_PASS', flush=True)
    torch.testing.assert_close(actual, eager, atol=1e-5, rtol=1.3e-6)
    actual.mean().backward()
    torch.npu.synchronize()
    print('COMPILED_BACKWARD_PASS', flush=True)


if __name__ == '__main__':
    main()
