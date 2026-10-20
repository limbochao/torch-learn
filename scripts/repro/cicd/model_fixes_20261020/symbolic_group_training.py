"""Dynamic spatial reduction and its pointwise producer, as in squeeze-excitation backward."""
import argparse

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def model(value, weight):
    value = F.conv2d(value, weight, stride=32)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--static', action='store_true')
args = parser.parse_args()
torch.manual_seed(0)
compiled = torch.compile(model, backend='inductor', dynamic=not args.static, fullgraph=True)
for batch, height in ((2, 224), (3, 256)):
    value = torch.randn(batch, 16, height, height, device='npu', requires_grad=True)
    weight = torch.randn(16, 16, 1, 1, device='npu', requires_grad=True)
    expected = model(value, weight)
    gradient = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (value, weight), gradient)
    actual = compiled(value, weight)
    actual_grads = torch.autograd.grad(actual, (value, weight), gradient)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-3)
    print(f'PASS dynamic={not args.static} batch={batch} height={height}', flush=True)
