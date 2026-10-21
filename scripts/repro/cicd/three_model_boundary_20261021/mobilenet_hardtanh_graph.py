"""Dynamic float hardtanh backward, without MobileNet or model inputs."""
import torch
import torch_npu


def backward(grad, value, lower, upper):
    return torch.ops.aten.hardtanh_backward.default(grad, value, lower, upper).sum(dim=0)


torch.manual_seed(0)
torch.npu.set_device(0)
value = torch.randn(16, 64, device="npu")
grad = torch.randn_like(value)
expected = backward(grad, value, 0.0, 6.0)
actual = torch.compile(backward, dynamic=True, fullgraph=True)(grad, value, 0.0, 6.0)
torch.npu.synchronize()
torch.testing.assert_close(actual, expected)
print("GRAPH_PASS")
