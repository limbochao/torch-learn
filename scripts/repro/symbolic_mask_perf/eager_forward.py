"""Eager reference for ``triton_poi_fused_add_gelu_split_with_sizes_229``."""

import torch


def eager_forward(bmm_26, arg1288_1):
    """Reconstruct the eager computation represented by ``new_kernel.py``."""
    add_14751 = torch.ops.aten.add.Tensor(bmm_26, arg1288_1)
    getitem_470 = torch.ops.aten.split_with_sizes.default(add_14751, [200, 960], -1)[0]
    mul_9678 = torch.ops.aten.mul.Tensor(getitem_470, getitem_470)
    mul_9679 = torch.ops.aten.mul.Tensor(mul_9678, getitem_470)
    mul_9680 = torch.ops.aten.mul.Tensor(mul_9679, 0.044715)
    add_14764 = torch.ops.aten.add.Tensor(getitem_470, mul_9680)
    mul_9681 = torch.ops.aten.mul.Tensor(add_14764, 1.5957691216057308)
    sigmoid_18 = torch.ops.aten.sigmoid.default(mul_9681)
    mul_9682 = torch.ops.aten.mul.Tensor(getitem_470, sigmoid_18)
    return mul_9682
