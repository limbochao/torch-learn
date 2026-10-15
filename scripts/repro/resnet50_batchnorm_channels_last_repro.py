"""复现 resnet50 训练时 BatchNorm 前向输出的 stride 断言失败。

环境：NPU_A5，pta_213_cicd，torch 2.13.0+cpu，torch_npu 2.13.0.dev20261013。
从原始编译缓存 buf20 -> buf22 提取：
    input: float32[32, 64, 112, 112], stride=(802816, 1, 7168, 64)
    aten._native_batch_norm_legit_no_training(input, weight, bias, mean, var, 0.1, 1e-5)[0]
    expected output stride=(802816, 1, 7168, 64)
    actual output stride=(802816, 12544, 112, 1)

运行：python resnet50_batchnorm_channels_last_repro.py
小输入：SHAPE=1,2,3,3 python resnet50_batchnorm_channels_last_repro.py
CHECK=0 可跳过 eager 对比；不需要 TorchBench、权重下载或数据集。
原始 --training 命令在前向就失败，节点实际使用 BatchNorm 的 running statistics。
因此保持 training=False，同时让输入、weight、bias 需要梯度，不使用 no_grad 编译。
还需复用 benchmarks/torchbench/npu_support.py::_patch_model_18 中的 BN decomposition 移除；
默认 decomposition 会把 BN 展开为逐元素运算，从而不经过原始失败的 NPU 算子。
"""

import os

# 与 TorchBench --npu-backend triton 一致，在导入 torch_npu 前设置。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"

import torch
import torch.nn.functional as F
import torch_npu
from torch._decomp import remove_decompositions
from torch._inductor.decomposition import decompositions


def batchnorm(x, weight, bias, running_mean, running_var):
    return F.batch_norm(
        x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5
    )


def main():
    shape = tuple(int(dim) for dim in os.environ.get("SHAPE", "32,64,112,112").split(","))
    check = os.environ.get("CHECK", "1") == "1"
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(memory_format=torch.channels_last).to("npu").requires_grad_()
    weight = torch.randn(shape[1], dtype=torch.float32).to("npu").requires_grad_()
    bias = torch.randn(shape[1], dtype=torch.float32).to("npu").requires_grad_()
    running_mean = torch.zeros(shape[1], device="npu", dtype=torch.float32)
    running_var = torch.ones(shape[1], device="npu", dtype=torch.float32)
    args = (x, weight, bias, running_mean, running_var)
    print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}", flush=True)
    for name, tensor in zip(("input", "weight", "bias", "running_mean", "running_var"), args):
        print(
            f"{name}: shape={tuple(tensor.shape)}, stride={tensor.stride()}, "
            f"dtype={tensor.dtype}, requires_grad={tensor.requires_grad}",
            flush=True,
        )
    print(f"backend=inductor, npu_backend=triton, dynamic=False, aclgraph=False, check={check}", flush=True)

    if check:
        with torch.no_grad():
            expected = batchnorm(*args)
        torch.npu.synchronize()
        print(f"eager: shape={tuple(expected.shape)}, stride={expected.stride()}", flush=True)

    # TorchBench 对 resnet50 本来就移除了此 decomposition；这里只保留与本用例有关的配置。
    remove_decompositions(decompositions, [torch.ops.aten._native_batch_norm_legit_no_training])
    print("torchbench config: _native_batch_norm_legit_no_training decomposition removed", flush=True)
    compiled = torch.compile(
        batchnorm,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "triton.cudagraph_trees": False, "size_asserts": True},
    )
    # 保留编译器生成的 stride 断言异常，原问题在 backward 之前发生。
    actual = compiled(*args)
    torch.npu.synchronize()
    print(f"compiled: shape={tuple(actual.shape)}, stride={actual.stride()}", flush=True)
    if check:
        torch.testing.assert_close(actual, expected)
        print("eager/compiled output check passed", flush=True)
    actual.sum().backward()
    torch.npu.synchronize()
    print("backward completed", flush=True)


if __name__ == "__main__":
    main()
