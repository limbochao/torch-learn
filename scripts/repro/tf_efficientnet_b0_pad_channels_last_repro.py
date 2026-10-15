"""复现 tf_efficientnet_b0 推理时 constant_pad_nd 输出的 stride 断言失败。

环境：NPU_A5，pta_213_cicd，torch 2.13.0+cpu，torch_npu 2.13.0.dev20261013。
原始编译缓存 buf23 -> buf25：
    input: float32[128, 96, 112, 112], stride=(1204224, 1, 10752, 96)
    aten.constant_pad_nd.default(input, [0, 1, 0, 1], 0.0)
    expected output: [128, 96, 113, 113], stride=(1225824, 1, 10848, 96)
    actual output stride=(1225824, 12769, 113, 1)

运行：python tf_efficientnet_b0_pad_channels_last_repro.py
小输入：SHAPE=1,2,3,3 python tf_efficientnet_b0_pad_channels_last_repro.py
CHECK=0 可跳过 eager 对比；不需要 timm、模型权重或数据集。
使用 NPU Triton 默认配置，无需修改 decomposition 或 fallback 列表。
"""

import os

# 与 benchmark --npu-backend triton 对应，在导入 torch_npu 前设置。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"

import torch
import torch.nn.functional as F
import torch_npu


def pad(x):
    return F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)


def main():
    shape = tuple(int(dim) for dim in os.environ.get("SHAPE", "128,96,112,112").split(","))
    check = os.environ.get("CHECK", "1") == "1"
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(memory_format=torch.channels_last).to("npu")
    print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}", flush=True)
    print(f"input: shape={tuple(x.shape)}, stride={x.stride()}, dtype={x.dtype}", flush=True)
    print(f"backend=inductor, npu_backend=triton, dynamic=False, aclgraph=False, check={check}", flush=True)
    compiled = torch.compile(
        pad,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "triton.cudagraph_trees": False, "size_asserts": True},
    )
    with torch.no_grad():
        if check:
            expected = pad(x)
            torch.npu.synchronize()
            print(f"eager: shape={tuple(expected.shape)}, stride={expected.stride()}", flush=True)
        # 保留编译器原始异常，不手写 stride 断言，也不调用已有缓存 kernel。
        actual = compiled(x)
        torch.npu.synchronize()
        print(f"compiled: shape={tuple(actual.shape)}, stride={actual.stride()}", flush=True)
        if check:
            torch.testing.assert_close(actual, expected)
            print("eager/compiled output check passed", flush=True)


if __name__ == "__main__":
    main()
