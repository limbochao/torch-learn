"""复现 YOLOv3 的 max_pool2d 输出 stride 断言失败（NPU_A5 / pta_213_cicd）。

从原始 torchinductor 缓存的 buf219 -> buf221 提取：
    input: float32[8, 512, 12, 16], stride=(98304, 1, 8192, 512)
    aten.max_pool2d_with_indices.default(input, [5, 5], [1, 1], [2, 2])[0]
    expected output stride=(98304, 1, 8192, 512)
    actual output stride=(98304, 192, 16, 1)

运行：python yolov3_maxpool_channels_last_repro.py
可选：SHAPE=1,2,3,3 CHECK=1 python yolov3_maxpool_channels_last_repro.py
CHECK=0 跳过 eager 对比；不需要 TorchBench、模型权重或数据集。
"""

import os

# 与 TorchBench --npu-backend triton 对应，在导入 torch_npu 前设置。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"

import torch
import torch.nn.functional as F
import torch_npu


def maxpool(x):
    return F.max_pool2d(x, kernel_size=5, stride=1, padding=2)


def main():
    shape = tuple(int(dim) for dim in os.environ.get("SHAPE", "8,512,12,16").split(","))
    check = os.environ.get("CHECK", "1") == "1"
    torch.manual_seed(0)
    # 直接构造 channels-last 输入，保留原图中池化收到的实际布局。
    x = torch.randn(shape, dtype=torch.float32).to(memory_format=torch.channels_last).to("npu")
    print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}", flush=True)
    print(f"input: shape={tuple(x.shape)}, stride={x.stride()}, dtype={x.dtype}", flush=True)
    print(f"backend=inductor, npu_backend=default (triton), dynamic=False, aclgraph=False, check={check}", flush=True)

    compiled = torch.compile(
        maxpool,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
            "size_asserts": True,
        },
    )
    with torch.no_grad():
        if check:
            expected = maxpool(x)
            torch.npu.synchronize()
            print(f"eager: shape={tuple(expected.shape)}, stride={expected.stride()}", flush=True)
        # 保留编译器原始异常；不手写 stride 断言、不调用缓存 kernel。
        actual = compiled(x)
        torch.npu.synchronize()
        print(f"compiled: shape={tuple(actual.shape)}, stride={actual.stride()}", flush=True)
        if check:
            torch.testing.assert_close(actual, expected)
            print("eager/compiled check passed", flush=True)


if __name__ == "__main__":
    main()
