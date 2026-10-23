"""独立运行用户附件中 kernel `_relu_1` 对应的 eager 子图和 Inductor 编译。

完整 kernel 名：
triton_per_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1
来源：用户附件 Pasted text.txt，AOT ID ['0_inference']。
wrapper 第 1894 行先完成 BN/ReLU，第 1903 行将 buf2 传给目标 kernel。
目标 DSL 只执行 max pooling，仅写出池化值，不写 indices。
原 Graph fragment 混入了上游 BN/ReLU，并缺失 getitem；此处按实际 load/store
边界保留 relu 输入，补全 max_pool2d_with_indices(...)[0]。

输入：float32[16, 64, 112, 112]，stride=(802816, 12544, 112, 1)。
输出：float32[16, 64, 56, 56]，stride=(200704, 3136, 56, 1)。
输入为固定 seed 的合成非负数据，不是真实捕获值；无需模型、数据集或原始缓存。
依赖 PyTorch 和对应设备的 Inductor/Triton 环境，NPU 另需 torch_npu。

运行（默认 NPU，自动执行 eager、compile 和精确结果对比）：
    python max_pool2d_kernel_1_eager_repro.py
    python max_pool2d_kernel_1_eager_repro.py --device cuda
    python max_pool2d_kernel_1_eager_repro.py --device cpu
导出生成代码：
    TORCH_LOGS=output_code python max_pool2d_kernel_1_eager_repro.py
"""

import argparse
import os

# 选择 NPU Triton backend，在导入 torch/torch_npu 之前设置。
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"

import torch


def eager_forward(relu):
    return torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])[0]


def make_inputs(device, seed=0):
    # 在 CPU 上生成同一份数据，再传到目标设备，便于跨设备对照。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    relu = torch.empty_strided(
        (16, 64, 112, 112),
        (802816, 12544, 112, 1),
        device="cpu",
        dtype=torch.float32,
    )
    relu.normal_(generator=generator).relu_()
    return (relu.to(device),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu", help="npu[:index], cuda[:index], or cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu

        print(f"torch_npu={torch_npu.__version__}", flush=True)

    inputs = make_inputs(args.device, args.seed)
    x = inputs[0]
    assert x.stride() == (802816, 12544, 112, 1)
    print(f"torch={torch.__version__}, device={x.device}, seed={args.seed}, input=synthetic", flush=True)
    print(f"input: shape={tuple(x.shape)}, stride={x.stride()}, dtype={x.dtype}", flush=True)
    print("backend=inductor, fullgraph=True, dynamic=False", flush=True)

    with torch.no_grad():
        expected = eager_forward(*inputs)
        assert tuple(expected.shape) == (16, 64, 56, 56)
        assert expected.dtype == torch.float32
        assert torch.isfinite(expected).all().item(), "non-finite eager output"
        print(f"eager: shape={tuple(expected.shape)}, stride={expected.stride()}", flush=True)

        compiled = torch.compile(eager_forward, backend="inductor", fullgraph=True, dynamic=False)
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"compiled: shape={tuple(actual.shape)}, stride={actual.stride()}", flush=True)
        print("PASS compiled vs eager (rtol=0, atol=0)", flush=True)


if __name__ == "__main__":
    main()
