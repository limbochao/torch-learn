import os


os.environ["INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"] = "1"
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
import triton
import triton.language as tl


NUM_SM_VALUES = (84, 128, 92)
BLOCK_VALUES = (256, 512, 1024)
CONFIGS = [
    triton.Config({"NUM_SM": num_sm, "BLOCK": block}, num_warps=4, num_stages=3)
    for num_sm in NUM_SM_VALUES
    for block in BLOCK_VALUES
]


@triton.autotune(configs=CONFIGS, key=[])
@triton.jit
def persistent_add_kernel(x_ptr, output_ptr, n_elements, BLOCK: tl.constexpr, NUM_SM: tl.constexpr):
    block_id = tl.program_id(0)
    block_count = tl.cdiv(n_elements, BLOCK)
    while block_id < block_count:
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, value + value, mask=mask)
        block_id += NUM_SM


def add(x):
    output = torch.empty_like(x)

    def grid(meta):
        return (meta["NUM_SM"],)

    persistent_add_kernel[grid](x, output, x.numel())
    return output


def make_input(numel):
    return torch.randn(numel, device="npu", dtype=torch.float32)


def run_and_check(compiled, numel, x=None):
    x = make_input(numel) if x is None else x
    actual = compiled(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, x + x)
    print(f"shape=({numel},) passed")


def main():
    torch.npu.set_device(0)
    first_input = make_input(262144)
    torch._dynamo.mark_dynamic(first_input, 0)
    compiled = torch.compile(add, backend="inductor", dynamic=None, fullgraph=True)
    print(f"group={os.environ['INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE']} configs={len(CONFIGS)}")
    run_and_check(compiled, first_input.numel(), first_input)
    run_and_check(compiled, 131072)


if __name__ == "__main__":
    main()
