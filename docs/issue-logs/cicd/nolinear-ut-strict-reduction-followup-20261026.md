---
title: no-linear 设备 UT 暴露的 strict_reduction 接口兼容问题（待定位）
published: false
---

# no-linear 设备 UT 暴露的 strict_reduction 接口兼容问题

状态：用户提供的 2026-10-26 流水线日志已确认失败，尚未修复，尚未在相同环境复测。
本文件仅在本地记录，不提交或推送；不为此扩大 [PR !48475](https://gitcode.com/Ascend/pytorch/pull/48475) 的修改范围。

## 失败现场

流水线以 Python 3.10.21 执行：

```bash
python test/_inductor/test_nolinear_fallback_fusion.py -v
```

原始代码版本为 `a741de5f5ee2ab26491bb607c9a42ead81a92ff2`，
设备用例为 `TestNoLinearFallbackFusionDevice.test_dynamic_training_with_forced_index_fallback`。
日志没有提供流水线实际安装的 PyTorch commit、torch_npu 完整版本和设备型号，这些信息需后续补齐。

11 个 ERROR 中，10 个来自 `unittest.TestCase.enterContext()` 在 Python 3.10 中不存在，
属于 UT 兼容性问题，已改用 `contextlib.ExitStack` 与 `addCleanup(stack.close)`。
另 1 个来自设备 UT 的 `mean` lowering，属于本文件跟踪的新问题：

```text
test_dynamic_training_with_forced_index_fallback
  -> compiled(compiled_input)
  -> aten.mean.dim
  -> torch_npu._inductor.lowering.mean
  -> sum_(x, axis, keepdim)
  -> torch._inductor.lowering.sum_
  -> make_reduction(..., strict_reduction=...)
TypeError: make_reduction() got an unexpected keyword argument 'strict_reduction'
```

错误发生在 lowering，尚未进入 scheduling、强制 index fallback 或目标 kernel 编译，
因此本日志不能判定 no-linear 融合复核修复是否有效，也没有生成这次失败对应的 kernel DSL。

## 源码证据

在上述代码版本中：

- `torch_npu/_inductor/lowering.py:145` 的 `make_reduction` 只接受
  `reduction_type`、`override_return_dtype`，没有 `strict_reduction`。
- 同文件第 177 行，以及 `torch_npu/_inductor/__init__.py:228`，
  会将 NPU 实现安装到公共 `torch._inductor.lowering.make_reduction`。
- `torch_npu/_inductor/lowering.py:405` 的 `mean` 调用社区 `sum_`。
  流水线 traceback 显示其已传入新的 `strict_reduction` 关键字，替换后的 NPU 函数无法接收。
- PR 三个功能提交均未修改上述 lowering 文件；该设备用例暴露了既有替换逻辑与流水线 PyTorch 接口的兼容问题。

上游当前 [lowering.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py)
的 `make_reduction` 已接受 `strict_reduction=False`，并将该值传给 `Reduction.create()`。
此链接为可变的 main 分支，仅用于说明接口形式，不能据此确定流水线使用的版本或引入该参数的提交。

## 保留后续复现入口

原设备 UT 的核心计算如下；输入为 contiguous FP32 NPU tensor，
尺寸依次为 `(2, 16, 65, 65)`、`(3, 16, 97, 97)`，使用动态 shape 并计算输入梯度。
下面保留核心图的手动复现候选，去掉了发生在更后面 scheduling 阶段的强制 fallback hook。
该简化版本尚未执行，不能声称已经独立复现。

```python
import torch
import torch_npu
from torch._inductor import config

torch.manual_seed(0)
weight = torch.randn(16, 16, 3, 3, device="npu")

def fn(x):
    value = torch.nn.functional.conv2d(x, weight, stride=32, padding=1)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()

print("torch:", torch.__version__)
print("torch commit:", torch.version.git_version)
print("torch_npu:", torch_npu.__version__)
torch._dynamo.reset()
try:
    with torch.npu.aclnn.flags(allow_hf32=False), config.patch({"fx_graph_cache": False}):
        compiled = torch.compile(fn, dynamic=True)
        for batch, side in ((2, 65), (3, 97)):
            x = torch.randn(batch, 16, side, side, device="npu", requires_grad=True)
            result = compiled(x)
            result.backward(torch.randn_like(result))
finally:
    torch._dynamo.reset()
```

完整原始 UT 可从 `327c8063699bd3910b75b3fca87d69010e55f31c` 的
`test/_inductor/test_nolinear_fallback_fusion.py` 查看，保留了强制 fallback、
分组记录以及 eager/compiled 前向与梯度比较。

## 当前处理与后续定位边界

按用户要求，从 PR 删除该设备 UT，保留 10 个原有逻辑 UT，
继续覆盖 no-linear 融合兼容性、分组、buffer 生存期和 benchmark 状态恢复。
没有修改 `make_reduction`、`mean`、公共 lowering 替换逻辑或设备编译实现。
删除用例不代表修复了接口问题，也减少了 PR 中端到端动态训练的自动回归覆盖。
此前 Python 3.11 环境中的设备 PASS 仅作为历史验证记录，不能替代此次流水线环境验证。

后续先采集失败环境的 PyTorch commit、两个 `make_reduction` 的实际签名及 `sum_` 源码，
确认配套版本和参数引入点；再检查 `strict_reduction` 在 IR、归约顺序与 NPU 路径中的语义。
不要仅通过 `**kwargs` 吞掉参数或跳过失败来宣称修复。
