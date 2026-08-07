---
title: NPU FX Graph Runnable 通用适配
---

# NPU FX Graph Runnable 通用适配

## 1. 适用范围

本文说明如何把 Inductor debug trace 中的 `fx_graph_runnable.py` 改造成可在 NPU 上独立运行、支持 static/dynamic/group 对照和 profile 的复现脚本。流程适用于模型级 FX graph，不绑定某个具体模型。

目标是复现目标 graph 的编译与 codegen 行为。若用占位实现替换了原环境的自定义算子，产物只能用于验证剩余 graph 的编译链路，不能直接代表原整网性能。

## 2. 起点选择

优先使用同版本、同设备生成的 NPU runnable。CUDA runnable 的 decomposition、融合边界、自定义 kernel 和输入 layout 可能与 NPU 不同，不应作为 NPU graph 的默认基线。

保留原始 runnable，只在副本中完成设备、输入和自定义算子适配。修改前先保存来源版本、graph 名称和原始 input count。

## 3. 环境变量和导入顺序

影响 torch_npu 初始化的开关必须在 `import torch` / `import torch_npu` 前设置，例如：

```python
import os

if os.getenv("EXECUTION") == "group":
    os.environ["INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"] = "1"

import torch
import torch_npu
```

脚本应显式提供 `static`、`dynamic`、`group` 模式。group 本质仍走 dynamic compile，只额外在导入前开启 group 环境变量。

## 4. 输入构造与符号化

`load_args` 应使用具体整数构造 tensor。不要把 `s0` 之类未绑定符号直接留在 Python shape 表达式中。输入创建完成后，再对共享同一符号的所有 tensor 维度调用 `torch._dynamo.mark_dynamic`：

```python
args = load_args(bs=runtime_bs)
for tensor, dim in marked_dims:
    torch._dynamo.mark_dynamic(tensor, dim, min=lower, max=upper)
compiled = torch.compile(module, backend="inductor", dynamic=None)
```

`dynamic=None + mark_dynamic` 让 Dynamo 从具体输入建立符号及 guard，也能精确控制哪些维度共享符号。static 模式不调用 `mark_dynamic`，并使用 `dynamic=False`。

多个 runtime shape 连跑时，首 shape 负责首次编译，后续 shape 用新的具体 tensor 调用同一个 compiled function。每轮都要重新建立所有受 batch size 影响的输入，不能只替换部分 tensor。

## 5. 自定义算子

先扫描 runnable 中所有非 `aten`/`prims` namespace，以及 higher-order Triton wrapper。按以下优先级处理：

1. 目标环境存在原实现：导入并注册原实现。
2. generated `output_code.py` 中存在可移植实现：按 NPU 支持的 API 还原，并验证 schema、shape、stride、dtype 和 format。
3. 无原环境且该算子不属于待测路径：注册同名、确定性的占位实现。

占位自定义算子至少需要一致的 schema、NPU implementation 和 fake/meta implementation。fake/meta 输出 shape 必须与真实实现一致；依赖数据决定 shape 的算子不能用固定 fake size 伪造，否则 AOTAutograd 生成的 size assertion 会在运行时失败。

用 `torch.library.opcheck` 检查 schema、fake tensor 和实现一致性。对 NPU format 敏感的 concat/copy，先统一 format 或实现等价的逐段写入，不能假设 `torch.cat` 接受不同 format 的输入。

占位值应确定性生成，避免随机结果让后续 shape/data-dependent path 不稳定。与此同时要明确：占位实现改变了计算和融合边界，不能用于原模型的端到端性能结论。

## 6. Higher-order Triton 调用

旧 runnable 中的 `torch.ops.higher_order.triton_kernel_wrapper_functional` 可能不被目标 Dynamo/torch_npu 版本支持。若必须保留自定义 Triton kernel，可以用 graph 可调用的普通 Python wrapper 包装 launch，并把实际 Triton runtime 调用放到 Dynamo 不继续 trace 的边界内。

该处理只绕开 unsupported higher-order operator；外层 module 仍通过 `torch.compile` 进入 Dynamo。wrapper 的返回 tensor、mutation 和 alias 语义必须与原 higher-order op 一致。

## 7. 运行与产物

建议每次运行记录：

- execution、首 shape、runtime shape sequence 和 group 开关。
- profiler 的 device kernel 明细和 step time。
- 本轮生成的 `output_code.py` 及相关 torchinductor Python/DSL 文件。
- 输入符号到 tensor/dim 的映射。

保留 `/tmp/torchinductor_root` 作为默认 cache 目录，便于发生编译或 runtime 错误后直接查看生成文件。是否清理由运行者在测试前决定；汇总脚本只复制与本轮相关的输出文件，不复制 autotune profiler 临时目录和整个 cache 树。

## 8. 验证顺序

1. eager 运行通过，自定义算子 shape/dtype/format 合法。
2. static compile 单 shape 通过并生成 `output_code.py`。
3. dynamic 首 shape 通过，后续 shape 复用 compiled function。
4. group 模式的 generated metadata 中出现 `group_enabled=True`。
5. profile 只覆盖 warmup 后的正式迭代，不混入 compile/autotune。

若只完成占位算子链路验证，应把结论写成“runnable 可编译/可执行”，不能写成“与原模型等价”。
