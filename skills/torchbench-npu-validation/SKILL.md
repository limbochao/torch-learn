---
name: torchbench-npu-validation
description: >-
  Run and interpret TorchBench runner model validation on NPU, including torchbench,
  huggingface and timm_models suites, eager controls, training support, accuracy and
  performance, and static or dynamic shapes. Use for model-level testing and missing
  default batch size diagnosis; single-kernel replay and codegen repair are separate workflows.
---

# TorchBench NPU Validation

用于在指定 NPU 环境验证模型整网，并把结论限定到实际执行的场景和代码版本。
使用当前仓库的 runner 和模型定义作为依据；不同版本的参数、模型支持范围和状态字符串可能不同。

## 确认环境与场景

先确定 suite、模型、training/inference、accuracy/performance、static/dynamic、dtype、batch size、
sequence length 或图像尺寸，以及目标 backend。用户指定几个模型定界时，不自动扩展到其他模型。
“全部场景”需要显式列出适用的场景组合；不支持的训练场景单独说明原因。

运行前确认：

- 容器、工作目录、Python 可执行文件、CANN 环境和空闲设备符合当前任务约定。
- `torch`、`torch_npu`、Triton、模型依赖版本及实际模块 `__file__`。
- runner、模型定义和修复代码的来源提交；使用 overlay 时记录加载路径及关键文件校验值。
- dataset、模型缓存和下载端点可用。按缺失依赖补齐，不因测试一个模型重装整套环境。

使用 NPU_A5 的 TorchBench 2.13 环境时，读取
[references/npu-a5.md](references/npu-a5.md)。其中路径为已知配置，执行前仍需核实。
升级后的环境不能直接复用旧环境的通过结论。

## 缺少默认 batch size 或不支持 training

先检查当前 suite 的模型定义、模型类的训练 batch size、训练入口和 runner 的 batch size 解析逻辑。
区分模型未实现训练、runner 漏接支持信息、依赖版本不匹配和普通配置缺失。
类中训练 batch size 为 `None` 是证据线索，不能单凭这一项断言模型只支持推理。

做 eager 对照时保持模型、training、performance/accuracy、dtype、shape 和输入设置一致：
从原命令去掉 `--backend ...`、`--npu-backend ...`，加 `--execution-mode eager`，
并确认当前 runner 真正绕过编译入口。不要擅自换成 `--backend eager` 代替用户要求的执行模式。
若 eager 同样在加载模型或选 batch size 时失败，优先定位公共 runner 或模型定义，
而不是把它记为 Inductor 编译缺陷。

显式指定 batch size 只能作为对照，不能把未实现的训练入口变成受支持场景。
建议修复时指出应修改模型支持声明、runner 校验/跳过逻辑还是 backend，并给出源码依据。
社区 CUDA 是否支持训练，应检查对应版本源码或实际运行；NPU 现象不能代替 CUDA 结论。

## 执行与持续监控

为每次运行使用独立输出目录，保存完整命令、环境摘要、日志、CSV 和 debug trace。
长任务使用会在 SSH 断开后继续运行的方式，并持久记录 PID、启动时间、运行状态和退出码。
SSH 失联只表示监控中断；恢复后先核对原进程，避免重复启动同一任务。

OOM 时可按需降低编译或预编译线程，先确认该版本实际读取的配置项，记录调整前后的值。
保留原 shape、精度阈值和运算路径；缩小输入得到的结果另列，不覆盖原场景结论。
单候选超时与整网累计耗时分别统计，采用本次任务约定的阈值。
候选不断切换时，整网运行较久不等于某个候选卡住。

## 判读结果

终态结合进程退出码、目标模型的 CSV 行和完整日志判断，不能只看退出码为 0 或某一条报错。

| 结果 | 应如何报告 |
| --- | --- |
| 依赖、数据、默认 batch size 或支持声明失败 | 环境/模型/runner 阶段失败，尚未完成编译或精度验证 |
| eager 两次结果不同 | eager 基准不稳定，不能直接归因为 Inductor 精度 |
| 编译或运行失败 | 精度检查未完成，保留失败阶段与 traceback |
| 部分候选编译失败，autotune 选出可用配置 | 记录候选异常及最终模型结果，不自动判整网失败 |
| 精度比较失败 | 同输入检查输出、loss、梯度、更新后参数，定位首个不一致项 |
| 目标模型明确通过 accuracy | 报告本次场景通过；保留 dtype、shape、阈值及实际代码版本 |

训练精度需核对 runner 实际收集哪些结果、优化器和迭代次数。
仅 loss 一致不能证明梯度和更新后参数一致；不要默认所有 runner 都会比较这些项目。
保留既有 RNG 重置、模型克隆及两次 eager 检查，不通过放宽阈值或跳过比较换取 PASS。
NPU 不支持 double 时，不把 dtype 回退后的 reference 描述为真实 FP64 基准。
`tol=0.01` 的含义由当前比较函数决定，不等于逐元素相对误差均小于 1%。

`--dynamic-shapes` 只说明启用了动态编译模式。若输入尺寸始终相同，明确注明未覆盖多尺寸切换。
需要验证尺寸变化时，让同一个 compiled callable 顺序执行指定形状，并保留 backward、数值和重编译证据。

## 交付

报告模型与场景、有效环境/修复版本、完整命令、最终状态、日志/CSV 路径和验证范围。
performance 结果区分首次编译耗时与稳定迭代耗时，不将 accuracy 运行的计时直接当正式性能结论。
区分本次实测、用户此前在其他机器的结果和源码推断。
后续需修复时，再进入单 kernel 编译重放或 codegen 定位流程。
