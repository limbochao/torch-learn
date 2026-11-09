---
name: inductor-eager-repro-extract
description: >-
  Extract an independent eager ATen subgraph from an Inductor generated kernel wrapper
  for CUDA/NPU codegen comparison, preserving input layouts and actual outputs. Use when
  asked for eager forward or portable standalone graph reproducers; fixed-config Triton
  compile/launch replay uses a different workflow.
---

# Inductor Eager Repro Extract

交付可单文件运行的 ATen 子图，用同一计算在不同设备生成 DSL，检查 NPU 修复与 GPU 计算语义是否一致。
先确认交付目标：eager 子图用于重新 codegen；固定配置 Triton 文件用于重放既有 kernel。
两者不能互相代替，也不要求一个 ATen 子图在不同后端融合成相同数量的 kernel。

## 从原始 wrapper 抽取

在可用的 `torch-learn` 仓库定位 `scripts/tools/extract_triton_kernel.py`，先查看参数实现或 `--help`。
输入使用同时包含目标定义、Graph fragment metadata 和实际 `.run(...)` 的原始 `output_code.py`。
不要用仅保留 Triton 函数体的文件冒充完整 wrapper。

```bash
python <torch-learn>/scripts/tools/extract_triton_kernel.py \
  <output_code.py> <kernel_name> --only-eager -o <eager_case.py>
```

只有动态维度无法从原 wrapper 唯一解析时才补 `--symbol-values`，值来自实际运行或 benchmark 输入。
生成的 `CASE` 是可导入描述，不代表执行文件已经实际运行过 eager。
将它整理为带入口、输入构造和检查的独立文件；不要把原 wrapper 的绝对路径或安装目录作为运行依赖。
独立输入与 CUDA 导出方式见 [references/extraction-and-comparison.md](references/extraction-and-comparison.md)。

找不到工具时先在工作区搜索；若只有原始图，依据实际输入、节点和输出重建最小子图并明确标注来源。
没有完整图证据时，指出缺失材料，不声称已经抽取出原 kernel 的等价计算。

## 校验子图边界

逐项对齐 Graph fragment、FX graph、wrapper 调用和实际 load/store：

- 输入以子图数据依赖为准；图内已计算的值不能再作为重复 placeholder 输入。
- tuple 输出补全必要的 `getitem`；未定义节点先回到原图确认来源。
- 只返回目标 kernel 实际写出的结果。为便于观察而额外返回 max、sum 等中间值可能破坏 fusion。
- metadata 混入相邻 kernel 的计算时，仅保留有来源证据的目标依赖闭包，记录删改依据。
- 保留算子分解、加法/归约次序、常量、eps、reshape/permute 和边界选项。
  不用更简洁的高级算子替换原 ATen 分解，以免改变 codegen 路径。
- `aten.*_backward` 子图可作为函数直接执行；不要因函数名为 eager forward 再加一次 backward。
  若问题在训练调度阶段，则保留真正的 forward、loss、backward 流程。

保存原始 metadata 和修正后版本，说明每项修正；原始 generated DSL 保持原格式。
同名 kernel 多次调用时，确认抽取工具所选调用代表目标失败输入，不能只按名称相同认定等价。

## 输入构造

固定 seed，保留 shape、stride、dtype、标量和必要的 storage offset。
可用 `torch.empty_strided` 构造布局后填值；存在跨输入别名或重叠时须显式重建共享 storage。
单独创建几个相同 stride 的 Tensor 不会保留别名关系。

输入值需满足运算条件并覆盖原问题：索引保留合法负端点和正值，token id 不超词表，
bool selector 包含两个分支，rstd 等具有约束的量取合法值。
自动生成的整数全零输入不能证明负索引路径正确。
数据相关故障应使用保留的真实输入或已证实能触发的值，随机 metadata 重建不保证复现。

明确标注随机合成输入与真实捕获值。缩小 batch/shape 或删除分支后重新验证触发条件，
尤其是 load liveness、资源不足和符号 group 问题，不能把缩小后通过当作原场景通过。

## 独立运行与对照

单文件包含目标函数、确定性输入、设备选择、直接执行入口，以及按需开启的 compile/eager 比较。
NPU 注册只在选择 NPU 时导入，CUDA 路径不依赖 torch_npu 或 NPU 专有自定义算子。
打印输入 shape/stride/dtype、静态/动态模式和结果；按算子需要检查输出及梯度，阈值显式记录。
有随机数、in-place 或状态更新时，比较前重置 RNG、克隆输入并保留别名，避免后一次读取已修改输入。

多 shape 故障用同一个 compiled callable 顺序执行原来两组尺寸，不为每组重建 compiled function。
导出后复制文件到独立目录实际运行，确认不依赖原 capture、辅助 Python 文件或原图缓存。
CPU eager 用于验证脚本和输入；CPU compile、CUDA compile、NPU compile 的结果分别记录。

对照 GPU/NPU 时，将 preserved/reduction axes 映射回真实维度，检查 mask、访问集合、单位元和输出。
不要求变量名、tile 数值或 load 行位置一致；GPU 与 NPU 的资源模型不同。
若原故障发生在 scheduling、Layout 符号证明或 range 检查，可能尚未生成 DSL：
保留对应 IR/表达式和派生小图，明确最终 GPU DSL 只展示成功路径。
CPU 通过、整网通过或 GPU DSL 看起来相似，均不能单独证明原内部缺陷已修复。

## 交付

给出独立脚本、原模型/场景/提交、原 wrapper/metadata 来源、边界修正、输入来源、运行命令和设备实测范围。
为 GPU 对照指出具体检查项；多个问题若共用同一小图，保留同源关系，不称为多个不同模型 kernel。
