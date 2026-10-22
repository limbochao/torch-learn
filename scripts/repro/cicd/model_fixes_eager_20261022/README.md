# 八类历史问题的独立 eager forward

用于在 GPU 上重新编译相同 ATen 图，查看社区 Inductor 生成的 Triton DSL，
与[八类 NPU 修复分析](../../../../docs/issue-logs/cicd/model-fix-dsl-explained-20261021.md)对照。
每个 `.py` 都包含 forward、输入构造和命令行入口，可单独复制运行，不需要模型、权重或同目录脚本。
CUDA 路径只需要 CUDA 版 PyTorch 及其兼容的 Triton；只有指定 `--device npu` 才导入 `torch_npu`。

## 问题、模型与入口

层级沿用历史定位记录，表示前一个阻塞解决后实际暴露的下一个问题。

| 问题 | 涉及模型 | 独立入口 | 历史层级 |
| --- | --- | --- | --- |
| reduction mask | GPT2ForSequenceClassification、BERT_pytorch | [01](01_reduction_mask.py) | 第一层 |
| reduction axes | BERT_pytorch | [02](02_reduction_axes.py) | 第二层 |
| negative index | 独立负索引回归，未证实模型触发 | [03](03_negative_index.py) | 不分配模型层级 |
| load liveness | AlbertForMaskedLM | [04](04_load_liveness.py) | 第一层，与 tiling 共同解决 |
| tiling resource | nfnet_l0、AlbertForMaskedLM | [05a](05a_tiling_pooling.py)、[05b](05b_tiling_softmax.py) | 见下文 |
| scheduler symbolic groups | nfnet_l0 | [06](06_scheduler_symbolic_groups.py) | 第二层 |
| symbolic strides | NFNet 派生小图 | [07](07_layout_symbolic_strides.py) | 第三层，仅小图 |
| range split | 同一 NFNet 派生小图 | [08](08_codegen_range_split.py) | 第四层，仅小图 |

共八类问题、九份入口。01 使用实际捕获的 GPT2 图代表 GPT2/BERT 共有的 mask 问题，
没有额外声称抽出了 BERT 的第二份 embedding 图。
05a 对应 NFNet 第一层；tiling 修复还覆盖 Albert 第一层，05b 单独覆盖 Albert 动态 softmax 第二层。

## 在 GPU 上运行

下面从本目录执行；单独复制文件后也使用相同参数。默认只执行 eager：

```bash
python 01_reduction_mask.py --device cuda
```

eager 自身不会生成 Triton DSL。加 `--compile` 后，用同一组输入执行社区 Inductor，
并比较 eager/compiled 的数值结果。开启 trace，保留生成代码和编译前后 IR：

```bash
case=01_reduction_mask
mkdir -p "out/$case"
TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCH_COMPILE_DEBUG=1 \
TORCH_COMPILE_DEBUG_DIR="$PWD/out/$case" \
TORCHINDUCTOR_CACHE_DIR="$PWD/out/$case/cache" \
python "$case.py" --device cuda --compile 2>&1 | tee "out/$case/run.log"
```

脚本打印 torch/CUDA/Triton 版本、GPU 型号和输入 shape/stride/dtype。
首次编译后，在 `out/<case>` 下查看 `output_code.py`、`fx_graph_readable.py`、
`ir_pre_fusion.txt`、`ir_post_fusion.txt`；06–08 还需查看 backward 的目录。
不同 PyTorch 版本的 trace 内容可能不同；缓存命中时换一个新的 cache 目录再导出。
原始 NPU 图来自 2.13 开发环境；GPU 对比请记录实际版本，版本差异也会影响 lowering 和 fusion。

批量运行全部入口：

```bash
set -o pipefail
for script in [0-9]*.py; do
    case=${script%.py}
    mkdir -p "out/$case"
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    TORCH_COMPILE_DEBUG=1 \
    TORCH_COMPILE_DEBUG_DIR="$PWD/out/$case" \
    TORCHINDUCTOR_CACHE_DIR="$PWD/out/$case/cache" \
    python "$script" --device cuda --compile 2>&1 | tee "out/$case/run.log" || break
done
```

04 保留 81 个输入、57 个大矩阵，共约 1.78 GiB FP32 矩阵输入，另有中间结果和编译开销。
编译内存不足时可继续降低并发；不要首先删掉分支，以免丢失原来的 load liveness 场景。
05a/05b 支持缩小输入，例如 `--batch 2 --height 224`、`--batch 1 --sequence 512`；
缩小后的结果只代表该 shape。

## 抽取方法与输入边界

01–05b 使用仓库的 [extract_triton_kernel.py](../../../tools/extract_triton_kernel.py)
`--only-eager` 流程，从原始 `output_code.py` 的 `Graph fragment` 生成 forward。
原始 metadata、补齐后的 metadata 和来源校验值保存在
[抽取证据目录](../../../../artifacts/cicd/model_fixes_eager_20261022/)。
没有改动抽取工具；各入口只增加了独立输入构造、设备选择和可选编译检查。

通用命令如下，`output_code.py` 必须是包含目标 kernel 定义和 `.run` 的原始 wrapper：

```bash
python scripts/tools/extract_triton_kernel.py output_code.py KERNEL_NAME \
    --only-eager --symbol-values '{"s21":128,"s65":288}' -o eager_case.py
```

只有存在动态维度时才传 `--symbol-values`，键名按具体图选择。
工具直接输出可导入的 `CASE`；这里再将其转为可以直接执行的单文件。
输入使用 `torch.empty_strided`，保留抽取 metadata 的 shape、stride、dtype。
张量数据由固定 seed 生成，不是原模型权重或运行时捕获值。

原始 metadata 不能直接等同于一个完整可运行的 kernel 子图，本次做了以下明确整理：

- 删除被图内计算覆盖的重复 placeholder，以及目标 kernel 不输出的中间结果。
  输出范围按原 `.run` 的实际 store 确定，不让导出的中间值破坏目标 fusion。
- 01 缺少 LayerNorm tuple 的 `getitem`，改为返回其第 0 项；同时返回残差和，
  对应原 kernel 写回的两个张量。iota 的 `npu:0` 改为跟随索引输入的 device。
- 05a 的原片段混入相邻卷积的 weight 标准化等节点，且以未定义的 `buf276` 返回。
  按实际 kernel 的单输入、单输出边界，仅保留原 `avg_pool2d` 节点。
- 05b 补上缺失的两个 shape 节点，分别读取 batch 和 sequence；保留双 max、条件选择、
  exp/sum/div 的原始 ATen 计算，没有替换成一个 `softmax` 调用。
- 03 显式使用 `[-8,-3,-1,0,4,7]`，避免自动生成的全零整数输入漏测负索引；
  01 的 token id 限制在合法词表范围；05b 的 selector 同时包含 true/false。
- 04 保留 12 路梯度的实际加法顺序及 81 个输入；rstd 构造为正值。

06–08 的原错误发生在 DSL 生成前，使用历史已验证的
[symbolic_group_training.py](../model_fixes_20261020/symbolic_group_training.py)提取其 forward。
三个文件各自完整包含该图，避免下载单个文件后还要寻找公共依赖。
它们是相同小图在不同编译阶段暴露的问题，不能把它们说成三个不同的模型 kernel。

## 每份用例需要比较什么

**01：embedding + residual + LayerNorm。** 输入隐藏维为 768，保留词表 `[50257,768]`、
序列长度 1024、eps `1e-5`。查看 GPU 对 768 扩展为二次幂 tile 后，
如何在计算均值和方差前屏蔽非法 lane（不属于原张量的 tile 元素）。
NPU 修复对应每次 sum 前选择单位元，不要求 GPU 生成相同的临时变量或 fusion 边界。

**02：训练反向的 bias gradient。** 原图输入 `[48,128,64]`，
经过 `[4,12,128,64] -> permute(0,2,1,3) -> [512,768]`，返回重排结果和 `[1,768]` 的 sum。
查看 GPU 是否保留 12×64 个 channel，只规约 batch/sequence。
这是把原反向 ATen 子图作为可直接调用的 forward 执行，不需要再做一次 backward。

**03：负索引。** table `[8,16]`，索引含 `-8`、`-1` 两个合法端点。
查看 GPU 的负索引回绕和边界检查，与 NPU 修复后的 wrap 语义对齐。
这一项来自独立回归，不能归到某个已验证模型的故障层级。

**04：Albert LayerNorm 参数梯度。** 原静态训练 performance 的 reduction 为 2048×4096，
原前向中 batch=4、sequence=512。返回 weight/bias 的两个 `[4096]` 梯度。
它执行的是已经分解的反向 ATen 图，因此直接调用即可生成对应计算。
查看 GPU 的多路归约、fusion 拆分和 load/consume 安排；NPU 修复主要缩短同时存活的 load 值的范围。
GPU 与 NPU 的资源模型不同，不能单凭一行 load 的位置判定修复等价。

**05a：NFNet pooling。** 默认 `s21=128,s65=288`，来自原 wrapper 的 benchmark 输入；
kernel 输入 `[128,256,72,72]`、输出 `[128,256,36,36]`。
保留 `avg_pool2d(kernel=2,stride=2,ceil_mode=True,count_include_pad=False)`。
比较四点均值的访存与 tiling 选择；NPU 修复恢复合法候选搜索，算子公式不变。

**05b：Albert softmax。** 默认 `s87=4,s40=512`，保留 64 个 head、除数 8、
bool selector，按实际 store 仅返回 softmax 概率。查看 GPU 如何保留完整的最后一维规约并划分非规约轴。
NPU 的 X2/R3 tile 数值是 NPU 容量约束下的选择，不要求 GPU 采用相同数值。

**06–08：双 shape 训练。** 同一个 compiled function 依次执行 `(batch,H)=(2,224),(3,256)`；
输入 channel=16，1×1 卷积 stride=32，随后空间均值、sigmoid、逐点乘，并比较输入/权重梯度。
默认动态 shape；`--no-dynamic` 可作静态对照。必须执行 backward，第二组 shape 也必须保留。

- 06：检查含 `q=floor((H-1)/32)` 的符号 group 是否能识别代数等价表达式；重点是 scheduler IR。
- 07：检查第二组 shape 的布局/整除证明是否保留合法整数表达式；原来在 DSL 前失败。
- 08：检查不能拆分的 fusion 候选是否被合法拒绝；原实现可能捕获异常后继续，
  因而最终数值 PASS 不证明该内部问题已消失，需同时查看编译日志和候选处理源码。

06–08 的 GPU DSL 只反映最终成功路径。要比较这三项修复，应同时检查 IR 和社区相应源码；
它们没有一份“修复前失败 DSL”可逐行比较。历史下层问题若被上层修复屏蔽，仍沿用
“未修复、影响被屏蔽、不阻塞当前模型验证”的标记，不因这些用例 PASS 就改成已修复。

## 验证范围

本地没有 GPU/NPU。本轮使用隔离的 `torch 2.9.1+cpu`，验证单文件 eager 可运行，
并做了部分 CPU Inductor 数值检查；结果见
[cpu_validation.json](../../../../artifacts/cicd/model_fixes_eager_20261022/cpu_validation.json)。
CPU 结果用于检查脚本和输入构造，不作为 CUDA DSL、NPU 修复或模型整网通过的证明。
GPU 对比结论待收集真实 CUDA 生成代码后补充。
