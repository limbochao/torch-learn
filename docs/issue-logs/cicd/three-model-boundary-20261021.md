---
title: MobileNet、fastNLP BERT、ConvNeXt 全量修复分支问题定界
published: false
---

# MobileNet、fastNLP BERT、ConvNeXt 全量修复分支问题定界

## 范围与验证基线

仅针对以下三个模型；用户所指的复测基线来自其他机器上的先前测试。
本轮不修改框架代码，也不修改已有修复分支。

| 模型全名 | 套件 | 指定场景 |
| --- | --- | --- |
| mobilenetv2_100 | timm_models | 动态训练；若未复现问题，补齐全部 8 项 |
| fastNLP_Bert | torchbench | 动态推理、静态训练 accuracy |
| convnextv2_nano.fcmae_ft_in22k_in1k | timm_models | 全部 8 项 |

8 项指静态/动态 shape × inference/training × performance/accuracy。
环境为 NPU_A5 的 TorchBench 2.13 专用环境，FP32、关闭 ACLgraph、默认 batch size。
`TORCHINDUCTOR_NPU_BACKEND=default`，命令行 `--backend inductor --npu-backend triton`。
`TORCHINDUCTOR_COMPILE_THREADS=1`，最初的 performance 任务使用 `TORCHNPU_PRECOMPILE_THREADS=4`，
后续启动的任务降为 1；单候选编译超时 60 分钟。

框架修复基线为 `debug/cicd-all-fixes-20261020`，commit
`5b18fac65380a14a404251de2eb6b89072c60911`。独立 overlay 使用该 commit 的完整 `_inductor`
Python 目录及 `utils/_dynamo.py`；执行前 184 个文件 SHA256 全部一致。
原安装包、框架工作树和修复分支未改动。native 库沿用专用环境的既有构建。
实际导入的 benchmark `common.py`、`torchbench.py` 与该 commit 的文件哈希相同。

版本：torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261019`、Triton `3.6.0`、CANN `9.1.0`、
timm `1.0.30`、transformers `4.36.0`、fastNLP `0.6.0`；device `Ascend950PR_9579`。

## 验证结果

矩阵仍在执行，最终结果待补齐。运行中的场景不记为通过，也不因整个模型编译较久记为单候选超时。
performance 通过要求进程正常退出、CSV 模型名正确且 speedup 大于 0；这不代表精度通过。
accuracy 通过要求正常退出并产生 `pass_accuracy`。

| 模型 | 场景 | 当前结果 | batch size |
| --- | --- | --- | --- |
| mobilenetv2_100 | 动态训练 performance | FAIL：反向 kernel 的 FP64 标量转换编译失败 | 128 |
| mobilenetv2_100 | 动态训练 accuracy | FAIL：同一反向 kernel 编译失败，`fail_to_run` | 8 |
| fastNLP_Bert | 动态推理 performance | FAIL：长度标量错误，后续申请约 3 TiB HBM | 1 |
| fastNLP_Bert | 动态推理 accuracy | PASS | 4 |
| fastNLP_Bert | 静态训练 accuracy | PASS | 4 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 静态推理 accuracy | FAIL：HF32 开启时整网输出未满足 `tol=0.001`；关闭 HF32 的诊断对照通过 | 8 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 静态训练 accuracy | FAIL：原生 GELU 的 channels-last 输出分配不支持，`fail_to_run` | 8 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 静态推理 performance | PASS；speedup `1.112756` | 128 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 静态训练 performance | RUNNING | 128 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 动态推理 performance/accuracy | RUNNING | 128/8 |
| convnextv2_nano.fcmae_ft_in22k_in1k | 动态训练 performance/accuracy | RUNNING | 128/8 |

## 问题层级与归属

层级只表示解决前一阻塞之后实际暴露的下一阻塞，不表示组件栈的上下位置。
本轮没有实施修复，因此不能把并存发现直接排列为第一、第二层。

| 涉及模型、场景 | 故障层级 | 问题 | 归属 | 当前阻塞性 |
| --- | --- | --- | --- | --- |
| mobilenetv2_100，动态训练 performance/accuracy | 第一层实际阻塞 | hardtanh 边界被生成为 FP64 标量参数，触发不受支持的 SIMT 转换 | Inductor NPU 参数类型适配；下层 BiSheng 同时存在不良错误处理 | 阻塞两个场景，未修复 |
| fastNLP_Bert，动态推理 performance | 第一层实际阻塞 | 最大 word-piece 长度的输出未写入，读取未初始化标量后异常分配 HBM | Inductor NPU codegen | 阻塞该场景，未修复 |
| fastNLP_Bert，同一长度图的 template 编译候选 | 并存发现，不续编号 | int64 间接索引的 `tensor<1xi64>` 转换失败 | Triton-Ascend 的 TritonToLinalg | 该候选失败后整网继续；不是最终 HBM 异常原因 |
| convnextv2_nano.fcmae_ft_in22k_in1k，静态训练 accuracy | 第一层实际阻塞 | GELU 接收到 channels-last tensor，原生输出分配报错 | torch_npu 原生 tensor factory 的布局限制；Inductor fallback 的输入布局适配也需检查 | 阻塞该场景，未修复 |
| convnextv2_nano.fcmae_ft_in22k_in1k，静态推理 accuracy | 该场景首个精度失败 | FP32 舍入差异在 HF32 卷积下被放大，超过整网容差 | 当前证据指向 HF32 精度配置与验证标准；未发现生成 kernel 的独立计算错误 | 原配置仍 FAIL；对照通过不覆盖原结果 |
| convnextv2_nano.fcmae_ft_in22k_in1k，动态推理 performance 的 template 候选 | 并存发现，不续编号 | `PlanMemory Traverse IR Failed` | BiShengIR 内存规划路径 | 候选失败后搜索继续，当前非终止错误 |

候选编译问题应标记为“候选失败已回退，当前非阻塞”，不能根据回退将其写成“修复后缺陷影响被屏蔽”。
动态 accuracy 的 batch 4 通过，也不能证明 batch 1 的最大值输出缺陷消失。

## mobilenetv2_100：FP64 标量参数未适配，不能仅归给下层

两个动态训练场景都在反向图编译失败。AOTAutograd 曾提示 eager backward compile 失败、暂时抑制异常；
实际训练需要 backward，后续重新编译仍失败，最终 performance 非零退出、accuracy 为 `fail_to_run`。
因此这里没有因抑制异常而整网通过，也不标为“不阻塞模型验证”。

### 固定候选与进一步缩小

失败 kernel 为
`triton_red_fused__native_batch_norm_legit_functional_div_expand_hardtanh_backward_native_batch_norm_backward_view_1`。
所有 tensor 参数均为 FP32，但 hardtanh 的两个边界参数 `in_ptr1`、`in_ptr2` 的 signature 是 `fp64`。
函数体将它们通过 `tmp1.to(tl.float32)`、`tmp4.to(tl.float32)` 转回 FP32，再与激活值比较。

skill 捕获 record 0 的配置为 `XBLOCK=64`、`R0_BLOCK=32`、`simt_only`、
`enable_auto_blockify=True`、8 warps、2 stages、`debug=False`，target `npu/Ascend950PR_9579/32`。
完整 kernel 在独立脚本中只执行一次 `triton.compile`，脱离 Inductor 与 overlay 后仍报：

```text
*** Bad machine code: Expected a register operand. ***
CALL_TES &__truncdfsf2
fatal error: error in backend: Found 2 machine code errors.
Failed to compile BiShengLIR for device
```

再缩成“将一个运行期 FP64 标量转成 FP32 并写出 64 个元素”的手写 kernel，仍复现同一错误。
仅把该标量的 signature 改成 `fp32`，相同 target 和 options 编译通过。
这证明下层转换路径可独立触发；**单凭这点不能排除 Inductor 生成了设备不支持的类型**。

另有不依赖模型的 `mobilenet_hardtanh_graph.py`：FP32 `grad/value[16,64]`、
边界 `0.0/6.0`，只编译 `hardtanh_backward(...).sum(dim=0)`，`dynamic=True`、`fullgraph=True`。
它同样生成 FP64 边界参数，最终在链接时报告 `undefined symbol: __truncdfsf2`，exit 1。
因此现象不依赖 MobileNet 的 BatchNorm 融合、权重、优化器或整网输入。
小图与完整 kernel 的最终错误文本不同，但都指向 FP64 到 FP32 的同一运行期转换。

### 参数类型生成点

小图的生成代码明确记录两个输入为 `Tensor "f64[][]cpu"`，wrapper 通过
`arg4_1.item()`、`arg5_1.item()` 把它们传给 kernel。
模型原 wrapper 同样将 `primals_384/385` 标为 CPU FP64 零维 tensor，
并通过 `primals_384.item()`、`primals_385.item()` 传入该反向 kernel 的第二、第三个参数。
`GraphLowering.is_unspec_arg()` 将输入中的 CPU 零维 tensor 识别为需要解包的标量；
这里 unspec 指保留为运行期参数、没有按某个 Python 常量值专门生成代码。

实际安装的社区 `torch/_inductor/codegen/triton_utils.py::signature_of()` 对 `TensorArg` 执行：

```python
typ = _type_of(arg.dtype)
if should_unwrap_unspec_arg(arg.buffer):
    new_typ = typ.lstrip("*")
    if new_typ in ["fp16", "bf16"]:
        return "fp32"
    return new_typ
```

因此 CPU FP64 tensor 的 `*fp64` 去掉指针标记后仍是 `fp64`，没有 NPU 类型适配。
`mobilenet_scalar_signature.py` 用 `TensorArg(dtype=torch.float64)` 直接复现该分支，
不加载模型、不分配设备 tensor、不进入编译器：

```text
UNWRAPPED_CPU_SCALAR_SIGNATURE fp64
AssertionError: CPU FP64 scalar is passed unchanged to an NPU kernel
```

**`_use_fp64_for_unbacked_floats=False` 不能修复本例。**
另一次小图配置对照仍生成 FP64 参数并报 `__truncdfsf2`、exit 1：该配置只控制 `SizeArg` 的分支，
不控制当前 `TensorArg` 解包分支。不能把对 Python float/SizeArg 的 API 对照误记成模型根因。

整合分支 `torch_npu/_inductor/codegen/triton.py::NPUTritonPrinter` 第 261 行附近已经将符号表达式里的
`tl.float64` 改为 `tl.float32`，但 `NPUTritonKernel.codegen_kernel()` 调用父类生成参数签名，
这条路径没有相应的 NPU scalar 类型适配。打印表达式的修复无法改变 signature 中的 `fp64`。
另一套 `triton_experimental` 的显式 downcast 代码不在本轮执行路径上，也不能作为 default 路径已覆盖的证据。
小图使用 `NPUIndexTritonKernel`，其 `signature_to_meta()` 也调用同一社区 `signature_of()`；
因此两套 default kernel 生成入口都需要覆盖这个输入边界。
已有 `lowering.py::_convert_element_type()` 的 FP64 fallback 仅处理显式 `prims.convert_element_type`，
当前 FX 的 `le.Tensor/ge.Tensor` 直接消费 CPU 标量，内部类型提升在生成 kernel 内留下 `.to(tl.float32)`，
并未经过上述显式转换算子的 fallback。

Ascend C 的 [SIMT 标量类型列表](https://asc.gitcode.com/guide/programming_guide/language_extension/simt_builtin_keywords.html)
列出 float 等类型，没有 double；结合当前 A5 固定候选的 FP64/FP32 编译对照，
应在 **Inductor 的 NPU 零维 CPU tensor 解包/标量签名生成边界**处理这类浮点参数，而不是把模型改成不训练。
同时应由 BiSheng 对不支持的转换给出明确诊断，避免产生非法机器指令。
这里没有声称所有 NPU 操作都不支持 FP64，也没有将 HCCL 的 FP64 通信支持等同于 SIMT 浮点计算支持。

本轮没有修改上述组件；完整 kernel 与缩小的转换 kernel 均保留，便于分别验证接口适配和下层诊断。

### skill 使用与独立复现

使用了 `inductor-kernel-compile-debug` 的 `_precompile_config` 定向捕获，journal record 0 为
`PYTHON_EXCEPTION`。自动 exporter 未覆盖这个 kernel 使用的 `triton_helpers.div_floor_integer` shim，
因此自动导出失败；最终脚本手工内联了实际安装版本的同名 helper，固定相同 ASTSource/config。
kernel body 与捕获版本经 AST 对比一致（仅 helper 引用改名和换行）。
独立脚本已在 capture 目录之外、原安装环境下复现，未要求通用 replay 框架或模型文件。

## fastNLP_Bert：长度输出遗漏写入，归属 Inductor

模型 `fastNLP/embeddings/bert_embedding.py` 先计算：

```python
batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False), 0)
max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()
```

后面第 511 行根据 `max_word_piece_length - real_word_piece_length` 分配 padding。
动态推理 performance 在此申请 `3072.05 GiB`，而设备空闲 HBM 约 `121.55 GiB`。
这是错误尺寸引发的运行期分配，不是编译线程导致的主机内存耗尽。

### 小图与生成代码证据

独立小图保留 mask、table gather、两次 sum 和 max，输入为 int64 `words[1,473]`、`table[2869]`，
words stride `(473,1)`；使用确定性合成数据，无模型或数据集依赖。
同一输入的 eager 与 compiled 对比为：

```text
EAGER 1165 [466] [1165]
COMPILED 1073758208 [466] [1165]
CHECK max False
CHECK mask True
CHECK seq_len True
CHECK lengths True
CHECK total True
```

错误标量的具体数值依赖未初始化内存，不能把 `1073758208` 当作固定复现要求。

生成 wrapper 分配 `buf2`，将其作为标量 `buf5` 复用，并传入 kernel 的 `in_out_ptr0`。
kernel 只写入 mask、逐元素长度、长度之和与有效词数，函数体从未访问 `in_out_ptr0`。
wrapper 随后把 `buf5` 作为 maximum 返回。
因此在进入 Triton 编译器之前，Inductor 已经生成缺少必需写入的程序。
这排除了把本问题归给模型申请策略或下层编译器数值误编译的解释。

### 源码位置

在上述框架 commit 的 `torch_npu/_inductor/codegen/triton.py`：

- `NPUIndexTritonKernel.store()` 第 3170 行附近：如果 value 属于 `reduction_result_vars`，
  写入被放进 `_deferred_reduction_stores`，而不是 `self.stores`。
- `codegen_body()` 第 2639 行附近：提前返回条件只检查 loads、stores、compute、post_loop_store
  及三个 original buffer，没有检查 `_deferred_reduction_stores`。

batch 1 的 `sum(...).max()` 中 max 被简化为标量 copy，不再生成新的计算语句。
只剩延迟写入时，`codegen_body()` 错判为空并返回。
只读运行期探针捕获到：

```text
STORE_PROBE buf5 0 tmp13 reduction_result True
deferred ['tl.store(in_out_ptr0 + (tl.arange(0,1)), tmp13, None)']
BODY_BEGIN {'loads': 0, 'compute': 0, 'stores': 0, 'post_loop_store': 0}
BODY_END deferred ['tl.store(in_out_ptr0 + (tl.arange(0,1)), tmp13, None)']
```

这条 store 已经产生，却没有被发射到最终 kernel。探针仅观察方法调用，没有修改生成结果。
后续修复应在 codegen 中确保仅包含延迟 reduction store 的阶段也能完整发射；本轮未实施或验证修复。

## fastNLP_Bert：独立编译候选失败，归属下层

目标 kernel 为 `triton_red_fused_eq_index_masked_fill_max_ne_s_0`。
使用 `inductor-kernel-compile-debug` skill 捕获并导出 journal record 0，
`R0BLOCK_SUB=1`、`compile_mode=simd_simt_template`、`multibuffer=True`、
`num_warps=1`、`num_stages=2`、`debug=False`，target `npu/Ascend950PR_9579/32`。

独立导出及精简版都只构造 `ASTSource`、`GPUTarget` 并直接调用一次 `triton.compile`，
不执行 Dynamo、Inductor codegen、autotune，也不加载模型。
在 capture 目录之外运行，精简版还去掉了修复 overlay，均复现：

```text
failed to legalize unresolved materialization from () to ('tensor<1xi64>')
tmp8 = tl.where(tmp7, tmp6, tmp0)
tmp10 = tl.load(in_ptr1 + tmp8, r0_mask)
Pipeline failed while executing [`TritonToLinalg` on 'builtin.module' operation]
```

这里的 gather 是合法的 int64 间接索引，候选 block 为 1；失败不涉及 HBM 分配或 UB 容量限制。
在固定 DSL 和配置、绕开 Inductor 后，转换 pass 仍然失败。
另一个直接编译对照只把 `compile_mode` 改为 `simt_only`，其余 DSL、block、signature、target、
attributes、num_warps、num_stages 和 multibuffer 均不变，结果 `COMPILE_PASS`、exit 0。
这排除了模型、Inductor 图转换、autotune 搜索和通用 Triton 语法错误，
把本编译错误限定在 Triton-Ascend 的 template 转换路径。本轮未进一步修复或定位该 pass 的内部实现。
模型日志证明该候选失败后继续执行后续 BERT 图，最后因前述错误标量分配 padding 而退出。
两项独立缺陷分别保留，不能把这个转换错误当作当前模型的第二层阻塞。

## ConvNeXt：训练 accuracy 的 GELU 布局错误

静态训练 accuracy 的 forward 已编译完成，在执行生成 wrapper 时首次失败：

```python
buf25 = empty_strided((8, 320, 56, 56), (1003520, 1, 17920, 320), device='npu', dtype=torch.float32)
triton_poi_fused_convolution_native_layer_norm_9.run(buf24, buf25, 8, 320, 3136, stream=raw_stream0)
buf26 = torch.ops.aten.gelu.default(buf25)
```

`buf25` 是逻辑 NCHW、物理 channels-last 的 FP32 tensor。异常发生在原生 `aten.gelu` 调用，
不是该 Triton kernel 的编译异常，也尚未进入实际 backward。

将模型 shape 原样抽出，完全不导入 Inductor，仅执行 `empty_strided(...).fill_(0.5)` 和
`torch.ops.aten.gelu.default(x)`，同样报：

```text
Only c10::MemoryFormat::Contiguous is supported for creating a npu tensor
ERR01007 OPS feature not supported
```

再缩小到 `[2,8,4,4]`、stride `(128,1,32,8)`，仍以 exit 1 复现。
同一输入先 `.contiguous()` 后原生 GELU 正常执行，exit 0、首元素约 `0.34573123`。
因此错误不依赖权重、优化器、AOTAutograd、Triton 编译或模型的融合代码。

代码证据位于修复基线 `torch_npu/csrc/aten/common/TensorFactories.cpp` 的 `empty_npu`：
第 192–196 行读取输出 `memory_format`，只接受 `Contiguous`，否则产生上述原文错误。
输入 tensor 可以通过 `empty_strided` 创建，失败的是 GELU 原生路径随后申请输出的内存格式。

**归属需要区分原生限制与编译集成边界。** 原生问题在不使用 Inductor 时即可复现，
应由 torch_npu 原生 GELU/输出分配路径支持该合法布局，或在入口明确转换。
同时，当前整网的这个布局由 Inductor 生成；fallback 到有布局限制的原生算子时，
NPU Inductor 也应约束输入布局，不能仅凭“原生报错”排除其适配责任。
这里不将它归为 Triton/BiSheng 的 kernel 编译错误，也没有声称仅修复原生分配就能保证训练整网通过。

简单入口为 `convnext_gelu_layout.py`，只有一个原生 GELU；`CONTIGUOUS=1` 执行对照。
原始 wrapper 片段、原生源码 guard、完整异常和小用例日志保存在 ConvNeXt artifacts 中。

## ConvNeXt：推理 accuracy 与 HF32 的精度边界

静态推理 accuracy 在 FP32、batch 8、`[8,3,288,288]` 输入下完成编译和执行，
得到 `fail_accuracy`，没有 graph break，整网容差为 `rtol=atol=0.001`。
原进程的卷积 HF32 为 True、矩阵乘 HF32 为 False。

| 相同 benchmark 的诊断对照 | 结果 | 可以说明什么 |
| --- | --- | --- |
| `aot_eager` | `pass_accuracy` | 仅使用 Dynamo/AOTAutograd 不产生该失败 |
| Inductor，卷积 HF32 改为 False，矩阵乘保持 False | `pass_accuracy`，退出后检查标志仍为 False | 原失败与 HF32 路径有关；保留原矩阵 FAIL |
| Inductor，保持原配置，只读观察生成 kernel | 整网仍 `fail_accuracy` | 观测未将原失败消除 |

只读探针对全部 32 个生成 kernel 做了 108 次输出检查，涵盖 LayerNorm 统计量与 affine 输出、
GRN 平方和/范数/逐点计算、残差加法、残差融合 LayerNorm、最后的池化及 LayerNorm。
参照结果使用每次实际 kernel 输入计算，在 `rtol=1e-4、atol=1e-5` 下，108 项的 mismatch 都是 0。
这是对实际输入和实际选中候选的检查，不是对所有 shape 或候选正确性的证明。
生成代码和每项数值记录均归档，避免仅凭整网最终小误差推断某个 kernel 错误。

移除 benchmark 后的 22 行模型入口 `convnext_inference_accuracy.py` 也已独立验证：
HF32=True 时 8000 个输出中 1 个不匹配、最大绝对误差 `0.0015096`、exit 1；
仅关闭 HF32 时不匹配为 0、最大绝对误差 `2.03e-6`、exit 0。

另提供不使用 Inductor、timm 或模型权重的 `convnext_hf32_sensitivity.py`：
同一 FP32 输入分别使用原生 LayerNorm 和等价均值/方差公式，接相同权重的原生 1×1 卷积。
CPU FP64 的 LayerNorm 作为独立精度参照，结果为：

| 比较 | 最大绝对差 | 超出 `rtol=atol=0.001` 的元素数 |
| --- | --- | --- |
| 原生 LayerNorm vs CPU FP64 | `9.54e-7` | 0 |
| FP32 公式 LayerNorm vs CPU FP64 | `7.15e-7` | 0 |
| 两个 FP32 LayerNorm 输出 | `4.77e-7` | 0 |
| 两个输出分别接相同原生卷积，HF32=True | `0.0062294` | 206 |
| 相同对照，HF32=False | `1.53e-5` | 0 |

该用例说明：无需任何生成 kernel，等价 FP32 归约的舍入差异也会在当前 HF32 卷积中被放大到超过验证容差。
结合整网 HF32 关闭后通过、实际全部生成 kernel 的逐输入对照均通过，
当前应定界为 **HF32 精度配置下的整网数值敏感性**，没有证据将本次失败定位为 Inductor 生成代码的独立计算缺陷。
建议先统一 FP32 accuracy 验证时的 HF32 策略，再判断是否需要针对 HF32 重新定义容差；本轮不提交精度策略或框架修复。
这不构成对 HF32 内部量化步骤的源码级定位，也不保证所有输入均通过；原配置的失败不会改记为 PASS。

另一个需要保留的验证限制是：原 benchmark 尝试将 NPU 参照转成 FP64 时，
原生路径提示设备不支持 double、转换为 float。因此不能把该 NPU 参照称为真正的 FP64 golden。
上面的缩小用例明确在 CPU 上计算 FP64 LayerNorm，避免混淆。

内置 `INDUCTOR_ASCEND_CHECK_ACCURACY=1` 还会改变 lowering 路径，在 GRN 的 `aten.add` 报
`NotImplementedError: Pointwise`，尚未到原失败位置。此项是诊断机制的限制，
不作为原模型的第二层问题；有效数值检查使用上述不改变 DSL 的运行时观测。

## ConvNeXt：候选编译的 PlanMemory 异常

以下编译候选异常单独保留，不能代替训练运行期错误或推理数值问题，也不排列为下一层阻塞。

动态推理 performance 的 `triton_poi_fused_add_native_layer_norm_permute_7` 是一个四维
masked load + add + store 程序，把两种布局的输入相加。
固定候选为 `T0BLOCK_SUB=1`、`Z1BLOCK_SUB=80`、`Y2BLOCK_SUB=16`、`X3BLOCK_SUB=6`，
两个 runtime block 参数 `T0BLOCK/Y2BLOCK` 保留为 i32，`ks0/ks1` 保留为 i64。
options 为 `simd_simt_template`、`multibuffer=True`、1 warp、2 stages、`debug=False`。
target 为 `npu/Ascend950PR_9579/32`，pointer attributes 为空。

skill 只加载该 kernel 的源码并捕获 `_precompile_config` record 0，自动导出成功。
完整独立导出和 95 行精简版都绕开 Inductor，固定 `ASTSource` 直接编译，均得到：

```text
[ConvertLinalgIRToBinary] encounters error:
LLVM ERROR: PlanMemory Traverse IR Failed!
```

精简版还在原安装环境、不加载修复 overlay 时复现。
这把崩溃定位到 Triton-Ascend 调用的 BiShengIR 内存规划路径，而非模型 forward 或 Inductor 执行期。
源码使用的非 2 次幂 `tl.arange(0,80/6)` 在 Triton-Ascend 扩展中合法，不能套用 CUDA 的限制来认定 DSL 非法。
仅将 `multibuffer=False` 的对照仍报相同 PlanMemory 错误，故不能归因于单独的多缓冲开关。
没有改动编译器或过滤配置，也没有对该闭源 pass 的内部实现作进一步根因声明。

该候选失败后，原进程继续编译 `_9` 等后续 kernel，说明它不是当时的终止错误。
其状态为“独立候选缺陷未修复，搜索继续，当前非终止错误”；不是“修复后缺陷影响被屏蔽”。
另一些候选明确报告 UB 容量超限，属于资源限制下的候选拒绝，不与此内部崩溃合并统计。

## 复现与证据

复现脚本与运行说明见
[`scripts/repro/cicd/three_model_boundary_20261021`](../../../scripts/repro/cicd/three_model_boundary_20261021/README.md)。
原始 graph、生成 wrapper/kernel、只读探针输出、独立编译日志保存在
[`artifacts/cicd/three_model_boundary_20261021`](../../../artifacts/cicd/three_model_boundary_20261021/)。
原始生成产物只去除环境路径、backend hash 等信息，保留 kernel 结构和格式。
