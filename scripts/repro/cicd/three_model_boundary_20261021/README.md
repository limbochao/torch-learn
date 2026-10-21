# 三模型定界复现

对应分析：[三模型问题定界](../../../../docs/issue-logs/cicd/three-model-boundary-20261021.md)。
仅做定界，不包含框架修复。模型验证使用 `debug/cicd-all-fixes-20261020` 的 `5b18fac653`。

## mobilenetv2_100：标量签名与独立编译

首选小图复现，无 MobileNet 或 timm 依赖：

```bash
TORCHINDUCTOR_NPU_BACKEND=default python mobilenet_hardtanh_graph.py
```

仅编译 FP32 `[16,64]` 的 hardtanh backward 与 sum，动态边界 `0.0/6.0` 被生成为 FP64 参数，
受影响版本报告 `undefined symbol: __truncdfsf2`。

进一步查看类型生成及下层转换：

```bash
python mobilenet_scalar_signature.py
python mobilenet_scalar_cast_compile.py
SCALAR_TYPE=fp32 python mobilenet_scalar_cast_compile.py
python mobilenet_backward_compile.py
```

第一个脚本只调用 Inductor 的 `signature_of()`，不加载模型或分配设备 tensor。
用 `TensorArg(dtype=torch.float64)` 复现 CPU 零维 tensor 解包，输出
`UNWRAPPED_CPU_SCALAR_SIGNATURE fp64` 并 assertion 失败。
它定位签名生成边界，不能代替完整模型验证。

第二个脚本只有一个手写 Triton kernel，固定 A5 的 SIMT 编译配置。
默认 FP64 标量转 FP32，编译器在 `__truncdfsf2` 报 `Bad machine code`；
FP32 signature 对照输出 `COMPILE_PASS fp32`。两项都只编译，不启动 kernel。
`_use_fp64_for_unbacked_floats=False` 只控制 SizeArg，不覆盖本例的 TensorArg；小图实测仍失败。

第三个脚本保留模型失败的完整反向 kernel 和固定候选，仍仅调用一次 `triton.compile`。
已使用 `inductor-kernel-compile-debug` 捕获；自动 exporter 缺少 `div_floor_integer` shim，
所以该脚本内联了实际 torch helper，其余 body/config 保持一致。
完整 kernel 与 27 行转换复现均已在原安装环境、capture 目录之外复现，无模型或 overlay 依赖。
这保留下层编译证据，同时用第一个脚本证明 default Inductor 的 NPU 标量适配存在缺口。

## fastNLP_Bert 长度标量错误：Inductor 小图

在已加载 CANN、支持 torch_npu Inductor 的 NPU_A5 环境中执行：

```bash
TORCHINDUCTOR_NPU_BACKEND=default TORCHINDUCTOR_COMPILE_THREADS=1 \
  python fastnlp_length_graph.py
```

脚本只有一个 compiled function，同一组确定性输入同时执行 eager 与 Inductor。
无需 fastNLP、TorchBench、BERT 权重或数据集。
输入为 contiguous int64 `words[1,473]`、`table[2869]`，`dynamic=True`，返回 5 个输出。
受影响版本的 maximum 输出与 eager 的 `1165` 不一致，其余输出一致；最终 assertion 失败。
未初始化标量的实际值不固定。该复现只比较标量，不按错误值申请巨量显存。

可用 `TORCH_COMPILE_DEBUG=1` 保存生成代码，检查 kernel 的 `in_out_ptr0` 是否存在实际写入。
这是 Inductor 的生成问题，应保留小图复现，不能只提供一个已经遗漏 store 的底层 kernel。

## fastNLP_Bert template 候选：下层编译复现

```bash
python fastnlp_gather_compile.py
```

这是使用 `inductor-kernel-compile-debug` skill 捕获 record 0、导出并精简后的单 kernel 程序。
保留原 kernel 计算、signature、constexpr、pointer attributes、target 和 compile options，
删除通用导出器中本用例未使用的 launch/helper 代码。
仅调用一次 `triton.compile`；不依赖模型、Inductor、capture 目录、原缓存或 overlay。
默认 target 固定为实际出错的 `Ascend950PR_9579`，需要兼容的 Triton-Ascend/CANN 环境。

配置为 `R0BLOCK_SUB=1`、`simd_simt_template`、multibuffer 开启、1 warp、2 stages。
受影响版本在 `TritonToLinalg` 报 `tensor<1xi64>` 无法合法化并非零退出。
完整 skill 导出和精简版均已在 capture 目录外执行复现，精简版还使用原安装环境验证。
此候选在模型中可以回退，独立编译 FAIL 与模型最终 FAIL 分别记录。

定界对照：复制该文件，只把 options 中的 `compile_mode` 改为 `simt_only`，其余参数不变，
在同一环境编译得到 `COMPILE_PASS`、exit 0。原始失败入口始终保持 template 配置。

## ConvNeXt PlanMemory 候选：下层编译复现

```bash
python convnext_planmemory_compile.py
```

skill 捕获 record 0 后导出、精简，保留完整 add/permute kernel、显式 signature、constexpr、target 和 options。
只编译一个 `simd_simt_template` 候选，不执行模型或启动 kernel。
配置为 sub-block `(1,80,16,6)`、multibuffer 开启、1 warp、2 stages，target 为实际 A5。
受影响环境报 `LLVM ERROR: PlanMemory Traverse IR Failed!`，exit 1。
自动导出及精简版均在 capture 目录之外复现；精简版还使用原安装环境验证。
仅关闭 multibuffer 的编译对照仍报同一错误。

这是独立候选异常。原矩阵的动态推理 performance 最终 PASS，候选失败已回退，当前非阻塞。
静态推理 accuracy 的数值失败另行分析。

## ConvNeXt 动态训练：SIMT 栈溢出候选

```bash
python convnext_layernorm_stack.py
NUM_WARPS=8 python convnext_layernorm_stack.py
```

109 行，只保留一个 LayerNorm 统计量 kernel、完整函数体、显式输入及一次直接编译、执行。
使用 skill 的 prerun metadata 捕获后整理，不加载模型、Inductor、autotuner、capture 文件或权重。
原配置 `XBLOCK=8/R0_BLOCK=128`、64 warps、`simt_only` 编译成功，执行时栈溢出、exit 1。
只改为 8 warps 后 exit 0，三个输出均通过 eager 对照，`rtol=atol=1e-4`。

也可保留 64 warps，只给本进程增加运行期栈，复现资源预算对照：

```bash
printf '%s\n' '{"defaultDevice":{"default_device":"0"},"StackSize":{"simt_stack_size":8192}}' > stack8192.json
TORCH_ACL_INIT_CONFIG_PATH="$PWD/stack8192.json" python convnext_layernorm_stack.py
```

此项实测 exit 0，三个输出对照均通过；不修改原安装配置。
默认每线程栈为 1152 Byte，两个对照说明失败取决于线程及栈资源预算。
原模型动态训练 performance 最终 PASS，故该失败候选标为“已回退、当前非阻塞”，不排列为下一层阻塞。
入口在原安装环境、不加载框架修复 overlay、脱离捕获目录时验证。

## ConvNeXt 训练 GELU：原生布局复现

```bash
python convnext_gelu_layout.py
CONTIGUOUS=1 python convnext_gelu_layout.py
```

13 行入口，不加载模型，也不调用 Inductor。默认构造 FP32 `[2,8,4,4]` channels-last tensor，
直接调用 `aten.gelu`，报 `Only c10::MemoryFormat::Contiguous is supported`、exit 1。
原模型 `[8,320,56,56]` 布局也已验证相同错误；缩小脚本避免大内存依赖。
contiguous 对照 exit 0。这证明原生路径的布局限制可以独立复现；
整网的这个布局由 Inductor 生成，其原生 fallback 入口的布局适配责任也不能排除。

## ConvNeXt 推理：HF32 数值敏感性

```bash
python convnext_hf32_sensitivity.py
```

只使用原生 PyTorch/torch_npu：两种等价 FP32 LayerNorm，接相同的原生 1×1 卷积。
LayerNorm 另与 CPU FP64 比较，避免把设备上被转换为 float 的 NPU double 当作 golden。
实测 LayerNorm 差约 `4.77e-7`；HF32 卷积后有 206 个元素超出 `rtol=atol=0.001`，
关闭 HF32 后为 0。该用例展示误差放大的机制，不代表原模型结果可改记为通过。

`convnext_inference_accuracy.py` 是移除 benchmark 的完整模型入口：保留预训练 timm 模型、
seed 1337、batch 8、测试分辨率 288 和输入标准化，与 eager 比较整网输出。
需要已缓存的模型权重或可访问权重下载源；`HF32=0` 用作精度配置对照。
原配置实测 8000 个输出中 1 个不匹配、exit 1；`HF32=0` 为 0 个、exit 0。
最大绝对误差由 `0.0015096` 降为 `2.03e-6`。完整记录见 artifacts。

## 模型验证命令

从 TorchBench runner 目录执行，`SUITE` 取 `timm_models` 或 `torchbench`，`MODEL` 取指定模型全名：

```bash
TORCHINDUCTOR_NPU_BACKEND=default TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCHNPU_PRECOMPILE_THREADS=4 \
python "${SUITE}.py" --performance --training --float32 \
  --backend inductor --npu-backend triton --disable-aclgraph \
  --only "$MODEL" --dynamic-shapes --repeat 1 --iterations 1 --output result.csv
```

静态 shape 去掉 `--dynamic-shapes`；推理将 `--training` 改为 `--inference`；
accuracy 将 `--performance` 改为 `--accuracy`，去掉 `--repeat 1`，改用 `--iterations 5`。
不设置 batch size，保留 runner 各场景的默认值。测试环境的 Inductor Python 文件须先核对为上述 commit。
本轮保存每场景进程退出码、CSV 和生成代码，单编译候选超时为 60 分钟。
