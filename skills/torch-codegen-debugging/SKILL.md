---
name: torch-codegen-debugging
description: Use this skill for PyTorch/torch_npu/Inductor codegen failures, wrong generated kernels, DSL/Triton lowering bugs, scheduler or tiling issues, compile-time crashes, or accuracy failures that appear only after torch.compile. 适用于 torch.compile 后的 codegen、scheduler、tiling、kernel type、generated DSL、精度异常定位。
---

# Torch Codegen Debugging

这个 skill 用于定位 PyTorch / torch_npu / Inductor 的 codegen 类问题。典型信号是 eager 正常、`torch.compile` 后失败，或者 generated DSL / Triton kernel 中出现可疑的 indexing、reshape、reduction、mask、store placement。

写作和记录采用中文为主、关键技术词保留英文的形式。保留 `codegen`、`lowering`、`scheduler`、`tiling`、`kernel type`、`DSL`、`axis`、`stride`、`mask`、`reduction`、`fallback` 等英文术语，分析过程用中文说明。避免过程性套话，直接写事实、证据和结论。

## 基本原则

先找 code evidence。不要只根据日志解释失败；优先拿到 generated code、FX graph、scheduler node metadata 或 source code。

保持 repro 足够小。一个 compiled function、一个 generated kernel 或一个 scheduler node，通常比模型级失败更容易分析。

区分 fact 和 hypothesis。没有被 DSL、源码或复现确认的行为，需要标记为 inference。

保护用户修改。worktree dirty 时先看相关 diff，不要回滚无关文件。

## 定位流程

### 1. 分类 failure

记录这些信息：

- failure stage：Dynamo、AOTAutograd、lowering、scheduling、codegen、kernel compile、runtime launch、accuracy check。
- device path：NPU target、kernel type、backend path、fallback behavior。
- symptom：exception、wrong output、illegal memory access、compile error、missing trace、precision mismatch。
- inputs：shape、stride、dtype、dynamic/static mode、影响 codegen 的 env flags。

如果是 accuracy issue，必须用同一组输入比较 eager 和 compiled output。

### 2. 构造或缩小 repro

优先写 standalone script：

- 使用 `torch.manual_seed` 构造 deterministic input。
- 打印 shape、stride、dtype、dynamic flag、check flag。
- 只 compile 目标 function。
- 可选 eager 对比和 `torch.testing.assert_close`。
- 通过 `CHECK`、`DYNAMIC`、shape 参数显式控制开关。

reduction kernel 要保留 non-contiguous stride pattern。把输入简化成 contiguous 可能会隐藏 bug。

### 3. 获取 generated artifact

选择侵入性最低的方法：

- kernel 已经生成时，优先打开 debug trace。
- codegen 阶段提前失败、trace 没有生成时，在 codegen 点附近加临时日志。
- 打印完整 generated kernel，不只贴失败行。
- 同时记录 metadata：`axis_names`、`tiling_axis`、`split_axis`、`low_dims`、`numof_reduction_axis`、`npu_kernel_type`、runtime block args、static axis values。

debug log 要聚焦且容易删除。最终代码不要留下无条件噪声日志，除非用户明确要求保留。

#### 使用现有工具抽取单个 Triton kernel

当工作区中存在 `torch-learn` 仓库时，优先使用
`scripts/tools/extract_triton_kernel.py` 从 Inductor `output_code.py` 提取可独立阅读和运行的 kernel，
不要手工复制不完整代码。先用 `rg --files` 定位脚本和 `output_code.py`：

```bash
python <torch-learn>/scripts/tools/extract_triton_kernel.py \
  /path/to/output_code.py \
  triton_poi_fused_add_0 \
  -o /path/to/triton_poi_fused_add_0.py
```

同名 kernel 有多次 `.run(...)` 时，工具只提取第一次调用的参数和依赖。输出包含公共导入、DSL、
`async_compile.wait(globals())`、`del async_compile`、benchmark input 和首次 launch。

需要同时查看 FX 语义时可以增加 `--include-eager`。生成的 `eager_forward(...)` 不会猜测
FX placeholder 与 Triton pointer 参数的对应关系；涉及 in-place、多输出或别名时，调用者仍需显式构造 eager 输入。

#### 记录最终 best tiling

需要确认最终 selected config 时，使用
`scripts/tools/autotune_tiling.py::BestTilingRecorder`，不要从 autotune 候选日志推测 winner：

```python
from scripts.tools.autotune_tiling import BestTilingRecorder

recorder = BestTilingRecorder("npu")  # 或 "cuda"
recorder.install()
try:
    recorder.start_capture()
    try:
        compiled_fn(*args)
    finally:
        records = recorder.stop_capture()
finally:
    recorder.uninstall()
```

hook 必须在目标 backend 导入后安装，并覆盖真正触发 autotune/dispatch 的调用。普通记录包含
`kernel_name`、`selected_config`、`runtime_blocks`；NPU group 还包含 `group_id` 和
`feature_inputs`。该工具使用 Inductor 内部接口，升级 torch/torch_npu 后必须在实际设备上确认 hook 仍生效。

不要把 recorder 扩展成特定测试脚本的 CSV 汇总器。场景 ID、shape、execution 和落盘格式由调用脚本负责。

### 3.1 定位 symbolic group 的通用证据链

分析 grouped autotune 时，按生成期到运行期依次收集，避免把不同阶段的概念混在一起：

1. generated metadata：`group_enabled`、`group_template`、`group_workload`、
   `primary_group_axis`、`static_split_axes`、`secondary_runtime_symbolic_axes`、
   `group_features`、`runtime_block_arg_names`。
2. candidate plan：每组 representative feature/axis values、reachable group、variants、policies。
3. runtime selection：真实 `feature_inputs`、解析出的 `group_id`、`selected_config`、
   `runtime_blocks` 和实际 grid。
4. device result：目标 kernel 的单次 device timing，以及 static/dynamic 对照。

明确区分四层信息：

- generated DSL body 决定数学计算、访问和 mask。
- representative 只用于某个 group 的 autotune benchmark input，不是当前 runtime shape。
- selected config 是该 group benchmark 后的 winner compile config。
- runtime blocks / grid 是 winner 在真实 runtime shape 上 materialize 的 launch 参数。

匹配 static、dynamic 和 group kernel 时，不要依赖 generated kernel 名称后缀。使用 graph fragment、输入输出、
迭代域、indexing 和数学表达式建立对应关系。若 DSL body 相同但性能不同，优先逐项对齐 selected config、
runtime blocks、grid、kernel type 和编译模式；若这些也相同，再检查 profile 噪声、cache 和调用上下文。

只把能跨 case 复用的检查步骤写入 skill。具体 kernel 名、shape、性能数字、临时 patch 和尚未在线上代码验证的
hypothesis 应进入 issue log，而不是固化为通用规则。

### 4. 机械阅读 DSL

把 generated symbols 映射回 tensor semantics：

- 识别 preserved axes，例如 `x0`，以及 reduction axes，例如 `r0`、`r1`、`r2`。
- 用 tensor shape / stride 对齐 `tl.load` address formula。
- 看 `base_*`、`loop_*`、mask 如何定义实际访问范围。
- 跟踪 value 在 `permute`、`reshape`、`sum`、`store` 前后的 layout。
- 检查 store 是发生在 full reduction 之后，还是落在 partial loop 内。

每个可疑点都要指出具体 generated operation，并说明 expected behavior。

### 5. 快速验证 DSL 假设

尝试修复前，优先直接修改 generated DSL 做 quick experiment。目标是先确认“期望 DSL 语义”是否真的能解决问题，再回到 codegen 侧找通用生成规则。

适合直接改 DSL 快速验证的场景：

- 调整 `permute` / `reshape` 顺序。
- 移动 accumulator 初始化、`tl.sum`、`tl.store` 的 loop 层级。
- 替换 reduction dim。
- 修改 mask 和 value 的组合方式。
- 固定某个 tiling 参数，观察 DSL 语义是否成立。

快速验证时可以用单 kernel 调试脚本、已有 debug trace 中的 kernel，或把 generated kernel 单独抽出来调用。不要每次都从 codegen 修改开始重新 compile；如果 DSL 修改都达不到预期，说明 hypothesis 本身可能不成立，需要先继续缩小问题。

直接改 DSL 只用于验证语义，不是最终修复。验证通过后，再把结论映射回 codegen 的 axis analysis、layout transform、loop emission 或 scheduler metadata。

### 6. 对比 version / path

不同版本、device 或 fallback path 行为不一致时：

- 先比较 generated kernel。
- 再比较决定 axis order、tiling、kernel type、reduction dim 的源码。
- 不要把能 fallback 的路径当作主路径正确性的证据。
- 如果一个 device 可以 fallback 到另一个 kernel type，另一个不能，优先分析 shared failing path。

### 7. 加 focused instrumentation

每个 instrumentation 只回答一个问题：

- axis order 如何选出来。
- reduction dim 如何计算。
- tiling config 如何选中。
- 哪个 node 或 store index 生成了某行 DSL。
- 哪个 branch 写出了 `prefix`、`post_loop_store`、`stores`。

临时日志可以用 `[DEBUG]` 前缀。完成定位后删除或加开关。

### 8. 设计 fix

fix 要对应 DSL defect：

- buffer 初始化在错误 loop 层级时，改 codegen emission site，而不是手改生成文本。
- reduction dim 错误时，修正 dim analysis。
- flatten 前 value layout 错误时，基于 axis metadata 计算 `permute` order，不硬编码 case。
- tiling config 暴露尾块问题时，只在精确条件下过滤或调整排序。

codegen 本身应该是一套通用规则。不要因为当前 case 暴露问题，就添加只识别该 shape、该 op、该 kernel name、该 tiling 的 special case，也不要优先用 fallback、过滤 config、绕过某条路径来掩盖语义问题。

第一目标始终是正向定位和通用解法：

- 从 tensor semantics、axis order、stride、mask、reduction layout 推导规则。
- 用 metadata 描述适用范围，例如 contiguous multi-reduction、non-contiguous stride、SIMD/SIMT template，而不是绑定某个 RMSNorm case。
- 保持 single-reduction、non-contiguous reduction、pointwise fused node 等既有路径行为不变。
- 如果只能给出次级解法，例如 fallback、config filter、局部 bypass，需要说明为什么通用解法当前不可达、风险是什么，并先向用户确认。

影响范围要窄。contiguous multi-reduction 的修复不应改写 single-reduction 或 non-contiguous behavior。

### 9. 验证

能跑时按三层验证：

- 本地语法和格式：`py_compile`、`git diff --check`。
- repro correctness：目标 device 上运行 `CHECK=1`。
- artifact validation：检查 regenerated DSL，确认原始错误行为消失。

最终报告包含 command、result、generated-kernel evidence，以及不能运行的测试和原因。

## 输出结构

面向用户汇报时按这个顺序：

1. repro 和 symptom。
2. generated DSL evidence。
3. source-code cause。
4. fix summary。
5. validation result。
6. remaining risk。

代码引用要具体，优先给 file path、function name 和 line number。
