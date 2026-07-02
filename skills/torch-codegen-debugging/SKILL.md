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
