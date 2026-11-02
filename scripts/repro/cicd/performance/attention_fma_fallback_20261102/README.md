# attention backward FMA fallback：CPU dispatch 对照

[分析文档][analysis] · [已采集结果][result] · [源码证据][source]

本脚本只核对一次 `torch.ops.prims.fma.default(a,b,c)` 实际执行几次乘加，
不执行 torch.compile、不修改编译器、不使用 NPU，也不提供 NPU 性能数据。

## 运行方式

需要带 `torch._inductor.inductor_prims` 的 PyTorch。
已有采集来自 torch `2.13.0+cpu`；后续版本的内部接口或运行时实现可能变化。
从仓库根目录运行：

```bash
python scripts/repro/cicd/performance/attention_fma_fallback_20261102/check_fma_dispatch.py \
  --output /tmp/attention-fma-dispatch.json
```

输入均为 CPU FP32：a/c shape `[2,4,32,32]`、stride `[4096,1024,32,1]`；
b shape `[2,4,32,1]`、stride `[128,32,1,1]`，用于广播。
小尺寸用于核对 dispatch 次数，不代表模型 attention 的真实尺寸。

在已观察缺陷的实现中，预期计数为 `prims::fma=1, aten::mul=2, aten::add=2`，
输出与 `(a*b)+c` 一致，进程退出码为 0。
若计数不同，脚本保存实际计数并返回非 0，表示没有重现原执行模式，不直接表示数值错误。

2026-11-02 已在指定框架环境运行等价的内联 CPU 对照，结果已归档。
本次新包装脚本只做本地语法和入口检查，没有将其表述为新的 NPU 验证。
归档结果验证了公共 prims 执行模式；LayoutLM 整模型反向 wrapper 的捕获仍未完成。

[analysis]:
  ../../../../../docs/issue-logs/cicd/performance/electra-layoutlm-train-gpu-inductor-gap-20261102.md
[result]:
  ../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_cpu_dispatch_result.json
[source]:
  ../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt
