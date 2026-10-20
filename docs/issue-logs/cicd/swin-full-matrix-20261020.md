---
title: Swin 全场景验证与编译问题定界（2026-10-20）
published: false
---

# Swin 全场景验证与编译问题定界

`swin_base_patch4_window7_224` 的 8 个 FP32 场景均已获得终态：4 项性能运行通过，4 项精度失败。
精度失败包括两项推理编译失败、静态训练编译失败，以及动态训练数值校验失败。
此前静态推理性能 batch 128 的通过结果，不能外推到精度场景默认 batch 8。

运行环境为 A5 / Ascend950PR_9579 的 Docker、`pta_213_cicd`，
PyTorch `2.13.0+cpu`、torch_npu `2.13.0.dev20261019`、Triton `3.6.0`、CANN `9.1.0`。
精确提交及环境变量见 [环境记录](../../../artifacts/cicd/swin_matrix_20261020/environment.json)。
全部场景使用 FP32、Inductor + Triton、关闭 aclgraph，初次运行的 compile threads 为 1。
训练性能场景因长时间编译，另启动 compile threads 2 的补跑，复用已有缓存并使用本任务相同的 0、1 卡。
最终结果以 `results.json` 中每项的 `compile_threads` 为准；补跑不构成独立、受控的性能对比。
补跑后容器内存一度达到约 157 GiB / 192 GiB，主动停止两项旧的重复性能任务以留出余量，
保留 compile threads 2 的同场景任务，最终采用其完整运行结果。旧任务没有功能终态，主动停止不计模型失败。
复现命令见 [脚本说明](../../../scripts/repro/cicd/swin_matrix_20261020/README.md)。

## 验证范围

覆盖静态/动态 shape × 推理/训练 × 性能/精度，共 8 个场景。
未指定 batch，性能使用模型表默认 128，精度由 runner 改为 8。
性能参数为 repeat 1、iterations 1；精度 iterations 5。
动态场景使用 `--dynamic-shapes`，并非任意动态 batch/分辨率组合的穷举。
runner 默认的静态场景传 `dynamic=None`，没有额外强制 `dynamic=False`。
[runner 源码证据](../../../artifacts/cicd/swin_matrix_20261020/source_evidence.md)。

| shape | 模式 | 指标 | batch | 最终状态 |
|---|---|---|---:|---|
| 静态 | 推理 | 性能 | 128 | PASS |
| 静态 | 推理 | 精度 | 8 | FAIL：kernel 9 编译失败 |
| 静态 | 训练 | 性能 | 128 | PASS：2 个图、9 次 graph break |
| 静态 | 训练 | 精度 | 8 | FAIL：kernel 47 编译失败 |
| 动态 | 推理 | 性能 | 128 | PASS |
| 动态 | 推理 | 精度 | 8 | FAIL：kernel 9 编译失败 |
| 动态 | 训练 | 性能 | 128 | PASS：2 个图、9 次 graph break |
| 动态 | 训练 | 精度 | 8 | FAIL_ACCURACY：已编译并运行，数值校验失败 |

8 项场景的 CSV 和日志摘录保存在
[cases](../../../artifacts/cicd/swin_matrix_20261020/cases)，
机器可读状态见 [results.json](../../../artifacts/cicd/swin_matrix_20261020/results.json)。

| 性能场景 | speedup | compiled latency / ms | compilation latency / s | 图数 | graph break |
|---|---:|---:|---:|---:|---:|
| 静态推理 | 0.758698 | 290.829608 | 42.511101 | 1 | 0 |
| 动态推理 | 0.767160 | 1033.452464 | 3567.624303 | 1 | 0 |
| 静态训练 | 0.878092 | 776.093184 | 9681.026301 | 2 | 9 |
| 动态训练 | 0.896634 | 755.881844 | 8531.254445 | 2 | 9 |

四项 speedup 均低于 1，运行通过但未获得相对 eager 的加速。
两项训练性能均记录 9 次 graph break（6 种），训练循环没有被捕获为完整单图。

共用前次 Swin 的 Inductor cache 和环境的 Triton cache，编译耗时不代表冷缓存耗时。
部分场景在不同 NPU 卡上并行运行；此轮重点是运行和编译正确性，不能据此进行受控性能比较。
精度基准尝试 FP64，但 NPU 报 double 不受支持并转成 float，故不能将其称为真正的 FP64 golden。
推理精度与静态训练精度的编译失败尚未进入完整模型误差判定；动态训练精度则是实际 accuracy mismatch。
本轮日志均未记录 OOM；从 03:31 起的监测至最终 05:59 检查，容器的历史 OOM kill 计数保持 378。
最终两个训练性能模型主进程均已退出。
[资源终检](../../../artifacts/cicd/swin_matrix_20261020/final_resource_check.json)。

## 推理精度：tiling 搜索未覆盖可用的小 tile

静态和动态推理精度均在 `triton_poi_fused_add_addmm_clone_native_layer__9` 抛出
`NoTritonConfigsError`。runner 最终返回 0，但 CSV 的 accuracy 为 `fail_to_run`，应记为失败。
额外使用 `--performance --batch-size 8` 也复现相同 kernel 的 UB overflow，说明并非仅由精度检查流程触发。
[batch 8 对照日志](../../../artifacts/cicd/swin_matrix_20261020/control_batch8.txt)。

[完整 kernel](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/kernel.py)
融合 addmm bias、window reverse、残差相加和 LayerNorm 的逐元素计算。
size hints 为 `y1=8, y4=7, y3=448, x2=128`，固定 `Y4BLOCK_SUB=7`、`X2BLOCK_SUB=128`。
[动态 source](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/dynamic_kernel.py)
与静态 source 的函数体相同。
tile 表示一次计算所处理的数据块；此处关注另两个 constexpr 维度 `Y1BLOCK_SUB`、`Y3BLOCK_SUB`。

独立预编译实际尝试 28 个配置（含 VF retry），没有成功配置：4 个 UB overflow，24 个
`LLVM ERROR: PlanMemory Traverse IR Failed!`。UB 是计算核片内缓冲区，UB overflow 与主机内存 OOM 不同。
[配置记录](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/configs.json)。

保持相同 kernel body 和编译选项，仅改变 constexpr tile：

| Y1BLOCK_SUB | Y3BLOCK_SUB | 独立结果 |
|---:|---:|---|
| 2 | 8 | UB：需 2,757,888 bits，可用 1,769,472 bits |
| 1 | 4 | PlanMemory Traverse IR Failed |
| 1 | 2 | PlanMemory Traverse IR Failed |
| 1 | 1 | 编译通过；实际运行与 eager 比较通过 |

`(1,1)` 单 kernel 的最大绝对误差为 `2.86102294921875e-6`，使用同组 seed 0 输入，
`atol=rtol=2e-5`。[诊断结果](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/kernel_variant_results.json)，
[运行精度结果](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/tile1_validation.json)。

源码位于安装版本的 `torch_npu/_inductor/runtime/tile_generator.py`：

- `TileGenerator.__init__()` 第 43 行将输入指针数按最多 3 个估算；第 47 行计算 `stop_numel=2730`。
- `add_to_configs()` 第 185–191 行在候选足够多时拒绝低于 `stop_numel + 100` 的 tile。
- `descend_tiling_axis()` 第 239–245 行达到 `stop_numel` 后结束缩小搜索。

默认搜索保留的最小 tile 为 `1*7*4*128=3584`；下一档 `1*7*3*128=2688` 已低于阈值，
因此 `(1,1)` 没有进入候选。仅将独立 TileGenerator 对象的 `stop_numel` 改为 0，
原始候选从 40 增为 58，覆盖 `(1,1)`。没有修改安装文件或模型运行逻辑。
[候选阈值实验](../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel/tiling_threshold_probe.json)，
[精确源码摘录与 SHA256](../../../artifacts/cicd/swin_matrix_20261020/source_evidence.md)。

这定位到 torch_npu Inductor 适配层的 tiling 搜索覆盖不足，而不是 PyTorch 上游 Inductor 已确认的缺陷。
此外，现有候选触发 CANN/BiSheng 的 PlanMemory 崩溃，是后端的另一个问题表现。
通用修复需要在原候选全部失败时仍有更小合法 tile 的搜索保障；此轮没有修改框架，也未验证完整模型修复。

## 静态训练精度：no-loop 推断阻断小 tile，回退配置过大

静态训练精度在 03:35:54 终止，CSV 为 `fail_to_run`。
致命 kernel 为 `triton_per_fused__unsafe_view_clone_native_lay_47`，归约轴长度为 `8*7*2*14=1568`，
输出通道数 512，计算 LayerNorm 的 weight、bias 梯度。
[完整 source](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/kernel.py)，
[原始 wrapper 分配与调用](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/wrapper_excerpt.py)。

独立捕获仅得到一个完整形状的候选和一次 VF retry，两者均失败：

```text
X0BLOCK_SUB=512
ub overflow, requires 180031488 bits while 2031616 bits available!
```

保持 kernel body、固定归约轴和编译选项，仅改变 `X0BLOCK_SUB`：512、16、8 失败，4、2、1 可编译。
其中 16 和 8 的 UB 需求分别为 5,820,416、3,913,728 bits。
[编译对照](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/variant_results.json)，
[便携独立回放](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/portable_compile.txt)。

源码定界到 torch_npu Inductor 适配层：

- `TileGenerator.rebuild_dynamic_axis()` 第 76–85 行：tiling axis 超过 3 个、未指定 no-loop axis 时，
  因低维 x0 的 `512*4=2048 bytes` 小于 4 KiB，将其加入 no-loop axis，没有计入完整归约工作集。
- `descend_all_low_dims()` 第 368–369 行跳过 no-loop axis，无法独立缩小 x0 的 sub tile。
  诊断记录中实际尝试的最小 `X0BLOCK_SUB=16`，估算 tile numel 为 28,672，高于预算 21,845，全部被拒。
- `_triton_config_npu_index_legacy()` 第 3630–3639 行在候选为空时回退到完整 size hints，
  产生 `X0BLOCK_SUB=512`。该回退没有解决资源约束，反而给出更大的工作集。

仅清除独立 TileGenerator 对象推断的 no-loop axis，候选数从 0 增加到 16，覆盖 sub tile 1、2、4、8。
[tiling 诊断](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/tiling_inspection.json)，
[fallback 源码与 SHA256](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/heuristics_source.txt)，
[TileGenerator 源码](../../../artifacts/cicd/swin_matrix_20261020/source_evidence.md)。

### 编译修正之外，还存在输入地址越界

原始 wrapper 为 `in_ptr0` 分配 `(1568,512)`、stride `(512,1)`，共 802,816 个元素。
DSL 的 `tmp0` load 地址为：

```text
x0 + 512*r5 + 3584*r2 + 25088*r4 + 50176*(r2//7) + 100352*r3
```

在合法的 `r3=7,r2=7,r4=1,r5=0,x0=0` 处，地址已为 802,816，超出最后一个元素 802,815。
整数枚举确认最大偏移 827,903，共 25,088 个元素读取超出逻辑范围；这些位置的 x0 mask 仍有效。
[CPU 地址核算结果](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/index_extent.json)。
这是 generated DSL 的地址问题，不能用更小的 tile 消除；尚未将它映射到生成该索引的具体源码转换分支。
当前原始模型先因编译失败终止，未观察到它实际触发运行时非法访问；小 tile 的未修改地址版本没有执行。

额外诊断同时将 tile 改为 4、将 `3584*r2` 改为 `3584*(r2%7)`，保留其他计算，
实际运行与同组 seed 0 输入的 eager window reverse、LayerNorm 参数梯度对照通过。
两个输出最大绝对误差分别为 `3.0517578125e-5`、`2.288818359375e-5`，`atol=2e-4,rtol=2e-5`。
这验证了单 kernel 的 tile 与地址诊断，尚未修改通用 codegen、未验证整模型修复。
[单 kernel 验证结果](../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel/variant_validation.json)。

## 训练编译耗时：CANN 后端长时间编译

静态、动态训练精度均在 `triton_per_fused__unsafe_view_clone_native_lay_42` 的候选编译阶段长时间运行过。
静态场景已越过该 kernel，最终失败于上述 kernel 47；不能把 kernel 42 当作其终止错误。
动态场景也完成该 kernel 的编译，之后因数值校验失败。
保存了 [静态 kernel](../../../artifacts/cicd/swin_matrix_20261020/training_kernel/static/kernel.py) 和
[动态 kernel](../../../artifacts/cicd/swin_matrix_20261020/training_kernel/dynamic/kernel.py)，两者函数体相同，
static axis metadata 不同。size hints 为 `z0=8, y1=14, x2=14, r3=512`，归约长度固定为 512，
计算包含 window indexing、梯度乘 weight 和两次 LayerNorm backward 归约。

所捕获候选采用 `--pure-simt --num-warps=8 --shared-mem-dynamic-size=122880`，
生成的 LLVM IR 约 1.64 MB。运行中的 BiSheng 进程持续占满一个 CPU 核心，未观察到 OOM。
独立重放 LLVM IR，加入 `-mllvm -debug-pass=Executions` 后，日志最后进入
`HiTPE Stall Cycle Refactor`，60 秒诊断限时内未离开该 pass。
[pass 日志摘录](../../../artifacts/cicd/swin_matrix_20261020/training_kernel/pass_trace_excerpt.txt)，
[探针结果](../../../artifacts/cicd/swin_matrix_20261020/training_kernel/pass_trace_result.json)。

原始 `-O2` 和 `-O0` 独立编译均未在 60 秒内完成；因此尚不能判断是 `-O2` 特有优化导致。
直接 LLVM 重放已绕过 Dynamo、Inductor 和 Triton 的前端编译，长耗时至少发生在 CANN 后端阶段。
独立限时探针未获得编译完成结果，也没有 CANN pass 内部源码，不能进一步认定内部算法缺陷、死循环或输出错误。
独立探针超时不代表整模型已超时；两项训练精度任务后来均越过该 kernel，已获得各自终态。

运行中的候选至少有一次在约 36 分钟的 LLVM 子编译后生成对象文件，随后同一 `bishengir-compile`
候选继续启动下一次 BiSheng 编译。不能因日志长时间未更新就判为死循环；候选总耗时与单个 LLVM 子进程耗时
应分别记录。

batch 128 的静态、动态训练性能任务也出现同类长时间编译，其入口名称为
`triton_per_fused__unsafe_view_clone_native_lay_44`，不能按编号与精度场景的 kernel 42 混为一项。
已从各自运行中的 CANN 编译临时目录保存
[性能场景 TTIR / LLVM IR](../../../artifacts/cicd/swin_matrix_20261020/performance_slow_kernel)，
便于后续直接重放；捕获时仍在编译，这不是模型最终失败的证据。
随后两项性能任务均成功完成，kernel 44 不应归类为最终编译失败。

## 动态训练精度：编译成功，数值校验失败

该场景运行到 5 步训练后，CSV 为 `fail_accuracy`，日志为 `allclose not within tol=0.02`；
捕获 3,468 个调用、1 个图、0 次 graph break，没有最终编译失败 kernel。
不能把它归为 UB overflow、CANN 慢编译超时或静态训练 kernel 47 的同一个失败。

首步 loss 已不同：eager 为 `0.5475974082946777`，compiled 为 `0.4442262053489685`。
后续 step 2–4 的 eager loss 为 `0.3560846448, 0.3080650866, 0.1982577890`，
compiled 为 `0.1378552616, 0.0620538704, -0.0357798375`。
runner 在两条路径前重置 seed、复制模型并重新初始化优化器；首步已不同，需先检查前向，不能只从反向 kernel 推断原因。

成功 backward trace 的对应 kernel 47 是线性 R 循环归约
`triton_red_fused__unsafe_view_clone_native_layer_norm_backward_permute_view_47`，
使用正确的 `%7` 地址表达式，与静态场景失败的多轴 persistent kernel 不同。
独立测试 46 次 Bernoulli 调用的 inplace/outplace 序列一致；小型 DropPath 在 eager、aot_eager、Inductor 下亦一致。
这些对照排除简单 Bernoulli API 替换这一解释。

### 已定位：persistent softmax 的归约 mask 被移除

使用成功 trace 中的 FX 前向和 compiled wrapper，以同一组生成输入、同一个 seed 1337 重放。
692 个返回项中，最早出现差异的已保存激活为第 188 项 `div`，即第一个 attention softmax；
前面的卷积、LayerNorm、window partition 对应激活在所用容差内一致。
这里使用 wrapper 生成的形状，浮点输入重新生成为 `N(0,0.05)`、CPU scalar 为 0.95、索引为 0，
是同输入的独立前向定界，不是原模型权重的训练重跑。
[逐返回项对照](../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel/forward_comparison.json)。

该激活来自 `triton_per_fused__softmax_add_clone_index_matm_13`。
逻辑归约长度为 49，`R3BLOCK_SUB=64`；代码直接对补齐后的 64 项执行 max、exp、sum，
没有在归约前用 `r3_mask` 排除无效项。
[原始 DSL](../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel/kernel.py)。

独立 kernel 在全零输入上得到每项 `1/64=0.015625`、每行和 `49/64=0.765625`，
而正确结果应为每项 `1/49≈0.02040816`、每行和 1。
只在 max 前增加 `tmp8 = tl.where(r3_mask, tmp8, float("-inf"))`，保留其余计算，
全零和同组 seed 0 随机输入均与 eager 一致，最大绝对误差为 0。
独立重放采用 `Z0BLOCK_SUB=X2BLOCK_SUB=1`、runtime block 16、grid 32，
用于隔离 DSL 的数值语义，未声称复用了整模型选中的 autotune 配置。
[数值证据](../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel/softmax_validation.json)。

安装版本 `torch_npu/_inductor/codegen/triton.py` 存在三处对应逻辑：

- `reduction()` 第 4086–4094 行在 persistent 路径收集 mask 时排除 r 轴。
- `filter_masks()` 第 4020–4035 行又跳过 persistent reduction 轴；独立对象对照确认输入 `r3_mask` 被删除。
- `reduction()` 第 4230 行附近直接令 `masked_value = value`，未为 max/sum 加归约默认值。

因此这是当前安装版本的 torch_npu Inductor codegen 问题，已生成不正确的 DSL，不能归因于 CANN 数值实现。
精确行号和安装文件 SHA256 见
[源码证据](../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel/codegen_source.txt)。
该安装文件与本地 `pytorch_new` checkout 不同，不能用本地文件的当前内容替代安装版本来推断行为。
诊断只修改独立 kernel / 生成代码副本，没有替换安装包，也没有验证完整训练链路修复。

同一前向 wrapper 中 13、23、31、41、49、59、66 共 7 类 softmax kernel 都有相同的归约 mask 缺失。
在生成代码副本的每个 max 前，仅将其输入用 `r3_mask` 置无效项为 `-inf`，其余计算保持不变，
重复同组输入前向：最终 logits 最大绝对误差从 `0.06009545` 降至 `5.0664e-7`，
第一个 softmax 的误差从 `0.00508456` 降至 `5.5879e-9`，两者均通过 `atol=2e-5,rtol=2e-4`。
仍有 24 个中间返回项超过该较严格容差，最大绝对误差 `5.6267e-5`；不能声称所有中间值或完整训练均通过。
[前向诊断副本结果](../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel/forward_masked_comparison.json)。

## 与此前结果的关系

[2026-10-19 的记录](cicd-static-compile-failures-20261019.md) 只确认静态推理性能 batch 128 通过，
所抽 kernel 19 的失败候选可由其他候选替代。本轮 batch 8 的 kernel 9 是所有候选失败导致的模型终止错误，
两者不矛盾。本轮材料与此前非致命候选分目录保留。
