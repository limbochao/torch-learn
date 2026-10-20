---
title: 四模型 Inductor 修复审查与复测
published: false
---

# 四模型 Inductor 修复审查与复测

## 范围和基线

本轮针对 [四模型完整矩阵](four-model-matrix-20261020.md) 的精度与编译失败实施框架修复。
模型为 AlbertForMaskedLM、GPT2ForSequenceClassification、nfnet_l0、BERT_pytorch。
基线 32 个场景中 18 PASS、14 FAIL。基线的 kernel、模型结果和独立 DSL 对照实验保留在
[`artifacts/cicd/four_model_matrix_20261020`](../../../artifacts/cicd/four_model_matrix_20261020/)。

环境为 NPU_A5 的 TorchBench 2.13 专用容器环境，FP32、关闭 ACLgraph。
静态/动态 shape，各自包含 inference/training × performance/accuracy。
performance 使用模型默认 batch size，不能用 accuracy 的小 batch 结果代替。
每个编译候选的超时为 60 分钟；performance PASS 只表示成功执行并产生正 speedup，不证明精度。

最终调试分支：`debug/cicd-all-fixes-20261020`，head `81f8f87a4e`，建立于 Ascend master `b9f98688f2`。
TA 仅做定界和独立 kernel 抽取，不修改 TA，也不在 Inductor 中添加针对 TA 的绕过。
下文结果是持续更新的验证记录，尚未写入结果的项目不表示通过。

## 分支组织

| 分支 | 修改类别 | 依赖 |
| --- | --- | --- |
| `bugfix/codegen-reduction-mask` | persistent reduction 的尾部 mask 和单位元 | master |
| `bugfix/codegen-reduction-axes` | 多轴规约保留输出维度、布局和 CSE shape | mask 分支 |
| `bugfix/codegen-negative-index` | 调试开关下的负索引语义与 no-wrap bounds | master |
| `bugfix/codegen-load-liveness` | reduction load 与消费它的计算相邻生成 | master |
| `bugfix/tiling-resource-budget` | 按 kernel type 更新搜索边界、保护可继续切分的轴 | master |

mask 原三个提交通过 interactive rebase 合为一个，合并前后 tree 完全相同。
旧 head 保存在 `backup/codegen-reduction-mask-before-squash-20261020`。
随后移到最新 master；mask 分支自身只含一个修复提交。
axes 采用有序依赖，复用 mask 布局构造，避免两个分支分别维护同一段代码。
各分支精确 revision 与 patch 保存在
[`branches.json`](../../../artifacts/cicd/model_fixes_20261020/branches.json)
和同目录 `patches/`，最终调试分支只包含上述 5 类修复，没有 TA 修改。

模型矩阵初版基于 master `bf4abcfc94`，整合分支随后移到 `b9f98688f2`。
新增 master 提交只修改测试以及 `decomposition.py` 的 `triton_experimental` RNG 注册路径，
本轮使用的 `triton` loader 不调用该函数；三个修复文件内容完全一致。
最新分支的独立包目录另含最新 `decomposition.py`，用于最终数值回归和 NFNet 动态场景复核。
该说明用于区分源码版本，不能把旧版运行结果标为新版实际运行。

## 修复与原失败的对应关系

| 修复类别 | 可以解释或解决的原问题 | 验证边界 |
| --- | --- | --- |
| reduction mask | GPT2 两个推理精度失败，BERT 四个精度失败中的前向部分 | GPT2、BERT 各 8 项通过；BERT 静态训练还需要 axes 修复 |
| reduction axes | BERT 静态训练反向 bias gradient 错误 | 同输入单 kernel、静态训练整模型通过；动态反向本来使用另一生成路径 |
| negative index | dump/check 开关使负索引失去 wrap 语义，另有 no-wrap bounds 传播错误 | 专门回归通过；本轮模型使用的索引未证明触发此缺陷，不将它计入原 14 项失败的修复数 |
| load liveness + resource tiling | Albert 四个训练编译失败，以及复测新暴露的动态 softmax 资源失败 | 原配置 DSL 对照和整模型 8 项通过 |
| kernel type 资源边界与候选搜索 | NFNet 四个动态场景的 pooling 编译失败 | 动态推理 performance/accuracy 已通过；两个动态训练场景仍在复测 |

前两类错误在 generated DSL 中已违反 tensor semantics，应由 Inductor 修复；负索引也在 DSL 生成前
改变了语义。后两类针对工作集排布和合法候选覆盖，属于 Inductor NPU 适配层资源管理。
下层对超出 UB 容量的 tile 拒绝编译，本身不能证明下层有 bug。

## Persistent reduction：必须在计算后重新应用 mask

归约 lane 指一次归约 tile 中的一个元素位置。例如隐藏维度 768 在统一 SIMT 路径上扩展到 1024，
多出的 256 个位置无效。即使 load 的 `other=0` 正确，`(value - mean) ** 2` 在这些位置也不再为零。
因此直接 `tl.sum(square, ...)` 会把无效位置计入方差。

原实现同时在 `IterationRangesEntryNPUIndex._codegen_mask`、`filter_masks`、`reduction` 中
省略 persistent reduction 的 mask，并直接使用 `masked_value = value`。
修复改为保留无法证明恒真的轴 mask，在最终规约前用 `tl.where` 选择规约单位元：
sum 使用 0，prod 使用 1，max 使用负无穷，min 使用正无穷。
整数极值按实际 value dtype 转换，空 mask 不生成空元组条件。

恒真 mask 的消除使用实际生成的 `BLOCK_SUB`：SIMD 可以使用非二次幂静态块，统一 SIMT 使用
`next_power_of_2`。只有正静态长度可被实际块大小整除时才消除；动态长度、向量化 split、
关闭 mask 优化的情形保留条件。静态块大小计算由常量生成和证明共同使用，避免规则漂移。
mask 根据 value 的 axis 顺序重建形状，并执行相同的 permute/reshape。

CUDA 参考：PyTorch `cf30153c4c1` 的 `torch/_inductor/codegen/triton.py::reduction`，
在 persistent 分支先取得 `ir.Reduction.default_value`，再通过 `_mask_value`/`where_cond` 处理。
`_has_constant_mask` 用可整除证明消除条件。NPU 的数学语义与它一致；NPU 的轴布局和非二次幂块
不能直接照搬 CUDA 的线性 index 或二次幂假设。
NPU 专用 SIMD Welford 路径仍独立处理，它使用完整静态规约大小，不能直接套用普通 tuple 归约逻辑。

该问题应由 Inductor 修复：生成的 DSL 在进入 TA 之前已把无效 lane 纳入规约。
可解决 GPT2/BERT embedding + norm 方差误差，以及 fused sum/prod/min/max 的同类 padding 问题。
TA masked load 是否还存在必须另外验证，不能仅从模型 PASS 推断。

## 多轴规约：只压平规约维，保留 channel 维

BERT 静态反向的 bias gradient 需要对输入 `[48,128,64]` 进行
`reshape(4,12,128,64).permute(0,2,1,3).reshape(512,768).sum(0)`。
原 `ReductionAnalysis.dense_size_str` 在 non-contiguous multi-reduction 情况把 R2、R3 和 X5 全部
压成一维，生成标量 sum，再广播到每个 channel。

修复以实际生成 value 的 `golden_var_list` 为布局依据，把 preserved axes 排在前面、
reduction axes 排在后面，仅合并后者，并在规约后恢复原布局的 broadcast 维。
规约 dim 等于 preserved axes 个数。CSE 是已生成表达式的复用缓存，其 shape 也要与 broadcast、packed value 和 accumulator
实际生成形状一致，否则后续 shape 推导仍会沿用错误元数据。
删除了 `analyze_reduction_dim` 中已被函数前部 return 覆盖的重复 multi-reduction 分支。

CUDA 的数学参考是保留输出维、只对目标 reduction dim 规约；其线性索引模型没有 NPU
`golden_var_list`/`dense_post_reduction_list` 的直接同名实现。因此对齐的是 tensor semantics，
不声称这段 NPU layout 算法来自 CUDA。此问题在 DSL 的 reshape 中已经形成，归属 Inductor。

修复后真实 BERT 静态训练 DSL 先生成
`tmp0.permute([2, 0, 1]).reshape([X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB])`，
再 `tl.sum(tmp3, 1).reshape(1, 1, X5BLOCK_SUB)`；X5 始终保留。
完整生成函数见
[`bert_bias_gradient.py`](../../../artifacts/cicd/model_fixes_20261020/generated/bert_bias_gradient.py)。

## 负索引：调试开关不能改变算子语义

原 `NPUIndexTritonKernel.__enter__` 内的 `CSEProxy.indirect_indexing`，在设置
`INDUCTOR_ASCEND_DUMP_FX_GRAPH` 或 `INDUCTOR_ASCEND_CHECK_ACCURACY` 时强制 `wrap_neg=False`。
例如长度 8 的 tensor 中，索引 -1 本应先变成 7；dump/check 开关不能取消这个转换。
修复删除环境变量分支，保留调用方传入的 `wrap_neg`。

审查还发现 no-wrap 的 bounds 传播与 CUDA 不一致：没有实际包装负数时，不能把 bounds 当作已加 size。
否则可能错误地消除负数的 lower-bound check。修复在 `wrap_neg=False` 时保留原 bounds，
只有实际 wrap 时才计算转换后的范围，并保留 dtype/shape 元数据。
参考为相同 PyTorch revision 的 `torch/_inductor/codegen/common.py::CSEProxy.indirect_indexing`。

## 编译失败：load 生存期与 tiling 是两个修复类别

### Albert 多路 LayerNorm 参数梯度融合

基线 kernel 融合 12 路梯度贡献，81 个 input load 被集中写到 `self.loads`，随后才写计算。
大量 tile 同时存活，导致 UB（NPU 的片上 unified buffer）容量不足。
独立 DSL 实验已证明：将 load 移到首次消费前，原来失败的部分 tile 可以编译并通过梯度数值对比。

`codegen-load-liveness` 在 reduction 内将输入 load 写到 compute buffer，保持与消费运算的生成顺序；
可提升到循环外的公共 load 仍保留在 prefix。indirect load 依赖计算结果，也进入 compute。
依赖 load 的 layout transform 写入同一 buffer，避免出现先使用后定义。
这是 NPU 的资源使用修复；CUDA 有不同的寄存器/共享内存资源模型，不存在可原样移植的 UB 调度规则。
CUDA `TritonKernel.get_load_buffer()` 同样把 indirect/tmpmask load 放在 compute，
将循环不变量放在 body；NPU 进一步把普通 reduction load 与消费计算相邻生成，以缩短 UB 生存期。
该资源策略不是照抄 CUDA 对普通 load 的默认安排。

### 搜索范围需要覆盖可编译 tile

`TileGenerator` 原来把 pointer 数量截断到 3，且切换 SIMD/SIMT 模式后没有更新容量和 stop 阈值。
NFNet 动态 avg_pool2d 的 pure SIMT 候选限制为每 tile 8 KiB，但旧搜索在 4096 个 FP32 元素处就停止，
因此所有搜索到的 SIMT 候选都被拒绝，只剩编译失败的 template 候选。

修复按实际 pointer 数量估计工作集，在 mode 切换时重新计算搜索范围，并让 SIMT 搜索进入 8 KiB 限制内。
每种 mode 独立清空 configs，修正原 `self.config` 的拼写错误，避免混入上一个 mode 的候选。
persistent/no-loop 轴不能继续切分；即使 pointer 估计过于保守，也保留最小合法 tile 供编译器验证。
该估计用于搜索，不声称可以精确预测编译器的 UB 分配。

最新整合分支上的 NFNet 动态推理 performance 已通过，batch=128，speedup=1.017493。
动态推理 accuracy 也已通过，在两台空闲设备上独立执行的相同场景均为 `pass_accuracy`，不重复计入总数。
其真实 autotune cache 中，三个 dynamic pooling kernel（后缀 21、32、40）选中
`XBLOCK=512, compile_mode=simt_only`。原先这里没有任何合法 SIMT 候选；这次通过的是修复后的
正常候选枚举与选择，未向模型注入固定配置。配置记录见
[`nfnet_pooling_selected_v4.json`](../../../artifacts/cicd/model_fixes_20261020/nfnet_pooling_selected_v4.json)，
原失败 kernel 对应的生成函数见同目录 `generated/nfnet_pooling_v4.py`。

复测又暴露 Albert 动态训练 performance 的 softmax kernel：X2=512、R3=512，约需 33.6 Mbits UB。
原因是 `rebuild_dynamic_axis` 在决定免循环轴时只看 X2 大小，漏算完整 persistent R3，
并以 `axis_name == 'r'` 判断实际名为 `r3` 的轴。修复先计入所有 persistent reduction 轴，
再判断哪些低维轴可以免循环，同时正确识别带编号的 reduction 轴。
这样 X2 保留为可切分轴，不需要绑定模型名、shape、kernel 名或手工指定 XBLOCK。

通过 `inductor-kernel-compile-debug` skill 重新捕获该 kernel，并从 journal 的 record 1 导出
compile-only 文件。独立导出与精简后的
[`albert_softmax_compile.py`](../../../scripts/repro/cicd/model_fixes_20261020/albert_softmax_compile.py)
均在捕获目录之外运行，默认 X2=512 复现 UB overflow（本次独立编译报告需要 41,984,000 bits，
可用 2,031,616 bits）；仅改为 `--xblock 8 --check` 后编译与 eager softmax 数值比较通过，
最大绝对误差 `6.98e-10`。生成函数保持原样，只移除导出工具为其它版本/launch 场景准备的公共 helper。
这是 DSL 对照实验；框架修复通过候选枚举寻找合法 tile，没有在框架中固定 X2=8。

Albert 动态训练 performance 修复后已通过，batch=4、speedup=1.017143。
结合初版其余 7 项 PASS，Albert 原有 4 个训练编译失败均已解决。
数值验证由 accuracy 场景承担，performance 的单次 speedup 不作为稳定性能提升结论。

## TA masked indirect load：独立定界

独立复现脚本为
[`masked_indirect_load.py`](../../../scripts/repro/cicd/model_fixes_20261020/masked_indirect_load.py)。
它直接运行一个 Triton kernel，没有 `torch.compile`、Inductor 和任何 reduction。
table 的有效 768 个元素设为 3，额外合法分配的元素设为 9；输出保留全部 1024 个 lane。
mask 为假时输出必须为 `other`，包括 0 和非零 -7 两种对照。
所有地址都有合法 backing storage，错误表现不依赖越界读碰巧读到什么。

基线证据：TTIR 中 `tt.load` 同时携带 mask 和 `other=0`，ttadapter IR 中对应的非结构化访问
变成覆盖全部 1024 元素的无条件 `memref.load`。
template 的尾部返回 9，pure SIMT 返回 0。由此可以在 Inductor 之外证明 TA lowering 不满足 load 语义。
源码参考为 `BlockDataParser::rewriteAddPtrToUnstrucMemAcc` 和
`LoadConverter::continueModifyFromAddPtrConverter`；安装二进制与本地源码 revision 尚未一一匹配，
因此实际归属证据以运行时 TTIR → ttadapter 转换和独立结果为准。

修复 Inductor 后，GPT2 全部 8 个场景通过，包含原先失败的静态/动态推理 accuracy。
在最新调试分支的独立包目录中运行本用例，template 的 `other=0` 和 `other=-7` 都仍返回尾部 9，
pure SIMT 的两个对照分别正确返回 0 和 -7；有效 768 个位置四种情况都正确。
TTIR 仍保留 mask/other，ttadapter 仍出现覆盖 1024 个元素的无条件 `memref.load`。
结果及 IR 见
[`ta_after_inductor_v4`](../../../artifacts/cicd/model_fixes_20261020/ta_after_inductor_v4/)。

因此两者是独立缺陷，在模型症状上存在耦合：新 Inductor 会在每次 sum/max 等操作前重新屏蔽
无效 lane，既保护均值，也保护方差，消除了 TA 错误尾值对这些规约结果的贡献。
这解释了 GPT2 可以恢复精度，但不代表 `tl.load` 的 mask/other 语义已修复。
独立用例完全不运行 Inductor，仍能复现，从而把剩余问题定界到 TA lowering。
依照任务边界，不改 TA，也不通过更换 kernel type 来规避该问题。

## 当前验证记录

数值回归入口：
[`validate_codegen.py`](../../../scripts/repro/cicd/model_fixes_20261020/validate_codegen.py)，
批量入口为同目录 `run_regressions.py`。

| 验证 | 当前结果 |
| --- | --- |
| mask codegen 单元检查 | 4/4 PASS |
| negative no-wrap bounds 单元检查 | 1/1 PASS |
| tiling resource 单元检查（含 softmax 轴保护） | 3/3 PASS |
| 静态/动态 embedding + norm、sum/prod/min/max、argmin/argmax、int32/int64 | 22/22 PASS |
| 静态/动态 BERT bias gradient、分离规约轴 | 4/4 PASS |
| 静态/动态负索引，none/dump/check 开关（包含 no-wrap bounds 修改） | 6/6 PASS |
| Albert 初版完整矩阵 | 7/8 PASS；动态训练 performance 暴露 softmax tiling 问题 |
| Albert softmax 通用修复后 | 动态训练 performance PASS；合计 8/8 PASS |
| GPT2 完整矩阵 | 8/8 PASS |
| BERT 完整矩阵 | 8/8 PASS |
| NFNet 完整矩阵 | 推理 4/4 PASS；训练四项仍在复测 |
| TA 独立 kernel 在最新 Inductor 修复环境下 | template 两个 other 对照仍失败；pure SIMT 两项通过 |

最新分支的 32 项数值回归全部通过，10 个独立子进程均正常退出；结果保存在
[`regressions_v4`](../../../artifacts/cicd/model_fixes_20261020/regressions_v4/)。
argmin/argmax 使用精确比较；prod 用 `value+0.99`，使 padded lane 上的表达式不是单位元 1。
这避免了原先 `value+1` 的对照在缺少 mask 时也可能通过的问题。

测试使用独立包目录加载本轮三个修复文件，v4 另加载 master 的 decomposition.py；原环境安装文件不变。
原安装 decomposition.py 与首轮 master 的差异也仅位于未使用的 TE 注册函数内，
为 matmul_backward 是否已注册的 guard，不影响本轮 triton loader 的结论。
完整模型入口为同目录 `run_matrix.py`，通过 `MATRIX_MODELS`、`MATRIX_CASES` 选择复测项。
阶段结果、源码哈希、分支提交与 generated DSL 保存在
[`artifacts/cicd/model_fixes_20261020`](../../../artifacts/cicd/model_fixes_20261020/)。
其中 `model_matrix.json` 合并各阶段最新结果，原始阶段结果分别保留；进行中状态不能作最终结论。

本轮覆盖 FP32 和上表注明的整数回归；未将其推广为所有 dtype、所有模型的验证结论。
tiling 工作集估算是搜索启发式，实际合法性仍由编译器确认；单次性能记录用于确认场景执行成功。
