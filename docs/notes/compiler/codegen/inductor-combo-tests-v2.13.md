---
published: false
---

# 2.13 Combo/foreach 最新结果与问题分析

<a id="validation213"></a>

<!-- combo-section:validation213:start -->
## 最新执行结果

2026-11-13 在 NPU_A5 Docker 的 `pta_213_case` 环境，按固定 CSV 全量重新收集和执行 105 个源方法、464 个参数实例。CSV 的 116 行包含 11
个随外层方法验证的嵌套函数。实例结果为 **419 PASS、35 FAIL、10 SKIP**；没有抽样或复用旧执行结果。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 2.10 重合范围 60 个方法 | 57 | 3 | 0 |
| 2.10 重合范围 413 个实例 | 410 | 3 | 0 |
| 重合范围外 45 个方法 | 6 | 30 | 9 |
| 重合范围外 51 个实例 | 9 | 32 | 10 |
| 全量 105 个方法 | 63 | 33 | 9 |
| 全量 464 个实例 | 419 | 35 | 10 |

### 验证版本与范围

torch_npu `2.13.0.dev20261113`，提交 `44555141711c0c825c5a201e303dcadad1f5fcb4`。

PyTorch `2.13.0+cpu`，提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`；Triton `3.6.0`。

物理卡 1、2、3，每卡两个进程，Inductor/Triton 缓存和日志独立；没有生产代码 overlay。

测试分支 `fix/combo-213-210-overlap`，提交 `a3cfe564cd60cb83f4c81bab5dcd388305992ff2`，main 基线
`91435bdf732e772c3240329797a9a29166ce533a`。

执行时间：北京时间 2026-11-13 18:58:49 至 2026-11-13 20:11:18。

Community 分支只包含与既定 60 个重合方法相关的适配，修改 3 个文件、30 个方法；范围外测试修改未迁入该分支。仅本轮验证副本加上 TMA 和三个 PDL 方法的硬件 SKIP，方法
body 和断言保持原内容。foreach 按已确认口径保留数值、mutation、布局和梯度检查，NPU 不要求 CUDA 单 kernel／FOREACH 专属结构。

执行前后 301 个测试文件、588 个 Torch/NPU Inductor Python 文件哈希一致；其中 194 个 NPU Inductor Python
文件与实际包提交逐项一致。三卡基础计算均通过。全部分片 JUnit 已独立重新解析，464 个实例无遗漏或跨分片重复。

[2.13 PASS 清单][pass213]保留 2.10 重合范围口径，当前为 410 个实际 PASS 实例，在 2.13 NPU 测试目录执行；[全范围
PASS][all-pass]另存在本轮证据目录。2.10 原生 397 条清单保持不变。既定 60 方法适配范围与 2.10 原生 54 方法清单的口径不同，本轮用 [60
方法范围清单][overlap]精确划分。

`test_combo_autotune_grouping_by_metadata_fingerprint` 直接调用社区 helper 和模拟 metadata；其 PASS 证明该 helper
的分组规则，本轮未验收真实 NPU 成员 metadata 到该分组的连接。

[当前 CSV][csv]保持原文件名、116 行顺序及五列，所有行均来自本轮结果。[方法与逐实例报告][jobs]、[完整环境和分片证据][evidence]、[JUnit
交叉核对][junit]及[Community 范围审计][scope]可追溯当前结论。

<!-- combo-section:validation213:end -->

<a id="analysis213"></a>

<!-- combo-section:analysis213:start -->
## 当前问题与修复顺序

重合范围外共有 39 个非 PASS 方法：34 个非硬件 FAIL／SKIP 全部进入下表，5 个硬件／平台协议方法保留 SKIP。P0 1 项先定位本轮多轴归约精度失败；P1 18
项恢复有效测试入口、eager 布局、metadata、统计和追踪；P2 14 项处理独立分块、成员调优、融合目标和 FTZ 数学路径；P3 1 项补齐峰值门控的有效验收。

以下是修复顺序与验收要求，本轮未实现生产修复。原测试提前 SKIP 是实际执行状态，不代表功能通过，也不作为后续规避目标的理由；非硬件旧 skip 应移除，未完成的目标保持 FAIL。当前 PASS
不保留旧失败分析。源码分析固定在实际安装的 NPU/PyTorch 提交，不把其他开发分支或最新 master 当作本轮运行代码。

### metadata 和统计分别缺什么

`optimize_mem` 两个方法检查同一个字段的两个图阶段；七个统计方法检查多种计数、布局信息和成本出口，不能写成“缺七种统计”。`combo_subkernel_meta`
已有真实成员轴、尺寸和模式，它不等于社区承载下表统计的 `inductor_meta_i` 字典。

| 字段／出口 | 具体作用 |
| --- | --- |
| `optimize_mem` | 推理／反向 True，训练前向 False，约束 autotune 输入备份和恢复。 |
| `reduction_hint_i` | 逐成员 INNER/OUTER 访问方向，用于候选分类和观察。 |
| `num_load` | 去除 CSE 缓存复用后实际 load 数，描述访存工作量。 |
| `num_store` | 保留的实际写回数；NPU store 路径目前未统一递增此计数。 |
| `num_reduction` | 归约 primitive 数，用于候选与工作量统计。 |
| `autotune_hints` | 如 ONE_ELEMENT_PER_THREAD 的候选提示，须来自真实成员。 |
| `atomic_add_found` | 是否存在原子加，约束候选及写回行为。 |
| `has_loadstore_with_contiguous_rdim` | 实际归约维是否连续，参与归约候选过滤；不是确定性支持开关。 |
| `no_x_dim` | 成员是否缺独立 X 维；需按 NPU 真实 split 轴和成员布局转译。 |
| `tiling_scores` | 各轴 tiling 的评价信息；需与 native 候选语义对应。 |
| `kernel_num_gb` | 实际访存字节的 GB 估算，供成本和带宽报告使用。 |
| `kernel_flop` | 按统一规则估算计算量，Combo 聚合成员值。 |
| benchmark `num_gb` | 独立 benchmark 带宽分子；当前 NPU Combo 模板收到硬编码 0。 |
| `add_persistent_rblock` | 成员增加大 R 候选的资格；字段缺失不证明 NPU 没有 persistent 归约。 |
| `disable_ftz` | 控制非正规数保留，需要编译器与数学路径消费，不能只导出文字。 |

计数与布局字段先由真实 load/store/reduction 生成，再由 standalone 与 Combo 统一导出；访存／FLOP 先形成有效的 standalone 估算基准，再聚合
Combo；成员 fingerprint 分组依赖完整 metadata。成员独立 block ABI／dispatch 接通后，才有独立默认候选、顺序调优和最大成员优先的 coordesc。NPU
已有消费真实成员轴约束的共享搜索，缺这些导出字段不证明整个归约／调优能力缺失。共享候选不能替代独立成员目标，补 0、空字典或日志字符串也不算完成。

有效 store 计数的缺口位于 Native Index 的 `CSEProxy.store/store_reduction`；no-linear 普通 kernel 继承社区
CSE，已有的计数逻辑与 NPU metadata 出口需分别核对。`kernel_num_gb` 是逻辑读写量的 GB 估算，不是硬件 profiler 测得的传输量。

当前安装的 Triton Ascend 使用 `NPUOptions`，其 [真实字段列表][triton-options] 未声明
`disable_ftz`；[原始编译器源码][triton-compiler] 已保存。这项验收还需核对通用编译选项及下层数学路径的实际消费，不能从 Python 字段缺失直接推定硬件能力。

本轮多轴归约的 P0 精度失败走 no-linear 普通 kernel；其 [原始生成
wrapper][accuracy-wrapper]和[按执行时间／输入布局匹配的来源][accuracy-source]已保存。该路径继承社区多轴 reduction DSL，再由 NPU
heuristic／编译器执行；不能按 Native Combo 的统计出口缺失来定界这项精度问题。

### 范围外逐用例修复顺序

| 顺序 | 优先级 | 用例 | 当前结果 | 具体待修内容 |
| ---: | --- | --- | --- | --- |
| 01 | P0 | [test_combo_kernel_per_config_subkernel_r0_r1](#repair-01) | FAIL | 非连续多轴 sum 的 FP32 结果超出原容差 |
| 02 | P1 | [test_foreach_basic](#repair-02) | SKIP | 取消旧 skip，验收图输入 foreach 节点的距离为 0 |
| 03 | P1 | [test_foreach_after_extern](#repair-03) | SKIP | 取消旧 skip，验收 mm 后 foreach 消费者的依赖距离 |
| 04 | P1 | [test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu](#repair-04) | SKIP | 开放 NPU 动态归约入口，保留编译期 autotune 目标 |
| 05 | P1 | [test_persistent_reduction_size_hint](#repair-05) | FAIL | 按 NPU 真实成员轴核对 persistent size_hints |
| 06 | P1 | [test_single_combo_kernels](#repair-06) | FAIL | 用原生 grid 验收两个 standalone，覆盖 benchmark 开关 |
| 07 | P1 | [test_combo_kernel_yz_overflow](#repair-07) | FAIL | 适配 NPU profiler 格式，验收大尺寸真实 launch grid |
| 08 | P1 | [test_combo_kernel_per_config_subkernel_block_size](#repair-08) | FAIL | 先修 eager channels_last contiguous，再验独立分块 |
| 09 | P1 | [test_combo_inductor_meta_has_optimize_mem](#repair-09) | FAIL | 补推理 optimize_mem=True 及输入保护消费契约 |
| 10 | P1 | [test_combo_inductor_meta_optimize_mem_false_in_training_forward](#repair-10) | FAIL | 补训练前向 optimize_mem=False 及输入保护契约 |
| 11 | P1 | [test_combo_kernel_per_subkernel_reduction_hint](#repair-11) | FAIL | 导出成员归约方向：行 INNER、列 OUTER |
| 12 | P1 | [test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields](#repair-12) | FAIL | 导出三成员真实计数、autotune 提示及 atomic 标志 |
| 13 | P1 | [test_combo_reduction_combo_grid_meta_has_per_subkernel_fields](#repair-13) | FAIL | 导出 sum/mean 成员真实 load/store/reduction 计数 |
| 14 | P1 | [test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel](#repair-14) | FAIL | 导出每成员连续归约维布局标志 |
| 15 | P1 | [test_combo_per_kernel_inductor_meta_matches_standalone](#repair-15) | FAIL | 对齐 standalone/Combo 计数、布局和 tiling_scores |
| 16 | P1 | [test_combo_kernel_num_gb_and_flop_match_standalone_sum](#repair-16) | FAIL | 补 Combo 访存 GB/FLOP 聚合及有效 standalone 基准 |
| 17 | P1 | [test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth](#repair-17) | FAIL | profile 模式输出真实 Combo kernel_num_gb |
| 18 | P1 | [test_benchmark_combo_kernel_emits_real_num_gb](#repair-18) | FAIL | benchmark num_gb 当前为 0，需真实访存 GB |
| 19 | P1 | [test_triton_kernel_to_post_grad_tracing_combo_kernel](#repair-19) | FAIL | 补真实 Combo 到 post-grad FX origins 的映射 |
| 20 | P2 | [test_combo_kernel_store_mask](#repair-20) | SKIP | 取消旧 skip，验收动态 mean Combo 的最终 store mask |
| 21 | P2 | [test_fuse_mix_order_reductions_combo_kernels](#repair-21) | FAIL | 支持共享输入的行／列归约联合执行目标 |
| 22 | P2 | [test_combo_kernel_scalar_store_broadcast](#repair-22) | FAIL | 支持标量写回、广播消费者与向量归约的合法组合 |
| 23 | P2 | [test_combo_kernel_no_bench_stitched_config](#repair-23) | FAIL | 接通三成员独立默认 X 分块及 body/grid/launcher |
| 24 | P2 | [test_combo_kernel_no_bench_persistent_reduction](#repair-24) | FAIL | 保留完整 R，接通 persistent 成员独立默认 X 分块 |
| 25 | P2 | [test_combo_kernel_no_bench_carve_out](#repair-25) | FAIL | 验收 bucketize 从不合适的默认 Combo 配置中分离 |
| 26 | P2 | [test_combo_kernel_max_autotune](#repair-26) | FAIL | 实现激活成员独立候选与完整 Combo 顺序调优 |
| 27 | P2 | [test_combo_kernel_max_autotune_with_reduction](#repair-27) | FAIL | 实现归约成员独立候选与完整 Combo 顺序调优 |
| 28 | P2 | [test_combo_autotune_many_subkernels](#repair-28) | FAIL | 实现六成员独立候选与顺序调优 |
| 29 | P2 | [test_combo_autotune_grouping](#repair-29) | FAIL | 按真实成员尺寸和 heuristic 输入划分调优组 |
| 30 | P2 | [test_combo_max_persistent_rblock_gated_on_per_subkernel](#repair-30) | FAIL | 区分共享／独立 X 分块的真实 R 资源门控 |
| 31 | P2 | [test_combo_kernel_coordesc_tunes_largest_subkernel_first](#repair-31) | FAIL | 独立成员坐标下降先调工作量最大的成员 |
| 32 | P2 | [test_combo_rms_norm_forwards_add_persistent_rblock](#repair-32) | FAIL | 接通 RMSNorm 成员大 R 候选资格及消费者 |
| 33 | P2 | [test_combo_triton_meta_has_disable_ftz](#repair-33) | FAIL | 接通 FTZ 字段、编译选项与非正规数数学路径 |
| 34 | P3 | [test_combo_kernel_peak_memory_wide_resnet](#repair-34) | FAIL | 补真实 gate 候选及隔离编译后的运行峰值验收 |

<a id="repair-01"></a>

#### 01 P0 `test_combo_kernel_per_config_subkernel_r0_r1`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：保留 permute 后的非连续 (16,32,64) 输入，sum(dim=(1,2)) 的 compiled 结果满足原精度。

当前失败／跳过：原数值比较失败：报错的 16 元素向量中 1 个输出超差，最大绝对误差 1.1920928955078125e-5，atol=1e-5。执行的是同一定义的两次普通归约
launch；后续数量断言未执行。

具体缺口／适配：本轮实际是 no-linear 普通多轴归约，一个 kernel 定义对两个输入调用两次，未形成 Combo。DSL 保留两条 R 轴、stride 地址与 FP32
累加；数值误差已观测，累积次序／tile／下层编译职责尚未定界。 本轮仅保存生成 wrapper 与布局 metadata，真实 tensor 值和 selected tile 未单独留存。

先修入口：先捕获并固定可复现的真实输入，用同一组输入对照 NPU eager 与该 no-linear 普通 kernel；定位
NPUNoLinearTritonScheduling/NPUTritonKernel、社区多轴 reduction DSL、NPU 候选 remap 和 Triton
数学路径。核对两条归约轴、partial loop、累加 dtype 和实际 selected tile，不能把修复归到 Combo 分区。 验收：原非连续输入与原容差通过，保留原归约元素集合和
dtype；不以放宽容差或跳过精度检查通过。数值通过后再验原分区目标。

证据：[E16][E16]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-02"></a>

#### 02 P1 `test_foreach_basic`

位置：`test_snode_runtime.py::InputDistanceTests`；本轮 SKIP 1。

功能目标：图输入上的 foreach_add 节点输入距离为 0。

当前失败／跳过：NPU does not support

具体缺口／适配：旧的无条件 NPU skip 挡住输入距离检查；这不是 FOREACH 分组融合验收。

先修入口：test_snode_runtime.py 的 decorator 和 _get_snodes 观测。 验收：取消旧 skip，确认 snodes 非空，再检查每个真实节点的
min/max_input_distance 为 0。

证据：[E10][E10]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-03"></a>

#### 03 P1 `test_foreach_after_extern`

位置：`test_snode_runtime.py::InputDistanceTests`；本轮 SKIP 1。

功能目标：mm extern 深度 0，后续 foreach_add 的真实依赖深度至少为 1。

当前失败／跳过：NPU does not support

具体缺口／适配：旧的无条件 NPU skip 挡住依赖距离检查。

先修入口：test_snode_runtime.py 的 decorator 和 _get_snodes 观测。 验收：保留原矩阵乘输入、非空节点、最小距离 0 和最大距离至少 1 的检查。

证据：[E10][E10]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-04"></a>

#### 04 P1 `test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu`

位置：`test_combo_kernels.py::ComboKernelDynamicShapesTests`；本轮 SKIP 1。

功能目标：不同 X 长度的 sum/mean/max 动态归约及编译期 autotune。

当前失败／跳过：requires xpu and triton

具体缺口／适配：requires_xpu 门控；主体没有 XPU 硬件 intrinsic，不能认作 NPU 功能不支持。

先修入口：测试设备门控及真实 NPU autotune/launch 观测。 验收：用 NPU 通用入口执行原动态形状约束及数值比较；保留编译期 autotune 目标。

证据：[E9][E9]、[E3][E3]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-05"></a>

#### 05 P1 `test_persistent_reduction_size_hint`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：两个 max/min persistent 成员的候选尺寸覆盖真实归约长度。

当前失败／跳过：native persistent heuristic 已匹配，但原固定 CUDA size_hints 检查失败。实际为 x0=1536、r1=32，共享 split extent
求和；最终数值比较尚未执行。

具体缺口／适配：原 FileCheck 固定 CUDA x/r0_ 字典；NPU 采用多轴 metadata 和 native heuristic。

先修入口：测试生成代码检查；NPUComboKernel.select_heuristics。 验收：核对 NPU persistent 模式、两个成员 16/32 的完整 R
及共享候选包络，保留数值比较。

证据：[E1][E1]、[E2][E2]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-06"></a>

#### 06 P1 `test_single_combo_kernels`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 2。

功能目标：尺寸差异过大的两个 pointwise 保持 standalone，覆盖 benchmark 开关。

当前失败／跳过：两种 benchmark 开关均在固定 Grid1D FileCheck 失败，实际普通 kernel 为 GridNpu；先前数值和原 kernel 计数检查已通过。

具体缺口／适配：Grid1D 是 CUDA 生成 ABI；实际定义数、benchmark probe 和执行次数必须分别核对。

先修入口：test_single_combo_kernels 的原生 NPU grid 和生成／launch 计数检查。 验收：两种开关都保留数值、两个独立计算和无 Combo 的目标，按真实
wrapper 与 launcher 验收。

证据：[E1][E1]、[E2][E2]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-07"></a>

#### 07 P1 `test_combo_kernel_yz_overflow`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：大尺寸 permute/clone 的 launch grid 不溢出。

当前失败／跳过：编译执行已完成；读取 NPU profiler 的 trace_json["traceEvents"] 时报 TypeError，实际 JSON 是列表，后续 grid
及最终数值断言未执行。

具体缺口／适配：测试使用 CUDA traceEvents 字典和固定三维 grid；NPU 采用一维 dispatch。

先修入口：测试真实 NPU launcher/grid 观测。 验收：保留大尺寸与数值检查，确认真实 NPU grid 的 X 合法、Y/Z 为 1，排除 autotune launch。

证据：[E1][E1]、[E9][E9]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-08"></a>

#### 08 P1 `test_combo_kernel_per_config_subkernel_block_size`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：八种 channels_last 转换的逐成员独立分块与实际 grid。

当前失败／跳过：先在 eager 的 contiguous(memory_format=channels_last) 报 NPU 仅支持 contiguous 格式；compiled、独立成员分块和
grid 检查均未开始。

具体缺口／适配：原 eager contiguous(memory_format=channels_last) 是前置阶段；独立成员 tile 和 CUDA profiler ABI 另有目标。

先修入口：先处理 NPU eager memory-format 路径，再接成员 block ABI 与原生 grid 观测。 验收：原 NCHW 转 channels_last 的
eager／compiled 都正确，再验独立分块与 launch；替代 stride 不代表布局已支持。

证据：[E1][E1]、[E9][E9]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-09"></a>

#### 09 P1 `test_combo_inductor_meta_has_optimize_mem`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：推理图显式传 optimize_mem=True，声明 autotune 输入备份／恢复的内存优化资格。

当前失败／跳过：推理图 relu/sigmoid 数值比较通过，随后缺 optimize_mem=True；输入备份／恢复的真实消费契约尚需另验。

具体缺口／适配：NPU Combo 未显式导出 optimize_mem；native benchmark 直接 clone，helper 尚非其消费路径。

先修入口：NPUComboKernel.jit_line、NPUCachingAutotuner 的输入备份／恢复调用链。 验收：显式推理字段正确；如果接通内存优化消费者，同时验证
mutation、alias 和恢复。

证据：[E2][E2]、[E5][E5]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-10"></a>

#### 10 P1 `test_combo_inductor_meta_optimize_mem_false_in_training_forward`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：训练前向显式传 optimize_mem=False，防止输入保存／恢复破坏训练状态。

当前失败／跳过：训练前向数值比较通过，随后缺 optimize_mem=False；本方法未进行反向梯度和 autotune 输入恢复验收。

具体缺口／适配：训练前向字段未导出，runtime 缺省为 True；当前 native clone 路径不能据此认定输入已污染。

先修入口：NPUComboKernel.jit_line 的图阶段值及 NPUCachingAutotuner 输入保护。 验收：训练前向 False，推理／反向另行核对；保留数值、mutation
和梯度保护。

证据：[E2][E2]、[E5][E5]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-11"></a>

#### 11 P1 `test_combo_kernel_per_subkernel_reduction_hint`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：分别记录行归约 INNER 与列归约 OUTER 的成员访问分类。

当前失败／跳过：行／列归约数值比较通过，随后缺 reduction_hint_0；INNER/OUTER 的成员分类检查未完成。

具体缺口／适配：jit_line 仅传首成员 hint，combo_grid_meta 未输出 reduction_hint_i；native 已有成员轴约束。

先修入口：NPUComboKernel.combo_grid_meta／autotune_subkernel_meta 及候选评价入口。 验收：按实际
stride/归约轴输出两个真实分类，保留数值；分类影响优化时还需验证真实候选与 winner。

证据：[E1][E1]、[E2][E2]、[E3][E3]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-12"></a>

#### 12 P1 `test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：三个 pointwise 成员分别带有效 load/store/reduction 计数、autotune_hints 和 atomic_add_found。

当前失败／跳过：relu/sigmoid/tanh 数值比较通过，随后缺 inductor_meta_0；成员真实计数、提示和 atomic 标志的检查未完成。

具体缺口／适配：缺 inductor_meta_i；NPU CSEProxy 的有效 store 路径也未递增 num_store。

先修入口：NPU CSEProxy.store／store_reduction、统一 standalone 与 Combo 的统计出口。 验收：逐成员核对真实
load/store、reduction=0、优化提示及 atomic 标志，不能填 0 或空字典。

证据：[E2][E2]、[E4][E4]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-13"></a>

#### 13 P1 `test_combo_reduction_combo_grid_meta_has_per_subkernel_fields`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：sum 和 mean 成员分别携带真实 load、最终 store 与 reduction 数量。

当前失败／跳过：sum/mean 数值比较通过，随后缺 inductor_meta_0；成员 load/store/reduction 统计断言尚未执行。

具体缺口／适配：缺 inductor_meta_i；归约计数入口已有，但最终归约写回统计未统一。

先修入口：NPU CSEProxy.store_reduction 与成员统计出口。 验收：与 standalone 的真实 load/store/reduction
对照，覆盖最终写回、广播及被消除的写回。

证据：[E2][E2]、[E4][E4]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-14"></a>

#### 14 P1 `test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：确定性归约携带逐成员 has_loadstore_with_contiguous_rdim。

当前失败／跳过：确定性配置下 sum/mean 数值比较通过，随后缺 inductor_meta_0；未完成成员连续归约维标志检查。

具体缺口／适配：缺成员字典；native load 未同步社区连续归约标志，不能将默认 False 当真实统计。

先修入口：NPU load 地址／stride 分析、成员 metadata 出口与 native 确定性候选选择。 验收：按真实连续归约维判断标志，保留数值；不能把缺字段解释成没有确定性归约。

证据：[E2][E2]、[E4][E4]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-15"></a>

#### 15 P1 `test_combo_per_kernel_inductor_meta_matches_standalone`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：Combo 成员与 standalone 的 load/store/reduction、atomic、no_x_dim、提示及 tiling_scores 一致。

当前失败／跳过：Combo 数值比较通过；standalone 解析为两个字典，Combo 成员字典解析为空，长度 2 对 0 而失败。必需统计字段仍需非空有效基准，不能用空字典对齐。

具体缺口／适配：成员统计出口缺失；standalone 也需补完整必需字段，空正则匹配不能充当基准。

先修入口：统一 native per-kernel metadata，并修测试字段解析与非空校验。 验收：两侧必需字段均非空，真实值按 NPU split 轴和布局对应，修计数后再比较。

证据：[E2][E2]、[E4][E4]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-16"></a>

#### 16 P1 `test_combo_kernel_num_gb_and_flop_match_standalone_sum`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：Combo 总访存字节／FLOP 与其 standalone 成员之和相等。

当前失败／跳过：原数值比较完成后，Combo kernel_num_gb 匹配为空；FLOP 聚合检查尚未执行，空 standalone 求和不是有效基准。

具体缺口／适配：Combo 未聚合 kernel_num_gb，普通 NPU 也未统一导出 kernel_flop。

先修入口：NPU 字节／FLOP 估算、standalone 出口与 Combo 聚合。 验收：先得到非空有效基准再聚合；本输入常规读写为 16384B，不能以空匹配求和 0 通过。

证据：[E2][E2]、[E12][E12]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-17"></a>

#### 17 P1 `test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：profile_bandwidth 只携带真实 kernel_num_gb，避免额外 FLOP 估算。

当前失败／跳过：数值比较通过，随后缺 kernel_num_gb；后续不应输出 kernel_flop 的检查未执行。

具体缺口／适配：普通 NPU 已按 profile 条件估算字节，Combo 未聚合并传出。

先修入口：NPUComboKernel.jit_line 的 profile 条件和成员字节数聚合。 验收：profile 模式有真实总字节、无 kernel_flop，并联验未开 profile
时不输出该字段。

证据：[E2][E2]、[E12][E12]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-18"></a>

#### 18 P1 `test_benchmark_combo_kernel_emits_real_num_gb`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：独立 benchmark 模板得到真实 GB 数，带宽报告才能使用有效分子。

当前失败／跳过：数值比较通过，随后未匹配到浮点 num_gb；NPU Combo benchmark 生成入口传入硬编码 0，后续 device 参数检查未执行。

具体缺口／适配：NPU Combo 调用 codegen_kernel_benchmark(num_gb=0)，当前模板收到硬编码零。

先修入口：NPUComboKernel.codegen_kernel 传真实成员字节估算之和。 验收：正的真实浮点 num_gb、正确的 NPU device 参数及模板实际消费，不能仅把 0 改成
0.0。

证据：[E2][E2]、[E12][E12]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-19"></a>

#### 19 P1 `test_triton_kernel_to_post_grad_tracing_combo_kernel`

位置：`test_provenance_tracing.py::TestProvenanceTracingArtifact`；本轮 FAIL 1。

功能目标：每个真实 Combo kernel 能回溯到其 post-grad FX 节点和成员 origins。

当前失败／跳过：Inductor 编译执行与 debug trace 收集已完成；Combo 到 post-grad 的映射实际为空，期望一项。第二个 AOTInductor 子场景尚未开始。

具体缺口／适配：define_kernel／codegen_comment 接收 Combo 外壳，而其 node=None，origins 未按真实分区展开。

先修入口：NPUTritonScheduling.codegen_combo_kernel 按 node_group 展开 get_nodes。
验收：核对实际分区、实际命名和缓存命中；Inductor 与 AOT 两入口都需映射到真实 FX origins。

证据：[E6][E6]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-20"></a>

#### 20 P2 `test_combo_kernel_store_mask`

位置：`test_torchinductor.py::CommonTemplate`；本轮 SKIP 1。

功能目标：动态两个 mean 在一个 Combo 中执行，store 使用合法尾部 mask。

当前失败／跳过：NPU does not support

具体缺口／适配：旧 NPU skip 阻断验收；解除后须先证明原图形成 Combo，再检查 store mask。

先修入口：test_torchinductor 的旧 skip、NPU 动态归约轴兼容性及 store mask 生成。 验收：原输入与数值保留，真实一个 Combo，最后写回 mask
覆盖有效输出；不能跳过数量检查直接称 mask 已实现。

证据：[E1][E1]、[E4][E4]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-21"></a>

#### 21 P2 `test_fuse_mix_order_reductions_combo_kernels`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：共享输入的行／列归约联合执行，消费者单独执行，其他独立归约形成 Combo。

当前失败／跳过：数值比较通过，随后 generated_kernel_count 期望 3、实际 4；尚需按实际生成链核对共享输入联合归约的缺口。

具体缺口／适配：NPU mix_order_reduction=False；这是共享输入 mixed-order 联合归约能力目标。

先修入口：NPU mixed-order 调度、lowering／成员资源分析，避免与独立成员 hint 混淆。 验收：原共享 x 图数值正确，证明行列联合归约、消费者隔离及独立 y/z
Combo，保留原 3-kernel 目标。

证据：[E1][E1]、[E15][E15]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-22"></a>

#### 22 P2 `test_combo_kernel_scalar_store_broadcast`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：标量加法、向量归约、读取该标量的归约在一次合法计算中组合。

当前失败／跳过：标量结果、向量归约和广播消费者的数值比较通过；随后 generated_kernel_count 期望 1、实际 2，单个 Combo 的组合目标未通过。

具体缺口／适配：需要协调标量 producer/consumer 依赖和不同成员轴 ABI；不能笼统写成融合策略差异。

先修入口：NPU scalar write／broadcast、垂直融合边界和 Combo 成员轴兼容性。 验收：原随机输入数值正确，真实一次 launch，标量只作合法写回且消费者读取正确值。

证据：[E1][E1]、[E15][E15]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-23"></a>

#### 23 P2 `test_combo_kernel_no_bench_stitched_config`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：无 autotune 时三个成员各自的默认 X 分块与 body/grid 一致。

当前失败／跳过：数值、单个真实 kernel 定义及 default/body 一致性检查已通过；最后缺 XBLOCK_0、XBLOCK_1、XBLOCK_2，空匹配不满足三成员独立配置。

具体缺口／适配：NPU default_config 使用共享 split block，未实现三个独立 XBLOCK_i 拼接。

先修入口：成员 block ABI、默认候选生成、dispatch 边界与 launcher 参数映射。 验收：数值通过且无 benchmark，三个真实独立默认配置与 body/grid
同源，不以共享块或空匹配通过。

证据：[E1][E1]、[E3][E3]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-24"></a>

#### 24 P2 `test_combo_kernel_no_bench_persistent_reduction`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：无 autotune 的三个 persistent 成员保留完整 R，X 默认分块独立。

当前失败／跳过：三个 sum 的数值与单定义检查通过，default_config 不含 R0_BLOCK 的检查也通过；随后缺 XBLOCK_0/1/2 的独立默认 X 配置，原方法未直接验证实际
winner/launch。

具体缺口／适配：静态完整 R 生成已有；成员独立 X 默认配置和 dispatch 尚未接通。

先修入口：成员 X block ABI／默认配置及 persistent R 常量生成。 验收：完整 R 在 body 固定，三个独立 X 默认值与实际 grid/launcher 一致，保留数值和无
benchmark。

证据：[E1][E1]、[E3][E3]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-25"></a>

#### 25 P2 `test_combo_kernel_no_bench_carve_out`

位置：`test_combo_kernels.py::ComboKernelTests`；本轮 FAIL 1。

功能目标：无 autotune 默认配置不适合 bucketize 时将其分离，另两个 pointwise 形成 Combo。

当前失败／跳过：数值比较通过，随后 async_compile.triton 定义数期望 2、实际 1；此计数不等于实际 launch 数，原 carve-out 分区目标未验收。

具体缺口／适配：需核对 native 默认配置接受／拒绝边界，不能直接照搬 CUDA 的候选数量门槛。

先修入口：NPU 默认候选／资源检查和 Combo partition 的 carve-out 决策。 验收：保留 bucketize 与其他输入，真实一个 standalone 加一个 Combo
且数值正确，说明拒绝理由。

证据：[E1][E1]、[E3][E3]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-26"></a>

#### 26 P2 `test_combo_kernel_max_autotune`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：三个激活成员各自生成候选并在完整 Combo 中顺序评价。

当前失败／跳过：编译执行已返回，但 Combo sequential autotune 日志匹配为 0；后续数值比较和单 kernel 断言未执行。

具体缺口／适配：native 当前共享 tile 搜索；原目标是逐成员候选／顺序调优及分组，缺社区 combo_tuning_groups 契约。

先修入口：先接成员 block ABI 和真实 metadata，再实现独立候选、完整 Combo 评价和可观测分组。 验收：保留数值和实际单 Combo，验证真实候选、组成员及
winner；不能只替换日志字符串通过。

证据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-27"></a>

#### 27 P2 `test_combo_kernel_max_autotune_with_reduction`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：sum/mean 成员各自的归约候选及顺序评价。

当前失败／跳过：assertLogs 捕获了日志，但 Combo sequential autotune 匹配为 0；后面的数值与 kernel 数量断言尚未执行。

具体缺口／适配：native 当前共享 tile 搜索；原目标是逐成员候选／顺序调优及分组，缺社区 combo_tuning_groups 契约。

先修入口：先接成员 block ABI 和真实 metadata，再实现独立候选、完整 Combo 评价和可观测分组。 验收：保留数值和实际单 Combo，验证真实候选、组成员及
winner；不能只替换日志字符串通过。

证据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-28"></a>

#### 28 P2 `test_combo_autotune_many_subkernels`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：六个不同工作量／算子的独立成员候选和顺序评价。

当前失败／跳过：六成员编译执行已返回，但 Combo sequential autotune 的日志匹配为 0；后续数值比较和单 kernel 检查未执行。

具体缺口／适配：native 当前共享 tile 搜索；原目标是逐成员候选／顺序调优及分组，缺社区 combo_tuning_groups 契约。

先修入口：先接成员 block ABI 和真实 metadata，再实现独立候选、完整 Combo 评价和可观测分组。 验收：保留数值和实际单 Combo，验证真实候选、组成员及
winner；不能只替换日志字符串通过。

证据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-29"></a>

#### 29 P2 `test_combo_autotune_grouping`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：相同 heuristic 输入的成员共享一次调优，不同尺寸分组。

当前失败／跳过：编译执行已返回，期望两个调优组而实际日志解析为空；后续数值比较和单 kernel 断言未执行。

具体缺口／适配：native 当前共享 tile 搜索；原目标是逐成员候选／顺序调优及分组，缺社区 combo_tuning_groups 契约。

先修入口：先接成员 block ABI 和真实 metadata，再实现独立候选、完整 Combo 评价和可观测分组。 验收：保留数值和实际单 Combo，验证真实候选、组成员及
winner；不能只替换日志字符串通过。

证据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-30"></a>

#### 30 P2 `test_combo_max_persistent_rblock_gated_on_per_subkernel`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1; PASS 1。

功能目标：共享 X 分块时应用联合 R 资源约束，真正独立 X 分块时取消该共享约束。

当前失败／跳过：per_subkernel=False 数值通过但缺 max_persistent_rblock=1024；True 数值与字段不存在的断言通过，该负向字段检查未证明独立 X
分块生效。

具体缺口／适配：原 max_persistent_rblock 是社区共享-X/HIP 约束字段；NPU 需验证其真实 UB／成员资源规则。

先修入口：NPU 共享／独立 block ABI 与每成员资源约束，不移植裸 4096 阈值。 验收：区分实际共享与独立模式，验证合法完整 R、资源受限候选及字段语义；开关 True 不等于独立 X
已实现。

证据：[E1][E1]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-31"></a>

#### 31 P2 `test_combo_kernel_coordesc_tunes_largest_subkernel_first`

位置：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；本轮 FAIL 1。

功能目标：逐成员坐标下降先修改工作量最大的第 1 个成员 X 分块。

当前失败／跳过：在 assertLogs 退出时失败，未捕获 coordinate_descent_tuner 的 DEBUG 日志；后续数值、多个成员配置及最大成员优先断言均未执行。

具体缺口／适配：共享 block 无独立成员坐标，不能验证仅先改变 XBLOCK_1 的目标。

先修入口：成员独立 ABI、coordesc 字段与工作量排序、完整配置评价。 验收：真实尝试多个不同成员配置，首步只改最大成员，并保持数值；日志缺失不代表数值错误。

证据：[E1][E1]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-32"></a>

#### 32 P2 `test_combo_rms_norm_forwards_add_persistent_rblock`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 1。

功能目标：两个 RMSNorm 成员的候选保留 standalone 的完整 R／大 R 优化资格。

当前失败／跳过：两个 RMSNorm 的数值比较通过，随后缺 inductor_meta_0；未执行到成员 add_persistent_rblock 检查。

具体缺口／适配：缺逐成员 add_persistent_rblock 出口；NPU 原图可已是完整-R persistent，不能据字段认定缺 persistent。

先修入口：native standalone／Combo 的实际 R 候选和资源约束，再定义该优化标志的等价消费。 验收：原 (2048,4096) 图数值正确，真实候选与 winner
对照；不能补无消费者标志冒充优化。

证据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-33"></a>

#### 33 P2 `test_combo_triton_meta_has_disable_ftz`

位置：`test_combo_kernels.py::ComboKernelMetadataTests`；本轮 FAIL 2。

功能目标：disable_ftz 开关从 Inductor 传到编译器和实际数学路径，控制非正规数保留。

当前失败／跳过：False/True 两个实例数值均通过，随后各缺对应 disable_ftz 字段；当前正值输入未覆盖非正规数，数学功能还需另行验收。

具体缺口／适配：NPU triton_meta 未导出该字段，当前安装 Triton 的 NPUOptions 未声明它；通用编译选项及数学路径消费仍需定界。

先修入口：NPU jit_line、Triton Ascend compiler NPUOptions、指数数学 lowering／下层消费。 验收：False/True 字段正确并有真实消费者；验证
exp/exp2 边界，不用只打印开关替代数值能力。

证据：[E2][E2]、[E8][E8]；[原方法与断言][test-index]；[本轮报告][jobs]。

<a id="repair-34"></a>

#### 34 P3 `test_combo_kernel_peak_memory_wide_resnet`

位置：`test_combo_kernels.py::ComboKernelPeakMemoryTests`；本轮 FAIL 1。

功能目标：峰值门控拒绝增加内存的合法 Combo 候选并降低运行峰值。

当前失败／跳过：tight 与 disabled 的首次执行峰值同为 616101888B，未满足下降断言；本轮没有额外捕获 gate 候选，不能据此判定 gate 未实现。

具体缺口／适配：源码已有 gate；原首次执行同时计入编译／autotune 内存，峰值不降不能直接证明 gate 缺失。

先修入口：用例候选有效性、正 memory delta 与 warm 运行峰值观测；再核对实际拒绝决策。 验收：证明存在会被门控拒绝的真实 NPU
Combo，并用相同原生内存统计隔离运行峰值；不只放宽阈值。

证据：[E7][E7]、[E13][E13]；[原方法与断言][test-index]；[本轮报告][jobs]。

### 保留 SKIP 的硬件／平台协议

`test_combo_kernel_tma_descriptor_block_name`：NVIDIA SM90+ TMA descriptor 搬运协议。 本轮 SKIP，未计为
PASS，也不进入 NPU 同协议实现队列。

`test_combo_kernel_amd_special_config_args`：ROCm/HIP 专属 compile options 与 constexpr 分离。 本轮 SKIP，未计为
PASS，也不进入 NPU 同协议实现队列。

`test_pdl_codegen_in_combo_kernel`：NVIDIA SM90+ CUDA grid dependency control 和 launch_pdl。 本轮
SKIP，未计为 PASS，也不进入 NPU 同协议实现队列。

`test_pdl_combo_kernel_pointwise`：NVIDIA PDL 每成员 wait／launch_dependents 顺序。 本轮 SKIP，未计为 PASS，也不进入
NPU 同协议实现队列。

`test_pdl_combo_kernel_reduction`：类级开启的 NVIDIA PDL 归约 launch 协议。 本轮 SKIP，未计为 PASS，也不进入 NPU 同协议实现队列。

XPU-only 的动态归约方法没有 XPU 专属 intrinsic，故已进入修复顺序；它的设备门控不能与 TMA／PDL／HIP 协议一并归为硬件差异。

### 2.10 重合范围的当前非 PASS

这些方法不进入上面的“范围外”排序，但仍保留本轮实际结果；非硬件 SKIP 不作为已完成支持。

`test_combo_kernels.py::test_combo_kernel_per_config_subkernel_per`（FAIL）：数值比较和
generated_kernel_count=1 通过，随后固定 CUDA XBLOCK : tl.constexpr 检查失败。本轮走共享 block 分支；需按 NPU 实际 split
block ABI 适配观测，并保留两个 sum 成员 R=64/512 的完整归约要求。缺该字符串不代表 persistent 归约未实现。

`test_combo_kernels.py::test_combo_kernel_per_config_subkernel_red`（FAIL）：两组 (32,64,128) 的
sum(dim=(0,2)) 原数值比较通过；随后 generated_kernel_count 期望 1、实际 2，尚未执行 X/R block 检查。原目标是将两条非连续多轴归约组合为单个
Combo，需核对成员轴、候选合法性和分区接受条件；保留 FAIL，不能仅取消数量目标或改期望为 2。

`test_torchinductor.py::test_combo_kernel_filter_cpu`（FAIL）：a*4、b+8、c.min(-1)、d.max(-1) 的数值比较通过；NPU
原分组目标为 pointwise 与 min/max 两个 Combo，generated_kernel_count 期望 2、实际 4。需核对 20x20/30x30 的不同尺寸 pointwise
和 min/max 成员的轴、tiling 与分区接受条件，完成实际分组目标；保留 FAIL，不能直接改为 4 或用 skip 规避。

以上原方法与断言见 [测试源码索引][test-index]，实际失败阶段见 [逐实例报告][jobs]，NPU 成员分区和共享 block ABI 见 [E1][E1]。

### 水平融合、垂直融合与 foreach 的关系

水平融合把没有 producer/consumer 依赖的独立成员打包进一个 Combo；垂直融合沿依赖链把生产与消费算子融进同一成员。`torch._foreach_*` 是对 tensor
列表逐项计算的前端 API；编译器的 FOREACH 分组融合路径把各项形成成员、吸收生产者／消费者链，再将独立成员水平打包。当前 NPU 未注册该 FOREACH
后端能力，普通算子的垂直融合入口仍可使用。

```text
成员 0：add(a0, a0) → foreach 第 0 项 → mul(d0, a0)
成员 1：mul(a1, a1) → foreach 第 1 项 → mul(d1, a1)
```

`test_non_foreach_consumer_producer_list` 体现这两条链；`test_scheduler_fusion_list` 体现 foreach→foreach
的成员配对融合，`test_single_list` 体现列表成员水平打包。社区原目标是单 kernel；本轮 foreach 按既定功能口径执行，所以数值 PASS 不证明完整 FOREACH
分组融合完成。

社区 2.13 支持上述水平／垂直组合，基础机制在 2.10 已存在，见版本源码 [E15][E15]。NPU 的普通 can_fuse_vertical 入口已有，backend features
当前未注册 FOREACH；应将完整列表分组与依赖链吸收作为 NPU 待适配功能，不能从 FOREACH 入口未启用推断普通垂直融合缺失，也不能把缺 mixed-order
联合归约或成员独立调优笼统写成融合策略差异。

<!-- combo-section:analysis213:end -->

[pass213]:
  ../../../../scripts/tests/combo_kernel_210/combo_native_213_passed_nodeids.txt
[csv]:
  inductor-combo-tests-native-v2.13-sync-64dc85c.csv
[jobs]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/jobs.json
[evidence]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/status.json
[junit]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/junit-cross-check.json
[scope]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/community-branch-scope.json
[overlap]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/overlap60-jobs.json
[all-pass]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/passed-nodeids.txt
[test-index]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/test-method-index.json
[accuracy-wrapper]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/generated/r0-r1-accuracy-1.py
[accuracy-source]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/generated/r0-r1-accuracy-source.json
[triton-options]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/installed-triton-source/options-ast-check.json
[triton-compiler]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/installed-triton-source/compiler.py
[E1]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/shared-block-abi.txt
[E2]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/combo-metadata.txt
[E3]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/native-tile-search.txt
[E4]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/cse-statistics.txt
[E5]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/graph-optimize-mem.txt
[E6]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/provenance-chain.txt
[E7]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/peak-memory-gate.txt
[E8]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/ftz-boundary.txt
[E9]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/platform-test-gates.txt
[E10]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/scheduler-input-distance.txt
[E11]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/community-combo-contract.txt
[E12]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/bandwidth-and-flop.txt
[E13]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/wrn-grouping-and-autotune-memory.txt
[E15]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/fusion-capabilities.txt
[E16]:
  ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/source/no-linear-multi-axis.txt
