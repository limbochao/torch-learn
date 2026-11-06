---
title: PyTorch v2.13.0 Combo/Foreach 用例在 v2.10.0 的存在性核对
---

# PyTorch v2.13.0 Combo/Foreach 用例在 v2.10.0 的存在性核对

## 最新 liulm 基线复测

最新的 0fd5a10 结果见[liulm 基线复测表](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md#原失败方法的最新复测)。
原 13 个失败方法、15 个实例：仅新包 1/15 通过，新包+F3 7/15 通过。
新包+F3 的全量结果由后台脚本刷新到新基线表；本页保留原版本对照和旧基线历史。

核对日期：2026-11-05。范围：用户提供的 116 条记录、8 个 `test/inductor/` 文件。

**结论：54 条独立测试方法在 v2.10.0 中已存在，11 条记录是已存在的嵌套函数，
51 条独立测试方法在 v2.10.0 中未找到同名定义。**

按“文件名 + 函数名”去重后，共 108 项：54 个已有独立测试方法、3 个已有嵌套函数名、
51 个未找到同名定义的测试方法。原清单的 8 条重复记录均来自嵌套函数。

可下载 [CSV 完整表格](inductor-combo-tests-v2.13-vs-v2.10.csv)，
包含测试文件、函数名、类型、版本存在性、缺失用例的能力分析、直接运行判断、NPU 实测结果及修复 commit。

## 核对口径与证据

- 比较官方 `pytorch/pytorch` 的版本标签；已用 `git ls-remote --tags origin` 确认本地标签与上游一致。
  - `v2.10.0`：`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`。
  - `v2.13.0`：`cf30153c4c131c8164ee7798e5022d810682e2cb`。
- 使用 `git show <tag>:test/inductor/<file>` 读取源码，通过 Python AST（语法树）识别函数及所属类/外层函数。
- “一致”表示已有对应方法或函数且实现一致；“存在（差异：…）”表示已存在但实现有变化。
  若版本间命名不一致，在存在性列中说明。
  本次已有条目的函数名、所属类与外层函数均一致。存在性列同时标注目标方法的实现差异，
  “差异”摘要统一描述从 2.10 到 2.13 的变化，包含方法体、装饰器、断言和方法内配置。
  54 个已有独立方法中，10 个有差异，44 个方法一致；11 条嵌套函数记录的函数实现均一致。
  “一致”仅表示目标函数定义及装饰器的语法树一致（忽略排版和注释），
  不表示文件导入、类初始化、辅助函数或参数化列表一致，也不表示可直接运行。
  这些共享依赖的差异与阻塞见“直接运行判断口径”和运行判断列。
- “嵌套”表示测试方法内部定义的被测函数，不作为独立测试方法计数。
  例如 `test_activations` 位于 `ComboKernelTests.test_activation_functions` 内部。
- “无”表示同文件未找到同名定义，并用 `git grep -n -w -F` 在 v2.10.0 的整个 `test/` 目录复查无匹配；
  这不等于被测能力不存在；功能支持另行对照实现及新增测试的提交分析。
  `test_user_streams.py` 在 v2.10.0 中整文件不存在。
- 所有 116 条记录均能对应到 v2.13.0 定义；保留清单中的 CUDA、XPU、CPU 名称，不按设备重新筛选。
- 存在性基于上游标签源码；直接运行兼容性结合用户提供的 2.10 NPU 报错及本地 2.13 NPU 测试文件核查。
  原始文件兼容性以静态分析及此前两个收集失败日志为依据；临时适配后的 NPU 实测与修复 commit 另列。

## 按文件汇总

| 文件 | 原始条数 | 去重项 | 已有独立方法 | 已有嵌套记录 | 未找到 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test_triton_kernels.py` | 1 | 1 | 0 | 0 | 1 |
| `test_torchinductor.py` | 5 | 5 | 4 | 0 | 1 |
| `test_combo_kernels.py` | 78 | 70 | 21 | 11 | 46 |
| `test_foreach.py` | 27 | 27 | 27 | 0 | 0 |
| `test_snode_runtime.py` | 2 | 2 | 0 | 0 | 2 |
| `test_unbacked_symints.py` | 1 | 1 | 1 | 0 | 0 |
| `test_provenance_tracing.py` | 1 | 1 | 1 | 0 | 0 |
| `test_user_streams.py` | 1 | 1 | 0 | 0 | 1 |
| **合计** | **116** | **108** | **54** | **11** | **51** |

## 直接运行判断口径

**判断对象是在 2.10 环境中，保留 2.13 测试文件及同目录依赖，直接用 pytest 选择目标用例。**
与用户实际运行方式一致，例如在 `Community-Use-Case-Testing/v2.13.0/inductor_npu` 下执行：

```bash
pytest -vs test_torchinductor.py -k test_rand_like_deterministic
pytest -vs test_combo_kernels.py -k test_activation_functions
```

- 不替换为 2.10 的导入、测试类或辅助文件，不预先删除新配置、不补接口。
  同名方法和方法体相同都不足以证明整份测试文件可直接运行。
- pytest 必须先导入完整模块并收集测试，`-k` 不能绕过文件级导入错误；
  之后仍需经过测试类初始化、测试执行和断言检查。
- “否”表示已发现上述任一阶段的明确阻塞；“待验证”表示尚无足够证据确认可运行。
  未找到同名定义的条目，其功能支持见存在性列；运行判断列“—”表示未单独作直接运行判定，
  不表示可以运行。上述文件级阻塞同样影响其中新增的方法。嵌套函数不作为独立用例判断。
- 54 个已有独立方法中，**53 个受已识别的文件级依赖阻塞，1 个在本轮 NPU 实测中执行失败**。
  53 个中只有日志所列的两个目标有用户实测证据，其余依据共享导入依赖推断，未逐个实测。
- `test_unbacked_symints.py` 未发现对 `test_torchinductor.py` 的同类导入依赖，
  本轮保留该文件原样并完成收集及执行，在 Combo benchmark 阶段因缺动态尺寸参数失败。
  因此该行的直接运行判断由此前“待验证”更新为“否”；失败详情见实测列 F1。

### 已确认的阻塞及影响范围

- `test_torchinductor.py`：用户实测缺少 `torch.testing._internal.common_dtype.highest_precision_float`，
  结果为 `collected 0 items / 1 error`；清单中该文件的 4 个已有方法均无法直接收集。
- `test_combo_kernels.py`：用户实测缺少 `requires_xpu_and_triton`，同样在收集阶段失败，
  影响该文件的全部 21 个已有独立方法。文件还会导入同目录的 `test_torchinductor.py`。
  三个相关 Combo 测试类的 `setUp()` 还设置 2.10 缺少的四个配置：
  `combo_kernel_per_subkernel_blocks`、`combo_kernel_max_distance`、
  `combo_kernel_peak_memory_increase_gb`、`combo_kernel_peak_memory_pct_threshold`。
  对 `test_torchinductor.py` 的依赖及 `setUp()` 配置问题为后续静态发现的阻塞，日志尚未执行到这些阶段。
- `test_foreach.py`：导入同目录 `test_torchinductor.py` 中的 `check_model`、`check_model_gpu`，
  因而受上述导入失败影响；其文件级算子列表还引用 2.10 没有的 `torch._foreach_clone`。
  即使只选择清单中的某个已有方法，也必须先完成这些文件级操作，影响全部 27 个已有方法。
- `test_provenance_tracing.py`：文件级导入同目录 `test_torchinductor.py` 的 `copy_tests`，
  因而清单中的 1 个已有方法也受上述导入失败影响。

以上依赖关系同时核对了本地 `Community-Use-Case-Testing/v2.13.0/inductor_npu/` 中的对应文件；
缺失的 PyTorch 符号、算子和配置与上游 `v2.10.0` 源码比对。
用户环境的两个 `ImportError` 为实际报错；Foreach、provenance 及初始化配置的影响为源码推断。
以上判断针对未适配的原始文件；下方 NPU 实测根据本轮授权临时适配测试依赖和 CUDA 专属检查。

## 51 个缺失用例的功能支持分析

**未找到测试不等于 2.10 不支持对应算子。** 逐项比较测试的配置、生成代码断言及生产实现后，分为：

| 判断 | 条数 | 含义 |
| --- | ---: | --- |
| 缺能力 | 33 | 缺少该测试要求的新增能力、融合策略、调优策略或元数据传递 |
| 缺修复 | 9 | 基础能力已有，但缺少测试针对的场景修复 |
| 部分支持 | 7 | 旧路径或默认行为已有；新路径、可配置行为缺失 |
| 已有行为 | 1 | 目标行为在 2.10 已有，该测试属于后来补充的覆盖 |
| 待验证 | 1 | 已有同体 GPU 测试；不足以判定 2.10 的 XPU 后端表现 |
| **合计** | **51** | **42 条缺少对应能力/策略或修复，7 条部分支持，1 条行为已有，1 条待验证** |

这里的“缺能力”只针对测试要求的具体行为，例如子 kernel 独立分块、Combo 的 PDL 接入、
带宽统计或融合内存限制，不能扩大为“不支持 pointwise、归约、Foreach、训练或 ResNet”。
“缺修复”表示源码缺少对应处理，不代表已在 2.10 环境实测复现该错误。
“部分支持”“已有行为”也不表示整份 2.13 测试可直接运行：文件导入、初始化配置和新辅助接口仍可能阻塞。

本节针对**上游 v2.10.0** 的实现进行静态判断；未检查 Ascend 分支是否另有回移，
本轮 NPU 实测另见实测列；未运行 CUDA、XPU 或 ROCm。PDL、TMA、HIP 等测试有各自硬件/后端条件，
不能据此推出 NPU 的支持情况。逐项结论已写入下方存在性列和 CSV。

### 能力判断的源码依据

以下依据均来自两个标签的生产代码和新增测试对应的提交；提交号用于本地复核。

- **独立分块和 grid**：2.10 的 `ComboKernel.get_block_args()` 使用共享 `XBLOCK/YBLOCK/R0_BLOCK`，
  `ComboKernelGrid.generate()` 已处理动态尺寸；2.13 增加独立 `XBLOCK_i` 等参数、
  `_handle_combo_kernel_per_subkernel_blocks()` 及 `SequentialFlattenComboKernelGrid`。
  关键提交 `767bcdfd29d`、`b553a3af6ed`。因此四个基本分块用例、block_size 和动态 grid 用例是部分支持。
  `767bcdfd29d` 同时新增二维归约及 3D pointwise 的单独分区处理；
  这两个用例检查排除 Combo 的策略，不能解释为 2.10 不支持这些普通 kernel。
- **分组策略**：2.10 的 `ForeachKernelSchedulerNode` 将节点上限写死为 8，
  缺少 `combo_kernel_max_num_nodes` 和 `combo_kernels_pointwise_only` 配置；
  2.13 可配置上限并可仅融合 pointwise，关键提交 `d69ec4e17a0`、`d451010f177`。
  `test_single_combo_kernels` 对应 `489edca3c95`：单节点分区改用普通 kernel，并复用 benchmark 结果。
- **不测速的独立块配置**：2.10 已有共享块的非 autotune 路径；2.13 在 `SIMDScheduling` 增加
  `_probe_subkernel_heuristic()`、`_stitch_no_bench_combo_config()`，拼接各子 kernel 的配置，
  并将不适合此路径的节点拆出。对应 `d18f5dcab25`，三个 no_bench 用例均检查这些新增处理。
- **调优策略**：2.10 已有基础 Combo autotune；缺少独立块的最大调优、串行调优及分组调优实现。
  对照 `runtime/triton_heuristics.py`，相关提交为 `b712e5c9183`、`048c2ae8480`、
  `e60f3fc26d5`、`559f2b27dfc`；最后一个按包含元数据的完整指纹分组。
  `b61524d2fcb` 还增加 Combo 独立块的坐标下降及最大子 kernel 优先调优。
  `57e7ded57bc` 按共享块/独立块模式控制 `max_persistent_rblock`；
  2.10 的 Combo 本身也未生成这个上限字段，不能满足该用例的两分支断言。
- **归约提示及候选配置**：2.10 的 Combo 根据选中子 kernel 传入单个 reduction_hint，
  缺少逐子 kernel 的 `reduction_hint_i`；`b4b3186829f` 增加该能力。
  普通 kernel 已有 `add_persistent_rblock`，但 Combo 未传递子级标记；`a4a3bfb06c3` 补齐。
  这不意味着 2.10 不能计算 RMSNorm。
- **子 kernel 元数据**：2.10 的 `ComboKernel.combo_grid_meta()` 没有 `inductor_meta_i`，
  `c399ebc6f2b` 将普通 kernel 的元数据抽为 `inductor_meta_per_kernel()` 并传给每个子 kernel，
  覆盖 pointwise、归约、确定性连续归约维及与普通 kernel 的元数据一致性四个用例。
- **内存优化和数值选项**：2.10 普通 kernel 已区分推理/反向和训练前向的 `optimize_mem`，
  Combo 未传该字段；`0274ad69c3e` 补齐两个场景。`a886dc98482` 让 Combo 复用
  `TritonKernel.triton_meta_common()`；2.10 还缺少该测试使用的 `eager_numerics.disable_ftz` 配置。
  FTZ 指将非正规小浮点数置零，测试检查禁用该行为的编译选项是否传递。
- **字节量和 FLOP**：2.10 普通 kernel 已有统计，但 `ComboKernel.jit_line()` 未汇总这些字段，
  benchmark 更是写死 `num_gb=0`；`b3fbb25559b` 增加子 kernel 统计汇总和真实 benchmark 字节量。
  因而三个正向统计用例缺相应能力；反向检查
  `test_combo_inductor_meta_no_kernel_num_gb_without_profile` 的“不输出字段”行为在旧版已有。
- **调度诊断与融合内存限制**：`cecfffb6d29` 为调度节点增加 `min_input_distance`、
  `max_input_distance` 和计算逻辑；两个 snode 用例检查该指标，2.10 已有 Foreach 算子。
  `2eaecea5b57` 增加 Combo 峰值内存评估及阈值配置；2.10 缺少该融合限制策略，
  不能据此认定 Wide ResNet 模型不受支持。
- **stream、PDL**：2.10 的 Combo 分组没有 `node_to_stream` 的分组及赋流处理，
  `216c03a9d36` 补齐该路径；这与基本 CUDA Stream API 是否存在是两回事。
  2.10 虽已有普通 Triton kernel 的 `enable_pdl` 配置及代码生成，
  Combo 尚未设置 `launch_pdl` 和过滤子 kernel 的 PDL 语句，`fdd12f1a340` 增加接入。
  PDL 是 CUDA 的程序化依赖启动能力，三个用例检查其在 Combo 中的处理。
- **浮点自减、归约选择及排序修复**：`9ebccc80ec1` 在 `TritonOverrides.sub()` 增加
  相同浮点表达式相减的局部改写；`a5c5d9fc4a7` 将全持久归约 Combo 的参数选择从首个子 kernel
  改为最大归约维；`2700915a75e` 强制 Combo 的 sort 使用持久归约。2.10 缺少这些处理。
- **混合顺序归约和 store 修复**：2.10 已有 `FusedMixOrderReductions`；`b13d56226f4`
  从 Combo 分组排除该节点及其消费者。`98c1c0c2ef8` 在 Combo 标量 store 中广播指针，
  `2c109a65f8e` 为 Combo 常量索引 store 保留必要的 mask；2.10 均缺少对应处理。
- **grid、TMA、HIP 修复**：`e9dfc90e074` 将超出上限的 grid Y 拆到 Z，2.10 没有该拆分。
  2.10 已有 `triton.use_tensor_descriptor`，但 Combo 缺少前导代码生成及完整延迟行的块名替换，
  `2ee4eba06fb` 补齐 TMA 路径，TMA 指 Tensor Memory Accelerator。
  `507f793e6e0` 让独立块配置中的 HIP 后端选项保持原名并传给编译器；
  旧版缺少独立块路径及此处理，不能解释为整个 ROCm 后端不受支持。
- **XPU 新增覆盖**：`188411752cd` 仅修改测试，新增 XPU 专用动态归约用例。
  2.10 的 `test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda` 使用通用
  `requires_gpu_and_triton` 和 `GPU_TYPE`，其被测函数及输入与该 XPU 用例相同。
  新测试名及 `requires_xpu_and_triton` 不存在不能证明旧版不支持该计算；
  该用例还设置编译期 autotune，XPU 的完整路径是否可用需设备验证。

## NPU_A5 功能验证（2.10 分支）

验证日期：2026-11-05。**基线已完成全部 62 个候选方法、409 个核定参数化实例。**
基线按源方法统计：44 个全部通过、18 个失败；按实例统计：389 个通过、20 个失败、0 个跳过。
随后应用 F1 修复，定向复测 6 个方法、6 个实例，5 个通过、1 个出现新的 F8 阻塞。
再基于 F1 应用 F3 修复，复测 7 个原 F3 方法，6 个通过、1 个越过 F3 后出现 F9；
另复测两个此前通过的 F1 方法，均通过，不重复计数。
合并未复测项的基线结果后，当前为 **55 个方法通过、7 个失败；400 个实例通过、9 个失败、0 个跳过**。
这是基线、F1 与 F3 定向复测结果的合并统计，未重新执行全部 409 个实例；表中使用各方法最新的实测状态。
两个测试侧阻塞的首次失败日志和原始基线结果继续保留。

| 测试文件 | 方法通过 / 总数 | 实例通过 / 总数 |
| --- | ---: | ---: |
| test_torchinductor.py | 2 / 4 | 2 / 6 |
| test_combo_kernels.py | 26 / 29 | 26 / 29 |
| test_foreach.py | 27 / 27 | 372 / 372 |
| test_unbacked_symints.py | 0 / 1 | 0 / 1 |
| test_provenance_tracing.py | 0 / 1 | 0 / 1 |

- 环境：NPU_A5 Docker，工作目录 `/home/l30023782/PTA_210/combo_kernel_ut`。
  使用允许共用的物理卡 1、2、3，子进程以 `ASCEND_RT_VISIBLE_DEVICES` 映射为逻辑卡 0。
- 安装包：`torch 2.10.0+cpu`、`torch_npu 2.10.0.post7+gite32cdeb`、Triton `3.6.0`，CANN `9.1.0`。
  torch_npu 提交为 `e32cdebf77e797fc85a1a8c3fe1e0eef050f9c0e`；
  已核对 7 个 NPU 安装源码文件的 SHA-256，均与本地目标分支一致；
  继承的社区 Combo 源码也与上游 v2.10.0 一致。
- 测试来源：远端 clone 超时，使用本机同仓库提交
  `be6049a0059677ca408da1360653b694b6d1afb5` 导出的 `v2.13.0/inductor_npu` 文件。
- 执行范围：54 个已有方法、7 个部分支持方法的共享块/固定上限分支、1 个已有行为方法。
  11 条嵌套函数记录随外层执行；42 个缺能力/缺修复方法和 1 个 XPU 专项方法不运行。
  预收集时发现的 21 个前缀同名额外实例已在执行前排除，核定实例总数为 409。
- 适配仅位于工作目录的测试文件及测试进程，不修改安装包。
  补测试导入符号，移除旧版没有的中性初始化配置和 `_foreach_clone` 参数，
  尝试运行原有宽泛 NPU skip 的目标方法；保留原输入、精度阈值、输出和梯度检查。
  CUDA kernel 数量、FileCheck、profiler/grid 和 foreach 生成代码结构检查不作为功能通过条件。
  因此“功能通过”不保证采用 Combo 融合，也不代表满足 CUDA 的分组、grid 或代码形式要求。
  原测试中的 `benchmark_combo_kernel` 路径保留，未关闭测速路径以绕过其执行错误。

验证脚本、命令和适配说明见
[执行说明](../../../../scripts/tests/combo_kernel_210/README.md)。
逐实例结果、源码快照、校验和及适配 diff 位于
[实测证据目录](../../../../artifacts/combo-ut-210-npu-a5-20261105/)。

### F1 修复与定向复测

- 修复提交：`6c6454567f795cb580ca07da68884e15bdd16c6a`，已推送至 `pytorch_new` 的 origin。
  开发分支为 `fix/v2.10.0-combo-benchmark-dynamic-args`，基于上述 `e32cdebf77` 基线。
  最终提交只保留 `NPUComboKernel.kernel_benchmark_extra_args()` 的 11 行修复代码，不包含新增测试。
- 复测日期及环境与基线相同；沿用原测试适配，未关闭原用例的 Combo benchmark，
  保留原输入、输出、突变及精度检查。
  通过进程内方法替换加载候选源码，不修改共享安装包；实测文件 SHA-256 与最终修复提交中的文件一致。
  实测发生在提交整理前，后续只删除了新增测试，修复代码未变；不能把安装包版本号写成已安装该提交。
- 五个动态激活、二维 blocking、mutated、no-autotune 及 round-robin 方法修复后通过，
  对应行记录修复 commit。`test_combo_kernel_size_hint_failure` 的 F1 已消除，
  但执行生成 wrapper 时触发 F8，仍记失败；该行的 commit 只表示已应用 F1 修复。
- 生成代码核对：动态激活 benchmark 的 `get_args()` 现在返回 14 个用户参数，
  包含此前缺失的 6 个动态尺寸，与 kernel 用户参数数量和顺序一致；运行时块参数由 launcher 补充。
  unbacked benchmark 也已补齐四个动态尺寸，随后执行阶段才发生 `u0` 未绑定错误。

复测逐阶段报告、生成代码及源码对照见
[F1 修复证据](../../../../artifacts/combo-f1-benchmark-fix-20261105/)。

### F3 修复与定向复测

- 修复提交：`e578ec524d592475fd52702b975ce03d1a95bbf2`，基于 F1 提交 `6c6454567f`，
  已推送至 origin 分支 `fix/v2.10.0-combo-partition-numellist`。
  仅修改 `NPUComboKernel.horizontal_partition()` 及其类型导入，新增 25 行、删除 3 行，没有新增测试。
- 社区自定义分区期间，将 NPU `NumelList` 转换为 SymPy 乘积表达式；通过 `finally`
  恢复原始分组对象，保留 NPU 后续 codegen 所需的各维信息。动态尺寸保留符号表达式，未强制转为整数。
- 2026-11-05 在相同 NPU_A5 Docker 和基线安装包上沿用既有测试适配，使用物理卡 2、3。
  进程内加载 F3 分区方法及已有 F1 benchmark 方法，未修改共享安装包；候选文件 SHA-256
  与最终提交一致，安装源码的 8 项校验仍匹配基线。保留原输入、输出和精度检查，未关闭 Combo benchmark。
- 原 F3 的 7 个方法全部越过 `NumelList.xreplace` 错误，其中 6 个完整通过；
  `test_helper_fn_defined` 在后续 Combo Triton 编译中因缺少 scan helper 定义失败，记录为 F9。
  7 个方法共观察到 13 次分区调用，原始分组对象均已恢复。
- F1 回归 `test_dynamic_shapes_2d_blocking`、`test_dynamic_shapes_activations` 均通过，
  另观察到 6 次分区调用及原分组恢复。它们仍归入此前的 F1 修复结果，不增加通过方法总数。

逐阶段结果、分区观察、生成 kernel 和源码对照见
[F3 修复与 F9 证据](../../../../artifacts/combo-f3-partition-fix-20261105/README.md)。
本次分析沿用上述已完成的远程复测证据，没有新增远程运行，也没有修复 F9。

### 已确认的失败原因

- **F1（6 个方法）：动态 Combo benchmark 入口缺参数。** 动态激活用例在预运行阶段报 launcher 缺少尺寸和块参数。
  已保存的生成 kernel 签名包括 6 个新增的 `y0_numel_i/x1_numel_i`，
  但该 benchmark 文件的 `get_args()` 只生成 6 个张量和 2 个基础符号，没有补上新增尺寸。
  NPU 的 `add_numel_to_args()` 使用轴名生成动态参数，继承的
  `kernel_benchmark_extra_args()` 仍按社区 range tree 的前缀寻找参数，导致 benchmark 输入不完整。
  基线失败发生在数值比较前，不能归为精度不一致。
  后续 F1 修复使 5 个方法通过，剩余 1 个方法越过该错误后暴露 F8，详见上述定向复测。
- **F2（1 个方法）：归约 store 的指针和值类型不匹配。** `test_combo_kernel_per_config_subkernel_red`
  对两个 `(32, 64, 128)` 张量沿 `(0, 2)` 求和。生成代码将归约结果经
  `triton_helpers.promote_to_tensor()` 转为块类型，却用标量 `out_ptr0 + x0` 执行 `tl.store`，
  Triton 以 `Value argument cannot be block type if pointer argument is not a block` 拒绝编译。
  NPU `store_reduction()` 只为非 persistent 多轴归约补块索引，该 persistent 路径仍使用标量指针。
  本轮保留编译失败，不修改后端。

- **F3（7 个方法）：Combo 分区与 NPU 尺寸分组类型不兼容。** NPU `group_fn()` 产生 `NumelList`，
  Combo benchmark 分区调用继承的社区长归约分类逻辑，直接把 `n.group[-1][-1]` 传给
  `shape_env.has_hint()`；后者按标量表达式调用 `xreplace`，触发
  `AttributeError: NumelList object has no attribute xreplace`。属于编译阶段的前后端接口差异。
  已由 `e578ec524d` 修复并复测：6 个通过，剩余 1 个出现独立的 F9，不能将该方法记为通过。

- **F4（1 个方法）：池化在线性与非线性路径分别失败。** `Combo=True` 进入 NPU 访存线性化的
  `transform_dims_in_indexing()`，经 `split_expression()` 构造 `FloorDiv` 子项后，
  `eliminate_zero_term()` 未检查化简结果类型便解包 `term.args`，
  触发 `not enough values to unpack (expected 2, got 0)`，属于线性化索引变换阶段的错误。
  `Combo=False` 的最终栈则进入 `_nolinear_triton_scheduling`，在 CSE 的
  `assert output_shape is not None` 处失败；不能将其也记为上述解包错误。
  普通节点调度源码会捕获线性 codegen 异常并回退非线性路径，但当前日志未记录被捕获的异常，
  尚不能确认该分支此前是否也遇到了同一个线性化错误。
  该 shape 断言还涉及版本适配：2.13 测试类新增开启 `runtime_triton_shape_assert`，
  而 2.10 的 `ShapePropagationOpsHandler.index_expr()` 返回 `None`，CSE 检查直接断言其非空。
  因此 `Combo=False` 应先按测试检查与旧框架不兼容处理，不能直接判为 NPU 普通功能缺陷。
  两个分支均未完成数值比较。
- **F5（1 个方法）：随机数在两个参数分支的失败不同。** `Combo=True` 生成代码将 `base_x0`
  转为 `tl.int64`，启用的 dtype 检查却断言 `tmp1.dtype == tl.int32`，触发 Triton 编译期断言。
  这是轴索引与 dtype 检查元数据的一致性问题；生成代码调用随机函数前还将索引转为 `tl.uint32`，
  当前不能据此认定随机数算法或数值结果错误。
  `Combo=False` 未进入该 Triton 编译错误，而是在非线性路径的 CSE 中触发
  `assert output_shape is not None`，与 F4 的该分支一样先按 shape 检查版本适配处理。
  两个分支均尚未执行到随机数确定性比较。

- **F6（1 个方法）：eager 不支持目标内存格式。** block_size 用例的第一步为
  `contiguous(memory_format=torch.channels_last)`，NPU eager 直接报告仅支持 contiguous memory format。
  尚未进入 `torch.compile`；保留原内存布局要求，不替换算子或改变输入来绕过。

- **F7（1 个方法）：provenance 映射为空。** Inductor 分支完成编译执行后，
  `_check_provenance_tracing_kernel_to_post_grad()` 读到的 `cppCodeToPost` 为零项，
  与预期的 kernel 到 `relu/sigmoid/tanh` 映射不符。该方法先测 Inductor、再测 AOTI，
  因前者断言失败，AOTI 分支未执行；此项是追溯元数据失败，未报告数值不一致。

- **F8（F1 修复后新增记录，1 个方法）：benchmark 尺寸收集向真实 wrapper 写入提前求值语句。**
  `test_combo_kernel_size_hint_failure` 已完成编译和 Combo benchmark，
  随后在执行 wrapper 时抛出 `UnboundLocalError: cannot access local variable 'u0'`。
  生成代码保留了如下顺序，`u0` 要等 `nonzero` 执行后才能确定：

  ```python
  _x0_numel = 1 + u0
  _x0_numel = 2*u0
  _x0_numel = u0 + ((-1)*s77)
  with torch.npu.utils.device(0):
      torch.npu.set_device(0)
      # Topologically Sorted Source Nodes: [nz], Original ATen: [aten.nonzero]
      buf0 = torch.ops.aten.nonzero.default(arg1_1)
      u0 = buf0.size(0)
  ```

  源码链路为普通 `TritonKernel.kernel_benchmark_extra_args()`
  → `add_numel_to_call_args("", ...)`
  → NPU `generate_node_numel_expr("", ...)`。
  对 `u0 + 1` 等复合尺寸，NPU helper 无条件调用 `self.writeline()`，
  因而在生成 benchmark 参数时向真实 wrapper 插入 `_x0_numel` 赋值。
  空 kernel 名字正是 benchmark 收集路径的标记；社区 `generate_numel_expr()`
  对空名字只返回参数对象，不写 wrapper，NPU helper 缺少对应处理。
  这解释了三条无 kernel 名前缀的赋值及其错误位置；最终 Combo 正常调用直接传尺寸表达式，
  并不需要这些提前赋值。

  **归类：与 F1～F7 的既有失败原因不同，新增 F8；属于 NPU wrapper 与 benchmark 的适配问题，应该修复。**
  `e32cdebf77` 基线中的上述两个 NPU 方法已包含同样逻辑，F1 提交未修改它们；
  因此是基线中已有、越过 F1 后实际暴露的缺陷，不是这次补动态参数新写入的 wrapper 行。
  目前只在该用例确认触发；未额外做禁用 Combo 等对照运行，不能扩大为所有 unbacked 场景都失败。
  建议让 benchmark 尺寸收集不产生 wrapper 写入，保留正常 kernel 调用的尺寸生成；本次仅分析，未修复 F8。
  完整 wrapper 与四段版本化源码保存在 F1 修复证据目录。

- **F9（F3 修复后新增记录，1 个方法）：NPU Combo 漏写子 kernel 的 scan helper 定义。**
  `test_helper_fn_defined` 计算 `x.sum(1), y.mean(1), z.cumsum(1)`，输入为连续 float32 张量，
  shape 分别为 `(16, 128)`、`(32, 128)`、`(32, 256)`。
  失败发生在 Combo benchmark 的 Triton 编译阶段，外层为 `NoTritonConfigsError`，
  内层为 `NameError('_triton_helper_fn_add0 is not defined')`；尚未完成最终输出比较。

  scan 即前缀累计计算。此处 `cumsum` 通过 `tl.associative_scan` 调用加法 helper，
  需要在同一 generated module 的全局作用域定义该函数。已保存的独立 scan kernel 有 1 个定义、1 次调用，
  Combo kernel 有 0 个定义、1 次调用，失败语句为：

  ```python
  tmp15, = tl.associative_scan((tmp13,), 1, _triton_helper_fn_add0)
  ```

  源码链路为 NPU `scan()` 调用继承的 `TritonKernel._lift_helper()`，后者生成函数定义并登记到
  `sub_kernel.helper_functions`；普通 NPU kernel 会输出该集合。
  上游 2.10 的 `ComboKernel.codegen_kernel()` 也会遍历各子 kernel 的 helper，按完整定义去重，
  然后输出到主 kernel 之前。NPU `NPUComboKernel.codegen_kernel()` 重写了生成流程，
  只拼接子 kernel body，遗漏 helper 集合的输出，造成调用存在而定义缺失。

  **归类：NPU Inductor Combo codegen 缺陷，应该修复；与 F1～F8 根因不同。**
  已比较基线 `e32cdebf77`、F1 `6c6454567f`、F3 `e578ec524d` 的该方法，语法树完全一致；
  缺陷在基线中已有，F3 仅使该用例到达后续失败阶段。上游 2.10 已有相应生成逻辑，
  不是依赖 2.13 新接口，也不能根据外层异常归因为 tiling 配置不足或 Triton 不支持 cumsum。
  当前仅确认此用例触发；其他需要动态 helper 的 Combo 路径可能受影响，尚未逐项验证。

  建议在 NPU Combo 中补齐子 kernel helper 的收集、去重及输出，放在 Combo 主 kernel 之前，
  复用上游行为，不为该函数名添加特例。修复后仍需复测此用例的输出、helper 定义及多子 kernel 去重，
  并检查是否有后续错误；本次只分析和记录，F9 尚未修复。

动态 grid 用例最初因上游 `ProfilerActivity` 没有 `NPU` 成员失败；
移除 CUDA profiler/grid 检查并保留两组动态形状的原始输出断言后，复测通过。
首次日志保存在证据目录的 `attempt1_profiler/`。
混合尺寸 mask 用例先通过数值比较，随后因生成代码未使用 CUDA 期望的 foreach decorator 而失败；
当前 NPU 选择普通 pointwise kernel。移除该结构检查后功能复测通过，首次日志位于 `attempt1_cuda_structure/`。

## 逐项对照表

存在性及直接运行列保留未应用修复的原始文件口径；实测及修复 commit 列记录临时适配后的最新功能验证，二者不能混用。
修复后通过的行记录对应 commit；仍失败的 unbacked 和 helper 行分别注明仅修复 F1、F3，
对应的 F8、F9 尚未修复。

“通过”以全部目标参数化实例完成为条件，跳过不计为通过。

### test_triton_kernels.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_combo_fp_self_sub_rewrite</code></td>
<td>无（缺修复：浮点自减的局部数值改写）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
</tbody>
</table>

### test_torchinductor.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_adaptive_avg_pool2d1</code></td>
<td>一致</td>
<td>有</td>
<td>否：文件导入 2.10 缺少的 highest_precision_float</td>
<td>失败（0/2）</td>
<td>—</td>
<td>F4：Combo=True 线性化索引拆分解包失败；Combo=False 非线性路径 output_shape 断言失败</td>
</tr>
<tr>
<td><code>test_mean</code></td>
<td>存在（差异：开启 Combo）</td>
<td>有</td>
<td>否：文件导入 2.10 缺少的 highest_precision_float</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_rand_like_deterministic</code></td>
<td>存在（差异：增加 random.seed）</td>
<td>有</td>
<td>否：文件导入 2.10 缺少的 highest_precision_float</td>
<td>失败（0/2）</td>
<td>—</td>
<td>F5：Combo=True 索引 dtype 检查不一致；Combo=False 触发旧框架 shape 检查断言</td>
</tr>
<tr>
<td><code>test_combo_kernel_store_mask</code></td>
<td>无（缺修复：动态 Combo 常量索引 store mask）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_filter_cpu</code></td>
<td>一致</td>
<td>有</td>
<td>否：文件导入 2.10 缺少的 highest_precision_float</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
</tbody>
</table>

### test_combo_kernels.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_activation_functions</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_activations</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_reduce_functions</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_reduce</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_mutated_args</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_mutated</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_reduce_split</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_blocking_partitioning</code></td>
<td>存在（差异：增加 C++ wrapper grid 检查）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_persistent_reduction_size_hint</code></td>
<td>无（缺修复：按最大归约维选择 size hint）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_sort_in_combo_kernel_forces_persistent_reduction</code></td>
<td>无（缺修复：Combo sort 强制持久归约）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_fuse_mix_order_reductions_combo_kernels</code></td>
<td>无（缺修复：混合顺序归约及消费者排除）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_scalar_store_broadcast</code></td>
<td>无（缺修复：标量 store 指针广播）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_single_combo_kernels</code></td>
<td>无（缺能力：单节点分区改用普通 kernel）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_poi</code></td>
<td>无（部分支持：共享块已有，独立块缺失）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_per</code></td>
<td>无（部分支持：共享块已有，独立块缺失）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_red_per</code></td>
<td>无（部分支持：共享块已有，独立块缺失）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_red</code></td>
<td>无（部分支持：共享块已有，独立块缺失）</td>
<td>有</td>
<td>—</td>
<td>失败（0/1）</td>
<td>—</td>
<td>F2：归约结果为块类型、store 指针为标量，Triton 编译失败</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_r0_r1</code></td>
<td>无（缺能力：二维归约排除 Combo 策略）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_poi_3d</code></td>
<td>无（缺能力：3D pointwise 排除策略）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_config_subkernel_block_size</code></td>
<td>无（部分支持：共享块已有，独立块及扁平 grid 缺失）</td>
<td>有</td>
<td>—</td>
<td>失败（0/1）</td>
<td>—</td>
<td>F6：NPU eager 不支持 channels_last contiguous，参考输出阶段即失败</td>
</tr>
<tr>
<td><code>test_combo_kernels_pointwise_only</code></td>
<td>无（缺能力：仅融合 pointwise 的开关）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_max_num_nodes</code></td>
<td>无（部分支持：节点上限固定 8，不可配置）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_combo_kernel_yz_overflow</code></td>
<td>无（缺修复：grid Y 溢出转移至 Z）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_no_bench_stitched_config</code></td>
<td>无（缺能力：免 benchmark 拼接独立块配置）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_no_bench_persistent_reduction</code></td>
<td>无（缺能力：免 benchmark 持久归约配置拼接）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_no_bench_carve_out</code></td>
<td>无（缺能力：免 benchmark 不适配节点拆出）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_activation_benchmark</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_activations</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_reduce_benchmark</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_reduce</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_mutated_benchmark</code></td>
<td>存在（差异：kernel 数 6/9→5～10）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_mutated</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_round_robin_dispatch</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_mutated</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_2d_blocking_benchmark</code></td>
<td>存在（差异：kernel 数 7～8→6）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_persistent_reduction_no_x_dim</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_combo_kernel_per_subkernel_reduction_hint</code></td>
<td>无（缺能力：逐子 kernel 归约方向提示）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_rms_norm_forwards_add_persistent_rblock</code></td>
<td>无（缺能力：转发持久归约候选标记）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_dynamic_shapes_grid_changes</code></td>
<td>无（部分支持：动态 grid 已有，独立块扁平 grid 缺失）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>移除 CUDA profiler/grid 检查；保留原输出比较</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_activations</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>6c6454567f</td>
<td>原 F1：动态 benchmark 缺尺寸参数；修复后输出/突变检查通过</td>
</tr>
<tr>
<td><code>test_activations</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_2d_blocking</code></td>
<td>存在（差异：kernel 数 7～8→6）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>6c6454567f</td>
<td>原 F1：动态 benchmark 缺尺寸参数；修复后输出/突变检查通过</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_reduce</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_reduce</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_mutated</code></td>
<td>存在（差异：增加 XPU 跳过条件）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>6c6454567f</td>
<td>原 F1：动态 benchmark 缺尺寸参数；修复后输出/突变检查通过</td>
</tr>
<tr>
<td><code>test_mutated</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_activations_no_autotune</code></td>
<td>存在（差异：kernel 数=5 改为源码编译调用数=1）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>6c6454567f</td>
<td>原 F1：动态 benchmark 缺尺寸参数；修复后输出/突变检查通过</td>
</tr>
<tr>
<td><code>test_activations</code></td>
<td>一致</td>
<td>嵌套</td>
<td>不适用：嵌套函数；所属文件同样导入失败</td>
<td>随外层方法验证</td>
<td>—</td>
<td>嵌套函数，不独立计数</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_persistent_reduction_no_x_dim</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_persistent_reduction_no_x_dim_2</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_2d_blocking_round_robin</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>6c6454567f</td>
<td>原 F1：动态 benchmark 缺尺寸参数；修复后输出/突变检查通过</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda</code></td>
<td>存在（差异：装饰器仅限 CUDA）</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后通过（1/1）</td>
<td>e578ec524d</td>
<td>原 F3：分区不兼容 NumelList；修复后输出检查通过</td>
</tr>
<tr>
<td><code>test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu</code></td>
<td>无（待验证：已有同体 GPU 测试，XPU 待验证）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>XPU 专项，不在 NPU 验证范围</td>
</tr>
<tr>
<td><code>test_combo_kernel_tma_descriptor_block_name</code></td>
<td>无（缺修复：Combo TMA 前导代码及块名替换）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_amd_special_config_args</code></td>
<td>无（缺修复：独立块路径的 HIP 编译参数传递）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_helper_fn_defined</code></td>
<td>一致</td>
<td>有</td>
<td>否：导入缺少 requires_xpu_and_triton</td>
<td>修复后仍失败（0/1）</td>
<td>e578ec524d（仅修复 F3，F9 未修复）</td>
<td>F3 已修复；新增 F9：NPU Combo 漏写 scan helper 定义，Triton 编译报 NameError</td>
</tr>
<tr>
<td><code>test_pdl_codegen_in_combo_kernel</code></td>
<td>无（缺能力：Combo PDL 代码生成）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_pdl_combo_kernel_pointwise</code></td>
<td>无（缺能力：Combo pointwise PDL 接入）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_pdl_combo_kernel_reduction</code></td>
<td>无（缺能力：Combo reduction PDL 接入）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_max_autotune</code></td>
<td>无（缺能力：独立块逐子 kernel 最大调优）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_max_autotune_with_reduction</code></td>
<td>无（缺能力：独立块归约最大调优）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_autotune_many_subkernels</code></td>
<td>无（缺能力：逐子 kernel 串行调优策略）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_autotune_grouping</code></td>
<td>无（缺能力：相同子 kernel 分组调优）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_autotune_grouping_by_metadata_fingerprint</code></td>
<td>无（缺能力：按完整元数据指纹分组）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_max_persistent_rblock_gated_on_per_subkernel</code></td>
<td>无（缺能力：持久归约块上限及分模式控制）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_coordesc_tunes_largest_subkernel_first</code></td>
<td>无（缺能力：Combo 坐标下降及大子 kernel 优先）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_inductor_meta_has_optimize_mem</code></td>
<td>无（缺能力：Combo 内存优化标记）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_inductor_meta_optimize_mem_false_in_training_forward</code></td>
<td>无（缺能力：训练前向内存优化标记）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_triton_meta_has_disable_ftz</code></td>
<td>无（缺能力：disable_ftz 配置及传递）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields</code></td>
<td>无（缺能力：子 kernel pointwise 元数据）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_reduction_combo_grid_meta_has_per_subkernel_fields</code></td>
<td>无（缺能力：子 kernel reduction 元数据）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel</code></td>
<td>无（缺能力：子 kernel 连续归约维元数据）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_per_kernel_inductor_meta_matches_standalone</code></td>
<td>无（缺能力：与普通 kernel 一致的子级元数据）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_num_gb_and_flop_match_standalone_sum</code></td>
<td>无（缺能力：Combo 字节量及 FLOP 汇总）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth</code></td>
<td>无（缺能力：带宽分析的 Combo 字节量）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_inductor_meta_no_kernel_num_gb_without_profile</code></td>
<td>无（已有行为：默认不输出统计字段；新增覆盖）</td>
<td>有</td>
<td>—</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_benchmark_combo_kernel_emits_real_num_gb</code></td>
<td>无（缺能力：benchmark 使用真实字节量）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_combo_kernel_peak_memory_wide_resnet</code></td>
<td>无（缺能力：按峰值内存限制融合）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
</tbody>
</table>

### test_foreach.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_single_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（32/32）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_single_scalar</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_single_scalar_tensor</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_scheduler_fusion_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（32/32）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_scheduler_fusion_scalar</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_broadcasting</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_type_promotion</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（20/20）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_kernel_split_arg_limit_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_fusion_duplicate_buffer_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（20/20）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_non_foreach_producer_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（32/32）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_non_foreach_producer_scalar</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_non_foreach_consumer_producer_list</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（32/32）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_non_foreach_consumer_producer_scalar</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（16/16）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_enable_dynamic_shapes_python_wrapper</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_decomp</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_fuse_concat</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_blocking</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（20/20）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_blocking_partitioning_elems</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（20/20）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_blocking_partitioning_mixed_sizes</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（20/20）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_reinplacing</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（4/4）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_reinplacing_mut_before</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（4/4）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_reinplacing_mut_after</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（4/4）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_aliasing</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_block_no_mixed_sizes_no_mask</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_2d_block_mixed_sizes_with_mask</code></td>
<td>存在（差异：启用子 kernel 独立块并检查生成代码）</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（1/1）</td>
<td>—</td>
<td>数值通过；移除 CUDA foreach 生成代码结构断言</td>
</tr>
<tr>
<td><code>test_foreach_map_backward_binary</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（10/10）</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><code>test_foreach_map_backward_unary</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖文件导入失败，且缺少 _foreach_clone</td>
<td>通过（4/4）</td>
<td>—</td>
<td>—</td>
</tr>
</tbody>
</table>

### test_snode_runtime.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_foreach_basic</code></td>
<td>无（缺能力：调度节点输入距离指标）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
<tr>
<td><code>test_foreach_after_extern</code></td>
<td>无（缺能力：外部算子后的输入距离指标）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
</tbody>
</table>

### test_unbacked_symints.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_combo_kernel_size_hint_failure</code></td>
<td>一致</td>
<td>有</td>
<td>否：本轮 NPU 实测 Combo benchmark 缺动态尺寸参数</td>
<td>修复后仍失败（0/1）</td>
<td>6c6454567f（仅修复 F1，F8 未修复）</td>
<td>F1 已修复；新增 F8：benchmark 收集尺寸时污染 wrapper，u0 在 nonzero 绑定前使用</td>
</tr>
</tbody>
</table>

### test_provenance_tracing.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_triton_kernel_to_post_grad_tracing_combo_kernel</code></td>
<td>一致</td>
<td>有</td>
<td>否：依赖的 test_torchinductor 导入失败</td>
<td>失败（0/1）</td>
<td>—</td>
<td>F7：Inductor 运行后追溯映射 cppCodeToPost 为空；AOTI 分支未执行</td>
</tr>
</tbody>
</table>

### test_user_streams.py

<table>
<thead><tr>
<th>用例 / 函数名</th>
<th>v2.10.0 存在性 / 差异</th>
<th>v2.13.0</th>
<th>2.13 用例能否直接在 2.10 运行</th>
<th>NPU_A5通过情况</th>
<th>修复commit</th>
<th>失败原因分析</th>
</tr></thead>
<tbody>
<tr>
<td><code>test_combo_kernel_fusion_within_same_stream</code></td>
<td>无（缺能力：按 stream 分组及传递 stream）</td>
<td>有</td>
<td>—</td>
<td>未运行</td>
<td>—</td>
<td>缺新增能力或场景修复，按范围排除</td>
</tr>
</tbody>
</table>

## 复核方法

在本地 CUDA 社区源码仓 `/home/qc/Repos/Ascend/pytorch` 执行，只读取标签，不切换或修改工作区：

```bash
git rev-parse 'v2.10.0^{commit}' 'v2.13.0^{commit}'
git ls-remote --tags origin refs/tags/v2.10.0 refs/tags/v2.13.0
git show v2.10.0:test/inductor/test_combo_kernels.py
git show v2.13.0:test/inductor/test_combo_kernels.py
```

对其余 7 个文件同样读取。AST 核对函数名、所属类与外层函数；
未找到的 51 个函数名逐行放入临时文本，再执行以下命令复核是否移动到其他测试文件：

```bash
git grep -n -w -F -f /tmp/missing_test_names.txt v2.10.0 -- test
```

本次全目录复查无匹配，退出码为 1。表格与 CSV 均保留全部 116 条记录；
汇总统计不包含参数化后生成的测试实例数。

功能分析进一步读取对应生产实现，并用 `git log v2.10.0..v2.13.0 -G` 定位新增测试的提交，
再对照 `git show <commit> -- torch/_inductor`，避免仅凭测试名或配置项缺失推断算子不受支持。
