---
published: false
---

# 2.13 Combo/foreach 最新结果与问题分析

<a id="validation213"></a>

<!-- combo-section:validation213:start -->
## 最新执行结果

2026-11-12 完成 15 个修改源方法的定向验证，包含动态 grid、store mask 和各类 SKIP 条件。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 本轮 15 个修改源方法及 15 个参数实例（最终结果） | 2 | 0 | 13 |
| 合并历史结果的 105 个源方法 | 74 | 14 | 17 |
| 合并历史结果的 314 个已选实例 | 281 | 15 | 18 |
| 2.10 基线 PASS 的 60 个重合方法 | 55 | 0 | 5 |

**本轮只定向验证 15 个修改方法，没有重新执行其余 90 个方法或 foreach。**
未复验的 14 个 FAIL 方法、15 个失败实例沿用参考报告，不能称为本轮新确认失败。
74 个 PASS 中有 14 个抽样 PASS；原清单 464 个泛化实例的 150 个未选择实例不计为 SKIP 或 PASS。
2.10 重合对照仍是既定的 60 方法适配基线，与 54 方法原生清单口径不同。
当前重合范围的 5 个 SKIP 为 4 个独立分块方法和 1 个 NPU CPU-filter 实例，不是计算错误。

[用例 CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv) 保持 116 行原顺序和五列。
11 行嵌套函数随外层方法验证，不另计源方法数。PASS 的当前分析统一为“—”。
本轮 13 个 SKIP 分别是独立分块待实现 8 个、CPU 过滤测试范围不适用 1 个、CUDA TMA/PDL 专属 4 个。
另外 4 个 SKIP 沿用参考报告。SKIP 不计为功能实现或功能修复。

动态 grid 从真实 launcher 观测启动范围，排除 autotune 的计时启动，
检查两组尺寸的 X grid 不同、Y/Z 为 1，并保留数值检查。
store mask 保留原输入数值检查，新增实际双成员 NPU Combo，检查两个分支的尾块边界及全部 store mask。
**新增 store-mask 场景在用例内关闭 dtype 调试断言；相关 codegen 诊断问题仍未修复。**
原标量输入仍被拆成三个普通 kernel，因此尚未覆盖社区原标量 Combo 写竞争路径，详见下文验证前提。

环境：NPU_A5/Ascend950PR，`pta_213_case`，PyTorch `2.13.0+cpu`、Triton `3.6.0`。
基础 wheel 为 `torch_npu 2.13.0+git64dc85c`，实际 NPU Python 为既有 `d12230b556` 覆盖版本。
本轮未修改生产代码或安装包；两阶段执行前后源码哈希一致，并与参考报告的 574 个安装文件哈希逐项一致。
物理卡 1、3，使用独立缓存，执行原生 `python -m pytest -vs`，无自定义插件。

测试分支 `fix/combo-213-npu-case-adaptation` 已推送本轮提交 `e71a557`。
14 个方法的正式报告来自 `4acf77e`，store mask 最终报告来自 `64015d3`；
这 14 个方法在最终提交中未改变。`e71a557` 相对最终验证版本仅修正诊断注释，执行 AST 一致。
逐实例实际提交保存在合并证据 `jobs.json` 的 `validation` 中，源码核对见 `source-validation.json`。

foreach 继续沿用 NPU 功能适配：保留数值、mutation 和梯度检查。
其抽样 PASS 不证明全部参数通过，也不证明 FOREACH 分组融合。

[执行方式](../../../../scripts/tests/combo_kernel_210/README-213.md)；
[本轮 JUnit、日志、精确 node ID、源码哈希和生成代码](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/)；
[2.10 重合方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/baseline-pass-comparison.json)。
未复验方法的实际验证版本保留在合并报告中，来源为对应参考证据。

<!-- combo-section:validation213:end -->

<a id="analysis213"></a>

<!-- combo-section:analysis213:start -->
## 当前问题与优先级

以下 14 个源方法仍保留最近失败报告，15 个参数实例；本轮没有重跑这些未修改的方法。
失败检查涵盖统计、追踪、后端专用调优协议和 eager 支持范围，不能把它们都算成计算错误。

| 优先级 | 问题组 | 方法数 | 当前判断 |
| --- | --- | ---: | --- |
| P1 | `optimize_mem` 图阶段语义 | 2 | Combo 未显式传入字段，runtime 默认 True；训练前向应按社区图阶段语义传 False。尚未观察到数值错误 |
| P1 | provenance 追踪 | 1 | `cppCodeToPost` 映射为空，NPU `define_kernel` 未接社区 origins 登记 hook |
| P1 | 真实统计与逐成员统计元数据 | 7 | 字节数/FLOP 聚合 3 个、逐成员元数据 4 个；需区分实际消费者要求，不能填零或空字典冒充统计 |
| P2 | 峰值内存门控 | 1 | 紧阈值与禁用阈值均为 616101888 字节；尚未证明是设计差异，保留严格峰值断言 |
| 待实现能力 | 逐成员调优契约 | 2 | INNER/OUTER hint、额外大 RBLOCK 候选尚未接入；共享 tiling 不能替代对应语义，也不能据字段缺失推定 NPU 没有 autotune |
| 待核实 | FTZ 数值语义 | 1 | 两个参数实例缺少 `disable_ftz` 接入；尚未证明硬件不支持保留非正规数 |
| 合计 | 仍失败 | 14 | 均为未在本轮重跑的既有失败报告 |

### 已验证的 NPU 测试口径

| 检查内容 | NPU 当前设计与验证要求 | 源码依据 |
| --- | --- | --- |
| persistent reduction size hint | split-axis 尺寸求和，其余轴取最大；不照搬 GPU 的二次幂 hint | `NPUComboKernel.select_combo_heuristics` |
| 普通 kernel 与 Combo grid | 普通 kernel 使用 `GridNpu`；Combo 使用一维 `SequentialComboKernelGrid`。动态尺寸用例从真实 launcher 读取 grid，排除 autotune 启动，检查两组尺寸的 X grid 不同及 Y/Z 为 1，并保留数值检查 | `CompiledKernel.launch_metadata`、`SIMDDispatch`、`SequentialComboKernelGrid` |
| mixed-order 数量 | NPU 关闭混合行列归约；行归约、列归约、消费者及 y/z Combo 共 4 个定义，并检查仅有一个双成员 Combo | `torch_npu/_inductor/config.py` 显式设置 `mix_order_reduction=False` |
| scalar broadcast 数量 | `(x0,r1)` 与 `(r0)` 轴布局使用不同分块接口，生成 2 个普通 kernel；检查数量、两类实际轴布局和数值 | `NPUComboKernel.horizontal_partition` 按 kernel 类型、模式和轴标记细分分区 |
| bucketize 数量 | bucketize 调用 ATen fallback；其余两个逐点成员合为 1 个 Triton Combo。检查 fallback 调用及双成员结构，不能把 Triton 定义数当作全部设备 launch 数 | `lowering_fallback_list.py` 注册 `aten.bucketize.Tensor`；实际 wrapper 保留对应调用 |
| 无 benchmark 的逐成员配置 | 原方法要求独立分块及逐成员默认配置拼接，NPU 待实现并 SKIP；共享 default_config 不作为等价覆盖 | 用例显式打开 `combo_kernel_per_subkernel_blocks` |
| max-autotune 与 grouping | 检查完整共享配置有多个实际候选、执行计时、选出赢家并保留一个 launcher；检查成员数量及输出数值。NPU 不要求 GPU 逐成员分组日志 | `_triton_config_npu_index_combo`、`ComboTileGenerator`、`NPUCachingAutotuner.autotune_to_one_config` |
| persistent RBLOCK 上限 | 不带 HIP 专用的 `max_persistent_rblock=1024`，NPU 按自身资源模型筛选共享候选 | NPU `jit_line` 与共享 tiling 候选生成 |
| store mask | 保留原输入数值检查；额外构造行数为 257/513 的兼容动态 mean，检查一个双成员 Combo、各分支边界和两个 store 的 `x0_mask` | 正式用例 FileCheck/JUnit；`compatible-combo-output.py` 为兼容输入独立探测证据 |
| TMA / PDL | TMA 依赖 NVIDIA SM90+ Tensor Memory Accelerator；PDL 依赖 SM90+ grid dependency control 及 CUDA 启动/同步协议。四个方法均明确 SKIP | 用例硬件条件、tensor descriptor 与 `tl.extra.cuda.gdc_*` 断言 |

调优检查只在用例内临时包装 `AsyncCompile.wait` 取得真实对象，并调用原函数；
不替换候选、计时或赢家。关闭缓存并将编译线程设为 1 是为了确保本次确实执行待检查路径，退出即恢复。
NPU 的共享 tiling 通过不代表 GPU 的独立逐成员分块或逐成员 coordesc 已实现。
普通 GPU 分支的原断言保留。
动态 grid 的 hook 只记录实际参数并调用原函数，不改 grid 或调优结果。

### 独立分块：待实现功能

以下 8 个源方法在 NPU 明确 SKIP，不再把共享分块下的数值 PASS 记为独立分块功能通过。
这是相对社区 2.10 新增的编译器优化能力，不属于 CUDA 专属硬件协议。

| 用例 | 待实现能力 |
| --- | --- |
| `test_combo_kernel_per_config_subkernel_poi` | 逐成员 pointwise 分块参数 |
| `test_combo_kernel_per_config_subkernel_per` | 逐成员 persistent reduction 分块 |
| `test_combo_kernel_per_config_subkernel_red_per` | persistent/普通 reduction 各自分块 |
| `test_combo_kernel_per_config_subkernel_red` | 逐成员 reduction 分块参数 |
| `test_combo_kernel_per_config_subkernel_block_size` | 逐成员 tile 及对应启动 grid；本轮因功能未实现提前 SKIP，不验证 channels-last eager 支持 |
| `test_combo_kernel_no_bench_stitched_config` | 无 benchmark 时拼接逐成员默认配置并与 body 对齐 |
| `test_combo_kernel_no_bench_persistent_reduction` | 无 benchmark 时生成逐成员 XBLOCK 和静态 reduction block |
| `test_combo_kernel_coordesc_tunes_largest_subkernel_first` | 独立分块基础上的逐成员坐标下降顺序 |

三个显式打开独立分块的继承测试类也标记 NPU SKIP，避免换类执行后误报功能通过；
本轮定向清单不另外展开这些继承实例。
`per_config_subkernel_r0_r1`、`per_config_subkernel_poi_3d` 在当前所选共享分块类中检查的是
不兼容布局拆成普通 kernel，原结构断言保留；它们的 PASS 不代表独立分块支持。

### CPU 过滤和 CUDA 硬件专用用例

`test_combo_kernel_filter_cpu` 原本位于 CPU/GPU 共用模板，并非只在 CPU 上执行。
CPU 实例验证禁用 Combo 融合，CUDA 实例验证融合数量；原 NPU 实例只有数值检查，
不能证明 CPU 节点过滤策略。本轮将 NPU 实例标记为测试范围不适用 SKIP，保留 CPU/CUDA 原路径。
这不表示 NPU 不支持用例中的乘法、加法、min 或 max。

TMA 用例依赖 NVIDIA SM90+ 的 Tensor Memory Accelerator，检查 tensor descriptor 生成。
三个 PDL 用例依赖 SM90+ grid dependency control 硬件和 CUDA 启动/同步协议，
包括 `launch_pdl`、`gdc_wait`、`gdc_launch_dependents`；均标记 CUDA 特有能力、NPU SKIP。
PDL reduction 不再以单纯数值 PASS 表示 PDL 支持。

store mask 的原输入在当前 NPU 路径生成三个普通 kernel，因此仍未覆盖社区原来的
“标量成员在 Combo 共享 block 中缺 mask”竞争路径。
新增的兼容双成员场景验证 NPU Combo 的实际尾块 store 保护，不能将两者当成完全相同的覆盖。

### store mask 的诊断验证前提

共用模板默认启用 `test_configs.runtime_triton_dtype_assert`。
独立探测日志显示，NPU 对动态整数 `ks0`（Triton signature 为 `i64`）生成了以下断言：

```python
tmp4 = ks0
tl.static_assert(tmp4.dtype == tl.float32)
```

整数在转换为 float32 之前即被要求具有 float32 dtype，
TTIR 检查失败后回退为两个普通 kernel，无法满足本例的真实 Combo 数量断言。
直接证据见 [TTIR 失败片段](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/dtype-debug-ttir-failure.txt)。
这是仍存在的独立诊断代码生成问题，未作为生产代码修复，也不额外计入本轮源方法 FAIL 数量。

新增 mask 场景仅在用例作用域内关闭该调试项，退出恢复。
随后实际验证一个双成员 Combo、两个分支各自的边界 mask、全部 store mask 和 eager/compiled 数值。
因此该 PASS 有明确的调试配置前提，不代表 dtype 诊断能力已修复。
TMA/PDL 的硬件协议来源保存在 [官方资料记录](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/cuda-protocol-references.json)。

### 仍需处理的具体检查

| 用例 | 当前检查点与依据 |
| --- | --- |
| `test_combo_inductor_meta_has_optimize_mem` | 推理缺 True 字段；NPU Combo `jit_line` 未写入，`cached_autotune` 默认 True |
| `test_combo_inductor_meta_optimize_mem_false_in_training_forward` | 训练前向缺 False 字段，影响 autotune 输入复制/恢复策略 |
| `test_triton_kernel_to_post_grad_tracing_combo_kernel` | `cppCodeToPost` 长度 0；应接 `set_kernel_post_grad_provenance_tracing` 并登记实际子节点 origins |
| `test_combo_kernel_num_gb_and_flop_match_standalone_sum` | Combo 未生成 standalone 统计之和 |
| `test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth` | profile 路径缺 `kernel_num_gb` |
| `test_benchmark_combo_kernel_emits_real_num_gb` | benchmark 未传入真实字节数估算；当前 NPU Combo 使用 `num_gb=0` |
| `test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields` | 缺少逐成员 load/store/reduction 等统计契约 |
| `test_combo_reduction_combo_grid_meta_has_per_subkernel_fields` | reduction 路径同类契约缺失 |
| `test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel` | 首先缺逐成员元数据；连续 reduction 语义还需按 NPU 实际地址轴推导 |
| `test_combo_per_kernel_inductor_meta_matches_standalone` | standalone 2 组、Combo 0 组；现有轴/tiling 约束不等同于统计字段 |
| `test_combo_kernel_peak_memory_wide_resnet` | 门控没有观察到峰值降低；还需将实际融合选择与 buffer 生命周期对应，不能直接放宽断言 |
| `test_combo_kernel_per_subkernel_reduction_hint` | 要求逐成员 `reduction_hint_i`；NPU 现有布局分区/共享搜索不能直接等同该契约 |
| `test_combo_rms_norm_forwards_add_persistent_rblock` | 要求将社区大 RBLOCK 优化逐成员传递给候选生成；NPU 尚未提供该专用候选契约 |
| `test_combo_triton_meta_has_disable_ftz` | 两个参数实例缺后端 FTZ 选项接入，需要核对真实非正规数行为 |

真实统计的普通 kernel 出口在 NPU `triton.py::codegen_kernel`；Combo `jit_line` 尚无对应聚合出口。
`combo_subkernel_meta` 只包含实际尺寸、轴和 tiling 约束，不能替代 load/store/FLOP 统计。
FTZ 指把非正规浮点数冲刷为零，不能仅凭字段出现宣布支持。
共用卡上的本轮验证不构成性能 benchmark，不能据 kernel 数量直接判断性能升降。

## 复现

使用证据目录中的精确 node ID 清单，在相同测试快照中执行原生 pytest：

```bash
conda activate pta_213_case
ASCEND_RT_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=/path/to/new-cache \
python -m pytest -vs \
  test_combo_kernels.py::ComboKernelTests::test_combo_kernel_dynamic_shapes_grid_changes \
  test_torchinductor.py::GPUTests::test_combo_kernel_store_mask_npu
```

复用执行脚本与失败项 manifest 可在卡 1、3 并行；每个方法输出独立日志与 JUnit，
完成后合并已有结果。安装源码前后哈希一致不代表全部历史结果已经重跑。

<!-- combo-section:analysis213:end -->
