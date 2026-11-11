---
published: false
---

# 2.13 Combo/foreach 最新结果与问题分析

<a id="validation213"></a>

<!-- combo-section:validation213:start -->

## 最新执行结果

2026-11-11 已基于 `sync/combo-kernel-master:d12230b556` 完成此前失败项复验。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 本次 33 个失败方法 | 1 | 32 | 0 |
| 本次 35 个失败参数实例 | 1 | 34 | 0 |
| 保留历史结果后的 105 个方法 | 69 | 32 | 4 |
| 保留历史结果后的 314 个已选实例 | 275 | 34 | 5 |
| 2.10 基线 PASS 的 60 个重合方法 | 60 | 0 | 0 |

**本轮只重新执行失败实例，没有重跑历史 PASS、SKIP 或 foreach 全参数。**
cumsum/helper 本次通过，其余 59 个重合方法沿用历史 PASS；总计 69 个 PASS 中有 14 个抽样 PASS。
原清单 464 个泛化实例中有 150 个未选择，不记为 SKIP 或 PASS。

重合对照使用既定的 60 方法“2.13 用例适配后在 2.10 执行”基线，
与 2.10 最新文档中的 54 方法原生用例清单口径不同，不混算两者结果。

[用例 CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv) 保留 116 行原顺序和五列，
嵌套函数随其外层方法验证。实例列保留实际复验版本，PASS 的当前分析为“—”。
当前最值得处理的是 optimize_mem、provenance 和真实统计/聚合，
详见 [最新问题分类、优先级与源码依据](inductor-combo-tests-v2.13.md#analysis213)。

运行环境是 NPU_A5/Ascend950PR 的 `pta_213_case`；PyTorch `2.13.0+cpu`、Triton `3.6.0`。
基础 wheel 仍为 `torch_npu 2.13.0+git64dc85c`，C++ 未重新构建；
实际将 d12230b556 的全部 191 个 Inductor Python 文件覆盖到安装目录，并核对前后哈希一致。
测试源码为 `c1ce0688898b9ca1edebc2a4ff4dbffd8b287f28`，189 个测试 Python 文件已核对哈希。
本轮使用物理卡 3、6 和独立缓存，直接执行原生 `python -m pytest -vs`，没有自定义插件。

foreach 沿用 2.10 的 NPU 功能适配：保留数值、mutation 和梯度断言，
不检查 CUDA/foreach 专用融合数量和生成文本。抽样 PASS 不证明完整泛化范围通过，也不证明 FOREACH 分组融合。
4 个历史 SKIP 不作为功能缺失证据，本轮没有重试。

[执行脚本及命令](../../../../scripts/tests/combo_kernel_210/README-213.md)；
[本轮日志、JUnit 和哈希证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-d12230b-failure-rerun/)；
[重合方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-d12230b-failure-rerun/baseline-pass-comparison.json)。
逐实例实际验证版本保存在证据 `jobs.json` 的 `validation` 字段。

<!-- combo-section:validation213:end -->

<a id="analysis213"></a>

<!-- combo-section:analysis213:start -->

## 当前问题与建议优先级

以下按源方法计数；多个用例可能属于同一个修复点，不应把方法数当作独立缺陷数。
判断依据是本次日志、生成代码和 `d12230b556` 的实际实现。PASS 项不保留旧失败分析。

| 优先级 | 问题组 | 方法数 | 已确认现象与建议 |
| --- | --- | ---: | --- |
| P1 | `optimize_mem` 图阶段语义未接入 | 2 | Combo 缺字段，runtime 默认 True；训练前向应为 False。优先对齐 `is_inference or is_backward`，同时验证推理和训练前向。当前未观察到数值错误 |
| P1 | provenance 追踪缺失 | 1 | `cppCodeToPost` 映射为空；NPU `define_kernel` 未接社区 provenance hook。展开 Combo 实际子节点登记 origins，覆盖代码复用路径 |
| P1 | 真实统计与逐子 kernel 元数据 | 7 | 带宽/FLOP 聚合 3 个、逐子 kernel 统计 4 个；补真实估算和一致性出口，不能用空字典或零值满足检查 |
| P2 | 融合数量和峰值内存差异 | 4 | 三个数量断言及一个峰值断言不满足。需要进一步记录分区、候选编译、回退和 buffer 生命周期，尚不能定性为主动收紧融合或功能缺失 |
| P2 | 独立分块与逐子 kernel 调优协议 | 9 | 默认独立 block、逐子 kernel reduction hint、大 RBLOCK 候选及 5 个 autotune 方法未对齐社区协议；需贯通参数、grid、launch 和候选消费者，不是只打印字段 |
| 适配 | 后端专用检查/资源策略 | 7 | 3 个 size-hint/grid/profiler 检查可做 NPU 等价适配；TMA 1 个、PDL 2 个为 CUDA 路径，另 1 个 max-persistent-RBLOCK 检查对应 HIP 资源过滤，不应直接复制硬件阈值 |
| 独立能力 | eager 内存格式 | 1 | `channels_last contiguous` 在进入 compile 前即失败；属于 eager 支持范围，替换输入布局会改变覆盖 |
| 待后端核实 | FTZ 数值语义 | 1 | 缺少 `disable_ftz` 元数据和等价后端选项接入；未证明硬件不支持保留非正规数，也不能只补字段宣布支持 |
| 合计 | 当前未通过方法 | 32 | 均不属于 2.10 基线已通过的 60 个重合方法 |

### P1 的具体用例和源码依据

| 用例 | 当前失败点 | 源码依据 |
| --- | --- | --- |
| `test_combo_inductor_meta_has_optimize_mem` | 缺少推理 True 字段 | NPU `triton_combo_kernel.py::jit_line` 未写入；runtime `cached_autotune` 默认 True |
| `test_combo_inductor_meta_optimize_mem_false_in_training_forward` | 缺少训练前向 False 字段 | 同上；社区按图阶段显式传入，影响 autotune 输入复制/恢复策略 |
| `test_triton_kernel_to_post_grad_tracing_combo_kernel` | 映射长度 0，预期 1 | NPU `scheduling.py::define_kernel` 缺 provenance hook；社区 `debug.py::set_kernel_post_grad_provenance_tracing` 登记实际 origins |
| `test_combo_kernel_num_gb_and_flop_match_standalone_sum` | Combo 没有 standalone 统计之和 | 普通 NPU `triton.py::codegen_kernel` 在 benchmark/profile 下写字节估算；Combo `jit_line` 无相同聚合出口 |
| `test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth` | 缺 `kernel_num_gb` | 最新调试器已执行，后续断言暴露相同统计遗漏 |
| `test_benchmark_combo_kernel_emits_real_num_gb` | benchmark 未发出要求的真实字节估计 | 需要将聚合字节估算传到生成 benchmark 的路径 |
| `test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields` | 缺 `inductor_meta_i` | 现有 `combo_subkernel_meta` 提供 NPU 轴/tiling 约束，并不包含同一统计契约 |
| `test_combo_reduction_combo_grid_meta_has_per_subkernel_fields` | 同上，reduction 路径 | 应与普通 kernel 统一真实 load/store/reduction 统计 |
| `test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel` | 首先缺 `inductor_meta_i` | 连续 reduction 标记还需按 NPU 地址轴推导，不能套用 CUDA 轴假设 |
| `test_combo_per_kernel_inductor_meta_matches_standalone` | standalone 2 组，Combo 0 组 | 应统一两种生成方式的有效统计及 tiling 语义 |

逐子 kernel 元数据是“每个融合成员各自的统计和约束”；它与 NPU 已有的轴元数据、
也与真正执行独立分块搜索的调优算法不同。补统计字段本身不能证明独立分块能力已实现。
5 个 autotune 方法的数值断言位于当前失败的行为/日志检查之后，尚未执行，不能标记数值 PASS。
仅凭日志缺失还不能证明 NPU 没有任何等价调优能力，需要比较实际候选、分组和赢家。
FTZ 指把非正规浮点数冲刷为零，涉及真实数值行为，不能以 metadata 是否存在代替后端验证。

### 需要先定位的四个项目

| 用例 | 本次实际结果 | 当前判断 |
| --- | --- | --- |
| `test_fuse_mix_order_reductions_combo_kernels` | kernel 数量预期 3、实际 4 | 分区/候选/回退原因待定位 |
| `test_combo_kernel_scalar_store_broadcast` | kernel 数量预期 1、实际 2 | 同上，需确认 scalar store 的候选是否被拒绝 |
| `test_combo_kernel_no_bench_carve_out` | 编译定义数量预期 2、实际 1 | 查大子 kernel carve-out 条件及 NPU 分区策略；定义数量不能直接等同实际 launch 数量 |
| `test_combo_kernel_peak_memory_wide_resnet` | tight=616101888、disabled=616101888 字节 | 未满足峰值降低断言，与上轮相同；尚未定位是否由候选回退、门控或生命周期导致 |

三个数量用例在失败断言前已通过数值比较。它们说明融合决策与用例预期不同，
并不直接证明生成计算错误。峰值用例还需要将实际 Combo 决策与内存分配对应起来；
在共用卡环境下，本次没有做独立性能 benchmark，不能据这些结果推断性能提升或回退。

## 未执行范围与复现

4 个未执行的 SKIP 保持当前状态：XPU 专用、ROCm 专用各一个，另外两个
`test_snode_runtime.py::InputDistanceTests` 方法仍被 NPU skip 排除。
它们不计入本轮失败复测，也不能由 skip 推断 foreach 算子不支持。
foreach 的既有功能结果沿用每方法最多两个代表实例的策略，抽样 PASS 不代表所有泛化实例通过。

复现单项时，在相同测试快照的 `v2.13.0/inductor_npu` 中执行：

```bash
conda activate pta_213_case
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=/path/to/new-cache \
python -m pytest -vs \
  test_combo_kernels.py::ComboKernelMetadataTests::test_combo_inductor_meta_optimize_mem_false_in_training_forward
```

完整失败实例清单、两阶段命令、JUnit 结果及实际代码哈希见
[本轮证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-d12230b-failure-rerun/)。
当前汇总表仍使用稳定文件名：
[CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)。

<!-- combo-section:analysis213:end -->
