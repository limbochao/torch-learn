---
published: false
---

# 2.13 Combo/foreach 最新结果与问题分析

<a id="validation213"></a>

<!-- combo-section:validation213:start -->
## 最新执行结果

2026-11-11 基于测试提交 `ff5edcd` 完成 NPU 用例适配后的失败项复验。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 本次修改的 16 个源方法 | 13 | 0 | 3 |
| 本次 17 个原失败参数实例 | 14 | 0 | 3 |
| 合并历史结果的 105 个源方法 | 82 | 16 | 7 |
| 合并历史结果的 314 个已选实例 | 289 | 17 | 8 |
| 2.10 基线 PASS 的 60 个重合方法 | 60 | 0 | 0 |

**本轮只复验已修改方法的原失败实例，没有跑全量，也没有重跑 foreach。**
其余报告沿用原始验证版本，未复验的 16 个失败方法不能称为本次新确认失败。
82 个 PASS 中有 14 个抽样 PASS；原清单 464 个泛化实例有 150 个未选择，不能计为 SKIP 或 PASS。
2.10 重合对照仍是既定的 60 方法适配基线，与 54 方法原生清单口径不同。

[用例 CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv) 保持 116 行原顺序和五列。
嵌套函数随外层方法验证，不另计源方法数。PASS 的当前分析统一为“—”。
新增 SKIP 是 1 个 TMA、2 个 PDL CUDA 专用协议检查，不计成功能通过。
其余 4 个 SKIP 沿用历史结果。

环境：NPU_A5/Ascend950PR，`pta_213_case`，PyTorch `2.13.0+cpu`、Triton `3.6.0`。
基础 wheel 为 `torch_npu 2.13.0+git64dc85c`，实际 Inductor Python 为既有 `d12230b556` 的 191 文件覆盖。
本轮未修改生产代码或安装包；执行前后安装源码哈希一致。
测试分支 `fix/combo-213-npu-case-adaptation` 只有一个提交 `ff5edcd`，已推送。
最终远端测试源码与该提交的本地源码逐文件核对一致。
物理卡 1、3 并行，使用独立缓存，执行原生 `python -m pytest -vs`，无自定义插件。

foreach 继续沿用 NPU 功能适配：保留数值、mutation 和梯度检查。
其抽样 PASS 不证明全部参数通过，也不证明 FOREACH 分组融合。

[执行方式](../../../../scripts/tests/combo_kernel_210/README-213.md)；
[本轮日志、JUnit、精确 node ID、源码哈希与生成代码](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-adaptation-20261111/)；
[2.10 重合方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-adaptation-20261111/baseline-pass-comparison.json)。
逐实例实际验证版本保存在合并证据 `jobs.json` 的 `validation` 中。

<!-- combo-section:validation213:end -->

<a id="analysis213"></a>

<!-- combo-section:analysis213:start -->
## 当前问题与优先级

以下 16 个源方法仍保留最近失败报告，17 个参数实例；本轮没有重跑这些未修改的方法。
失败检查涵盖统计、追踪、后端专用调优协议和 eager 支持范围，不能把它们都算成计算错误。

| 优先级 | 问题组 | 方法数 | 当前判断 |
| --- | --- | ---: | --- |
| P1 | `optimize_mem` 图阶段语义 | 2 | Combo 未显式传入字段，runtime 默认 True；训练前向应按社区图阶段语义传 False。尚未观察到数值错误 |
| P1 | provenance 追踪 | 1 | `cppCodeToPost` 映射为空，NPU `define_kernel` 未接社区 origins 登记 hook |
| P1 | 真实统计与逐成员统计元数据 | 7 | 字节数/FLOP 聚合 3 个、逐成员元数据 4 个；需区分实际消费者要求，不能填零或空字典冒充统计 |
| P2 | 峰值内存门控 | 1 | 紧阈值与禁用阈值均为 616101888 字节；尚未证明是设计差异，保留严格峰值断言 |
| 能力差异 | GPU 逐成员专用调优协议 | 3 | INNER/OUTER hint、额外大 RBLOCK 候选、按最大子 kernel 优先 coordesc；NPU 共享 tiling 不直接提供这些 GPU 专用协议，不能据字段/日志缺失推定没有 autotune |
| 独立能力 | eager 内存格式 | 1 | `channels_last contiguous` 在进入 compile 前即失败；当前未替换输入布局，避免改变覆盖 |
| 待核实 | FTZ 数值语义 | 1 | 两个参数实例缺少 `disable_ftz` 接入；尚未证明硬件不支持保留非正规数 |
| 合计 | 仍失败 | 16 | 均不属于已 PASS 的 60 个重合方法 |

### 已验证的 NPU 测试口径

| 检查内容 | NPU 当前设计与验证要求 | 源码依据 |
| --- | --- | --- |
| persistent reduction size hint | split-axis 尺寸求和，其余轴取最大；不照搬 GPU 的二次幂 hint | `NPUComboKernel.select_combo_heuristics` |
| 普通 kernel 与 Combo grid | 普通 kernel 使用 `GridNpu`；Combo 使用一维 `SequentialComboKernelGrid`，实测启动 Y/Z 均为 1，保留大尺寸输出数值检查 | `jit_line`、`SIMDDispatch`、`SequentialComboKernelGrid` |
| mixed-order 数量 | NPU 关闭混合行列归约；行归约、列归约、消费者及 y/z Combo 共 4 个定义，并检查仅有一个双成员 Combo | `torch_npu/_inductor/config.py` 显式设置 `mix_order_reduction=False` |
| scalar broadcast 数量 | `(x0,r1)` 与 `(r0)` 轴布局使用不同分块接口，生成 2 个普通 kernel；检查数量、两类实际轴布局和数值 | `NPUComboKernel.horizontal_partition` 按 kernel 类型、模式和轴标记细分分区 |
| bucketize 数量 | bucketize 调用 ATen fallback；其余两个逐点成员合为 1 个 Triton Combo。检查 fallback 调用及双成员结构，不能把 Triton 定义数当作全部设备 launch 数 | `lowering_fallback_list.py` 注册 `aten.bucketize.Tensor`；实际 wrapper 保留对应调用 |
| 默认共享分块 | grid 元数据和每个分支都使用同一 runtime block；persistent reduction 的静态 `R1BLOCK_SUB` 仍需生成 | `combo_grid_meta`、`SIMDDispatch._calculate_split_blocks`、`codegen_static_numels_sub_kernel` |
| max-autotune 与 grouping | 检查完整共享配置有多个实际候选、执行计时、选出赢家并保留一个 launcher；检查成员数量及输出数值。NPU 不要求 GPU 逐成员分组日志 | `_triton_config_npu_index_combo`、`ComboTileGenerator`、`NPUCachingAutotuner.autotune_to_one_config` |
| persistent RBLOCK 上限 | 不带 HIP 专用的 `max_persistent_rblock=1024`，NPU 按自身资源模型筛选共享候选 | NPU `jit_line` 与共享 tiling 候选生成 |
| TMA / PDL | CUDA tensor descriptor 与 `tl.extra.cuda.gdc_*` 专用检查在 NPU 明确 SKIP | 用例 SM90 条件与 CUDA 指令断言 |

调优检查只在用例内临时包装 `AsyncCompile.wait` 取得真实对象，并调用原函数；
不替换候选、计时或赢家。关闭缓存并将编译线程设为 1 是为了确保本次确实执行待检查路径，退出即恢复。
NPU 的共享 tiling 通过不代表 GPU 的独立逐成员分块或逐成员 coordesc 已实现。
普通 GPU 分支的原断言保留。

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
| `test_combo_kernel_coordesc_tunes_largest_subkernel_first` | 要求独立 `XBLOCK_i` 的特定坐标下降顺序；NPU 当前搜索完整共享配置 |
| `test_combo_kernel_per_config_subkernel_block_size` | eager 的 channels-last 内存格式即不支持，尚未进入 Combo 编译 |
| `test_combo_triton_meta_has_disable_ftz` | 两个参数实例缺后端 FTZ 选项接入，需要核对真实非正规数行为 |

真实统计的普通 kernel 出口在 NPU `triton.py::codegen_kernel`；Combo `jit_line` 尚无对应聚合出口。
`combo_subkernel_meta` 只包含实际尺寸、轴和 tiling 约束，不能替代 load/store/FLOP 统计。
FTZ 指把非正规浮点数冲刷为零，不能仅凭字段出现宣布支持。
共用卡上的本轮验证不构成性能 benchmark，不能据 kernel 数量直接判断性能升降。

## 复现

使用证据目录中的精确 node ID 清单，在相同测试快照中执行原生 pytest：

```bash
conda activate pta_213_case
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=/path/to/new-cache \
python -m pytest -vs \
  test_combo_kernels.py::ComboKernelTests::test_combo_kernel_scalar_store_broadcast
```

复用执行脚本与失败项 manifest 可在卡 1、3 并行；每个方法输出独立日志与 JUnit，
完成后合并已有结果。安装源码前后哈希一致不代表全部历史结果已经重跑。

<!-- combo-section:analysis213:end -->
