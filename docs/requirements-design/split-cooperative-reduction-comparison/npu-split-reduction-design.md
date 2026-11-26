---
title: NPU Split Reduction 当前实现方案
---

# NPU Split Reduction 当前实现方案

更新日期：2026-11-26。状态：NPU 选择器与标定准入接口已实现，性能阈值待实测验收。

本方案让 NPU 自己决定普通 reduction 的分片数 `S`，继续复用上游的 partial/combine IR 构造。
当前默认 `auto` 没有标定策略时返回 `S=1`；固定候选公式已经可以开发和测试，
但不能将公式实现完成视为默认开启 split 的性能依据。

本文记录 `pytorch_new` 的 `feat/npu-num-splits` 工作分支，基于开发时的 master
`b91c0f0d065ac8b4698ff4802358939945f7660f`；不表示这些改动已合入上游。
上游源码分析参照 PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`。

相关入口：

- [需求规格草案](npu-split-reduction-requirements.md)：目标能力与验收要求，范围不等于当前实现范围。
- [数据测试脚本说明](../../../scripts/tests/npu_split_calibration/README.md)：测试场景、计时、补点、holdout、百分比采集及离线合并。
- [原始适配 PR #50315](https://gitcode.com/Ascend/pytorch/pull/50315/diffs)。
- [既有 split/cooperative 原型对照](results.md)：问题背景，不作为本选择器的实测阈值。

## 1. NPU 接管的部分

上游 `Reduction.num_splits()` 同时选择 reduction hint 与分片数。
GPU 的 `InductorChoices.reduction_split_factor()` 使用 SM 数、warp 大小、每 SM 线程数、
每线程工作量及 GPU 架构相关阈值；这些假设不能直接解释 NPU Vector 核的并行度和开销。

NPU 因此接管合法性检查、固定候选 `S` 和收益准入，保留上游分层 reduction 的数据流：

```text
Reduction.create
  └─ NPU num_splits → (ReductionHint, S)
       ├─ S=1 → 原有普通 reduction
       └─ S>1 → 上游 create_multilayer / create_multilayer_helper
                   ├─ partial：每个输出拆成 S 份
                   └─ combine：合并 partial，恢复原输出 shape/dtype
                         ↓
                 两阶段分别进入 NPU tiling/codegen
```

上游已经负责分片索引、非整除尾部 mask、归约单位元、中间张量与 combine。
这里不维护第二套 partial/combine kernel，也不让 autotune 搜索多个 `S`。
普通 tiling autotune 仍会为既定的两个阶段选择各自的局部配置。

目录组织与上游保持一致，使用现有模块，不增加独立的 `split_reduction.py`：

| 职责 | 上游 PyTorch | NPU 实现 |
| --- | --- | --- |
| 选择入口、hint 与合法性 | `torch/_inductor/ir.py: Reduction.num_splits` | `torch_npu/_inductor/ir.py: num_splits`，增加 NPU 标定准入 |
| 候选分片数 | `torch/_inductor/choices.py: InductorChoices.reduction_split_factor` | `torch_npu/_inductor/choices.py: reduction_split_factor` |
| partial/combine IR | `torch/_inductor/ir.py: Reduction.create_multilayer*` | 直接复用上游 |
| 模式和策略配置 | 上游 `split_reductions` 总开关 | `torch_npu/_inductor/compile_options.py` 与 `_compat/inductor.py` 注册和兼容 |
| 缓存中的设备资源约束 | FX/AOT 图缓存 | NPU `ir.py` 补充 selector version 与 `NPU_DEVICE_LIMIT` |

NPU 入口对非 NPU 设备调用保存的上游 selector。接管范围是默认 NPU Triton backend，
实验后端及具有 `REDUCE_TO_SINGLE_ELEMENT` 特性的路径不通过本策略拆分。

## 2. 固定候选公式

定义 `X` 为输出元素数，`R` 为每个输出的归约元素数，`C` 为所请求设备的有效 Vector 核数。
`C` 通过 NPU 设备属性查询并遵守现有 `NPU_DEVICE_LIMIT`，不使用 GPU SM/warp/thread 属性。

当前候选版本为 `npu-fill-v2`：

```text
S_fill = ceil(C / X)
S = max(1, min(S_fill, C, R - 1, floor(R / 256)))
```

`ceil(C/X)` 估计在每个 program 处理一个输出时，补足一 wave 并行度需要多少分片。
`floor(R/256)` 避免把工作拆得过细。**256 是候选最小工作量假设，不是收益阈值，也不是 UB 容量限制。**
真实 program 数和 R tile 由后续 tiling 决定，不能从 `X*S` 推断实际 grid。
当 `X>=C` 或 `R` 太短时，公式自然返回 1。

这个公式只给出一个候选；准入表回答“该候选在哪些范围值得启用”。
要修改每 program 输出数、wave 数、最小分片工作量或公式，须升级 selector 版本并重新标定，
不能沿用旧公式测出的准入区间。

上游递归构造保持原样。若 `S` 受填核数限制，partial 的输出数量 `X*S` 已足够；
若受最小工作量限制，partial 的归约长度 `ceil(R/S)` 不足以再拆成两个合法分片。
因此生产实现不需要修改上游递归，也不需要通过全局 depth 状态抑制嵌套拆分。

## 3. 合法性和当前启用范围

| 维度 | 当前行为 |
| --- | --- |
| 普通 IR reduction | `sum/prod/min/max/any/xor_sum` |
| sum/prod/min/max dtype | 输入与输出 IR dtype 均限定 FP16、BF16、FP32、int32、int64 |
| any dtype | 输入和输出均为 bool |
| xor_sum dtype | 输入和输出均为 int32/int64 |
| 特殊路径 | arg reduction、Welford、online-softmax、scan、dot/fmax 返回 `S=1` |
| shape | 只接受实际静态维度；动态 X/R 返回 `S=1`；空输出或 `R<2` 返回 `S=1` |
| 读取索引 | 必须可证明为非负常量 stride 的 affine 表达式；未知、间接、负 stride 或纯 broadcast 归约回退 |
| auto 上下文 | FX 图恰有一个受支持的独立归约，并且只含一个零偏移的连续读取 |
| force 上下文 | 可测试通过合法性检查的更多布局和融合；仍不绕过 dtype、shape、读取索引及后端检查 |

这里声明的是 selector 能力。某个 ATen overload 若仍在安装后端的 fallback 表中，
不会因为 selector 支持对应 IR 类型就自动恢复 lowering。本改动不调整产品 fallback 表。
mean 是 sum 加缩放，all 可能分解为取反和 any；不能由 API 名称推断它们满足当前独立归约的 auto 条件。

`num_splits` 发生在最终 scheduler fusion 之前。独立 sum 的收益不能直接推断到
RMSNorm、softmax、归约前逐元素计算、多个归约或未知 consumer。
所以 auto 暂时要求上述严格图上下文，匹配 X/R 仍不足以放行这些扩展场景。
后续扩展须同时增加可识别的上下文 guard 和对应完整 forward 的性能验收。

## 4. 动态 shape：上游行为与 NPU 选择

当前上游先检查 unbacked symbol 是否具有显式 hint，缺少 hint 时保守返回 `S=1`。
其余情况下用 `optimization_hint(X/R)` 给 GPU 启发式选择一个编译期 `S`。
构造分片时仍使用真实的 shape 表达式，例如 `ceildiv(R,S)`、索引和 mask；
shape 约束、guard 与重编译属于其余编译链路的责任。**optimization hint 本身不是运行时范围 guard。**

这意味着代表尺寸上选出的 `S` 可能在同一动态图的其他运行尺寸上仍合法，但性能并不相同。
上游这种选择机制也不意味着会 autotune 不同的 `S`。

NPU 首版同时对 backed 和 unbacked symbolic X/R 返回 `S=1`，
不使用代表尺寸命中静态收益区间，不为这项优化额外特化动态 shape。
后续若支持动态拆分，需要真实的范围 guard/dispatch、该范围的性能证据以及缓存/重编译契约。
标定套件的动态诊断用于观察复用、重编译和不同 runtime shape 的行为，不会扩大静态准入域。

## 5. partial 精度复用上游

在本文参照的上游版本中，`Reduction.create_multilayer_helper()` 按 **dst_dtype**
确定中间存储类型：dst_dtype 为 FP16/BF16 时 intermediate 使用 FP32，其余使用 dst_dtype。
这样跨 kernel 保存 partial 时，不因写回低精度而额外引入一次舍入。
最终 combine 仍以原始 dst_dtype 构造输出。

`config.triton.codegen_upcast_to_fp32` 影响 kernel 内的代码生成精度，
与这里的跨阶段 intermediate 存储规则不是同一层控制；该 helper 的存储选择没有读取这个开关。
NPU selector 不复制该 dtype 推导，也不接管 partial 分配。
精度相关配置纳入策略环境指纹，避免用不同精度设置下的数据选择阈值。

测试 hook 中的 `partial_dtype_expected` 只是基于这条上游规则的观测注释，
不改变实际 intermediate 类型。实际 dtype、分配与计算类型仍需从生成 IR/code 核对。

## 6. 模式、准入表与缓存

| 配置 | 语义 |
| --- | --- |
| 上游 `split_reductions=False` | 总开关优先，NPU 不拆分 |
| `split_reduction_mode="off"` | 返回 `S=1` |
| `split_reduction_mode="auto"` | 默认；合法、图上下文受支持、环境一致且命中标定区间才拆分 |
| `split_reduction_mode="force"` | 使用固定候选，绕过收益准入；其余合法性检查仍执行 |
| `split_reduction_policy=None` | 默认没有标定策略，auto 返回 `S=1` |

force 不保证 `S>1`，也不提供性能保证。该模式用于验证能力和收集候选收益。
配置走 NPU 既有 compile options；新版本使用 `npu.<key>`，旧版本使用裸 key，
可通过 `torch_npu._compat.inductor.compat_compile_option()` 得到当前版本对应名称。

产品策略结构如下，尖括号处须由对应环境的实测验收填入，不能作为可运行阈值示例：

```text
schema: "npu-split-policy-v1"
status: "VALIDATED_MEASURED_DOMAIN"
selector_version: "npu-fill-v2"
target: <与 _policy_target(device_name, vector_cores) 完全一致的环境快照>
rules:
  - key: {op: <IR type>, src_dtype: <dtype>, dst_dtype: <dtype>, X: <精确 X>}
    r_min: <通过标定和独立 holdout 的下界>
    r_max: <通过标定和独立 holdout 的上界>
```

规则使用精确的 op/src-dtype/dst-dtype/X 和两端有界的闭区间 R。
不由某个“首次获益 R”推断所有更长 R 都获益，也不向邻近 X、其他 dtype 或设备外推。
图上下文与连续读取由 selector 另行检查，不由表中的 R 区间代替。

`target` 包含设备型号、有效 Vector 核数、torch/git/torch_npu/Triton/CANN 版本、
精度与 tiling 相关配置，以及关键编译源码摘要；摘要同时覆盖 NPU `ir.py` 和 `choices.py`。
策略 schema 或规则格式错误会显式报错；状态、版本、环境或区间不匹配时不准入。

mode 和 policy 通过 compile config 参与图缓存与编译进程传递，
selector version 与 `NPU_DEVICE_LIMIT` 另加入 FX/AOT 持久化图缓存键。
策略变化应以新 compile 配置生成新的 callable；不要在已编译 callable 上修改环境后期待重新选 S。

## 7. 数据如何决定启用范围

标定主指标限定为 profiler 中完整 forward 的 device kernel duration 总和，包含 partial 与 combine；
直接执行并解析 kernel_details.csv，不使用 NPU Graph、Event 区间或 host 等待时间决定阈值。
旧 Event 采集结果无效，不能续用于性能准入。

完整的测试矩阵、模块分工、执行方式、产物字段与多机合并流程统一维护在
[标定脚本 README](../../../scripts/tests/npu_split_calibration/README.md)。

默认单阶段入口先做 quick 筛查，用较少的静态连续归约点定位收益变化区间；
筛查结果标为 `SCREENING_ONLY`，不生成可用准入规则。profiler 原始数据只保留 kernel_details.csv，
实际 S/launch/tiling 与精度证据保存在结果元数据中；完整验证仍需补齐下面的区间和上下文要求。

数据决策分为候选验证、区间收敛和产品验收三步：

1. 在同一卡、同输入和同测量配置下配对比较 S=1 与固定候选的完整 forward。
   必须先通过精度与真实 split/codegen 证据检查；性能包含 partial、combine、临时读写及融合变化。
2. 根据全部采样结果补测收益交界、S/tiling 切换、已获益区间内部和缺失上下文。
   对每个拟准入区间冻结规则，再使用未参加拟合的 R 与新输入 seed 做逐条 holdout。
   进入 split 的验收点必须全部满足收益裕量；失败或覆盖不全时不生成有效准入结论。
3. 将实验结果映射到当前产品 selector 的严格上下文，在实际产品编译配置、目标设备及独立时段复测。
   通过后才生成产品策略并启用 auto。应同时检查产品原有基线，避免只证明相对实验 plain 的收益。

测试套件输出的 schema 是 `npu-split-table-v3`，产品接口是 `npu-split-policy-v1`。
两者的环境、上下文和合法性检查不同；**目前没有可直接导入的转换工具，不能只改 schema/status 就加载到产品。**
特别是 `ordinary_ir` profile 在 worker 内恢复部分 fallback lowering，
与产品安装默认路径属于不同实验域，不能混用准入结果。

套件允许探索更多布局、融合和动态 hint，也提供抑制嵌套 split 的实验 hook。
这些用于收集证据；产品仍使用前述静态、可证明读取与严格 auto 上下文，且保留上游递归构造。
套件默认参数下的固定 S 与当前产品公式相同，更改套件候选参数后不再对应这版产品策略。

“进入 split 后有收益”通过有限域准入、逐点裕量、独立 holdout 和域外回退来约束。
有限采样无法证明区间内每一个整数 R、任意输入数值和任意运行负载都必然更快。
当前仍需完成实际数据标定与产品路径验收，不能发布无条件性能保证。

## 8. 已完成验证与剩余验收

| 验证 | 已有结果 | 结论边界 |
| --- | --- | --- |
| selector 主机验证 | 目录调整到 `ir.py/choices.py` 后，11 项检查通过，并通过语法与 diff 检查 | 覆盖固定 S、递归终止、类型/shape/stride、准入、分派、配置与缓存；不证明设备收益 |
| NPU 编译执行验证 | 目录调整前的实现，在 NPU_A5 的 `pta_213_case` 环境通过 11 项主机检查及 3 项真实 NPU 编译测试 | 涵盖 FP16/BF16/FP32 partial/combine、尾块、合成 auto 策略和动态回退；通过进程内源码覆盖验证，未全量重编最新 master |
| 采集编排验证 | 百分位切分、分卡、续跑和离线合并的 12 项 CPU/模拟 worker 集成检查通过 | 含缺片、重叠、环境失配、失败保留及合并后续跑；不包含新增 NPU 性能数据 |

设备测试中的 auto 区间是用于验证控制逻辑的合成规则，不是实测性能阈值。
目录调整后的最终源码仍需在目标安装/构建环境做设备复验。
旧 Event 全量采集已停止，性能数据作废；profiler 版本须重新采集并验收，当前没有可上线阈值。

剩余验收包括：完整采集与补点结束、逐规则 holdout、当前产品路径复验、
独立时段/设备稳定性，以及需求规格中尚未覆盖的数值压力、生命周期和模型回归。
原始测量、IR/code、日志及缓存保存在仓库外；本仓库保留方案、脚本和使用说明。
