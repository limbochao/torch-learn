---
title: UB overflow retry：怎样选最小 tiling，怎样决定缩小哪根轴
published: false
---

# UB overflow retry：怎样选最小 tiling，怎样决定缩小哪根轴

读这段实现，先记住两条规则：

1. **找最小 tiling：把一个 config 中各 tiling block 的长度相乘，选乘积最小的原配置。**
   单轴比较 XBLOCK，多轴比较 X0×X1 等乘积；它选的是继续搜索的起点。
2. **选缩小轴：先排除不能改的轴，再按 metadata 中的顺序，缩小第一个仍大于 1 的轴。**
   第一根可变轴缩到 1 后才动下一根；它没有分析哪根轴导致 UB overflow。

因此，“找到最小 tiling”是从已有候选中选起点；“锁定应该缩小的轴”在当前实现里，
实际是确定哪些轴允许缩小、按什么顺序缩小。**当前代码不能定位最占 UB 的轴，也不能保证这个顺序最合适。**
编译报错中的所需/可用 UB 只用于估算要缩小到什么体积，不用于选择轴。

| 问题 | 当前实现怎样决定 |
| --- | --- |
| 最小 tiling 是哪个？ | 将每个原始 config 中参与 tiling 的 block 长度相乘，取乘积最小的配置；乘积相同，优先 multibuffer=False |
| 可以缩小哪些轴？ | 从 metadata 的 tiling_axis 出发，只保留 DSL 对应的 constexpr block，排除 no-loop 轴和 persistent 模式下的 R 轴 |
| 当前这一步缩小哪个轴？ | 按 tiling_axis 的原顺序，找第一个长度大于 1 的可变轴；先把它逐步缩到 1，再动后面的轴 |
| UB 报错起什么作用？ | 用所需/可用 UB 的比例估算目标体积，决定下降到哪里开始追加；不从报错中识别哪根轴最耗 UB |

本文对应 [`448cd7439d`][commit]、[PR !48470][pr] 和 [Issue #5625][issue]。
生成函数见 [`runtime/tile_generator.py`][generator]，调用方见
[`runtime/triton_heuristics.py`][heuristics]，路径相对于 `torch_npu/_inductor/`。
历史初始候选搜索方案见[第 05 项详解](model-fix-05-tiling-resource-20261022.md)。

## 1. 如何找到最小的 tiling

### 从第一轮已经生成的 configs 中选，先不创造新配置

`configs` 就是第一轮编译使用的原始候选。函数先确认每个候选能对应到有效的 UB overflow 报告，
且参与计算的 block 长度都是正整数，再在这些候选中选一个起点，代码中叫 `seed`。

这里的“最小”指已有候选中的最小体积，不是试编译得出的最小可运行配置，也不是直接把所有轴设成 1。
完整的选择过程如下：

```text
第一轮原始 configs
    → 保留能匹配 UB 报告、block 长度有效的配置
    → 为每个配置计算 tile 体积
    → 选体积最小的一项
    → 体积相同则优先 multibuffer=False
    → 得到一个完整 config，作为继续搜索的起点 seed
```

### 单轴：直接比较 XBLOCK

NFNet 原始 20 个 config 包含 4 种 XBLOCK。一个 XBLOCK 可以搭配多个编译选项，
所以 config 数量大于 tile 大小的种类数。

| 原始 XBLOCK | 用于比较的体积 | 是否最小 |
| ---: | ---: | --- |
| 32768 | 32768 | 否 |
| 16384 | 16384 | 否 |
| 8192 | 8192 | 否 |
| 4096 | 4096 | 是，从这种大小的原配置继续搜索 |

有多个 XBLOCK=4096 的配置时，再按 multibuffer 排序，优先 False。
其他条件也相同，则保留原列表顺序。最终选中的是带有编译模式等全部选项的一个 config。

### 多轴：比较各 block 的乘积，不能逐轴拼一个新最小值

假设一个 kernel 有两个可变轴，下表是为解释规则构造的候选：

| config | X0BLOCK_SUB | X1BLOCK_SUB | 体积 X0×X1 | multibuffer | 选择结果 |
| --- | ---: | ---: | ---: | --- | --- |
| A | 32 | 64 | 2048 | False | 体积更大 |
| B | 16 | 64 | 1024 | True | 最小体积，但 tie-break 排在 C 后 |
| C | 32 | 32 | 1024 | False | 选 C 作为起点 |

C 的第一根轴 32 比 B 的 16 大，但总体积相同，因此按照 multibuffer 选择 C。
这里不会把 B 的 X0=16 与 C 的 X1=32 拼成 `(16, 32)`；那是一个尚未生成的新配置。
也不会因为 B 的某一根轴更小，就直接认定 B 是最小 tiling。

对应代码为：

```python
# block_names 是 DSL 中参与 tiling 的 constexpr block 名称。
def volume(cfg):
    return functools.reduce(lambda a, b: a * b, (
        cfg.kwargs[name] for _, name in block_names
    ), 1)

seed = min(seeds, key=lambda cfg: (
    volume(cfg),
    bool(cfg.kwargs.get("multibuffer", False)),
))
```

没有设置 multibuffer 时，排序值也按 False 处理；后端实际默认行为仍由编译器决定。
选择 seed 不按实际 UB 需求排序，也不按性能排序。固定的 R 轴如果属于 block_names，同样参与体积乘积。

### seed 和 minimum_volume 分别有什么用

| 名称 | 从哪里取得 | 后续用途 |
| --- | --- | --- |
| seed | 能匹配有效 UB 报告的原始配置中选最小的一项 | 复制整个 config，继承它的编译模式、multibuffer 等选项，然后缩小允许修改的 block |
| minimum_volume | 所有 block 长度有效的原始配置中的最小体积 | 规定新增配置必须严格小于第一轮原始最小体积 |

正常整轮 UB 失败且配置能匹配时，两者通常对应相同的最小体积。
代码仍将它们分开：seed 需要有容量反馈，minimum_volume 用来限制新配置不能重复搜索原有大小区间。

## 2. 如何锁定应该缩小的轴

### 先识别轴，再筛选允许修改的 block

轴信息来自 codegen 写入 generated DSL 的 metadata。
[`create_inductor_meta()`][codegen] 将 tiling 轴的 sorted_order、sorted_axis 的名字和 no-loop 标记分别写成
`tiling_axis`、`axis_names`、`no_loop_axis`；no-linear 路径也会写入相应轴信息及 remap 标记。
retry 读取这些结果，不重新从 tensor shape、load 数量或 UB 报错推导轴。

以一个 X 轴和一个 R 轴的 kernel 为例：

```python
axis_names = ["x0", "r1"]
tiling_axis = [0, 1]
constexpr_names = {"X0BLOCK_SUB", "R1BLOCK_SUB"}
no_loop_axis = []
persistent_reduction = True
```

这些字段依次回答四个问题：

| 判断 | x0 轴 | r1 轴 |
| --- | --- | --- |
| 是否在 tiling_axis 中？ | 是，编号 0 | 是，编号 1 |
| 找哪个编译期 block？ | X0BLOCK_SUB | R1BLOCK_SUB |
| 对应参数是否声明为 constexpr？ | 是，可以参与候选体积计算 | 是，可以参与候选体积计算 |
| 是否属于 no-loop 或受保护的 persistent R？ | 否，允许缩小 | 是，persistent R 保持不变 |
| 最终结果 | 进入 shrink_names | 不进入 shrink_names |

因此最终得到：

```python
block_names = [(0, "X0BLOCK_SUB"), (1, "R1BLOCK_SUB")]
shrink_names = ["X0BLOCK_SUB"]
```

block_names 是“计算体积时包含哪些轴”，shrink_names 是“实际可以改哪些参数”。
上例的体积仍是 X×R，但 retry 只减小 X，R 始终不变。

参数名的拼法由 requires_no_linear_block_remap 决定：为 True 时使用 `<AXIS>BLOCK`，
否则使用 `<AXIS>BLOCK_SUB`。所以 NFNet 示例查找 XBLOCK，普通多轴例子查找 X0BLOCK_SUB。
没有对应 constexpr 参数的轴，例如直接写死在 DSL 中的大小，不会被改写。
运行时 X0BLOCK 与编译期 X0BLOCK_SUB 也要区分：前者是运行时参数时，不在对应 constexpr 候选中。

### no-loop 与 persistent R 为什么要保留

`no_loop_axis` 是已有 codegen 给出的限制，表示该轴按当前生成方式不应继续切成更多循环分块。
如果上例将轴 0 也放入 no_loop_axis，则 X 也被保护，最终没有可缩小轴，函数直接返回空列表。

persistent 模式的 R 轴按当前规则保持完整归约块。要切小 R，需要生成的归约循环和累加逻辑也能覆盖剩余元素；
本次 retry 只改 config，不重新生成 DSL，因此保护这类归约轴。当前调用方对 PERSISTENT_REDUCTION 和 TEMPLATE
两种 heuristic 都启用这一保护，规则是过滤名字以 `r` 开头的轴。

实际筛选代码是：

```python
shrink_names = [name for axis, name in block_names
                if axis not in fixed_axes
                and not (persistent_reduction and axis_names[axis].startswith("r"))]
```

### 多根轴都能缩小时，不分析谁更耗 UB，按原顺序逐根缩小

假设没有固定轴、没有 persistent R 约束，得到：

```python
shrink_names = ["X0BLOCK_SUB", "X1BLOCK_SUB"]
```

起点为 `(32, 32)` 时，下降轨迹是：

```text
(32, 32) → (16, 32) → (8, 32) → (4, 32) → (2, 32) → (1, 32)
                                                               ↓
                                  (1, 4) ← (1, 8) ← (1, 16)
```

每一步只改一个参数；每次都从 shrink_names 开头找第一个大于 1 的轴。
因此先把 X0 缩到 1，才会开始缩小 X1。若 tiling_axis 给出的顺序相反，优先级也会相反。
函数没有采用“先缩最长轴”“X/R 交替缩小”或“找 UB 贡献最大的轴”等策略。

可以用下表直接判断同一个 `(32, 32)` 起点会先缩哪根轴：

| metadata 条件 | 得到的 shrink_names | 第一次下降 |
| --- | --- | --- |
| tiling_axis=[0,1]，两根 X 轴都允许改 | [X0BLOCK_SUB, X1BLOCK_SUB] | (32,32) → (16,32) |
| tiling_axis=[1,0]，两根 X 轴都允许改 | [X1BLOCK_SUB, X0BLOCK_SUB] | (32,32) → (32,16) |
| tiling_axis=[0,1]，但 no_loop_axis=[0] | [X1BLOCK_SUB] | (32,32) → (32,16) |
| 第二根是 R1，且 persistent_reduction=True | [X0BLOCK_SUB] | (32,32) → (16,32)，R1 始终不变 |
| 两根轴都受保护 | [] | 不追加配置 |

表中均假定对应 block 是 constexpr；前两行也说明：即使 shape 和 UB 报错完全相同，
只要 metadata 的轴顺序不同，下降路径就可能不同。UB 容量数字本身不能告诉 retry 应该选 X0 还是 X1。

对 2 的幂大小，每步减半；其他正整数先降到严格小于当前值的最大 2 的幂，例如 3000→2048、3→2。
对应代码为：

```python
for name in shrink_names:
    value = candidate.kwargs[name]
    if value > 1:
        candidate.kwargs[name] = 1 << ((value - 1).bit_length() - 1)
        return True
return False
```

这条轨迹只规定下一步怎么走。哪些位置真正返回为新 config，还要看下一节的 UB 目标和数量限制。

### 将“选最小”和“选轴”串成一个双轴例子

继续使用上面的 A、B、C 三个原配置，假设两根 X 轴都可变，tiling_axis=[0,1]。
以下容量数据为构造示例，用来逐步展示函数的决定，不代表实测 kernel：

| 步骤 | 依据 | 本例结果 |
| --- | --- | --- |
| 选原始最小配置 | A 体积 2048，B/C 均为 1024；C 的 multibuffer=False | seed=C，即 (32,32) |
| 找可变轴及顺序 | 两个 block 都是 constexpr；没有固定轴；顺序为 [0,1] | shrink_names=[X0BLOCK_SUB,X1BLOCK_SUB] |
| 估算目标体积 | C 需要 256 KiB，可用 160 KiB；1024×160/256×0.8=512 | target_volume=512 |
| 下降到目标 | X0 排在前面且 32>1，所以只缩 X0 | (32,32) → (16,32)，体积降到 512 |
| 收集最多 4 项 | 每次仍从第一个可变轴开始检查 | (16,32)、(8,32)、(4,32)、(2,32) |

这次最多收集 4 项就停止，X0 尚未降到 1，所以 X1 一次也没改。
如果这个 kernel 必须缩小 X1 才能避开 UB overflow，这条下降路径可能仍然全部失败。
“缩小了总 tile 体积”和“找到了有效配置”之间，仍需真实编译验证。

## 3. 两个选择完成后，怎样得到实际追加的 config

NFNet 的实际情况可以把整个过程串起来：

```text
原始 XBLOCK：32768、16384、8192、4096
    ↓ 比较体积
seed：XBLOCK=4096，multibuffer=False
    ↓ 筛选可缩小轴
shrink_names：[XBLOCK]，只有一个可改的 block
    ↓ 用编译器报错估算目标
需要 256 KiB、可用 216 KiB，目标体积为 2764
    ↓ 沿 XBLOCK 下降，找到不超过目标的第一个位置
4096 → 2048
    ↓ 从此位置继续收集，最多 4 项
2048、1024、512、256
```

容量公式为：

```python
target_volume = min(
    minimum_volume - 1,
    volume(seed) * available * 4 // (required * 5),
)
```

本例计算为 `min(4095, floor(4096 × 216 / 256 × 0.8)) = 2764`。
它假设 UB 需求与 tile 体积近似成比例，保留 20% 可用容量余量。这里是粗略估算，编译器仍可能拒绝结果。

函数先在内存中下降到目标，再从那里收集新配置：

| 阶段 | 做什么 | 是否产生编译尝试 |
| --- | --- | --- |
| 找起点 | 比较原有 configs，选 seed | 否 |
| 找可变轴 | 从 metadata 和 constexpr 筛选 shrink_names | 否 |
| 定位追加位置 | 沿下降轨迹走到不超过 target_volume 的位置 | 否，中间位置不编译 |
| 收集候选 | 从该位置起最多检查 4 个位置，保留比原始最小体积更小、且 key 不重复的项 | 返回列表，交给外层统一编译 |

例如目标若是 600，会先经过 `4096→2048→1024→512`，再返回 `512、256、128、64`。
2048 和 1024 只在内存中经过，不会作为新增候选编译；这是规则示意，并非本例 NFNet 的实际报错。

当 R 固定为 512、X 从 32 开始时，体积用 X×512 计算，但仍只沿 X 缩小。
若估算目标为 6553，会经过 `(32,512)→(16,512)→(8,512)`，再返回 X 为 8、4、2、1 的四项。
即使固定轴使最小可达体积高于估算目标，只要它仍小于原始最小体积，也允许交给编译器尝试。

## 4. 入参、匹配和停止条件补充

```python
def ub_overflow_retry_configs(
    configs, failures, inductor_meta, constexpr_names, runtime_block_names=(),
    persistent_reduction=False, max_configs=4,
):
```

| 入参 | 含义 |
| --- | --- |
| configs | 第一轮原始 Config 列表，调用方先去重；包含各 block 的实际值与其他编译选项 |
| failures | `(失败配置, 所需 UB bit 数, 可用 UB bit 数)` 的列表，来自已完成的原始和 VF 编译尝试 |
| inductor_meta | 包含 axis_names、tiling_axis、no_loop_axis 和 requires_no_linear_block_remap 的轴信息 |
| constexpr_names | DSL 编译期参数名集合，用来确认哪些 block 大小由 config 控制 |
| runtime_block_names | 运行时 block 参数名；匹配 UB 报告和去重时忽略其值 |
| persistent_reduction | 是否保护 R 轴，默认 False；调用方对 PERSISTENT_REDUCTION、TEMPLATE 传 True |
| max_configs | 最多检查的追加候选位置数，默认 4；过滤或走到底时可能返回更少 |

返回值是新的 Config 列表。函数深拷贝 seed，只改允许缩小的 block，保持其余编译选项，
不修改原始 configs，也不为每个新 tile 再枚举不同 mode、warps 和 multibuffer 组合。

原 config 与实际编译变体可能带有不同的运行时 block 字段，因此用 key 匹配反馈。
key 忽略 runtime_block_names 和 split_blocks，保留其他 kwargs、warps/stages 等选项，
并单独保留 VF 开关的布尔值。原始 VF=False 与重试 VF=True 的报告不会混用。
同 key 有多份有效报告时，使用 required/available 比例较小的一份。

没有轴信息、没有对应 constexpr block、没有有效 UB 反馈或没有可变轴，均返回空列表。
收集阶段要求新体积严格小于原始最小值，并按上述 key 去重。
达到最多 4 个检查位置，或所有可变轴都已降到 1，就结束；被过滤的位置不会额外补齐。

整轮原始编译及已有 VF 重试全部结束，且所有失败都匹配有效 UB overflow 容量报告，外层才允许调用。
仅普通 NPUCachingAutotuner 支持本次追加，costmodel 等子类、自定义和 grouped 场景不追加。
新候选只编译一轮，至少一项成功后才更新 configs、candidate plan 和编译结果；失败则保留原状态，
重新抛出第一轮错误。单配置编译函数与基线一致。

已有 VF 重试会复制 config 并设置 enable_vf_fusion=True。NFNet 重放环境中该选项已弃用并被忽略，
验证摘要中的“VF 20 项”表示 Inductor 层尝试次数。新批次传 allow_vf_retry=False，只禁止再补 VF 编译，
不会改写 config 中的 VF 开关。

## 5. 可直接执行的候选生成示例


在安装了本文提交对应 torch_npu 的环境中运行以下代码。它调用真实候选生成函数，
使用构造的容量反馈演示 NFNet 数值，不加载模型，也不运行设备 kernel。

```bash
python - <<'PY'
from torch._inductor.runtime.triton_heuristics import Config
from torch_npu._inductor.runtime.tile_generator import ub_overflow_retry_configs

configs = [
    Config({
        "XBLOCK": size,
        "compile_mode": "simd_simt_template",
        "multibuffer": False,
    }, num_warps=1)
    for size in (32768, 16384, 8192, 4096)
]
failures = [(cfg, cfg.kwargs["XBLOCK"] * 512, 1769472) for cfg in configs]
meta = {
    "axis_names": ["x"],
    "tiling_axis": [0],
    "requires_no_linear_block_remap": True,
}
extra = ub_overflow_retry_configs(configs, failures, meta, {"XBLOCK"})
print("original:", [cfg.kwargs["XBLOCK"] for cfg in configs])
print("retry:", [cfg.kwargs["XBLOCK"] for cfg in extra])
assert [cfg.kwargs["XBLOCK"] for cfg in extra] == [2048, 1024, 512, 256]
PY
```

预期输出：

```text
original: [32768, 16384, 8192, 4096]
retry: [2048, 1024, 512, 256]
```

## 6. 已有验证、搜索范围与局限


去敏后的数量、容量和版本记录见[验证摘要][validation]。

| 验证 | 实际代码版本 | 结果和边界 |
| --- | --- | --- |
| NFNet 原始 kernel 的编译重放，2026-10-24，Ascend A5 | 历史提交 `66ac527207` | 原始 20 项和 VF 20 项全部 UB overflow；追加 4 项全部编译成功；共 44 次编译尝试，最终候选数 24；未 launch kernel |
| UB retry 单测，2026-10-24，PyTorch/torch_npu 2.13 环境 | 当前提交 `448cd7439d` | 18 项通过，覆盖候选生成、串并行、整批结果边界、固定轴、混合错误和 retry 失败保留原异常 |
| precompile 无设备初始化邻近回归，同日、同环境 | 当前提交 `448cd7439d` | 2 项通过 |

NFNet 编译重放早于最终的“保留第一轮错误”调整；两个版本的 `tile_generator.py` 文件内容完全相同，
Git blob 均为 `bb4476e5d6082d8ba6eb764ce5746d6dc9e4f6ce`。
候选生成数值与该次重放一致，但不将历史重放写成当前最终提交的整模型验证。
本次文档整理未重跑设备测试；文中的一维和固定 R 示例另经 CPU 执行真实 helper 的函数体核对，
使用轻量配置对象，不代表真实编译或设备精度检查。
双轴示例也用同一 helper 函数体核对了 5 种情况：正常轴顺序、反向轴顺序、固定 X0、保护 persistent R1、
全部轴受保护；返回配置均与表格一致，原始 config 保持不变。

原始轮次没有全部 UB 失败时，新搜索开销为零；触发时最多额外编译 4 个新候选。
定位目标阶段可能经过多个更大 tile，但只是整数计算，不增加编译次数。
本次不扩大正常路径的初始候选列表，也不追加所有 compile mode、multibuffer 和多轴组合。

这个限制保留了明确的适用边界：

- 单条下降路径可能漏掉其他轴组合里的可行配置，找到的配置也未必是性能最优。
- 体积与 UB 需求近似成比例是启发式；20% 余量不是容量证明。
- 只保留一个 seed 的其他选项，可能漏掉其他选项搭配下才可编译的配置。
- 固定轴或固定 buffer 已经占满 UB 时，缩小允许修改的轴也可能无效。
- 重试配置需要真实编译和后续 autotune，编译成功本身不证明整模型精度或性能。

[commit]: https://gitcode.com/luqichao/pytorch_new/commit/448cd7439d90e47f3a7418468838d83340b263a8
[pr]: https://gitcode.com/Ascend/pytorch/merge_requests/48470
[issue]: https://gitcode.com/Ascend/pytorch/issues/5625
[generator]: https://gitcode.com/luqichao/pytorch_new/blob/448cd7439d90e47f3a7418468838d83340b263a8/torch_npu/_inductor/runtime/tile_generator.py
[heuristics]: https://gitcode.com/luqichao/pytorch_new/blob/448cd7439d90e47f3a7418468838d83340b263a8/torch_npu/_inductor/runtime/triton_heuristics.py
[codegen]: https://gitcode.com/luqichao/pytorch_new/blob/448cd7439d90e47f3a7418468838d83340b263a8/torch_npu/_inductor/codegen/triton.py
[validation]: ../../../artifacts/cicd/ub_overflow_retry_20261024/validation_summary.json
