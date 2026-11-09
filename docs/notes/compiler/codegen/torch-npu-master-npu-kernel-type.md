---
title: torch_npu master 的 npu_kernel_type 选择逻辑
---

# torch_npu master 的 npu_kernel_type 选择逻辑

本文回答两个问题：`npu_kernel_type` 的四个值分别代表什么，以及一个计算图在什么条件下生成这些值。

分析日期：2026-10-28。源码为官方 `Ascend/pytorch` 的 `master`，
通过 `git fetch upstream master` 更新后，固定在提交
[`9377b9df839283ca9afded838d82de985f101e7c`][revision]。
全文源码链接均固定到该提交，未使用本地 feature/bugfix 分支的实现。

## 先看场景对照

`npu_kernel_type` 是 **Inductor 自动生成 Triton kernel 时的 codegen / 候选编译模式分类**。
需要结合生成类、IR 索引和 autotune 配置理解，不能只按 ATen 算子名判断。

下表中“保留”表示后续没有其他赋值覆盖，且该路径成功生成代码。
“A5”指源码中的 `is_ascend950` 条件，即 SoC version `>= 260`；
下文“非 A5”统一指这个条件为假。平台判断见 [config.py:30][soc]。

| 场景 | 生成的 `npu_kernel_type` | 直接依据 |
| --- | --- | --- |
| 非 A5，正常走 `NPUIndexTritonKernel`，没有后续覆盖 | `simd` | 构造函数的默认值。[源码][default] |
| A5，正常走 Index kernel，没有后续覆盖 | `simt_template` | 默认值不区分计算类别或 static/dynamic。[源码][default] |
| A5，indirect/masked 重分类，默认模式，未命中 indirect sum 特判 | `simd_simt_mix` | 分类函数返回默认模式。[源码][indirect] |
| A5，默认模式，命中直接 indirect load 或一层 add/mul 后 sum | `simt_only` | sum 特判。[源码][classifier] |
| A5，indirect/masked 重分类，显式配置 `simt_only` | `simt_only` | 直接返回配置。[源码][classifier] |
| A5，命中 indirect/masked 重分类，显式设置模式为 `simt_template` | `simt_template` | 非 mix 模式直接返回环境配置。[源码][classifier] |
| A5，索引表达式 `can_split_all=False` | 当场写成 `simt_only` | 当前 Index kernel 内赋值并 return。[源码][linearization] |
| 开启 Welford，persistent，`welford_reduce`，归约轴全静态 | 当场写成 `simd` | finalize 阶段覆盖。[源码][welford] |
| Index codegen 异常，改走 no-linear 并成功生成 | `simd_simt_mix` | no-linear 固定写 mix。[源码][combined]、[源码][nolinear] |
| lowering 直接走 extern/fallback | 此部分没有该字段 | 没有自动生成对应 Triton kernel。[源码][fallback] |

表中的默认值只适用于相应 codegen 路径。一个模型可以包含多个 Triton kernel、extern 调用或模板 kernel，
同一模型内出现多个 type 是正常的。

## 四个值与三个不同层次

枚举定义在 [`codegen/triton_utils.py:26`][enum]，这里没有额外的 `dynamic`、`reduction` 或 `fallback` 枚举。

| 枚举 | metadata 字符串 | `compile_mode()` 的返回值 | 在 torch_npu 中的作用 |
| --- | --- | --- | --- |
| `SIMD` | `simd` | `simd` | SIMD codegen/候选路径；正常非 A5 Index kernel 的默认分类，以及特定 Welford 路径的显式选择。 |
| `SIMT_ONLY` | `simt_only` | `simt_only` | 纯 SIMT 编译候选；Index codegen 中也会进入 unified 分支。 |
| `SIMT_TEMPLATE` | `simt_template` | **`simd_simt_template`** | SIMD+SIMT 模板编译模式；A5 Index kernel 的默认分类。 |
| `SIMD_SIMT_MIX` | `simd_simt_mix` | `simd_simt_mix` | 候选集合的分类；正常配置生成流程会先展开为具体模式，再交给编译器。 |

要分清三个层次：

1. **生成期分类**：`inductor_meta['npu_kernel_type']`，决定 codegen 分支和候选模式空间。
2. **某个编译候选的模式**：`Config.kwargs['compile_mode']`，进入 Triton 编译 options。
3. **编译产物的执行属性**：binary metadata 中的 `parallel_mode`、`is_pure_simt`、`mix_mode` 等。

因此，看到 `npu_kernel_type='simd_simt_mix'`，只能说明候选集合按 mix 处理，
还不能确定最终选中哪个模式；看到 `simt_template`，实际传给编译器的字符串则是 `simd_simt_template`。
前者要查 best config，执行属性要继续查 binary metadata。
参数落盘和 launch 判断见 [`runtime/triton_heuristics.py:1239`][binary]。

这里的 SIMD/SIMT 是编译模式命名。仅凭 torch_npu 这一层不能把某个 type 推导成固定的设备指令比例。

术语说明：

- **pointwise / reduction**：描述计算和迭代方式；例如逐输出位置计算，或沿指定轴聚合。
  它们与本字段是不同维度，不能由 `triton_poi_*` / `triton_red_*` 名称推导 type。
- **indirect 索引**：地址的一部分来自 tensor 中读出的值，例如 `weight[index[i]]`。
  transpose/stride 不连续本身不等于 indirect。
- **masked subblock**：LoopBody 中由条件控制执行的子图，与 DSL 中普通的尾块 mask 不是同一个概念。
- **索引拆分/线性化**：分析地址表达式能否拆到 codegen 使用的迭代轴上；本文的 no-linear
  指放弃这套 Index codegen 方案，改用另一个生成类，不表示数学运算没有线性关系。
- **persistent reduction**：由 codegen 的启发式决定的归约方案，通常让一个归约 tile 覆盖完整归约范围；
  不能只根据 Python 中用了 `sum` 或 `var` 判断它是否开启。
- **autotune**：生成并编译多组 tile/编译选项，在可用候选中选择配置。

## 从入口到结果的完整流程

下面按普通自动生成 kernel 的入口画出流程；这是 codegen 路径图，
其中 `simt_only` 是一次字段赋值，no-linear 是一次 backend 回退。

```text
NPUCombinedScheduling.codegen_node
  |
  +-- 尝试 NPUTritonScheduling -> NPUIndexTritonKernel
  |     |
  |     +-- 构造时：A5 -> simt_template；非 A5 -> simd
  |     |
  |     +-- 遍历 node_schedule，transform_dims_in_indexing
  |     |     |
  |     |     +-- 建立 body.indexing（尚未建立时）
  |     |     |     +-- generate_indirect_replacements
  |     |     |           +-- 命中 indirect/masked 节点且配置非空 -> 重分类
  |     |     |
  |     |     +-- 分析索引表达式
  |     |           +-- can_split_all=True -> 继续
  |     |           +-- False，A5 -> type=simt_only，当前 transform 返回
  |     |           +-- False，非 A5 -> 抛 ValueError
  |     |
  |     +-- finalize：特定 Welford 条件成立 -> type=simd
  |     |
  |     +-- 生成 DSL / metadata，并检查 TTIR
  |           +-- 成功 -> metadata 写出当前 type
  |
  +-- 上述 codegen 调用抛异常 -> no-linear fallback
        +-- 重建 group，重新检查融合，必要时拆成多个 kernel
        +-- NPUNoLinearTritonScheduling -> NPUTritonKernel
              +-- metadata 固定写 npu_kernel_type=simd_simt_mix
```

入口与异常捕获见 [`codegen/npu_combined_scheduling.py:165`][combined]；
两个 scheduling 类对应的 kernel 类见 [`codegen/scheduling.py:158`][scheduling]。
Index 构造函数调用 `decide_codegen_dims_in_kernel`，后者先遍历索引再 finalize，
见 [`codegen/triton.py:1217`][decide]、[`triton.py:1478`][transform-schedule]。
Index scheduling 在代码生成后调用 `make_ttir_for_check`，见 [`scheduling.py:272`][ttir-check]。

**type 没有单调优先级或“设过 simt_only 就锁定”的机制。**
源码直接修改同一个 `V.kernel.npu_kernel_type`：多个 LoopBody 的 indirect 分类、
索引拆分失败和后面的 Welford 选择都可能赋值。
因此应检查实际执行顺序和最后写出的 metadata，不能把上述条件当成无顺序的互斥标签。
例如后一个 body 的重分类可以覆盖前一个 body 在拆分失败时写入的值。
写入点见 [indirect 分类][indirect]、[拆分失败][linearization]、[Welford][welford]。

## indirect/masked 如何覆盖默认值

### 配置读取与作用范围

[`config.py:345`][indirect-config] 只在 A5 上读取 `INDUCTOR_INDIRECT_MEMORY_MODE`。
非 A5 的内部值保持为 `None`。

| 环境变量 | A5 内部值 | 命中重分类时的行为 |
| --- | --- | --- |
| 未设置 | `simd_simt_mix` | 先尝试 indirect sum 特判，否则 mix。 |
| `simd_simt_mix` | `simd_simt_mix` | 同上。 |
| `simt_only` | `simt_only` | 直接选择 `SIMT_ONLY`，不执行 mix 的 sum 特判。 |
| `simt_template` | `simt_template` | 直接选择 `SIMT_TEMPLATE`，并启用对应 indirect lowering override。 |
| `fallback` | `None` | 不执行这处重分类，相关算子加入 lowering fallback list。 |
| 其他字符串，包括 `simd` | `simd_simt_mix` | 非法值被重置为默认 mix；不能用该变量强制 SIMD。 |

这不是全图的 kernel type 强制开关。例如设置 `simt_only` 后，
一个未触发重分类、没有其他覆盖的普通 A5 `relu` Index kernel，仍保留默认的 `simt_template`。

配置还会影响更早的 lowering：

- `simt_template` 会把 embedding、gather、index、index_put、scatter 等加入 NPU override 列表，
  见 [`lowering_override_list.py:34`][override]。
- 以 embedding 为例，模板模式还检查 weight 是否有 size=1 的维度、是否为 `BaseView`；
  不符合条件时仍使用社区 embedding lowering，见 [`lowering.py:495`][embedding]。
- `None` 会把包括 cat、embedding、gather、index、scatter 在内的一批算子加入 fallback list，
  见 [`lowering_fallback_list.py:452`][fallback]。

所以这个变量既改变前面的 IR 形成方式，也影响后面的 type 分类。
不能认为切换变量只是在完全相同 DSL 上更换一个编译参数。

### 真正触发分类的是 LoopBody 节点名称

调用链为：

```text
transform_dims_in_indexing
  -> generate_body_indexing（body.indexing is None 时）
     -> generate_indirect_replacements
        -> define_npu_kernel_type
```

`generate_indirect_replacements` 遍历 root graph：

1. 对 `index_select`、`gather_template`、`indexput_template`、`scatter_template` 名称先走专门分支并 `continue`。
2. 其他节点通过 `get_indirect_var(node.name)` 检查名称中是否出现 `indirect` 或 `masked`。
3. 命中且 `inductor_indirect_memory_mode` 非空，才把 `define_npu_kernel_type(self)` 返回值写入 kernel。

见 [`codegen/ir.py:1830`][indirect]、[`triton_utils.py:58`][name-match]。
这里会识别 `masked`，所以 **mix 不能反向证明 kernel 一定包含 tensor 驱动的间接地址**；
某些 masked body 也会触发这处分类。普通 `x < numel` 尾块 mask 本身则不能作为该分支已触发的证据。

### 默认 mix 中的 indirect sum 特判很窄

[`define_npu_kernel_type`][classifier] 没有完整的表达式模式匹配器，主要依赖节点名和固定参数位置：

- 只遍历 `loop_body.root_block.graph`，没有递归扫描所有 subblocks。
- 只处理名称 **恰好为 `reduction`** 的节点，`reduction_1` 不会进入同一判断。
- 读取 `node.args[3]` 为 reduction type，必须等于 `sum`。
- 读取 `node.args[4]` 为被归约值，只向下看直接 load，或者一层名称包含 `mul` / `add` 的节点及其直接 load 输入。
- 对相应 load 的 index key，在 `loop_body.indexing` 中检查字符串是否包含 `indirect`。

命中后返回 `simt_only`，否则返回配置中的 `simd_simt_mix`。
源码注释给出的理由是 indirect load + sum 使用 `simt_only` 更快；本文没有重新测量性能。

以下是 **IR 语义示意**，不是仅凭 Python 表达式就保证得到某个 type 的端到端实测结论：

```text
sum(load(weight, indirect0))
    -> 若节点名和 args 位置满足上述约束：simt_only

sum(mul(load(weight, indirect0), load(scale, x0)))
    -> 若 mul 的直接输入被识别：simt_only

max(load(weight, indirect0))
    -> 不满足 sum 特判：保留 mix

sum(sin(load(weight, indirect0)))
    -> 不属于直接 load 或一层 add/mul：保留 mix
```

融合、分解、cast 或 reduction 命名变化，都可能让数学上相近的计算落到不同分支。
尤其不能把这段代码概括成“所有 gather + reduction 都是 simt_only”。

## 索引拆分失败与 no-linear 是两种回退

### 在 Index kernel 内改 type

[`transform_dims_in_indexing`][linearization] 对索引逐项调用 `analyze_expression`。
当 `analyse_res['can_split_all']` 为假：

- A5：设置 `V.kernel.npu_kernel_type=SIMT_ONLY`，从当前 transform 函数返回。
- 非 A5：抛出 `ValueError`，可进入外层 no-linear fallback。

A5 分支本身没有创建新的 `NPUTritonKernel`，也没有把算子交还 eager；
仍是在 `NPUIndexTritonKernel` 路径内继续执行外层流程。
而且这一判断没有“必须 dynamic shape”的前置条件，静态表达式也可能拆分失败。

### 更换整个 codegen backend

[`NPUCombinedScheduling.codegen_node`][combined] 捕获 Index codegen 调用抛出的 `Exception`，
恢复 buffer 状态，重建 scalar groups，重新检查融合后调用 no-linear backend。
原来一个 fused kernel 在这里可能被拆成多个 kernel。

no-linear 使用 `NPUTritonKernel`，其 metadata **无条件写成 `simd_simt_mix`**，
并附加 `requires_no_linear_block_remap=True`，见 [`triton.py:945`][nolinear]。
因此出现 mix 至少有 indirect/masked 重分类和 no-linear fallback 两种来源。

这里捕获的是 codegen 调用范围内的异常，包括其中 TTIR 检查暴露的错误；
不能推导成“后续任意 Triton 编译失败、launch 失败都自动回到 no-linear”。

## dynamic shape 与 Welford 的特殊情况

### dynamic shape 没有独占一个 type

当前 `NPUIndexTritonKernel.__init__` 仅按平台设置默认 type，未按 static/dynamic 分支赋值。
后续应检查真实索引是否能拆分、是否触发 indirect/masked，以及是否发生外层异常回退。

这份 master 中还保留了：

- `should_skip_linearization_on_a5`：判断动态范围或 `ModularIndexing` / `FloorDiv` 中的符号边界。
- `is_linear_dynamic_reduction`：检查索引，以及 argmax/argmin/Welford 等 reduction。

但对固定提交的全仓检索，以及 `_inductor` Python 源码的 AST 引用扫描中，
**两个函数都只有定义，没有发现引用或调用点**。
实际 `transform_dims_in_indexing` 从基础 indexing 生成直接进入表达式分析，未调用它们。
见 [`ir.py:1464`][unused-dynamic]、[`ir.py:1543`][linearization]。

因此，不能据这些函数的注释写成“dynamic pointwise 必然 simt_only”
或“dynamic argmax 一定按 unsafe reduction 检查回退”。当前生效路径要沿实际调用链判断。

### Welford 可以把 A5 默认模板模式覆盖为 SIMD

[`_select_vectorized_welford_axis`][welford] 在 finalize 阶段依次检查：

1. `npu_config.enable_welford` 开启，且 `self.persistent_reduction=True`。
2. `find_reduction_node()` 找到的 reduction type 为 `welford_reduce`。
3. `reduction_axis_list()` 中每个轴的 length 是 `int` 或 `sympy.Integer`。

通过后立即设置 `self.npu_kernel_type=SIMD`，然后才寻找可向量化的非 reduction 轴。
即使后面找不到候选轴直接返回，这次 `simd` 赋值也已经发生。
要求的是 reduction 轴静态，并非所有输入维度都必须静态。

`TORCHINDUCTOR_ENABLE_WELFORD` 默认是 `0`，见 [`config.py:191`][welford-config]。
仓库的 `test_welford_simd_codegen` 显式打开配置，检查生成代码含
`npu_kernel_type': 'simd'`，见 [`test_var_mean.py:88`][welford-test]。
这是已有测试中的预期，本文未在 NPU 上重新运行该测试。

## type 确实改变 DSL 生成方式

在 Index codegen 中，[`is_unified_simt_kernel`][unified] 返回真的集合为：

```python
{NPUKernelType.SIMT_ONLY, NPUKernelType.SIMD_SIMT_MIX}
```

`SIMT_TEMPLATE` **不在这个集合中**。这也是不能把名字含 `simt` 的 type 当成同一路径的原因。

| 行为 | `simd` / `simt_template` | `simt_only` / `simd_simt_mix` |
| --- | --- | --- |
| 静态 persistent reduction block size | 使用静态长度 | 使用 `next_power_of_2(length)`。 |
| 静态 no-loop 轴 `BLOCK_SUB` | 使用轴长度 | 向上取到 2 的幂。 |
| `triton_meta['configs']` | 此处不补 `config_of(signature)` | 此处补入。 |
| reduction / mask 处理 | 各自分支 | unified 分支；例如部分非末轴 reduction 关闭 persistent。 |

分别见 [`triton.py:4048`][unified]、[`triton.py:2546`][static-block]、
[`triton.py:2471`][unified-meta]、[`triton.py:1538`][finalize]、[`triton.py:4112`][mask]。
上表只描述 `NPUIndexTritonKernel`，no-linear 使用另一个生成类。

所以修改 generated metadata 的 type，可能改变 autotune 却没有重新生成匹配的 DSL；
分析或对比时应同时核对 DSL 和候选模式。

## mix 最终会编译哪些模式

### 普通 autotune：当前只显式生成两类

`FASTAUTOTUNE` 默认是 `0`，见 [`config.py:393`][fasta-config]。
[`_triton_config_npu_index_legacy`][candidates] 在非 FastA 分支遇到 mix 时，依次：

```text
set_kernel_type(SIMT_ONLY)      -> 生成 compile_mode=simt_only 的候选
set_kernel_type(SIMT_TEMPLATE)  -> 生成 compile_mode=simd_simt_template 的候选
```

**这里没有第三次 `set_kernel_type(SIMD)`。**
非 mix 输入则直接按自身 type 生成候选。
`TileGenerator.fill_config` 把 `npu_kernel_type.compile_mode()` 写入配置，
见 [`tile_generator.py:240`][fill-config]。

`SIMT_ONLY` 还会使用不同的 tiling 限制和 warp 搜索：例如 sub-block 的 2 次幂处理、
当前 8 KiB 候选大小过滤，以及 `num_warps=[8,16,32,64]` 扩展。
其他 type 则走 multibuffer 扩展，见 [fill_config/候选过滤][fill-config]、
[`tile_generator.py:498`][extra-options]。这些是候选策略，不应视为设备容量的通用规格。

### FastA：expert 候选显式包含三类

当 `FASTAUTOTUNE=1` 时使用 `FastATileGenerator`。
其 `FASTA_SETTING.use_tile_exp` 默认开启，mix 分支依次生成：

```text
SIMT_ONLY      -> simt_only
SIMT_TEMPLATE  -> simd_simt_template
SIMD           -> simd
```

见 [`fasta_autotune.py:395`][fasta-candidates]。
但 metadata type 不为 `SIMD` 时，FastA autotuner 会设置 `use_origin_autotuner=True`，
使用 expert configs 交回原 autotuner，见 [`fasta_autotune.py:910`][fasta-fallback]。
“FastA 参与候选生成”和“最终使用 FastA 搜索算法”需要区分。

仓库的 [indirect memory 配置文档][upstream-doc] 泛称 mix 支持三种方案。
结合当前实现，应补充上述开关条件：普通分支显式展开两类，FastA expert 分支显式展开三类。
候选还可能被过滤、剪枝或编译拒绝，枚举出某类候选不保证该类最终可执行或胜出。

### 两种默认值及显式 options 的优先级

这里还有两处不同阶段的默认值：

- 候选生成读取 metadata 时，缺失 `npu_kernel_type` 默认按 `simd`，见 [候选生成入口][candidates]。
- `_precompile_config` 读取具体候选时，缺失/空 `compile_mode` 默认按 **`simd_simt_template`**，
  见 [`triton_heuristics.py:1104`][compile-options]。

候选为空时的保底 `Config` 没有写 `compile_mode`，因此也会进入第二个默认值。
这意味着 metadata 为 `simd`，在候选生成失败、走保底配置等边界情况下，
也不能只靠 metadata 断言编译器实际收到 `simd`。

编译 options 先由候选/默认值建立，再应用 `triton_meta['npu_compile_options']`，
最后应用 `cfg.kwargs` 中 Triton `NPUOptions` 支持的选项，见
[`parse_triton_ascend_options` / `_precompile_config`][compile-options]。
有显式选项时，以真正送入 `triton.compile` 的 options 为准。

symbolic grouped autotune 对每个 representative 仍调用 legacy 候选生成器，
没有新增第五种 type；但不同 group 可以选择不同配置。
见 [`triton_config_npu_index`][grouped-entry]、[`_triton_config_npu_index_grouped:3345`][grouped-candidates]。

## 拿到一个 kernel 后怎么判断

按以下顺序查看，通常可以解释“为什么这个 kernel 是这个 type”：

1. **确认实际产物和版本**：从对应进程的 `output_code.py` 读取 metadata，固定 torch_npu 提交。
   先确认目标计算究竟是自动生成 Triton、模板，还是 extern。
2. **查生成类与平台**：Index 类的 A5 默认值为 template，非 A5 默认值为 SIMD；
   `requires_no_linear_block_remap=True` 是这份代码中 no-linear 路径的识别线索。
3. **查配置和 IR**：记录 `INDUCTOR_INDIRECT_MEMORY_MODE`，检查 root graph 是否出现 indirect/masked 名称；
   若是 `simt_only`，继续分辨 sum 特判、显式配置和 `can_split_all=False`。
4. **查覆盖与异常**：检查 Welford 配置、persistent 状态、reduction 轴长度，以及 Index codegen 的异常日志。
   `simd` 在 A5 上可能来自 Welford；mix 可能来自 no-linear。
5. **查最终候选与 binary**：记录 selected config 的 `compile_mode`、`num_warps`、multibuffer、tile，
   再核对 binary 的 `parallel_mode` / `is_pure_simt`；grouped autotune 还要记录真实 `group_id`。

手写/算法模板路径也要单独判断。例如 `triton_heuristics.template()` 直接使用空 kwargs 的 `Config`，
没有经过上面的 Index 候选生成器，见 [`triton_heuristics.py:3886`][template]。
因此该字段不是 torch_npu 全部 kernel 的统一分类接口，也不能用它区分 Cube、Vector、aclnn 等所有执行形态。

## 逻辑混乱的具体来源与验证范围

当前实现同时存在几种不同职责：

- **字段承担多种含义**：既影响生成期 DSL，又表示 autotune 的候选集合；mix 本身没有唯一 winner。
- **赋值分散、存在覆盖**：平台默认值、LoopBody 名称匹配、索引拆分失败、Welford 都能改字段。
- **同一个结果有多个入口**：例如 mix 同时来自 indirect/masked 和整个 no-linear backend。
- **配置同时改变 lowering 和分类**：开关前后可能不是同一份 IR/DSL。
- **名字与实际调用不完全直观**：`simt_template` 编译为 `simd_simt_template`，
  却不属于 `is_unified_simt_kernel()`；dynamic 辅助函数存在但没有接入调用链。
- **候选模式依赖 autotune 实现**：普通 mix 展开两类，FastA expert 展开三类；
  metadata 缺字段和候选缺 compile mode 又使用不同默认值。

本文的验证为固定提交的静态源码审阅，覆盖全仓字段检索、赋值/调用关系核对，
以及隔离执行原始枚举和分类函数：四个枚举的字符串映射、直接 indirect sum、
一层 mul 后 sum、普通索引 sum、非 sum、`reduction_1`、两种显式模式共 7 个分类分支检查通过。
两个 dynamic 辅助函数在 `_inductor` Python AST 中也未发现名称或属性引用。

另外，使用替身 tile generator 隔离执行原始候选调度函数，确认普通分支的四种模式选择和
空候选保底配置的字段；隔离执行 Welford 选择函数，确认其在没有向量轴候选时也已写入 `simd`。
这些检查验证 Python 控制流，不包含真实 tiling 生成、Triton 编译或设备结果。

本文未进行 NPU 编译或 launch 验证；IR 示例用于说明源码判断条件，
实际某个模型的最终 type 和 winner 仍应由上述 generated artifacts 验证。

[revision]: https://gitcode.com/Ascend/pytorch/commit/9377b9df839283ca9afded838d82de985f101e7c
[soc]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/config.py#L30
[enum]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton_utils.py#L26
[default]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L1048
[nolinear]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L945
[decide]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L1217
[transform-schedule]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L1478
[finalize]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L1538
[welford]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L1561
[unified-meta]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L2471
[static-block]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L2546
[unified]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L4048
[mask]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton.py#L4112
[indirect]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/ir.py#L1830
[classifier]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/ir.py#L1718
[name-match]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/triton_utils.py#L58
[linearization]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/ir.py#L1543
[unused-dynamic]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/ir.py#L1464
[combined]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/npu_combined_scheduling.py#L165
[scheduling]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/scheduling.py#L158
[ttir-check]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/codegen/scheduling.py#L272
[indirect-config]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/config.py#L345
[welford-config]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/config.py#L191
[fasta-config]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/config.py#L393
[override]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/lowering_override_list.py#L34
[embedding]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/lowering.py#L495
[fallback]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/lowering_fallback_list.py#L452
[candidates]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L3701
[fill-config]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/tile_generator.py#L240
[extra-options]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/tile_generator.py#L498
[fasta-candidates]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/fasta_autotune.py#L395
[fasta-fallback]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/fasta_autotune.py#L910
[compile-options]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L1071
[binary]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L1239
[grouped-entry]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L3223
[grouped-candidates]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L3345
[template]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/runtime/triton_heuristics.py#L3886
[welford-test]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/test/_inductor/test_var_mean.py#L88
[upstream-doc]: https://gitcode.com/Ascend/pytorch/blob/9377b9df839283ca9afded838d82de985f101e7c/torch_npu/_inductor/docs/feature/non_contiguous_accesses/INDUCTOR_INDIRECT_MEMORY_MODE.md
