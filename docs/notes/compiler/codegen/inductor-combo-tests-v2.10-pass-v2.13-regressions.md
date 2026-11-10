---
published: false
---

# 2.10 已通过用例在 2.13 Combo 同步分支上的优先验收

优先范围取自 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 中“最新结果”以“通过”开头的行。
共 60 个独立源方法，均包含在 2.13 的 105 方法清单中。11 行嵌套函数随外层验证，不重复计数。
本轮完整任务仍在执行，尚不能声明这 60 个方法全部通过。

[2.10 基线表](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv)；
[2.13 当前表](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)；
[自动生成的逐方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-interface-fix/baseline-pass-comparison.json)。
逐方法状态以后者和当前表为准，本文记录源码分析，不固定运行中的统计。

## 版本与验收口径

2.10 基线为 `a8dcca84bf`，62 个执行方法中 60 PASS、2 FAIL，407/409 个实例通过。
它使用的是经过 NPU 功能适配的 2.13 用例，不是完全原样的 CUDA 用例。
[适配脚本](../../../../scripts/tests/combo_kernel_210/adapt_combo_210.py) 明确移除了所选方法的
`generated_kernel_count`、CUDA `FileCheck` 断言，并将动态 grid 用例改为保留两组动态输入数值比较。
因此不能从旧表的 PASS 推导“融合数和 CUDA 相同”，也不能只比较方法名就认定测试条件完全一致。

2.13 当前环境为 torch `2.13.0+cpu`、torch_npu `2.13.0+git64dc85c`、Triton `3.6.0`，
叠加 Python 修复 `5a5bda0ae3`，测试快照 `a889082`。
foreach 已对齐 2.10 功能验证口径；其他文件保留当前测试快照中的断言。
各阶段使用独立新缓存。安装文件哈希由运行脚本检查，不能仅以包版本字符串判断实际修复版本。

验收同时记录原始 pytest 状态和失败阶段：

- 数值未完成、编译或运行崩溃：功能尚未通过。
- 数值通过、数量或生成代码检查失败：原始状态仍为 FAIL，需要单独核对融合策略和测试适配。
- SKIP：能力未验证，不能替代 PASS；残留的旧 NPU skip 需要单独处理。
- 尚未执行：保持待执行，不复用 2.10 的 PASS。

## 优先级一：动态 benchmark 的旧接口未替换

已触发的方法：

- `test_dynamic_shapes_2d_blocking`
- `test_dynamic_shapes_mutated`
- `test_dynamic_shapes_persistent_reduction_no_x_dim`
- `test_dynamic_shapes_2d_blocking_round_robin`

调用链是 `benchmark_combo_kernel → generate_combo_kernel_code → codegen_kernel_benchmark →
kernel_benchmark_extra_args`，最后报：

```text
AttributeError: 'SizeVarAllocator' object has no attribute 'size_hint'
```

NPU 源码 `torch_npu/_inductor/codegen/triton_combo_kernel.py::kernel_benchmark_extra_args`
仍调用 `V.graph.sizevars.size_hint(length, fallback=...)`。
社区 2.13 对应方法已调用 `optimization_hint()`；`SizeVarAllocator` 的优化提示接口也已更名。
这是明确的 2.13 接口适配遗漏，不是旧 F11 再次导致中间 buffer 残留 store。
例如 `test_dynamic_shapes_mutated` 当前在生成 benchmark 时已失败，尚未到达旧 F11 的运行检查。

处理方向是保留 NPU 动态轴参数顺序和实际长度来源，改用 2.13 的优化提示接口。
修复后仍须验证动态输入和 mutation 数值，不能把越过该 AttributeError 等同于完整修复。
原始报告见当前证据目录的 `68-`、`70-`、`72-`、`74-` 开头 XML。

## 优先级二：普通回退生成重复的输入释放

重合方法为 `test_combo_kernel_per_config_subkernel_red`。
输入是两个 shape 为 `(32, 64, 128)` 的 contiguous 张量，dtype 为默认 float32，
对 `(0, 2)` 两个维度归约，输出各为 64 个元素。
当前报错发生在生成的 Python wrapper 运行阶段：

```python
triton_red_fused_sum_0.run(arg0_1, buf0, 64, 4096, stream=raw_stream0)
del arg0_1
buf1 = empty_strided_npu((64, ), (1, ), torch.float32)
triton_red_fused_sum_0.run(arg1_1, buf1, 64, 4096, stream=raw_stream0)
del arg1_1
del arg0_1
del arg1_1
```

这段摘录保留真实语句顺序；省略了输入解包、对齐复制、注释和返回语句。
第二次 `del arg0_1` 抛出 `UnboundLocalError`，不是数量断言或数值容差问题。
原始异常见当前证据目录的 `17-test_combo_kernel_per_config_subkernel_red.xml`。

2.10 的审计记录显示，该方法原本就是 Combo TTIR 校验失败后回退普通 kernel，最终功能 PASS：
[历史路线证据](../../../../artifacts/combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/audit.test_combo_kernels.test_combo_kernel_per_config_subkernel_red.jsonl)。
因此不能把 2.10 PASS 解释为该输入的 Combo 主路径已经支持。

源码检查发现两层生命周期管理存在交互：

1. Combo 回退先保存并清空外层 `pending_frees`，逐个调用普通 `codegen_node()`，结束时恢复并释放整个集合。
2. 2.13 普通非线性回退新增 `_codegen_nolinear_groups()`，按重新分组后的 `last_usage` 逐组释放。
3. 社区 `WrapperCodeGen.codegen_free()` 对 `InputBuffer` 写出 `FreeLine` 后立即返回，
   不加入用于可复用 buffer 的 `freed` 集合；再次恢复外层释放集合时不能依赖该集合自动去重。

生成代码已确认重复释放。定向日志也确认 Combo 编译失败后进入普通 kernel 回退，
两个普通 Index kernel 又分别失败并进入 `_codegen_nolinear_groups()` 的非线性回退。
结合源码，可定位到组内输入释放后再次恢复外层 `pending_frees` 的清理冲突。
尚未通过修改清理集合后的复测确认最终修复点；
不能只删生成代码中的第二个 `del`，也不能粗暴延长所有 buffer 的生命周期。
两版社区的 InputBuffer 分支相同，当前差异重点在 NPU 2.13 新回退与移植的 Combo 清理流程组合。

该方法在回退前还出现 `R2BLOCK_SUB is not defined`，说明存在更早的 Combo 静态块参数定义问题。
2.10 历史 Combo 失败为 scalar pointer 写入 block value；当前定向日志也有这类候选失败。
应分别处理缺失参数定义、候选合法性与回退生命周期，不能把所有异常合并为一个已确认根因。

## 优先级三：功能通过，但融合数量检查与旧基线口径不同

`test_combo_kernel_per_config_subkernel_per` 的 eager/compiled 数值比较已通过，
失败为 `generated_kernel_count` 预期 1、实际 2。
2.10 的适配脚本删除了这一数量断言及 CUDA DSL 结构检查，所以旧 PASS 不能证明它也生成 1 个 kernel。
历史审计只证明走过 linear Combo 路径，未提供该方法最终数量与分区分布：
[2.10 路线证据](../../../../artifacts/combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/audit.test_combo_kernels.test_combo_kernel_per_config_subkernel_per.jsonl)。

最新定向日志确认 `2 nodes partitioned into [2] groups`，随后 Combo 编译报
`NameError('R1BLOCK_SUB is not defined')`，再回退两个普通 kernel。
因此本例实际数量变化来自编译回退，不是上游或 NPU 主动拆为两个分区，也不是虚增计数。

源码对照发现移植时 `codegen_static_numels_sub_kernel()` 的条件从：

```python
if tree.is_no_loop_axis or (tree.is_reduction and sub_kernel.persistent_reduction):
```

变为：

```python
if tree.is_no_loop_axis:
```

原来的第二个分支为 persistent reduction 轴生成 body 内静态 `R*BLOCK_SUB`。
当前生成代码继续引用该变量，但这个分支已删除；这与本次日志的未定义变量直接对应。
属于明确的移植差异，应恢复正确的静态定义规则并验证两种 reduction kernel 类型。
本次只定位、未修改安装源码，不能宣称修复后已 PASS。

## 测试适配差异与环境失败

| 重合方法 | 当前证据 | 是否可归因为实现同步遗漏 |
| --- | --- | --- |
| `test_combo_kernel_dynamic_shapes_grid_changes` | NPU profiler 返回事件列表；测试按 `traceEvents` 字典读取，直接 TypeError。2.10 已移除该 profiler/grid 检查并保留两组动态输入数值检查 | 测试适配尚未对齐；后续数值和动态 grid 能力仍待验证 |
| `test_rand_like_deterministic` | Combo=True 实例仍由源码中的“NPU Inductor does not support ComboKernel yet”主动 skip；2.10 适配曾去掉它 | 旧 skip 残留，不能据此判定算子或随机数实现缺失 |
| `test_type_promotion` | `_foreach_clamp_max` 实例在编译子进程初始化设备时超时，随后 context 为空；其余实例数值检查有通过报告 | 当前是设备初始化失败证据；需重试该实例后才能判断是否另有编译或精度问题 |

`test_combo_kernel_filter_cpu` 虽尚待调度，源码也有宽泛的 NPU skip；该方法同属 60 个验收项，
后续不能将其 SKIP 当作满足基线。当前不因静态发现 skip 就提前改变运行结果。

## 已有五个修复是否被漏带

以 `a8dcca84bf` 的有效补丁逐项检查当前 2.13 源码：

| 旧修复 | 当前源码检查 | 验证边界 |
| --- | --- | --- |
| F10/F12：删除 inplace store 状态 | Combo 聚合 `inplaced_to_remove`，最终 `code.getvalue()` 在 Combo handler 中渲染 | `test_mean` 已 PASS；其他目标以运行表为准 |
| F11：移除中间 buffer store | Combo 聚合 `removed_buffers` | `test_mutated_benchmark` 已 PASS；动态 mutation 当前被旧 size_hint 接口提前阻塞 |
| F8：benchmark 尺寸收集副作用 | `generate_node_numel_expr()` 的空名称提前返回仍存在 | `test_combo_kernel_size_hint_failure` 尚待执行，不能仅凭代码存在宣布通过 |
| F4：FloorDiv 无进位拆分 | 两个除法分析 helper 与旧修复的 AST 一致；mask 过滤为后续主动恢复 | pooling 的 Combo=False/True 已在本轮 2/2 PASS |
| F5：值为 1 的常量 metadata | 普通 Index 和 Combo 的 `equal_1_arg_indices` 登记均存在 | 随机数 Combo=True 仍被测试 skip，需放开适用条件后验证 |

因此当前证据不支持“旧五个修复整体没同步”的结论。优先处理确实暴露的 2.13 接口和回退协议差异，
并将基线功能检查与额外的 CUDA 结构检查分开说明。

## 定向复现与结果更新

在相同 2.13 包和 Python 叠加源码下，使用独立新缓存执行原生 pytest：

```bash
ASCEND_RT_VISIBLE_DEVICES=<allowed-card> \
TORCHINDUCTOR_CACHE_DIR=<fresh-cache> INDUCTOR_ASCEND_LOG_LEVEL=DEBUG \
python -m pytest -vs --tb=short --log-file=<debug-log> --log-file-level=DEBUG \
  test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_red \
  test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_per
```

该定位运行与完整任务分开，不覆盖完整结果，也不修改安装源码。
2026-11-10 定向执行上述两个原始方法，耗时约 104 秒，均仍 FAIL，安装源码前后哈希一致。
已确认两个方法都分为 `[2]` 单个 Combo 分区，随后分别出现缺失静态 BLOCK_SUB 的编译异常并回退。
证据摘要见
[定向调度与失败证据](../../../../artifacts/combo-213-baseline-pass-diagnosis-20261110/evidence.json)。
`publish_plain_pytest.py` 每次发布均重新读取 2.10 基线，生成 60 方法状态对照，
在同一 116 行表格的“当前分析”中标出非 PASS 的优先项，不增加表格列；PASS 项不保留历史失败说明。
