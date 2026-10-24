---
title: 修复 ⑥：generate_node_schedule 如何匹配符号 group 并安排归约前后节点
published: false
---

# 修复 ⑥：generate_node_schedule 如何匹配符号 group 并安排归约前后节点

对应整合提交 `e1429cc917ec8a6415a4bae51d694b1b81857c3d`，主题为
`fix(inductor): compare symbolic scheduling group extents`。
生产修改在 `torch_npu/_inductor/codegen/scheduling.py`，
完整范围和新增测试见[补丁][patch]，八项关系见[总览][overview]。

2026-10-24 对融合与回退流程的进一步核对确认：index 接受的融合组重建为 no-linear 标量 group 后，
需要重新执行目标 backend 的融合检查。独立方案见
[no-linear 回退融合复核详解](model-fix-06-nolinear-fusion-recheck-20261024.md)，对应 PR !48469。
本文继续记录 `e1429cc917` 的历史实现及验证，不代表新方案已通过设备验证。

## 这个提交解决什么问题

**解决动态尺寸下数学上相同的迭代规模被误判为不同 group，导致调度阶段编译失败的问题。**

- **触发场景**：NFNet 动态训练中，同一个融合归约的前置逐点计算和归约节点，
  用不同形式的符号表达式描述输入元素总量。例如 `A*q**2 + 2*A*q + A` 与 `A*(q**2 + 2*q + 1)`。
- **修复前**：相关调度路径未能证明这些表达式相等，无法把节点匹配到归约体内或归约后，
  抛出 `unexpected group`。此时尚未生成可执行的 Triton kernel。
- **修复后**：先统一尺寸容器的表示，再利用已有符号约束证明等值；
  两类 NPU scheduler 都使用该逻辑，把前置计算、归约、后置计算安排到正确位置。
  无法证明相等的尺寸仍不能直接视为匹配。

[捕获的 IR][ir]展示了表达式差异，[修复后 DSL][dsl]和[检查记录][checks]展示了正确调度。
该修复对应 NFNet 已观察到的 `unexpected group`；后续双 shape 小图暴露的 Layout/range 异常
分别由第 7、8 项处理。

## 这个函数的产物是节点序列，不是设备上的 sum

Inductor 将若干图节点融合到一个 kernel 后，还需要安排它们在归约循环内外执行。
`generate_node_schedule(nodes, numel, rnumel)` 接收节点和当前 kernel 的迭代规模，
返回节点及 `DisableReduction`、`EnableReduction` 标记组成的序列。
后续 codegen 根据这个序列输出循环、计算和 store。

这里的 group 是节点迭代规模的描述，不是分布式 process group。
节点的 `group` 里包含设备信息及 `(node_numel, node_rnumel)`：

| 字段 | 含义 | 例子 |
| --- | --- | --- |
| `numel` | 当前 kernel 保留方向的元素量 | batch×channel |
| `rnumel` | 当前 kernel 每个输出需要归约的元素量 | height×width |
| 节点 `(X*R,1)` | 在全部输入位置运行的 pointwise producer | 先做乘法再进入 sum |
| 节点 `(X,R)` | 对 R 个输入求一个结果的 reduction | 空间 sum |
| 节点 `(X,1)` | 消费归约结果的 pointwise epilogue | sum 后计算 sigmoid 梯度 |

producer 指给后面运算生产中间值的节点；epilogue 指归约结束后消费结果的逐点部分。
这些都是节点职责，不是模型里必须存在的 Python 函数名。

## 为什么编译器需要匹配 group

以 `output[x] = sum_r(input[x,r] * weight[x,r]) * scale[x]` 为语义示意：

```text
归约体内：按 (x,r) 生成乘法，累积 sum
归约结束：按 x 取完整 sum，再乘 scale，写出 output[x]
```

如果把最后一次乘法放到归约循环内部，可能读到部分 sum 或重复写回。
如果把 producer 放到循环外，又可能失去 r 的迭代环境。
group 匹配帮助调度器区分“适合主归约体”和“需要归约结束后执行”的节点。
它不是整个融合合法性的唯一判断，地址布局和范围兼容性仍要单独检查。

## 已有 `_generate_npu_node_schedule()` 在做什么

这个 helper 在本提交之前已经存在，不能把它整段说成本提交新写的算法。
本提交修改其中的尺寸比较，并扩大两条 scheduler 使用它的范围。

主要判断为：

```python
def fits_in_main_body(n):
    _, (node_numel, node_rnumel) = n.group
    return (same_size(node_numel, numel) and same_size(node_rnumel, rnumel)) or (
        same_size(node_numel, numel * rnumel) and same_size(node_rnumel, 1)
    )

def fits_outside_reduction(n):
    _, (node_numel, node_rnumel) = n.group
    return same_size(node_numel, numel) and same_size(node_rnumel, 1) and rnumel != 1
```

第一个接受 reduction 自身 `(X,R)` 以及输入侧 pointwise `(X*R,1)`。
第二个识别输出侧 pointwise `(X,1)`。
都匹配不上就抛 `unexpected group`，拒绝当前调度解释。

helper 还保留了以下机制：

| 状态或 helper | 用途 |
| --- | --- |
| `done` | 防止同一节点被重复加入序列 |
| `not_ready_yet_nodes` | 记录在归约结束前尚不能作为完整结果消费的节点 |
| `requires_closing_previous_reduction()` | 根据祖先依赖判断是否先结束上一段归约 |
| `current_loop_buffer_usage` | 跟踪当前循环使用的 buffer，为内存复用判断提供依据 |
| `maybe_split_index` | 必要时插入归约段切换标记的位置 |
| `end_current_reduction_loop()` | 输出 Disable/Enable 标记并重置该段状态 |

这些已有逻辑解释了为何不能为了修一个比较而绕过整个调度过程。

## 问题中的两个表达式为什么“看起来不同但值相同”

NFNet 动态图中，令 `q=FloorDiv(H-1,32)`，`A=1536*batch`，
捕获的[真实调度 IR][ir]包含等价于以下形式的 group：

```text
producer  = (A*q*q + 2*A*q + A, 1)
reduction = (A, q*q + 2*q + 1)
epilogue  = (A, 1)
```

producer 的元素数等于 `A*(q*q+2*q+1)`。
SymPy 的 `==` 主要比较表达式结构，展开式和乘积式可能返回 False。
这不是证明两者的数学值不相等。例如整数 3×4 与 12 是同一个数，
但符号树 `Mul(A, Add(...))` 与 `Add(Mul(...),...)` 不是同一种结构。
动态图中的不同阶段可能独立 expand/simplify，不能要求它们始终保留相同书写形式。

## 改动 1：`same_size()` 先归一化容器，再请求等值证明

新实现完整主体为：

```python
left = left.numels() if isinstance(left, NumelList) else left
right = right.numels() if isinstance(right, NumelList) else right
if left == right:
    return True
return sizevars.statically_known_equals(left, right)
```

第一、二行处理 `NumelList`：它是 NPU 用于保存分轴尺寸的 tuple 类，
例如 `(16,batch)`；`.numels()` 返回乘积 `16*batch`。
这里要比较的是迭代总量，而不是容器的轴分解结构。

第三、四行保留结构相等的快路径。静态数字或已经规范成相同形式时，无需额外证明。
最后一行才请求尺寸系统证明两边相等；无法证明就返回 False。

旧 helper 也曾在“两边都是 `sympy.Expr`”时调用 `statically_known_equals`，
所以不能把这次修改概括成“以前所有比较都只有 `==`”。
问题还包括 NumelList/int 未进入该证明分支，以及实际非 Welford 调度根本没有调用这个 helper。

`statically_known_equals()` 通过 `statically_known_true(Eq(...))` 使用现有符号知识，
不新加 guard。guard 是编译产物运行前要检查的输入条件；
此处没有通过“给当前样例加一个相等约束”强行接受未知关系。

## 改动 2、3：两类 scheduler 都改用统一 helper

修改的两个入口分别为：

```python
class NPUNoLinearTritonScheduling(TritonScheduling):
    def generate_node_schedule(self, nodes, numel, rnumel):
        return _generate_npu_node_schedule(self, nodes, numel, rnumel)

class NPUTritonScheduling(TritonScheduling):
    def generate_node_schedule(self, nodes, numel, rnumel):
        return _generate_npu_node_schedule(self, nodes, numel, rnumel)
```

两者分别配合不同 NPU kernel 生成路径。
旧实现只有启用 Welford 且节点中存在 Welford group 时才用 NPU helper；
其他情况返回社区父类调度器，非 Welford 分支还重新取得原始 group 规模再调用父类。
这些早退分支在本提交中全部删除。

为什么一定要改入口？如果只修 `same_size()`，NFNet 的非 Welford sum 仍走父类，
修好的比较函数没有机会处理这次错误。
两条入口统一后，NumelList 归一化和符号证明才真正作用于当前失败路径。
这也意味着补丁影响面大于一个比较表达式，必须纳入调度回归评估。

## 用本提交的单元走一次完整返回值

新增 `test_expanded_pointwise_and_reduction_groups` 构造 producer、reduction、epilogue 三节点。
两类 scheduler 和 Welford 开/关都运行；一条路径还使用 NumelList 表示 group。
预期返回：

```python
[pointwise, reduction, DisableReduction, epilogue, EnableReduction]
```

前两节点属于主归约体；epilogue 前结束归约，执行后恢复生成器的归约状态标记。
`EnableReduction` 不是要求把 epilogue 重新执行一次，也不是启动另一个设备 kernel。

另一个单元构造没有相等证明的独立符号 x、y，要求仍抛 `unexpected group`，
同时比较调用前后的 `ShapeEnv.guards`，确认没有新增约束。
这是正例与拒绝例的共同验证，而不是把所有 group 都判作兼容。

## 为什么不只在 `group_fn()` 里 expand

`group_fn()` 和 `_canonical_group_size()` 确实会整理尺寸，
但不同节点、不同容器仍可能独立形成表达式。
在真正比较的边界使用等值证明，才能让所有来源遵守同一种“尺寸相等”的语义。
修改最终 DSL 的 sum 无法解决此故障，因为旧调度器在生成 DSL 之前就已抛异常。

## 模型必要性、结果与风险

NFNet 动态训练原本抛 `unexpected group`，修复后两个历史动态训练场景通过。
[修复后反向 DSL][dsl]显示先完成空间 sum，再执行 sigmoid 梯度的 epilogue。
单元两个方法及其组合检查通过，见[检查记录][checks]。
第二组动态 shape 后来暴露的 Layout/range 问题属于第 7、8 项，不能算成本项又一次失败。

本提交无条件改变两类 NPU scheduler 的路由，覆盖静态、动态、Welford 和非 Welford 场景。
风险包括更广的循环切换、依赖处理、融合拓扑兼容性和符号证明开销。
总元素数相同并不代表任意地址布局都可融合，后续兼容性检查仍必须成立。
现有证据不覆盖所有 scan、多段依赖 reduction 或全部布局；本次没有重跑设备矩阵。

[overview]: model-fix-dsl-explained-20261021.md
[patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/scheduler-symbolic-groups.patch
[ir]: ../../../artifacts/cicd/model_fixes_20261020/scheduler/nfnet_backward_fused_group_ir.txt
[dsl]: ../../../artifacts/cicd/model_fixes_20261020/generated/nfnet_backward_v5.py
[checks]: ../../../artifacts/cicd/model_fixes_20261020/final_checks/scheduler.txt
