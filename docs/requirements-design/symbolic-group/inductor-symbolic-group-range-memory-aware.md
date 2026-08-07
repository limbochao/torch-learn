---
title: Range/Memory-Aware Symbolic Group 分组设计草案
---

# Range/Memory-Aware Symbolic Group 分组设计草案

> **状态：设计草案。实现状态：未实现。验证状态：未验证。**
>
> 本文不描述 `torch_npu v2.10.0` 的线上行为，也不能作为当前代码已经具备该能力的依据。

## 1. 背景

现有经验档位只根据 workload feature 的固定边界构造 group representative。若符号的真实取值范围很窄，固定档位可能生成无意义的 group；若 representative 对应的 tensor 规模超出实际可分配内存，group autotune 可能在构造 benchmark input 或运行 candidate 时失败。

目标是在生成分组前同时考虑符号语义约束和 kernel 输入输出的内存约束，只在可行范围内构造 workload 档位。

## 2. 设计原则

处理顺序必须固定为：

```text
semantic_range
      INTERSECT
memory_feasible_range
          |
          v
effective_symbol_range
          |
          v
workload/numel range
          |
          v
经验档位裁剪或动态档位生成
```

不能先按经验档位生成 representative，再单独过滤超内存的 representative。后者无法反映符号有效域，也可能留下覆盖不完整或边界不连续的分组。

## 3. 范围定义

### 3.1 Semantic range

`semantic_range` 来自 ShapeEnv、guard 和 `mark_dynamic` 提供的上下界。范围必须是运行时合法的整数区间；无法取得可靠边界时，不应自行假设一个无限范围。

### 3.2 Memory-feasible range

对每个 kernel 输入输出 tensor，根据 shape expression、stride/storage size 和 dtype byte width 建立占用函数：

```text
tensor_bytes_i(s) = required_storage_elements_i(s) * dtype_bytes_i
kernel_bytes(s) = sum(tensor_bytes_i(s))
```

初版设计阈值：

- 单个 tensor 最大占用 `10 GiB`。
- kernel 全部输入输出合计最大占用 `20 GiB`。

上述数值只是待验证的设计参数，不是当前实现常量。workspace、mutation、alias 和多输出是否需要采用更保守的峰值模型，仍需实测确认。

`memory_feasible_range` 是同时满足所有单 tensor 限制和 kernel 总量限制的符号区间。若表达式非单调、包含多个相互独立符号，或无法保守求界，应放弃 grouped plan，回退普通 dynamic。

### 3.3 Effective range

```text
effective_symbol_range = semantic_range ∩ memory_feasible_range
```

- 交集为空：不做分组，回退普通 dynamic。
- 交集只有一个值：没有跨 shape 分组收益，可走普通 dynamic 或单配置路径。
- 无法可靠求交集：不做推测，回退普通 dynamic。

## 4. 从符号范围到 workload 范围

分组依据是 feature，而不一定是原始符号。应在 `effective_symbol_range` 上求 feature 的最小值和最大值，例如：

```text
elementwise_numel = product(all pointwise axes)
outer = product(non-reduction axes)
reduction = product(reduction axes)
```

只有一个动态符号且其余因子静态时，可以直接反算 workload 区间。多个相关符号需要利用已有等式约束；多个独立符号无法保守映射为可代表的单一范围时，初版应回退普通 dynamic。

## 5. 档位生成

先把现有经验边界与有效 workload 区间求交：

1. 删除区间外且不会形成有效分界的边界。
2. 保留落在区间内、能形成至少两个非空区间的经验边界。
3. 若经验边界不能有效覆盖该范围，再根据区间跨度生成动态边界。

动态边界的数量、间距和最小 bucket 宽度尚未定型。可优先评估按倍率分割，并用 representative 的可分配性、candidate 性能稳定性和额外编译成本决定最终策略。本文不把某一套动态边界公式定义为已确认方案。

每个 closed bucket 的 representative 必须落在自身有效区间内；open tail 也必须受 `effective_symbol_range` 上界约束，不能继续使用无限尾部假设。

## 6. 回退条件

任一条件成立时，整个 kernel 回退普通 dynamic，而不是保留部分不完整 group：

- 有效符号范围为空或不可求。
- tensor storage expression 无法安全估算。
- workload 在有效域上无法保守求界。
- representative 无法映射回合法 axis values。
- 分组后没有 reachable group。

## 7. 待验证项

- `10 GiB / 20 GiB` 是否适用于不同 NPU 型号和并发场景。
- storage size、workspace、alias、mutation 的峰值内存估算是否完整。
- 经验边界裁剪后，性能是否仍稳定覆盖整个 bucket。
- 动态边界的数量与额外编译/autotune 成本。
- range/memory 限制触发回退时，普通 dynamic 是否能稳定运行。

只有完成目标设备正确性、OOM 防护和跨 shape 性能验证后，本文状态才能从“未实现、未验证”更新。
