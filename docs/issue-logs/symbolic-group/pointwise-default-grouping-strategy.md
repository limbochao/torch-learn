---
title: Pointwise default symbolic group 分组策略实现说明
---

# Pointwise default symbolic group 分组策略实现说明

## 1. 背景

当前 NPU Inductor 的 symbolic grouped autotune 会为动态 shape kernel 构造若干代表 shape，
并在运行时根据输入 shape 选择对应的 group 和 autotune 结果。原有 pointwise default 分组将
所有轴直接组成一个 `outer_product` feature，bucket 固定为：

```text
4096 * vector_core
```

这种方式没有区分动态 split 轴、静态 split 轴和普通 tiling 轴。对于 pointwise kernel，
尤其是动态 split 轴前后存在静态 split 轴时，单个 program 的工作量和 grid 并行度会随轴布局
发生不同变化，使用单一 numel bucket 容易让代表 shape 与实际运行场景不匹配。

本次修改只作用于 `pointwise` template 中未被识别为 `elementwise`、`broadcast` 或
`transpose` 的 default 分支。reduction、persistent reduction、elementwise、broadcast
和 transpose 的既有分组逻辑保持不变。

## 2. 代码调用链

分组 metadata 的生成路径如下：

```text
SplitTiling.select_split_tiling_axis()
        |
        +-- _classify_split_axes()
        |       `-- dynamic_split_axes / static_split_axes
        |
        +-- _build_grouped_meta()
                |
                `-- _build_group_features()
                        |
                        `-- _pointwise_default_feature()
```

生成的 `GroupFeatureSpec` 会经过 `create_inductor_meta()` 写入 generated kernel 的
`inductor_meta["group_features"]`，之后由 runtime grouped autotuner 使用：

```text
group_features
    |
    +-- build_group_representatives()
    |       `-- 生成每个 group 的 benchmark axis values
    |
    `-- NPUSymbolicGroupedAutotuner._runtime_feature_inputs()
            `-- _resolve_group_id()
                    `-- 运行时选择 group
```

相关源码位于：

- `pytorch_new/torch_npu/_inductor/codegen/split_tiling.py`
- `pytorch_new/torch_npu/_inductor/runtime/symbolic_grouping.py`
- `pytorch_new/torch_npu/_inductor/runtime/triton_heuristics.py`

## 3. pointwise layout 与 default 分支的关系

`_pointwise_layout_kind()` 根据 load/store index 中的轴 stride 顺序识别 pointwise 的访问布局：

| 返回值 | 含义 |
| --- | --- |
| `None` | 普通 pointwise/default |
| `broadcast` | 存在 broadcast 访问 |
| `transpose` | 不同访问的 stride 顺序不一致 |
| `transpose_broadcast` | 同时存在 transpose 和 broadcast |

`_build_group_features()` 当前按以下优先级选择策略：

1. reduction / persistent reduction 使用 reduction 专用 feature；
2. `workload == "elementwise"` 使用 elementwise 全轴乘积；
3. `pointwise_layout` 为 broadcast 或 transpose-broadcast 时，按 primary axis 使用既有 bucket；
4. `pointwise_layout == "transpose"` 时，按 transpose 专用 bucket；
5. 其余情况进入本次新增的 pointwise default 策略。

因此，`pointwise_layout` 只是访问布局分类，不代表新的分组特征。后续 elementwise 需要采用
类似策略时，可以复用轴结构分析，但不应把 elementwise 逻辑继续堆叠到 layout 判断分支中。

## 4. 当前 pointwise default 策略

### 4.1 支持范围

当前新策略只处理恰好一个动态 split 轴：

```text
dynamic_split_axes 数量 == 1
```

如果存在多个动态 split 轴，当前实现返回不可用 feature，由现有 grouped codegen guard
决定是否关闭 grouped plan；没有动态 split 轴的 pointwise default 也不构造本策略的 feature。

这不是对多动态轴场景的最终策略设计，而是为了避免在 primary axis 降级其他动态轴后，
仍使用无法准确描述剩余动态轴的代表 shape。

### 4.2 轴角色

设唯一动态 split 轴为 `D`。静态 split 轴按照 `kernel.sorted_axis` 中相对于 `D` 的位置分为：

```text
prefix static split axes: D 之前的静态 split 轴
suffix static split axes: D 之后的静态 split 轴
```

静态 split 轴仍然保留在 `kernel.split_axis` 中，不修改轴定义，也不把它们降级为普通 tiling 轴。

区别只发生在分组特征计算中：

- prefix static split 参与 group 的原始 feature value；
- suffix static split 不参与该 feature value，但会影响 launch policy，并触发额外的 `V/2` bucket；
- dynamic split 轴 `D` 是 primary group axis。

### 4.3 tiling product 与 base

`tiling_product` 遍历 `kernel.tiling_axis`，但排除所有仍然属于 `kernel.split_axis` 的轴：

```text
tiling_product = product(length(axis) for axis in tiling_axis if axis not in split_axis)
```

因此静态 split 轴不会因为同时具有 tiling 属性而被计入 tiling product。随后计算：

```text
base = ceil(4096 * vector_core / tiling_product)
upper = max(base, 8 * vector_core)
```

这里 `base` 表示在当前普通 tiling 工作量下，动态 split 轴对应的典型规模；它不是完整
kernel numel，也不是静态 split 轴与动态 split 轴的乘积。

### 4.4 bucket 边界

没有 suffix static split 时，边界为：

```text
(2 * vector_core, upper)
```

有 suffix static split 时，增加半 vector core 的场景：

```text
(vector_core / 2, 2 * vector_core, upper)
```

边界最终会排序并去重。bucket 的语义仍是已有实现的左开右闭区间，加上最后一个 open bucket：

```text
(0, b0]
(b0, b1]
...
(last, +inf)
```

open bucket 的 representative 使用最后边界的两倍，沿用现有 `build_group_representatives()`
约定。

## 5. `C_prefix * D` 与 bucket 坐标分离

当动态 split 轴之前存在静态 split 轴时，用户侧定义的分组值是：

```text
raw_feature_value = C_prefix * D
```

其中 `C_prefix` 是 prefix static split axes 的长度乘积。但 bucket 边界必须仍按照只有动态轴
`D` 时的规则计算，不能把边界整体乘以 `C_prefix`。

为此，`GroupFeatureSpec` 新增：

```python
bucket_factor: int = 1
```

pointwise default feature 的设置为：

```text
axis_names = prefix_static_axis_names + (dynamic_axis_name,)
bucket_factor = C_prefix
```

运行时先计算原始乘积，再归一化后 bucketize：

```text
bucket_coordinate = raw_feature_value // bucket_factor
group_id = bucketize(bucket_coordinate, buckets)
```

代表 shape 生成时也使用同一规则。对于一个 bucket，代表的动态轴值直接取该 bucket 的上界，
原始 feature value 再由 `C_prefix * D_rep` 得到。这样可以同时满足：

1. group feature 保留真实的 prefix split 工作量；
2. bucket 边界与“只有一个动态 split 轴”的策略一致；
3. benchmark representative 和 runtime group selection 使用同一坐标系。

旧的 metadata 没有 `bucket_factor` 时，payload/runtime 默认使用 `1`，因此已有 reduction、
elementwise、broadcast 和 transpose feature 的行为不变。

## 6. 代表 shape 与 launch policy

`build_group_representatives()` 仍负责枚举全部理论 group，并将无法构造代表值的 group 标记为
unreachable。对当前 pointwise default 的单一动态轴场景：

- prefix static split 轴从 `axis_static_values` 读取固定值；
- 动态轴使用 bucket 上界或 open bucket 的两倍；
- suffix static split 轴不属于 group feature，但仍作为静态轴写入 benchmark axis environment；
- runtime launch policy 继续把非 primary runtime block 作为静态 block，并据此计算 prior programs。

因此，本次实现没有改变 split axis 的 grid 语义，只改变 representative 和 group id 的分类坐标。

## 7. 为 elementwise 扩展保留的结构

当前 `GroupFeatureSpec.bucket_factor` 已经是通用 metadata 字段，不属于 pointwise 专用字段。
后续 elementwise 采用类似策略时，建议将 `_build_group_features()` 进一步拆成：

```python
context = _build_group_context(
    dynamic_split_axes,
    static_split_axes,
    primary_axis,
)

if workload == "elementwise":
    return _build_elementwise_group_features(context)

return _build_pointwise_group_features(context, pointwise_layout)
```

推荐把以下内容放入共享的 `group_context`：

- dynamic split axes；
- static split axes；
- prefix/suffix 关系；
- 非 split tiling product；
- prefix factor；
- `vector_core` 和 `base`；
- 是否存在 suffix static split。

`workload` 只表达计算 workload 子类，`pointwise_layout` 只表达访问布局，具体 bucket 策略由
各 workload builder 自己决定。这样后续 elementwise 可以复用轴结构分析和 bucket normalization，
而不需要依赖 pointwise default 的分支顺序。

## 8. 修改范围、验证与风险

本次修改文件：

| 文件 | 作用 |
| --- | --- |
| `torch_npu/_inductor/codegen/split_tiling.py` | 生成 pointwise default feature 和 bucket 边界 |
| `torch_npu/_inductor/runtime/symbolic_grouping.py` | 保存 bucket_factor、生成 representative、兼容旧 payload |
| `torch_npu/_inductor/runtime/triton_heuristics.py` | runtime group id 按归一化坐标解析 |

已完成的本地验证：

```text
python -m py_compile \
  torch_npu/_inductor/codegen/split_tiling.py \
  torch_npu/_inductor/runtime/symbolic_grouping.py \
  torch_npu/_inductor/runtime/triton_heuristics.py
git diff --check
```

本地环境没有 PyTorch/NPU runtime，因此尚未完成目标设备上的 kernel compile、selected config、
runtime group id 和性能验证。当前主要风险是：

- 多动态 split 轴仍未纳入新策略；
- symbol range 约束下 representative 是否可达，仍依赖现有 grouped plan 的可达性检查；
- 不同 pointwise layout 的旧策略尚未迁移到新的 tiling-product 分组框架。

这些限制是当前实现范围，不应从本次 pointwise default 修改推断为所有 pointwise workload
已经采用统一的新策略。
