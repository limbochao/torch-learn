---
title: DLRM dynamic=True vs dynamic=False codegen 对比分析
published: false
---

# DLRM dynamic=True vs dynamic=False codegen 对比分析

## 分析对象

本次对比基于同一模型、同一输入 shape 下的四份 artifact：

- `artifacts/dlrm-dynamic-vs-static-codegen/static_fx_graph_readable.py`
- `artifacts/dlrm-dynamic-vs-static-codegen/dynamic_fx_graph_readable.py`
- `artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py`
- `artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py`

其中 `readable` 反映 FX graph 级语义，`output_code` 反映最终 Inductor/NPU codegen 结果。

## 先给结论

这组对比里，`dynamic=True` 和 `dynamic=False` 的前端语义没有变，真正变化发生在 codegen
阶段：

1. `readable` 图几乎等价，主要区别只是 batch 维从常量 `128` 变成了符号 `Sym(s20)`。
2. `dynamic=True` 的 `output_code` 明显更大，不是因为数学语义变了，而是因为：
   - A5 路径遇到 dynamic shape 时会跳过 memory access linearization，并把 kernel type
     强制改成 `SIMT_ONLY`。
   - embedding/sum 这一段没有收敛成 static 那种单个 `[B, 50, 20, 8] -> sum(dim=2)` 的
     reduction kernel，而是保留成 50 组 `slice + embedding + sum`，然后再被 scheduler
     按 `max_fusion_size=64` 切成 `21 + 21 + 8` 三个大 reduction kernel。
   - 由于 sparse 分支不再产出单个 `[B, 50, 8]` buffer，后续 `cat` 也无法沿用 static
     那条融合路径，只能先 materialize 一个 50 输入的 `aten.cat.default`，再继续后面的
     融合。
3. 这些动态 DSL 虽然“长得很大”，但从地址公式、slice 覆盖范围、reduce 维度和后续
   `aten.cat.default` 的顺序看，当前 artifact 本身没有暴露出错误语义；它更像是
   “较差的 fusion 形态”，不是“错误的 fusion 语义”。

下面把每一处差异拆开。

## 1. readable 图的差异

### 1.1 事实：op 语义完全一致

`static_fx_graph_readable.py` 和 `dynamic_fx_graph_readable.py` 的 ATen op 计数一致，
都是：

- `aten.embedding.default`: 50
- `aten.slice.Tensor`: 50
- `aten.sum.dim_IntList`: 50
- `aten.addmm.default`: 14
- `aten.relu.default`: 13
- `aten.cat.default`: 3
- `aten.bmm.default`: 1
- `aten.sigmoid.default`: 1

从图上也能直接看到两边都还是 50 组：

- `slice -> embedding -> sum`
- 然后 `cat`
- 然后 dense MLP
- 然后 `cat + bmm + triu/where + view + final addmm + sigmoid`

可见：

- static 起点：[`static_fx_graph_readable.py:1`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_fx_graph_readable.py#L1)
- dynamic 起点：[`dynamic_fx_graph_readable.py:1`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_fx_graph_readable.py#L1)
- static embedding 分支和后续主干：
  [`static_fx_graph_readable.py:353`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_fx_graph_readable.py#L353)
- dynamic embedding 分支和后续主干：
  [`dynamic_fx_graph_readable.py:353`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_fx_graph_readable.py#L353)

### 1.2 差异只在 shape 表达

`readable` 层的主要差异只有两类：

1. batch 维从常量 `128` 变成了符号 `Sym(s20)`。
2. 参数编号整体后移了一位，因为 dynamic 图把 `s20` 当作显式 shape 输入传入。

例如：

- static 输入是 `arg0_1: "i64[128, 1000]"`。
- dynamic 输入变成
  `arg0_1: "Sym(s20)"` 和 `arg1_1: "i64[s20, 1000]"`。

这个差异本身不改变语义，只是给后端 codegen 引入了 symbolic range。

## 2. output_code 的核心差异

### 2.1 kernel 总体规模差异

四份文件的行数已经说明了问题：

- `static_output_code.py`: 782 行
- `dynamic_output_code.py`: 1880 行

按 kernel 看更直观：

| artifact | kernel 数量 | 关键特征 |
| --- | --- | --- |
| static | 8 | 1 个 embedding reduction kernel，多个 `simd` pointwise kernel |
| dynamic | 10 | 3 个 embedding reduction kernel，全部 pointwise kernel 退成 `simt_only` |

尤其是 embedding/sum 段：

- static: `triton_red_fused_3`
  [`static_output_code.py:228`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L228)
- dynamic:
  `triton_red_fused_embedding_slice_sum_3/4/5`
  [`dynamic_output_code.py:366`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L366),
  [`dynamic_output_code.py:783`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L783),
  [`dynamic_output_code.py:1137`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L1137)

### 2.2 static 的 sparse 分支是什么形态

static 的关键 reduction kernel `triton_red_fused_3` 对应的 graph fragment 是：

1. `arg0_1` reshape 成 `[128, 50, 20]`
2. 对这个 3D index 做一次 embedding，得到 `[128, 50, 20, 8]`
3. 沿 dim=2 做一次 sum，得到 `[128, 50, 8]`

对应代码：

- graph fragment:
  [`static_output_code.py:216`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L216)
- kernel body:
  [`static_output_code.py:228`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L228)

kernel 里的核心地址公式也和这个语义一致：

```python
tmp0 = tl.load(in_ptr0 + (r3 + 20*y1 + 1000*z0), ...)
tmp6 = tl.load(in_ptr1 + (x2 + 8*tmp4), ...)
tmp8 = tl.sum(_tmp8, 2)
tl.store(out_ptr0 + (x2 + 8*y1 + 400*z0), tmp8, ...)
```

这里：

- `z0` 对应 batch
- `y1` 对应 50 个 field
- `r3` 对应每个 field 的 20 个 id
- `x2` 对应 embedding dim=8

所以 static 的 sparse 分支本质上是：

```text
feat_ids[B, 1000]
  -> reshape[B, 50, 20]
  -> embedding[B, 50, 20, 8]
  -> sum(dim=2)
  -> sparse_embedding[B, 50, 8]
```

### 2.3 dynamic 的 sparse 分支是什么形态

dynamic 没有得到上面那种单个 `[B, 50, 8]` sparse buffer，而是生成了 3 个大 kernel：

- `triton_red_fused_embedding_slice_sum_3`: 21 个 field
- `triton_red_fused_embedding_slice_sum_4`: 21 个 field
- `triton_red_fused_embedding_slice_sum_5`: 8 个 field

然后用一次真实的 `aten.cat.default` 把 50 个 `[B, 8]` 输出拼成 `[B, 400]`：

- 三个 reduction kernel 调用和 cat：
  [`dynamic_output_code.py:1678`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L1678)

从 DSL 可以直接看到每个 kernel 都是在把多个固定 offset 的 slice 手工展开。

以 `triton_red_fused_embedding_slice_sum_3` 为例，它覆盖这些 offset：

```text
0, 20, 40, ..., 400
```

`triton_red_fused_embedding_slice_sum_4` 覆盖：

```text
420, 440, 460, ..., 820
```

`triton_red_fused_embedding_slice_sum_5` 覆盖：

```text
840, 860, 880, ..., 980
```

也就是刚好把 `1000 = 50 * 20` 的第二维完整切完，没有重叠也没有缺口。

每一组的内部结构都一样。还是以第一个 kernel 为例：

```python
tmp0 = tl.load(in_ptr0 + (r2 + 1000*y0), ...)
tmp10 = tl.load(in_ptr0 + (20 + r2 + 1000*y0), ...)
...
tmp181 = tl.load(in_ptr0 + (400 + r2 + 1000*y0), ...)
...
tmp6 = tl.load(in_ptr1 + (x1 + 8*tmp4), ...)
...
tmp8 = tl.sum(_tmp8, 1)
...
tl.store(out_ptr0 + (x1 + 8*y0), tmp8, ...)
tl.store(out_ptr1 + (x1 + 8*y0), tmp17, ...)
...
```

这里每个 `_tmp*` accumulator 对应一个 field 的 `embedding + sum(dim=1)` 结果，
最后分别写到不同的 `out_ptr*`。这说明它并没有换算子语义，只是把 21 个
`slice + embedding + sum` 放进了同一个 DSL 里。

## 3. 为什么 readable 几乎不变，但 output_code 差这么大

### 3.1 dynamic shape 会直接改 kernel type

这是最上游、最确定的一条原因。

NPU A5 路径在 `transform_dims_in_indexing()` 里有明确逻辑：

- 如果检测到 dynamic shape，直接跳过 memory access linearization。
- 同时把 `V.kernel.npu_kernel_type` 设成 `SIMT_ONLY`。

源码：

- [`torch_npu/_inductor/codegen/ir.py:1148`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/codegen/ir.py#L1148)

关键代码是：

```python
if should_skip_linearization_on_a5(self.var_ranges, self.indexing):
    V.kernel.npu_kernel_type = NPUKernelType.SIMT_ONLY
    return
```

如果表达式分析失败，A5 还会再次退化成 `SIMT_ONLY`：

- [`torch_npu/_inductor/codegen/ir.py:1169`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/codegen/ir.py#L1169)

这就解释了为什么 dynamic 的那些 `addmm + relu` kernel，虽然 kernel body 基本没变，
但 `inductor_meta['npu_kernel_type']` 全部从 static 的 `simd` 变成了
`simt_only`：

- static 示例：
  [`static_output_code.py:70`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L70)
- dynamic 示例：
  [`dynamic_output_code.py:70`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L70)

这类差异属于“执行模板变化”，不是“数学语义变化”。

### 3.2 embedding 这种 indirect load + reduction，本来就倾向走 SIMT_ONLY

除了 dynamic shape 触发的通用退化，NPU 还专门对 indirect load + sum 做了
kernel type 判定：

- [`torch_npu/_inductor/codegen/ir.py:1254`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/codegen/ir.py#L1254)

注释写得很直接：

```python
For indirect load + sum pattern: simt_only is faster
```

也就是说，`embedding -> sum` 这类模式在 NPU 上本身就是重点走 SIMT 路线的对象。

### 3.3 dynamic 的 3 个大 DSL，为什么正好是 21 + 21 + 8

这是 scheduler 的 fusion 限制，不是随便切的。

PyTorch Inductor 有一个全局限制：

- `max_fusion_size = 64`
  [`torch/_inductor/config.py:531`](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/_inductor/config.py#L531)

而在 fusion 决策里，超过这个上限会直接拒绝 fusion：

- [`torch/_inductor/choices.py:248`](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/_inductor/choices.py#L248)

```python
and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
```

dynamic 这条链上，每个 field 仍然是 3 个独立节点：

```text
slice + embedding + sum
```

所以：

- 21 个 field = 63 个节点，刚好还能 fuse
- 22 个 field = 66 个节点，超过 64，不能继续 fuse

这正好解释了为什么 dynamic 最终被切成：

- 21
- 21
- 8

这是一个完全能用源码解释的数字，不是巧合。

### 3.4 为什么 static 没有出现这 21 + 21 + 8 的切分

static 最终拿到的是单个 `reshape -> embedding -> sum` 形态的 sparse kernel，
而不是 50 组独立的 `slice + embedding + sum`。

从 artifact 看，static 的 graph fragment 已经变成了：

```text
reshape[128, 50, 20]
  -> embedding
  -> sum(dim=2)
```

dynamic 没拿到这一步，因此才会被后面的 fusion size 限制切开。

代码层面，NPU 确实有一个专门想做这件事的 pass：

- `batch_embedding_fusion_pass`
  [`ascend_graph_pass.py:2493`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py#L2493)

它的目标非常明确：

```text
把多次 slice -> embedding -> reduce
合并成一次 reshape -> embedding -> reduce
```

而且它在分组时并没有把 symbolic batch 直接排除掉，因为 `_symbolic_shape_key()`
专门把 `torch.SymInt` 转成字符串参与分组：

- [`ascend_graph_pass.py:2082`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py#L2082)

它的 Pattern C 实现也正是 static artifact 里看到的那种目标形态：

- [`ascend_graph_pass.py:2350`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py#L2350)

但是，这个 pass 只要 fake-mode 的 reshape/embedding/reduce/select 任一步报错，
就会静默 `return False`：

- [`ascend_graph_pass.py:2391`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py#L2391)
- [`ascend_graph_pass.py:2435`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py#L2435)

所以这里能确定的事实是：

1. static 最终 artifact 已经是 `reshape -> embedding -> reduce` 形态。
2. dynamic 最终 artifact 仍然是 50 组 `slice -> embedding -> sum`。
3. 仓里存在一个正是为了把后者变成前者的 pass。

但仅凭现有 artifact，还不能 100% 证明“dynamic 一定是这个 pass 失败了”，
因为也可能是 dump 时机不同，或者还有别的 lower-level merge 路径参与。

更稳妥的表述是：

- `batch_embedding_fusion_pass` 是最符合 static 产物形态的源码证据。
- dynamic 最终没有拿到这类融合结果。
- 精确是哪一步阻止了 dynamic 走到同样形态，需要再开 pass log 或临时加日志确认。

这一点我在本文里把它标为 `inference`，不把它写成已证实结论。

## 4. 为什么后面的 DSL 也跟 static 不一样

### 4.1 sparse 分支产物变了，后面的 cat 就必须改写

static sparse 分支产出的是单个 `buf25: [128, 50, 8]`：

- [`static_output_code.py:670`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L670)

因此 static 可以直接把它当成一整块 sparse embedding，再和 dense 分支一起往
`[128, 408]` 的 alias buffer 里写：

- `triton_poi_fused_addmm_cat_relu_triu_view_wher_4`
- `triton_poi_fused_addmm_relu_5`

对应：

- [`static_output_code.py:676`](../../artifacts/dlrm-dynamic-vs-static-codegen/static_output_code.py#L676)

dynamic 没有这个 `[B, 50, 8]` buffer，而是先得到 50 个 `[B, 8]` buffer。
于是它必须先做一次 50 输入的 `aten.cat.default`：

- [`dynamic_output_code.py:1715`](../../artifacts/dlrm-dynamic-vs-static-codegen/dynamic_output_code.py#L1715)

然后才有后面的：

- `triton_poi_fused_addmm_cat_relu_triu_view_wher_6`
- `triton_poi_fused_addmm_cat_relu_7`
- `triton_poi_fused_cat_triu_view_where_zeros_lik_8`

所以“后面看起来融合算子都不一样”，根因不是后半段自己分叉了，而是前面的 sparse
分支已经换了 buffer 形态。

### 4.2 为什么 dynamic 这里没有把 50 输入 cat 继续 fuse 掉

NPU lowering 默认就不走 pointwise cat，而是直接 `ConcatKernel.create(...)`：

- [`torch_npu/_inductor/lowering.py:959`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/lowering.py#L959)

```python
if torch._inductor.config.force_pointwise_cat:
    return pointwise_cat(inputs, dim)

return TensorBox(ir.ConcatKernel.create(inputs, dim))
```

这意味着 dynamic 一旦真的落回 50 输入 `aten.cat.default`，后面就会先 materialize
一个 cat buffer，而不是像 static 那样继续保持“单个 `[B, 50, 8]` 中间结果”的形态。

## 5. dynamic 这几个超大 DSL 到底是怎么生成的

### 5.1 生成机制

这几个大 DSL 的生成机制并不神秘，本质上是：

1. 前端保留了 50 个 `slice + embedding + sum`。
2. scheduler 尝试把相邻、可融合的 node 横向融合。
3. 每 21 个 field 可以 fuse 成一个 kernel，再多就被 `max_fusion_size=64` 卡住。
4. codegen 对每个 fused group 直接做常量展开，于是一个 kernel 里出现 21 套
   `load indices -> gather embedding -> reduce -> store`。

所以 DSL 变“大”，本质上是“同一个模板被常量展开了 21 次”，不是生成器突然把别的算子
错误卷进来了。

### 5.2 为什么看起来融合算子不一样

看起来不一样，是因为 static 和 dynamic 的融合边界不同：

- static sparse 分支像是：

```text
reshape + embedding + sum
```

- dynamic sparse 分支像是：

```text
(slice + embedding + sum) * 21
```

两边最后做的事情仍然是“把 50 个 field 的 20 个 id 规约成 50 个 8 维向量”，
只是 static 选择了“先 reshape 再统一规约”，dynamic 选择了“先切 50 段再批量展开规约”。

## 6. 这些大 DSL 是否正确

### 6.1 从地址公式看，它们和 static 是等价的

static 的核心地址公式是：

```python
in_ptr0 + (r3 + 20*y1 + 1000*z0)
```

这等价于：

```text
feat_ids[z0, 20*y1 + r3]
```

dynamic 把它拆成多个常量 offset：

```python
in_ptr0 + (offset + r2 + 1000*y0)
```

其中 `offset` 依次取：

```text
0, 20, 40, ..., 980
```

这等价于把 static 里的 `y1` 手工常量展开：

```text
offset = 20 * y1
```

所以从 index 公式上，dynamic 的 3 个大 kernel 正是在手工实现 static 那个
`reshape[B, 50, 20]` 的第二维。

### 6.2 从 reduction 维度看，它们也正确

dynamic 每个大 kernel 都把：

- `y0` 当 batch
- `r2` 当 20 个 id 的 reduce 维
- `x1` 当 embedding dim=8

在 loop 结束后执行：

```python
tmp8 = tl.sum(_tmp8, 1)
```

这里 reduce 的正是 20 这一维，和原始 `torch.sum(emb, dim=1)` 的语义一致。

### 6.3 从输出拼接顺序看，也没有错位

dynamic 最终用：

```python
torch.ops.aten.cat.default([buf25, buf26, ..., buf74], 1)
```

把 50 个 `[B, 8]` 按 field 顺序拼成 `[B, 400]`。

而三个 kernel 覆盖的 offset 顺序也是单调递增的：

```text
0..400, 420..820, 840..980
```

对应 field 编号正好是：

```text
0..20, 21..41, 42..49
```

所以 cat 的顺序和原始 50 个 group_sum 的顺序一致。

### 6.4 当前 artifact 没显示出错误语义

基于当前四份 artifact，我能确认的是：

1. dynamic 的超大 DSL 没有漏 field，也没有重复 field。
2. 每个 field 的 reduce 维仍然是 20。
3. 后续 cat 的顺序和原始 field 顺序一致。
4. pointwise kernel 的 body 大多和 static 一致，只是 kernel type 从 `simd`
   退成了 `simt_only`。

所以“DSL 很大”本身不是错误证据。当前更合理的判断是：

- `dynamic=True` 走到了语义正确但形态更差的 codegen 路径。

如果要把“是否真的数值正确”从静态分析提升为已验证结论，还需要同一组输入下再跑一遍：

- eager
- static compile
- dynamic compile

做数值对比，并最好把生成 trace 一起保留下来。

## 7. 这次对比里最重要的几个根因

按优先级排，我认为最关键的是这三条：

1. `dynamic=True` 在 A5 上触发了
   `skip linearization -> force SIMT_ONLY`
   [`codegen/ir.py:1148`](https://gitcode.com/luqichao/pytorch_new/blob/50471c33618f9547c3616779ce3369a8d4b869a9/torch_npu/_inductor/codegen/ir.py#L1148)。
2. dynamic sparse 分支没有收敛成 static 的单个 `reshape -> embedding -> sum` 形态。
3. 保留下来的 50 组 `slice + embedding + sum` 又被
   `max_fusion_size=64`
   [`torch/_inductor/config.py:531`](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/torch/_inductor/config.py#L531)
   精确切成了 `21 + 21 + 8`。

这三条叠起来，正好解释了：

- 为什么 dynamic DSL 更大
- 为什么 kernel type 变了
- 为什么后续 cat/where/view 的融合边界也跟着变了

## 8. 还缺什么验证

当前文档已经能解释 artifact 差异和大部分源码原因，但还有两个点如果要彻底坐实，
建议下一轮补：

1. 打开 `batch_embedding_fusion_pass` 的日志，确认 static 是否真的命中了它，
   dynamic 又是在什么位置返回 `False`。
2. 跑同一组输入的 eager/static/dynamic 数值比对，把“看起来正确”升级成
   “已经验证正确”。

在不加额外日志、不跑环境的前提下，本文已经把当前四份 artifact 能证明的部分尽量证明完了。
