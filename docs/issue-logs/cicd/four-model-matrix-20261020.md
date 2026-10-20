---
published: false
---

# Albert、GPT2、NFNet、BERT 八场景测试与问题定界

本轮固定 FP32，分别执行静态/动态 shape、推理/训练、performance/accuracy，共 32 项。
32 项原始测试已全部结束：18 项通过，8 项编译失败，6 项精度失败。
失败均已定位到具体生成代码，并以单 kernel 或模型级诊断实验完成问题定界。

| 模型 | 原始失败场景 | 问题归属 |
| --- | --- | --- |
| AlbertForMaskedLM | 静态/动态训练 performance、accuracy，共 4 项 | Inductor 的 load 排布、资源估算和 tiling 候选覆盖 |
| GPT2ForSequenceClassification | 静态/动态推理 accuracy，共 2 项 | Inductor 的归约 mask；独立存在 TA masked indirect load lowering 错误 |
| nfnet_l0 | 四个动态场景，共 4 项 | Inductor pooling tiling 没有搜索可编译的小 tile |
| BERT_pytorch | 四个 accuracy 场景，共 4 项 | 前向同 GPT2 的两类 mask 问题；静态训练另有 Inductor backward 错误归约通道轴 |

## 环境、命令和结果判定

- 设备：Ascend950PR_9579，NPU_A5 容器。
- PyTorch：2.13.0+cpu，提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`。
- torch_npu：2.13.0.dev20261019，提交 `2c8682df9271e545e7f50f8c45e2395563a05483`。
- Triton：3.6.0；CANN：9.1；transformers：4.36.0；timm：1.0.30。
- 使用 Ascend `benchmarks/torchbench` runner，`--backend inductor --npu-backend triton --disable-aclgraph`。
- `TORCHINDUCTOR_COMPILE_THREADS=1`。单次 `triton.compile` 60 分钟超时，保存源码、配置和编译输入。
- 不覆盖模型默认 batch size。HuggingFace 性能测试为 4，精度为 1；BERT 性能为 16，精度为 4。
- NFNet 静态推理、训练性能默认 batch size 均为 128，accuracy 均为 8。
- performance 使用 `--repeat 1 --iterations 1`；accuracy 使用 `--iterations 5`。

performance 通过表示 runner 完成并输出正数 speedup，不表示该场景数值正确；accuracy 以 CSV 的
`pass_accuracy` 为准。进程返回 0 但 CSV 为 `fail_accuracy` / `fail_to_run` 仍记失败。
队列共用编译缓存，且诊断与其他任务并行，时延和编译时间用于运行记录，不作为受控性能基线。
动态场景使用 `--dynamic-shapes`，没有额外扩展到多个 batch/sequence length 的遍历。
NFNet 静态训练 performance 记录 9 次 graph break、6 类原因，输出 2 个 unique graphs；
本轮未要求 `fullgraph=True`，该项通过表示默认 runner 的模型级执行完成，不代表整步训练捕获为一张图。

[环境记录](../../../artifacts/cicd/four_model_matrix_20261020/environment.json)、
[逐场景状态、命令和 CSV](../../../artifacts/cicd/four_model_matrix_20261020/matrix.json)。

| 模型 | 静态推理性能 | 静态推理精度 | 静态训练性能 | 静态训练精度 | 动态推理性能 | 动态推理精度 | 动态训练性能 | 动态训练精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlbertForMaskedLM | 通过 | 通过 | 编译失败 | 编译失败 | 通过 | 通过 | 编译失败 | 编译失败 |
| GPT2ForSequenceClassification | 通过 | 精度失败 | 通过 | 通过 | 通过 | 精度失败 | 通过 | 通过 |
| nfnet_l0 | 通过 | 通过 | 通过 | 通过 | 编译失败 | 编译失败 | 编译失败 | 编译失败 |
| BERT_pytorch | 通过 | 精度失败 | 通过 | 精度失败 | 通过 | 精度失败 | 通过 | 精度失败 |

## Albert：load 排布与 tiling 搜索使工作集超过 UB

四个训练场景的致命 kernel 都在计算 12 层 LayerNorm 的参数梯度，融合了 81 个输入 load。
UB 是设备核内的 Unified Buffer，容量固定；这里的 UB overflow 不是主机内存 OOM。
BiSheng 报出资源溢出，不能仅凭这个报错认定下层编译器存在语义错误。

| 场景 | 致命 kernel 后缀 | 原始选中配置之一 | 原始结果 | 定界实验 |
| --- | --- | --- | --- | --- |
| 静态训练性能 | `per_*backwar_14` | X=1，R=2048，multibuffer=False | 需要 6,030,080 bits，UB 仅 2,031,616 bits | 相同配置，仅延后 load 即编译通过 |
| 静态训练精度 | `per_*backwar_11` | X=16，R=512，multibuffer=False | UB overflow | 原始 DSL 补充 X=1 可编译；相同 X=16 延后 load 也可编译 |
| 动态训练性能 | `red_*backwar_14` | X=80，R=128，multibuffer=False | 需要 34,177,024 bits | 原始 DSL 补充 X=1 可编译；X=80 仅延后 load 仍失败 |
| 动态训练精度 | `per_*backwar_11` | X=8，R=512，multibuffer=False | UB overflow | 原始 DSL 补充 X=1 可编译；相同 X=8 延后 load 也可编译 |

静态性能 kernel 与此前抽取版本去掉 decorator 后 AST 一致，因此是已知 load 排布问题的再次复现。
本轮仍重新执行了原始/诊断配置，未以历史结果替代本轮验证。
初版捕获文件名没有包含 constexpr，因而同源码、同编译选项的不同 tile 记录会覆盖，
现有 Albert 捕获仅代表各编译选项最后保存的失败配置，不能据文件数统计全部候选数。
捕获脚本已修正 key，加入 signature/constants/attrs；后续新场景会分别保存各 tile。

安装版本中有两个直接相关的问题点：

1. `codegen/triton.py` 的 `load()` 将普通 load 写入 `self.loads`；`write_pointwise()` 统一先输出
   loads，再输出 compute。多个独立梯度贡献融合后，会同时维持大量输入块存活。
2. `runtime/tile_generator.py:43` 将 `input_ptr_num` 截断到最多 3，再以该数估算 tile 容量。
   它不能代表本例 81 个输入与多组 accumulator 的存活工作集，模型搜索未找到可编译配置；固定小 tile 的直接编译提供了可行的反例。

结论：已在 **Inductor NPU 适配层** 定位到 load/compute 排布和资源估算、候选覆盖的问题。
修复应根据依赖缩短存活区间，并让 tiling 估算考虑实际工作集。动态性能实验也说明，仅移动 load
并不能使任意大 tile 变得合法。没有证据把这些资源拒绝直接归为 TA/NPUIR 的语义 bug。

四个诊断配置均已执行 kernel，并用同一输入的 eager 参数梯度比较通过，最大绝对误差不超过 `3.67e-4`
（`atol=2e-3, rtol=1e-3`）。这验证了诊断改动没有破坏原始归约语义，尚未替代整模型训练修复验证。

[原始与诊断编译用例](../../../scripts/repro/cicd/four_model_matrix_20261020/AlbertForMaskedLM/)、
[动态配置对照](../../../artifacts/cicd/four_model_matrix_20261020/analysis/Albert_configs/results.json)、
[安装版本源码片段](../../../artifacts/cicd/four_model_matrix_20261020/installed_sources/triton.py.txt)、
[tiling 估算](../../../artifacts/cicd/four_model_matrix_20261020/installed_sources/tile_generator.py.txt)。

## GPT2、BERT 前向：Inductor 方差 mask 和 TA 间接 load 两个独立问题

历史 `bench_acc_nodata` 中的 GPT2 embedding + LayerNorm、BERT embedding + sample-std normalization
用例与本轮生成代码一致。历史目录没有输入数据；复现使用固定 seed 构造输入，不依赖缺失的 `data.pth`。

### Inductor 在 persistent reduction 中丢失尾部 mask

逻辑 hidden size 为 768，生成代码的 reduction block 为 1024，因此多出 256 个 lane。
即便 masked load 正确将这些 lane 填为 0，减去均值后它们也成为 `-mean`，平方后不再为 0：

```python
mean = tl.sum(value, axis) / 768
centered = value - mean
squared = centered * centered
variance_sum = tl.sum(squared, axis)  # 将额外 256 个 mean² 也累加
```

正确的方差归约应对 `squared` 应用 reduction mask。GPT2 的分母为 768、epsilon 为 `1e-5`；
BERT 使用 sample variance，分母为 767，且 epsilon 是在 sqrt 之后加 `1e-6`，复现保留了这一差别。

安装版本 `codegen/triton.py` 中，`reduction()` 对单轴 persistent reduction 排除了 r-axis mask，
`filter_masks()` 也跳过 persistent reduction axis，之后直接使用 `masked_value = value`。
该假设无法保证非线性计算后的 padded lane 仍是归约单位元，属于 **Inductor codegen 问题**。

### TA 把 masked indirect load 转换成无条件读取

补方差 mask 后，默认 SIMD/template-SIMT 路径仍错误；中间值比较发现均值也错。
进一步把间接 load 的全部 1024 个 lane 写出，发现 padded lane 没有返回 `other=0`。

独立小用例仅含 token index load、masked table load 和 store。有效元素设为 3，尾部底层存储设为 9：

| 编译路径 | 有效 768 个 lane | 额外 256 个 lane | 期望 |
| --- | --- | --- | --- |
| `simd_simt_template` | 3 | **9** | 0 |
| `simt_only` | 3 | 0 | 0 |

[最小复现](../../../scripts/repro/cicd/four_model_matrix_20261020/masked_indirect_load.py) 不调用 Inductor。
[TTIR](../../../artifacts/cicd/four_model_matrix_20261020/masked_indirect_load/tail_tmp2.ttir.txt) 保留了 mask 和零填充值，
而 [ttadapter IR](../../../artifacts/cicd/four_model_matrix_20261020/masked_indirect_load/tail_tmp2.ttadapter.txt)
已变成遍历全部 1024 个元素的 `memref.load`，没有 mask 分支。这是在进入 BiSheng 后续编译前就已发生的
**Triton-Ascend lowering 语义错误**，不应归到 Inductor 或仅归因于后续 NPUIR/BiSheng 优化。

本地 TA 源码对应 `BlockDataParser::rewriteAddPtrToUnstrucMemAcc()` 和
`LoadConverter::continueModifyFromAddPtrConverter()`：后者构造无条件 `memref::LoadOp`，没有处理
原始 load 的 mask/other。源码版本对应关系仍需与实际安装的 TA 构建核对；上述阶段定界直接由运行产物证明。

### 对照验证

- 原始 DSL 在纯 SIMT 路径仍失败；仅补方差 mask 后纯 SIMT 通过，证明 Inductor 问题独立存在。
- 默认路径中仅补方差 mask 仍失败；均值与方差均显式 mask 后通过，证明还需要处理 TA 间接 load。
- GPT2 静态、动态推理的完整前向：相同 seed、相同浮点输入、合法三角 causal mask，原始 26 个输出
  中 24 个 `past_key_values` 不一致；仅修改 kernel 0 和 3 的归约 mask 后，26 个输出全部通过。
- BERT 静态训练前向：原始保存给 backward 的中间输出存在差异；同一输入补 mask 后 264 个输出全部通过。
- BERT 静态模型推理精度测试：只注入上述 mask 修改后，5 步测试由 `fail_accuracy` 变为 `pass_accuracy`。
  同样修改尚不能使训练通过，训练还有下一节的问题。
- 注入已确认的 DSL 修改后，GPT2 静态/动态推理，以及 BERT 静态/动态推理和训练，
  六个原始失败 accuracy 场景的 5 步模型级复测均变为 `pass_accuracy`。

完整前向比较使用合成参数和固定 token，证明的是生成图对同一输入的行为；它不能替代原始模型级精度测试。
这些实验没有替换安装目录中的框架文件，也没有把诊断通过计入原始矩阵通过数。

[原始/诊断单 kernel 脚本](../../../scripts/cicd-four-model-matrix-20261020.md)、
[GPT2 静态原始比较](../../../artifacts/cicd/four_model_matrix_20261020/analysis/GPT2_s_infer_acc/triangular_comparison.json)、
[GPT2 静态补 mask 比较](../../../artifacts/cicd/four_model_matrix_20261020/analysis/GPT2_s_infer_acc_both_masks/comparison.json)、
[GPT2 动态补 mask 比较](../../../artifacts/cicd/four_model_matrix_20261020/analysis/GPT2_d_infer_acc_both_masks/comparison.json)。

## BERT 静态 backward：归约误包含应保留的通道轴

本轮 `triton_per_fused_clone_matmul_backward_sum_tra_11` 与历史抽取用例有相同错误。
输入 `[48,128,64]` 先按 `[batch=4,head=12,seq=128,channel=64]` 重排，输出 `[512,768]`，
再沿 batch/seq 求和，得到每个 head/channel 的 bias gradient。

生成的 load 结果布局为 `[R2=4,R3=128,X5]`，其中 X5 是必须保留的 channel：

```python
tmp1 = tl.reshape(tmp0, [R2BLOCK_SUB * R3BLOCK_SUB * X5BLOCK_SUB])
tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
```

它把 X5 也展平求和，随后把同一个 scalar 存入一组 channel。tile=1 时掩盖错误，tile=8 时
单 kernel 的 sum 最大绝对误差为 187.4362；transpose 输出仍然正确。
只把 reshape 改为 `[R2BLOCK_SUB * R3BLOCK_SUB, X5BLOCK_SUB]` 并沿第 0 维求和后，两个 tile 都通过，
sum 最大绝对误差约 `9.54e-6`。

仅注入前向 mask 时，模型静态训练仍为 `fail_accuracy`；再注入上述 backward 归约修正后，
原始默认 batch=4、5 步训练精度测试变为 `pass_accuracy`。动态训练日志仅触发前向 mask 修改，已能使模型精度通过。对应动态 backward 使用 `red_*_12` / `red_*_13`：
load/accumulator 形状为 `[XBLOCK,R0_BLOCK]`，最后 `tl.sum(...,1)`，通道轴保留，
没有静态 kernel 11 的 flatten 错误。

[backward 单 kernel 验证](../../../artifacts/cicd/four_model_matrix_20261020/bert_backward_validation.json)、
[模型级静态复测](../../../artifacts/cicd/four_model_matrix_20261020/bert_model_all_fixes_results.json)、
[动态 BERT 和 GPT2 模型复测](../../../artifacts/cicd/four_model_matrix_20261020/model_diagnostic_results.json)。

问题点位于安装版本 `codegen/kernel_analysis.py` 的 `ReductionAnalysis.dense_size_str()`：
多轴 reduction 被判断为 non-contiguous 时，它将所有 dense size 相乘；`analyze_reduction_dim()` 又固定返回 0。
随后 `codegen/triton.py::reduction()` 按这个形状生成错误归约。这是 **Inductor 生成的 DSL 已经错误**，
不需要下层 miscompile 才能解释结果。通用修复必须保留所有非归约轴，并按实际布局组合、移动 reduction axes。

[原始复现](../../../scripts/repro/cicd/four_model_matrix_20261020/bert_backward_original.py)、
[仅修归约的对照](../../../scripts/repro/cicd/four_model_matrix_20261020/bert_backward_fixed.py)、
[安装版本 ReductionAnalysis](../../../artifacts/cicd/four_model_matrix_20261020/installed_sources/kernel_analysis.py.txt)。

## NFNet 动态场景：pooling 的 tiling 搜索没有覆盖可编译的小 tile

四个动态场景都在前向 average-pool 编译阶段失败。推理对应
`triton_poi_fused__native_batch_norm_legit_avg_pool2d_convolution_view_21`，训练对应
`triton_poi_fused_avg_pool2d_15`。性能和精度各自捕获的同名 kernel body 的 AST 一致。
推理 kernel 名虽含 convolution，实际 DSL 只做四次相邻元素 load、相加、除以 4 和 store，
即 stride=2、kernel=2 的 average-pool，没有卷积计算。

将 `p = (ks2 - 1) // 4 + 1` 代入原始地址式，四个输入位置为
`x2 * p² + 2 * row * p + 2 * col + {0, 1, p, p+1}`，符合 2×2 pooling 语义。
`ks0` 是输出宽度，`ks1` 是输出空间元素数；`x2` 合并 batch/channel。

| kernel | 原始最小 XBLOCK=4096，multibuffer=False | 仅缩小 XBLOCK |
| --- | --- | --- |
| 推理 `_21` | 需要 2,097,152 bits，UB 仅 1,769,472 bits | 2048、1024、512 都编译通过；与 eager 最大绝对误差为 0 |
| 训练 `_15` | 需要 2,883,584 bits，UB 仅 1,769,472 bits | 2048、1024、512 都编译通过；与 eager 最大绝对误差为 0 |

推理输入使用固定 seed 的 `[2,256,72,72]`，运行时参数为 `(36,1296,288)`；
训练为 `[2,256,56,56]`，运行时参数为 `(28,784,224)`。两者均保留连续布局和原始动态整数参数，
用同一输入的 `torch.nn.functional.avg_pool2d(data, 2)` 比较，`rtol=atol=1e-6`。
原始源码不变，只有 constexpr tile 缩小，因此直接证明原始搜索遗漏了可行配置。

安装版本的候选生成记录给出以下事实：

1. `TileGenerator.__init__()` 按两个 pointer 参数估算容量，`max_numel_threshold=32768`，
   `stop_numel=4096`。pointer 参数数并不代表四次 load、地址计算和中间值的实际存活工作集。
2. `descend_tiling_axis()` 到 4096 就停止，template 路径最终只有 XBLOCK=4096/8192/16384/32768。
3. `triton_config_npu_index()` 先切换到 SIMT_ONLY 搜索；`add_to_configs()` 又拒绝超过 8 KiB 的
   SIMT tile，即 FP32 最多 2048。搜索在 4096 停止，所以本例实际生成 **0 个 SIMT 候选**。
   `set_kernel_type()` 只更新类型，没有同步重算构造时的资源阈值。
4. fallback 的 `_remap_fallback_block_subs()` 把 `XBLOCK_SUB` 改为该一维 kernel 的 constexpr
   `XBLOCK`；最终失败的是这些过大的真实向量 tile。

使用 `inductor-kernel-compile-debug` skill 对单条推理 kernel 重放：40 次编译尝试全部失败，
最小 tile 仍为 4096，均使用 `simd_simt_template`。其中 journal id=9 是
XBLOCK=4096、multibuffer=False、未启用 VF fusion 的固定配置。
skill 捕获保留了 direct_compile 元数据；自动导出器尚不支持 `div_floor_integer`，
独立用例将已核对的安装版本 helper 原样内联，保留 floor division 的负数修正规则。
最终 compile-only 推理用例为 46 行，包含完整 kernel、一个 helper 和一次 `triton.compile`，没有 autotune。

结论：这是 **Inductor NPU tiling 的资源估算和候选覆盖问题**。下层对超 UB 配置的拒绝合理，
同一条 DSL 在同一编译路径缩小 tile 后编译、数值均通过，当前没有证据归为 TA/NPUIR 语义 bug。
通用修复应使资源估算、停止条件和 kernel type 的约束一致，并在资源拒绝后覆盖更小 tile；
不应将这个模型的 XBLOCK=2048 硬编码为框架规则。

诊断脚本只给动态 pooling 增加 XBLOCK=2048，保留原始候选，未修改安装目录。
动态推理 performance 已完成并输出正数 speedup，默认 batch=8 的 5 步 accuracy 也变为 `pass_accuracy`。
原始矩阵四项失败保持不变，训练场景目前验证到单 kernel，未声称整模型训练已修复。

[四场景原始 kernel 与失败配置](../../../artifacts/cicd/four_model_matrix_20261020/kernel_sources/nfnet_l0/)、
[候选生成记录](../../../artifacts/cicd/four_model_matrix_20261020/analysis/NFNet_d_infer_perf/tiling_generation.json)、
[skill 编译记录摘要](../../../artifacts/cicd/four_model_matrix_20261020/analysis/NFNet_d_infer_perf/skill_capture_summary.json)、
[推理数值对照](../../../artifacts/cicd/four_model_matrix_20261020/nfnet_direct/nfnet_pool_validation_fixed_grid.json)、
[训练数值对照](../../../artifacts/cicd/four_model_matrix_20261020/nfnet_train_direct/nfnet_pool_validation_fixed_grid.json)、
[模型级诊断结果](../../../artifacts/cicd/four_model_matrix_20261020/nfnet_model_diagnostic_results.json)、
[安装版本 tiling 源码](../../../artifacts/cicd/four_model_matrix_20261020/installed_sources/tile_generator.py.txt)、
[runtime 配置源码](../../../artifacts/cicd/four_model_matrix_20261020/installed_sources/triton_heuristics.py.txt)。

## 诊断验证的范围

- 原始 32 项已全部执行完成；诊断通过未覆盖原始矩阵的失败状态。
- Albert 四个诊断配置完成单 kernel 数值验证；NFNet 动态训练的两个失败完成 pooling 单 kernel 定界，
  没有将这些结果外推为整模型训练修复通过。
- GPT2/BERT 六个原始精度失败场景，以及 NFNet 动态推理 performance/accuracy，均完成模型级诊断复测。
- 通用框架修复不在本轮实现范围内；文中的 DSL 修改和附加 tile 仅用于验证问题点。
- 动态测试沿用 runner 的 `--dynamic-shapes` 场景，没有额外验证多组运行时尺寸。
