---
title: CICD 静态 shape 模型编译失败定界（2026-10-19）
published: false
---

# CICD 静态 shape 模型编译失败定界

执行环境为 A5 / Ascend950PR_9579 的 Docker 容器，使用 `pta_213_cicd`。
框架版本、安装提交见 [environment.json](../../../artifacts/cicd/compile_failures_20261019/environment.json)。
单 kernel 复现与验证命令见 [脚本说明](../../../scripts/repro/cicd/compile_failures_20261019/README.md)。

后续 Swin 全场景测试在 batch 8 的静态、动态推理精度场景发现真实致命编译失败，
见 [2026-10-20 全场景记录](swin-full-matrix-20261020.md)。下文 Swin 通过结论限于本轮静态推理性能 batch 128。

UB（Unified Buffer）是计算核上的片内缓冲区。本文中的 UB overflow 是 kernel 的编译资源分配失败，
与主机内存 OOM、NPU HBM 分配失败不同；降低 compile threads 不会缩小单个 kernel 的 UB 工作集。

## swin_base_patch4_window7_224：本轮通过，历史明确错误来自 runner 模型表

使用 timm_models、静态 shape、FP32 推理，默认 batch 128，compile threads 2。
当前整模型完成编译与性能测试，捕获 1 个图、0 次 graph break；CSV 记录编译耗时
1242.324152 秒，speedup 为 0.756102。本次使用独立 Inductor cache，Triton cache 为环境共享缓存，
因此该耗时不代表所有层级均为空缓存的冷编译。性能模式没有执行完整模型 eager 精度对照。
[运行结果](../../../artifacts/cicd/compile_failures_20261019/swin_base_patch4_window7_224/model_results.csv)，
[完整日志](../../../artifacts/cicd/compile_failures_20261019/swin_base_patch4_window7_224/model.txt)。

历史第一份明确异常为 `KeyError: 'swin_base_patch4_window7_224'`，发生在
`benchmarks/torchbench/timm_models.py:176` 的 `TIMM_MODELS[model_name]`，尚未进入编译。
该字典在 86–97 行从 `timm_models_list.txt` 构建，故属于 runner 模型表缺项，与 Inductor 无关。
当前模型表已有 `swin_base_patch4_window7_224 128`，本次没有修改该文件。
[历史异常](../../../artifacts/cicd/compile_failures_20261019/swin_base_patch4_window7_224/historical_keyerror.txt)，
[源码摘录](../../../artifacts/cicd/compile_failures_20261019/source_evidence.md)。

历史后两份补充模型表后的日志只停在 scheduler `Error in codegen` 和 `low_dims is null`，
没有最终 traceback 或完整终态。本轮也出现相同日志，随后继续生成 kernel 并最终通过，
所以不能仅据这两类日志将历史运行定为 Inductor 编译失败。当前版本不同，也不能倒推历史环境必定通过。

为了保留可研究的失败配置，额外抽取了 kernel
`triton_red_fused__unsafe_view_add_addmm_clone_native_layer_norm_permute_roll_view_19` 的一个配置：

```text
XBLOCK=3072, R0_BLOCK=2, compile_mode=simd_simt_template, multibuffer=False
ConvertLinalgIRToBinary
ub overflow, requires 4031488 bits while 1769472 bits available!
```

独立脚本和脱离原 capture 的便携回放均复现这一异常。但该 kernel 的 294 个候选中有
128 个编译成功、166 个失败，整体预编译成功；它不是模型终止错误。
`runtime/triton_heuristics.py:910–963` 捕获单个候选异常，仅在所有候选和 VF retry 都无成功结果时
抛出 `NoTritonConfigsError`。完整模型生成的 71 个 kernel 合计 50,520 个候选，解释了较长的搜索过程，
不能把长时间无终态直接视为编译失败。本轮未观察到 OOM。

结论：本轮未复现 Swin 编译失败，无法据现有证据认定其存在 Inductor 缺陷；历史可明确归因的
`KeyError` 是 runner 配置问题。已保留失败候选用于后续编译器研究，并明确标注为非致命配置。

## AlbertForMaskedLM：Inductor 的 load 排布扩大存活工作集

场景为 huggingface、静态 shape、FP32 训练。整模型使用 `TORCHINDUCTOR_COMPILE_THREADS=1` 重现：

```text
No valid triton configs for kernel triton_per_fused_add_native_layer_norm_backwar_14
ConvertLinalgIRToBinary
ub overflow, requires 37225216 bits while 2031616 bits available!
```

目标 kernel 的 16 个编译候选全部失败，包含 VF retry。最小非归约 tile 为 `X0BLOCK_SUB=1`，
归约轴固定为 `R1BLOCK_SUB=2048`。独立导出选择该最小 tile，仍报：

```text
ub overflow, requires 6030080 bits while 2031616 bits available!
```

[完整 generated kernel](../../../artifacts/cicd/compile_failures_20261019/AlbertForMaskedLM/kernel.py)
融合了 12 组 LayerNorm backward 的参数梯度贡献。矩阵地址为 `x0 + 4096*r1`，
统计量地址为 `r1`；每组计算两个长度为 4096 的输出向量，最后按图中的顺序相加。
开头连续发出 81 个输入 load，再统一发出 compute 和 reduction：

```python
tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
# ...直到 tmp191 的输入加载...
tmp3 = tmp1 - tmp2
tmp5 = tmp3 * tmp4
tmp6 = tmp0 * tmp5
tmp8 = tl.sum(tmp6, 0).reshape(1, X0BLOCK_SUB)
```

定界实验只调整语句顺序：把每个 load 移到首次使用之前，保留全部输入、计算、tile、归约轴和
最终加法顺序。该版本编译成功，实际运行与 eager 对照通过，最大绝对误差分别为
`2.74658203125e-4` 和 `1.8310546875e-4`。
[验证结果](../../../artifacts/cicd/compile_failures_20261019/validation_results.json)。

源码位置为安装版本 `torch_npu/_inductor/codegen/triton.py`：

- `load()`，4533–4550 行：普通 load 写入独立的 `self.loads` 缓冲区。
- `codegen_body()` 中的 `write_pointwise()`，2654–2667 行：先整体输出 loads，再输出 compute。
- `TileGenerator.__init__()`，`runtime/tile_generator.py` 39 行：输入数按最多 3 个估算，
  没有表达这个融合 kernel 的实际存活输入和中间值数量。

见 [源码快照摘录](../../../artifacts/cicd/compile_failures_20261019/source_evidence.md)。

结论：在 Inductor 层已经定位到可操作的问题点——大量独立贡献融合后仍统一前置 load，扩大了
向量后端看到的存活工作集；资源估算未覆盖该形态。BiShengIR 报错是最终表现。
仅改变生成语句排布即可编译并保持结果，为此归因提供了对照证据；不据此排除后端调度也可进一步优化。
后续通用修复应围绕 load/compute 的依赖与存活区间处理，或准确约束融合资源，不能只特判模型名。

## phlippe_densenet：Inductor 轴元数据与 tiling 候选不匹配

场景为 torchbench、静态 shape、FP32 训练，整模型以 compile threads 1 重现：

```text
No valid triton configs for kernel triton_poi_fused__native_batch_norm_legit_no_t_14
ConvertLinalgIRToBinary
ub overflow, requires 78249984 bits while 1769472 bits available!
```

目标 kernel 的 72 个候选全部失败，包含 VF retry。`Y0BLOCK_SUB` 的候选集合为
`[512, 1024, 2048, 2304, 2344, 3072, 8192, 16384]`；不存在 128 或更小的候选。
独立导出选择 512，仍报 `requires 2445312 bits while 1769472 bits available`。

[完整 generated kernel](../../../artifacts/cicd/compile_failures_20261019/phlippe_densenet/kernel.py)
执行 avg_pool2d、推理语义 BatchNorm、ReLU 和 concat 输出写入。虽然模型场景为训练，
这个具体子图中的 BatchNorm 操作为 `_native_batch_norm_legit_no_training`。

存在可以直接核对的元数据矛盾：

```text
axis_names: ['y0', 'x1']
split_axis: [0, 1]
runtime_block_arg_names: ('Y4BLOCK', 'Y5BLOCK')
```

kernel 参数保留 `Y4BLOCK, Y5BLOCK`，body 却不使用它们，也没有 `tl.program_id`，
而是循环覆盖整个 `y0_numel=131072`；`X1BLOCK_SUB` 在 body 固定为 16。
候选配置却尝试把 `X1BLOCK_SUB` 设为 1、2、6，这些值不会改变实际 DSL 的固定 16。
因此候选生成的 tile 估算与实际 DSL 不一致。

源代码涉及安装版本 `torch_npu/_inductor/codegen/triton.py`：

- `_apply_guarded_expansions()`，3780 行起：对同一维度的不同展开方式应用统一轴转换。
- `_remove_expansion_dims_from_axes()`，3681 行起：移除、重排 `tiling_axis` 和 `sorted_axis`，
  清空 `low_dims`，但没有同步重建 `split_axis`。
- `create_inductor_meta()`，1888–1930 行：`axis_names` 来自 `sorted_axis`，
  `split_axis` 和 runtime block 名来自另一份 `split_axis` 列表，遗留状态随之进入元数据。
- `add_autotune_args()`，2026 行起：runtime block 参数也直接来自该 split 列表。
- `runtime/tile_generator.py` 的 `calculate_config_numel()`、`add_to_configs()` 和
  `descend_split_tiling()`：按元数据候选计算 tile 大小和停止条件，未基于实际 DSL 固定维度重新校验。

定界实验保留原 DSL 和原编译选项，仅把 `Y0BLOCK_SUB` 改为 128、64、32，三者均编译成功。
128 的执行结果与 eager 的 avg_pool2d / BatchNorm / ReLU 对照通过，包含非连续 concat 输出视图；
其 storage offset 为 4096，stride 为 `(20480, 256, 16, 1)`，未写入的前 16 个 channel 保持 NaN。

结论：属于 Inductor 的轴状态一致性及 tiling 候选生成问题，已经定位到统一轴转换后 split 状态未同步
和实际固定维度未被候选资源估算正确表达两个位置。最终报错来自后端 UB 分配。
完整修复应在确定最终轴结构后统一重建 split/tiling/参数元数据，并保证资源估算对应真实 body；
本次缩小 tile 是定界实验，尚未作为通用修复提交。

## cm3leon_generate：已抽取和复现 IR 转换失败

场景为 torchbench、静态 shape、FP32 推理。整模型使用 compile threads 1 重新复现：

```text
No valid triton configs for kernel triton_per_fused__to_copy_add_embedding_mul_na_1
ConvertTritonIRToLinalgIR
PassManager::run failed
```

[完整 kernel](../../../artifacts/cicd/compile_failures_20261019/cm3leon_generate/kernel.py)
包含 embedding、加法及 LayerNorm。归约长度为 1536，body 固定 tile 为 2048。
原候选和 VF retry 共 2 个配置均失败，导出的单 kernel 程序复现同一 IR 转换异常。
本任务不要求对该模型进一步定责；没有把外层 `InductorError` 直接当作 Inductor 归因结论。

## 验证边界

其余三个历史缓存的 generated body 与当前整模型重新生成的 body 经 AST 比较一致。
本轮保留当前整模型失败日志及单 kernel 失败日志，配置见各模型的 `configs.json`、`journal.jsonl`。
所有模型保持默认 batch 和静态 shape；未通过修改 batch、dtype 或 fallback 来改变目标用例。

诊断版本只验证了目标 kernel 的结果与资源行为，没有运行通用 codegen 修复后的完整模型回归。
未修改 torch_npu 安装目录或仓库业务源码。

## 整模型复现命令

在目标 Docker 工作目录激活 `pta_213_cicd` 并加载既有 `env.sh` 后，使用 runner 目录：

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
cd "$TORCHBENCH_RUNNER_DIR"
python huggingface.py --performance --repeat 1 --iterations 1 --training --float32 \
  --backend inductor --npu-backend triton --disable-aclgraph --only AlbertForMaskedLM --output albert.csv
python torchbench.py --performance --repeat 1 --iterations 1 --training --float32 \
  --backend inductor --npu-backend triton --disable-aclgraph --only phlippe_densenet --output densenet.csv
python torchbench.py --performance --repeat 1 --iterations 1 --inference --float32 \
  --backend inductor --npu-backend triton --disable-aclgraph --only cm3leon_generate --output cm3leon.csv
TORCHINDUCTOR_COMPILE_THREADS=2 python timm_models.py --performance --repeat 1 --iterations 1 \
  --inference --float32 --backend inductor --npu-backend triton --disable-aclgraph \
  --only swin_base_patch4_window7_224 --output swin.csv
```

采集时额外启用 `TORCH_COMPILE_DEBUG=1`；前三个模型用编译追踪工具包裹 runner，
只筛选目标 kernel。没有传 `--dynamic-shapes`，没有覆盖默认 batch。
