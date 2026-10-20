# CICD 静态 shape 编译失败 kernel

在 A5 / Ascend950PR_9579、`pta_213_cicd` 环境中复现，日期为 2026-10-19。
未传 `--batch-size`，使用 runner 的模型默认 batch；仅将编译并发降为 1 或 2。

后续 [Swin 全场景测试](../swin_matrix_20261020/README.md) 在 batch 8 推理精度场景发现致命的 kernel 9 编译失败。
本目录 Swin 的通过结果仅覆盖静态推理性能 batch 128。

三个失败模型目录的 `compile_repro.py` 是独立的单 kernel、单配置编译脚本，直接调用 Triton 的
`ASTSource`、`GPUTarget` 和 `triton.compile`。无需模型、数据集、原始 Inductor cache 或 capture 目录。
默认只编译，不分配模型输入，不执行训练/推理。

| 模型 | suite / 场景 | kernel | 独立复现 |
|---|---|---|---|
| swin_base_patch4_window7_224 | timm_models / 静态推理 FP32 | `triton_red_fused__unsafe_view_add_addmm_clone_native_layer_norm_permute_roll_view_19` | 整模型通过；仅保留非致命 UB 溢出候选 |
| AlbertForMaskedLM | huggingface / 静态训练 FP32 | `triton_per_fused_add_native_layer_norm_backwar_14` | UB overflow |
| phlippe_densenet | torchbench / 静态训练 FP32 | `triton_poi_fused__native_batch_norm_legit_no_t_14` | UB overflow |
| cm3leon_generate | torchbench / 静态推理 FP32 | `triton_per_fused__to_copy_add_embedding_mul_na_1` | ConvertTritonIRToLinalgIR 失败 |

Swin 本轮默认 batch 128 的整模型编译和推理通过，因此没有可认定为模型终止原因的失败 kernel。
其目录中的 `candidate_compile_repro.py` 只复现一个被 autotune 正常淘汰的配置；同一 kernel 的
294 个配置中有 128 个编译成功、166 个失败。不能将这个单配置失败当成 Swin 模型编译失败。

## 运行

先进入目标 Docker 工作目录并激活 `pta_213_cicd`、加载 CANN，再执行：

```bash
CASE_DIR=/path/to/torch-learn/scripts/repro/cicd/compile_failures_20261019
python "$CASE_DIR/AlbertForMaskedLM/compile_repro.py"
python "$CASE_DIR/phlippe_densenet/compile_repro.py"
python "$CASE_DIR/cm3leon_generate/compile_repro.py"
python "$CASE_DIR/swin_base_patch4_window7_224/candidate_compile_repro.py"  # 非致命候选
```

在本次记录的环境中，上述四个独立命令预期非零退出并打印表中的异常。未来环境运行成功，表示该配置的
编译问题可能已修复；不能仅据此认定完整模型已通过。

独立编译复现的签名、constexpr、target 和有效 compiler options 都内嵌在文件中。
部分 kernel 的 `*_BLOCK` 是运行时参数，另一些是 constexpr；以脚本内签名和 constants 为准。
`*_BLOCK_SUB` 控制向量 tile 的大小；tile 是单次向量计算处理的数据块。

## 定界实验

```bash
python "$CASE_DIR/validate_variants.py" --model all --output ./validation_results.json
```

该命令运行两个诊断版本并与 eager 比较，预期退出 0：

- Albert：`defer_loads.py` 保留全部 12 组计算、同一输入布局、同一配置和求和顺序，
  仅将 load 语句移动到首次使用前。矩阵输入为 `(2048, 4096)`、stride `(4096, 1)`，
  行统计输入为 `(2048,)`，两个输出均为 `(4096,)`。需要约 2 GB 输入显存。
- phlippe_densenet：保留原 DSL，将 `Y0BLOCK_SUB` 从 512 改为 128；输入为
  `(128, 64, 32, 32)`、stride `(65536, 1024, 32, 1)`。保留 concat 输出视图的
  stride `(20480, 256, 16, 1)` 和 4096 元素 storage offset，检查未写区域保持 NaN。

随机输入使用 seed 0；不保存真实模型输入。Albert 两个输出的最大绝对误差分别为
`0.000274658203125`、`0.00018310546875`，检查阈值为 `atol=0.002, rtol=0.001`；
phlippe_densenet 三个输出均通过 `atol=rtol=2e-5` 检查。

这些是用来验证归因的 DSL 实验；没有修改安装的 torch_npu，也没有完成通用 codegen 修复。

## 证据

- [完整分析](../../../../docs/issue-logs/cicd/cicd-static-compile-failures-20261019.md)
- [环境版本](../../../../artifacts/cicd/compile_failures_20261019/environment.json)
- [原始 kernel、日志和配置](../../../../artifacts/cicd/compile_failures_20261019/)
- [源码证据与原始行号](../../../../artifacts/cicd/compile_failures_20261019/source_evidence.md)

`kernel.py` 保持 generated body 的原始格式，仅去除 backend_hash 和环境路径。
`journal.jsonl` 保存所选失败配置。前三个失败模型的 `configs.json` 汇总整模型目标 kernel 的候选结果；
Swin 的配置汇总来自该 kernel 的独立预编译回放，`candidate_inventory.json` 则来自当前整模型生成的
71 个 kernel，记录总计 50,520 个候选。
没有把其他 kernel 中被 autotune 正常淘汰的配置误记为模型的终止错误。

历史缓存与当前整模型重新生成的三个 kernel body 经 AST 比较一致。当前 torch_npu 版本为
`2.13.0.dev20261019`，与历史日志中 10 月 14 日使用的环境版本有差异；本次三个整模型和独立 kernel
均在当前版本重新验证。

Swin 的完整结果见其 artifacts 目录中的 `model_results.csv`、`model.txt` 和历史 runner `KeyError` 日志。
