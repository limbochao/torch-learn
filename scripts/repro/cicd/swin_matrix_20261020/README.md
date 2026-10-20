# Swin 全场景验证与失败 kernel

模型为 `swin_base_patch4_window7_224`，suite 为 `timm_models`。
验证覆盖静态/动态 shape × 推理/训练 × 性能/精度，共 8 个场景。
使用 A5 / Ascend950PR_9579、`pta_213_cicd`、FP32、Inductor + Triton，关闭 aclgraph。
环境版本见 [environment.json](../../../../artifacts/cicd/swin_matrix_20261020/environment.json)。

本轮 8 项均已结束：4 项性能运行通过，4 项精度失败。
推理精度均在 kernel 9 编译失败，静态训练精度在 kernel 47 编译失败，动态训练精度为实际数值不一致。
四项性能 speedup 均低于 1，两项训练性能各有 9 次 graph break。
完整 CSV、时延与定界结论见 [问题记录](../../../../docs/issue-logs/cicd/swin-full-matrix-20261020.md)。

## 重跑矩阵

在目标 Docker 的 TorchBench 工作目录内激活环境并加载 CANN 后执行：

```bash
CASE_DIR=/path/to/torch-learn/scripts/repro/cicd/swin_matrix_20261020
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/swin_matrix_cache"
export TORCH_COMPILE_DEBUG=1
bash "$CASE_DIR/run_torchbench_matrix.sh" --models "$CASE_DIR/models.csv" \
  --output-dir "$PWD/swin_matrix_results"
```

`run_torchbench_matrix.sh` 是本次运行的脚本快照，只将默认目录改为相对当前工作目录。
可通过 `TORCHBENCH_DIR`、`TORCHBENCH_DATA_PATH` 覆盖路径。
默认性能测试为 batch 128、repeat 1、iterations 1；精度测试由 runner 自动使用 batch 8、iterations 5。
动态场景传 `--dynamic-shapes`；它验证 runner 给定的输入，不代表任意 batch 或分辨率均已覆盖。
可用 `--shape static|dynamic --mode inference|training --metric performance|accuracy` 选择单个场景。

本次测试共用此前 Swin 的 Inductor cache 和环境的 Triton cache，因此编译耗时不表示冷缓存耗时。
训练任务与部分后续场景在不同 NPU 卡上并行，性能数字仅用于记录运行结果。
训练性能场景另有 compile threads 2、同卡复用缓存的补跑；每项最终采用的编译并发见结果 JSON。
精度模式应检查 CSV 中的 `pass_accuracy`；runner 返回 0 而 CSV 为 `fail_to_run` 仍是失败。

## 推理精度编译失败

静态和动态推理精度均失败于 `triton_poi_fused_add_addmm_clone_native_layer__9`。
静态 source、28 个失败候选及 tiling 诊断见
[static_inference_kernel](../../../../artifacts/cicd/swin_matrix_20261020/static_inference_kernel)。

```bash
python "$CASE_DIR/static_inference_kernel/compile_repro.py"
python "$CASE_DIR/probe_kernel.py"
python "$CASE_DIR/validate_tile1.py"
```

第一条独立编译一个原始配置，预期以 UB overflow 非零退出。
第二条比较 `(Y1BLOCK_SUB, Y3BLOCK_SUB)` 为 `(2,8)`、`(1,4)`、`(1,2)`、`(1,1)` 的配置，
只修改 constexpr tile，保留 kernel body 和编译选项；预期前三项失败、最后一项通过。
第三条实际运行 `(1,1)` 配置，与同组随机输入的 eager 结果比较，预期通过。
输入保留 NCHW stride、window reverse 排布和 LayerNorm 统计量，详见脚本。
这些是单 kernel 定界实验，没有修改框架，也不表示完整模型已经修复。

## 训练精度编译失败与地址核算

静态训练精度最终失败于 `triton_per_fused__unsafe_view_clone_native_lay_47`。
source、wrapper 调用、失败候选和对照结果见
[training_accuracy_kernel](../../../../artifacts/cicd/swin_matrix_20261020/training_accuracy_kernel)。

```bash
python "$CASE_DIR/training_accuracy_kernel/compile_repro.py"
python "$CASE_DIR/probe_training_kernel.py"
python "$CASE_DIR/audit_training_index_extent.py"
python "$CASE_DIR/validate_training_variant.py"
```

原始 `X0BLOCK_SUB=512` 配置预期 UB overflow；只修改这个常量后，16、8 仍失败，4、2、1 可以编译。
TileGenerator 将 x0 推断为 no-loop axis，无法缩小到可用 tile，候选为空后又回退到完整的 512。
仅在诊断对象中清除推断的 no-loop axis 可得到 16 个配置，覆盖 1、2、4、8；没有修改安装版本。

第三条只在 CPU 上核算原始 DSL 的输入地址范围：802,816 个元素的输入，最大读取偏移 827,903，
其中 25,088 个元素读取超出逻辑范围。因此原始 kernel 的小 tile 只做编译，没有执行。
第四条同时将 tile 改为 4、将地址中的 `3584*r2` 改为 `3584*(r2%7)`，
再与 eager 的 window reverse 和 LayerNorm 参数梯度计算比较，预期通过。
该实验验证两处诊断修改，不能表示框架通用修复或整模型精度已通过。

## 动态训练精度失败

原始模型编译并运行完成，CSV 为 `fail_accuracy`。前向定界定位到 attention softmax 的归约 mask 缺失：
49 个有效元素补齐到 64 个后，错误地全部参与 max/exp/sum。
安装版本的 `reduction()`、`filter_masks()` 都跳过了 persistent reduction 的 r 轴 mask。
源码及同输入前向对照见
[dynamic_accuracy_kernel](../../../../artifacts/cicd/swin_matrix_20261020/dynamic_accuracy_kernel)。

```bash
python "$CASE_DIR/dynamic_accuracy_kernel/validate_softmax.py"
python "$CASE_DIR/dynamic_accuracy_kernel/probe_reduction_mask_filter.py"
```

第一条直接运行原始 DSL 及仅补 mask 的诊断版本，不调用 Inductor autotuner。
预期原始版本全零输入输出 `1/64`、每行和 `49/64`，随机输入也不匹配 eager；补 mask 后两种输入均通过。
固定独立配置为 sub tile 1、runtime block 16、grid 32，不代表整模型 autotune 选择。
第二条对安装版本的 filter 方法使用独立合成对象，验证 `r3_mask` 被删除；不分配 NPU tensor、不改安装文件。
这些结果定位了生成 DSL 的数值错误，尚未验证完整训练链路修复。

## 训练精度慢编译

已抽取 `triton_per_fused__unsafe_view_clone_native_lay_42` 的 source、TTIR 和 LLVM IR：
[training_kernel](../../../../artifacts/cicd/swin_matrix_20261020/training_kernel)。
静态与动态 source 的函数体相同，axis static metadata 不同。
下列命令直接调用 CANN BiSheng，不运行模型、不分配 NPU 输入：

```bash
ARTIFACT_DIR=/path/to/torch-learn/artifacts/cicd/swin_matrix_20261020
python "$CASE_DIR/compile_training_llvm.py" \
  --ir "$ARTIFACT_DIR/training_kernel/static/kernel.ll" \
  --output-dir "$PWD/swin_llvm_O2" --timeout 60 --trace-passes
```

本环境中独立探针在 60 秒后返回 124，日志最后进入 `HiTPE Stall Cycle Refactor` pass。
`--optimization O0` 可运行优化级别对照；原始 O0 探针也未在 60 秒内完成。
探针的 `PROBE_TIMEOUT` 是人为设置的诊断时间界限，与整模型终态分开记录。

结论与矩阵结果见 [问题记录](../../../../docs/issue-logs/cicd/swin-full-matrix-20261020.md)。
