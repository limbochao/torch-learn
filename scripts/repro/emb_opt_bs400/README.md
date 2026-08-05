# emb_opt NPU repro

本目录保存 embedding 模型的 NPU `torch.compile` 复现材料。当前主脚本
`fx_graph_runable_npu.py` 基于原始 BS200 NPU runnable 改造，支持调整 batch size、静态编译、普通动态编译、
symbolic group autotune 和 NPU profile。

原模型依赖的 `qianchuan_triton` 算子在本地没有原始实现。主脚本通过 `torch.library` 注册同名的确定性 NPU
占位算子。`ascend_triton` 的 4 个算子则仅在运行环境没有原始注册时使用占位实现。sequence concat 相关实现
参考 CUDA 产物，使用 offsets 按 batch 交错复制两个输入；flash attention 直接复制 query。它们保留
external/fallback 边界及输出 shape、dtype、device，但没有复现原始自定义 kernel 的性能。因此该脚本用于
分析 Inductor 生成 kernel 的 dynamic/group 行为，不能用于整网正确性验证，也不能把占位算子耗时当作原模型
耗时。由于原始 storage 未保存，脚本会将 `arg147` 的 7 组序列长度替换为均匀分配且总和为当前 batch size
的确定性值，避免随机输入破坏 sequence concat 后续的 shape guard。

## 运行整图

直接执行脚本，通过 `--execution` 选择编译方式，通过 `--bs` 指定输入 batch size：

```bash
# dynamic=False，不 mark 维度，不开启 group
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution static --bs 200

# dynamic=None，mark 所有 batch 输入维度，不开启 group
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution dynamic --bs 200

# dynamic=None，mark 所有 batch 输入维度，开启 symbolic group autotune
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution group --bs 200
```

通过 `--bs-sequence` 可以在一个进程中按顺序运行多个 batch size。dynamic/group 使用第一个 BS 编译，后续 BS
复用同一个 compiled callable；static 为每个 BS 分别执行静态编译。`--bs-sequence` 会覆盖 `--bs`：

```bash
# 首 shape 为 BS400，随后复用符号化 kernel 运行 BS200 和 BS100
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py \
    --execution group --bs-sequence 400,200,100
```

不同执行方式可以复用同一个 `RUN_ID`：

```bash
RUN_ID=emb_opt_01 python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution static --bs 200
RUN_ID=emb_opt_01 python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution dynamic --bs 200
RUN_ID=emb_opt_01 python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution group --bs 200
```

单 BS 的默认结果目录保持不变：

```text
prof_log/fx_graph_runable_npu/<RUN_ID>/bs_<BS>/<execution>/
```

BS sequence 的结果目录为：

```text
prof_log/fx_graph_runable_npu/<RUN_ID>/bs_sequence_<BS0>-<BS1>-.../<execution>/
    step_000_bs_<BS0>/profiles/
    step_001_bs_<BS1>/profiles/
    ...
```

其中包含：

- `profiles/`：torch_npu profiler 原始结果。
- `performance.csv`：每个 sequence step 一行的 device 侧汇总耗时，包括 `sequence_index`、`compile_bs`、
  `runtime_bs` 和 `bs_sequence`。
- `torch_compile_debug/`：Inductor debug trace，包括生成成功时的 `output_code.py`。

Inductor、Triton 编译 cache，以及 autotune 使用当前工作目录生成的 `profile_result/`、`profile_results/`
均写入本次进程独立的系统临时目录，退出时自动删除，不再保存到 `prof_log`。

可用环境变量为 `PROFILE_ROOT`、`RUN_ID`、`WARMUP`、`ACTIVE`、`REPEAT`。profile 默认配置为
`WARMUP=1 ACTIVE=10 REPEAT=1`。

## 废弃实现

`deprecated/` 保存此前从 CUDA runnable 改造成 NPU 的版本及其自定义 Triton kernel 适配。该版本与原始
NPU 图的融合和 external op 边界存在差异，只保留用于历史对照，不再作为测试入口。

## 抽取单个 Triton kernel

`scripts/tools/extract_triton_kernel.py` 从 Inductor `output_code.py` 中抽取指定 kernel 的定义和首次调用所需参数。
若同一 kernel 被调用多次，只采用源码中第一次 `.run(...)` 的参数：

```bash
python scripts/tools/extract_triton_kernel.py \
  /home/qc/Repos/Ascend/tmp/emb_opt_bs400/ouput_code_npu.py \
  triton_poi_fused_amax_gt_sign_view_0 \
  -o scripts/repro/emb_opt_bs400/kernels/triton_poi_fused_amax_gt_sign_view_0.py
```

工具会保留生成文件的公共导入、kernel 前的 FX/ATen 注释和 Triton DSL，并从
`benchmark_compiled_module` 读取原始输入构造，再反向收集首次调用依赖的 shape、buffer、numel 和 stream。
