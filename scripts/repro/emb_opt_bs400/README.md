# emb_opt_bs400 NPU repro

本目录保存 embedding 模型 `bs=400` 的 NPU `torch.compile` 复现材料：

- `fx_graph_runable_npu.py`：完整 FX graph，支持动态 group 和静态编译，并采集 NPU profile 与 Inductor 生成代码。
- `fx_graph_custom_kernels_npu.py`：原模型中 user autotune Triton kernel 的 NPU 适配与注册。
- `kernels/`：从 Inductor `output_code.py` 独立抽取的 Triton kernel。

## 运行整图

直接执行脚本，通过 `--execution` 选择编译方式：

```bash
# mark_dynamic + dynamic=None，并开启 symbolic group autotune
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution dynamic

# dynamic=False，并关闭 symbolic group autotune
python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution static
```

两次执行使用相同的 `RUN_ID`，结果会分别放在 `<RUN_ID>/dynamic` 和 `<RUN_ID>/static`：

```bash
RUN_ID=emb_opt_bs400_01 python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution dynamic
RUN_ID=emb_opt_bs400_01 python scripts/repro/emb_opt_bs400/fx_graph_runable_npu.py --execution static
```

默认结果根目录为 `prof_log/fx_graph_runable_npu/<RUN_ID>/<execution>/`，其中：

- `profiles/`：torch_npu profiler 原始结果。
- `performance.csv`：整次调用的 device 侧汇总耗时。
- `torch_compile_debug/` 和 `torchinductor/` 相关目录：Inductor debug 与生成代码。

可用环境变量：`PROFILE_ROOT`、`RUN_ID`、`WARMUP`、`ACTIVE`、`REPEAT`。profile 默认配置为 `WARMUP=1 ACTIVE=10 REPEAT=1`。

## 抽取单个 Triton kernel

`scripts/tools/extract_triton_kernel.py` 从 Inductor `output_code.py` 中抽取指定 kernel 的定义和首次调用所需参数。若同一 kernel 被调用多次，只采用源码中第一次 `.run(...)` 的参数：

```bash
python scripts/tools/extract_triton_kernel.py \
  /home/qc/Repos/Ascend/tmp/emb_opt_bs400/ouput_code_npu.py \
  triton_poi_fused_amax_gt_sign_view_0 \
  -o scripts/repro/emb_opt_bs400/kernels/triton_poi_fused_amax_gt_sign_view_0.py
```

工具会保留生成文件的公共导入、kernel 前的 FX/ATen 注释和 Triton DSL，并从 `benchmark_compiled_module` 读取原始输入构造，再反向收集首次调用依赖的 shape、buffer、numel 和 stream。
