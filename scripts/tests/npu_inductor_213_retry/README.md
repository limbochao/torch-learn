# PyTorch 2.13 NPU Inductor 失败与跳过重试

该脚本复测既有 331 行清单中的失败、超时、收集异常及可解除的跳过，保持原测试源码不变。
运行记录见 [重试结果](../../../artifacts/npu-inductor-213-retry-20261011/README.md)。

## 执行

先激活原测试环境，并将 `CASE_DIR` 设置为待测仓库的 `test/inductor` 目录。
`run_retry.py`、`isolated_plugin.py`、`retry_gates.py` 需位于同一个目录。

```bash
python run_retry.py failures-plan.json failures --case-dir "$CASE_DIR" --cards 0,1,2
python run_retry.py dynamic-plan.json dynamic --case-dir "$CASE_DIR" --cards 3,4
python run_retry.py skips-plan.json skips --case-dir "$CASE_DIR" --cards 5,6
python run_retry.py xfails-plan.json xfails --case-dir "$CASE_DIR" --cards 0
python run_retry.py skips_followup-plan.json skips_followup --case-dir "$CASE_DIR" --cards 1,2,3
python run_retry.py dynamic_followup-plan.json dynamic_followup --case-dir "$CASE_DIR" --cards 4
```

各计划来自结果目录。每行一个独立 pytest 进程和 Inductor 缓存目录，超时上限为 900 秒。
三个主要计划可以并行；`xfails` 在 `failures` 结束后运行。
`skips_followup` 只补跑初版插件未解除 `unittest.case.skip` 的 30 行；使用当前插件首次运行时无需补跑。
`dynamic_followup` 进一步解除第 292 行仅限 FBCODE 的装饰器；该函数体不依赖 FBCODE 专用 API。
原清单第 230 行是跨设备测试，在普通失败项结束后使用全设备可见性；源码实际访问设备 0、1。
其他原始跨设备行若用于本脚本，也会使用全设备可见性。
完整 invocation、pytest 分阶段结果、日志和 JUnit XML 保存在输出目录的 `results/` 下。
已存在 `.invocation.json` 的行不会重复执行。恢复之前须先确认对应 driver 和测试进程已结束。

默认环境为 `TORCH_TRANSFER_TO_NPU=1`、`TORCH_NPU_DEVICE_CAPABILITY=8.0`，
`TORCHINDUCTOR_COMPILE_THREADS`、`TORCHNPU_PRECOMPILE_THREADS` 和 `OMP_NUM_THREADS` 均为 2。
`--python` 可指定另一个已配置好的测试环境解释器。

## 跳过条件

插件仅在计划设置 `RETRY_UNSKIP=1` 时生效，并在当前 pytest 进程内修改测试辅助逻辑。
原测试文件、编译器、生成 kernel、输入及断言均保持不变；进程结束即失效。

- 将通用 GPU/CUDA/Triton 可用性 skip decorator 按明确的 reason 白名单解除。
  该环境 NPU 可用、Triton 已安装，但社区 `HAS_GPU` 检测仍为假。
  解除的是测试入口门控，不能据此认定 CUDA 专属功能受 NPU 支持。
- `RETRY_DYNAMIC_GATE=1` 只开启动态形状测试模块 `setUp` 使用的 `HAS_GPU` 门控。
  原 `TestInductorDynamicNPU` 节点须先通过真实收集结果映射到 `TestInductorDynamicPRIVATEUSE1`。
- `RETRY_EXTENSION_GATE=1` 开启测试辅助 `HAS_GPU_AND_TRITON`，使 extension 测试导入其 Triton 依赖。
- `RETRY_NPU_MEMORY=1` 为大张量测试增加基于 `torch.npu.mem_get_info` 的实际空闲内存检查。
  内存不足仍会跳过，不直接忽略内存检查。
- 慢用例使用其已有 `PYTORCH_TEST_WITH_SLOW=1` 开关。
- `RETRY_FBCODE_FLOAT_ITEM=1` 只解除 `TestInductorDynamic.test_float_item_return` 的 FBCODE 装饰器，
  不改变全局 `IS_FBCODE` 或其他空 reason 的 skip。
- CPU 专属、CUTLASS、TMA 等不在白名单内的限制保留；XFAIL 标记保留。

`retry_gate_events` 记录插件解除的装饰器和能力门控，以及实际内存检查数值。
装饰器事件可能包括同一文件中被导入但未选中执行的方法；实际执行范围以 `collected` 和 `reports` 为准。

## 汇总

```bash
python summarize.py "$BASELINE_RESULTS" "$RETRY_WORKDIR" "$OUTPUT_DIR" --require-complete
```

汇总保留全部 331 行的原始顺序，同时标记上一轮结果、当前结果、调用次数及本轮未重跑的行。
`node-results.json` 保留各次尝试；最终结果使用对应行最后一次尝试。
归档输出对个人目录、进程 ID 和临时路径作去敏，原始日志仍保留在测试工作目录中。
