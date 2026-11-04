# FMA fallback 与 attention constexpr 临时对照

`run.py` 在原 TorchBench runner 进程内安装一个临时改动，不修改安装目录。
结论直接维护在对应模型分析文档中。

- `baseline`：保持原实现。
- `no_fma_fallback`：在导入 torch/torch_npu 前安装单模块 source loader，仅从 `FALLBACK_LIST`
  删除 `prims.fma` 和其 default overload，避免后端自动加载使修改发生在 lowering 注册之后。
- `bind_constexpr`：仅在 `NPUIndexTritonKernel.codegen_kernel` 中补齐 `equal_1_arg_indices` 对应的常量值 1。

在指定 TorchBench 容器工作目录激活环境，选择允许且空闲的物理卡后运行：

```bash
python <torch-learn>/scripts/repro/cicd/performance/fma_seed_validation/run.py \
  --runner <runner-dir>/huggingface.py --variant baseline --output-dir <output>/metadata -- \
  --only OPTForCausalLM --device npu --training --performance --float32 --batch-size 2 \
  --backend inductor --npu-backend triton --disable-aclgraph --iterations 20 --repeat 3 \
  --enable-profiler 0 --prof-output-path <output>/profile --output <output>/benchmark.csv
```

为各 variant 使用不同的 Inductor cache 和 debug 输出目录，关闭 FX graph cache，保留全部生成代码。
性能使用稳定 step 的设备 kernel duration 汇总；同时核对生成 wrapper 中的 FMA 外部调用、
首层 attention 的 Grid1D/GridNpu 及 checker 日志。精度使用原 runner 的 `--accuracy --accu-summary`，
保持同一模型、batch 和 FP32，不以编译成功代替精度通过。
`environment.json` 保存版本、关键源码 SHA256、实际 lowering 入口以及安装文件未改动的检查。
该 runner 的原 `--performance` 路径没有调用 `run_n_iterations`，单传 `--enable-profiler` 不会采样。
脚本在原 `run_performance_test` 两个函数预热完成后，补充相同输入的 Eager/Inductor 设备采样：
各执行 `iterations` 步，前 10 步 warmup；原性能计时流程继续执行。
补充的函数源码与 SHA256 保存在 metadata；`index_codegen_failures.json` 保存回退前的异常。
可先用 `--check-setup-only` 检查修改安装，不运行模型。

完成后运行 `summarize.py --job-root <job-root> --output <summary.json>`，
按 CSV 的真实 Step Id 统计整步和逐 kernel duration，并单独统计 softmax backward 后的
Mul/Add/Mul/Add 窗口；不使用主机 wall time 代替设备时间。

`validate_index_constants.py` 用于在保留安装版本的环境中，加载 master 修复前后的完整
`NPUIndexTritonKernel.codegen_kernel` 方法，执行修复提交中的元数据回归测试：

```bash
python <torch-learn>/scripts/repro/cicd/performance/fma_seed_validation/validate_index_constants.py \
  --before <master-codegen.py> --after <fixed-codegen.py> \
  --test-file <fixed-repo>/test/_inductor/test_index_triton_constants.py \
  --test-support-dir <fixed-repo>/test --output-dir <metadata-results>
```

输入文件应从文档指定的基线/修复提交提取；脚本记录源码 SHA256，要求原始组 3 项失败、
修复组 3 项通过，并检查安装文件未变化。只回放进程内方法、构造元数据，不编译/运行设备 kernel，
也不将结果表述为完整 master 构建或整网性能回归。
