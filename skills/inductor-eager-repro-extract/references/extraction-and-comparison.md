# 抽取接口与 GPU 对照

本 reference 说明现有抽取器的接口和常见边界；升级后按工具当前实现核对。
工具位于 `torch-learn/scripts/tools/extract_triton_kernel.py`，不随本 skill 复制维护。

## 动态符号与 CASE

```bash
python <torch-learn>/scripts/tools/extract_triton_kernel.py \
  <original_output_code.py> <kernel_name> --only-eager \
  --symbol-values '{"s_batch":4,"s_seq":512}' -o eager_case.py
```

符号名为示意，必须换成当前 metadata 中真实存在的名称。工具会拒绝未知或无法唯一解析的符号。
一次具体符号绑定不是多 shape 验证；抽取器生成的邻近 sample 也不一定满足模型语义约束。

当前输出包括 `eager_forward`、`make_inputs(binding, device)`、`SAMPLE_BINDINGS`、
`COMPILE_BINDINGS`、`DYNAMIC_DIMS` 和 `CASE`。
可导入后这样检查基线：

```python
binding = CASE["compile_bindings"][0]
args, kwargs = CASE["make_inputs"](binding, "cpu")
result = CASE["forward"](*args, **kwargs)
```

交付独立脚本时把函数、实际输入构造和入口留在同一文件。
默认 `rand_strided` 是合成输入，不会保证索引合法、负索引覆盖或别名正确；据目标语义调整构造。
如使用真实数据替代随机值，说明来源及设备映射方式。

## 易丢失的语义

| 场景 | 保留的信息 | 常见误改 |
| --- | --- | --- |
| LayerNorm/多轴归约 | eps、轴顺序、尾部 mask、实际输出 | 合并输出轴到归约轴，或增加无关 tuple 输出 |
| 负索引 | 合法负端点、上界、原 table 布局 | 全零整数输入完全漏掉 wrap |
| 多路梯度/资源不足 | 原分支数量、load/consume 依赖、加法顺序 | 删除大矩阵后不再触发原资源场景 |
| softmax 分解 | selector、缩放常量、max/exp/sum/div 及最终概率输出 | 改成高级 softmax 或额外返回归约中间值 |
| 符号布局/调度 | 两组 shape、同一 callable、backward 和原符号表达式 | 每次重新 compile，或只做 eager forward |

## 生成 CUDA DSL

下面的运行参数是建议给独立入口提供的接口，不是抽取器自动生成的命令行参数。
确保脚本已实现这些参数；不要对只有 `CASE` 定义的文件执行后声称 GPU 验证通过。

```bash
TORCH_COMPILE_DEBUG=1 \
TORCH_COMPILE_DEBUG_DIR="$PWD/cuda_debug" \
TORCHINDUCTOR_CACHE_DIR="$PWD/cuda_cache" \
python standalone_case.py --device cuda --compile
```

保留运行日志、实际设备/软件版本和 generated `output_code.py`。
逐项标注 ATen 子图对应的 CUDA kernel 与 NPU kernel；不能按生成名称后缀匹配。
若需要 winner tile，读取实际 selected config，不能把失败候选或第一个候选当最终配置。
修复等价性检查包括访问集合、保留轴、归约轴、mask 后的单位元及输出；
性能与资源调度单独评估。DSL 生成前失败的问题，应对照 IR 和职责相同的社区源码。
