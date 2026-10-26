# DINOv2 SIMT reduction 精度复现

`vit_base_patch14_dinov2.lvd142m` 最终 LayerNorm 参数梯度 kernel 的独立复现。
依赖 PyTorch、torch_npu 和支持 Ascend A5 的 Triton-Ascend，不依赖 timm 或模型权重。

## 文件

- `original_kernel.py`：完整原始 `@triton.jit` 函数，函数体逐字保留；移除 Inductor autotune decorator，改由脚本指定配置。
- `minimal_kernel.py`：只保留 beta 梯度的简化版本，用于缩小下层编译问题。
- `run.py`：分别运行上述 kernel，打印实际输出、CPU 参考值和误差。
- [原始 wrapper 和验证证据](../../../../artifacts/cicd/dinov2_simt_reduction_20261026/)：保留原 FX fragment、metadata、首次调用和实际选中配置；个人路径与 backend hash 已去敏。

## 运行

在 Ascend A5 环境加载对应 PyTorch / CANN 环境后，从仓库根目录运行：

```bash
# 原始完整 kernel、原始错误配置
python scripts/repro/cicd/dinov2_simt_reduction_20261026/run.py

# 保持 simt_only，仅缩小 XBLOCK；--check 失败时返回非零
python scripts/repro/cicd/dinov2_simt_reduction_20261026/run.py --case small_x --check

# 原始完整 kernel 的四组配置对照
python scripts/repro/cicd/dinov2_simt_reduction_20261026/run.py --case all

# 简化 kernel 的同样配置对照
python scripts/repro/cicd/dinov2_simt_reduction_20261026/run.py --kernel minimal --case all
```

脚本使用逻辑设备 0，物理设备通过 `ASCEND_RT_VISIBLE_DEVICES` 选择。
默认只打印结果；加 `--check` 会检查所有选中配置，因此原错误配置预期触发 `AssertionError`。

## 输入与预期

全部输入为 FP32 合成数据，保留模型原 shape/stride，并非模型中间张量实际数值。
这组数据用于核对 DSL 数学语义，不要求它们来自一次真实 LayerNorm forward。

| 参数 | shape | stride | 值 |
| --- | --- | --- | --- |
| 上游梯度 | [16,768] | [768,1] | 1/(16×768) |
| LayerNorm 输入 | [16,1370,768] | [1052160,768,1] | 1 |
| mean | [16,1370,1] | [1370,1,1] | 0 |
| rstd | [16,1370,1] | [1370,1,1] | 1 |

两个输出是 gamma / beta 梯度，均为 `[768]`、stride `[1]`。
每个通道应累加 16 个样本的贡献，期望均为 `1/768 = 0.0013020833333333333`。
简化 kernel 只输出 beta 梯度。

原环境实测：A5 / Ascend950PR_9579，PyTorch 2.13.0+cpu，native torch_npu
2.13.0.dev20261019，Triton 3.6.0，CANN 9.1；基线 Python overlay 来自
`bugfix/cicd_issue_fix@848e9feaa86f1cefe82ea1307c62c2d8b89bfc6f`。

| XBLOCK | R0_BLOCK | num_warps | compile_mode | 实际值 / 期望值 |
| ---: | ---: | ---: | --- | ---: |
| 32 | 64 | 8 | simt_only | 5/16，错误 |
| 8 | 64 | 8 | simt_only | 1，正确 |
| 32 | 16 | 8 | simt_only | 1，正确 |
| 32 | 64 | 8 | simd_simt_template | 1，正确 |

各配置均为 `num_stages=2, multibuffer=False`。这些是配置对照，不代表下层缺陷已修复。

## DSL 语义

`r=0..21919` 覆盖 `16×1370` 个 batch/token 位置；`r//1370` 是 batch，
`r%1370==0` 仅让 CLS token 的梯度参与计算。循环先在每个 reduction lane 上累加，
最后 `tl.sum(..., 1)` 应汇总所有 lane。原 DSL 的索引和 mask 覆盖全部 16 个有效贡献。

错误配置逐 batch one-hot 探针只保留 batch 0、3、5、10、15 的贡献。
相同 DSL 在其他配置下正确，独立纯 Triton 也能复现错误，责任范围目前缩小到
Triton-Ascend 的 `simt_only` 编译/执行路径；具体 lowering 或 BiShengIR pass 尚未确定。

## 整网范围

原场景为 timm、FP32 training accuracy、dynamic 编译、batch 16、SGD lr=0.01。
实际输入固定为 `[16,3,518,518]`，未覆盖多尺寸切换。
5 iterations 通过，50 iterations 失败。训练每步更新权重，容差为 atol=rtol=0.02，
因此短迭代通过不能证明该梯度数学正确。

此前仅将本 kernel 切换为正确的 `simd_simt_template` 后，整网 50 步仍失败。
相同权重下逐步对照的误差远小于自由训练轨迹差异；两份 eager 仅给其中一份初始
`norm.weight` 加一个 ULP，也会在长训练中分离。因此不能把剩余差异直接记为第二个 DSL 缺陷。
完整状态见验证证据中的 `validation_summary.json`。


2026-10-26 补充实测：只将目标 kernel 手动固定为 `8×64、8 warps、simt_only`，
对应 grid 为 96；其他 kernel、50 iterations、输入、优化器和阈值不变。
真实模型首次 kernel 输入按 CPU 公式核对，gamma 最大误差 `4.656612873077393e-10`，beta 误差为 0。
整网仍 `fail_accuracy`：prediction 最大差 `5.3725438`，梯度最大差 `1.0161488`，
71/174 个梯度张量超阈值；更新后参数 0/174 个张量超阈值。
这组最终误差与此前 template 对照一致；本次未重复逐步相同权重诊断。
原始摘要见证据目录的 `model_tiling_accuracy.txt` 和 `model_kernel_actual_input_check.json`。

归档后的 `run.py --case all` 与 `run.py --kernel minimal --case all` 已在上述 A5 环境重新执行：
原始/简化 kernel 均复现原配置 5/16 错误，其余三组配置均通过数值检查。
完整输出保存在证据目录 `standalone_results.json`。默认仅打印结果，进程退出 0 不代表所有配置正确。
