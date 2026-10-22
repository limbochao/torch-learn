# 公共 symbolic stride 取模问题的 CUDA 对照

[单文件脚本](symbolic_stride_cuda_probe.py)不导入 `torch_npu`，可复制到安装社区 CUDA PyTorch 的机器运行。
目标是区分公共符号表达式缺陷、当前版本 Layout 的缺陷，以及真实 CUDA 图是否自然触发。
对应[八类修复分析第 7 项](../../../../docs/issue-logs/cicd/model-fix-dsl-explained-20261021.md#7-symbolic-stride整数整除证明不能引入倒数)。

## 元数据验证：不需要 GPU

```bash
python symbolic_stride_cuda_probe.py --mode metadata
```

脚本创建有具体提示值的 ShapeEnv 符号 `H=256`，约束 `H>=33`，构造
`a=FloorDiv(H-1,32)+1`、`b=a**2`。分别测试 b 的因式形式和展开形式，并交换比较方向。
每一次观察都使用新的 ShapeEnv，避免前一次添加的 guard 影响下一次。

| 输出 method | 实际检查内容 |
| --- | --- |
| `percent_expr` | 原始 `Eq(a % b, 0)` 在 ShapeEnv 中求值，并打印负指数节点 |
| `pythonmod_expr` | 同样的操作数，改用社区已有 `PythonMod` |
| `native_layout` | 直接调用安装版本的 `Layout._stride_expr_ge_or_false` |
| `pythonmod_candidate` | 直接调用 c74c45c960 的候选比较函数，不安装全局 patch |

这里没有 tensor 分配或设备 kernel，CUDA 环境上的 metadata 结果也不能叫作 CUDA kernel 失败。
2.7.1/2.10 等没有该 Layout helper 的版本会将 `native_layout` 标记为 SKIP，
不会人为补上 2.13 的方法并冒充旧版本原生行为。

`status=REPRODUCED` 表示原始表达式或原方法观察到精确的 `ValueRangeError: Invalid ranges [0:-1]`。
原始路径在其他版本可能 PASS；两种结果都应保留实际版本信息，不能预设所有 SymPy/PyTorch 组合都会失败。
候选方法错误或返回错误真值记为 ERROR，并使脚本返回非零。预期缺陷的 REPRODUCED 不计作脚本自身失败。

## CUDA 实际编译：先跑原生版本

建议先使用出现新 Layout helper 的社区 2.13 环境。
前向为 1×1 conv(stride=32) → spatial mean → sigmoid → 逐点乘；
同一个 `torch.compile(dynamic=True, fullgraph=True)` 依次执行 `(batch,H)=(2,224),(3,256)`，
输入 `[batch,16,H,H]`、weight `[16,16,1,1]`，FP32、contiguous。
脚本用同一输入和同一上游梯度比较 eager/compiled 的前向、输入梯度和权重梯度；不允许静默 eager fallback。

从脚本所在目录执行，命令使用独立进程、独立缓存，避免旧缓存跳过待观察的编译：

```bash
mkdir -p out/stride_native
set -o pipefail
TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR="$PWD/out/stride_native/cache" \
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$PWD/out/stride_native" \
python symbolic_stride_cuda_probe.py --mode all --device cuda --traceback \
  2>&1 | tee out/stride_native/run.log
```

`layout_compare` 打印前八次真实调用的操作数，`layout_error` 记录比较点实际异常，
`layout_summary` 给出总调用/异常数。observer 只是记录并调用原方法；默认不安装候选。
版本、SymPy、CUDA build、GPU 型号以及是否已存在 NPU patch 标记均输出。
`npu_patch_marker=true` 时原方法已被替换，不能把该次运行当作未修改的社区基线。

若 CUDA 默认 layout optimization 避开了该调用，可在另一个新目录、新进程中增加
`--disable-layout-optimization`，该开关临时设置 `layout_optimization=False`，
用于探索 convolution backward 的布局约束路径；这也不保证一定出现目标 stride。
保留默认与关闭优化两种结果，不将关闭优化的结果冒充默认配置行为。

## PythonMod 编译对照

```bash
mkdir -p out/stride_pythonmod
TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR="$PWD/out/stride_pythonmod/cache" \
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$PWD/out/stride_pythonmod" \
python symbolic_stride_cuda_probe.py --mode compile --device cuda --patch-pythonmod --traceback \
  2>&1 | tee out/stride_pythonmod/run.log
```

该参数只替换当前脚本进程的 Layout 比较方法，不修改安装文件，并在退出比较上下文时恢复原始描述符。
对照两边必须使用相同的 layout optimization 设置；若基线关闭优化，候选也加相同开关。
没有 helper 的旧版本不能运行这个编译 patch 对照，但仍可跑原生 compile 及独立 expression 检查。

## 如何判断结果

- expression 出现 REPRODUCED：公共符号表达式问题成立，不要求 NPU 或非线性化。
- `native_layout` 出现 REPRODUCED：该安装版本的实际 Layout 方法也可在手动构造的操作数上触发。
- CUDA compile 出现对应的 `layout_error`：真实 GPU 编译图确实进入该失败路径。
  仅有其他编译异常不算同一个问题，应看 traceback 和实际操作数。
- CUDA compile PASS、`calls=0`：该图未调用此 helper，不能证明 helper 没有缺陷。
- CUDA compile PASS、`calls>0`：只证明已观察到的操作数及当前配置通过，不能证明任意表达式均安全。
- 即使最终数值 PASS，若 `errors>0`，也不能称为首次布局检查无异常；可能有上层重试。
- 基线对应异常、候选无对应异常且数值通过：支持该表达式替换修复这条 CUDA 路径，仍非全模型回归证明。

本地 2026-10-22 的环境为 PyTorch 2.9.1+cpu、SymPy 1.14.0。
展开形式的原始取模在两个方向均出现目标 ValueRangeError，PythonMod 表达式及候选比较通过；
原生 Layout helper 不存在，CUDA 不可用，分别明确 SKIP。
完整 JSON 输出见[CPU 对照记录](../../../../artifacts/cicd/model_fixes_20261020/final_checks/stride_cuda_probe_cpu_20261022.jsonl)。
同一本地环境另以 `--mode compile --device cpu --disable-layout-optimization` 检查脚本的真实编译分支，
两组 shape 的前向、输入梯度、权重梯度比较通过；Layout helper observer 因旧版本不存在而 SKIP。
这些都不是 2.13 CUDA 运行结果。无需模型权重、数据集、NPU 后端或原始编译缓存。
