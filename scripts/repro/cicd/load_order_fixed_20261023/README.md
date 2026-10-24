# Albert 固定 tiling 的 load 顺序对照

从历史静态 training performance 失败 kernel 抽取两份 DSL，直接调用 Triton 编译器。
不调用 `torch.compile`、Inductor codegen、heuristics decorator 或 autotune。

## 文件与约束

- `kernel_front.py`：原始 DSL，81 条输入 load 集中在计算前。
- `kernel_near.py`：同一份 DSL，仅把各 load 移到首次使用之前。
- `run.py`：独立单文件，内嵌两份 DSL、固定 tiling、编译与数值检查逻辑；只复制此文件即可执行。
- `compare.py`：固定配置编译；传入 `--check` 时再直接启动已编译 binary，并比较 eager 结果。
- `manifest.json`：原文件来源、SHA256、固定参数与语句对照信息。

两份 kernel 的参数、load 表达式及 mask 相同。循环体内语句的 AST 多重集合相同；
去掉 load 后，剩余算术、归约和 store 的语句及顺序完全相同。
没有移动 store、改变归约算法或缩小 tile。

```text
X0BLOCK_SUB = 1
R1BLOCK_SUB = 2048（DSL 内的常量）
x0_numel = 4096
r1_numel = 2048
X0BLOCK = 64
grid = (64, 1, 1)
num_warps = 1
num_stages = 2
compile_mode = simd_simt_template
target = GPUTarget("npu", "Ascend950PR_9579", 32)
```

这里有两个层次：`X0BLOCK=64` 表示一个 program 总共覆盖 64 个输出列；
`X0BLOCK_SUB=1` 表示每次循环只计算其中 1 列。
每条矩阵 load 的逻辑张量形状是 `(R1BLOCK_SUB, X0BLOCK_SUB)=(2048, 1)`，
一个 program 循环 64 次，每次完整归约 2048 行。矩阵输入的 shape 是 `(2048, 4096)`，
stride 是 `(4096, 1)`。两份 DSL 的这些参数完全一致。

`--check` 使用 seed 0 的合成 FP32 输入，保留矩阵 `(2048, 4096)` 的布局和列访问 stride 4096，
以及各 `(2048,)` 的广播向量；81 个输入与两个输出独立分配，不构造别名。
编译本身不依赖张量值。

## 运行

在指定的 NPU Docker 工作目录激活 `pta_213_cicd` 和 CANN 环境。
只需把 [run.py](run.py) 复制到工作目录，在该目录执行：

```bash
python run.py
```

不需要同时复制 `compare.py`、`kernel_front.py`、`kernel_near.py` 或 `manifest.json`。
也可以从 `torch-learn` 仓库根目录执行：

```bash
python scripts/repro/cicd/load_order_fixed_20261023/run.py
```

默认依次比较 `multibuffer=False/True` 下的集中 load 和就近 load，共四次固定配置编译。
每次运行自动新建当前工作目录的 `load_order_results_<时间戳>/`，缓存也放在该结果目录内。
脚本打印实际 torch、torch_npu、Triton 版本；完整模块路径、torch_npu git 版本等保存在各自的 JSON 中。
无需重新生成模型 kernel，无需安装 04 的 codegen patch。

运行开始会直接显示以下 tiling 信息；编译后逐项显示结果，UB 溢出时额外打印所需 bits 和可用 bits。
也可执行 `python run.py --show-config` 只查看配置，该参数无需导入 torch_npu 或 Triton。

```text
Total extent:       x0_numel=4096, r1_numel=2048
Grid:               (64, 1, 1)
Program coverage:   X0BLOCK=64 output columns
DSL subtile:        X0BLOCK_SUB=1, R1BLOCK_SUB=2048
Load tensor shape:  (R1BLOCK_SUB, X0BLOCK_SUB) = (2048, 1), FP32
Loop per program:   64 x-iterations, each reducing all 2048 rows
```

设备可正常运行时，用同一入口增加数值检查：

```bash
python scripts/repro/cicd/load_order_fixed_20261023/run.py --check
```

`--check` 对所有编译通过的版本直接启动 binary，与 eager 结果比较。
默认每个子进程（编译与可选数值检查合计）限时 180 秒，超时后结束该用例的进程组并继续下一项。
可使用 `--timeout 300` 调整；`--multibuffer false` 只跑一组；`--output <新目录>` 指定结果位置。
输出目录已存在时拒绝执行，避免覆盖已有证据。

输出含义：

- `REPRODUCED`：集中 load 明确出现 UB 容量错误，就近 load 编译通过；若带 `--check`，还要求数值通过。
- `NOT_REPRODUCED`：两版均通过，当前环境未复现该问题。
- `ERROR`：出现其他失败、就近 load 失败或超时；查看对应的 `stdout.log` 和 `traceback.txt`。

汇总保存在 `summary.json`。前两种结论返回退出码 0，`ERROR` 返回 1。
未加 `--check` 时 `numeric_status=NOT_RUN`，编译通过不代表数值检查通过。

以下是保留的拆分版入口，单文件复现不依赖这些文件。
如需分别编译某个版本，在本目录执行：

```bash
export TRITON_CACHE_DIR="$PWD/cache"
python compare.py --order front --output results/front_mbfalse
python compare.py --order near --output results/near_mbfalse
python compare.py --order front --multibuffer true --output results/front_mbtrue
python compare.py --order near --multibuffer true --output results/near_mbtrue
```

单独检查就近 load 版本：

```bash
python compare.py --order near --check --output results/near_checked
```

底层 `compare.py` 每次保存 `result.json`，成功编译时保存文本 IR，失败时保存 traceback。
底层入口会捕获编译异常，应检查 JSON 中的状态；统一入口 `run.py` 则会自动汇总并设置退出码。

## 2026-10-23 执行结果

实际激活环境为 torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261019`、Triton `3.6.0`，
CANN 路径指向 9.1.0。登录 banner 中的 2.7.1 不是激活后的 torch_npu 版本。
实际模块路径、安装代码校验值和 CUDA `get_load_buffer()` 内容均保存在环境记录中。
安装的 NPU codegen 未检测到 04 的特征注释；本实验不执行该 codegen 文件。

| multibuffer | 集中 load | 就近 load |
| --- | --- | --- |
| False | UB overflow：需要 6,030,080 bits，可用 2,031,616 bits | 编译通过 |
| True | UB overflow：需要 11,339,008 bits，可用 2,031,616 bits | 编译通过 |

另外重放了原始记录中显式传入 `enable_vf_fusion=True` 的参数组合，结果相同。
本栈成功候选返回的有效 metadata 中该字段仍为 None，且发生缓存命中，不能算作新的独立编译模式。

失败发生在 `ConvertLinalgIRToBinary` 的 `buildFinalHIVMPipelines`，错误来自下层 UB 资源检查。
对照证明当前下层工具链对 load 顺序敏感，并不证明具体哪个排程或分配 pass 存在错误。
04 在此作为 NPU codegen 的资源规避；该实验不能证明所有候选只能通过这一种改法编译。

本次未执行目标 kernel 的数值复验。默认映射设备的运行探测中，1 元素 `torch.ones` 已分配，
但在同步时 45 秒超时；更早的设备属性探测还出现 507899 警告。
编译 PASS 只表示生成 binary 成功，不能当作本次数值 PASS；也没有执行 CUDA 设备对照。

结果见 [summary.json](../../../../artifacts/cicd/load_order_fixed_20261023/summary.json)，
具体日志和 IR 位于同目录的 `results/`。
