# Cooperative reduction：多卡一键测试

这个测试包从实际 `torch.compile` 入口测 NPU plain、GPU plain/cooperative；
NPU cooperative 使用手写 origin/debug DSL，联合搜索 RS 与 BLOCK。
CUDA 测试不依赖 torch_npu、NPU 硬件或 PyTorch 源码仓。

## 准备与选卡

需要 Linux、Python 3.10+、具有 cooperative 配置的 CUDA PyTorch/Triton，或已配置的 NPU 环境。
当前测试目标为 PyTorch 2.13。绘图需要 Matplotlib 3.8+：

```bash
python -m pip install 'matplotlib>=3.8'
tar -xzf cooperative-compile-matrix.tar.gz
cd cooperative-compile-matrix
```

CUDA 的 `--devices 0,1` 是**当前 Python 进程可见的逻辑卡号**。
若外层设置过 `CUDA_VISIBLE_DEVICES`，这里填写过滤后重新编号的卡号，也可以直接填写可见 GPU 的完整 UUID。
例如外层 `CUDA_VISIBLE_DEVICES=6,7` 时，本脚本填写 `--devices 0,1`。
脚本先解析实际 UUID，再为每个子进程只暴露一张卡；不会隐式使用其他卡。

NPU 的 `--devices` 填**物理卡号**，脚本通过 ACL 核对物理映射。
若外层设置了 `ASCEND_RT_VISIBLE_DEVICES`，所选卡必须属于该列表。
启动前自行检查当前机器的卡占用、可用显存和使用约定；本脚本不驱逐其他进程，也不自动选择空闲卡。

## 一键运行

先做小矩阵，检查当前机器的软件栈与多卡流程：

```bash
bash run_matrix.sh --backend cuda --devices 0,1 --smoke --out cuda-smoke
```

正式 GPU 全矩阵：

```bash
bash run_matrix.sh --backend cuda --devices 0,1 --out cuda-full
```

也可以使用 UUID，或只使用单卡：

```bash
bash run_matrix.sh --backend cuda --devices GPU-实际UUID1,GPU-实际UUID2 --out cuda-full
bash run_matrix.sh --backend cuda --devices 0 --out cuda-single
```

NPU 在已激活的 torch_npu / Triton-Ascend 环境中运行，下例卡号须替换为当前允许使用的设备：

```bash
bash run_matrix.sh --backend npu --devices 1,2 --smoke --init-timeout 900 --out npu-smoke
bash run_matrix.sh --backend npu --devices 1,2 --init-timeout 900 --out npu-full
```

如需限核，先设置该环境支持的限核配置，例如 A5 的 `NPU_DEVICE_LIMIT=4,8` 表示 8 个 Vector 核。
脚本记录物理核数、有效核数和 runtime 限额，超预算候选仍会实测并保留正确性与超时结果。

默认参数如下：

| 项目 | 正式矩阵 | `--smoke` |
| --- | --- | --- |
| 计算 | fp32 `(a+b).sum(-1)` / `(a+b).amax(-1)` | 相同 |
| X | 1、7、56 | 1 |
| R | 4K、8K、16K、32K、64K、128K、256K、512K、1M | 4K、32K |
| NPU DSL RS | 1、2、4、8、16、32、48、56 | 1、2、8 |
| NPU DSL BLOCK | 256、512、1024、2048、4096、8192 | 512、2048 |
| DSL 粗测 | 每配置 3 个样本 | 相同 |
| DSL 复测候选 | 最快正确的 3 个；另保留最快的首期接入候选 | 1 个；同样保留接入候选 |
| 独立复测轮数 | 3 | 1 |
| 正式计时 | 每轮 7 个样本，中位数；最终取各轮中位数 | 相同 |

一个操作/形状整体分给一张卡：同卡执行 plain、候选搜索和复测。
多张卡按操作/形状轮流分配，每张卡最多有一个正在执行的测试进程，避免同卡并发相互污染。
54 组形状不会在每张卡重复执行。若需要每张卡的完整结果，分别运行单卡命令并使用不同输出目录。
不同型号 GPU 的结果保留各自设备名称，绘图不把它们合成同一条曲线。

可缩小或扩展范围，例如：

```bash
bash run_matrix.sh --backend npu --devices 2 --operations sum \
  --rows 1,7 --rnumels 4096,32768 --splits 1,2,4,8 \
  --blocks 512,1024,2048 --topk 2 --confirm-rounds 2 --out npu-custom
```

BLOCK 必须是 2 的幂。“最佳”只表示公布搜索空间内通过检查并完成复测的最快配置。
`block_search_boundary=true` 表示最佳 BLOCK 落在边界，需用新输出目录扩展 `--blocks` 再测，
不能把有限扫描称为全局最优。RS=1 保留协议耗时，但不参与实际分片配置的最优选择。

## 策略、计时与精度

| 结果组 | 实际策略 |
| --- | --- |
| NPU / GPU plain | `torch.compile(fullgraph=True, dynamic=False)`；关闭 cooperative、force 和上游 split |
| GPU cooperative | `torch.compile`；开启 cooperative 与 force，关闭上游 split |
| GPU natural | 相同图，开启 cooperative、关闭 force 和上游 split；单独记录实际命中或回退 |
| NPU origin / debug | SIMD DSL；相同连续分片和 all-read 合并，分别调优 RS/BLOCK |

compile 分组显式开启 `max_autotune=True`，保留编译器自己的候选与实际选中配置。
GPU 的 RS 由上游决定，不手工搜索 RS。默认选择不会覆盖全部形状，尤其 X=56，
因此全形状性能曲线明确标注 **forced cooperative**；自然选择结果写入独立文件。
RS=1 或没有生成 cooperative 的结果不会冒充实际 cooperative 性能。

计时覆盖完整编译 callable 或完整 DSL 计算，包含其必要设备操作；编译、autotune、CPU 参考和拷回在计时外。
使用外部 Graph + device Event，默认每图 16 次调用、每样本 10 次 replay；慢点根据 pilot 减少调用数。
两端均使用同地址重复回放的缓存口径，样本范围表示观测波动，不是置信区间。
compile 每轮实际选择的配置全部保存，若各轮选择不同则 `selection_varied=true`。

输入用固定种子在 CPU 构造；参考先做 fp32 加法，再做 fp64 sum/amax。
sum 采用 `atol=0.02, rtol=2e-4`；有限输入的 amax 要求精确一致。
检查首次输出、最后一次 Graph replay 输出和两组变化输入；DSL 另外检查多 phase 严格整数交换。
只有所有检查和复测通过的配置参加最佳选择。

错误但已完成的候选保留诊断耗时，启动失败或超时没有可用设备耗时。
debug 的首期接入候选还要求 `X*RS<=有效Vector核数` 及已观测的 AutoBlockify 映射；
这只是已验证 A5 SIMD 路线的候选条件，仍需后端契约确认，不是任意 NPU 的通用合法性证明。
origin 的驻留/映射条件另行确认，不直接套用 debug 的协议。

## 结果与 DSL / tiling

运行结束自动汇总，不需要手工寻找临时缓存：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 测试脚本指纹、完整参数、设备 UUID/物理卡及限额 |
| `results.json` | 每个操作/形状的已选结果与独立复测记录 |
| `candidates.json` | 所有扫描、复测、错误和超时候选 |
| `natural.json` | GPU 自然选择实际命中/回退情况 |
| `figures/results.csv` | 主结果、配置、精度状态、同设备收益；GPU 108 行，NPU 162 行 |
| `figures/candidates.csv` | 全部候选，包含错误配置的已完成耗时 |
| `figures/natural-selection.csv` | 自然选择与强制测试分开 |
| `figures/same-config-sync.csv` | NPU 同 RS/BLOCK 的 origin/debug 对照及状态 |
| `figures/latency.svg` / `speedup.svg` | 延迟与同设备 plain / candidate 收益；大于 1 表示加速 |
| `figures/selected-rs.svg` / `selected-block.svg` | 实际选中 RS；NPU DSL 另给 BLOCK |
| `cases/` | 每组 wrapper/DSL、选中配置、完整 metadata、可导出的 IR、精度和原始计时 |

SVG 同时生成 PNG。CUDA 与 NPU 的完整主结果合并后为 270 行，GPU natural 另存。
生成 kernel 的 `ir/` 中保留后端可提供的 TTIR、adapter、LLVM IR 或 PTX；无法导出的阶段不会伪造文件。
NPU 每个 DSL RS/BLOCK 候选均导出产物。`cache/` 是可再生成的编译缓存，不是唯一的证据来源。
各 compile 记录另保存实际 PyTorch/Triton/torch_npu 版本和安装源码摘要。
回传结果时保留除 `cache/` 外的整个输出目录。

## 刷新与合并图表

在原机器重建结果，或把整个结果目录带到无卡机器执行：

```bash
python plot_compile_matrix.py --input cuda-full --out refreshed-figures
```

合并多台 GPU 与 NPU 的数据：

```bash
python plot_compile_matrix.py --input npu-full --input cuda-4090 --input cuda-other \
  --out combined-figures
```

每个输入目录需保留 `run.json`、`results.json`，以及要导出的候选和自然选择 JSON。
绘图不会混用旧 `reduction_matrix.py` 的固定 BLOCK 数据。
若测试环境没有 Matplotlib，可加 `--no-plot`：仍自动生成 CSV，在另一台机器执行上述命令生成 SVG。

## 续跑与失败处理

```bash
bash run_matrix.sh --backend cuda --devices 0,1 --out cuda-full --resume
```

续跑必须保持设备顺序、源码、搜索参数、限额和计时参数一致。
已有成功、错误和超时点都不重跑；重测或更换参数请使用新目录。
不要用 smoke 目录接着运行正式矩阵。
同一输出目录有进程锁，不能同时启动两个采集器。

默认冷初始化/编译无进展时限 300 秒，执行无进展时限 30 秒；可调整 `--init-timeout`、`--point-timeout`。
NPU 冷编译或共享机器建议使用 `--init-timeout 900`；小范围验证曾出现首次冷编译超过 300 秒，
同一配置随后完成编译与检查。编译阶段超时不能当作精度错误或 kernel 延迟。
只终止本脚本创建的子进程组，不操作其他进程。
退出码 0 表示请求的主结果全部可用；2 表示已完成汇总但有无正确配置或未实际分片的分组；
1 表示环境/脚本错误；130 表示中断。逐候选错误不会因为扫描进程退出而被标为 PASS。

NPU 全量调优和超时候选可能耗时较长。新机器先运行 `--smoke`，再启动完整矩阵。
