# NFNet：抽取 eager 计算，追查 index codegen 首次异常

[nfnet_index_fallback_eager.py](nfnet_index_fallback_eager.py) 可单独复制运行，包含输入构造、
eager、`torch.compile(backend="inductor", fullgraph=True, dynamic=True)` 和数值检查。
不依赖模型、权重、原始 trace 或同目录文件。默认 CPU eager；加 `--compile` 才编译。

对应[问题记录](../../../../docs/issue-logs/cicd/model-fix-06-nolinear-fusion-recheck-20261024.md)。
历史日志只保存了 no-linear 回退后的 `unexpected group`，首次 index 异常仍需重新捕获。
这里提供两个入口，不能把二者当作相同的原始模型图。

## `--case original`：历史 NFNet backward 原始 ATen 子图

默认入口直接执行失败融合组 `op59/op60/op61` 的计算，不再调用 backward。
从历史 `fx_graph_transformed.py` 抽取 16 条 ATen 表达式，保留原节点名、常量、
乘加顺序、空间 sum 和 sigmoid 导数，没有改写成高级 SiLU/backward 算子。

原始文件在本地的来源为
`/home/qc/Repos/Ascend/tmp/cicd-model-fixes-20261020/nfnet_scheduler/`：

- `fx_graph_transformed.py` 第 358–386 行，SHA-256：
  `e3ea51e8ecc4ee9779c9ee454f2e8347c3a9e19543280008979a4010fcc1d80f`。
- `fx_graph_runnable.py::load_args` 记录 `s28=128, s63=224`；SHA-256：
  `53318d40227509463c023e3da9d39ebeba9343a61edcf2dfb44968ea5c27d3f7`。
- 对照仓库中保留的[原始 scheduler IR][original-ir]。

令 `S=floor((H-1)/32)+1`，输入边界如下，均为 float32、storage offset=0：

| 输入 | 原 IR 对应值 | shape / stride |
| --- | --- | --- |
| `getitem_114` | `buf6` | `[B,1536,S,S]` / `[1536*S*S,S*S,S,1]` |
| `getitem_132` | `buf53` | 同上 |
| `add_1604` | 同名保存的前向值 | 同上 |
| `convolution_71` | 同名保存的前向值 | 同上 |
| `convolution_73` | 同名 gate 输入 | `[B,1536,1,1]` / `[1536,1,1,1]` |
| `convert_element_type_default_171/169/167` | 三个独立 CPU scalar tensor | `[]` / `[]` |

输出严格按融合组对外依赖返回：`mul_tensor_102=buf59` 和 `mul_2040=buf61`。
`buf59` 还被组外 `op71` 使用，因此不能只返回归约结果。`buf60=sum_20` 为内部结果，
不额外返回；原 FX 的 `mul_2038` 属于组外消费，未纳入抽取范围。

输入值是 seed=0 的合成值，三个 CPU scalar 使用 `1.25/0.2/0.75`，不是捕获的模型数据。
第一组 `[128,1536,7,7]` 来自原 trace；第二组 `[3,1536,8,8]` 是补充的多 shape 检查。
原 IR 没有显示这八个边界输入之间的别名，本脚本分别分配它们，不重放整网的 buffer 复用。

```bash
python nfnet_index_fallback_eager.py --case original --device cpu
python nfnet_index_fallback_eager.py --case original --device npu:0 --compile --index-debug
python nfnet_index_fallback_eager.py --case original --device cuda:0 --compile
```

**符号边界：** `original` 的 `H` 只用于入口外构造输入。切断上游后，Dynamo 看到的是
`S` 本身的动态符号，不能保证恢复原来的 `FloorDiv(H-1,32)+1` 或原融合决策。
此入口用于阅读和重新 codegen 原始计算；数值通过不证明复现了首次 index 失败。
需要保留该符号来源时使用下面已有失败证据的训练入口。

## `--case training`：保留符号来源的训练入口

forward 与问题记录中的[派生小图](../model_fixes_eager_20261022/06_scheduler_symbolic_groups.py)相同：

```python
def training_forward(value, weight):
    value = F.conv2d(value, weight, stride=32)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()

compiled = torch.compile(training_forward, backend="inductor", fullgraph=True, dynamic=True)
```

同一个 compiled callable 依次执行 `(2,16,224,224)` 和 `(3,16,256,256)`，
权重为 `[16,16,1,1]`，均为连续 float32。保留真正的 forward 和 backward，
向 eager/compiled 传入相同随机输出梯度，不在 forward 后追加 `.sum()`。
检查输出（rtol/atol=0.0002）以及输入和权重梯度（rtol/atol=0.002）。

这是历史模型派生图，channel=16，不能称作历史 1536-channel backward 的原样重放。
[2026-10-24 对照日志](../../../../artifacts/cicd/nolinear_fusion_recheck_20261024/npu/parent_control.txt)
已记录它在 `dc67771ff2` combined scheduler 下自然回退并报 `unexpected group`；
带重新分组修复的版本可能正常运行，仍需查看首次 index 异常。

在目标容器工作目录、已激活匹配 torch/torch_npu/CANN 环境并选定可用物理卡后执行：

```bash
set -o pipefail
run_dir=$(mktemp -d "$PWD/nfnet_index.XXXXXX")
TORCHINDUCTOR_NPU_BACKEND=default FAST_RUN_WITH_MAX_TILING_NUM=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 TORCHNPU_PRECOMPILE_THREADS=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$run_dir/debug" \
TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache" TRITON_CACHE_DIR="$run_dir/triton_cache" \
python nfnet_index_fallback_eager.py --case training --device npu:0 --compile --index-debug \
    2>&1 | tee "$run_dir/run.log"
```

`--index-debug` 启用现有 combined scheduler 的异常日志，输出 `[index-fallback]`、
失败节点和完整 traceback，位置在进入 no-linear 回退之前；没有强制制造失败或修改 scheduler。
应从这里读取最内层异常，再判断是否确为线性化失败。正常回退后的 PASS 仍可能伴随首次 index 失败。
未出现该日志时，先确认缓存、版本和实际 backend，不能据此直接认定原缺陷已修复。

需要 CUDA 对照时，改为 `--device cuda:0 --compile` 并去掉 NPU 专用参数和环境变量。
`--no-dynamic` 可作静态对照。trace 中查看前向及反向的 FX graph、fusion IR 和 generated code；
不同设备不保证融合边界相同。

## 本次检查范围

16 条抽取表达式的 AST 已逐条对齐原始 transformed FX graph。
在独立目录仅复制这一个脚本，使用 torch `2.9.1+cpu`，两个入口的两组 shape 均完成
CPU eager 和动态 `torch.compile` 数值对照，训练入口同时检查输入/权重梯度。
未执行 NPU/CUDA 验证，未重新确认首次 index 异常。

[original-ir]: ../../../../artifacts/cicd/model_fixes_20261020/scheduler/nfnet_backward_fused_group_ir.txt
