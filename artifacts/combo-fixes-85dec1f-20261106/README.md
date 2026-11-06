# Combo 六个失败问题的定向修复验证

后续 F4 已加入余数不进位证明并合入单个提交，见 [F4 优化与 2/2 复测](../combo-f4-nocarry-20261106/README.md)。本目录保留原轮次提交及结果。

基线 `85dec1f1a10ee42ebce87f1a1f8d2095909c0af6`，最终安装源码 `bc9d3c4ee973dde621ebf61600967ffbd66d0571`。
分支 `fix/combo-f10-f12-f11-f8-f4-f5-v2.10`，已推送 origin。
NPU Ascend950PR，2026-11-06；详细包版本和安装源码校验见 [baseline_environment.json](baseline_environment.json) 与 [committed_installation.json](committed_installation.json)。

原新包 8 个方法、10 个实例全部失败；本次修复后 8/8 方法、10/10 实例通过。
未运行全量，F6/F7 不在本次范围；原测试适配和数值、梯度、突变断言没有变更。
生产提交累计四个 Python 文件，新增 30 行、删除 12 行；没有新增正式测试或调试日志。

## 问题、代码与修复

| 问题 | 修复提交 | 修复入口 | 单项验证 |
| --- | --- | --- | --- |
| F10 / F12 | `64a3e7e8b9` | NPUComboKernel.codegen_kernel | [2/2 PASS](fix_f10/evidence/functional_results.json) |
| F11 | `9b1a894d65` | NPUComboKernel.codegen_kernel | [3/3 PASS](fix_f11/evidence/functional_results.json) |
| F8 | `fdf73f229e` | _NPUKernelCodegenMixin.generate_node_numel_expr | [1/1 PASS](fix_f8/evidence/functional_results.json) |
| F4 | `a2f7b18da9` | eliminate_zero_term / split_expression / NPUIndexTritonKernel.filter_masks | [2/2 PASS](fix_f4_complete/evidence/functional_results.json) |
| F5 | `bc9d3c4ee9` | NPUIndexTritonKernel.codegen_kernel / NPUComboKernel.jit_line | [2/2 PASS](f5_constants_only/evidence/functional_results.json) |

源码改动完整保存在 [fix-series.patch](fix-series.patch)，提交对应关系见 [fix_commits.json](fix_commits.json)。
F10 与 F12 是同一个缺陷，使用同一个修复提交，未为同一修复创建空提交。

### F10 / F12：删除状态未进入最终 store 渲染

子 kernel 已标记某些 inplace store 应删除，但 Combo 拼接后的 DeferredLine 在错误的 V.kernel 上下文中求值。
因而 `mean = sum / N` 写回后，同一输出地址又被 sum 覆盖；数学计算本身正确，最终输出错误。
修复把子 kernel 的 `inplaced_to_remove` 汇总到 Combo，最终 `code.getvalue()` 在 Combo 的 kernel handler 下执行。
删除状态保留在 Combo 本地，验证成功前不修改全图状态，避免失败后 fallback 被污染。
F12 的 test_mean 同源，直接 DSL 修改和正式源码修复都同时覆盖两个方法。

### F11：已删除的中间 buffer 仍生成 store

`out_ptr1` 等中间 buffer 已不在最终签名中，但对应 store 没有被 DeferredLine 删除，导致未定义参数的编译错误。
在相同渲染规则下进一步汇总 `removed_buffers`，三个突变/dispatch 方法均实际走 `linear_combo` 并通过。
与 F10 共用渲染机制，但所需删除状态不同，单独提交。

### F8：benchmark 收集尺寸时生成了运行期语句

benchmark 用空 kernel_name 调用尺寸发现函数，旧实现向 wrapper 提前写入 `_x0_numel = 1 + u0`。
此时 nonzero 结果尚未产生，u0 还未绑定，因此 wrapper 执行失败。
空名称调用现在直接返回 SymbolicCallArg；真实 kernel 的非空名称调用保留原有尺寸化简和语句生成。

### F4：线性化除法与 pooling 访问 mask

原方法固定输出 `(6,6)` 与 `(2,5)`，输入依次为 `(2,4,16,16)`、`(2,4,3,3)`、`(2,4,6,6)`，
包含 Combo False/True 以及原有低精度检查。逐层修复如下：

1. SymPy 的 FloorDiv 构造可立即求值为常量；eliminate_zero_term 先确认仍是 FloorDiv，再解包两个参数。
2. `floor((a+b)/d)` 不能普遍展开为 `floor(a/d)+floor(b/d)`。例如 `(8+10)//3=6`，拆开后为 5。
   只把已证明可整除的项移出，其他项保持在同一个 FloorDiv 中，保留余数进位。
3. 3×3 输入上采样到 6×6 时，pooling 窗口边界 mask 依赖展开轴，而 load 地址没有使用这些轴。
   原 filter_masks 根据地址轴的子集关系删除这些语义 mask，四个邻居被无条件求和，分母却只计一个有效元素。
   保留 TritonCSEVariable 语义 mask 后，load 在 mask 定义后生成，窗口外元素不参与计算；已有 axis mask 过滤保持原规则。

[单 kernel DSL 快速实验](pool_mask_dsl.py)只调整四个 load 的位置和窗口 mask，
[结果](pool_mask_dsl.log)为 `PASS max_abs=0.0`，确认了 mask 语义。
正式修复随后通过完整原方法，未限定输入形状、算子名或强制回退。

### F5：值为 1 的 constexpr 元数据缺失

新基线没有复现旧的 int32/int64 索引检查错误，因此没有修改 index_expr 或 RNG 算法。
当前实际失败是 `load_seed_offset1` 参数被签名标记为 constexpr，但 triton_meta.constants 缺少数值 1，
进入 Triton IR 后成为 None。普通 Index kernel 与 Combo 都需要补充。
使用社区 `equal_1_arg_indices(signature)` 注册元数据后，False/True 两个实例都通过。
`f5_constants_only` 单独证明不需要此前诊断时尝试的 dtype 修改。

## 最终安装源码验证

逐项阶段使用带 SHA256 的源码函数 overlay，实际加载入口记录在每轮 production.*.jsonl 中。
这些阶段的安装包始终保持原始 85dec1f；overlay 只在测试进程内生效。
最终从已推送提交提取四个完整 Python 文件覆盖安装目录后，执行 [installed_validation](installed_validation/evidence/functional_results.json)：
**10/10 PASS**，不加载函数修复 overlay 或旧 F3 overlay。

[完整性检查](installed_validation/evidence/integrity_checks.json)核对每个实例的 setup/call/teardown、
实例清单和 11 项安装源码前后哈希；四个改动文件与提交的哈希另见安装记录。
`combo_sources/` 和 audit.*.jsonl 保留实际生成路径，功能 PASS 与 Combo 路径是否使用分开记录。

运行使用仓库的可复用脚本：

```bash
python scripts/tests/combo_kernel_210/run_validation.py \
    --root <prepared-failed-cases> --devices 4,5,6 --expected-instances 10 --detach
```

准备目录保留原适配测试和本轮 [8 个方法清单](selected_methods.json)。
脚本自动收集、执行、汇总，使用独立目录和缓存，不覆盖历史结果。
下载后用 `archive_fix_series.py` 批量去敏归档，每轮 archive_manifest.json 保留原始及归档文件的 SHA256。
日志中的个人路径和 backend_hash 已去敏；生成 DSL 保持原格式。

`dsl_mean_store` 为两个 mean 用例的直接 DSL 诊断；`metadata_f10` 为状态传播假设实验。
`fix_f4` 与 `fix_f4_floor_carry_v2` 是 F4 逐层暴露问题的失败证据，不能当作最终修复结果。

## 影响范围与限制

修复涉及 Combo 删除状态、benchmark wrapper 尺寸收集、FloorDiv 化简、Index 语义 mask 和 constexpr 元数据。
mask 保留可能影响相关 kernel 的编译耗时及生成形态，本轮验证仅证明目标用例功能正确，未评价性能。
未运行全量，也没有在其他硬件或社区 CUDA 上验证；不能外推到所有动态尺寸和算子组合。

[生成代码核对](codegen_checks.json)记录 mean 最终写回次数、F11 pointer 声明、pooling 语义 mask 的定义与使用顺序，以及 RNG seed offset 的常量元数据。
