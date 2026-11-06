# 0fd5a10 Combo 验证证据

验证日期：2026-11-06。基线为 `liulm/v2.10.0-combokernel-dev` 的
`0fd5a10da2a3c33cab8c200f5778d711e490ff6b`。最新结果已刷新：全量 55/62 个方法结束，其中 46 个通过、8 个失败、1 个执行异常；其余状态按实际记录展示。

原 wheel 的 `scheduling.py` 多一个缩进空格；安装后只删除此空格，生产安装目录不再叠加其他修改。
“新包+F3”通过测试进程内 overlay 替换已有分区方法，安装目录校验前后不变。
环境为 Docker、Ascend950PR、torch 2.10.0+cpu、torch_npu 2.10.0.post7+git0fd5a10、Triton 3.6.0、CANN 9.1.0。
用户授权共用物理卡 1–3；性能、benchmark 选择和融合收益不在本轮结论范围内。

## 证据索引

- `install/installation.json`：wheel 校验值、源码匹配、原样语法失败与前置修正。METADATA 仍为旧 a369f6c，模块源码实际为 0fd5a10。
- `install/scheduling.raw.py`、`install/scheduling-indentation.patch`：原始错误文件及一个空格的修正。原文件有预期的语法错误。
- `expected_*source_sha256.json` 位于 `execution/`，记录前置修正前后的 11 项校验；`installed_source/` 为修正后的实际安装源码。
- `validation_summary.json`：各阶段独立统计；不会将定向结果外推为全量通过。
- `failed_first/`：仅新包对原 13 个失败方法、15 个实例的复测。1 个方法/实例通过，12 个方法、14 个实例失败。
- `with_fixes/`：同一清单叠加 F3，7 个方法/实例通过，6 个方法、8 个实例失败。
- 各阶段 `functional_results.json`、`test_*.jsonl`、`jobs.jsonl`、`run_summary.json`：完整失败堆栈、逐阶段报告和方法返回值。
- 各阶段 `integrity_checks.json`：所有预期节点均上报，检查 setup/call/teardown，实际安装源码在运行前后保持一致。
- 各阶段 `audit.*.jsonl`：Combo 尝试、回退、错误堆栈，以及 F3 实际加载记录。原 F2 在新包里仍编译失败，但回退后通过。
- `combo_sources/` 保留检查前的源码模板，其中 Placeholder.KERNEL_NAME 在实际编译检查时替换为合法 kernel 名；模板不作为独立脚本运行。
- `with_fixes/combo_sources/audit.test_combo_kernels.test_helper_fn_defined/`：含 helper 定义和 scan 调用的真实 Combo 源码，配合通过记录验证 F9。
- `with_fixes/generated_on_failure/ComboKernelDynamicShapesTests.test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda/`：F10 失败时的生成代码。
  其中 `wz/cwzw6l3ip7esl3fltygji4wpw7duhp3ravrqni2plxnq4xvtcd54.py` 是实际 wrapper，可见 mean 被第二个 sum store 覆盖。
- `generated_on_failure/` 仅保留 F10 现场及 F8 对应 wrapper；其他缓存和原始完整日志保留在验证环境。
- `scripts/`、`execution/`：实际使用的 runner、依赖/设备适配、只读观测 hook、F3 方法和控制流程快照，作为证据保存。
  这些不新增测试方法；测试计算文件与上一轮逐字节一致，见各阶段 `test_equivalence.json`。
- `archive_manifest.json`：原始和脱敏文件的 SHA-256。home 路径、PID、backend_hash 已脱敏，生成代码格式保留。

## 执行方式

使用已适配的 [Combo 2.10 测试脚本](../../scripts/tests/combo_kernel_210/README.md) 和本目录观测插件。
原测试输入、精度阈值、梯度、突变检查保持不变；CUDA 专属结构检查的适配沿用上一轮。
每个方法启动独立 pytest 进程，各物理卡独立缓存；通过结果不保证采用 Combo 融合。

```bash
cd <validation-root>/<phase>
COMBO_PHYSICAL_DEVICES=1,2,3 COMBO_WITH_F3=0 python run_combo_selected.py
# with_fixes 及最终 full 阶段使用 COMBO_WITH_F3=1；仅替换旧 F3 分区方法。
python export_combo_results.py
python capture_combo_evidence.py
```

原样 wheel 的语法错误先单独记录，随后仅修正该缩进；原始包不具备直接开始本清单的条件。
旧版 F1 实现已在新包中，未重复覆盖。helper 修复来自 `98448611ecb64cbbca6df1f247ed28072ec896e7`。
F3 overlay 来自 `e578ec524d592475fd52702b975ce03d1a95bbf2`，只替换 `horizontal_partition()`，因此不会回滚 helper 修复。

定向阶段的原始证据包校验值为：
`f9c17e12e1abe724f7bc2f6ae3d59117efeba19ac7bdebea7444f21bbadea188`。

详细结果和责任边界见[验证表](../../docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md)。
