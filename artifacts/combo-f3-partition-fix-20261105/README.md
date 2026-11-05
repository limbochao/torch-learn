# F3 修复复测与 F9 helper 缺失证据

验证日期：2026-11-05。设备：NPU_A5（Ascend950PR），Docker 内使用物理卡 2、3。
环境：torch 2.10.0+cpu、torch_npu 2.10.0.post7+gite32cdeb、Triton 3.6.0、CANN 9.1.0。

安装基线为 `e32cdebf77e797fc85a1a8c3fe1e0eef050f9c0e`。
F3 修复提交为 `e578ec524d592475fd52702b975ce03d1a95bbf2`，
父提交为 F1 修复 `6c6454567f795cb580ca07da68884e15bdd16c6a`；
已推送至 origin 分支 `fix/v2.10.0-combo-partition-numellist`。

## 实际加载与结果

通过进程内方法替换加载 `NPUComboKernel.horizontal_partition()` 和已有 F1 的
`kernel_benchmark_extra_args()`，共享安装包保持不变。
候选文件 SHA-256 与最终提交一致，具体值见 `validation_summary.json`；
`installed_source_checks.json` 中的 8 项安装源码校验仍匹配基线。
其他代码沿用基线，不能将安装包版本号改写为 F3 提交。

沿用既有测试依赖适配和 CUDA 专属结构检查的移除，保留输入、输出和精度比较；
Combo benchmark 保持开启。执行方式沿用
[原功能验证脚本说明](../../scripts/tests/combo_kernel_210/README.md)。
当次在独立验证目录使用候选加载插件，命令形式为：

```bash
cd <validation-root>/Community-Use-Case-Testing/v2.13.0/inductor_npu
ASCEND_RT_VISIBLE_DEVICES=<allowed-physical-device> \
PYTHONPATH=<validation-root> \
COMBO_SELECTED_MANIFEST=<validation-root>/combo_npu_a5_selected.json \
COMBO_RESULTS_JSONL=<validation-root>/evidence/<method>.jsonl \
TORCHINDUCTOR_CACHE_DIR=<validation-root>/cache/<device> \
TORCHINDUCTOR_COMPILE_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p combo_validation_plugin -p f3_validation_patch -q -rA \
    --tb=short --disable-warnings \
    test_combo_kernels.py::ComboKernelDynamicShapesTests::test_helper_fn_defined
```

其中 `f3_validation_patch` 是当次验证专用加载器：从候选源码加载上述两个方法，
观察社区分区入口的尺寸类型，并在返回时检查 `node.group` 恢复为原始对象。
该命令用于说明当次运行方式；重新执行需准备原测试适配、候选源码及相同加载器。
本目录 generated code 为只读分析产物，不作为独立可执行 repro 维护。

七个原 F3 方法中六个通过，`test_helper_fn_defined` 越过 F3 后因 F9 失败。
`test_*.jsonl` 保留逐阶段结果，六个通过方法的 setup/call/teardown 均通过；
七个方法的 `*.partitions.json` 共记录 13 次分区调用，原分组对象均已恢复。
`f1_regression/` 保存二维 blocking、activations 两个 F1 回归方法的结果，均通过，
共 6 次分区观察也确认分组恢复。未关闭 benchmark 绕过故障。

当前合并统计为 55/62 个方法、400/409 个实例通过；其余 7 个方法、9 个实例失败。
这是全量基线、F1 和 F3 定向复测的合并状态，不是一次全量重跑；
F1 回归两个已通过方法不重复增加总数。

## F9 的生成代码证据

`test_helper_fn_defined` 的被测函数为：

```python
def fn(x, y, z):
    return x.sum(1), y.mean(1), z.cumsum(1)
```

三个输入均为连续 float32 张量，shape 为 `(16, 128)`、`(32, 128)`、`(32, 256)`，
stride 分别为 `(128, 1)`、`(128, 1)`、`(256, 1)`。
它位于动态测试类中，实际输入尺寸按该类设置符号化；保存的生成签名也包含动态尺寸参数。
原测试的 helper 定义次数结构检查不是本轮功能通过的前提；当前在执行编译结果前就已失败，
因此移除 CUDA 结构检查不会消除这个编译错误。

- `standalone_scan_with_helper.py`：当次独立 scan kernel，含一个 `@triton.jit` 加法 helper 定义及一次调用。
- `combo_scan_missing_helper.py`：当次失败的完整 Combo kernel，含三个子 kernel，scan 分支有一次 helper 调用，零个定义。
- `helper_failure.txt`：保留失败阶段与内层 `NameError` 的 traceback。

Triton 在解析如下调用时找不到全局函数，外层调优器最终报告 `NoTritonConfigsError`：

```python
tmp15, = tl.associative_scan((tmp13,), 1, _triton_helper_fn_add0)
```

这份 generated DSL 缺少必需定义，当前责任模块是 NPU Inductor Combo codegen。
外层错误不证明 tiling 搜索不足，也不能解释为 Triton 缺少 cumsum 支持。
独立 kernel 只用于确认定义的生成与保存，不能由其源码单独推定任意独立 scan 场景数值正确。

## 源码原因与修复方向

- `source/upstream_lift_helper.py`：上游 2.10 的 `TritonKernel._lift_helper()` 生成 helper，
  将定义登记到 `self.helper_functions` 并返回函数名。NPU `scan()` 沿用这一登记机制。
- `source/upstream_combo_codegen_kernel.py`：上游 2.10 的 Combo 遍历所有子 kernel 的
  `helper_functions`，用 `seen_helpers` 按完整定义去重，并在主 kernel 之前输出。
- `source/npu_combo_codegen_kernel.py`：NPU 基线的重写方法只拼接 body，没有合并 helper。

对基线、F1、F3 三个提交的 NPU `codegen_kernel()` 做 AST 比较，结果完全一致，
见 `validation_summary.json` 的 `source_evidence`。F9 在基线中已存在，
F3 修复让这个用例越过分区失败后暴露 F9；它不是 F3 新增的代码缺陷，也不依赖 2.13 新接口。

建议在 NPU Combo 中补齐上游的收集、去重及输出逻辑，不针对 `_triton_helper_fn_add0` 特判。
修复范围是 Combo 中依赖子 kernel helper 的生成路径；目前仅此用例实际确认失败，
其他 helper 场景的影响仍需验证。未来应复测当前用例的输出与生成定义，并覆盖多个子 kernel
共享同一 helper 时的去重，确认没有后续编译或运行错误。

本次只分析 F9，未修改生产代码或新增远程运行。F9 状态为未修复、仍阻塞该用例。
本目录保留最终有效复测的证据；首次观察脚本参数名错误的无效尝试未计入结果。
产物已泛化个人路径和完整 backend_hash，保留生成 kernel 排版及计算逻辑。
