# 最新 main 上的 NPU adapter 验证（2026-10-15）

本轮接续会话 `01a12f0c-7bd7-7103-81c4-9815b767ddd4`，将原适配分支的最新修改迁移到最新主线，
并以实际运行结果更新 [331 行总表](../results.csv)。

## 代码与环境

- 仓库：[Community-Use-Case-Testing](https://github.com/limbochao/Community-Use-Case-Testing)。
- 基线 `main`：`762c46edcf42bcfabfa63758be23ba3ee4fa3e98`。
- 来源 `ut/adapter_to_npu`：`2b5b9b2c1e80a8bd7b4299bae12de2a35c1034a5`。
- 新分支：[ut/adapter_to_npu_main_20261015](https://github.com/limbochao/Community-Use-Case-Testing/tree/ut/adapter_to_npu_main_20261015)。
- 验证提交：`99d0217eb9e0b5f51feb02ef6ec3c0b547dbf30b`。
- NPU_A5 Docker，Conda `pta_213_case`，PyTorch `2.13.0+cpu`，torch_npu `2.13.0+git472321b`。
- 测试目录：`v2.13.0/inductor_npu`；使用物理卡 1、3、4、5、6，每个测试进程只暴露一张卡。

主线已包含 static launcher、动态 shape、NPU 内存检查及四个代码生成断言的适配。
迁移时保留主线的新兼容处理，去除已经合入的重复修改；最终相对主线仅有两个文件的改动：

- `test_inplacing_pass.py`：将 `const` 建在 `GPU_TYPE` 指定的设备。
- `test_torchinductor.py`：NPU 环境放行 round value-expression 用例的 CUDA/Triton 装饰器，
  保留主线 `has_triton=False` 兼容模式。

文件哈希和已跟踪文件无修改的检查见 [environment.json](environment.json)。

## 范围与结果

原 98 行重跑清单、后续 24 行适配记录，以及 12 行 inplacing/round/compile 用例，共 134 行、156 个节点。
完整目标及节点映射见 [resolved-targets.json](resolved-targets.json)，无遗漏节点。
各文件分别收集，按文件分批运行；没有加载 retry_gates，也没有运行时拦截 skip。

| 阶段 | 行结果 | 节点结果 |
| --- | --- | --- |
| 原安装环境 `stock/` | 125 PASS、9 FAIL | 147 PASS、9 FAIL |
| 后端修复复测 `with-backend-fix/` | 7 PASS | 7 PASS |
| 每行取最后一次实测结果 | 132 PASS、2 FAIL | 154 PASS、2 FAIL |

五个 round 用例失败于 FX `dtype_optimal_pass` 对 `SymFloat` 读取 `.dtype`；
`test_compile_decorator` 和 `test_mark_unbacked_slice` 失败于 CPU PyTorch 缺少
`_cuda_cudaCachingAllocator_is_enabled`。已发布的后端提交 `4f5017beec` 修复这两个原因。
应用 [原提交补丁](backend-4f5017beec.patch) 后，七个原始 NPU 节点全部通过。
补丁只应用在本轮工作目录下的独立 `torch_npu` 副本，通过 `PYTHONPATH` 加载；原安装文件哈希保持不变。
见 [补丁前后哈希](backend-overlay.json) 和 [复测终态](backend-retest-completed.json)。

两个不支持的数据类型用例未修改：

- 第 2 行 `test_unsigned_integers`：`aclnnInplaceZero` 拒绝 `DT_UINT64`。
- 第 17 行 `test_any`：`libdevice.isinf` 对 `triton.language.float64` 抛出 `KeyError`，导致编译失败。

这两行本次确实执行并失败，因此记录为 FAIL，附不支持原因；没有把失败转换成 pytest SKIP。
其他 197 行没有重新执行，保留其原结果和日期。总表合并后的统计不代表本轮全量执行了 331 行。

## 证据与复现

- [summary.json](summary.json)：本轮、原环境、补丁环境及总表统计，含逐行前后结果。
- [node-results.json](node-results.json)：原环境与补丁环境的节点、结果、原因、命令、卡号和 JUnit 路径。
- [validation.json](validation.json)：节点覆盖、退出码一致性、行序与未复测行保持不变的检查。
- `stock/job-*/` 与 `with-backend-fix/job-*/`：JUnit XML、命令和进程退出结果。
  原始 pytest 日志保留在远端本轮验证目录中。

在 NPU_A5 Docker 的 `pta_213_case` 中检出验证提交，进入 `v2.13.0/inductor_npu`，
按相应 `command.json` 的 `env` 和 `cmd` 复现。路径中的 `<test-env>`、`<case-repo>` 和
`<validation-workdir>` 分别替换为 Conda 环境、用例仓库和本次验证目录。
七个依赖后端修复的用例必须先应用 `4f5017beec` 或使用包含等价修复的安装包；
只切换用例分支时，应以总表的 `NPU_A5原安装环境结果` 为准。
