---
title: Combo 五个修复提交逐行评审与上游对照
---

# Combo 五个修复提交逐行评审与上游对照

这组文档对应 [PR !7](https://gitcode.com/liujm2001/pytorch/merge_requests/7) 的五个提交。
每篇包含触发场景、失败证据、调用链、完整原始 diff、每一条非空新增/删除行的解释、上游对照、
正确性与性能风险，以及已有 NPU 验证。F10 和 F12 同源，因此六个问题写成五篇，和实际提交一致。

## 阅读入口

| PR 项 | 文档 | 修复提交 | 单提交修改量 | 上游关系 |
| --- | --- | --- | --- | --- |
| 1 | [F10 / F12：inplace 删除状态与最终渲染](inductor-combo-fix-f10-f12.md) | `64a3e7e8b9` | 1 文件，+5 / -1 | 沿用社区删除契约，适配 NPU 候选校验时序 |
| 2 | [F11：已删除中间 buffer 的残留 store](inductor-combo-fix-f11.md) | `9b1a894d65` | 1 文件，+2 / -1 | 同一删除契约的另一类状态，依赖第 1 项 |
| 3 | [F8：benchmark 尺寸收集副作用](inductor-combo-fix-f8.md) | `fdf73f229e` | 1 文件，+3 / -0 | 社区 2.10、2.13 已有对应空名称约定 |
| 4 | [F4：FloorDiv 与 pooling mask](inductor-combo-fix-f4.md) | `53f8af09f1` | 2 文件，+49 / -10 | NPU 专有优化修正，数学及访问语义对齐 |
| 5 | [F5：equal-to-one 常量 metadata](inductor-combo-fix-f5.md) | `a8dcca84bf` | 2 文件，+7 / -2 | 两版社区普通 Triton、Combo 均有同一登记行为 |

修改量由每个 `git show --numstat` 得到，包含注释和空白行，不表示有效语句数量。
整组相对基线累计为 **4 个文件，新增 65 行、删除 13 行**；
逐提交相加会重复计算被后续提交改写的注释，因此与累计差异不完全相同。

## 固定版本与证据口径

| 项目 | 版本 / 范围 |
| --- | --- |
| NPU 基线 | `85dec1f1a10ee42ebce87f1a1f8d2095909c0af6`，F3 已合入 |
| 五修复最终源码 | `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e` |
| 社区 v2.10.0 | `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` |
| 社区 v2.13.0 | `cf30153c4c131c8164ee7798e5022d810682e2cb` |
| 设备 / 软件 | Ascend950PR；torch 2.10.0+cpu；torch_npu 包元数据 2.10.0.post7+git85dec1f；Triton 3.6.0 |
| 实际加载方式 | 在上述包中安装最终提交的四个完整 Python 源文件；不加载函数修复或旧 F3 overlay |
| 最新全量完成 | 2026-11-07 01:31:59（北京时间） |
| 验证结果 | 62 个源方法中 60 PASS；409 个实例中 407 PASS、2 FAIL；0 skip、0 执行异常 |
| 六个修复问题 | 8 个方法、10 个实例全部 PASS；11 项安装源码运行前后哈希一致 |

包元数据仍显示基线 commit，因此不能仅凭 `torch_npu.__version__` 声称实际运行的也是未修复基线。
安装校验和最新执行上下文记录了四个 Python 文件对应的最终 commit；文档按实际加载源码归因。
相反，后续分支 HEAD 即使发生合并，也不能据此把先前结果改写成对新 HEAD 的独立运行。

2026-11-06 各单项诊断阶段曾使用测试进程中的函数 overlay；最终定向及最新全量使用完整安装文件，
两种阶段在文档中明确区分。F4 合并优化后变为 `53f8af09f1`，F5 因重放变为 `a8dcca84bf`；
最新全量已经覆盖这两个最终提交，不只沿用改写提交前的结果。

## “与上游对齐”如何判定

这里固定比较社区 2.10.0 和 2.13.0 的实际源码，不把“算子含义相同”写成“实现逐行一致”。

- F8、F5 有可以直接对应的社区分支或赋值语句，属于补齐上游已有调用契约 / metadata 行为。
- F10/F12、F11 使用社区同一套 DeferredLine 删除规则，但 NPU 先校验候选再提交 graph 状态，
  所以局部聚合与 handler 切换是 NPU 流程适配。
- F4 的拆分 helper 和额外 mask 过滤属于 NPU 逻辑，修复需遵守 FloorDiv 与 masked load 语义，
  不能用不存在的社区同名实现作为直接移植依据。

[社区 2.10 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)、
[社区 2.13 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)
与[NPU 调用链](../../../../artifacts/combo-five-fixes-review-20261107/npu-call-chain.md)
保留了固定版本的原样片段。每段来源文件、原始行区间和 SHA256 均记入
[manifest](../../../../artifacts/combo-five-fixes-review-20261107/manifest.json)。
这些片段用于复核，不宣称已经执行社区 CUDA 或其他硬件对照测试。

## 已有验证如何复查

[最新结构化结果](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)
记录最终十个目标实例和总统计；
[逐实例报告](../../../../artifacts/combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/functional_results.json)
与[完整性检查](../../../../artifacts/combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/integrity_checks.json)
包含实际执行结果及安装源码校验。
[现有总表](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md)继续维护最新状态，
本文档保存每个修复的解释，不用历史失败覆盖当前 PASS。

目标覆盖如下：

| 修复 | 源方法 | 最新实例结果 |
| --- | --- | --- |
| F10 / F12 | `test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda`、`test_mean` | 2/2 PASS |
| F11 | `test_mutated_benchmark`、`test_round_robin_dispatch`、`test_dynamic_shapes_mutated` | 3/3 PASS |
| F8 | `test_combo_kernel_size_hint_failure` | 1/1 PASS |
| F4 | `test_adaptive_avg_pool2d1`，Combo False/True | 2/2 PASS |
| F5 | `test_rand_like_deterministic`，Combo False/True | 2/2 PASS |

原社区 2.13 测试树需要已有的 2.10/NPU 导入及设备适配，不能直接复制未适配文件就认为可以执行。
执行沿用同一适配树、输入、stride、dtype、数值/梯度/突变断言和阈值。
CUDA 专属 kernel 数、源码结构或部分 grid 断言由既有功能适配处理，故功能 PASS 不等于满足 CUDA 的融合形式或性能要求。
本次文档任务没有更改适配或重新跑环境。

## 需要复测时的复用方式

使用[现有执行脚本说明](../../../../scripts/tests/combo_kernel_210/README.md)准备新的独立 phase。
定向清单用现有的
[八个方法清单](../../../../artifacts/combo-fixes-85dec1f-20261106/selected_methods.json)，
精确实例可使用本次从完整 409 节点中筛出的
[十个 node ID](../../../../artifacts/combo-five-fixes-review-20261107/target-nodeids.json)。
安装文件校验必须匹配要验证的提交；若换了新包，应重新生成校验契约，不能把旧结果当成新包结果。

在已准备的远端 Docker 工作目录内运行，变量分别指向新 phase 与已获授权的物理卡列表：

```bash
python run_validation.py --root "$COMBO_PREPARED_ROOT" \
    --devices "$COMBO_PHYSICAL_DEVICES" --expected-instances 10 --detach
```

不要带 `--with-f3`：本基线已包含 F3。脚本自动收集、执行、汇总及校验，避免重复人工查看日志。
若只复测某一篇，准备对应方法子集和匹配的 node ID 清单，预期实例数使用上表中的 1、2 或 3。
这条命令是复用说明，本文编写阶段没有执行它。

## 性能结论与剩余问题

当前只有功能和生成代码证据，没有性能 benchmark。前 1、2、3、5 项没有新增设备侧算法工作，
但这不构成所有模型无性能回退的测量结论。F4 涉及索引表达式、范围分析和语义 mask，
对编译时间及运行代码形态的影响更直接，应重点看对应文档的风险分析。
共享卡执行耗时不能作为稳定性能结论。

最新仍失败的是 F6（eager channels_last contiguous 不支持）和 F7（provenance 映射为空，AOTI 未执行）。
它们没有包含在本组五个修复中，也不能把总计 407/409 简写成全量全部通过。
本组文档只解释已提交的五个修复，剩余问题状态以总表为准。
