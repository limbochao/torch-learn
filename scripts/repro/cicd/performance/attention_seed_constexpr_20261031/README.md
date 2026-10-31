# 首层 attention seed constexpr：TTIR 对照复现

[分析文档][analysis] · [原始 checker][checker] · [已采集结果][results]

目标是复现 Electra Train 首层多轴 softmax + dropout 在读取 seed offset 时的 TTIR 失败，
并用两种只改变 metadata 的对照定位常量绑定问题。
脚本从归档的生成代码创建临时副本，不修改已安装的 torch / torch_npu / Triton。

## 环境和运行

已采集环境：Ascend950PR_9579，torch `2.13.0+cpu`，torch_npu `2.13.0.dev20261030`，
Triton `3.6.0`。需要相应 NPU 版 Triton、CANN 和 torch_npu 环境。
`_triton_make_ttir` 等是该版本的内部接口，后续版本接口或修复状态可能不同。

在目标容器的工作目录中激活上述环境，按环境约定选择允许使用的空闲物理卡，
设置 `ASCEND_RT_VISIBLE_DEVICES`，然后从仓库根目录运行：

```bash
python scripts/repro/cicd/performance/attention_seed_constexpr_20261031/check_ttir.py \
  --output-dir /tmp/attention-seed-ttir
```

可用 `--case original`、`--case constant_bound` 或 `--case runtime_i32` 只运行一组。
输出目录保存生成副本、成功的 TTIR、异常日志和 `ttir_check_results.json`。
所有指定 case 均符合下表预期时退出码为 0；环境导入失败或结果不同则退出非 0。
输出日志可能包含本机路径，归档前需去敏。

| case | metadata | 当前缺陷版本的预期 |
| --- | --- | --- |
| original | `load_seed_offset: constexpr`，`constants={}` | 在 seed load 处出现 NoneType 异常 |
| constant_bound | 仅补 `constants["load_seed_offset"]=1` | TTIR 生成成功 |
| runtime_i32 | 仅将该参数的 signature 改为 `i32` | TTIR 生成成功 |

## 输入和验证边界

计算体固定为 `[32, 4, 512, 512]` 的 FP32 attention scores，contiguous stride 为
`[1048576, 262144, 512, 1]`，缩放系数 1/8，dropout 概率 0.1，static shape。
seed 来自一维 int64 buffer，出错的首层 offset 为 1。

本脚本只生成 TTIR，不分配上述真实输入、不执行设备 kernel，也不测性能或验证整模型精度。
它保留原 checker 的所有输出、mask、索引和计算体；配置由同一个 `_select_ttir_test_config` 选择。
原始采集的三组配置一致，详见归档结果。

2026-10-31 已完成的远程对照结果保存在 artifacts 中。
本次文档归档只对新包装脚本做了本地语法和入口检查，未重新运行 NPU 对照。
若框架已修复，original 可能通过，此时脚本会用非 0 退出码提示原缺陷未重现。

[analysis]:
  ../../../../../docs/issue-logs/cicd/performance/electra-layoutlm-train-first-attention-20261031.md
[checker]:
  ../../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/failed_first_attention_checker.py
[results]:
  ../../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/ttir_check_results.json
