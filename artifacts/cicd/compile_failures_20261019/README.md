# CICD compile failure evidence

使用说明、问题定界和验证命令见
[脚本说明](../../../scripts/repro/cicd/compile_failures_20261019/README.md)。

每个模型保存完整 generated kernel、所选 compile journal、候选配置汇总、整模型日志和独立编译日志。
仅去除个人路径、临时目录标识和 backend_hash；kernel body 保持原始格式。
