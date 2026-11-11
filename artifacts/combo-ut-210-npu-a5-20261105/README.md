# NPU_A5 Combo/Foreach 2.10 验证证据

验证日期：2026-11-05。62 个方法全部结束：44 个全部通过、18 个失败。
409 个核定参数化实例中，389 个通过、20 个失败、0 个跳过；未执行核定范围外实例。

- `functional_results.json`：逐方法最终结果；同一实例复测时使用最新结果。
- `collection.json`：409 个核定实例；`collection_preflight.json` 保留最初的前缀匹配收集结果。
  `excluded_prefix_collisions.json` 记录执行前排除的 21 个额外实例。
- `jobs.jsonl`：64 次任务的 start/finish、命令、物理卡和返回值；含两个测试侧适配复测。
  pilot 为 3 个方法，remaining 为 59 个方法，两个复测各 1 个方法。
- `test_*.jsonl/.log/.xml`：逐阶段报告、控制台输出和 JUnit 结果。
  复测 JSONL 保留前后两次记录；`attempt1_profiler/`、`attempt1_cuda_structure/` 单独保存首次失败。
- `environment.json`：真实安装包版本；SSH 登录欢迎页的旧版本提示不作为环境依据。
- `installed_source/`、`installed_source_sha256.json`：8 个安装源码快照与校验和；
  7 个 NPU 文件匹配目标分支，1 个社区文件匹配上游 v2.10.0。
- `original_tests/`、`executed_tests/`、`test_source_sha256.json`：5 个测试文件的原始及执行版本。
  三个 `.patch` 和 `adaptations.json` 记录测试侧适配；未修改安装包。
- `generated_python/`：最终缓存中保留的生成 Python 文件；manifest 含校验和。
  `generated_on_failure/` 是部分失败发生后、tearDown 清理前捕获的缓存，可能含同卡较早生成的文件。
- `scripts/`：最终执行脚本快照；`initial_scripts/` 保留早期快照，数值测试逻辑不因观察性捕获而改变。

用例通过表示适配后的功能检查通过，不保证使用 Combo，也不保证满足 CUDA 的 grid、融合数量或性能要求。
失败的具体归因和逐项表格见
[验证报告](../../docs/notes/compiler/codegen/inductor-combo-tests-v2.10.md#native210)。
