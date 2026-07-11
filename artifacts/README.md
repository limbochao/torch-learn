# artifacts

本目录保存问题定位和源码分析中需要长期引用、但不适合作为正文或可执行脚本维护的生成产物。

产物按主题建立子目录，并由对应的 `docs/issue-logs/` 或 `docs/notes/` 文档引用。临时日志和可重新生成且没有引用价值的文件不提交。

## 当前主题

- `dlrm-dynamic-vs-static-codegen/`: DLRM 在 `dynamic=True` 和 `dynamic=False` 下的 FX readable graph 与
  Inductor generated code 对照产物。
