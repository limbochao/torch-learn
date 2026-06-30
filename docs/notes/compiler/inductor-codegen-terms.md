---
title: Inductor Codegen 术语说明
---

# Inductor Codegen 术语说明

- DSL：Inductor 生成的类 Triton Python kernel。
- reduction lane：参与求和的维度。
- tile：一个 program 内被切分出来的子块。
- `tl.load` / `tl.sum` / `tl.store`：分别对应读、归约、写回。
- stride：张量在内存中的步长。
- tiling：把大维度拆成多个块处理。

