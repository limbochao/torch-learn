# F4：证明余数不进位后保留安全拆分

F4 单个修复提交 `53f8af09f144aaaec685562acd4c3a3ffaffb71a`，分支 `fix/combo-f10-f12-f11-f8-f4-f5-v2.10`，最新安装源码 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`。
已推送 origin；相对前一轮 `bc9d3c4ee973dde621ebf61600967ffbd66d0571` 仅 `torch_npu/_inductor/codegen/ir.py` 变化。

## 改动与安全条件

先提取能够证明整除的项，再尝试证明剩余各项余数之和小于除数。
新 helper `_floor_div_terms_do_not_carry` 要求各项为整数、除数为正整数，
合并 graph 的符号范围与 range tree 的有效循环索引范围，用 `PythonMod` 和 `bound_sympy` 计算余数上界。
对静态整数除数先约化整数系数，如 `(10*x)%8 == (2*x)%8`，以保留更多可证明的机会。
仅当所有余数上界总和严格小于除数才展开；未知范围、符号不明、暂不支持的范围分析保留合并除法。
`PythonMod` 保留负数项的 Python floor/mod 语义。多项加法检查总和，不能只检查两两相加。

`0≤x≤1、0≤y≤5` 时，`(10*x+y)//8` 的余数上界之和为 `2+5=7<8`，
可拆成 `(10*x)//8+y//8`，再消掉恒为 0 的后项，实际结果为 `(5*x)//4`。
y 上界变为 6 时余数可能恰好进位，拒绝拆分。
原 pooling 边界 `(8*i+10)//3` 在 i=0..5 时可能进位，继续合并。

## 验证

- [原 F4 方法](f4_nocarry_validation/evidence/functional_results.json)：Combo False/True，**2/2 PASS**。
  保留 `(2,4,16,16)`、`(2,4,3,3)`、`(2,4,6,6)` 输入及原有低精度、数值和梯度检查。
  原适配文件与上一轮安装源码验证的 SHA256 逐项一致，没有改动断言或精度阈值。
- [符号等价检查](symbolic_checks.json)：500 组随机范围、13,982 个取值点，含安全拆分、等号进位、
  三项共同进位、负数项、未知范围及不支持表达式。69 个域证明可拆分，所有取值点均与合并除法一致。
  [执行脚本快照](symbolic_check_snapshot.py) 归档为诊断证据，没有向生产仓新增正式测试。
- [安装校验](installation.json)与[运行完整性](f4_nocarry_validation/evidence/integrity_checks.json)：
  安装的是已推送提交的完整 ir.py；其余源码不变，11 项源码运行前后校验一致；无函数修复或 F3 overlay。
- [生成代码检查](codegen_checks.json)：语义 mask 在 load 前定义并用于 load，原 pooling 边界仍保留可能进位的合并除法。
  审计记录的 5 次 linear_combo 尝试全部成功。

环境为 NPU_A5 Docker、Ascend950PR，torch 2.10.0+cpu、torch_npu 包元数据 2.10.0.post7+git85dec1f、Triton 3.6.0。
仅复测本次 F4，使用物理卡 2；其余结果保留[上一轮 10/10 验证](../combo-fixes-85dec1f-20261106/README.md)。
使用已有 run_validation.py/run_combo_selected.py 汇总，未重新执行全量。

## 提交与风险

原 F4 a2f7b18da9 已与本次优化合并为 `53f8af09f1`，F4 仍只有一个提交。
后续 F5 从 bc9d3c4ee9 重放为 `a8dcca84bf`，补丁内容逐字一致，其结果仍引用原验证，未声称新版本重跑 F5。
分支从基线起仍有五个修复提交，生产提交只包含修复代码。
原始 [F4 完整补丁](f4-complete.patch)与[本次优化差异](optimization.patch)分别保存。

本次新增证明只在编译期执行，不增加运行时判断或新 shape guard；不能证明时保留原合并表达式。
风险是范围分析与新拆分机会可能改变生成代码和编译耗时；此次仅证明上述功能用例通过，没有测量性能收益或回退。
这是 NPU split_expression 的优化，社区 2.10/2.13 无相同 helper，不能称为逐行同步上游。
