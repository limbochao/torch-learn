# PTA FX 图回归用例

比较 NPU eager 与 Inductor 编译结果，并检查 `fx_graph_readable.py` 是否符合对应 PyTorch 版本的预期。

| 用例 | 覆盖内容 |
| --- | --- |
| `test_aten_triu_inplace_in_graph.py` | 图内原地 triu，静态和动态形状 |
| `test_nanquantile_defaul.py` | Tensor q 的 nanquantile |
| `test_nanquantile_scale.py` | 标量 q 的 nanquantile |
| `test_quantile_scalar.py` | 标量 q 的 quantile |

`testutilsops.py` 是这些用例共用的测试基类。

## 版本预期

版本判断使用 `current_version >= pt212__version`，依据 `torch.__version__` 选择 content。
PyTorch 2.12 起使用新预期；更早版本保留原有预期及 2.9 分支。

- triu：原生 PyTorch [PR #175582](https://github.com/pytorch/pytorch/pull/175582)，
  提交 `2f93063b00b`，修改 `meta_copy_` 的形状检查，使本例动态掩码节点从 `ge_22` 变为 `ge_26`。
- quantile：原生 PyTorch [PR #176804](https://github.com/pytorch/pytorch/pull/176804)，
  提交 `be7dbd8a340`，新增 Inductor `lerp.Tensor` 分解，插值部分改用 `addcmul`。
- 保留 addcmul 的配套修改：[PR #175309](https://github.com/pytorch/pytorch/pull/175309) 和
  [PR #175839](https://github.com/pytorch/pytorch/pull/175839)。

已检查发布标签的提交祖先关系和源码：上述变更不在 2.10.0、2.11.0 中，首次进入正式版本 2.12.0。
该版本边界面向正式发布版本；自定义回移提交或开发版需按实际源码确认。

## 运行

需要已配置 CANN 的 NPU 容器、匹配的 torch / torch_npu / Triton，以及 PyTorch 内部测试模块所需依赖。
在容器工作目录内进入本目录，激活目标 Python 环境，按脚本逐个运行。例如：

```bash
cd scripts/tests/pta_case
export ASCEND_RT_VISIBLE_DEVICES=2
python test_aten_triu_inplace_in_graph.py
python test_nanquantile_defaul.py
python test_nanquantile_scale.py
python test_quantile_scalar.py
```

保留了原用例的缓存清理行为：每个脚本启动时删除 `/tmp/torchinductor_root` 和当前目录的
`torch_compile_debug`。应串行运行；需要保留某次产物时，在启动下一个脚本前复制其目录。
如果另设 `TORCHINDUCTOR_CACHE_DIR`，重跑时应使用新缓存目录，避免缓存命中导致缺少 FX trace。

输出包括 unittest 结果、预期与实际的 FX 文本，以及 `torch_compile_debug/` 内的图和生成代码。

## 验证

2026-10-07 在 NPU_A5 上验证：

| PyTorch | torch_npu | 结果 |
| --- | --- | --- |
| 2.10.0+cpu | 2.10.0.post7.dev20261007 | 四个脚本，共 5 项通过 |
| 2.13.0+cpu | 2.13.0.dev20261007 | 四个脚本，共 5 项通过 |

验证副本仅移除了上述缓存删除语句，使用独立工作目录与缓存，保留全部数值和 FX 文本断言。
随后版本判断改为现有 `Version` 对象比较风格，对上述两个正式版本的分支选择不变。
2.11 和 2.12 只核对了源码边界，未执行对应环境的运行验证。
