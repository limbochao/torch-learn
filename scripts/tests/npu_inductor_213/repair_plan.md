# 101 个失败及跳过用例的简单修复评审

基于 2026-10-08 的 247 项测试归档（提交 `c55748f`），共同评审 87 个失败及 14 个跳过逻辑用例。
原状态分别保留；这里的 14 项不包含已执行后的 xfail，也不将解除跳过直接记为通过。
优先考虑修改测试代码或 `torch_npu/_inductor`；需要其他模块的项在 CSV 中单独标明。
87 个失败项的修复建议目前仍是源码评审；14 个跳过项已完成隔离测试副本的 NPU 临时验证。
最终 8 项通过、2 项失败、4 项仍跳过，其中 Conv/BN 有一项需要局部关闭卷积 HF32。
原测试结果保持不变，见 [临时验证说明](skip_validation.md) 和
[临时验证 CSV](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation.csv)。

- [逐用例修复建议 CSV（101 行）](../../../artifacts/npu-inductor-213-batches-20261008/repair-triage.csv)
- [跳过项适配建议 CSV（14 行）](../../../artifacts/npu-inductor-213-batches-20261008/repair-skip-triage.csv)
- [按共同修改点合并的 CSV（40 组）](../../../artifacts/npu-inductor-213-batches-20261008/repair-groups.csv)
- [评审数据](../../../artifacts/npu-inductor-213-batches-20261008/repair-plan.json)
- [补充源码证据](../../../artifacts/npu-inductor-213-batches-20261008/repair-source-evidence.json)
- [原始结果与验证环境](README.md)

下表分类保留补测前的方案评审，实际验证结论另列于 CSV 的临时验证列。
这里的“简单”指修改位置少、原因明确、可保留测试目标和断言；并不只按代码行数判断。
删除断言、关闭被测编译模式、把不支持的功能直接 skip，都不能算功能修复。

| 分类 | 失败项数 | 跳过项数 | 合计 | 含义 |
|---|---:|---:|---:|---|
| A：优先尝试简单修复 | 25 | 6 | 31 | 当前首错或跳过条件有明确的局部适配方向 |
| B：有条件的小改候选 | 8 | 5 | 13 | 先确认确定性、依赖、精度、模板或性能条件 |
| C：改变 dtype 覆盖的简单适配 | 4 | 0 | 4 | 可增加受支持类型的 NPU 变体，原 fp64 缺口仍保留 |
| D：其他模块的局部修复候选 | 2 | 0 | 2 | eager 右移广播、合法空 unpool；不能只修改 Inductor |
| E：不列入简单修复 | 48 | 3 | 51 | 需要功能开发、跨模块适配或进一步定位 |

**31 是优先尝试的小改候选数（25 个失败、6 个跳过），不是预计修复后通过数。**
当前报告主要定位首次错误，修改后仍需继续执行剩余分支；参数化实例也不能只验证其中一个。

## 跳过项：6 项优先适配，5 项带条件试跑，3 项暂不解除

| 修复组 | 原序号 / 用例 | 项数 | 建议 |
|---|---|---:|---|
| S01 / A | 163、165、166、167：大张量 reduction/pointwise/offset/stride | 4 | 补 NPU 内存检查，满足原门槛后执行原方法 |
| S02 / A | 204：`test_upsample_cat_conv` | 1 | 增加 NPU 分支，保留原模型、输入和 common 断言 |
| S03 / A | 245：`test_float_item_return` | 1 | 对已验证 NPU 配置细化 FBCODE 限制，保留 fullgraph 和标量捕获 |
| S04 / B | 197：`test_linear_dynamic_maxautotune` | 1 | 改按实际 NPU Triton 模板能力决定是否执行 |
| S05 / B | 96：`test_progressive` | 1 | 先确认 NPU 后台优化、模板、测时和性能条件 |
| S06 / B | 201、202：Conv/BN 融合 | 2 | 启用 slow 测试并核验 NPU 精度，细化历史 GPU 限制 |
| S07 / B | 50：`test_ensure_fp4_dtype_registered` | 1 | 真实 CUTLASS 依赖可导入后，作为主机端映射测试执行 |
| S08 / E | 42、45、46：TMA lookup 与多模板配置 | 3 | 当前不建议只删除 skip；仍缺少 NPU lookup/模板接入 |

跳过项 CSV 新增“是否建议解除跳过”列，可直接筛选优先项或条件项。
以下是补测前提出的方案；原结果仍为跳过，解除跳过后的结果单独记录。

S01 的首个阻塞并不是实际内存不足，而是 `_has_sufficient_memory` 不认识 `npu`。
可在测试文件增加局部 NPU 大张量适配器，让 NPU 检查真实可用内存/进程配额，其他后端仍使用原装饰器。
也可统一修改 `torch/testing/_internal/common_device_type.py`，但这是额外的测试基础设施模块，CSV 已标明。
原用例的 4/4/3/2 GiB 门槛和 cpp-wrapper 额外分配估计必须保留。
不能缩小 `2^32`、`2^31+1`、`2^30` offset 或 stride=32 来“通过”，那会丢失 64 位索引的目标。
满足门槛后建议逐项串行执行；实际内存不足仍允许带具体原因跳过。

S02 的函数体是普通 nearest upsample、cat 和 Conv2d，没有 CUDA 专用代码或 CPU 文本断言，
因此可先针对 NPU 细化笼统设备限制，再运行原 `common` 的数值和低精度检查。
S03 的函数体也没有直接 FBCODE import；可针对实际支持的 NPU 配置试启用，
同时补返回标量为 `3.0` 的断言。不能全局设置 `IS_FBCODE=True`、允许 graph break 或关闭 fullgraph。

S04/S05 的 `IS_BIG_GPU` 最终使用 `is_big_gpu`，非 ROCm、非 XPU 分支采用至少 68 个 SM 的门槛。
这不能直接当作 NPU Triton 能力判断。源码评审已确认，NPU 自己的 `use_triton_template` 检查
NPU 设备/dtype、max_autotune、TRITON backend 和 `TRITON_TEMPLATES`，不使用该 SM 门槛。
但它存在不代表特定输入一定有合法候选，所以仍列为条件项。
S04 必须保留 TRITON-only 并确认实际模板，不能改成 ATEN 回退后宣称覆盖原路径。
S05 还必须保留后台优化计数、300 秒超时、max-autotune 产物和优化版快于 baseline 的原要求。
第 96 项有两个收集实例；补测已在两者复现异步 lowering 状态恢复失败，完整结果见临时验证说明。

S06 注释明确记录了历史 GPU 精度问题，不能简单把它解释成没有 NPU 算子。
第 201 项先需要 `PYTORCH_TEST_WITH_SLOW=1`，随后还要处理函数体的 CPU-only 条件；
第 202 项只有后者。适配 NPU 后必须保留原容差、低精度检查和 1D/2D/3D 等组合，
不能跳过精度不符的组合。两项也可独立执行 CPU 版本，但不计作 NPU 通过；
若要证明 Conv/BN 确实融合，还需图或生成代码证据。

S07 调用的是 CUTLASS 的 Python 类型映射，没有运行 GPU kernel。
因此可先确认真实 `cutlass`/`cutlass_api` 在目标环境可导入，再按依赖可用性执行，
不必把 CUDA 硬件可用性当成唯一前提。当前直接缺少 `cutlass_api`，光删除 skip 会变成导入失败；
包是否还有 CUDA 运行依赖须确认，本轮没有安装。即使该项通过，也不证明 NPU FP4 算子可用。

S08 中第 42 项明确测试实际 TMA 模板，保留能力限制。
第 45/46 项的多配置/混合模板意图可在 NPU lookup 接入后新增原生变体，
但要真实验证两个配置或两种模板，不能都替换成同一个普通模板。
当前还受 R17 的 key/模板接入缺口阻塞；把兼容 capability 改为 9.0 或令 TMA 检查恒真不构成适配。

## 失败项中优先修改的 7 组

表中编号是原累计 CSV 的“序号”，重名方法仍按测试类区分。

| 修复组 | 原序号 / 数量 | 修改位置 | 建议 |
|---|---|---|---|
| R01 | 132–136 / 5 | NPU Inductor 的 `dtype_optimal_pass` | 先确认 `Tensor.to` 和 Tensor 元数据，再访问 `.dtype` |
| R02 | 51–55 / 5 | `test_inplacing_pass.py` | 将 0 维常量按测试输入设备创建 |
| R03 | 68、69 / 2 | `test_torchinductor.py` 的 custom-op 测试 helper | 为同一实现增加 `PrivateUse1` 注册 |
| R04 | 74、82、83、183、184、187、240、241 / 8 | `torch_npu/_inductor/__init__.py` | 只在 CUDA 为迁移别名时排除重复设备条目 |
| R05 | 243 / 1 | 动态 shape 常量折叠用例 | 将全部 `source_codes` 拼接后做原负向代码检查，允许空列表 |
| R06 | 48 / 1 | `test_utils.py` 的方法归类 | 将纯 FakeTensor FLOPs 检查放入不做设备实例化的测试类 |
| R07 | 190、195、196 / 3 | `test_torchinductor.py` 的负向用例 | 保留非法输入，为 NPU 选择精确的异常类型和信息 |

建议先做 R01、R02、R05，它们的修改边界最直接；随后处理 R03、R04，再检查 R06、R07。
完整位置、原失败原因、源码和日志引用见逐用例 CSV。

R01 的首错来自 `SymFloat` 被当作 Tensor 读取 `.dtype`。拟议条件顺序如下，后续 cast 逻辑不变：

```python
if node.op == "call_method" and node.target == "to" and node.args:
    input_node = node.args[0]
    input_fake = (
        input_node.meta.get("example_value")
        if hasattr(input_node, "meta") else None
    )
    if not isinstance(input_fake, torch.Tensor):
        continue
    input_dtype = input_fake.dtype
    # 继续原有 Tensor cast 处理。
```

这段是修改方向示例，尚未应用。必须同时回归原 Tensor cast 和 `arange` 分支，不能关闭整个 pass。

R02 要保留 `index_put/index_put_`、live Tensor、view 和输入突变检查。
例如调整初始化顺序为先确定 `device = GPU_TYPE`，再创建 `const = torch.tensor(0.0, device=device)`。
不建议在每次 forward 中增加 `.to()`，以免把额外搬运混入本来要检查的图。

R03 的注册 helper 本身位于 `test/inductor/`，属于允许修改的测试代码，不需要改生产 dispatcher。
正确 Meta 的 stride 应保持为 1040；错误 Meta 用例仍需验证原 storage offset 对齐错误。
如果注册修复后发现 NPU 未生成对应对齐断言，应继续记录后续缺口，不能改为接受任意异常。

R04 的公共首错发生在 benchmark 默认设备推断：NPU 初始化同时登记 `cuda` 和 `npu`，
而 transfer 使二者可用性都为真。可在 NPU 初始化中有条件排除 CUDA 别名，无须直接改社区 helper。
必须保留真实 CUDA/XPU 的情况，不能无条件把全局设备表改成只有 NPU。
第 82 项直接测试 profiler benchmark，也可以只在该测试调用处传 `device_type=device`。
第 83 项的 helper 还有后续循环，第 187 项还有 BMM 分解断言，首错修复不代表这些检查已通过。

R05 可对 `"\n".join(source_codes)` 执行原 `check_not`，然后继续全部数值和不同 shape 断言。
已有 3 个函数 × 两种 dynamic 设置的独立探针表明零 kernel、返回原输入是可能的合法结果。
不能因代码列表为空直接退出测试。

R06 的方法体检查 FakeTensor FX 元数据，不使用 `self.device`。
调整归类后仍须执行全部 FLOPs 正反例；它只能证明主机端元数据逻辑，不能记为 NPU 流功能通过。

R07 不能使用 `assertRaises(Exception)` 或宽泛的 `error` 正则。
第 196 项还包含 no-grad 和启用 autograd 的 compiled 分支，必须分别确认对应错误。
第 195、196 项对当前原始输入的拒绝，不足以证明所有受支持 dtype 之间的混用都被正确拒绝。

## 失败项中需要先确认条件的 8 项

| 修复组 | 原序号 | 修改位置 | 前置条件 |
|---|---|---|---|
| R09 / 6 项 | 86、87、89、90、91、92 | NPU autotuner `_bench_with_launch_args` | 候选和 reset/clone 行为满足确定性契约 |
| R10 / 1 项 | 94 | extension 测试的模块路径与 worker fixture | 子进程在创建时就能导入模块，已有 worker 也得到处理 |
| R11 / 1 项 | 93 | dummy PrivateUse1 测试进程和清理逻辑 | 子进程未加载 NPU backend，父进程注册保持完整 |

R09 有直接接口对照：当前社区 `CachingAutotuner.bench` 已向 benchmark 传入
`is_vetted_benchmarking=True`，NPU 对应调用没有传递。补一个参数可能解除首错，
但必须确认 NPU reduction、atomic、自定义 kernel 和输入突变候选的适用范围。
不能仅为通过测试全局关闭 deterministic 或无条件批准所有 benchmark。

R10 原测试已经修改父进程 `sys.path`，重复追加无效。
必须处理生成代码 import 和跨进程 pickle 的同一个模块身份，以及 worker 已启动的情况。
第 95 项独立运行还有 AIVEC 地址错误，不能因模块导入修好就把它算作通过。

R11 测试的是 dummy backend 注册，不是已占用 PrivateUse1 的 NPU runtime 再注册一次。
在干净进程运行该测试可保留目标；清理逻辑只能删除测试自己注册的对象。
仅禁止删除 `torch.npu` 不能解决原注册冲突。

## 可以简单改用例，但不能算原 fp64 修复的 4 项

| 原序号 / 用例 | 可新增的 NPU 变体 | 仍缺少的覆盖 |
|---|---|---|
| 17 `test_any` | float32 的 any/all、isinf 与 Inf 输入 | float64 isinf；也不能据数值通过推断 static launcher 已启用 |
| 131 `test_round_correctness` | float32 round 边界输入 | 原最高精度 float64 路径 |
| 210 `test_device_assert` | 全 1 的索引输入 `y` 使用 float32 | float64 trunc；不是设备 assert 硬件行为的单独证明 |
| 224 `test_to_dtype` | bf16/fp32/bool 转换，参照已有 MPS 分支的适配思路 | double load、运算和转换 |

如果当前目标是验证 NPU 已支持类型，可优先增加这些明确命名或参数化的变体。
不要覆盖原始 dtype 的记录，也不要全局把 `highest_precision_float(npu)` 改为 float32，
因为这里的证据是特定算子/lowering 缺口，不是所有 NPU 算子都不支持 double。
第 182 项 `test_linear_float64` 的目标本身就是 float64，不能采用此类换 dtype 方案修复。

## 涉及其他模块的局部修复候选

| 原序号 | 问题 | 必须标明的其他模块 | 判断 |
|---|---|---|---|
| 73 | Tensor 右移输出按 self 而非广播结果分配 | `op-plugin` 的 eager C++ 算子 | 局部修改较明确；需核对远端对应源码并构建验证 |
| 223 | 合法空 unpool 输出被 CANN 参数校验拒绝 | eager 算子/Meta，可能涉及 `op-plugin`、CANN | 条件性局部修复；需确认 1D/2D/3D 与索引校验契约 |

这两项都先在 eager 失败，所以只改 `torch_npu/_inductor` 无法使原用例完成参考结果。
不能用测试里预先 expand 输入或把空输出尺寸改为正数来掩盖原缺陷。

## 失败项中不建议按简单修改处理的 48 项

完整分组和原序号见 `repair-groups.csv` 中 E 类的 R 分组；跳过项的 S08 另见上文。主要包括：

- static/fast launcher、lookup table：缺少正向功能接入，换名称或放宽断言不能补齐能力。
- graph capture/allocator：移除检查或关闭被测模式会丢失原目标；还需确认 NPU 等价运行时接口。
- BMM 分解、卷积布局策略：数值正常不等于优化策略符合要求，不能只替换 FileCheck 字符串。
- fp64 linear、complex64 logaddexp、channels_last：涉及 eager 或编译器能力，不是测试文本小修。
- sparse gather Meta：`torch_npu/op_plugin/meta` 在允许的 `_inductor` 目录之外，且存在分解重入。
- 构造器身份：涉及 `torch_npu/contrib/transfer_to_npu.py` 和可能的 Dynamo 规则；恢复全局绑定的探针不是生产修复。
- P2P、UDTK 地址错误、fallback 序列化：还缺少可支持局部修复的根因证据。

对确实不支持的 CUDA 专用或 dtype 功能，精确 skip 是可讨论的测试矩阵维护，但不计为修复通过。
CSV 中“是否涉及其他模块”保留“是”“可能”“未定”的区别，不把尚未确认的修改范围写死。
