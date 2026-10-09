# NPU Inductor 剩余失败、跳过及预期失败用例的全量评审

本轮以完整 PR #47384 加回 Ascend driver/helper 后的总表为起点，逐项评审 100 行：
82 失败、14 跳过、4 行含预期失败。原 247 行顺序、测试目的和历史结果全部保留。
14 个原跳过项中已有 9 个适配版通过，本轮将其作为适配回归，不能算作新发现的修复。

最终累计为 **177 通过、65 失败、4 跳过、1 项中止未完成**。
本轮新增 21 项通过，另确认 9 项已有 skip 适配。BMM 候选因追加动态复用精度回归撤回，
188/189 虽在候选下通过原检查，累计仍保留失败；83 在 1191 秒后中止，不计通过。
首轮执行 37 行、第二轮 23 行，去重为 51 行，其中 42 行来自本次 100 行评审范围，
9 行为此前通过项的相关回归。另有两项代码采集及 BMM 同图复用探针，不重复计入逻辑行数。

- [247 行最新总表](../../../artifacts/npu-inductor-remaining-repairs-20261009/results.csv)
- [100 行逐项必要性、修复方向、负责模块及停止原因](../../../artifacts/npu-inductor-remaining-repairs-20261009/remaining-triage.csv)
- [最终统计](../../../artifacts/npu-inductor-remaining-repairs-20261009/progress.json)
- [逐项评审原始数据](../../../artifacts/npu-inductor-remaining-repairs-20261009/review.json)
- [源码对照](../../../artifacts/npu-inductor-remaining-repairs-20261009/source-evidence.json)
- [首轮补丁清单](../../../artifacts/npu-inductor-remaining-repairs-20261009/first/changes.json)
- [第二轮补丁清单](../../../artifacts/npu-inductor-remaining-repairs-20261009/second/changes.json)
- [覆盖和 hash 校验](../../../artifacts/npu-inductor-remaining-repairs-20261009/validation.json)

## 统计口径

“累计最新结果”合并本轮候选补丁/测试适配的完整方法结果与此前未重跑项的结果，
不是宣称 247 项在一个新配置中重新全量通过。“本轮通过对应修改”说明每个通过项采用的改动，
“本轮修改清单”记录实际组合安装的全部补丁；前者是源码对应关系，不是对每个补丁做逐个移除实验。
原始用例的“补丁状态”和历史通过原因仍单列，不将测试副本的通过倒写到历史列。

逻辑行与参数实例分别计数。第 83 行指向外层 `test_basic_npu` 中的嵌套
`test_conv_bn_eval`，必须执行全部循环，不能只跑第一组输入。
第 76/77 行包含 Benchmarker 抽象基类的负向检查；通过不代表基类增加了 NPU 测时实现。

## 实际尝试的修改

| 原序号 | 改动 | 保留的验证目标 |
|---|---|---|
| 51–55 | 全局 0 维常量在输入设备创建 | 输入存活、view 别名、原地修改和输出正确性 |
| 68、69 | custom-op 的同一 `fn` 注册到 PrivateUse1 | stride、storage_offset、正向及错误 Meta 对齐检查 |
| 76、77 | 显式断言基类 `NotImplementedError`，替换宽泛 xfail | TritonBenchmarker 的 timing/counter 与基类负向契约 |
| 78、79 | 显式断言设备推断 `ValueError`；`make_sum` 改为 `make_params` | 无设备、多设备输入真实到达设备检查 |
| 132–136 | dtype pass 先确认 `Tensor.to` 和 Tensor/FakeTensor 输入 | 符号标量 round 原结果和动态行为 |
| 86、87、89–92 | 仅内建 POINTWISE kernel 增加 vetted benchmark 标记 | 确定性保持开启，padding 代码和计数断言不变 |
| 187–189 | 尝试小点积分解，追加复用验证失败后撤回 | 原阈值检查曾通过，但动态范围复用仍有精度缺陷 |
| 190、195、196 | 按 NPU 的 eager/compiled 精确错误文本断言 | 原非法 dtype 输入保持拒绝，保留 autograd 负向检查 |
| 243 | 对 `"\n".join(source_codes)` 执行原 `check_not` | 零 kernel 合法，数值及第二种 shape 检查仍执行 |
| 96、97 | `force_fallback` 接受单个 HigherOrderOperator key | 父进程注册完整恢复、原 handler 清理、后台编译 |

首轮的 9 个 skip 适配回归保持原 shape、dtype、存储大小和容差。
163/165/166/167 按实际可用 NPU 内存检查；197 按真实 Triton 模板能力判断；
201/202/204 细化历史 GPU skip；245 保留 fullgraph/capture 设置并断言返回 Python 标量 3.0。
201 继续使用局部卷积 `allow_hf32=False`，202 保持默认；HF32 是卷积计算精度选项，
关闭它用于使 eager 与融合路径以完整精度比较，不提高 `assert_close` 容差。

## Inductor 修复的源码依据

### 符号标量被误当作 Tensor

`dtype_optimal_pass` 原先在确定 `call_method` 是否为 Tensor 的 `to` 之前读取输入 `.dtype`。
`SymFloat` 是符号浮点标量，没有这个属性。补丁先检查 `node.target == "to"` 和输入存在，
再用 `isinstance(input_fake, torch.Tensor)` 判断后读取 dtype。
该改动不绕过 round 的追踪或编译；5 个 round 原用例及相关既有通过项均做了回归。

### BMM 小 K 点积分解候选已撤回

**该补丁没有保留为成功修复。** 原 188/189 用例在候选下通过，但追加同一编译图的
K=3→32 复用时，fp16 输出 4/4 不符，最大绝对差 8.5234375。
候选已从当前准备脚本移除，并在验证环境恢复原 BMM lowering。
第二轮补丁仅作为失败尝试的证据保存；188/189 的累计状态仍为失败。
见 [撤回记录](../../../artifacts/npu-inductor-remaining-repairs-20261009/bmm-withdrawal.json)。
恢复原 BMM lowering 后，同一探针在 fp16/bf16/fp32 × K=3/32/33/64 的 12 组输入全部通过，
每种 dtype 仅编译 1 张图，确认此次回归由候选引入。
对照见 [候选错误](../../../artifacts/npu-inductor-remaining-repairs-20261009/bmm-reuse.log) 与
[原 lowering 结果](../../../artifacts/npu-inductor-remaining-repairs-20261009/bmm-reuse-baseline.json)。

社区 `tuned_bmm` 对浮点小点积采用乘法加归约，让相邻运算能够融合。
NPU 自己注册的 lowering 覆盖了该实现，原版本仅有 CPU 的分解分支。
候选曾为 NPU 的 fp16/bf16/fp32 接入同类规则：两输入 dtype 相同，M/N 的优化 hint 为 1，
K 没有静态已知的 `>32` 证明时，生成 `sum(mul(unsqueeze(a), unsqueeze(b)), axis=2)`。

hint 只用于选择优化，不把动态维度强制固定成 1。该乘法归约对一般 M/N 也合法；
K 使用静态关系判断，避免按一次运行的具体 K 生成不能复用的缓存图。
原测试要求 K=32 无 extern BMM、静态 K=33 保留 extern BMM，动态范围 1..64 也要满足其代码检查。
整数 dtype 不进入此分解，普通矩阵 BMM 的相邻用例另作回归。

生成代码证据分别为 [K=32 归约](../../../artifacts/npu-inductor-remaining-repairs-20261009/generated/row-188-wrapper-1.py)、
[K=33 extern BMM](../../../artifacts/npu-inductor-remaining-repairs-20261009/generated/row-188-wrapper-2.py) 和
[动态 K 归约](../../../artifacts/npu-inductor-remaining-repairs-20261009/generated/row-189-wrapper-1.py)。
前者包含 `tmp2 = tmp0 * tmp1`、`tl.sum(tmp2, 1)`，K=33 产物保留 `extern_kernels.bmm(...)`。
产物仅去除环境路径和 backend hash，保持原生成格式。这些是已撤回候选的产物。
动态归约的 tile 与运行时 extent/cache 复用需要继续定位；不能仅凭原用例一次 K=3 的运行，
推断后续 K 值都正确。完整追加脚本为 [probe_remaining_bmm.py](probe_remaining_bmm.py)。

第 187 项在修复后进入 float64 eager Matmul，因 `DT_DOUBLE` 不受支持失败；整项仍失败。
不能删除 dtype 循环或只统计此前通过的低精度分支。

### HigherOrderOperator 的子进程恢复

探针确认父进程 fallback registry 包含 `invoke_quant`、`with_effects` 等 HigherOrderOperator。
它们表示接收函数/子图的高阶算子，不是普通 `OpOverload`。
子进程恢复时原 `force_fallback`
只接受后者而断言失败。补丁允许这两类单个 key，仍保存和恢复原 handler，不过滤掉非 OpOverload 条目。
完整异步/progressive 流程的最终结果以逐项阶段报告为准，不将消除第一条断言等同于全部通过。

复测已越过 HOO 首错，随后子进程 `config.patch(input.config)` 报
`torch._inductor.config.npu_backend does not exist`。需要适配 NPU 后端与配置注册的子进程初始化，
具体涉及 `torch_npu/_inductor/__init__.py`、`torch_npu/utils/_dynamo.py` 和社区 `compile_fx_ext.py`。
`warm_pool` 早于后端/config 注册是一条源码线索，尚未证明为唯一原因；本轮没有过滤配置键或禁用异步。

### Conv/BN 的未完成复测

第 83 行局部关闭 HF32 后运行 1191 秒仍未返回，随后主动中止。独立诊断在 120/240 秒打印的
Python 栈位于 `fuse_attention::_get_sfdp_patterns → transfer_to_npu::decorated → Tensor 构造`。
这说明诊断调用仍在模式初始化中等待，不足以确认 CANN 或具体 kernel 的根因。
需继续捕获 Tensor 构造/拷贝的原生栈与设备任务；本项保留“中止/未完成”，不计精度修复通过。
证据见 [诊断栈](../../../artifacts/npu-inductor-remaining-repairs-20261009/diagnostic/case-83.log)。

## 不能只改用例或 Inductor 的阻塞

| 范围 | 当前证据与负责模块 | 处置 |
|---|---|---|
| 48 | 类迁移后，`op-plugin` 的 `convolution_meta` 把单元素 stride 解包成两值 | 修 Meta，保留 1D 覆盖 |
| 73 | `__Rshift__KernelNpuOpApi.cpp` 的输出分配不符合右移广播形状 | 修 op-plugin eager 分配并构建验证 |
| 82 | NPU profile 无 `.profiler.kineto_results.events()` | 适配 TorchProfilerBenchmarker 与 `torch_npu.profiler` 的真实事件接口 |
| 128、182、187 | CANN eager 拒绝 complex64 LogAddExp 或 Double Matmul/Addmm | 明确算子/精度能力；不改 dtype 伪装原项通过 |
| 17、131、210、224 | fp64 libdevice/Trunc/f64 load 等缺口 | triton-ascend CANN libdevice、编译器类型和相应 op-plugin/ACLLN 算子适配 |
| 209 | sparse gather backward Meta 构造/迁移递归 | op-plugin gather Meta 与 torch decomposition/prims 的设备迁移 |
| 223 | 合法空 unpool 输出在 ACLNN 参数检查失败 | op-plugin MaxUnpool eager/Meta 边界处理 |
| 226 | 公共 channels_last 接口在 eager 拒绝 | torch_npu 分配/复制、Meta 与 Inductor layout 契约 |
| 230 | 无编译的 4 元素跨卡复制也在 peer access 报 207001 | `NPUPeerToPeerAccess.cpp`、CANN runtime 与驱动/资源配置 |
| 242、244 | transfer 包装构造器后，unbacked 符号表达式被要求特化 | `transfer_to_npu.py` 与 Dynamo 构造器身份规则 |
| 246 | 生成的整数符号幂被转成 float64 `libdevice.pow` | NPU symbolic codegen；需整数范围证明或完整 libdevice 支持 |

86/87/89/91/92 的 benchmark 首错已消除，但 padding 布局文本或优化计数断言仍不满足。
其中 87 还含固定设备编号假设；只改编号不能证明 NPU 实现了被测 padded storage。
这些项保留失败，需要 NPU wrapper/scheduler 的 padding 实现，不能删除计数或代码断言。

44/75/247 需要 NPU allocator 和真实 capture/replay 集成；不能令 CUDA capability 检查恒真，
或关闭原模式的图捕获来记通过。94/95 的最新独立报告在模式初始化暴露异步设备错误，
最终归属还缺少故障 kernel 和 launch 实参，只能明确到扩展 backend、NPU launcher、
triton-ascend/CANN 这些待核对模块，不能凭异步报错位置定责。

## 当前没有必要强行通过的原目标

- static/fast CUDA launcher 需要 CUDA 二进制、C++ launcher 和参数 ABI；NPU 若启用等价功能，
  才应移植完整测试。只把 `cubin` 改成 `npubin` 不足以验证功能。
- 42/45/46 的 CUDA TMA 能力门槛保留。通用多配置/混合模板目标在 NPU lookup key、模板 UID、
  hash 和候选契约接入后需要等价测试；当前不能伪造 TMA capability。
- 50 的 CUTLASS 主机 dtype 注册可在真实依赖可导入时选测；当前缺少 `cutlass_api`，
  没有必要为 NPU 基础验收安装 CUDA 专属依赖。
- 93 的 dummy PrivateUse1 注册应在未加载真实 NPU runtime 的上游环境独立测试。
  一个进程已经注册 `npu` 后再注册第二个 runtime，不属于修复 NPU 注册功能。
- 203 的原 CUDA channels_last 启发式不能直接当作 NPU 必达布局目标；应先定义 NPU 等价策略，
  而不是让代码检查接受任意实际出现的调用。

27/29/31–39 的 lookup table 逻辑值得保留，但需要稳定 NPU 架构 key、模板/config/hash 集成。
49 的设备 TFLOPS 模型、63 的训练 range 标注，也需对应 NPU 性能/Profiler 接口支持。
逐项表明确区分“当前没有必要强行移植原 CUDA 目标”和“功能启用后必须补等价测试”。

## 复现与边界

环境沿用 torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0` 和 Ascend950PR_9579。
PR HEAD 为 `16437be0ffc9d67c8b27855a234243e35385c58b`，共同前提为 PR 迁移文件加回原 driver patch。
所有执行位于已授权验证容器的测试工作目录；源码改动先推验证分支，再从容器拉取。
测试副本与 site-package 候选分别保留 before/after SHA256，原测试文件不修改。

```bash
python scripts/tests/npu_inductor_213/prepare_remaining_repairs.py \
  --test-dir "$ORIGINAL_TEST_DIR" --site-dir "$ORIGINAL_SITE_DIR" \
  --output-dir "$REPAIR_ROOT"
```

生成脚本只准备候选，不安装。各轮 `changes.json` 和 `preflight.json` 记录实际安装快照；
第二轮的安装前基线来自首轮备份，避免对已有补丁重复应用。
runner 在每项运行前检查 site-package 文件 hash，并执行原完整方法的所有参数实例：

```bash
python scripts/tests/npu_inductor_213/run_remaining_repairs.py \
  --root "$REPAIR_ROOT" --round second --device 0 188 189 190

python scripts/tests/npu_inductor_213/summarize_remaining_repairs.py \
  --first-dir "$FIRST_RAW_DIR" --second-dir "$SECOND_RAW_DIR" \
  --output-dir artifacts/npu-inductor-remaining-repairs-20261009
```

代码候选与测试适配以补丁形式归档，尚未作为生产仓 PR 合入。局部通过不能证明其余模型/形状性能不变。
验证结束后，5 个临时修改的安装文件均已恢复原快照，原测试文件 hash 未变化；
见 [环境恢复检查](../../../artifacts/npu-inductor-remaining-repairs-20261009/postflight.json)。
本轮保留的未修项已经完成必要性和方向评审，仍需功能开发或底层模块适配；
总表不把“未实现”“仍失败”“跳过”写成已解决。
