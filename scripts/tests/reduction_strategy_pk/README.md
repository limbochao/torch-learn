# Split / cooperative 的 profiler 对照

`run.py` 从同一 `common.py::eager_function` 构造 199 组 forward。GPU 测量原生 Inductor 的 plain、split、forced cooperative、natural；plain 明确关闭 split/cooperative。NPU 先测同样关闭两项特性的 Inductor baseline，再测 `npu_kernels.py` 的匹配算法原型。两端均不修改安装包，不用 device Event / wall time 作为性能数据。

最终对照见[结果报告](../../../docs/requirements-design/split-cooperative-reduction-comparison/results.md)。测试设计与口径维护在本说明中。脚本使用 PyTorch 2.13、Triton 和 torch_npu 的内部接口，升级环境后先执行 smoke。

新机器推荐使用 `one_click.sh`：自动检查实际设备与接口、冻结源码、执行测试、汇总/绘图及审计。完整命令和独立包搬运方式见[一键使用说明](ONE_CLICK_README.md)。`run.py` 的直接入口保留原实验默认参数；一键入口支持自动检测 NPU 核预算、指定 GPU 小用例卡和全程 no_grad。

## 运行

在已授权容器中激活对应 Python 环境，先检查卡号和资源占用。CUDA 必须填写完整物理 UUID；第一个设备承接大用例，后续设备只执行 `X*R<=1048576` 的小用例，并限制本进程的 PyTorch allocator 为设备容量的 3.5%。NPU 填物理卡号，脚本会通过 ACL 验证实际映射。

以下低层入口在脚本目录中执行，所有运行输出均写入项目外。推荐的一键入口同样默认写入项目外，并拒绝项目内的 `--out`。

```bash
experiment_root="$HOME/torch-learn-runs/reduction-strategy-pk"
PK_PROFILE_GRAPH=1 python run.py --backend cuda --devices GPU-UUID6,GPU-UUID7 --smoke --out "$experiment_root/cuda-smoke"
PK_PROFILE_GRAPH=1 NPU_DEVICE_LIMIT=28,56 python run.py --backend npu --devices 2,3 --smoke --out "$experiment_root/npu-smoke"

PK_PROFILE_GRAPH=1 python run.py --backend cuda --devices GPU-UUID6,GPU-UUID7 --out "$experiment_root/cuda-full"
PK_PROFILE_GRAPH=1 NPU_DEVICE_LIMIT=28,56 python run.py --backend npu --devices 2,3 --out "$experiment_root/npu-full"
```

可以使用 `--case-ids sum-x1-r32768-float32-contiguous,...` 选择用例。`--emit-manifest cases.json` 离线导出完整清单。
每张卡一个串行子进程队列，同一形状的所有策略同卡配对。`--resume` 要求设备、case、源码摘要完全一致；已有终态记录不重跑，定向复测应选新目录。默认单次无进展期限为 900 秒，只有本 runner 启动的子进程组会在期限后被终止。

共享卡资源变化时，可以在原协调进程停止后使用 `continue_run.py` 继续相同的冻结源码与清单；`--adopt` 只接受该实验目录中仍在执行的 worker PID，保留它们完成当前 case。续跑记录会保存新的设备分配，已完成结果不覆盖。每次取新任务前查询 HBM 占用，超过 95% 时暂缓该卡取任务。`monitor_load.py` 可记录 `npu-smi` 设备占用采样；它不提供计时数据，也不能准确拆分其他进程造成的影响。

对 NPU 已确认入围的候选，可以使用 `--tuning-source <prior-run>` 独立复测。此模式从原结果取每种策略已进入确认阶段且精度通过的配置，重新执行原生 compile baseline、DSL 校验、三会话 profile 与变化输入检查，不再次搜索其它配置。结果记录原调优文件的 SHA256；它用于分析负载和重测变化，不代表重新找到全空间最优。

便携运行时将本目录 Python 文件及 `scripts/tools/autotune_tiling.py` 放在同一目录。`BestTilingRecorder` 只用于观察实际选中配置；Inductor 自身仍按原生 autotuner 工作。其内部选型机制并未改成 profiler，**本实验的策略比较和 NPU 手动候选搜索均只使用 profiler device duration**。

## 搜索空间

NPU 初筛：S=`2,4,8,14,28,56,112`；R≤4096 时 BLOCK=`64,256,512`，并过滤 `S>max(2,ceil(R/BLOCK))` 的候选，其余 R 使用 BLOCK=`512,2048`。cooperative 过滤 `S>56` 或 `X*S>56`；split 允许多 wave。每个策略初筛最好的两个配置继续搜索 BLOCK=`256,1024,4096`。split 还检查交错分片；RMSNorm/softmax 另试 `E=floor(56/X)`、`min(112,2*S)` 的输出并行度。split 的 E 与第一阶段 S 无须相同。

粗筛每候选 3 组×3 次调用；每种策略前两名与 baseline 做 3 次独立 profiler 会话，每会话 3 组×5 次调用。策略交替顺序执行。最终从通过变化输入检查的配置中，按 3 个会话中位数的中位数选择。所有候选与异常均保留；“最快”只指此有限、分阶段搜索，未穷举 BLOCK/分片/算法。

主体 R 扫描为 `256,1024,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304`。当 `X*R>33554432` 时两端确认批次均改为 3 次，限制 Graph 输出保留内存；解析和归一化按记录的 `confirmation_batch` 进行。原始 91 组运行快照仍可重现，新增清单单独保存在 `cases-expanded.json`。

X=56 没有满足单 wave 约束的 RS>1 cooperative，记录 `NOT_APPLICABLE`。这组用于展示输出并行度已经足够时的适用范围。

## profiler 口径

每个测量组内重复完整 forward，并在组末同步，编译、warmup、参考计算、精度检查与拷回都在采集外。GPU 包含 kernel 及 device memset/memcpy，因此 semaphore 初始化不会被漏掉。NPU 使用 Level0 的 `kernel_details.csv`，避免采集不需要的硬件指标。

`PK_PROFILE_GRAPH=1` 将每组 batch 次调用 capture 成一个 Graph，采集前额外预热三次 replay，然后在 profiler 标签内 replay。两端均解析完整 kernel 序列，计时来源仍是 profiler duration。未设置时为 direct callable。运行 manifest 和结果明确记录提交方式，禁止将不同方式合并成同一主结果。此选项用于检查和减轻 direct 提交时观察到的共享运行时波动，不保证排除所有竞争或同步问题。

CUDA 按同步包围的 CPU range 归属 device events，CPU duration 不计入结果。实测 NPU 的 host/device 时间戳有偏移，不能用 CPU 区间直接包含设备时间戳。NPU 采用单 stream 的完整设备事件顺序：split 必须严格匹配 `partial,finish`，cooperative 必须严格匹配 `cooperative`；每批次数、全部事件消费、baseline 重复 kernel 模式也须一致。任意多余、缺失或模式不符会报错。设备 duration 始终原样取自 profiler，时间戳只辅助排序。

CUDA 的两个大输出 case 曾在第三 profiler 会话缺少第一个 semaphore fill 事件。当前采集器在正式标签前增加 `pk-primer` 区间，单独预热完整 callable/graph；这些事件明确列入 `excluded_profiler_primer`，原始 trace 保留。正式 `pk:<strategy>:<repeat>` 区间仍须满足全部操作数量检查。原始失败不补零或插值，只对失败 case 完整重测。NPU 不执行这项 CUDA 采集适配。

指标是 `sum(device operation duration)/forward 次数`，不包含 kernel 间空隙。它不测 CPU launch 开销，也不能直接等价于整网端到端时延。相同地址热输入反复执行，输入大于 L2 的用例仍保留实际访存特征。共享卡上的抢占、时钟和干扰无法由本测试完全排除，汇总会标记会话中位数变化超过 15% 的配对。

## 结果与校验

- `run.json`：完整清单、设备和源码摘要。
- `cases/<id>/result.json`：环境、正确性、全部候选、最终配置与各次计时。
- `output_code_*.py`、`ir/`：完整生成代码、metadata 与可获取的编译 IR。
- `search/`、`refine/`、`confirm-*/`：原始 profiler 与逐组 device events。
- `progress.json`、`active.json`、`run.log`：失败或中断时的证据。

CPU 以固定随机种子先生成 fp32，再转换输入 dtype。eager 显式 `a.float()+b.float()`；输出均为 fp32，低精度用例考察输入带宽/转换，不声称测了低精度累加。stride-2 与转置 view 按真实 stride 分配并复用。

sum 对照 CPU fp64 累加（fp32 加法之后），`atol=0.02, rtol=3e-4`；amax 要求完全一致；RMSNorm 为 `atol=3e-5, rtol=3e-4`，softmax 为 `atol=1e-8, rtol=3e-4`。最终候选检查两组变化输入，每组连续调用三次；cooperative 另外检查两个 epoch、四个 phase 的严格整数交换。错误配置的性能不参与选择。

```bash
python summarize.py --input "$experiment_root/cuda-full" --input "$experiment_root/npu-full" --out "$experiment_root/figures"
python summarize.py --input "$experiment_root/cuda-main" --replace-failed "$experiment_root/cuda-profiler-primer" --out "$experiment_root/figures"
python verify_evidence.py --input "$experiment_root/collected-evidence" --output "$experiment_root/verification.json"
```

汇总只把实际生成 split 与 RS>1 cooperative 的配对纳入严格胜负统计。其余 requested-split/ordinary 对照单列。
首轮采集 hook 仅记录通用 `Reduction.create_multilayer`，遗漏 `OnlineSoftmaxReduction` 子类；汇总器根据完整保存的 DSL 验证 `(X*S,ceil(R/S)) -> (X,S)` 两阶段后纠正标注，原始记录保留。后续 runner 同时 hook 三种 reduction 类，不改变计算或性能选项。

`--replace-failed` 仅替换主运行中非 PASS 的 case，并保存原状态和证据路径；不会按速度挑选较快复测来覆盖正常记录，也不允许混合提交方式。证据审计检查原运行源码摘要、全部归档 SHA256、Graph 输出与变化输入检查，并从保留的 trace/CSV 独立重算三个确认会话。审计在临时目录解析，不改写原始证据。

`--confirmation-selection` 配合 `--confirmed` 用于完整应用采集前声明的固定候选复测清单。多个清单按参数顺序应用，全部确认输入必须与清单完全对应；缺失、失败、重复、来源或提交方式不符均报错。速度变慢的记录也会进入最终快照。CSV/JSON 的 `confirmation_history` 保留来源链，原搜索和每轮独立确认另存。固定候选摘要与配置集合由跨归档证据审计核对。

确认阶段的 ERROR 以新目录完整重跑，仍使用原 `--tuning-source` 的候选集合。汇总通过 `--replace-failed-confirmed` 接收这些重跑，只允许替换现有非 PASS 的确认记录，并核对用例、调优来源、声明的确认运行及提交方式；`confirmation_retry` 保存被替换的失败证据。已 PASS 的确认不能使用这个参数替换。

本次先确认 97 组，再对其中仍有任一策略会话变化超过 15% 的 40 组加控制点确认 43 组。两份清单及其选择理由、输入摘要保存在项目外原始数据目录的 `npu-remeasure-selection.json` 与 `npu-final-confirmation-selection.json`。

完整历史矩阵的参数与路径组合见本文末尾“最终快照的离线重算”；所有原始输入均取自项目外的数据目录。

审计同时检查完整清单覆盖、基线开关、相同 eager/输入/精度函数、实际输入 stride、NPU debug barrier 与驻留约束、严格整数交换、三次独立确认、逐样本 device duration 和操作次数。原始 search/refine profiler 要与候选集合和前两名入围规则相符；确认 profiler 核对全部入围候选，包含未最终选中的配置。逐样本浮点比较采用 `1e-12` 的相对/绝对容差，以兼容不同 Python 版本的浮点求和；操作数和批次数要求精确一致。使用 `--tuning-source` 的证据须与其原搜索归档一起传给审计器，便于核对原始摘要与完整固定候选集合。

## 配置阶段与专项诊断

CSV/JSON 的 `*_config_phase` 区分 `initial_compile` 与 `dsl_launch`。前者来自原生 `run_and_get_code` 初始编译，后者是 NPU 原型的实际候选 launch 参数。初始编译使用 `no_grad`，后续默认 grad context 的调用可能再次编译；不得仅凭初始元数据推定后续 Graph 的全部内部 tile。

`compile_context_probe.py` 对指定 plain case 分别记录初始编译与默认 grad context 调用的 unique graph 计数、完整生成代码、选中配置和输出精度，**不采集性能**。`--source` 应指向冻结运行源码目录，并包含 `run.py`、`common.py`、`npu_kernels.py` 和 `autotune_tiling.py`。必须在授权远端环境中先设置可见设备；`--device` 用于核验实际物理映射，不负责选择可见设备。

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID6 python compile_context_probe.py \
  --source frozen-source --backend cuda --device GPU-UUID6 \
  --case-ids sum-x1-r262144-float32-contiguous,softmax-x1-r32768-float32-contiguous \
  --out "$experiment_root/cuda-context-probe"

ASCEND_RT_VISIBLE_DEVICES=6 NPU_DEVICE_LIMIT=28,56 python compile_context_probe.py \
  --source frozen-source --backend npu --device 6 \
  --case-ids sum-x1-r262144-float32-contiguous,softmax-x1-r32768-float32-contiguous \
  --out "$experiment_root/npu-context-probe"
```

`verify_cuda_launch.py` 离线核对全部确认会话 `device_events.json` 中的实际 Triton kernel 名称、thread block 与记录配置，并核对 cooperative 的 RSPLIT/XBLOCK grid。覆盖缺失或参数不一致时返回非零退出码。它不重新计时，也不能从 launch 参数确认内部循环 R tile；原始 trace 到 device events 的对应由 `verify_evidence.py` 独立核验。

## 最终快照的离线重算

torch-learn 仅包含最终结果和图表，不包含历史采集的全量原始数据。若已经另行取得原始数据，可在项目外进行下列离线复核。从 `torch-learn` 根目录执行，将 `evidence` 改成原始数据的实际外部路径。汇总无需 GPU/NPU 环境；绘图需要 matplotlib/numpy，`--no-plot` 可只导出表格。完整快照使用全部预先声明的确认，失败重跑只替换相应 ERROR。

十个历史证据包、`archive-sha256.json` 和固定确认清单均保存在项目外。尚未展开时，下面的命令逐包校验 SHA256，再按清单恢复完整展开目录；需要约 0.8 GB 额外磁盘空间。

```bash
evidence=/data/reduction-strategy-pk/split-cooperative-profile-20261123
python - "$evidence" <<'PY'
import hashlib
import json
import sys
import tarfile
from pathlib import Path

base = Path(sys.argv[1]).resolve()
for item in json.loads((base / "archive-sha256.json").read_text())["archives"]:
    archive = base / item["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != item["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {archive}")
    destination = base / item["directory"]
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if (not target.is_relative_to(destination) or
                    not (member.isfile() or member.isdir())):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
        package.extractall(destination)
    print(f"Verified and extracted: {archive.name}")
PY
```

```bash
evidence=/data/reduction-strategy-pk/split-cooperative-profile-20261123
pk=scripts/tests/reduction_strategy_pk

python "$pk/summarize.py" \
  --input "$evidence/gpu-expanded-evidence/cuda-expanded" \
  --input "$evidence/npu-main-evidence/npu-main" \
  --input "$evidence/npu-expanded-evidence/npu-expanded" \
  --replace-failed "$evidence/npu-retry-evidence/npu-main-retry" \
  --confirmation-selection "$evidence/npu-remeasure-selection.json" \
  --confirmed "$evidence/npu-remeasure-evidence/npu-remeasure" \
  --confirmed "$evidence/npu-expanded-remeasure-evidence/npu-expanded-remeasure" \
  --confirmation-selection "$evidence/npu-final-confirmation-selection.json" \
  --confirmed "$evidence/npu-c2-main/npu-final-confirm-main" \
  --confirmed "$evidence/npu-c2-expanded/npu-final-confirm-expanded" \
  --replace-failed-confirmed "$evidence/npu-c2-retry1/npu-final-confirm-retry-exchange" \
  --replace-failed-confirmed "$evidence/npu-c2-retry2/npu-final-confirm-retry-other" \
  --out /data/reduction-strategy-pk/final-rebuilt

python "$pk/verify_evidence.py" \
  --input "$evidence/gpu-expanded-evidence" \
  --input "$evidence/npu-main-evidence" \
  --input "$evidence/npu-retry-evidence" \
  --input "$evidence/npu-expanded-evidence" \
  --input "$evidence/npu-remeasure-evidence" \
  --input "$evidence/npu-expanded-remeasure-evidence" \
  --input "$evidence/npu-c2-expanded" \
  --input "$evidence/npu-c2-main" \
  --input "$evidence/npu-c2-retry1" \
  --input "$evidence/npu-c2-retry2" \
  --output "$evidence/offline-evidence-verification.json"

python "$pk/verify_cuda_launch.py" \
  --input "$evidence/gpu-expanded-evidence/cuda-expanded" \
  --output "$evidence/cuda-launch-verification.json"
```

上述全量审计命令会从项目外的 profiler 重新解析，不复用已有审计缓存。历史 `final-evidence-verification.json` 使用当时 auditor 的增量结果：九个包的基础审计加最后一个 main 包，并重新检查跨包输入/候选来源契约。历史 auditor 原样保存在外部原始数据目录的 `auditor-used.py`，基础报告和其 SHA256 一起保留。当前 auditor 另支持从新运行的 `execution_policy` 检查可配置 Vector 预算；旧运行缺少该字段时仍按 56 核检查。仓库仅保留[最终验证结论](../../../docs/requirements-design/split-cooperative-reduction-comparison/assets/final/validation-summary.json)。
