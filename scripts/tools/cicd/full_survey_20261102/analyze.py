import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Analyze/render the 2026-11-02 CICD performance snapshot.")
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
BASE = args.work_dir.resolve()
ROOT = args.input_root.resolve()
DATA = json.loads((BASE / "inventory.json").read_text())
VIEW = {"view", "reshape", "_unsafe_view", "t", "transpose", "permute", "detach", "squeeze", "unsqueeze"}
PAT = {
    "sum": ("aclnnReduceSum_", "sum"),
    "tanh_backward": ("aclnnTanhBackward_", "tanh_backward"),
    "tanh": ("aclnnTanh_", "tanh"),
    "gelu": ("aclnnGelu", "gelu"),
    "gelu_backward": ("aclnnGeluBackward", "gelu_backward"),
    "threshold_backward": ("aclnnThresholdBackward_", "threshold_backward"),
    "add": ("aclnnAdd_", "add"),
    "mul": ("aclnnMul_", "mul"),
    "native_layer_norm_backward": ("aclnnLayerNormBackward_", "native_layer_norm_backward"),
    "avg_pool2d": ("aclnnAvgPool2d_", "avg_pool2d"),
    "scatter_add": ("aclnnScatterAdd_", "scatter_add_"),
    "nll_loss_backward": ("aclnnNLLLossBackward_", "nll_loss_backward"),
    "_log_softmax_backward_data": ("aclnnLogSoftmaxBackward_", "_log_softmax_backward_data"),
}


def families(d):
    return {a.split(".")[-1] for a in d["aten"]} - VIEW


def match(n, d):
    return n == d["name"] or d["name"].endswith(n) or d["name"].startswith(n)


def defs(a, device, n):
    return [d for d in a["code"][device] if match(n, d)]


def eager(a, device, fs):
    out = {}
    for n, g in a["profile"][device + "_Eager"]["groups"].items():
        for f in fs:
            if f not in PAT:
                continue
            p = PAT[f][0 if device == "NPU" else 1]
            if (
                (n.startswith(p) and not (f == "gelu" and n.startswith("aclnnGeluBackward")))
                if device == "NPU"
                else n == p
            ):
                out[n] = g
    return out


def total(groups):
    return sum(g["ms"] for g in groups.values())


def gpu_cost(a, fs):
    out = {}
    for n, g in a["profile"]["GPU_Inductor"]["groups"].items():
        dd = defs(a, "GPU", n)
        if any(families(d) & fs for d in dd):
            out[n] = g
        elif not n.startswith("triton"):
            for f in fs:
                if f in PAT and n == PAT[f][1]:
                    out[n] = g
    return out


ALL = {}
for case, a in DATA.items():
    issues = []

    def add(kind, title, ng, ne, ge, gg, component, detail, bound=True, notes=""):
        ni = total(ng)
        nev = total(ne)
        gev = total(ge)
        giv = total(gg)
        if min(ni, nev, gev, giv) <= 0:
            return
        budget = ni - nev * giv / gev
        if budget <= max(0.001, ni * 0.01):
            return
        issues.append(
            dict(
                kind=kind,
                title=title,
                npu_kernels=list(ng),
                gpu_kernels=list(gg),
                ne=nev,
                ni=ni,
                ge=gev,
                gi=giv,
                npu_eager_groups=list(ne),
                gpu_eager_groups=list(ge),
                npu_gain=1 - ni / nev,
                gpu_gain=1 - giv / gev,
                budget_ms=budget,
                share=ni / a["profile"]["NPU_Inductor"]["total_ms"],
                model_reduction=budget / a["profile"]["NPU_Inductor"]["total_ms"],
                gain_pp=budget / a["profile"]["NPU_Eager"]["total_ms"] * 100,
                component=component,
                detail=detail,
                bound=bound,
                notes=notes,
                npu_selected_ms={n: g["ms"] for n, g in ng.items()},
                npu_selected_count={n: g["count"] for n, g in ng.items()},
            )
        )

    # Core-family conservative screen. Only pure kernels or explicit fused families are selected.
    configs = [
        (
            "gelu",
            "GELU 前向外部调用边界",
            {"gelu"},
            "NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合",
            "NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。",
        ),
        (
            "ln_backward",
            "LayerNorm backward 归约调度",
            {"native_layer_norm_backward"},
            "NPU Inductor codegen/reduction tiling 与 runtime autotune",
            "检查该 kernel 的 reduction size_hints、split_axis 和 program 并行度；GPU 对应归约 kernel 列于本项。优先检查归约维切分与多阶段归约，不能据此断言 Triton_ascend/npuir 有缺陷。",
        ),
        (
            "pool",
            "avg_pool2d 生成 kernel",
            {"avg_pool2d"},
            "NPU Inductor avg_pool2d lowering、pointwise tiling/runtime 配置",
            "该 kernel 是 avg_pool2d 本身的生成实现；先核对输出点并行映射及 tile，必要时选择 ACLNN 回退。若相同 DSL/配置仍慢，再交 Triton_ascend/npuir。",
        ),
        (
            "relu_grad_reduce",
            "ReLU backward 与 bias 梯度归约",
            {"threshold_backward", "sum"},
            "NPU Inductor scheduler 融合决策、codegen/reduction tiling",
            "具体检查 threshold_backward 输出写回与沿 batch/edge 维 sum 的联合调度；GPU 将 pointwise 与 reduction 分拆或采用不同归约 kernel。重点是归约 kernel 的并行度和多阶段策略。",
        ),
        (
            "tanh_grad_reduce",
            "tanh backward 与 sum 梯度归约",
            {"tanh_backward", "sum"},
            "NPU Inductor scheduler、codegen/reduction tiling",
            "该融合包含 tanh 导数与按列梯度归约；GPU 为 tanh_backward 与 sum 生成独立或分阶段 kernel。检查输出通道少、归约维大时是否分派过少 program。",
        ),
        (
            "gelu_grad_reduce",
            "GELU backward 与 sum 梯度归约",
            {"gelu_backward", "sum"},
            "NPU Inductor scheduler、codegen/reduction tiling",
            "具体检查融合 gelu_backward 后的 bias sum；GPU 相应融合/归约成本全部计入上界。优化输出维分块、归约维切分和融合取舍。",
        ),
        (
            "affine_grad_reduce",
            "outer reduction：IR 分段计划缺失与 tiling 候选漏搜",
            {"sum", "add", "mul"},
            "NPU Inductor IR `num_splits` 与 runtime `TileGenerator`",
            "MobileBERT Train 的四组 R=16384、X=128 outer reduction 缺少 partial/final 计划，"
            "且候选遗漏更快的宽 X tile；逐 kernel 证据及其他宽度的边界见模型文档。",
        ),
        (
            "sum",
            "独立 sum 生成归约",
            {"sum"},
            "NPU Inductor codegen/reduction tiling、runtime autotune",
            "选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。",
        ),
    ]
    for kind, title, fs, component, detail in configs:
        if kind == "affine_grad_reduce" and not case.startswith("MobileBert"):
            continue
        if kind == "tanh_grad_reduce" and not case.startswith("lennard"):
            continue
        ng = {}
        for n, g in a["profile"]["NPU_Inductor"]["groups"].items():
            dd = defs(a, "NPU", n)
            if dd and all(families(d) and families(d) <= fs for d in dd):
                # For mixed families require sum + the activation, so attribution stays concrete.
                if len(fs) > 1 and not any("sum" in families(d) and len(families(d)) > 1 for d in dd):
                    continue
                ng[n] = g
            elif kind == "gelu" and n.startswith("aclnnGelu") and not n.startswith("aclnnGeluBackward"):
                ng[n] = g
        add(
            kind,
            title,
            ng,
            eager(a, "NPU", fs),
            eager(a, "GPU", fs),
            gpu_cost(a, fs),
            component,
            detail,
            notes="保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。",
        )
    # Scatter core cost: GPU fused kernels include index/mul overhead, so GPU cost is an upper bound.
    if case.startswith("basic_gnn"):
        ng = {n: g for n, g in a["profile"]["NPU_Inductor"]["groups"].items() if n.startswith("aclnnScatterAdd_")}
        add(
            "scatter",
            "scatter_add 外部调用阻断 gather/mul 融合",
            ng,
            eager(a, "NPU", {"scatter_add"}),
            eager(a, "GPU", {"scatter_add"}),
            gpu_cost(a, {"scatter_add"}),
            "NPU Inductor aten.scatter_add lowering、scatter/gather scheduler fusion",
            "NPU wrapper 保留 aten.scatter_add 外部调用及独立 gather/index_select；GPU 生成含 scatter_add 的 Triton kernel，可连同 index_select/mul/new_zeros 融合。优先补这个具体 lowering/融合边界；原子写入实现再由 Triton_ascend/CANN 承接。",
            notes="GPU fused kernel 的全部耗时均收费给 scatter_add 核心；GPU Eager 仅计 scatter_add_，因此是保守 GPU 收益下界。",
        )
    # Loss backward: count ALL forward+backward loss kernels as upper cost, also count prepare_softmax_online.
    if case.endswith("Train"):
        fs = {"nll_loss_backward", "_log_softmax_backward_data"}
        ng = {
            n: g
            for n, g in a["profile"]["NPU_Inductor"]["groups"].items()
            if n.startswith(("aclnnNLLLossBackward_", "aclnnLogSoftmaxBackward_"))
        }
        gg = {
            n: g
            for n, g in a["profile"]["GPU_Inductor"]["groups"].items()
            if any(t in n for t in ["log_softmax", "nll_loss", "prepare_softmax_online"])
        }
        add(
            "loss_backward",
            "NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵",
            ng,
            eager(a, "NPU", fs),
            eager(a, "GPU", fs),
            gg,
            "NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering",
            "两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。",
            notes="GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。",
        )
    # Forward loss: exact NPU tail after final projection, Eager GPU core lower bound.
    if case.endswith("Infer"):
        regions = {}
        for backend in ["Eager", "Inductor"]:
            path = ROOT / case / f"{case}_{backend}_NPU.csv"
            rows = list(csv.DictReader(path.open()))
            per = []
            names = set()
            selected_counts = defaultdict(float)
            for step in sorted({r["Step Id"] for r in rows}):
                rr = [r for r in rows if r["Step Id"] == step]
                end = next((i for i in range(len(rr) - 1, -1, -1) if rr[i]["Name"].startswith("aclnnNLLLoss_")), None)
                if end is None:
                    break
                start = next((i for i in range(end - 1, -1, -1) if "MatMulV3" in rr[i]["Name"]), None)
                if start is None:
                    break
                part = rr[start + 1 : end + 1]
                names.update(r["Name"] for r in part)
                for row in part:
                    selected_counts[row["Name"]] += 0.1
                per.append(sum(float(r["Duration(us)"]) for r in part) / 1000)
            if len(per) == 10:
                regions[backend] = {n: g for n, g in a["profile"]["NPU_" + backend]["groups"].items() if n in names}
                # Repeated copies in other layers must NOT be charged at model aggregate cost.
                for n in regions[backend]:
                    cost = (
                        sum(
                            float(r["Duration(us)"])
                            for step in sorted({r["Step Id"] for r in rows})
                            for rr in [[r for r in rows if r["Step Id"] == step]]
                            for end in [
                                next(i for i in range(len(rr) - 1, -1, -1) if rr[i]["Name"].startswith("aclnnNLLLoss_"))
                            ]
                            for start in [next(i for i in range(end - 1, -1, -1) if "MatMulV3" in rr[i]["Name"])]
                            for r in rr[start + 1 : end + 1]
                            if r["Name"] == n
                        )
                        / 10000
                    )
                    regions[backend][n] = dict(regions[backend][n], ms=cost, count=selected_counts[n])
        if len(regions) == 2:
            ge = {
                n: g
                for n, g in a["profile"]["GPU_Eager"]["groups"].items()
                if n in ["_log_softmax", "nll_loss_forward"]
            }
            gg = {
                n: g
                for n, g in a["profile"]["GPU_Inductor"]["groups"].items()
                if any(t in n for t in ["log_softmax", "nll_loss", "prepare_softmax_online"])
            }
            add(
                "loss_forward",
                "log_softmax→NLLLoss 的完整 log probability 写回",
                regions["Inductor"],
                regions["Eager"],
                ge,
                gg,
                "NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion",
                "外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。",
                notes="NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。",
            )
    # First attention + duplicate FMA: established defects retained, measured current CSV.
    if case.endswith("Train") and any(
        "torch.ops.prims.fma.default(" in p.read_text() for p in (ROOT / case).glob("NPU_g*_output_code.py")
    ):
        gg = a["profile"]["NPU_Inductor"]["groups"]
        first = [(n, g) for n, g in gg.items() if "all_amax_div_isfinite_matmul_native_dropout_sub" in n]
        later = [(n, g) for n, g in gg.items() if n.startswith("triton") and "softmax_all_amax" in n]
        if case in ["ElectraForCausalLM_Train", "LayoutLMForMaskedLM_Train"] and first and later:
            n, g = first[0]
            ln, lg = later[0]
            budget = g["ms"] - lg["ms"] / lg["count"]
            issues.append(
                dict(
                    kind="attention_seed",
                    title="首层 attention constexpr/checker 回退",
                    npu_kernels=[n],
                    gpu_kernels=[
                        k
                        for k in a["profile"]["GPU_Inductor"]["groups"]
                        if "softmax" in k and "dropout" in k and "backward" not in k
                    ],
                    ni=g["ms"],
                    budget_ms=budget,
                    share=g["ms"] / a["profile"]["NPU_Inductor"]["total_ms"],
                    model_reduction=budget / a["profile"]["NPU_Inductor"]["total_ms"],
                    gain_pp=budget / a["profile"]["NPU_Eager"]["total_ms"] * 100,
                    component="NPU Inductor NPUIndexTritonKernel.codegen_kernel、TTIR checker constexpr 参数传递及回退 tiling",
                    detail=f'首层 kernel 为 {g["ms"]:.6f} ms/step；后续同类 {ln} 为 {lg["ms"]/lg["count"]:.6f} ms/次。此前源码/TTIR checker 已确认 seed offset constexpr 缺失常量触发回退。GPU 同类 attention 没有这一首层回退。实际 compile_mode 不能仅凭 metadata 标签区分。',
                    notes="此项已由历史 checker 复现定位，是编译器缺陷；预算采用同设备后续同类 kernel 控制，不冒充按 GPU 局部提升比计算的预算。见历史首层 attention 文档与完整源码证据。",
                    bound=False,
                )
            )
        rows = list(csv.DictReader((ROOT / case / f"{case}_Inductor_NPU.csv").open()))
        ns = defaultdict(float)
        ns_count = defaultdict(float)
        nsteps = 10
        window = []
        for step in sorted({r["Step Id"] for r in rows}):
            rr = [r for r in rows if r["Step Id"] == step]
            for i, r in enumerate(rr):
                if "softmax" in r["Name"].lower() and "backward" in r["Name"].lower():
                    part = rr[i + 1 : i + 5]
                    if len(part) == 4 and all(
                        part[j]["Name"].startswith("aclnnMul_" if j % 2 == 0 else "aclnnAdd_") for j in range(4)
                    ):
                        for z in part[:2]:
                            ns[z["Name"]] += float(z["Duration(us)"]) / 10000
                            ns_count[z["Name"]] += 0.1
        if ns:
            budget = sum(ns.values())
            issues.append(
                dict(
                    kind="fma_duplicate",
                    title="prims.fma metadata 重复执行第一组 Mul/Add",
                    npu_kernels=list(ns),
                    gpu_kernels=[
                        k
                        for k in a["profile"]["GPU_Inductor"]["groups"]
                        if ("softmax_backward" in k and "log_softmax" not in k)
                        or ("attention" in k and "backward" in k)
                    ],
                    ni=budget,
                    budget_ms=budget,
                    share=budget / a["profile"]["NPU_Inductor"]["total_ms"],
                    model_reduction=budget / a["profile"]["NPU_Inductor"]["total_ms"],
                    gain_pp=budget / a["profile"]["NPU_Eager"]["total_ms"] * 100,
                    npu_selected_ms=dict(ns),
                    npu_selected_count=dict(ns_count),
                    component="PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行",
                    detail="softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。",
                    notes="仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。",
                    bound=False,
                )
            )
    issues.sort(key=lambda z: z["budget_ms"], reverse=True)
    # Top non-MM kernels: candidates requiring alignment, never claim regression solely from absolute I costs.
    top = []
    for n, g in sorted(a["profile"]["NPU_Inductor"]["groups"].items(), key=lambda p: -p[1]["ms"]):
        if any(
            z in n
            for z in [
                "MatMul",
                "Matmul",
                "Mm_",
                "Addmm_",
                "Convolution",
                "Conv2D",
                "npu_fusion_attention",
                "FlashAttention",
                "FlashAttentionScore",
            ]
        ):
            continue
        if not n.startswith("triton") and not any(
            z in n for z in ["BroadcastTo", "Softmax", "Gelu", "Scatter", "LayerNorm", "Pool"]
        ):
            continue
        if any(n in issue["npu_kernels"] for issue in issues):
            continue
        dd = defs(a, "NPU", n)
        fs = set().union(*(families(d) for d in dd)) if dd else set()
        gpu = [k for k in a["profile"]["GPU_Inductor"]["groups"] if any(families(d) & fs for d in defs(a, "GPU", k))]
        top.append(
            dict(
                name=n,
                ms=g["ms"],
                count=g["count"],
                share=g["ms"] / a["profile"]["NPU_Inductor"]["total_ms"],
                aten=sorted(fs),
                gpu=gpu[:8],
                gpu_all_count=len(gpu),
            )
        )
        if len(top) == 3:
            break
    ALL[case] = dict(issues=issues, candidates=top)

# Give the known first-attention defect an additional four-way conservative screen.
# Charge all 12 eager attention windows to the one slow compiled occurrence. GPU dropout
# mixes attention/FFN shapes; bound the 12 required calls without assuming equal durations.
for case in ["ElectraForCausalLM_Train", "LayoutLMForMaskedLM_Train"]:
    a = DATA[case]
    issue = next(z for z in ALL[case]["issues"] if z["kind"] == "attention_seed")
    rows = list(csv.DictReader((ROOT / case / f"{case}_Eager_NPU.csv").open()))
    ne = 0.0
    eager_names = set()
    steps = sorted({r["Step Id"] for r in rows})
    for step in steps:
        rr = [r for r in rows if r["Step Id"] == step]
        end = next(i for i, r in enumerate(rr) if r["Name"].startswith("aclnnNLLLossBackward_"))
        bmm = [i for i, r in enumerate(rr[:end]) if "BatchMatMulV3" in r["Name"]]
        assert len(bmm) == 24, (case, step, len(bmm))
        for layer in range(12):
            for row in rr[bmm[2 * layer] + 1 : bmm[2 * layer + 1]]:
                ne += float(row["Duration(us)"]) / len(steps) / 1000
                eager_names.add(row["Name"])
    eg = a["profile"]["GPU_Eager"]["groups"]
    dropout = eg["native_dropout"]
    k = 12
    assert eg["_softmax"]["count"] == k
    dropout_lb = max(
        k * dropout["min_us"] / 1000,
        dropout["ms"] - (dropout["count"] - k) * dropout["max_us"] / 1000,
    )
    ge = eg["_softmax"]["ms"] + dropout_lb
    gi = sum(a["profile"]["GPU_Inductor"]["groups"][n]["ms"] for n in issue["gpu_kernels"])
    control = issue["budget_ms"]
    budget = issue["ni"] - ne * gi / ge
    assert budget > 0
    issue.update(
        ne=ne,
        ge=ge,
        gi=gi,
        npu_gain=1 - issue["ni"] / ne,
        gpu_gain=1 - gi / ge,
        npu_eager_groups=sorted(eager_names),
        gpu_eager_groups=["_softmax", "native_dropout (12 calls, min/max lower bound)"],
        budget_ms=budget,
        control_budget_ms=control,
        model_reduction=budget / a["profile"]["NPU_Inductor"]["total_ms"],
        gain_pp=100 * budget / a["profile"]["NPU_Eager"]["total_ms"],
        notes=(
            "保守四方筛选：NPU 将全部 12 层前向两次 BMM 之间的 Eager 工作收费给首层，"
            "Inductor 只计首层；GPU compiled 计全部 12 层前向 softmax/dropout kernel，"
            "GPU Eager 计全部 softmax，以及混合 dropout 中 12 次 attention 调用的下界。"
            "dropout 下界=max(12×min, total−(count−12)×max)，不按平均值分摊混合形状。"
            f"因此这里是很保守的提升比缺口预算；恢复到 NPU 后续层实现的独立控制估算为 {control:.6f} ms，"
            "与本预算口径不同，不能相加。根因仍采用历史 checker/source 证据。"
        ),
    )
    ALL[case]["issues"].sort(key=lambda z: z["budget_ms"], reverse=True)

(BASE / "analysis.json").write_text(json.dumps(ALL, ensure_ascii=False, indent=2) + "\n")
print("issues", sum(len(a["issues"]) for a in ALL.values()), "scenarios", sum(bool(a["issues"]) for a in ALL.values()))
for c, a in ALL.items():
    if a["issues"]:
        print(c, "; ".join(f'{z["kind"]}:{z["budget_ms"]:.3f}' for z in a["issues"]))
