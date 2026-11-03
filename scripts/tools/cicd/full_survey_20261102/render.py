import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description="Render the 2026-11-02 CICD performance snapshot.")
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True)
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
BASE = args.work_dir.resolve()
ROOT = args.input_root.resolve()
REPO = args.repo.resolve()
DATA = json.loads((BASE / "inventory.json").read_text())
ANALYSIS = json.loads((BASE / "analysis.json").read_text())
DOC = REPO / "docs/issue-logs/cicd/performance"
ART = REPO / "artifacts/cicd/performance/full-survey-20261102"
(DOC / "models").mkdir(exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def clean(s):
    s = re.sub(r"(['\"]backend_hash['\"]\s*:\s*)['\"][A-Za-z0-9]+['\"]", r"\1'<redacted>'", s)
    s = re.sub(r"^# kernel path:.*$", "# kernel path: <generated-cache>", s, flags=re.M)
    s = re.sub(r"/home/[^\s\'\"\),]+", "<workspace>", s)
    s = re.sub(r"/data/[^\s\'\"\),]+", "<data>", s)
    return s


def match(n, d):
    return n == d["name"] or d["name"].startswith(n) or d["name"].endswith(n)


def defs(a, dev, n):
    return [d for d in a["code"][dev] if match(n, d)]


def full(a, dev, n):
    return " / ".join(dict.fromkeys(d["name"] for d in defs(a, dev, n))) or n


def code(s):
    return "`" + s + "`"


def pct(x):
    return f"{100*x:.2f}%"


def nums(x):
    return f"{x:.6f}"


def short(s):
    return s.replace("|", "\\|")


def common_issue_share_cells(samples, profiles):
    def number(value):
        return f"{value:.2f}" if value < 0.1 else f"{value:.1f}"

    def typical_range(values):
        typical = number(median(values)) + "%"
        if len(values) == 1:
            return typical
        return f"{typical}（{number(min(values))}–{number(max(values))}%）"

    costs = [100 * issue["share"] for _, issue in samples]
    budgets = [100 * issue["model_reduction"] for _, issue in samples]
    gap_shares = [
        100 * issue["budget_ms"] / profiles[case]["whole_model"]["deficit_ms"]
        for case, issue in samples
        if profiles[case]["whole_model"]["deficit_ms"] > 0
    ]
    gap_share = f"{number(median(gap_shares))}%（{len(gap_shares)}例）" if gap_shares else "—"
    return typical_range(costs), typical_range(budgets), gap_share


# Full selected definitions, including metadata, plus actual run/fallback call sites.
for case, a in DATA.items():
    out = ART / case
    out.mkdir(exist_ok=True)
    issues = ANALYSIS[case]["issues"]
    candidates = ANALYSIS[case]["candidates"]
    for dev in ["NPU", "GPU"]:
        names = {n for z in issues for n in z["npu_kernels" if dev == "NPU" else "gpu_kernels"]}
        if dev == "NPU":
            names.update(z["name"] for z in candidates)
        else:
            names.update(n for z in candidates for n in z["gpu"])
        chosen = {(d["file"], d["line"]) for n in names for d in defs(a, dev, n)}
        blocks = []
        calls = []
        for f in sorted((ROOT / case).glob(dev + "_g*_output_code.py")):
            source = f.read_text()
            lines = source.splitlines()
            for d in a["code"][dev]:
                if d["file"] != f.name or (d["file"], d["line"]) not in chosen:
                    continue
                blocks.append(
                    f'# Source: {case}/{f.name}:{d["line"]}\n'
                    + d["metadata"]
                    + "\n"
                    + "\n".join(lines[d["line"] - 1 : d["end_line"]])
                    + "\n"
                )
            selected_names = {
                d["name"] for d in a["code"][dev] if d["file"] == f.name and (d["file"], d["line"]) in chosen
            }
            positions = []
            for i, line in enumerate(lines):
                if any(n + ".run(" in line for n in selected_names) or any(
                    t in line
                    for t in [
                        "torch.ops.aten.nll_loss",
                        " = torch.ops.aten._scaled_dot_product_",
                        "torch.ops.aten._log_softmax_backward",
                        "torch.ops.prims.fma.default(",
                        "torch.ops.aten.gelu",
                        "torch.ops.aten.scatter_add",
                        "extern_kernels.bmm(" if case.startswith("torch_multimodal_clip") else "<unused>",
                        "extern_kernels.mm(" if case.startswith("torch_multimodal_clip") else "<unused>",
                    ]
                ):
                    positions.append(i)
            covered = set()
            for i in positions:
                start = max(0, i - 5)
                end = min(len(lines), i + 5)
                new = [j for j in range(start, end) if j not in covered]
                if not new:
                    continue
                calls.append(f"# Source: {case}/{f.name}:{new[0]+1}\n" + "\n".join(lines[j] for j in new) + "\n")
                covered.update(new)
        (out / (dev + "_kernels.txt")).write_text(
            clean("\n\n".join(blocks)).rstrip("\n") + "\n"
            if blocks
            else "# No selected generated kernel. See wrapper calls.\n"
        )
        (out / (dev + "_wrapper_calls.txt")).write_text(
            clean("\n\n".join(calls)).rstrip("\n") + "\n" if calls else "# No selected external wrapper calls.\n"
        )

# Compact, portable evidence tables. Full metadata stays in selected kernel artifacts.
profiles = {
    case: {
        "model": a["model"],
        "mode": a["mode"],
        "profile": a["profile"],
        "summary": a["summary"],
        "whole_model": a["whole_model"],
    }
    for case, a in DATA.items()
}
(ART / "profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n")
(ART / "issues.json").write_text(json.dumps(ANALYSIS, ensure_ascii=False, indent=2) + "\n")
manifest = []
for f in sorted(ROOT.rglob("*")):
    if f.is_file() and f.suffix in [".csv", ".py"]:
        manifest.append({"file": str(f.relative_to(ROOT)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
(ART / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
with (ART / "issues.csv").open("w") as f:
    fields = [
        "case",
        "kind",
        "npu_kernels",
        "ne",
        "ni",
        "ge",
        "gi",
        "npu_gain",
        "gpu_gain",
        "budget_ms",
        "share",
        "model_reduction",
        "gain_pp",
        "component",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for c, zs in ANALYSIS.items():
        for z in zs["issues"]:
            row = {k: z.get(k, "") for k in fields}
            row["case"] = c
            row["npu_kernels"] = ";".join(z["npu_kernels"])
            w.writerow(row)

# Extra candidate analysis is specific to generated statements, not a claim of a measured regression.
SPECIAL = {
    "nfnet_l0_Train": "`triton_poi_fused_avg_pool2d_42` 对 f32[128,1536,14,14] 做 2×2/stride=2 pooling。DSL 将输出组织成 y0=1376256、x1=7，固定 X1BLOCK_SUB=7，只沿 y0 分配 program，每个输出用四条 stride=2 load，再加和除 4；没有复杂数学运算。优先核对宽度 7 的非整齐 tile、y0 分块和实际生成的访存/向量指令。资料可确认是这个生成实现异常，尚不能区分 tile 选择与 Triton_ascend/npuir 编译质量。",
    "basic_gnn_edgecnn_Train": "`triton_red_fused_sum_threshold_backward_3` 的输入为 f32[200000,64]；split_axis=[0] 只沿 64 个输出通道分 program，200000 条边在 program 内 loop_r1 归约，并在同一循环写回 ReLU 导数。应在 NPUIndexTritonKernel.reduction / triton_config_npu_index 检查归约维拆分，或拆开 pointwise 与 bias sum 并采用多阶段归约。当前源码没有沿 edge 维分派独立 program；实际 tile/config 仍待补。",
    "deit_tiny_patch16_224.fb_in1k_Train": "`triton_red_fused_native_layer_norm_backward_vi_7` 计算参数梯度，按 192 个通道分 program，在 r1=25216 的循环中累计两个归约，load 索引为 x0+192*r1。这是参数梯度的大外轴归约，并非 LayerNorm 输入梯度的逐行归约。具体应检查 codegen 的归约维切分、多阶段参数梯度与双输出调度；不要泛化成所有 LayerNorm 都慢。",
    "lennard_jones_Train": "`triton_per_fused_sum_tanh_backward_2` 对 f32[4096,16] 计算 tanh 导数与列和；size_hints={x0:16,r1:4096}、split_axis=[0]，最多只有 16 个 x0 program（配置合并通道后还会更少），整个 4096 维在一个 persistent tile 内归约。优先优化小输出维情况下的分阶段归约/融合策略。GPU 把导数 pointwise 和 sum 分拆，避免把所有并行度压在 16 个输出上。",
    "MobileBertForMaskedLM_Train": "除 `_5` 外六组归约已在指定分支复测：`_11`、`_9`、`_7`、`_13` 确认编译收益缺口，主要入口为 IR `num_splits` 和 `TileGenerator`。四组自动分段外推约 50 ms/step；另发现每步 24 次重复 bias sum。`_20` 有 NPU 可达性实测但 GPU 单项缺口未确认，`_0` 分段无收益，不能给七组统一归因。见[首项分析](../mobilebert-parameter-reduction-20261103.md)与[其余 kernel 分析](../mobilebert-remaining-reductions-20261103.md)；没有修改框架或验证整网修复。",
    "AlbertForMaskedLM_Train": "额外优先核对 `_20`（90.679 ms）和 `_17`（51.366 ms）。`_20` 的名字含 matmul_backward，实际 DSL 为多个 tl.sum、permute 和累加，并没有执行裸矩阵乘；其 x4/x5=64/64、r2/r3=4/512，外层 x4 使用 Python range。`_17` 同时读取 12 个来源并逐个归约。应检查共享参数梯度累加导致的多归约融合、codegen 轴映射和寄存器压力。GPU 采用多条 sum/transpose/activation 归约。当前没有把这两个 kernel 的完整语义逐调用对齐，不能把 142.044 ms 全部当作可回收收益。",
    "MobileBertForMaskedLM_Infer": "`triton_red_fused__log_softmax_12` 的 4.669 ms 是具体候选，但 GPU 的 prepare_softmax_online 已耗时约 6.105 ms，不能看到 NPU 比自身 Eager 慢就认定 GPU 有更高收益；该局部保守筛选未通过，暂不承诺优化预算。",
    "beit_base_patch16_224_Train": "`triton_per_fused__safe_softmax_add_clone_index_10` 为明确的大 kernel（34.230 ms），但 GPU Eager/Inductor 均使用 efficient_attention，不能拿 NPU softmax 单独对比 GPU 完整 attention。需要以两次 BMM 及 softmax/dropout 构成完整前向区域再比较；已确认的 metadata 重复调用独立计入前项。",
    "torch_multimodal_clip_Infer": "额外具体点为 `aclnnBatchMatMul_BroadcastToAiCore_BroadcastTo`（约 2.231 ms）。NPU wrapper 的 QKV projection 为 bmm，右矩阵 (50,768,2304)、stride=(0,1,768)；GPU 把同一投影展平为 mm((1600,768),(768,2304))。NPU Eager 没有该 BroadcastTo kernel。优先检查 NPU matmul→bmm lowering/布局选择以及 CANN 对 stride=0 batch 的展开；这是插入的广播开销，若可消除，该项成本就是回收上限，需逐次 profiler 关联后确认，不能连同 BMM 全部算收益。",
    "torch_multimodal_clip_Train": "前向另有 `aclnnBatchMatMul_BroadcastToAiCore_BroadcastTo`（约 2.243 ms）；NPU 使用带 stride=0 权重的 QKV bmm，GPU 使用展平 mm。检查 matmul→bmm lowering/布局与 CANN batch 广播展开，属于额外 materialization 候选。2.243 ms 是消除该额外 kernel 的上限，尚未计入达到 GPU 局部提升比的预算。",
}


def evidence_links(case):
    p = "../../../../../artifacts/cicd/performance/full-survey-20261102/" + case + "/"
    return f"[NPU 完整 DSL]({p}NPU_kernels.txt)、[GPU 完整 DSL]({p}GPU_kernels.txt)、[NPU 调用边界]({p}NPU_wrapper_calls.txt)、[GPU 调用边界]({p}GPU_wrapper_calls.txt)"


def kernel_hint(a, dev, n):
    dd = defs(a, dev, n)
    if not dd:
        return "外部调用；见 wrapper 调用边界"
    d = dd[0]
    path = ROOT / (a["model"] + "_" + a["mode"]) / d["file"]
    body = "\n".join(path.read_text().splitlines()[d["line"] - 1 : d["end_line"]])
    hints = re.search(r"size_hints=(\{[^\n]+\})", body)
    split = re.search(r"'split_axis': (\[[^\]]*\])", body)
    params = re.findall(r"#   %[^\n]*PlaceHolder[^\n]*", d["metadata"])
    inp = ""
    if params:
        p = re.search(r'Tensor "([^"]+)"', params[0])
        inp = p.group(1) if p else ""
    value = f'{d["file"]}:{d["line"]}；' + ("输入 " + code(inp) + "；" if inp else "")
    if hints:
        value += "size_hints=" + code(hints.group(1)) + "；"
    if split:
        value += "split_axis=" + code(split.group(1)) + "；"
    if len(dd) > 1:
        locations = ", ".join(dict.fromkeys(f'{v["file"]}:{v["line"]}' for v in dd))
        value = f"同名/截断名对应 {len(dd)} 个定义（{locations}）；以下尺寸仅展示首个入口：" + value
    return value.rstrip("；")


for model in sorted({a["model"] for a in DATA.values()}):
    text = [
        "---",
        "title: " + json.dumps("CICD 性能分析：" + model, ensure_ascii=False),
        "---",
        "",
        "# " + model + "：Infer / Train 编译收益差距",
        "",
        "按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。",
        "",
        "局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。",
        "",
    ]
    if model == "MobileBertForMaskedLM":
        text += [
            "2026-11-03 更新：[首项分析](../mobilebert-parameter-reduction-20261103.md)及[其余归约分析](../mobilebert-remaining-reductions-20261103.md)已在物理卡 1、2、`pta_213_cicd`、`test/cicd_benchmark` 对应安装版本复测；下方仍是原七组粗筛统计，不能把 `_20`、`_0` 一并视为已确认单项问题。",
            "",
        ]
    for mode in ["Infer", "Train"]:
        text += ["## " + mode, ""]
        case = model + "_" + mode
        if case not in DATA:
            text += ["本批目录没有此场景，无法给出性能结论。", ""]
            continue
        a = DATA[case]
        zs = ANALYSIS[case]
        p = a["profile"]
        ne, ni, ge, gi = [p[k]["total_ms"] for k in ["NPU_Eager", "NPU_Inductor", "GPU_Eager", "GPU_Inductor"]]
        text += [
            f'Batch={a["summary"]["NPU"].get("batch_size","未知")}；四份数据均按 10 个采样 step 归一化为 ms/step。',
            "",
            "| 设备 | Eager ms | Inductor ms | 提升比 |",
            "| --- | ---: | ---: | ---: |",
            f"| NPU | {ne:.6f} | {ni:.6f} | {pct(1-ni/ne)} |",
            f"| GPU | {ge:.6f} | {gi:.6f} | {pct(1-gi/ge)} |",
            "",
            f"达到 GPU **整网**提升比时，NPU 整步目标为 {ne*gi/ge:.6f} ms；当前相对该目标的差额为 {ni-ne*gi/ge:.6f} ms。该整步差额只作背景，不分摊给单个 kernel。",
            "",
            evidence_links(case) + "。",
            "",
        ]
        missing_gpu = [n for n in p["GPU_Inductor"]["groups"] if n.startswith("triton") and not defs(a, "GPU", n)]
        if missing_gpu:
            text += [
                "GPU profile 中有以下 kernel 未在本批 wrapper 找到同名定义，需要补充对应编译图；"
                "本场景不依赖这些缺失 DSL 做额外的算子族收益归因：",
                "",
                *["- " + code(n) for n in missing_gpu],
                "",
            ]
        if not zs["issues"]:
            text += ["本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。", ""]
        for i, z in enumerate(zs["issues"], 1):
            component, detail = z["component"], z["detail"]
            if case == "MobileBertForMaskedLM_Train" and z["kind"] == "affine_grad_reduce":
                component = "NPU Inductor `ir.py::num_splits` 的 outer reduction 分段策略，以及 runtime `TileGenerator` 候选生成"
                detail = "2026-11-03 单项深挖确认 `_11`、`_9`、`_7`、`_13` 的归约计划/候选问题，并发现 24 次重复 bias sum。GPU 对位已纠正；`_20` 单项 GPU 收益缺口待确认，`_0` 实测不支持强行分段。详见[逐 kernel 定位](../mobilebert-remaining-reductions-20261103.md)；下表保留七组原操作族粗筛口径。"
            text += [
                f'### {i}. {z["title"]}',
                "",
                f"**优化入口：{component}。** {detail}",
                "",
                "| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |",
                "| --- | ---: | ---: | ---: |",
            ]
            for n in z["npu_kernels"]:
                cost = z.get("npu_selected_ms", {}).get(n, p["NPU_Inductor"]["groups"][n]["ms"])
                count = z.get("npu_selected_count", {}).get(n, p["NPU_Inductor"]["groups"][n]["count"])
                text.append(f'| {code(short(full(a,"NPU",n)))} | {count:.1f} | {cost:.6f} | {pct(cost/ni)} |')
            text += ["", f'本项 kernel/区间合计 **{z["ni"]:.6f} ms，占整步 {pct(z["share"])}**。', ""]
            if "ne" in z:
                text += [
                    "| 局部口径 | Eager ms | Inductor ms | 提升比 |",
                    "| --- | ---: | ---: | ---: |",
                    f'| NPU（Eager 上界 / 选中 kernel 成本下界） | {z["ne"]:.6f} | {z["ni"]:.6f} | ≤{pct(z["npu_gain"])} |',
                    f'| GPU（Eager 下界 / compiled 上界） | {z["ge"]:.6f} | {z["gi"]:.6f} | ≥{pct(z["gpu_gain"])} |',
                    "",
                    f'判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **{(z["gpu_gain"]-z["npu_gain"])*100:.2f} 个百分点**。{z["notes"]}',
                    "",
                    "NPU Eager 对账项：" + ", ".join(code(n) for n in z["npu_eager_groups"]) + "。",
                    "",
                    "GPU Eager 对账项：" + ", ".join(code(n) for n in z["gpu_eager_groups"]) + "。",
                    "",
                ]
            else:
                text += [z["notes"], ""]
            text += [
                "GPU 对应实现/保守收费项：",
                "",
                "| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |",
                "| --- | ---: | ---: | ---: |",
            ]
            for n in z["gpu_kernels"]:
                g = p["GPU_Inductor"]["groups"][n]
                text.append(
                    f'| {code(short(full(a,"GPU",n)))} | {g["count"]:.1f} | {g["ms"]:.6f} | {pct(g["ms"]/gi)} |'
                )
            text += [
                "",
                f'**贡献粗估：{z["budget_ms"]:.6f} ms/step，相当于当前 NPU Inductor 整步耗时的 {pct(z["model_reduction"])}，使整网提升比增加 {z["gain_pp"]:.3f} 个百分点，单独达到该预算后整网提升比约为 {pct(1-ni/ne+z["budget_ms"]/ne)}。**',
                "",
                "该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。",
                "",
                "直接代码证据：",
                "",
            ]
            for n in z["npu_kernels"][:3]:
                text.append("- " + code(full(a, "NPU", n)) + "：" + kernel_hint(a, "NPU", n) + "。")
            text += [""]
            if z["kind"] in ["attention_seed", "fma_duplicate"]:
                text += [
                    "历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。",
                    "",
                ]
        if case in SPECIAL:
            text += ["### 额外具体检查点", "", SPECIAL[case], ""]
        text += [
            "### 未计入收益的具体候选",
            "",
            "以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。",
            "",
            "| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |",
            "| --- | ---: | ---: | --- |",
        ]
        for c in zs["candidates"]:
            fs = set(c["aten"])
            if any("softmax" in f for f in fs):
                where = "NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界"
            elif any("norm" in f or f == "sum" for f in fs):
                where = "NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约"
            elif "gelu_backward" in fs:
                where = "NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界"
            elif "index_select" in fs or "scatter_add" in fs:
                where = "NPU Inductor gather/scatter lowering 与 producer-consumer 融合"
            elif "BroadcastTo" in c["name"]:
                where = "NPU Inductor 布局/materialization 与 CANN 广播路径；先定位触发的外部调用"
            else:
                where = "NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层"
            text.append(
                f'| {code(short(full(a,"NPU",c["name"])))} | {c["ms"]:.6f} / {c["count"]:.1f} | {pct(c["share"])} | {where} |'
            )
        text += [""]
        for c in zs["candidates"]:
            text += [
                "- "
                + code(full(a, "NPU", c["name"]))
                + "："
                + kernel_hint(a, "NPU", c["name"])
                + "。GPU 检查入口："
                + (", ".join(code(n) for n in c["gpu"][:3]) or "需对照外部 ACLNN/CUDA 调用和 FX 图")
                + "。"
            ]
        text += [
            "",
            "本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。",
            "",
        ]
    (DOC / "models" / (model + ".md")).write_text("\n".join(text).rstrip("\n") + "\n")

# Aggregate overview and complete coverage table.
allissues = [(c, z) for c, a in ANALYSIS.items() for z in a["issues"]]
counts = Counter(z["kind"] for _, z in allissues)
text = [
    "---",
    "title: CICD 全量性能分析：编译收益差距与共性问题",
    "---",
    "",
    "# CICD 全量性能分析：编译收益差距与共性问题",
    "",
    f"本批共 **40 个模型、78 个场景**（Infer 40、Train 38），已逐场景读取四份设备耗时 CSV，并索引 **4,379 个生成 kernel 定义**及其 wrapper/FX。按模型分别归档，均分 Infer / Train。本轮没有访问 NPU_A5，没有修改编译器。",
    "",
    f'保守局部筛选与已知重复执行/回退机制共列出 **{len(allissues)} 个场景级检查项，覆盖 {sum(bool(a["issues"]) for a in ANALYSIS.values())} 个场景**；每项都给出具体 kernel、代码位置、NPU/GPU 做法、占比及贡献预算。其余场景列出明确候选和缺失证据，不强行认定劣化。',
    "",
    "2026-11-03 补充：[MobileBERT 逐 kernel 深挖](mobilebert-remaining-reductions-20261103.md)已在指定环境与物理卡 1、2 实测；四组确认 IR 分段归约/候选问题，另两组不计入已确认单项缺口。约 50 ms/step 为小图外推。本文原 40 模型统计及粗筛预算保持不变。",
    "",
    "## 比较方法与数字边界",
    "",
    "`r_N = 1 - N_I/N_E`，`r_G = 1 - G_I/G_E`。只筛选同一语义计算的 `r_N < r_G`；裸 MM/AddMM 若 GPU 编译侧本身没有收益，不以 NPU/GPU 绝对耗时差列为优化点。",
    "",
    "局部预算为 `B = N_I - N_E × G_I/G_E`；kernel 占比为 `N_I/T_NI`；达到局部目标对当前整步的耗时贡献为 `B/T_NI`，对整网提升比的贡献为 `100×B/T_NE` 个百分点。**不使用整网 Eager 比例缩放单个 kernel。**",
    "",
    "逐调用的 GPU CSV 没有 shape，它是按 kernel 名聚合的。对有混合融合边界的操作族，本轮采用保守办法：NPU Eager 收费整个相关操作族，NPU compiled 只计列出的纯实现/明确融合实现；GPU Eager 只计对应核心操作，GPU compiled 将全部覆盖这些核心操作的生成 kernel 及附带融合工作都收费。故 NPU 提升比是上界、GPU 提升比是下界。若上界仍小于下界，才列入。",
    "",
    "这种操作族边界是粗筛，不等于已经逐层逐 shape 对齐。预算依赖核心计算语义可对应；需要用 FX 与逐次 profile 最终确认。文档中的 budget 是目标预算/保守差距估算，**不是已验证可全部回收的收益**。首层 attention 补充了全 12 层收费给首层的保守四方筛选，并另列同设备后续层控制估算；FMA 重复执行使用直接多出的调用成本，不冒充 GPU 局部比例预算。多项预算不可简单相加。",
    "",
    "单位为设备 kernel duration 之和的 ms/step，不是主机 wall time。NPU 每场景按 CSV 中 10 个 Step Id 归一化，GPU 按 summary/profile_steps=10 归一化；四份数据 batch 一致。当前 NPU 数据与此前两模型数据有小幅变化，本文全部重新计算。原始 profile 的完整软件提交与构建身份缺失；历史源码/CPU 验证单独引用，不视为当前重新实测。",
    "",
    "## 共性问题及优化责任",
    "",
    "本表按“占整网缺口”的中位数降序排列。",
    "以下占比按已列出该问题的场景逐个计算，再取中位数及最小–最大范围，不按跨模型毫秒数加权；仅有一个场景时列单值。",
    "**问题耗时占比**为本项 kernel/区间耗时除以当前 NPU Inductor 整步耗时；"
    "**劣化预算占比**为本项预算 B 除以同一个整步耗时，表示该项若达到局部目标可减少多少整步耗时。",
    "**占整网缺口**为 B / D，其中 D = T_NI − T_NE × T_GI/T_GE，"
    "只统计 D > 0 的场景，表示该项预算可解释多少达到 GPU 整网提升比所需的差距。"
    "它与 kernel 耗时占比不同，可能因其他区域的收益抵消而超过 100%；各类存在重叠，不能相加。",
    "",
    "| 具体问题 | 场景数 | 问题耗时/整步：中位（范围） | 劣化预算/整步：中位（范围） | 占整网缺口：中位 | 优化入口与 GPU 做法 |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
labels = {
    "attention_seed": (
        "首层 attention constexpr/checker 回退",
        "NPUIndexTritonKernel.codegen_kernel 的 constexpr 常量绑定及 checker 回退；GPU 无此首层异常。",
    ),
    "fma_duplicate": (
        "prims.fma metadata 第一组 Mul/Add 重复执行",
        "make_prim.meta 调用真实 impl_aten；_prim_impl 随后再次执行；GPU 融合在 kernel 内。已核对本批 wrapper 的 prims.fma 调用与 profile 连续调用窗口。",
    ),
    "loss_forward": (
        "Infer log_softmax→NLLLoss 完整矩阵写回",
        "NPU nll_loss_forward decomposition/lowering 使消费者留在外部调用；GPU 融合 target 提取和归约。",
    ),
    "loss_backward": (
        "Train NLLLossBackward→LogSoftmaxBackward 完整 dlogP",
        "NPU 两条 decomposition/lowering 边界；GPU loss derivative 融合。",
    ),
    "gelu": (
        "GELU 前向外部调用阻断 bias/pointwise 融合",
        "NPU activation lowering/fusion；GPU 含 GELU 的融合 kernel。",
    ),
    "scatter": (
        "GNN scatter_add 与 gather/mul 分离",
        "NPU scatter_add lowering 和 scheduler fusion；GPU 生成含 scatter 的融合 kernel。",
    ),
    "ln_backward": (
        "LayerNorm backward 归约映射",
        "NPU reduction codegen/tiling/runtime autotune；GPU 采用不同归约拆分。",
    ),
    "sum": ("具体 sum kernel 的归约调度", "NPU codegen 归约维切分、多阶段选择；GPU sum kernel 作为保守费用上界。"),
    "relu_grad_reduce": ("ReLU backward+bias sum", "NPU pointwise/归约融合与并行映射；GPU 分拆或不同归约。"),
    "gelu_grad_reduce": ("GELU backward+bias sum", "NPU 融合/归约 tiling；GPU 相应 fused/多阶段归约。"),
    "tanh_grad_reduce": (
        "tanh backward+bias sum",
        "NPU 输出维很小时的 program 并行度和分阶段归约；GPU pointwise/reduction 分拆。",
    ),
    "affine_grad_reduce": (
        "MobileBERT add/mul 多输出归约",
        "四组已确认 IR 分段/候选问题，另有 24 次重复 bias sum；`_20`、`_0` 单独保留边界。见[实测细分](mobilebert-remaining-reductions-20261103.md)；本行仍为七组历史粗筛。",
    ),
    "pool": (
        "NFNet avg_pool2d 生成实现",
        "NPU avg_pool2d lowering、pointwise tiling/runtime；先判断是否应回退 ACLNN。",
    ),
}
common_rows = []
for kind, count in counts.most_common():
    samples = [(case, issue) for case, issue in allissues if issue["kind"] == kind]
    cost_share, budget_share, gap_share = common_issue_share_cells(samples, DATA)
    gap_values = [
        issue["budget_ms"] / DATA[case]["whole_model"]["deficit_ms"]
        for case, issue in samples
        if DATA[case]["whole_model"]["deficit_ms"] > 0
    ]
    sort_share = median(gap_values) if gap_values else float("-inf")
    row = f"| {labels[kind][0]} | {count} | {cost_share} | {budget_share} | {gap_share} | {labels[kind][1]} |"
    common_rows.append((sort_share, row))
text.extend(row for _, row in sorted(common_rows, key=lambda item: item[0], reverse=True))
text += [
    "",
    "归约类具体代码入口为 `NPUIndexTritonKernel.reduction`、`codegen_kernel`，配置侧为 `triton_config_npu_index` 与 `NPUCachingAutotuner.autotune_to_one_config`；融合检查入口为 NPU scheduling 的 `can_fuse` / `candidate_tilings`。这些入口定位到 Inductor 的具体功能模块；除已复核的 constexpr/FMA 机制及 MobileBERT 专项实测外，不把首查模块写成已证实根因。相同 DSL/config 仍慢才能继续判断 Triton_ascend/npuir；外部 scatter/broadcast 则还涉及 CANN。",
    "",
    "## 优先检查的具体 kernel",
    "",
    "下表每个场景取本轮最大一个预算项，按预算排序。相邻的单个 kernel 成本和融合操作族预算不是同一个概念，完整明细见模型文档。",
    "",
    "| 场景 | 具体 NPU kernel / 点 | 本项 ms / 整步占比 | 预算 ms | 整网耗时贡献 | 提升比贡献 pp |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for c, a in sorted(ANALYSIS.items(), key=lambda v: -max([z["budget_ms"] for z in v[1]["issues"]] or [0])):
    if not a["issues"]:
        continue
    z = a["issues"][0]
    model = DATA[c]["model"]
    n = max(
        z["npu_kernels"],
        key=lambda n: z.get("npu_selected_ms", {}).get(n, DATA[c]["profile"]["NPU_Inductor"]["groups"][n]["ms"]),
    )
    text.append(
        f'| [{c}](models/{model}.md) | {code(short(full(DATA[c],"NPU",n)))} | {z["ni"]:.3f} / {pct(z["share"])} | {z["budget_ms"]:.3f} | {pct(z["model_reduction"])} | {z["gain_pp"]:.3f} |'
    )
text += [
    "",
    "还应优先核对 Albert Train 的 `_clone_matmul_backward_permute_20`（90.679 ms）与 `_add_mul_pow_sum_tanh_tanh_bac_17`（51.366 ms），以及 CLIP QKV projection 引出的 BroadcastTo。它们已有具体代码节点，但尚未闭合四方语义边界，所以没有放进“可与 GPU 提升比对齐”的预算表。Albert 的 `_20` 实际是 sum/permute/累加，不能被名字中的 matmul_backward 误导为裸 MM。",
    "",
    "## 40 个模型的完整入口",
    "",
    "| 模型 | Infer：NPU / GPU 提升比 | Train：NPU / GPU 提升比 | 列出项数 Infer / Train |",
    "| --- | ---: | ---: | ---: |",
]
for model in sorted({a["model"] for a in DATA.values()}):
    vals = []
    ic = []
    for mode in ["Infer", "Train"]:
        c = model + "_" + mode
        if c in DATA:
            w = DATA[c]["whole_model"]
            vals.append(pct(w["NPU_gain"]) + " / " + pct(w["GPU_gain"]))
            ic.append(str(len(ANALYSIS[c]["issues"])))
        else:
            vals.append("数据缺失")
            ic.append("—")
    text.append(f"| [{model}](models/{model}.md) | {vals[0]} | {vals[1]} | {ic[0]} / {ic[1]} |")
text += [
    "",
    "## 证据、复算及待补验证",
    "",
    "- [逐 kernel 四方原始汇总](../../../../artifacts/cicd/performance/full-survey-20261102/profiles.json)、[检查项与候选 JSON](../../../../artifacts/cicd/performance/full-survey-20261102/issues.json)、[预算 CSV](../../../../artifacts/cicd/performance/full-survey-20261102/issues.csv)。",
    "- 每个模型链接到该批 NPU/GPU 的完整选中 generated kernel（保留原格式），以及实际 run/外部调用边界；只去除了环境路径、backend_hash 等信息。",
    "- [decomposition/fallback 源码参考](../../../../artifacts/cicd/performance/full-survey-20261102/component_policy_reference.txt)：历史安装源码的 loss/GELU 排除策略，以及本地 checkout 的 scatter fallback 和归约入口；入口参考不等于 profile 构建身份。",
    "- [输入文件 SHA256 清单](../../../../artifacts/cicd/performance/full-survey-20261102/source_manifest.json)覆盖原始 CSV、wrapper、FX，原始输入不上传，防止大体积/环境路径公开。",
    "- [本地复算脚本及用法](../../../../scripts/tools/cicd/full_survey_20261102/README.md)。脚本不导入 torch、不连接远端，只读用户提供数据。",
    "- 当前校验范围为全部 CSV 的单位/步数/batch、预算公式、40 模型/78 场景覆盖、已选择 kernel 的代码/调用证据和链接检查；没有修复性能/精度验证。",
    "- 后续只对选中的最大点补 GPU 逐调用 shape/stride、NPU/GPU selected config、固定输入单 kernel 与 IR；预算最大的 kernel 优先，避免再次泛化成整条反向链路。",
    "- PLBart Train、TrOCR Train 各有 3 个 GPU profile 名称未在本批 wrapper 匹配到定义，均已在模型页列明；需要补对应编译图，不基于这些缺失 DSL 新增算子族收益归因。",
    "",
    "历史：[首层 attention 根因](electra-layoutlm-train-first-attention-20261031.md)。此前[两模型绝对耗时差分析](electra-layoutlm-train-gpu-inductor-gap-20261102.md)已标记口径被替代，其中 MM/AddMM 70.049 ms 不再作为编译收益优化依据。",
    "",
]
(DOC / "full-survey-20261102.md").write_text("\n".join(text))
print(
    "rendered",
    len(DATA),
    "scenarios;",
    len(allissues),
    "issues; artifacts",
    sum(f.stat().st_size for f in ART.rglob("*") if f.is_file()),
)
