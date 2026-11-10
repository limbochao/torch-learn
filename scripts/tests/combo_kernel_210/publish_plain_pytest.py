"""Check one remote plain-pytest run; publish its table and evidence when finished."""

import argparse
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[3]


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--table-name", help="Optional stable table name when the evidence run name changes")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--stop-timer")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("name must contain only letters, digits, dots, hyphens and underscores")
    table_name = args.table_name or args.name
    if not table_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("Invalid table name")
    lock = Path("/home/qc/Repos/Ascend/tmp") / (args.name + ".publish.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        remote = Path(args.remote_root)
        code = ("import json;from pathlib import Path;"
                f"p=Path({str(remote / 'results/status.json')!r});"
                "print('COMBO_STATUS='+json.dumps(json.loads(p.read_text()) if p.exists() "
                "else {'stage':'starting'}))")
        output = command(["ssh", "NPU_A5", "python -c " + shlex.quote(code)])
        state = json.loads(next(line.split("=", 1)[1] for line in output.splitlines()
                                if line.startswith("COMBO_STATUS=")))
        print(json.dumps(state, ensure_ascii=False))
        terminal = state["stage"] in ("complete", "error")
        if not terminal and not args.initial:
            return
        artifact = REPO / "artifacts" / args.name
        artifact.mkdir(parents=True, exist_ok=True)
        command(["scp", "-r", "NPU_A5:" + str(remote / "results") + "/.", str(artifact)])
        command(["scp", "NPU_A5:" + str(remote / "run.log"), str(artifact / "run.log")])
        selected = json.loads((artifact / "manifest.json").read_text())
        layout = artifact / "table_rows.json"
        table_rows = json.loads(layout.read_text()) if layout.exists() else selected
        jobs_file = artifact / "jobs.json"
        jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else []
        by_method = {(row["file"], row["class"], row["name"]): row for row in jobs}
        columns = ["测试文件", "函数名", "最新结果", "实例结果", "当前分析"]
        rows = []
        for row in table_rows:
            source_name = row.get("source_method", row["name"])
            job = by_method.get((row["file"], row["class"], source_name), {})
            reports = job.get("reports", {})
            counts = Counter(report["status"] for report in reports.values())
            detail = next((r["detail"] for r in reports.values() if r["status"] != "PASS"), "")
            status = job.get("status", "待执行")
            analysis = "—" if status == "PASS" else detail.split("\n\n")[0][:600]
            if status == "FAIL" and "node_can_linear" in detail:
                analysis = ("NPU 调度接口同步不完整：codegen_combo_kernel 调用未定义的 node_can_linear；"
                            "调用在 try 之前，无法进入回退。2.10 的同名方法读取线性模式配置，"
                            "当前 2.13 未保留该方法和配置，需要按 2.13 调度策略适配。")
            elif status == "FAIL" and "'tuple' object has no attribute 'tiling'" in detail:
                analysis = ("分区接口未对齐 2.13：NPU generate_combo_kernel_code 仍构造四元 tuple，"
                            "上游 horizontal_partition 要求 NodeInfo，并读取 .tiling 和 .features；"
                            "在生成、执行 kernel 前即失败。")
            elif (status == "FAIL" and row["name"] == "test_single_combo_kernels"
                  and "GridNpu" in detail and "Grid1D" in detail):
                analysis = ("执行已越过同步接口错误；当前失败为生成代码断言要求 CUDA Grid1D，"
                            "NPU 普通 kernel 使用 GridNpu。数值及该实例的 kernel 数量检查已通过，"
                            "需区分测试中的后端特定检查与实际代码生成问题。")
            elif status == "FAIL" and "only supported contiguous memory format" in detail:
                analysis = ("用例 eager 基准调用 channels_last contiguous 即失败，尚未进入 torch.compile；"
                            "属于 NPU eager 内存格式支持限制，不能归因为 ComboKernel 编译失败。")
            elif status == "FAIL" and row["name"] == "test_combo_kernel_peak_memory_wide_resnet":
                analysis = ("收紧阈值与禁用 combo 的峰值均为 616101888 字节，未满足峰值下降断言；"
                            "尚未确认融合决策与内存统计根因，不据此断言为代码生成错误。")
            if not analysis:
                analysis = state.get("error", "待执行")
            if state["stage"] == "error" and status in ("待执行", "PENDING", "RUNNING"):
                status = "ERROR"
            instances = "; ".join(f"{key} {value}" for key, value in sorted(counts.items()))
            if row.get("nested"):
                status = "随外层验证/" + status
                instances = "随 " + source_name + " 验证"
            rows.append([row["file"], row["name"], status,
                         instances, analysis])
        target = REPO / "docs/notes/compiler/codegen" / (table_name + ".csv")
        with target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        notes = target.with_suffix(".md")
        summary = Counter(job["status"] for job in jobs)
        instances = Counter(report["status"] for job in jobs for report in job.get("reports", {}).values())
        if state.get("foreach_functional"):
            policy = ("foreach 已按 2.10 的 NPU 功能验证口径调整：取消本轮 27 个方法的主动 NPU skip，"
                      "NPU 上不检查 kernel 数量及 foreach 专用生成代码；数值、原地修改、梯度断言保留。"
                      "其他后端保留原融合检查。功能 PASS 不证明 FOREACH 分组融合生效。")
        else:
            policy = ("保留指定目录已有的 NPU 适配和全部原始断言，没有沿用 2.10 foreach 放宽数量检查的修改。"
                      "该目录的 foreach 方法存在主动 NPU skip，原因为“不支持 NPU FOREACH 分组融合”。"
                      "这类结果保留为 SKIP，不能作为数值正确或融合成功的证据。")
        environment = "使用原生 pytest，无自定义 pytest 插件；设置选卡变量。"
        if state.get("cache_root"):
            environment += (f"另设独立 TORCHINDUCTOR_CACHE_DIR：`{state['cache_root']}`，"
                            "避免 Python 修复后复用旧编译代码；未清理共享缓存。")
        text = (f"# 2.13 ComboKernel 同步分支验证\n\n"
                f"使用 NPU_A5 的 `pta_213_case` 环境及本地 `sync/combo-kernel-master` 对应包。\n"
                f"torch_npu：`{state.get('torch_npu', '待获取')}`，提交：`{state.get('torch_npu_git', '待获取')}`。\n"
                f"NPU Python 修复叠加提交：`{state.get('npu_fix_commit') or '无'}`。\n"
                f"PyTorch：`{state.get('torch', '待获取')}`，Triton：`{state.get('triton', '待获取')}`。\n"
                f"测试快照：`{state.get('test_commit', '待获取')}`，目录为 `v2.13.0/inductor_npu`。\n\n"
                f"状态：`{state['stage']}`；{len(table_rows)} 行表格、{len(selected)} 个独立源方法，"
                f"{state.get('instances', '待收集')} 个参数化实例。方法结果：`{dict(summary)}`。\n"
                f"已报告实例结果：`{dict(instances)}`。\n\n"
                f"[用例表]({target.name})；"
                f"[执行脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)；"
                f"[原始证据](../../../../artifacts/{args.name}/)。\n\n"
                + environment + policy + "\n\n"
                "表格中的异常摘要来自 JUnit 报告；尚未逐项定位的错误不能直接据此归因于 ComboKernel。"
                "PASS 项不保留历史失败分析。\n")
        if layout.exists():
            text += ("\n表格行顺序与 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 一致。"
                     "嵌套函数随外层方法验证，不作为独立 pytest 方法重复计数；不沿用 2.10 的排除范围。\n"
                     f"已复用相同包、相同测试文件及相同 node ID 的既有结果："
                     f"`{state.get('reused_methods', 0)}` 个方法。\n")
        if terminal:
            text += (f"\n执行前后安装的 Inductor Python 源码哈希一致："
                     f"`{state.get('installed_source_unchanged', False)}`。\n")
        notes.write_text(text)
        paths = [str(path.relative_to(REPO)) for path in (artifact, target, notes)]
        # Raw logs/JUnit/CSV evidence can contain original CRLF or trailing spaces.
        command(["git", "diff", "--check", "--", str(target), str(notes), "scripts/tests/combo_kernel_210"],
                cwd=REPO)
        if args.push:
            if command(["git", "branch", "--show-current"], cwd=REPO).strip() != "main":
                raise RuntimeError("Refusing to publish from a branch other than main")
            command(["git", "add", "--", *paths], cwd=REPO)
            diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=REPO)
            if diff.returncode == 1:
                command(["git", "commit", "--only", "-m", "docs: publish " + args.name, "--", *paths], cwd=REPO)
            elif diff.returncode:
                raise RuntimeError("git diff failed")
            command(["git", "push", "origin", "main"], cwd=REPO)
            local = command(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
            actual = command(["git", "ls-remote", "origin", "refs/heads/main"], cwd=REPO).split()[0]
            if local != actual:
                raise RuntimeError("Remote main does not match the published commit")
            print("PUBLISHED", local)
        if terminal and args.stop_timer:
            command(["systemctl", "--user", "stop", args.stop_timer])


if __name__ == "__main__":
    main()
