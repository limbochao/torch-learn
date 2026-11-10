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
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--stop-timer")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("name must contain only letters, digits, hyphens and underscores")
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
        jobs_file = artifact / "jobs.json"
        jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else []
        by_method = {(row["file"], row["name"]): row for row in jobs}
        columns = ["测试文件", "函数名", "最新结果", "实例结果", "当前分析"]
        rows = []
        for row in selected:
            job = by_method.get((row["file"], row["name"]), {})
            reports = job.get("reports", {})
            counts = Counter(report["status"] for report in reports.values())
            detail = next((r["detail"] for r in reports.values() if r["status"] != "PASS"), "")
            status = job.get("status", "待执行")
            analysis = "—" if status == "PASS" else detail.split("\n\n")[0][:600]
            if not analysis:
                analysis = state.get("error", "待执行")
            if state["stage"] == "error" and status in ("待执行", "PENDING", "RUNNING"):
                status = "ERROR"
            rows.append([row["file"], row["name"], status,
                         "; ".join(f"{key} {value}" for key, value in sorted(counts.items())), analysis])
        target = REPO / "docs/notes/compiler/codegen" / (args.name + ".csv")
        with target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        notes = target.with_suffix(".md")
        summary = Counter(row[2] for row in rows)
        text = (f"# 2.13 ComboKernel 同步分支验证\n\n"
                f"使用 NPU_A5 的 `pta_213_case` 环境及本地 `sync/combo-kernel-master` 对应包。\n"
                f"torch_npu：`{state.get('torch_npu', '待获取')}`，提交：`{state.get('torch_npu_git', '待获取')}`。\n"
                f"PyTorch：`{state.get('torch', '待获取')}`，Triton：`{state.get('triton', '待获取')}`。\n"
                f"测试快照：`{state.get('test_commit', '待获取')}`，目录为 `v2.13.0/inductor_npu`。\n\n"
                f"状态：`{state['stage']}`；{len(selected)} 个源方法，"
                f"{state.get('instances', '待收集')} 个参数化实例。方法结果：`{dict(summary)}`。\n"
                f"实例结果：`{state.get('instance_results', {})}`。\n\n"
                f"[用例表]({target.name})；"
                f"[执行脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)；"
                f"[原始证据](../../../../artifacts/{args.name}/)。\n\n"
                "使用原生 pytest，无自定义 pytest 插件；仅设置选卡变量。"
                "保留指定目录已有的 NPU 适配和全部原始断言，没有沿用 2.10 foreach 放宽数量检查的修改。\n\n"
                "表格中的异常摘要来自 JUnit 报告；尚未逐项定位的错误不能直接据此归因于 ComboKernel。"
                "PASS 项不保留历史失败分析。\n")
        if terminal:
            text += (f"\n执行前后安装的 Inductor Python 源码哈希一致："
                     f"`{state.get('installed_source_unchanged', False)}`。\n")
        notes.write_text(text)
        paths = [str(path.relative_to(REPO)) for path in (artifact, target, notes)]
        command(["git", "diff", "--check"], cwd=REPO)
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
