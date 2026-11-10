"""Check one native 2.10 run and archive completed results into its own table."""

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
import uuid

from archive_liulm_run import clean, clean_value


def remote(host, root, code):
    command = "cd " + shlex.quote(root) + " && python - <<'PY'\n" + code + "\nPY"
    output = subprocess.check_output(["ssh", host, command], text=True, timeout=180)
    return json.loads(next(line[7:] for line in output.splitlines() if line.startswith("RESULT=")))


def classify(method):
    counts = method["counts"]
    total = method["expected_instances"]
    if not method["job_finished"] or method["completed_instances"] != total or method["returncode"] not in (0, 1):
        return "执行异常", "实例未完整结束或 pytest 进程异常，不能计为通过"
    if any(row["phase"] != "call" and row["outcome"] == "failed" for row in method["failures"]):
        status = "执行异常"
    elif counts.get("failed", 0):
        status = "失败"
    elif counts.get("skipped", 0):
        status = "跳过"
    elif counts.get("passed", 0) == total and method["returncode"] == 0:
        return "通过", "—"
    else:
        return "执行异常", "报告与进程退出码不一致"
    details = []
    for failure in method["failures"]:
        text = failure.get("longrepr", "")
        lines = [line.strip() for line in text.splitlines() if re.match(r"E\s+", line)]
        summary = " / ".join(lines[-3:]) if lines else " ".join(text.splitlines()[-3:])
        summary = clean(summary)[:650]
        if summary not in details:
            details.append(summary)
    return status, "；".join(details[:2]) or "见本轮日志，根因待分析"


def publish(repo, artifact, raw, state):
    context = json.loads((raw / "validation_context.json").read_text())
    selected = json.loads((raw / "combo_npu_a5_selected.json").read_text())
    result = json.loads((raw / "evidence/functional_results.json").read_text())
    assert state["stage"] == "complete", "Run incomplete; archive evidence without replacing result table"
    check = json.loads((raw / "evidence/integrity_checks.json").read_text())
    assert check["all_expected_instances_reported"] and check["installed_source_unchanged"]
    assert len(selected) == len(result["methods"]) == 54
    assert result["expected_instances"] == result["completed_instances"]
    table_path = repo / "docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.csv"
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    methods = {(row["file"], row["name"]): row for row in result["methods"]}
    assert set(methods) == {(row["测试文件"], row["函数名"]) for row in rows}
    method_counts = Counter()
    for row in rows:
        method = methods[row["测试文件"], row["函数名"]]
        status, analysis = classify(method)
        method_counts[status] += 1
        counts = method["counts"]
        detail = "、".join(f"{name} {counts[key]}" for key, name in (
            ("passed", "PASS"), ("failed", "FAIL"), ("skipped", "SKIP")) if counts.get(key))
        row.pop("执行状态", None)
        row["最新结果"] = f"{status}（{detail}；共 {method['expected_instances']}）"
        row["当前分析"] = analysis
    stream = io.StringIO(newline="")
    fields = ["测试文件", "函数名", "存在性/实现差异", "2.10 原生执行说明", "最新结果", "当前分析"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table_path.write_text(stream.getvalue(), encoding="utf-8-sig")
    summary = {"method_counts": dict(method_counts), "instance_counts": result["instance_counts"],
               "source_methods": 54, "instances": result["expected_instances"], "context": context,
               "validation_complete": True}
    (artifact / "summary.json").write_text(json.dumps(clean_value(summary), ensure_ascii=False, indent=2) + "\n")
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(state["started"] + 8 * 3600))
    finished = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(state["finished"] + 8 * 3600))
    text = ["# 2.10 原生 Combo/foreach 用例执行结果", "",
            f"执行时间：{started} 至 {finished}（北京时间）。", "",
            "使用用户指定的 `v2.10.0/test_inductor` 测试目录，保留原始参数、配置、数值及结构断言。",
            "仅为 Combo 文件补充与 foreach 一致的 NPU 初始化；测试插件只负责设备初始化、选择和记录。", "",
            f"测试源码提交：`{context['test_source_commit']}`。", "",
            f"安装包：`torch_npu {context['packages']['torch_npu']}`；"
            f"`torch {context['packages']['torch']}`；`triton {context['packages']['triton']}`。", "",
            "本轮没有叠加生产代码修复。安装源码运行前后 SHA256 一致。", "",
            f"54 个源方法：" + "，".join(f"{key} {value}" for key, value in method_counts.items()) + "。",
            f"参数化实例共 {result['expected_instances']} 个：`{result['instance_counts']}`。", "",
            "方法只有在其全部实例通过时才计为 PASS；跳过和执行异常单独记录。",
            f"用户预期 51 个方法 PASS，实际为 {method_counts.get('通过', 0)} 个，"
            f"差值 {method_counts.get('通过', 0) - 51:+d}。旧 2.13 适配版曾删除部分结构断言，结果不直接等价。", "",
            "2.10 原版 `test_mean` 不主动开启 Combo，其通过仅说明原始配置下通过。", "",
            "[完整 CSV 表](inductor-combo-tests-native-v2.10.csv)；"
            f"[本轮证据](../../../../artifacts/{artifact.name}/summary.json)。", "",
            "| 测试文件 | 函数名 | 最新结果 | 当前分析 |", "| --- | --- | --- | --- |"]
    for row in rows:
        values = [row[key].replace("|", "\\|").replace("\n", " ")
                  for key in ["测试文件", "函数名", "最新结果", "当前分析"]]
        text.append("| " + " | ".join(values) + " |")
    table_path.with_suffix(".md").write_text("\n".join(text) + "\n")
    return summary


def commit_results(repo, raw, artifact):
    scopes = [str(artifact.relative_to(repo)),
              "docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.csv",
              "docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.md"]

    def git(*arguments, env=None):
        return subprocess.check_output(["git", *arguments], cwd=repo, env=env)

    def other_index():
        return [entry for entry in git("ls-files", "--stage", "-z").split(b"\0") if entry and not any(
            entry.split(b"\t", 1)[1].decode() == scope or
            entry.split(b"\t", 1)[1].decode().startswith(scope + "/") for scope in scopes)]

    assert git("branch", "--show-current").decode().strip() == "main"
    before = other_index()
    index = raw / ("commit-index-" + uuid.uuid4().hex)
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    try:
        git("read-tree", "HEAD", env=env)
        git("add", "-f", "--", *scopes, env=env)
        git("diff", "--cached", "--check", "--", *scopes[1:], env=env)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo, env=env).returncode:
            git("commit", "-m", "docs: record native 2.10 daily Combo validation", env=env)
            git("reset", "HEAD", "--", *scopes)
        assert before == other_index(), "Unrelated staged files changed"
        commit = git("rev-parse", "HEAD").decode().strip()
        git("push", "origin", "HEAD:main")
        assert git("ls-remote", "origin", "refs/heads/main").decode().split()[0] == commit
        return commit
    finally:
        index.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="NPU_A5")
    parser.add_argument("--remote-phase", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-native210-daily20261110")
    parser.add_argument("--push", action="store_true", help="Commit only the result table/evidence and push main")
    parser.add_argument("--stop-timer")
    args = parser.parse_args()
    if args.stop_timer:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+\.timer", args.stop_timer)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    lock = (args.raw_root / "publisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.raw_root / "publisher_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("state") in ("complete", "validation_error"):
        return
    state = remote(args.host, args.remote_phase,
                   "import json,pathlib; print('RESULT='+json.dumps(json.loads("
                   "pathlib.Path('validation_status.json').read_text())))")
    if state["stage"] not in ("complete", "error"):
        status_path.write_text(json.dumps({"state": "waiting", "validation": state}, indent=2) + "\n")
        print("Run remains active; next timer invocation will check again.")
        return
    code = """import hashlib,json,pathlib,tarfile
root=pathlib.Path.cwd()
bundle=root/'native-results.tar.gz'
with tarfile.open(bundle,'w:gz') as tar:
 for name in ['validation_context.json','validation_status.json','validation.log',
              'combo_npu_a5_selected.json','expected_source_sha256.json']:
  tar.add(root/name,arcname=name)
 for path in (root/'evidence').rglob('*'):
  if path.is_file() and not set(path.relative_to(root/'evidence').parts)&{'generated_python','generated_on_failure'}:
   tar.add(path,arcname=str(path.relative_to(root)))
print('RESULT='+json.dumps({'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}))
"""
    receipt = remote(args.host, args.remote_phase, code)
    bundle = args.raw_root / "native-results.tar.gz"
    subprocess.run(["scp", "-q", args.host + ":" + args.remote_phase + "/native-results.tar.gz", str(bundle)],
                   check=True, timeout=180)
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == receipt["sha256"]
    with tarfile.open(bundle) as tar:
        tar.extractall(args.raw_root, filter="data")
    artifact = args.repo / "artifacts" / args.artifact
    artifact.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in args.raw_root.rglob("*"):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".py", ".log"):
            continue
        if path.name.startswith("publisher"):
            continue
        original = path.read_bytes()
        if path.suffix == ".json":
            content = json.dumps(clean_value(json.loads(original)), ensure_ascii=False, indent=2) + "\n"
        elif path.suffix == ".jsonl":
            content = "\n".join(json.dumps(clean_value(json.loads(line)), ensure_ascii=False)
                                for line in original.decode().splitlines()) + "\n"
        else:
            content = clean(original.decode())
        target = artifact / path.relative_to(args.raw_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        manifest.append({"path": str(target.relative_to(artifact)),
                         "original_sha256": hashlib.sha256(original).hexdigest(),
                         "archived_sha256": hashlib.sha256(content.encode()).hexdigest()})
    (artifact / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if state["stage"] == "complete":
        summary = publish(args.repo, artifact, args.raw_root, state)
        status = {"state": "complete", "method_counts": summary["method_counts"],
                  "instance_counts": summary["instance_counts"], "artifact": str(artifact)}
        if args.push:
            status["commit"] = commit_results(args.repo, args.raw_root, artifact)
    else:
        status = {"state": "validation_error", "validation": state, "artifact": str(artifact)}
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))
    if args.stop_timer:
        subprocess.run(["systemctl", "--user", "stop", args.stop_timer], check=True)


if __name__ == "__main__":
    main()
