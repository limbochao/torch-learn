"""Wait for one run_validation.py phase, archive its evidence and publish refreshed latest results."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

from archive_liulm_run import clean, clean_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-phase", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-liulm-0fd5a10-20261106")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--once", action="store_true", help="Check once for a systemd timer; return if still running")
    parser.add_argument("--stop-timer", help="Stop this user timer after results have been published")
    args = parser.parse_args()
    if args.stop_timer:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+\.timer", args.stop_timer), "Invalid user timer name"
    args.raw_root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[x for x in sys.argv[1:] if x != "--detach"]]
        with (args.raw_root / "publisher.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.raw_root / "publisher.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"pid": process.pid, "status": str(args.raw_root / "publisher_status.json")}))
        return

    lock = (args.raw_root / "publisher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A publisher is already running for this phase")

    def stop_timer():
        if args.stop_timer:
            subprocess.run(["systemctl", "--user", "stop", args.stop_timer], check=True)

    status_file = args.raw_root / "publisher_status.json"
    terminal_states = {"complete", "published_with_execution_errors", "validation_error"}
    if status_file.exists() and json.loads(status_file.read_text()).get("state") in terminal_states:
        stop_timer()
        return

    def status(state, **extra):
        (args.raw_root / "publisher_status.json").write_text(
            json.dumps({"state": state, "time": time.time(), **extra}, ensure_ascii=False, indent=2) + "\n")

    def git(*command, env=None):
        return subprocess.check_output(["git", *command], cwd=args.repo, env=env)

    try:
        status("waiting_for_retest")
        probe = ("import json;from pathlib import Path;p=Path(" + repr(args.remote_phase) + ");"
                 "print('RETEST_STATUS='+json.dumps(json.loads((p/'validation_status.json').read_text())))")
        while True:
            output = subprocess.check_output(["ssh", args.host, "python -c " + shlex.quote(probe)],
                                             text=True, timeout=60)
            state = json.loads(next(x[len("RETEST_STATUS="):] for x in output.splitlines()
                                    if x.startswith("RETEST_STATUS=")))
            if state["stage"] in {"complete", "error"}:
                break
            if args.once:
                status("waiting_for_retest", validation_state=state)
                return
            time.sleep(max(60, args.interval))
        status("archiving_retest", validation_state=state)
        archive = ".retest-evidence.tar.gz"
        bundle_script = ("import hashlib,tarfile;from pathlib import Path;p=Path(" + repr(args.remote_phase)
                         + ");names=['evidence','validation_status.json','combo_npu_a5_selected.json',"
                         "'expected_nodes.json','expected_source_sha256.json','validation.log',"
                         "'validation_context.json'];t=tarfile.open(p/" + repr(archive) + ",'w:gz');"
                         "[t.add(p/n,arcname=n) for n in names if (p/n).exists()];t.close();"
                         "print(hashlib.sha256((p/" + repr(archive) + ").read_bytes()).hexdigest()+'  '"
                         + "+" + repr(archive) + ")")
        command = "python -c " + shlex.quote(bundle_script)
        output = subprocess.check_output(["ssh", args.host, command], text=True, timeout=180)
        expected_hash = next(x.split()[0] for x in output.splitlines() if x.endswith("  " + archive))
        bundle = args.raw_root / archive
        subprocess.run(["scp", "-q", args.host + ":" + args.remote_phase + "/" + archive, str(bundle)],
                       check=True, timeout=180)
        assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected_hash
        with tarfile.open(bundle) as tar:
            tar.extractall(args.raw_root, filter="data")
        phase = Path(args.remote_phase).name
        artifact = args.repo / "artifacts" / args.artifact
        destination = artifact / "retests" / phase
        destination.mkdir(parents=True, exist_ok=True)
        receipts = []
        for path in args.raw_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".py", ".log"}:
                continue
            relative = path.relative_to(args.raw_root)
            if path.name.startswith("publisher") or set(relative.parts) & {
                    "installed_source", "executed_tests", "original_tests", "scripts", "generated_python"}:
                continue
            raw = path.read_bytes()
            content = raw.decode()
            if path.suffix == ".json":
                content = json.dumps(clean_value(json.loads(content)), ensure_ascii=False, indent=2) + "\n"
            elif path.suffix == ".jsonl":
                content = "\n".join(json.dumps(clean_value(json.loads(x)), ensure_ascii=False)
                                    for x in content.splitlines()) + "\n"
            else:
                content = clean(content)
            relative = Path(*relative.parts[1:]) if relative.parts[0] == "evidence" else relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            receipts.append({"artifact": str(relative), "original_sha256": hashlib.sha256(raw).hexdigest(),
                             "archived_sha256": hashlib.sha256(content.encode()).hexdigest()})
        (destination / "archive_manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")
        context_path = destination / "validation_context.json"
        context = json.loads(context_path.read_text()) if context_path.exists() else {}
        if context.get("full_suite_run") and state["stage"] != "complete":
            status("validation_error", validation_state=state, archived_evidence=str(destination),
                   reason="Full validation or source integrity failed; live result table was not updated")
            stop_timer()
            return
        result = json.loads((destination / "functional_results.json").read_text())
        if state["stage"] == "complete":
            check = json.loads((destination / "integrity_checks.json").read_text())
            assert check["all_expected_instances_reported"] and check["installed_source_unchanged"]
            expected = json.loads((destination / "expected_nodes.json").read_text())
            collected = json.loads((destination / "collection.json").read_text())
            assert set(expected) == {x["nodeid"] for x in collected}
            assert result["expected_instances"] == result["completed_instances"] == len(expected)
            if state.get("with_f3"):
                audits = list(destination.glob("audit.test_*.jsonl"))
                assert len(audits) == len(result["methods"])
                for path in audits:
                    events = [json.loads(line) for line in path.read_text().splitlines()]
                    assert any(e.get("event") == "existing_fix_loaded" and e.get("source_sha256")
                               == "510a63aa914a3384d68d5aac662aab05c7919f21584f109e04fdcc1476fcd77c"
                               for e in events)
            if context:
                assert context.get("with_f3_overlay", False) == state.get("with_f3", False)
        index = destination.parent / "index.json"
        phases = json.loads(index.read_text()) if index.exists() else []
        if phase not in phases:
            phases.append(phase)
        index.write_text(json.dumps(phases, indent=2) + "\n")
        subprocess.run([sys.executable, str(Path(__file__).with_name("update_liulm_results.py")),
                        "--repo", str(args.repo), "--artifact", args.artifact], check=True)
        status("publishing_retest", counts=result["instance_counts"])
        scopes = ["artifacts/" + args.artifact,
                  "docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv"]

        def other_index():
            return [r for r in git("ls-files", "--stage", "-z").split(b"\0") if r and not any(
                r.split(b"\t", 1)[1].decode() == s or r.split(b"\t", 1)[1].decode().startswith(s + "/")
                for s in scopes)]

        assert git("branch", "--show-current").decode().strip() == "main"
        before = other_index()
        env = dict(os.environ, GIT_INDEX_FILE=str(args.raw_root / ("commit-index-" + uuid.uuid4().hex)))
        git("read-tree", "HEAD", env=env)
        git("add", "-f", "--", *scopes, env=env)
        git("diff", "--cached", "--check", "--", *scopes[1:], env=env)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=args.repo, env=env).returncode:
            message = ("docs: record Combo full validation for " + context["installed_commit"][:10]
                       if context.get("full_suite_run") else "docs: record Combo targeted validation")
            git("commit", "-m", message, env=env)
            git("reset", "HEAD", "--", *scopes)
        assert before == other_index()
        commit = git("rev-parse", "HEAD").decode().strip()
        git("push", "origin", "HEAD:main")
        for attempt in range(3):
            try:
                assert git("ls-remote", "origin", "refs/heads/main").decode().split()[0] == commit
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(3)
        status("complete" if state["stage"] == "complete" else "published_with_execution_errors",
               commit=commit, counts=result["instance_counts"], phase=phase)
        stop_timer()
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
