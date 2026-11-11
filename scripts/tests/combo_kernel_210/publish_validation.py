"""Wait quietly for remote full validation, then archive, refresh tables and push scoped results."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid


def refresh_snapshot(args):
    from archive_liulm_run import clean_value

    remote = """import datetime,json
from pathlib import Path
r=Path(REMOTE_ROOT)
e=r/'full/evidence'
jobs={}
for line in (e/'jobs.jsonl').read_text().splitlines():
 try:
  x=json.loads(line);jobs[(x['file'],x['name'])]=x
 except json.JSONDecodeError: pass
methods=[]; audits={}
for m in json.loads((r/'full/combo_npu_a5_selected.json').read_text()):
 key=(m['file'],m['name']);job=jobs.get(key,{})
 stem=m['file'].removesuffix('.py')+'.'+m['name']
 records=[]
 f=e/(stem+'.jsonl')
 if f.exists():
  for line in f.read_text().splitlines():
   try: records.append(json.loads(line))
   except json.JSONDecodeError: pass
 row={**m,'job_started':bool(job),'job_finished':job.get('event')=='finish',
      'returncode':job.get('returncode'),'records':records}
 if row['job_finished']:
  a=e/('audit.'+stem+'.jsonl')
  if a.exists(): audits[stem]=[json.loads(x) for x in a.read_text().splitlines()]
  if row['returncode'] not in (0,1): row['execution_log']=(e/(stem+'.log')).read_text()[-16000:]
 methods.append(row)
s={'captured_at':datetime.datetime.now().astimezone().isoformat(),
   'pipeline':json.loads((r/'pipeline_status.json').read_text()),
   'collection':json.loads((e/'collection.json').read_text()),'methods':methods,'audits':audits}
print('SNAPSHOT_JSON='+json.dumps(s))
""".replace("REMOTE_ROOT", repr(args.remote_root))
    output = subprocess.check_output(["ssh", args.host, "python -c " + shlex.quote(remote)],
                                     text=True, timeout=90)
    snapshot = json.loads(next(x[len("SNAPSHOT_JSON="):] for x in output.splitlines()
                               if x.startswith("SNAPSHOT_JSON=")))
    raw = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    (args.raw_root / "full_snapshot.json").write_text(raw)
    artifact = args.repo / "artifacts" / args.artifact
    destination = artifact / "execution/full_snapshot.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = json.dumps(clean_value(snapshot), ensure_ascii=False, indent=2) + "\n"
    destination.write_text(clean)
    receipt = {"original_sha256": hashlib.sha256(raw.encode()).hexdigest(),
               "archived_sha256": hashlib.sha256(clean.encode()).hexdigest(),
               "captured_at": snapshot["captured_at"], "artifact": "execution/full_snapshot.json"}
    (artifact / "execution/full_snapshot_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True,
                        help="Parent containing install/ and the three phase directories")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-liulm-0fd5a10-20261106")
    parser.add_argument("--interval", type=int, default=180, help="Quiet background check interval in seconds")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true",
                        help="Archive current reports without waiting or pushing")
    args = parser.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    if args.snapshot_only:
        snapshot = refresh_snapshot(args)
        print(json.dumps({"captured_at": snapshot["captured_at"],
                          "finished_methods": sum(m["job_finished"] for m in snapshot["methods"])}))
        return
    status_path = args.raw_root / "publisher_status.json"
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[x for x in sys.argv[1:] if x != "--detach"]]
        with (args.raw_root / "publisher.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.raw_root / "publisher.pid").write_text(str(child.pid) + "\n")
        print(json.dumps({"pid": child.pid, "status": str(status_path), "log": str(args.raw_root / "publisher.log")}))
        return

    # Keep one publisher per evidence directory, including across detached invocations.
    lock = (args.raw_root / "publisher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A publisher is already running for this raw directory")

    def status(state, **extra):
        status_path.write_text(json.dumps({"state": state, "time": time.time(), **extra}, indent=2) + "\n")

    def git(*command, env=None):
        return subprocess.check_output(["git", *command], cwd=args.repo, env=env)

    try:
        status("waiting_for_full_validation")
        probe = ("import json; from pathlib import Path; p=Path(" + repr(args.remote_root) + "); "
                 "s=json.loads((p/'pipeline_status.json').read_text()); "
                 "print('JOB_STATUS='+json.dumps({'ready':(p/'full/evidence/integrity_checks.json').exists(),"
                 "'state':s.get('state'),'error':s.get('error')}))")
        failures = 0
        execution_error = False
        while True:
            try:
                result = subprocess.check_output(["ssh", args.host, "python -c " + shlex.quote(probe)],
                                                 timeout=60, text=True)
                data = json.loads(next(line[len("JOB_STATUS="):] for line in result.splitlines()
                                       if line.startswith("JOB_STATUS=")))
                failures = 0
            except (subprocess.SubprocessError, StopIteration, json.JSONDecodeError):
                failures += 1
                if failures >= 10:
                    raise RuntimeError("Repeated SSH/status failures; check publisher.log")
                time.sleep(max(60, args.interval))
                continue
            if data["state"] == "error":
                refresh_snapshot(args)
                execution_error = True
                break
            if data["ready"]:
                break
            time.sleep(max(60, args.interval))
        if not execution_error:
            status("downloading_evidence")
            names = ["install", "failed_first/evidence", "with_fixes/evidence", "full/evidence",
                     "resource_authorization.json", "expected_raw_source_sha256.json", "expected_source_sha256.json",
                     "run_focused_then_wait.py", "run_full_after_docs.py", "docs_push_receipt.json",
                     "pipeline_status.json"]
            archive = ".automatic-publish-evidence.tar.gz"
            command = ("cd " + shlex.quote(args.remote_root) + " && tar -czf " + shlex.quote(archive) + " "
                       + " ".join(shlex.quote(x) for x in names) + " && sha256sum " + shlex.quote(archive))
            output = subprocess.check_output(["ssh", args.host, command], timeout=180, text=True)
            expected_hash = next(line.split()[0] for line in output.splitlines() if line.endswith("  " + archive))
            bundle = args.raw_root / archive
            subprocess.run(["scp", "-q", args.host + ":" + args.remote_root + "/" + archive, str(bundle)],
                           check=True, timeout=300)
            assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected_hash
            with tarfile.open(bundle) as tar:
                tar.extractall(args.raw_root, filter="data")
            scripts = Path(__file__).resolve().parent
            artifact = args.repo / "artifacts" / args.artifact
            subprocess.run([sys.executable, str(scripts / "archive_liulm_run.py"), "--raw-root", str(args.raw_root),
                            "--artifact-root", str(artifact)], check=True)
            subprocess.run([sys.executable, str(scripts / "update_liulm_results.py"), "--repo", str(args.repo),
                            "--artifact", args.artifact], check=True)
            summary = json.loads((artifact / "validation_summary.json").read_text())
            assert summary["full_complete"]
            for phase in ["failed_first", "with_fixes", "full"]:
                check = json.loads((artifact / phase / "integrity_checks.json").read_text())
                assert check["all_expected_instances_reported"] and check["installed_source_unchanged"]
            for entry in json.loads((artifact / "archive_manifest.json").read_text()):
                digest = hashlib.sha256((artifact / entry["artifact"]).read_bytes()).hexdigest()
                assert digest == entry["archived_sha256"]
        else:
            scripts = Path(__file__).resolve().parent
            subprocess.run([sys.executable, str(scripts / "update_liulm_results.py"), "--repo", str(args.repo),
                            "--artifact", args.artifact], check=True)
            artifact = args.repo / "artifacts" / args.artifact
            summary = json.loads((artifact / "validation_summary.json").read_text())
            receipt = json.loads((artifact / "execution/full_snapshot_manifest.json").read_text())
            expected_hash = receipt["original_sha256"]
            digest = hashlib.sha256((artifact / receipt["artifact"]).read_bytes()).hexdigest()
            assert digest == receipt["archived_sha256"]
        status("publishing_results", summary=summary, raw_archive_sha256=expected_hash)
        assert git("branch", "--show-current").decode().strip() == "main", "Expected main; no automatic branch switch"
        scopes = ["artifacts/" + args.artifact,
                  "docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv"]

        def unrelated_index():
            return [line for line in git("ls-files", "--stage", "-z").split(b"\0")
                    if line and not any(line.split(b"\t", 1)[1].decode() == p
                                        or line.split(b"\t", 1)[1].decode().startswith(p + "/") for p in scopes)]

        before = unrelated_index()
        temporary_index = args.raw_root / ("publish-index-" + uuid.uuid4().hex)
        env = dict(os.environ, GIT_INDEX_FILE=str(temporary_index))
        git("read-tree", "HEAD", env=env)
        git("add", "-f", "--", *scopes, env=env)
        git("diff", "--cached", "--check", "--", *scopes[1:], env=env)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=args.repo, env=env).returncode:
            git("commit", "-m", "docs: refresh latest liulm Combo validation results", env=env)
            git("reset", "HEAD", "--", *scopes)
        assert before == unrelated_index(), "Unrelated staging changed"
        commit = git("rev-parse", "HEAD").decode().strip()
        git("push", "origin", "HEAD:main")
        for attempt in range(3):
            try:
                remote = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(3)
        assert remote == commit
        status("published_with_execution_errors" if execution_error else "complete",
               commit=commit, summary=summary, raw_archive_sha256=expected_hash)
        print(json.dumps({"commit": commit, "phase_results": summary["phase_results"],
                          "unreviewed_full_failures": summary.get("unreviewed_full_failures", [])}), flush=True)
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
