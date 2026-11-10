"""Finish active pytest jobs, run other files first, then finish foreach."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_plain_pytest import fingerprint, read_reports, save, write_table


TERMINAL = {"PASS", "FAIL", "SKIP", "ERROR"}


def process_state(pid):
    # Fields after comm start at field 3.  Linux keeps the wait status (field
    # 52) while an exited child is a zombie and its original parent is stopped.
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rindex(")") + 2:].split()
    return fields[0], fields[19], int(fields[49])


def key(job):
    return job["file"], job["class"], job["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / "results"
    handoff = json.loads((root / "handoff.json").read_text())
    state = json.loads((output / "status.json").read_text())
    jobs = json.loads((output / "jobs.json").read_text())
    by_key = {key(job): job for job in jobs}
    before = json.loads((output / "installed_sha256.json").read_text())
    assert fingerprint([Path(path) for path in before]) == before, "Installed sources changed"
    previous_pid = handoff["previous_runner_pid"]
    command = Path(f"/proc/{previous_pid}/cmdline").read_bytes()
    assert b"run_plain_pytest.py" in command
    assert Path(f"/proc/{previous_pid}/cwd").resolve() == root
    assert process_state(previous_pid)[0] in ("T", "t"), "Original dispatcher must be stopped"
    adopted = {}
    for index, pid in handoff["active_processes"].items():
        job = next(job for job in jobs if job["index"] == int(index))
        assert job["status"] == "RUNNING" and os.getpgid(pid) == pid
        adopted[int(index)] = {"pid": pid, "start_ticks": process_state(pid)[1], "job": job}
    state.update(stage="testing", execution_order="non_foreach_first", devices=[1, 2, 3, 6],
                 previous_runner_pid=previous_pid, controller_pid=os.getpid(),
                 phase="non_foreach_first", non_foreach_devices=[2, 3], foreach_devices=[1, 2, 3, 6])
    state["phase_runs"] = {
        "non-foreach-first": {"devices": [2, 3], "cache_root": str(root / "cache-non-foreach-first")},
        "foreach-last": {"devices": [1, 2, 3, 6], "cache_root": str(root / "cache-foreach-last")},
    }
    save(output / "execution-order.json", handoff)
    old_parent_stopped = True
    copied = set()

    def publish():
        save(output / "jobs.json", jobs)
        write_table(output, jobs)
        state["completed"] = sum(job["status"] in TERMINAL for job in jobs)
        state["method_results"] = dict(Counter(job["status"] for job in jobs))
        state["instance_results"] = dict(Counter(
            report["status"] for job in jobs for report in job.get("reports", {}).values()
        ))
        save(output / "status.json", state)

    def finish_active():
        nonlocal old_parent_stopped
        for index, item in list(adopted.items()):
            pid, job = item["pid"], item["job"]
            proc_status, ticks, raw_exit = process_state(pid)
            assert ticks == item["start_ticks"], "Active pytest PID changed"
            if proc_status != "Z":
                if time.time() > job["started"] + args.timeout:
                    if "terminated" not in item:
                        os.killpg(pid, signal.SIGTERM)
                        item["terminated"] = time.time()
                    elif time.time() - item["terminated"] > 15:
                        os.killpg(pid, signal.SIGKILL)
                continue
            code = 124 if "terminated" in item else os.waitstatus_to_exitcode(raw_exit)
            stem = f"{index:02d}-{job['name']}"
            reports = read_reports(output / (stem + ".xml"), job["nodes"])
            counts = Counter(report["status"] for report in reports.values())
            status = "ERROR" if code not in (0, 1) or len(reports) != len(job["nodes"]) else (
                "FAIL" if counts["FAIL"] or counts["ERROR"] else "SKIP" if counts["SKIP"] else "PASS"
            )
            job.update(status=status, returncode=code, reports=reports, finished=time.time(),
                       seconds=round(time.time() - job["started"], 2), adopted_pid=pid)
            del adopted[index]
            print("ACTIVE_FINISHED", job["name"], status, flush=True)
        if not adopted and old_parent_stopped:
            assert b"run_plain_pytest.py" in Path(f"/proc/{previous_pid}/cmdline").read_bytes()
            # Its children have exited and their real wait statuses were saved.
            # Never resume the old queues, which still contain the original order.
            os.kill(previous_pid, signal.SIGKILL)
            old_parent_stopped = False
            print("OLD_DISPATCHER_RETIRED", previous_pid, flush=True)

    def merge_phase(phase):
        directory = root / "phases" / phase / "results"
        status_path = directory / "status.json"
        if not status_path.exists():
            return {"stage": "starting"}
        phase_state = json.loads(status_path.read_text())
        save(output / (phase + "-status.json"), phase_state)
        for name in ("command.json", "manifest.json"):
            source = directory.parent / name
            if source.exists():
                shutil.copy2(source, output / (phase + "-" + name))
        jobs_path = directory / "jobs.json"
        if jobs_path.exists():
            for incoming in json.loads(jobs_path.read_text()):
                job = by_key[key(incoming)]
                assert job["nodes"] == incoming["nodes"], "Changed parameterized instances"
                index = job["index"]
                job.update(incoming, index=index, phase_results=str(directory))
                source_stem = f"{incoming['index']:02d}-{incoming['name']}"
                target_stem = f"{index:02d}-{job['name']}"
                token = phase, index
                if token not in copied:
                    for extension in (".log", ".xml"):
                        source = directory / (source_stem + extension)
                        if source.exists() and (extension == ".log" or job["status"] in TERMINAL):
                            shutil.copy2(source, output / (target_stem + extension))
                    if job["status"] in TERMINAL:
                        copied.add(token)
        if phase_state["stage"] == "complete":
            assert phase_state["installed_source_unchanged"]
            assert json.loads((directory / "installed_sha256.json").read_text()) == before
        return phase_state

    def start_foreach():
        directory = root / "phases/foreach-last"
        directory.mkdir(parents=True, exist_ok=False)
        pending = [job for job in jobs if job["file"] == "test_foreach.py" and job["status"] == "PENDING"]
        selected = [{field: job[field] for field in ("file", "class", "name")} for job in pending]
        save(directory / "manifest.json", selected)
        if not selected:
            return False
        command = [sys.executable, "-u", str(root / "run_plain_pytest.py"),
                   "--test-root", state["test_root"], "--manifest", str(directory / "manifest.json"),
                   "--output", str(directory / "results"), "--devices", "1,2,3,6",
                   "--test-commit", state["test_commit"], "--npu-fix-commit", state["npu_fix_commit"],
                   "--foreach-functional", "--cache-root", str(root / "cache-foreach-last")]
        with (directory / "run.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=root, env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="1"),
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (directory / "run.pid").write_text(str(process.pid) + "\n")
        save(directory / "command.json", command)
        print("FOREACH_LAST_STARTED", len(selected), "methods", process.pid, flush=True)
        return True

    try:
        publish()
        while True:
            finish_active()
            phase_state = merge_phase("non-foreach-first")
            state["phase_state"] = phase_state["stage"]
            publish()
            if not adopted:
                if phase_state["stage"] == "error":
                    raise RuntimeError("Non-foreach phase failed: " + phase_state.get("error", "unknown"))
                if phase_state["stage"] == "complete":
                    break
            time.sleep(args.poll_seconds)
        assert all(job["status"] in TERMINAL for job in jobs if job["file"] != "test_foreach.py")
        state["phase"] = "foreach_last"
        if start_foreach():
            while True:
                phase_state = merge_phase("foreach-last")
                state["phase_state"] = phase_state["stage"]
                publish()
                if phase_state["stage"] == "error":
                    raise RuntimeError("Foreach phase failed: " + phase_state.get("error", "unknown"))
                if phase_state["stage"] == "complete":
                    break
                time.sleep(args.poll_seconds)
        assert all(job["status"] in TERMINAL for job in jobs)
        assert fingerprint([Path(path) for path in before]) == before, "Installed sources changed"
        state.update(stage="complete", finished=time.time(), installed_source_unchanged=True)
        publish()
        print("COMPLETE", json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(stage="error", error=repr(error), finished=time.time())
        publish()
        raise


if __name__ == "__main__":
    main()
