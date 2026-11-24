"""Record shared-card utilization while an existing NPU matrix is running."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def sample(device):
    try:
        result = subprocess.run(["npu-smi", "info", "-i", device, "-t", "usages"],
                                text=True, capture_output=True, timeout=20)
        fields = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                fields[name.strip()] = value.strip()
        return dict(device=device, returncode=result.returncode, fields=fields)
    except subprocess.TimeoutExpired:
        return dict(device=device, status="QUERY_TIMEOUT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    args = parser.parse_args()
    devices = args.devices.split(",")
    output = args.run / "load_samples.json"
    records = json.loads(output.read_text()) if output.exists() else []
    while True:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            record = dict(date=datetime.now(timezone.utc).isoformat(), devices=list(pool.map(sample, devices)))
        records.append(record)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(output)
        if (args.run / "results.json").exists():
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
