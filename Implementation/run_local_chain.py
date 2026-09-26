"""Automated Local Inference Chain: FRANCE -> US

Monitors the currently running France inference process.
As soon as France finishes and verifies, it automatically launches US inference.
Runs completely headlessly overnight with zero user intervention required.
"""
import os
import sys
import time
from pathlib import Path
import psutil

IMPL_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
FRANCE_PID = 5812
LOG_DIR = IMPL_DIR / "runs"
US_LOG = LOG_DIR / "us_stream.log"
FRANCE_OUTPUT = IMPL_DIR / "runs" / "default" / "output" / "part_match_FRANCE.tsv"


def wait_for_france() -> bool:
    print(f"[{time.strftime('%X')}] Watcher started. Monitoring France inference (PID {FRANCE_PID})...", flush=True)
    while psutil.pid_exists(FRANCE_PID):
        try:
            p = psutil.Process(FRANCE_PID)
            if p.status() == psutil.STATUS_ZOMBIE:
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(15)

    print(f"[{time.strftime('%X')}] France inference process has exited.", flush=True)
    time.sleep(5)  # Allow file handles to flush

    if FRANCE_OUTPUT.is_file() and FRANCE_OUTPUT.stat().st_size > 1000:
        print(f"[{time.strftime('%X')}] Verified: France partition output exists ({FRANCE_OUTPUT.stat().st_size:,} bytes).", flush=True)
        return True
    else:
        print(f"[{time.strftime('%X')}] Warning: France output missing or empty at {FRANCE_OUTPUT}.", flush=True)
        return False


def run_us_inference() -> None:
    print(f"[{time.strftime('%X')}] Starting US streamed inference (663,106 queries)...", flush=True)
    cmd = [
        PYTHON_EXE,
        "-u",
        str(IMPL_DIR / "51_stream_country.py"),
        "--country", "US",
        "--shard-size", "25000",
    ]
    with US_LOG.open("w", encoding="utf-8") as f_out:
        p = psutil.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT, cwd=str(IMPL_DIR))
        print(f"[{time.strftime('%X')}] US inference launched with PID {p.pid}. Logging to {US_LOG}", flush=True)
        p.wait()

    print(f"[{time.strftime('%X')}] US inference completed with exit code {p.returncode}.", flush=True)


if __name__ == "__main__":
    import subprocess
    france_ok = wait_for_france()
    if france_ok:
        run_us_inference()
    else:
        print(f"[{time.strftime('%X')}] Exiting: France did not produce valid output.", flush=True)
        sys.exit(1)
