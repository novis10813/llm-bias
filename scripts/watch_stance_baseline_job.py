#!/usr/bin/env python3
"""Read-only lab observer; never launches or stops jobs. Linux publication is no-replace."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

_MARKER = "__LAB_SCRIPT_BEGIN__"

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o",
               "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


def simple_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("expected a simple host alias or job ID")
    return value


def remote_path(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", value) or any(
        part in (".", "..") for part in value.split("/")
    ) or value.startswith("-"):
        raise argparse.ArgumentTypeError("expected a safe relative remote-home path")
    return value


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def retries(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected at least one attempt")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True, type=simple_name)
    p.add_argument("--job-id", required=True, type=simple_name)
    p.add_argument("--remote-output", required=True, type=remote_path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--remote-lab-dir", default=".lab", type=remote_path)
    p.add_argument("--connection-retries", default=3, type=retries,
                   help="maximum consecutive failed observation attempts")
    p.add_argument("--retry-delay", default=5.0, type=positive)
    p.add_argument("--check-interval", default=30.0, type=positive)
    p.add_argument("--observation-window", default=300.0, type=positive)
    p.add_argument("--retrieval-timeout", default=600.0, type=positive)
    return p


def observation_script(args: argparse.Namespace) -> str:
    # Canonical lab order: exact session, then exit file, then missing.
    directory = '"$HOME"/' + shlex.quote(f"{args.remote_lab_dir}/jobs/{args.job_id}")
    return f'''job={shlex.quote(args.job_id)}
dir={directory}
SECONDS=0
while :; do
  tmux_error=$(tmux has-session -t "=$job" 2>&1)
  tmux_code=$?
  if (( tmux_code == 0 )); then
    if (( SECONDS >= {math.ceil(args.observation_window)} )); then echo RUNNING; exit 0; fi
  elif [ -f "$dir/exit_code" ]; then
    printf 'EXIT %s\\n' "$(tr -d '[:space:]' < "$dir/exit_code")"; exit 0
  else
    if (( tmux_code == 1 )); then
      case "$tmux_error" in
        ""|"no server running on "*|"error connecting to "*" (No such file or directory)"|"can't find session: "*)
          echo MISSING; exit 0 ;;
      esac
    fi
    printf 'tmux observation failed (exit %s): %s\\n' "$tmux_code" "$tmux_error" >&2
    exit 2
  fi
  sleep {min(args.check_interval, args.observation_window)}
done
'''


def publish(staging: Path, destination: Path) -> None:
    """Atomic Linux rename with RENAME_NOREPLACE, including empty destinations."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def watch(args: argparse.Namespace, *, run=subprocess.run, sleep=time.sleep) -> tuple[dict, int]:
    report = dict(job_id=args.job_id, host=args.host, remote_status="unknown",
                  remote_exit_code=None, monitor_error=None, retrieval_status="not_attempted",
                  retrieval_error=None, final_output=None, staging_location=None,
                  artifact_validation_status="not_available", research_eligible=False)
    destination = args.output_dir.absolute()
    report_path = args.report.absolute()
    try:
        if os.path.lexists(destination) or os.path.lexists(report_path):
            raise FileExistsError("output directory or report already exists")
        if report_path == destination or destination in report_path.parents:
            raise ValueError("report must be outside the fetched directory")
        failures = 0
        command = ["ssh", *SSH_OPTIONS, args.host, "bash -l -s"]
        script = f"echo {_MARKER}\n{observation_script(args)}"
        while True:
            try:
                result = run(command, input=script, capture_output=True, text=True,
                             timeout=math.ceil(args.observation_window) + min(
                                 args.check_interval, args.observation_window) + 90)
                if result.returncode:
                    raise RuntimeError(f"ssh exit {result.returncode}: {result.stderr.strip()}")
                _, found, output = result.stdout.partition(_MARKER + "\n")
                if not found:
                    raise ValueError("missing observation script marker")
                status = output.strip()
                if status not in ("RUNNING", "MISSING") and not re.fullmatch(r"EXIT -?[0-9]+", status):
                    raise ValueError(f"malformed status: {status!r}")
            except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
                failures += 1
                if failures >= args.connection_retries:
                    raise RuntimeError(f"observation attempts exhausted: {exc}") from exc
                sleep(args.retry_delay)
                continue
            failures = 0
            if status == "RUNNING":
                continue
            if status == "MISSING":
                report["remote_status"] = "lost"
            else:
                code = int(status.split()[1])
                report.update(remote_exit_code=code, remote_status="succeeded" if code == 0 else "failed")
            break
    except Exception as exc:
        report["monitor_error"] = str(exc)

    if report["remote_status"] != "unknown":
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
            report["staging_location"] = str(staging)
            # Fetch an ordinary basename into a unique container: legacy scp
            # rejects '/.' with its default filename checks enabled.
            fetched = staging / Path(args.remote_output).name
            source = f"{args.host}:" + shlex.quote(args.remote_output)
            result = run(["scp", *SSH_OPTIONS, "-r", source, str(staging)],
                         capture_output=True, text=True, timeout=args.retrieval_timeout)
            if result.returncode:
                raise RuntimeError(f"scp exit {result.returncode}: {result.stderr.strip()}")
            if fetched.is_symlink() or not fetched.is_dir():
                raise RuntimeError("scp did not fetch the expected non-symlink directory")
            publish(fetched, destination)
            staging.rmdir()
            report.update(retrieval_status="succeeded", final_output=str(destination),
                          staging_location=None, artifact_validation_status="pending")
        except Exception as exc:
            report.update(retrieval_status="failed", retrieval_error=str(exc))

    success = report["remote_status"] == "succeeded" and report["retrieval_status"] == "succeeded" and not report["monitor_error"]
    try:
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True)
            stream.write("\n")
    except Exception as exc:
        report["report_error"] = str(exc)
        success = False
    return report, 0 if success else 1


def main() -> int:
    report, code = watch(parser().parse_args())
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
