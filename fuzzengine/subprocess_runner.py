#!/usr/bin/env python3
"""Subprocess supervision helpers for fuzz/replay jobs.

Native crashes in Unicorn or extension modules can terminate the Python
process before runtime hooks run.  The only reliable generic signal is the
child process return code, so fuzzing campaigns should execute each expensive
seed/replay in a supervised subprocess and feed the return code to
``CrashDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Dict, Mapping, Optional, Sequence

try:
    from .crash_detector import CrashDetector, CrashDetectorConfig, CrashReport
except ImportError:  # Allow ``python3 fuzzengine/subprocess_runner.py``.
    from crash_detector import CrashDetector, CrashDetectorConfig, CrashReport

try:
    from lsgemu.artifact_io import atomic_json_dump
except ImportError:  # pragma: no cover - supports running fuzzengine standalone.
    atomic_json_dump = None


@dataclass
class SupervisedRunResult:
    command: Sequence[str]
    returncode: int
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    crash_report: CrashReport

    def to_dict(self) -> Dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "crash_report": self.crash_report.to_dict(),
        }


def run_supervised(
    command: Sequence[str],
    *,
    timeout: Optional[float] = None,
    detector: Optional[CrashDetector] = None,
    firmware: Optional[str] = None,
    input_id: Optional[str] = None,
    seed_path: Optional[object] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[object] = None,
    tail_bytes: int = 65536,
) -> SupervisedRunResult:
    """Run one replay/fuzz command in a child process and classify its exit."""
    detector = detector or CrashDetector()
    timed_out = False
    run_env = None
    if env is not None:
        run_env = os.environ.copy()
        run_env.update({str(key): str(value) for key, value in env.items()})
    process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=run_env,
            cwd=str(cwd) if cwd is not None else None,
            start_new_session=(os.name != "nt"),
        )
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = int(process.returncode)
        stdout_tail = stdout[-tail_bytes:].decode("utf-8", errors="replace")
        stderr_tail = stderr[-tail_bytes:].decode("utf-8", errors="replace")
        stop_reason = "timeout_reached" if timed_out else ("completed" if returncode == 0 else "process_exit")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        partial_stdout = exc.stdout or b""
        partial_stderr = exc.stderr or b""
        if process is not None:
            _terminate_process(process)
            try:
                stdout, stderr = process.communicate(timeout=2.0)
            except Exception:
                stdout, stderr = b"", b""
        else:
            stdout, stderr = b"", b""
        stdout = stdout or partial_stdout
        stderr = stderr or partial_stderr
        returncode = int(process.returncode) if process is not None and process.returncode is not None else -999
        stdout_tail = _decode_tail(stdout, tail_bytes)
        stderr_tail = _decode_tail(stderr, tail_bytes)
        stop_reason = "timeout_reached"

    crash_report = detector.detect(
        {
            "source_layer": "process",
            "stop_reason": stop_reason,
            "process_returncode": returncode,
            "timed_out": timed_out,
        },
        returncode=returncode,
        stdout=stdout_tail,
        stderr=stderr_tail,
        firmware=firmware,
        input_id=input_id,
        seed_path=seed_path,
        metadata={
            "source_layer": "process",
            "command": list(command),
            "timed_out": timed_out,
            "cwd": str(cwd) if cwd is not None else None,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    )
    return SupervisedRunResult(
        command=list(command),
        returncode=returncode,
        timed_out=timed_out,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        crash_report=crash_report,
    )


def _decode_tail(value: object, limit: int) -> str:
    if isinstance(value, bytes):
        return value[-limit:].decode("utf-8", errors="replace")
    return str(value or "")[-limit:]


def _terminate_process(process: subprocess.Popen) -> None:
    """Terminate the child process group so timed-out emulators do not linger."""
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows is not the deployment target.
            process.terminate()
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=2.0)
        return
    except Exception:
        pass
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except Exception:
            pass
def _load_config(path: Optional[str]) -> CrashDetectorConfig:
    if not path:
        return CrashDetectorConfig()
    return CrashDetectorConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one fuzz/replay command under crash supervision")
    parser.add_argument("--config", help="crash detector config JSON")
    parser.add_argument("--timeout", type=float, help="wall-clock timeout in seconds")
    parser.add_argument("--firmware", help="firmware label/path for the crash report")
    parser.add_argument("--input-id", help="seed/input identifier")
    parser.add_argument("--seed-path", help="optional seed path for sha256 attribution")
    parser.add_argument("--out", help="optional output JSON path")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")

    result = run_supervised(
        command,
        timeout=args.timeout,
        detector=CrashDetector(_load_config(args.config)),
        firmware=args.firmware,
        input_id=args.input_id,
        seed_path=args.seed_path,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        if atomic_json_dump is not None:
            atomic_json_dump(payload, Path(args.out), indent=2, ensure_ascii=False)
        else:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
