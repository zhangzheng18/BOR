#!/usr/bin/env python3
"""Run the three contribution-level LSGEmu ablations on CNC, Gateway, and PLC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRCV4_ROOT = PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.deployment_config import apply_config_from_argv, configured_path

apply_config_from_argv()

ELFMULTIFUZZ_ROOT = configured_path("ELFMULTIFUZZ_ROOT", PROJECT_ROOT / "datasets" / "elfmultifuzz")
RUN_OUTPUT_ROOT = configured_path("LSGEMU_RUN_OUTPUT_DIR", PROJECT_ROOT / ".lsgemu_runs")

TARGETS = {
    "CNC": ELFMULTIFUZZ_ROOT / "P2IM" / "CNC" / "CNC.elf",
    "Gateway": ELFMULTIFUZZ_ROOT / "P2IM" / "Gateway" / "Gateway.elf",
    "PLC": ELFMULTIFUZZ_ROOT / "P2IM" / "PLC" / "PLC.elf",
}

ABLATIONS = {
    "no_semantic_obligation_v2": Path("experiments/lsgemu_no_semantic_obligation.py"),
    "no_scoped_replay_v2": Path("experiments/lsgemu_no_scoped_replay.py"),
    "no_context_event_replay": Path("experiments/lsgemu_no_context_event_replay.py"),
}


def latest_report(output_dir: Path, firmware: Path) -> Path | None:
    candidates = sorted(output_dir.glob(f"{firmware.stem}*_interleaved_report.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted(output_dir.glob(f"{firmware.stem}*_lsgemu_result.json"))
    if candidates:
        return candidates[-1]
    return None


def report_row(ablation: str, target: str, report_path: Path | None, returncode: int, elapsed: float) -> dict:
    row = {
        "ablation": ablation,
        "target": target,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "report": str(report_path) if report_path else "",
    }
    if report_path and report_path.exists():
        with report_path.open() as f:
            report = json.load(f)
        row.update({
            "covered_bbs": report.get("covered_bbs"),
            "total_bbs": report.get("total_bbs"),
            "coverage_rate": report.get("coverage_rate"),
            "valid_covered_bbs": report.get("valid_covered_bbs"),
            "valid_total_bbs": report.get("valid_total_bbs"),
            "valid_coverage_rate": report.get("valid_coverage_rate"),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument("--time", type=float, default=60.0)
    parser.add_argument("--output-root", default=str(RUN_OUTPUT_ROOT / "p2im_contribution_ablation"))
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=[])
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), action="append", default=[])
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    targets = args.target or list(TARGETS)
    ablations = args.ablation or list(ABLATIONS)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for ablation in ablations:
        script = SRCV4_ROOT / ABLATIONS[ablation]
        for target in targets:
            firmware = TARGETS[target]
            output_dir = output_root / ablation / target
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(script),
                str(firmware),
                "--time",
                str(args.time),
                "--output-dir",
                str(output_dir),
                "--log-level",
                str(args.log_level),
                *args.extra_args,
            ]
            if args.config:
                cmd.extend(["--config", str(args.config)])
            print(" ".join(cmd), flush=True)
            start = time.time()
            if args.dry_run:
                returncode = 0
            else:
                with (output_dir / "run.log").open("w") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(SRCV4_ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    returncode = proc.returncode
            elapsed = time.time() - start
            summary_rows.append(
                report_row(ablation, target, latest_report(output_dir, firmware), returncode, elapsed)
            )
            with (output_root / "summary.json").open("w") as f:
                json.dump(summary_rows, f, indent=2)

    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
