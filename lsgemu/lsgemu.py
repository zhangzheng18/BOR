#!/usr/bin/env python3
"""
Unified LSGEmu entrypoint backed by the local HistoricalRunner stack.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    project_root_text = str(PROJECT_ROOT)
    if project_root_text in sys.path:
        sys.path.remove(project_root_text)
    sys.path.insert(0, project_root_text)

    from lsgemu.deployment_config import apply_config_from_argv, configured_path

    apply_config_from_argv()

    # Apply the run profile BEFORE importing heavy modules: runner_common and
    # others freeze LSGEMU_* path globals at import time, so a profile applied
    # only in main() would silently void its path-type entries.
    from lsgemu.run_profile import apply_run_profile_from_argv

    apply_run_profile_from_argv()

    from lsgemu.historical_runner import HistoricalRunner, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from lsgemu.firmware_input import (
        BinConversionSpec,
        DEFAULT_BINTOELF_CACHE,
        DEFAULT_BINTOELF_ROOT,
        FirmwareInputError,
        ResolvedFirmwareInput,
        resolve_firmware_input,
    )
    from lsgemu.run_profile import apply_run_profile_environment, load_run_profile
    from lsgemu.run_paths import unique_firmware_run_directory
    from lsgemu.analysis.snapshot_memory import configure_snapshot_storage_for_run
else:
    from .deployment_config import apply_config_from_argv, configured_path

    apply_config_from_argv()

    # Apply the run profile BEFORE importing heavy modules (see comment in the
    # package-mode import branch above).
    from .run_profile import apply_run_profile_from_argv

    apply_run_profile_from_argv()

    from .historical_runner import HistoricalRunner, PROJECT_ROOT as RUNNER_PROJECT_ROOT
    from .firmware_input import (
        BinConversionSpec,
        DEFAULT_BINTOELF_CACHE,
        DEFAULT_BINTOELF_ROOT,
        FirmwareInputError,
        ResolvedFirmwareInput,
        resolve_firmware_input,
    )
    from .run_profile import apply_run_profile_environment, load_run_profile
    from .run_paths import unique_firmware_run_directory
    from .analysis.snapshot_memory import configure_snapshot_storage_for_run


def _load_llm_config(project_root: Path) -> Optional[dict]:
    llm_config_path = configured_path("LSGEMU_LLM_CONFIG", project_root / "LLM.yaml")
    if not llm_config_path.exists():
        return None

    try:
        with llm_config_path.open() as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _llm_config_path(project_root: Path) -> Optional[Path]:
    path = configured_path("LSGEMU_LLM_CONFIG", project_root / "LLM.yaml")
    return path if path.exists() else None


def _parse_int(value: str) -> int:
    return int(str(value), 0)


def _resolve_cli_firmware_input(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> ResolvedFirmwareInput:
    conversion_values = (
        args.bin_arch,
        args.bin_bits,
        args.bin_endian,
        args.bin_base,
        args.bin_entry,
        args.bin_entry_offset,
        args.bin_thumb,
        args.bin_paddr,
        args.bin_permissions,
        args.bin_align,
        args.bin_bss_size,
    )
    conversion_requested = args.firmware_format == "bin" or any(
        value is not None for value in conversion_values
    )
    bin_spec = None
    if conversion_requested:
        required = {
            "--bin-arch": args.bin_arch,
            "--bin-bits": args.bin_bits,
            "--bin-endian": args.bin_endian,
            "--bin-base": args.bin_base,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "BintoElf conversion requires explicit layout options: "
                + ", ".join(missing)
            )
        if args.entry_point is not None or args.load_base is not None:
            parser.error(
                "BintoElf conversion cannot be combined with --entry-point/--load-base; "
                "use --bin-entry/--bin-base so the generated ELF and report agree"
            )
        if args.bin_entry_offset is not None and int(args.bin_entry_offset) < 0:
            parser.error("--bin-entry-offset cannot be negative")
        bin_spec = BinConversionSpec(
            arch=str(args.bin_arch),
            bits=int(args.bin_bits),
            endian=str(args.bin_endian),
            base_address=int(args.bin_base),
            entry=int(args.bin_entry) if args.bin_entry is not None else None,
            entry_offset=(
                int(args.bin_entry_offset)
                if args.bin_entry_offset is not None
                else None
            ),
            thumb=args.bin_thumb,
            physical_address=(
                int(args.bin_paddr) if args.bin_paddr is not None else None
            ),
            permissions=str(args.bin_permissions or "r-x"),
            segment_align=(
                int(args.bin_align) if args.bin_align is not None else 0x1000
            ),
            bss_size=(
                int(args.bin_bss_size) if args.bin_bss_size is not None else 0
            ),
        )
    try:
        return resolve_firmware_input(
            args.firmware,
            input_format=args.firmware_format,
            bin_spec=bin_spec,
            bintoelf_root=args.bintoelf_root,
            cache_root=args.bin_cache_dir,
        )
    except FirmwareInputError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error should not return")


class LSGEmu:
    def __init__(
        self,
        firmware_path,
        execution_time_minutes=20,
        baseline_instructions=500000,
        output_dir: Optional[str] = None,
        entry_point: Optional[int] = None,
        load_base: Optional[int] = None,
        firmware_input_provenance: Optional[dict] = None,
        execution_thumb_override: Optional[bool] = None,
    ):
        self.firmware_path = Path(firmware_path).resolve()
        self.firmware_input_provenance = dict(firmware_input_provenance or {})
        source_record = dict(self.firmware_input_provenance.get("source") or {})
        self.source_firmware_path = Path(
            str(source_record.get("path") or self.firmware_path)
        ).resolve()
        self.artifact_stem = self.source_firmware_path.stem
        self.execution_time_limit = execution_time_minutes * 60
        self.baseline_instructions = baseline_instructions
        self.entry_point_override = entry_point
        self.load_base_override = load_base
        self.start_time = None
        self.project_root = RUNNER_PROJECT_ROOT
        default_output_root = configured_path(
            "LSGEMU_RUN_OUTPUT_DIR",
            configured_path("LSGEMU_SOURCE_ROOT", self.project_root) / ".lsgemu_runs",
        )
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            # No explicit --output-dir: derive a per-run unique directory so
            # concurrent/sequential runs of the same firmware never overwrite
            # each other's reports or constraint artifacts.
            self.output_dir = unique_firmware_run_directory(
                default_output_root,
                self.source_firmware_path,
                namespace="lsgemu",
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configure_snapshot_storage_for_run(
            self.output_dir,
            run_id=os.environ.get("LSGEMU_ATTEMPT_ID") or None,
        )
        self.llm_config = _load_llm_config(self.project_root)
        self.llm_config_path = _llm_config_path(self.project_root)
        self.constraint_file = self._artifact_path("_lsgemu_constraints.json")
        llm_model = None
        if self.llm_config:
            llm_model = self.llm_config.get("llm", {}).get("model")

        print(f"\n{'=' * 80}")
        print(f"{'LSGEmu':^80}")
        print(f"{'=' * 80}\n")
        print(f"固件: {self.source_firmware_path.name}")
        if self.source_firmware_path != self.firmware_path:
            print(f"分析ELF: {self.firmware_path}")
        print(f"执行时间: {execution_time_minutes}分钟")
        print(f"基准最大指令数: {baseline_instructions}")
        print(f"约束文件: {self.constraint_file}")
        if llm_model:
            print(f"LLM: {llm_model}")

        print("\n[1/4] 静态分析...")
        self.runner = HistoricalRunner(
            self.firmware_path,
            max_snapshots=5,
            use_llm=self.llm_config is not None,
            llm_config=self.llm_config,
            llm_config_path=self.llm_config_path,
            constraint_file=str(self.constraint_file),
            entry_point_override=self.entry_point_override,
            load_base_override=self.load_base_override,
            firmware_input_provenance=self.firmware_input_provenance or None,
            execution_thumb_override=execution_thumb_override,
        )
        self.result = self.runner.result
        self.static_bbs = self.runner.static_bbs
        self.static_bb_set = self.runner.static_bb_set
        self.emulator = self.runner.emulator
        self.mmio_handler = self.runner.mmio_handler
        self.llm_guide = self.runner.llm_guide
        self.register_tracer = self.runner.register_tracer
        self.global_coverage = self.runner.global_coverage
        self.entry_point = self.runner.prepared.entry_point
        if self.entry_point_override is not None:
            self.entry_point = int(self.entry_point_override) & ~1
            self.runner.prepared.result.arch_info.entry_point = self.entry_point
            self.emulator.arch_info.entry_point = self.entry_point
            self.emulator.entry_point = self.entry_point
            print(f"      入口覆盖: 0x{self.entry_point:08x}")
        if self.load_base_override is not None:
            print(f"      运行基址: 0x{int(self.load_base_override) & 0xFFFFFFFF:08x}")

        print(f"      总BB: {self.runner.prepared.total_bbs}")
        if self.runner.prepared.valid_bb_set:
            print(f"      Valid BB: {self.runner.prepared.valid_total_bbs}")
            print(f"      Ghidra粗粒度BB: {self.runner.prepared.ghidra_total_bbs}")
        print(f"      MMIO访问: {len(self.result.mmio_accesses)}")
        self.start_time = time.time()

    def _time_remaining(self):
        return self.execution_time_limit - (time.time() - self.start_time)

    def _artifact_path(self, suffix: str) -> Path:
        return self.output_dir / f"{self.artifact_stem}{suffix}"

    def run_baseline(self):
        """基准运行（IntelligentEmulator已集成死锁检测和LLM求解）"""
        print("\n[2/4] 基准运行（智能循环检测+LLM求解）...")
        log_file = self._artifact_path("_execution.log")
        remaining = max(0.0, self._time_remaining())
        configured_timeout = os.environ.get("LSGEMU_BASELINE_TIMEOUT_SECONDS")
        if configured_timeout:
            try:
                baseline_timeout = max(1.0, float(configured_timeout))
            except ValueError:
                baseline_timeout = None
        else:
            reserve = min(max(30.0, self.execution_time_limit * 0.15), 300.0)
            baseline_timeout = max(1.0, remaining - reserve) if remaining > reserve else max(1.0, remaining)
        baseline = self.runner.run_baseline(
            max_instructions=self.baseline_instructions,
            trace_output_path=log_file,
            timeout_seconds=baseline_timeout,
        )
        metadata = self.runner.phase_metadata["baseline"]
        print(f"      覆盖: {len(baseline)} BBs")
        print(f"      分支快照: {metadata['branch_snapshots']}")
        print(f"      日志: {log_file}")

    def run_isr(self):
        print("\n[3/4] ISR探索...")
        covered = self.runner.context_recovery.cold_isr(
            max_instructions=10000,
            count_coverage=True,
        )
        print(f"      覆盖: {len(covered)} BBs")
        print(f"      ISR数量: {self.runner.phase_metadata['isr']['irq_count']}")

    def run_branches(self):
        print("\n[4/4] 分支探索...")
        branches = self.runner.conditional_branches()
        unexplored = [bb for bb in branches if bb.start_addr not in self.global_coverage]
        print(f"      总分支: {len(branches)}, 当前未覆盖: {len(unexplored)}")

        reservoir_budget = min(max(int(self._time_remaining() // 2), 0), 600)
        if reservoir_budget > 0:
            reservoir_covered = self.runner.run_reservoir_branch_exploration(
                time_limit_seconds=reservoir_budget,
                max_tasks=1000,
            )
            reservoir_meta = self.runner.phase_metadata["branch_reservoir"]
            print(f"      蓄水池路径: {reservoir_meta['forced_directions']} 次, 新增 {reservoir_meta['new_bbs']} BBs")
            print(f"      蓄水池覆盖: {len(reservoir_covered)} BBs")

        remaining = max(0, int(self._time_remaining()) - 10)
        if remaining <= 0:
            print("      时间预算不足，跳过 MMIO 分支重放")
            return

        branch_covered = self.runner.run_mmio_branch_exploration(
            time_limit_seconds=remaining,
            max_branches=len(unexplored) if unexplored else None,
        )
        branch_meta = self.runner.phase_metadata["branch_mmio"]
        print(f"      方向回放: {branch_meta['directions_replayed']} 次")
        print(f"      应用约束: {branch_meta['constraints_applied']} 条")
        print(f"      新增覆盖: {branch_meta['new_bbs']} BBs")
        print(f"      分支覆盖: {len(branch_covered)} BBs")

        naturalization_remaining = max(0, int(self._time_remaining()))
        naturalization_summary = self.runner.path_naturalization_ledger.summary()
        if (
            naturalization_remaining > 0
            and int(naturalization_summary.get("obligations", 0) or 0) > 0
        ):
            naturalized = self.runner.run_path_naturalization(
                time_limit_seconds=min(60, naturalization_remaining),
                max_paths=32,
                max_attempts_per_path=8,
                max_fact_candidates_per_edge=4,
                max_values_per_source=4,
                max_compound_candidates=4,
                replay_instructions=120000,
                replay_timeout=1000000,
            )
            naturalization_meta = self.runner.phase_metadata.get("path_naturalization", {})
            print(
                "      无强制路径验证: "
                f"晋升 {naturalization_meta.get('promoted_paths', 0)} 条, "
                f"自然覆盖 {len(naturalized)} BBs"
            )

    def run(self):
        self.run_baseline()
        self.run_isr()
        self.run_branches()

        total_time = time.time() - self.start_time
        entry_covered = (self.entry_point & ~1) in self.global_coverage
        output = self._artifact_path("_lsgemu_result.json")
        report = self.runner.save_report(
            output,
            execution_time_seconds=total_time,
            extra={
                "entry_point_covered": entry_covered,
                "runner": "HistoricalRunner",
                "constraint_file": str(self.constraint_file),
            },
        )

        print(f"\n{'=' * 80}")
        print("结果")
        print(f"{'=' * 80}\n")
        print(f"覆盖: {report['covered_bbs']}/{report['total_bbs']} BBs ({report['coverage_rate']:.2f}%)")
        if report.get("valid_total_bbs"):
            print(f"Valid覆盖: {report['valid_covered_bbs']}/{report['valid_total_bbs']} BBs ({report['valid_coverage_rate']:.2f}%)")
        if report.get("reachable_total_bbs"):
            print(
                f"可达覆盖: {report['covered_reachable_bbs']}/{report['reachable_total_bbs']} BBs "
                f"({report['reachable_coverage_rate']:.2f}%)"
            )
        print(f"入口点: {entry_covered}")
        print(f"时间: {total_time / 60:.1f}分钟")
        print(f"\n保存: {output}")
        return report


_MANAGED_INTERLEAVED_OPTIONS = frozenset({
    "--baseline-instructions",
    "--config",
    "--entry-point",
    "--execution-mode-override",
    "--firmware",
    "--firmware-input-provenance-json",
    "--load-base",
    "--output-dir",
    "--run-profile",
    "--total-time-minutes",
})


def _reject_managed_cli_options(argv: list[str], options: frozenset[str], owner: str) -> None:
    """Reject managed options smuggled in via passthrough/forwarded args.

    The wrapper composes these options itself from its own CLI contract; a
    forwarded duplicate would silently override the wrapper's choice (argparse
    last-wins) and e.g. redirect --output-dir or change --firmware without any
    error. Both ``--opt value`` and ``--opt=value`` forms are checked.
    """
    for token in argv:
        if not token.startswith("-"):
            continue
        name = token.split("=", 1)[0]
        if name in options:
            raise SystemExit(
                f"{owner}: option {name} is managed by the lsgemu wrapper and "
                f"must not be passed through: {' '.join(argv)}"
            )


def _run_interleaved_pipeline(
    args,
    forwarded_args: list[str],
    firmware_input: ResolvedFirmwareInput,
) -> None:
    from lsgemu import cached_interleaved_runner as interleaved

    _reject_managed_cli_options(
        list(args.interleaved_arg), _MANAGED_INTERLEAVED_OPTIONS, "lsgemu interleaved passthrough"
    )
    _reject_managed_cli_options(
        forwarded_args, _MANAGED_INTERLEAVED_OPTIONS, "lsgemu forwarded args"
    )

    output_dir = args.output_dir
    if output_dir is None:
        default_output_root = configured_path(
            "LSGEMU_RUN_OUTPUT_DIR",
            configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT) / ".lsgemu_runs",
        )
        output_dir = unique_firmware_run_directory(
            default_output_root,
            firmware_input.source_path,
            namespace="lsgemu_interleaved",
        )

    baseline_instructions = args.max_instructions
    if baseline_instructions is None:
        baseline_instructions = 0

    interleaved_argv = [
        "lsgemu-interleaved",
        "--firmware",
        str(firmware_input.analysis_path),
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--total-time-minutes",
        str(float(args.time)),
        "--baseline-instructions",
        str(int(baseline_instructions)),
        "--skip-cold-isr",
        "--firmware-input-provenance-json",
        json.dumps(
            dict(firmware_input.provenance),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    ]
    if firmware_input.execution_thumb_override is not None:
        interleaved_argv.extend([
            "--execution-mode-override",
            "thumb" if firmware_input.execution_thumb_override else "arm",
        ])
    if args.entry_point is not None:
        interleaved_argv.extend(["--entry-point", hex(int(args.entry_point) & 0xFFFFFFFF)])
    if args.load_base is not None:
        interleaved_argv.extend(["--load-base", hex(int(args.load_base) & 0xFFFFFFFF)])
    if args.run_profile:
        interleaved_argv.extend(["--run-profile", str(Path(args.run_profile).resolve())])
    interleaved_argv.extend(args.interleaved_arg)
    interleaved_argv.extend(forwarded_args)

    old_argv = sys.argv
    try:
        sys.argv = interleaved_argv
        interleaved.main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("firmware")
    parser.add_argument(
        "--firmware-format",
        choices=["auto", "elf", "raw", "bin"],
        default="auto",
        help=(
            "Input handling: auto preserves native ELF and legacy raw behavior; "
            "bin activates explicit BintoElf wrapping."
        ),
    )
    parser.add_argument(
        "--bintoelf-root",
        default=str(DEFAULT_BINTOELF_ROOT),
        help="BintoElf project root containing bin2elf/__init__.py.",
    )
    parser.add_argument(
        "--bin-cache-dir",
        default=str(DEFAULT_BINTOELF_CACHE),
        help="Deterministic generated-ELF cache root.",
    )
    parser.add_argument("--bin-arch", default=None, help="Raw BIN architecture, e.g. arm.")
    parser.add_argument("--bin-bits", type=int, default=None, help="Raw BIN architecture width.")
    parser.add_argument(
        "--bin-endian",
        choices=["little", "big"],
        default=None,
        help="Raw BIN byte order.",
    )
    parser.add_argument("--bin-base", type=_parse_int, default=None, help="Raw BIN virtual load base.")
    bin_entry_group = parser.add_mutually_exclusive_group()
    bin_entry_group.add_argument(
        "--bin-entry",
        type=_parse_int,
        default=None,
        help="Raw BIN absolute entry address.",
    )
    bin_entry_group.add_argument(
        "--bin-entry-offset",
        type=_parse_int,
        default=None,
        help="Raw BIN entry offset relative to --bin-base.",
    )
    parser.add_argument(
        "--bin-thumb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set or clear ARM32 Thumb entry state for the wrapped image.",
    )
    parser.add_argument("--bin-paddr", type=_parse_int, default=None, help="Raw BIN physical load address.")
    parser.add_argument(
        "--bin-permissions",
        default=None,
        help="Generated PT_LOAD permissions in rwx order; default r-x.",
    )
    parser.add_argument(
        "--bin-align",
        type=_parse_int,
        default=None,
        help="Generated PT_LOAD alignment; default 0x1000.",
    )
    parser.add_argument(
        "--bin-bss-size",
        type=_parse_int,
        default=None,
        help="Zero-filled bytes appended to the generated PT_LOAD memory size.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.",
    )
    parser.add_argument("--time", type=float, default=60.0)
    parser.add_argument(
        "--run-profile",
        default=None,
        help="YAML/JSON profile containing LSGEMU_* environment switches for reproducible runs.",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=0,
        help="Baseline instruction cap. 0 means no instruction cap; wallclock and quiescence limits still apply.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mode",
        choices=["interleaved", "simple"],
        default="interleaved",
        help="interleaved runs the full scheduler; simple preserves the legacy four-stage wrapper.",
    )
    parser.add_argument(
        "--entry-point",
        type=_parse_int,
        default=None,
        help="Override the analyzed runtime entry point, e.g. 0x01021060 for a raw BIN vendor container.",
    )
    parser.add_argument(
        "--load-base",
        type=_parse_int,
        default=None,
        help="Map raw BIN bytes at this runtime base in addition to 0x0, e.g. 0x01021000.",
    )
    parser.add_argument(
        "--interleaved-arg",
        action="append",
        default=[],
        help="Forward one raw argument to the interleaved scheduler. Repeat for option/value pairs.",
    )
    args, forwarded_args = parser.parse_known_args()

    if args.run_profile:
        profile = load_run_profile(args.run_profile)
        applied = apply_run_profile_environment(profile)
        print(f"Run profile: {profile.get('profile_file')} ({len(applied)} environment entries)")

    firmware_input = _resolve_cli_firmware_input(parser, args)
    if firmware_input.converted:
        conversion = dict(firmware_input.provenance.get("conversion") or {})
        print(
            "[LSGEmu] BintoElf input: "
            f"{firmware_input.source_path} -> {firmware_input.analysis_path} "
            f"(cache_hit={bool(conversion.get('cache_hit'))})"
        )

    if args.mode == "interleaved":
        _run_interleaved_pipeline(args, forwarded_args, firmware_input)
        return

    lsgemu = LSGEmu(
        firmware_input.analysis_path,
        execution_time_minutes=args.time,
        baseline_instructions=args.max_instructions,
        output_dir=args.output_dir,
        entry_point=args.entry_point,
        load_base=args.load_base,
        firmware_input_provenance=dict(firmware_input.provenance),
        execution_thumb_override=firmware_input.execution_thumb_override,
    )
    lsgemu.run()


if __name__ == "__main__":
    main()
