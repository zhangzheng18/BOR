#!/usr/bin/env python3
"""
Batch-run the local HistoricalRunner pipeline across all elfmultifuzz firmware.

Each firmware gets:
1. baseline
2. ISR exploration
3. branch reservoir exploration
4. contextual ISR exploration
5. contextual ISR reservoir exploration

Reservoir checkpoints are recorded every N seconds and now include the current
branch-choice path plus total covered BBs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import (
        DEFAULT_VALID_BB_ROOT,
        HistoricalRunner,
        PROJECT_ROOT as RUNNER_PROJECT_ROOT,
    )
    from lsgemu.stat_elfmultifuzz_valid_coverage import discover_cases
else:
    from .deployment_config import apply_config_from_argv, configured_path
    from .historical_runner import (
        DEFAULT_VALID_BB_ROOT,
        HistoricalRunner,
        PROJECT_ROOT as RUNNER_PROJECT_ROOT,
    )
    from .stat_elfmultifuzz_valid_coverage import discover_cases

apply_config_from_argv()


def load_llm_config(project_root: Path):
    path = configured_path("LSGEMU_LLM_CONFIG", project_root / "LLM.yaml")
    if not path.exists():
        return None, None
    with path.open() as f:
        return yaml.safe_load(f), path


def artifact_path(firmware: Path, suffix: str) -> Path:
    return firmware.with_name(f"{firmware.stem}{suffix}")


def workspace_artifact_path(output_dir: Path, firmware: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{firmware.stem}{suffix}"


def seed_constraint_file(firmware: Path, destination: Path):
    if destination.exists():
        return
    source = artifact_path(firmware, "_lsgemu_constraints.json")
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def append_jsonl(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


@contextmanager
def temporary_env(updates: Dict[str, Optional[str]]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_csv_filter(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def case_matches(case, families: List[str], names: List[str]) -> bool:
    if families and case.family.lower() not in families:
        return False
    if names:
        haystacks = {
            case.name.lower(),
            case.rel_path.lower(),
            case.elf_path.name.lower(),
            case.elf_path.stem.lower(),
        }
        if not any(any(token in hay for hay in haystacks) for token in names):
            return False
    return True


def safe_phase_meta(meta: Dict[str, object]) -> Dict[str, object]:
    keep = {
        "covered_bbs",
        "new_bbs",
        "forced_directions",
        "forced_hit_tasks",
        "forced_missed_tasks",
        "full_hit_tasks",
        "partial_hit_tasks",
        "contexts_selected",
        "context_attempts",
        "reservoir_new_branch_points",
        "pending_paths",
        "run_result",
        "main_path_status",
        "branch_snapshots",
        "vector_isrs",
        "isr_attempts",
        "isr_started",
        "target_bbs",
        "target_covered_bbs",
        "target_remaining_bbs",
        "frontier_target_bbs",
    }
    result = {}
    for key, value in meta.items():
        if key not in keep:
            continue
        result[key] = value
    return result


def write_campaign_outputs(output_dir: Path, started: float, selected_cases, results: List[Dict[str, object]]):
    summary_json = output_dir / "campaign_summary.json"
    summary_csv = output_dir / "campaign_summary.csv"
    payload = {
        "runner": "elfmultifuzz_campaign",
        "valid_root": str(DEFAULT_VALID_BB_ROOT),
        "cases_total": len(selected_cases),
        "cases_completed": len([item for item in results if item.get("status") == "completed"]),
        "cases_failed": len([item for item in results if item.get("status") == "failed"]),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    with summary_json.open("w") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "status",
        "family",
        "name",
        "rel_path",
        "firmware",
        "covered_bbs",
        "total_bbs",
        "coverage_rate",
        "valid_covered_bbs",
        "valid_total_bbs",
        "valid_coverage_rate",
        "execution_time_seconds",
        "report_file",
        "checkpoint_file",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})


def run_case(case, args, llm_config, llm_config_path, output_root: Path) -> Dict[str, object]:
    firmware = case.elf_path.resolve()
    case_output_dir = output_root / case.rel_path
    constraint_file = workspace_artifact_path(case_output_dir, firmware, "_lsgemu_constraints.json")
    seed_constraint_file(firmware, constraint_file)
    trace_file = workspace_artifact_path(case_output_dir, firmware, "_reservoir_execution.log")
    report_file = workspace_artifact_path(case_output_dir, firmware, "_reservoir_test_result.json")
    branch_catalog_file = workspace_artifact_path(case_output_dir, firmware, "_branch_catalog.json")
    progress_file = case_output_dir / "progress.json"
    checkpoint_file = case_output_dir / "coverage_checkpoints.jsonl"
    reservoir_state_file = case_output_dir / "reservoir_state.json"
    target_provenance_file = case_output_dir / "target_provenance.jsonl"

    started = time.time()
    with temporary_env({
        "LSGEMU_RESERVOIR_PROGRESS": str(progress_file),
        "LSGEMU_COVERAGE_CHECKPOINT": str(checkpoint_file),
        "LSGEMU_COVERAGE_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
        "LSGEMU_RESERVOIR_STATE": str(reservoir_state_file),
        "LSGEMU_RESERVOIR_RESUME": "1" if args.resume_reservoir_state else None,
        "LSGEMU_TARGET_PROVENANCE": str(target_provenance_file),
        "LSGEMU_RESERVOIR_ISR_CONTEXT_LIMIT": str(args.reservoir_isr_context_limit),
    }):
        runner = HistoricalRunner(
            firmware,
            max_snapshots=args.max_snapshots,
            use_llm=llm_config is not None,
            llm_config=llm_config,
            llm_config_path=llm_config_path,
            constraint_file=str(constraint_file),
        )

        baseline = runner.run_baseline(
            max_instructions=args.baseline_instructions,
            trace_output_path=trace_file,
            quiescence_bb_limit=args.baseline_quiescence_bbs,
        )
        isr = runner.context_recovery.cold_isr(
            max_instructions=args.isr_instructions,
            count_coverage=True,
        )
        reservoir = runner.run_reservoir_branch_exploration(
            time_limit_seconds=args.reservoir_time * 60,
            replay_instructions=args.replay_instructions,
            replay_timeout=args.replay_timeout_us,
            max_tasks=args.max_tasks,
            max_tasks_per_root=args.max_tasks_per_root,
            merge_probe_bbs=args.merge_probe_bbs,
            relax_root_occurrences=not args.exact_root_occurrences,
            ordered_path_constraints=args.ordered_path_constraints,
            strict_subtree_pruning=args.strict_subtree_pruning,
            prioritize_static_potential=args.prioritize_static_potential,
            skip_stale_tasks=args.skip_stale_tasks,
            defer_stale_tasks=not args.no_defer_stale_tasks,
            stale_replay_interval=args.stale_replay_interval,
            state_signature_pruning=not args.no_state_signature_pruning,
            state_probe_bbs=args.state_probe_bbs,
            edge_saturation_pruning=not args.no_edge_saturation_pruning,
            edge_zero_new_limit=args.edge_zero_new_limit,
            dynamic_frontier_seeding=not args.no_dynamic_frontier_seeding,
            isolate_path_coverage=args.isolate_path_coverage,
            phase_name="branch_reservoir",
            reset_state=True,
        )

        contextual_isr = set()
        contextual_isr_reservoir = set()
        switch_frontier_targeted = set()
        switch_frontier_rounds = []
        frontier_targeted = set()
        frontier_target_rounds = []
        contextual_isr_frontier_targeted = set()
        contextual_isr_frontier_rounds = []
        contextual_isr_enabled = args.contextual_isr_contexts != 0 or (
            args.contextual_isr_include_reservoir and args.contextual_isr_reservoir_contexts != 0
        )
        if contextual_isr_enabled:
            contextual_isr = runner.context_recovery.contextual_isr(
                max_contexts=args.contextual_isr_contexts,
                max_isrs=args.contextual_isr_max_isrs,
                max_instructions=args.contextual_isr_instructions,
                replay_timeout=args.replay_timeout_us,
                time_limit_seconds=args.contextual_isr_time * 60 if args.contextual_isr_time > 0 else None,
                include_reservoir_contexts=args.contextual_isr_include_reservoir,
                max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
            )
        if args.contextual_isr_reservoir_time > 0 and contextual_isr_enabled:
            contextual_isr_reservoir = runner.context_recovery.isr_reservoir(
                time_limit_seconds=args.contextual_isr_reservoir_time * 60,
                max_instructions=args.contextual_isr_reservoir_instructions,
                replay_timeout=args.replay_timeout_us,
                max_tasks=args.contextual_isr_reservoir_max_tasks,
                max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                dynamic_frontier_seeding=not args.no_contextual_isr_reservoir_dynamic_frontier_seeding,
                context_snapshots=True,
                max_contexts=args.contextual_isr_contexts,
                include_reservoir_contexts=args.contextual_isr_include_reservoir,
                max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                phase_name="contextual_isr_reservoir",
            )

        frontier_analysis_before = runner.uncovered_coverage_summary()
        switch_frontier_targets = set()
        if args.switch_frontier_time > 0 and not args.frontier_targeted_switch_only:
            for round_index in range(max(1, args.switch_frontier_rounds)):
                current_switch_targets = runner.frontier_target_bbs(
                    min_uncovered_successors=max(1, args.switch_frontier_min_uncovered_successors),
                    switch_only=True,
                    max_targets=args.switch_frontier_max_targets,
                )
                if round_index == 0:
                    switch_frontier_targets = set(current_switch_targets)
                if not current_switch_targets:
                    break
                phase_name = "switch_frontier_targeted" if round_index == 0 else f"switch_frontier_targeted_round_{round_index + 1}"
                round_coverage = runner.run_reservoir_branch_exploration(
                    time_limit_seconds=args.switch_frontier_time * 60,
                    replay_instructions=args.replay_instructions,
                    replay_timeout=args.replay_timeout_us,
                    max_tasks=args.max_tasks,
                    max_tasks_per_root=args.max_tasks_per_root,
                    merge_probe_bbs=args.merge_probe_bbs,
                    relax_root_occurrences=not args.exact_root_occurrences,
                    ordered_path_constraints=args.ordered_path_constraints,
                    strict_subtree_pruning=args.strict_subtree_pruning,
                    prioritize_static_potential=args.prioritize_static_potential,
                    skip_stale_tasks=args.skip_stale_tasks,
                    defer_stale_tasks=not args.no_defer_stale_tasks,
                    stale_replay_interval=args.stale_replay_interval,
                    state_signature_pruning=not args.no_state_signature_pruning,
                    state_probe_bbs=args.state_probe_bbs,
                    edge_saturation_pruning=not args.no_edge_saturation_pruning,
                    edge_zero_new_limit=1,
                    dynamic_frontier_seeding=not args.no_dynamic_frontier_seeding,
                    isolate_path_coverage=args.isolate_path_coverage,
                    target_bbs=current_switch_targets,
                    prioritize_target_bbs=True,
                    continue_target_merges=True,
                    target_probe_bbs=min(args.state_probe_bbs, 512) if args.state_probe_bbs and args.state_probe_bbs > 0 else 512,
                    phase_name=phase_name,
                    reset_state=True,
                )
                switch_frontier_targeted.update(round_coverage)
                phase_meta = runner.phase_metadata.get(phase_name, {})
                switch_frontier_rounds.append({
                    "round": round_index + 1,
                    "phase": phase_name,
                    "candidate_targets": len(current_switch_targets),
                    "covered_bbs": len(round_coverage),
                    "new_bbs": phase_meta.get("new_bbs", 0),
                    "target_hit_tasks": phase_meta.get("target_hit_tasks", 0),
                    "target_bbs_discovered": phase_meta.get("target_bbs_discovered", 0),
                })
                if phase_meta.get("new_bbs", 0) <= 0:
                    break
        frontier_targets = set()
        if args.frontier_targeted_time > 0:
            for round_index in range(max(1, args.frontier_targeted_rounds)):
                current_frontier_targets = runner.frontier_target_bbs(
                    min_uncovered_successors=args.frontier_targeted_min_uncovered_successors,
                    switch_only=args.frontier_targeted_switch_only,
                    max_targets=args.frontier_targeted_max_targets,
                )
                if round_index == 0:
                    frontier_targets = set(current_frontier_targets)
                if not current_frontier_targets:
                    break
                phase_name = "frontier_targeted" if round_index == 0 else f"frontier_targeted_round_{round_index + 1}"
                round_coverage = runner.run_reservoir_branch_exploration(
                    time_limit_seconds=args.frontier_targeted_time * 60,
                    replay_instructions=args.replay_instructions,
                    replay_timeout=args.replay_timeout_us,
                    max_tasks=args.max_tasks,
                    max_tasks_per_root=args.max_tasks_per_root,
                    merge_probe_bbs=args.merge_probe_bbs,
                    relax_root_occurrences=not args.exact_root_occurrences,
                    ordered_path_constraints=args.ordered_path_constraints,
                    strict_subtree_pruning=args.strict_subtree_pruning,
                    prioritize_static_potential=args.prioritize_static_potential,
                    skip_stale_tasks=args.skip_stale_tasks,
                    defer_stale_tasks=not args.no_defer_stale_tasks,
                    stale_replay_interval=args.stale_replay_interval,
                    state_signature_pruning=not args.no_state_signature_pruning,
                    state_probe_bbs=args.state_probe_bbs,
                    edge_saturation_pruning=not args.no_edge_saturation_pruning,
                    edge_zero_new_limit=1,
                    dynamic_frontier_seeding=not args.no_dynamic_frontier_seeding,
                    isolate_path_coverage=args.isolate_path_coverage,
                    target_bbs=current_frontier_targets,
                    prioritize_target_bbs=True,
                    continue_target_merges=True,
                    target_probe_bbs=min(args.state_probe_bbs, 512) if args.state_probe_bbs and args.state_probe_bbs > 0 else 512,
                    phase_name=phase_name,
                    reset_state=True,
                )
                frontier_targeted.update(round_coverage)
                phase_meta = runner.phase_metadata.get(phase_name, {})
                frontier_target_rounds.append({
                    "round": round_index + 1,
                    "phase": phase_name,
                    "candidate_targets": len(current_frontier_targets),
                    "covered_bbs": len(round_coverage),
                    "new_bbs": phase_meta.get("new_bbs", 0),
                    "target_hit_tasks": phase_meta.get("target_hit_tasks", 0),
                    "target_bbs_discovered": phase_meta.get("target_bbs_discovered", 0),
                })
                if phase_meta.get("new_bbs", 0) <= 0:
                    break
        contextual_isr_frontier_targets = set()
        if args.contextual_isr_frontier_time > 0 and contextual_isr_enabled:
            for round_index in range(max(1, args.contextual_isr_frontier_rounds)):
                current_contextual_frontier_targets = runner.frontier_target_bbs(
                    min_uncovered_successors=args.contextual_isr_frontier_min_uncovered_successors,
                    switch_only=args.contextual_isr_frontier_switch_only,
                    max_targets=args.contextual_isr_frontier_max_targets,
                )
                if round_index == 0:
                    contextual_isr_frontier_targets = set(current_contextual_frontier_targets)
                if not current_contextual_frontier_targets:
                    break
                phase_name = (
                    "contextual_isr_frontier_targeted"
                    if round_index == 0 else f"contextual_isr_frontier_targeted_round_{round_index + 1}"
                )
                round_coverage = runner.context_recovery.isr_reservoir(
                    time_limit_seconds=args.contextual_isr_frontier_time * 60,
                    max_instructions=args.contextual_isr_reservoir_instructions,
                    replay_timeout=args.replay_timeout_us,
                    max_tasks=args.contextual_isr_reservoir_max_tasks,
                    max_tasks_per_isr=args.contextual_isr_reservoir_max_tasks_per_isr,
                    dynamic_frontier_seeding=not args.no_contextual_isr_reservoir_dynamic_frontier_seeding,
                    context_snapshots=True,
                    max_contexts=args.contextual_isr_contexts,
                    include_reservoir_contexts=args.contextual_isr_include_reservoir,
                    max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
                    target_bbs=current_contextual_frontier_targets,
                    prioritize_target_bbs=True,
                    phase_name=phase_name,
                )
                contextual_isr_frontier_targeted.update(round_coverage)
                phase_meta = runner.phase_metadata.get(phase_name, {})
                contextual_isr_frontier_rounds.append({
                    "round": round_index + 1,
                    "phase": phase_name,
                    "candidate_targets": len(current_contextual_frontier_targets),
                    "covered_bbs": len(round_coverage),
                    "new_bbs": phase_meta.get("new_bbs", 0),
                    "target_hit_tasks": phase_meta.get("target_hit_tasks", 0),
                    "target_bbs_discovered": phase_meta.get("target_bbs_discovered", 0),
                })
                if phase_meta.get("new_bbs", 0) <= 0:
                    break
        frontier_analysis_after = runner.uncovered_coverage_summary()

        elapsed = time.time() - started
        branch_catalog = runner.save_branch_catalog(branch_catalog_file)
        static_reachability = runner.static_reachability_summary()
        # P1-D（k.5 C7）：K4 面在 save_report 前从 runner 状态构建（extra 会
        # 顶层并入报告并落盘）。
        k4_covered = runner.validate_coverage(runner.global_coverage)
        k4_validated = set(runner._canonical_evidence_partitions()["validated"])
        k4_symbol_watch = build_k4_symbol_watch(
            args.k4_symbol_watch, k4_covered, k4_validated
        )
        report = runner.save_report(
            report_file,
            execution_time_seconds=elapsed,
            extra={
                "runner": "elfmultifuzz_campaign",
                "constraint_file": str(constraint_file),
                "branch_catalog_file": str(branch_catalog_file),
                "static_reachability": static_reachability,
                "component_bbs": {
                    "baseline": len(baseline),
                    "isr": len(isr),
                    "reservoir": len(reservoir),
                    "contextual_isr": len(contextual_isr),
                    "contextual_isr_reservoir": len(contextual_isr_reservoir),
                    "contextual_isr_frontier_targeted": len(contextual_isr_frontier_targeted),
                    "switch_frontier_targeted": len(switch_frontier_targeted),
                    "frontier_targeted": len(frontier_targeted),
                },
                "contextual_isr_frontier_targeting": {
                    "candidate_targets": len(contextual_isr_frontier_targets),
                    "max_targets": args.contextual_isr_frontier_max_targets,
                    "min_uncovered_successors": args.contextual_isr_frontier_min_uncovered_successors,
                    "switch_only": args.contextual_isr_frontier_switch_only,
                    "rounds_requested": args.contextual_isr_frontier_rounds,
                    "rounds_executed": len(contextual_isr_frontier_rounds),
                    "round_results": contextual_isr_frontier_rounds,
                },
                "switch_frontier_targeting": {
                    "candidate_targets": len(switch_frontier_targets),
                    "max_targets": args.switch_frontier_max_targets,
                    "min_uncovered_successors": args.switch_frontier_min_uncovered_successors,
                    "rounds_requested": args.switch_frontier_rounds,
                    "rounds_executed": len(switch_frontier_rounds),
                    "round_results": switch_frontier_rounds,
                },
                "frontier_targeting": {
                    "candidate_targets": len(frontier_targets),
                    "max_targets": args.frontier_targeted_max_targets,
                    "min_uncovered_successors": args.frontier_targeted_min_uncovered_successors,
                    "switch_only": args.frontier_targeted_switch_only,
                    "rounds_requested": args.frontier_targeted_rounds,
                    "rounds_executed": len(frontier_target_rounds),
                    "round_results": frontier_target_rounds,
                },
                "uncovered_analysis_before_frontier": frontier_analysis_before,
                "uncovered_analysis_final": frontier_analysis_after,
                "branch_catalog_summary": {
                    "static_conditional_branch_points": branch_catalog["static_conditional_branch_points"],
                    "main_path_branch_events": branch_catalog["main_path_branch_events"],
                    "main_path_branch_points": branch_catalog["main_path_branch_points"],
                    "reservoir_discovered_branch_events": branch_catalog["reservoir_discovered_branch_events"],
                    "reservoir_new_branch_points": branch_catalog["reservoir_new_branch_points"],
                    # P1-F（cycle3 k.5 C4）：发现点 oracle 事件通道的可见性键
                    #（事件本体在 {FW}_branch_catalog.json 的
                    # reservoir_discovery_events；摘要只报基数防报告膨胀）。
                    "reservoir_discovery_event_count": len(
                        branch_catalog.get("reservoir_discovery_events", []) or []
                    ),
                    "reservoir_discovery_events_truncated": int(
                        branch_catalog.get("reservoir_discovery_events_truncated", 0) or 0
                    ),
                },
                # P1-D（k.5 C7）：K4 地址+符号双写（偶基+|1 双口径）。
                "k4_symbol_watch": k4_symbol_watch,
                "checkpoint_file": str(checkpoint_file),
            },
        )

    phase_metadata = {
        name: safe_phase_meta(meta)
        for name, meta in runner.phase_metadata.items()
    }
    return {
        "status": "completed",
        "family": case.family,
        "name": case.name,
        "rel_path": case.rel_path,
        "firmware": str(firmware),
        "covered_bbs": int(report.get("covered_bbs", 0)),
        "total_bbs": int(report.get("total_bbs", 0)),
        "coverage_rate": float(report.get("coverage_rate", 0.0)),
        "valid_covered_bbs": int(report.get("valid_covered_bbs", 0)),
        "valid_total_bbs": int(report.get("valid_total_bbs", 0)),
        "valid_coverage_rate": float(report.get("valid_coverage_rate", 0.0)),
        "execution_time_seconds": elapsed,
        "report_file": str(report_file),
        "checkpoint_file": str(checkpoint_file),
        "progress_file": str(progress_file),
        "target_provenance_file": str(target_provenance_file),
        "phase_metadata": phase_metadata,
    }


def build_k4_symbol_watch(spec, covered, validated):
    """K4 地址+符号双写（P1-D，cycle3 k.5 C7）。

    ``spec`` = ``"symbol@0xADDR[,...]"``；地址取**偶基**（函数首，Thumb
    位剥离），同时写 ``address``（偶基）与 ``address_thumb``（|1）双口径，
    covered/validated 成员判定在偶基上（covered 集存 BB 起址=偶地址）。
    纯函数：两个成员集合由调用方给（campaign 侧 = runner 的
    validate_coverage(global_coverage) 与 canonical validated 分区）。
    """
    covered_set = {int(item) & 0xFFFFFFFF for item in covered or []}
    validated_set = {int(item) & 0xFFFFFFFF for item in validated or []}
    entries = []
    for item in filter(None, (part.strip() for part in str(spec or "").split(","))):
        symbol, _, raw_address = item.rpartition("@")
        if not symbol or not raw_address:
            continue
        try:
            address = int(raw_address, 0) & 0xFFFFFFFE
        except ValueError:
            continue
        entries.append({
            "symbol": symbol,
            "address": f"0x{address:08x}",
            "address_thumb": f"0x{address | 1:08x}",
            "covered": address in covered_set,
            "validated": address in validated_set,
        })
    return {
        "enabled": bool(entries),
        "entries": entries,
        "covered_count": sum(1 for entry in entries if entry["covered"]),
        "validated_count": sum(1 for entry in entries if entry["validated"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument("--valid-root", default=str(DEFAULT_VALID_BB_ROOT))
    parser.add_argument(
        "--output-dir",
        default=str(configured_path("LSGEMU_RUN_OUTPUT_DIR", configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4") / ".lsgemu_runs") / "elfmultifuzz_campaign"),
    )
    parser.add_argument("--families", default=None, help="Comma-separated family filter, e.g. P2IM,HALucinator")
    parser.add_argument("--match", default=None, help="Comma-separated substring filter on case name/path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases; 0 means all")
    parser.add_argument("--resume-skip-existing", action="store_true", help="Skip cases with existing result json in output dir")
    parser.add_argument(
        "--k4-symbol-watch",
        default=None,
        help=(
            "K4 符号观察表 'symbol@0xADDR[,...]'——报告内 k4_symbol_watch "
            "按地址（偶基）+符号双写 covered/validated 成员判定"
        ),
    )
    parser.add_argument("--baseline-instructions", type=int, default=500000)
    parser.add_argument("--baseline-quiescence-bbs", type=int, default=50000)
    parser.add_argument("--max-snapshots", type=int, default=5)
    parser.add_argument("--isr-instructions", type=int, default=10000)
    parser.add_argument("--reservoir-time", type=int, default=20, help="Branch reservoir minutes per firmware")
    parser.add_argument("--contextual-isr-contexts", type=int, default=32)
    parser.add_argument("--contextual-isr-max-isrs", type=int, default=0)
    parser.add_argument("--contextual-isr-instructions", type=int, default=15000)
    parser.add_argument("--contextual-isr-time", type=int, default=1)
    parser.add_argument("--contextual-isr-include-reservoir", action="store_true")
    parser.add_argument("--contextual-isr-reservoir-contexts", type=int, default=16)
    parser.add_argument("--contextual-isr-reservoir-time", type=int, default=1)
    parser.add_argument("--contextual-isr-reservoir-instructions", type=int, default=15000)
    parser.add_argument("--contextual-isr-reservoir-max-tasks", type=int, default=500)
    parser.add_argument("--contextual-isr-reservoir-max-tasks-per-isr", type=int, default=64)
    parser.add_argument("--no-contextual-isr-reservoir-dynamic-frontier-seeding", action="store_true")
    parser.add_argument("--contextual-isr-frontier-time", type=int, default=0, help="Optional target-guided contextual ISR frontier pass in minutes")
    parser.add_argument("--contextual-isr-frontier-rounds", type=int, default=1)
    parser.add_argument("--contextual-isr-frontier-max-targets", type=int, default=0)
    parser.add_argument("--contextual-isr-frontier-min-uncovered-successors", type=int, default=1)
    parser.add_argument("--contextual-isr-frontier-switch-only", action="store_true")
    parser.add_argument("--frontier-targeted-time", type=int, default=2, help="Post-pass frontier-targeted reservoir minutes per firmware")
    parser.add_argument("--frontier-targeted-rounds", type=int, default=1, help="Repeat frontier-targeted passes this many times, recomputing immediate frontier after each round")
    parser.add_argument("--frontier-targeted-max-targets", type=int, default=0, help="0 or negative means all immediate uncovered frontier BBs")
    parser.add_argument("--frontier-targeted-min-uncovered-successors", type=int, default=1, help="Only use covered frontier predecessors with at least this many uncovered successors")
    parser.add_argument("--frontier-targeted-switch-only", action="store_true", help="Only target TBB/TBH frontier successors in the post-pass")
    parser.add_argument("--switch-frontier-time", type=int, default=0, help="Optional high-fanout switch-only frontier stage before general frontier targeting, in minutes")
    parser.add_argument("--switch-frontier-rounds", type=int, default=1, help="Repeat the switch-only frontier stage this many times")
    parser.add_argument("--switch-frontier-min-uncovered-successors", type=int, default=2, help="Only use switch frontier predecessors with at least this many uncovered successors")
    parser.add_argument("--switch-frontier-max-targets", type=int, default=0, help="0 or negative means all switch-only frontier targets")
    parser.add_argument("--replay-instructions", type=int, default=15000)
    parser.add_argument("--replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--max-tasks", type=int, default=1000)
    parser.add_argument("--max-tasks-per-root", type=int, default=0)
    parser.add_argument("--merge-probe-bbs", type=int, default=128)
    parser.add_argument("--exact-root-occurrences", action="store_true")
    parser.add_argument("--ordered-path-constraints", action="store_true")
    parser.add_argument("--strict-subtree-pruning", action="store_true")
    parser.add_argument("--prioritize-static-potential", action="store_true")
    parser.add_argument("--skip-stale-tasks", action="store_true")
    parser.add_argument("--no-defer-stale-tasks", action="store_true")
    parser.add_argument("--stale-replay-interval", type=int, default=16)
    parser.add_argument("--no-state-signature-pruning", action="store_true")
    parser.add_argument("--state-probe-bbs", type=int, default=512)
    parser.add_argument("--no-edge-saturation-pruning", action="store_true")
    parser.add_argument("--edge-zero-new-limit", type=int, default=3)
    parser.add_argument("--no-dynamic-frontier-seeding", action="store_true")
    parser.add_argument("--isolate-path-coverage", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--reservoir-isr-context-limit", type=int, default=96)
    parser.add_argument("--resume-reservoir-state", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_root = Path(args.valid_root).resolve()
    cases, _alias_map = discover_cases(valid_root)
    families = parse_csv_filter(args.families)
    names = parse_csv_filter(args.match)
    selected_cases = [case for case in cases if case_matches(case, families, names)]
    selected_cases.sort(key=lambda item: (item.family, item.name))
    if args.limit and args.limit > 0:
        selected_cases = selected_cases[:args.limit]

    llm_config, llm_config_path = load_llm_config(RUNNER_PROJECT_ROOT)
    campaign_log = output_dir / "campaign_events.jsonl"
    results: List[Dict[str, object]] = []
    started = time.time()

    append_jsonl(campaign_log, {
        "status": "campaign_start",
        "timestamp": started,
        "cases_total": len(selected_cases),
        "output_dir": str(output_dir),
    })

    for index, case in enumerate(selected_cases, start=1):
        firmware = case.elf_path.resolve()
        case_output_dir = output_dir / case.rel_path
        report_file = workspace_artifact_path(case_output_dir, firmware, "_reservoir_test_result.json")
        if args.resume_skip_existing and report_file.exists():
            try:
                with report_file.open() as f:
                    existing = json.load(f)
                result = {
                    "status": "completed",
                    "family": case.family,
                    "name": case.name,
                    "rel_path": case.rel_path,
                    "firmware": str(firmware),
                    "covered_bbs": int(existing.get("covered_bbs", 0)),
                    "total_bbs": int(existing.get("total_bbs", 0)),
                    "coverage_rate": float(existing.get("coverage_rate", 0.0)),
                    "valid_covered_bbs": int(existing.get("valid_covered_bbs", 0)),
                    "valid_total_bbs": int(existing.get("valid_total_bbs", 0)),
                    "valid_coverage_rate": float(existing.get("valid_coverage_rate", 0.0)),
                    "execution_time_seconds": float(existing.get("execution_time_seconds", 0.0) or 0.0),
                    "report_file": str(report_file),
                    "checkpoint_file": str(case_output_dir / "coverage_checkpoints.jsonl"),
                }
                results.append(result)
                append_jsonl(campaign_log, {
                    "status": "case_skipped_existing",
                    "timestamp": time.time(),
                    "case_index": index,
                    "case_total": len(selected_cases),
                    "rel_path": case.rel_path,
                    "report_file": str(report_file),
                })
                write_campaign_outputs(output_dir, started, selected_cases, results)
                continue
            except Exception:
                pass

        append_jsonl(campaign_log, {
            "status": "case_start",
            "timestamp": time.time(),
            "case_index": index,
            "case_total": len(selected_cases),
            "rel_path": case.rel_path,
            "firmware": str(firmware),
        })
        try:
            result = run_case(case, args, llm_config, llm_config_path, output_dir)
            results.append(result)
            append_jsonl(campaign_log, {
                "status": "case_complete",
                "timestamp": time.time(),
                "case_index": index,
                "case_total": len(selected_cases),
                "rel_path": case.rel_path,
                "covered_bbs": result["covered_bbs"],
                "total_bbs": result["total_bbs"],
                "coverage_rate": result["coverage_rate"],
                "valid_covered_bbs": result["valid_covered_bbs"],
                "valid_total_bbs": result["valid_total_bbs"],
                "valid_coverage_rate": result["valid_coverage_rate"],
                "execution_time_seconds": result["execution_time_seconds"],
                "checkpoint_file": result["checkpoint_file"],
                "report_file": result["report_file"],
            })
        except Exception as exc:
            logging.exception("Campaign failed for %s", case.rel_path)
            result = {
                "status": "failed",
                "family": case.family,
                "name": case.name,
                "rel_path": case.rel_path,
                "firmware": str(firmware),
                "error": str(exc),
                "report_file": str(report_file),
                "checkpoint_file": str(case_output_dir / "coverage_checkpoints.jsonl"),
            }
            results.append(result)
            append_jsonl(campaign_log, {
                "status": "case_failed",
                "timestamp": time.time(),
                "case_index": index,
                "case_total": len(selected_cases),
                "rel_path": case.rel_path,
                "error": str(exc),
            })
        write_campaign_outputs(output_dir, started, selected_cases, results)

    append_jsonl(campaign_log, {
        "status": "campaign_complete",
        "timestamp": time.time(),
        "cases_total": len(selected_cases),
        "cases_completed": len([item for item in results if item.get("status") == "completed"]),
        "cases_failed": len([item for item in results if item.get("status") == "failed"]),
        "elapsed_seconds": time.time() - started,
    })

    completed = [item for item in results if item.get("status") == "completed"]
    print(f"cases_total={len(selected_cases)}")
    print(f"cases_completed={len(completed)}")
    print(f"cases_failed={len([item for item in results if item.get('status') == 'failed'])}")
    print(f"campaign_log={campaign_log}")
    print(f"summary_json={output_dir / 'campaign_summary.json'}")
    print(f"summary_csv={output_dir / 'campaign_summary.csv'}")


if __name__ == "__main__":
    main()
