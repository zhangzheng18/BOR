#!/usr/bin/env python3
"""
Gateway ensemble reservoir test.

Single-run branch exploration can lose coverage because pruning state,
edge-saturation state, and global coverage are shared across strategies. This
entrypoint runs several replay strategies in isolated HistoricalRunner
instances, then merges only the dynamically validated BB coverage.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yaml

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import (
        HistoricalRunner,
        PreparedFirmware,
        DEFAULT_REAL_GATEWAY,
        PROJECT_ROOT as RUNNER_PROJECT_ROOT,
    )
else:
    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import (
        HistoricalRunner,
        PreparedFirmware,
        DEFAULT_REAL_GATEWAY,
        PROJECT_ROOT as RUNNER_PROJECT_ROOT,
    )

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


def load_report_coverage(path: Optional[str]) -> Set[int]:
    if not path:
        return set()
    try:
        with Path(path).open() as f:
            data = json.load(f)
    except Exception:
        return set()
    covered = data.get("covered_bb_list", []) if isinstance(data, dict) else []
    result = set()
    for item in covered:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


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


def phase(name: str, minutes: float, **kwargs) -> Dict[str, object]:
    return {"phase_name": name, "minutes": minutes, **kwargs}


def strategy_profiles() -> Dict[str, List[Dict[str, object]]]:
    """Return strategy plans ordered by expected marginal utility."""
    quick = [
        {
            "name": "three_stage_finaliso_ctx",
            "isr_reservoir_minutes": 1.0,
            "isr_reservoir_max_tasks": 180,
            "isr_reservoir_max_tasks_per_isr": 32,
            "contextual_isr_contexts": 4,
            "phases": [
                phase("branch_reservoir", 1.0, edge_zero_new_limit=1),
                phase("diverse_static_potential", 0.5, prioritize_static_potential=True, edge_zero_new_limit=1, reset_state=True),
                phase("diverse_exact_occurrence", 0.5, relax_root_occurrences=False, edge_zero_new_limit=1, reset_state=True),
                phase("branch_reservoir_final", 2.0, isolate_path_coverage=True, edge_zero_new_limit=1, reset_state=True),
            ],
        },
        {
            "name": "frontier_narrow_isr",
            "isr_reservoir_minutes": 1.0,
            "isr_reservoir_max_tasks": 180,
            "isr_reservoir_max_tasks_per_isr": 32,
            "phases": [
                phase("branch_reservoir", 3.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "legacy_state_sig",
            "phases": [
                phase("branch_reservoir", 3.0, edge_saturation_pruning=False, dynamic_frontier_seeding=False),
            ],
        },
        {
            "name": "ordered_seq",
            "phases": [
                phase("branch_reservoir", 1.5, ordered_path_constraints=True, edge_saturation_pruning=False),
            ],
        },
    ]

    balanced = [
        {
            "name": "three_stage_finaliso_ctx",
            "isr_reservoir_minutes": 3.0,
            "isr_reservoir_max_tasks": 500,
            "isr_reservoir_max_tasks_per_isr": 64,
            "contextual_isr_contexts": 8,
            "phases": [
                phase("branch_reservoir", 3.0, edge_zero_new_limit=1),
                phase("diverse_static_potential", 1.0, prioritize_static_potential=True, edge_zero_new_limit=1, reset_state=True),
                phase("diverse_ordered_constraints", 1.0, ordered_path_constraints=True, edge_zero_new_limit=1, reset_state=True),
                phase("diverse_exact_occurrence", 1.0, relax_root_occurrences=False, edge_zero_new_limit=1, reset_state=True),
                phase("branch_reservoir_final", 6.0, isolate_path_coverage=True, edge_zero_new_limit=1, reset_state=True),
            ],
        },
        {
            "name": "frontier_narrow_isr",
            "isr_reservoir_minutes": 3.0,
            "isr_reservoir_max_tasks": 360,
            "isr_reservoir_max_tasks_per_isr": 32,
            "phases": [
                phase("branch_reservoir", 6.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "legacy_state_sig",
            "phases": [
                phase("branch_reservoir", 6.0, edge_saturation_pruning=False, dynamic_frontier_seeding=False),
            ],
        },
        {
            "name": "ordered_seq",
            "phases": [
                phase("branch_reservoir", 2.0, ordered_path_constraints=True, edge_saturation_pruning=False),
            ],
        },
        {
            "name": "edge_sat1",
            "phases": [
                phase("branch_reservoir", 5.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "stale64",
            "phases": [
                phase("branch_reservoir", 5.0, stale_replay_interval=64, edge_zero_new_limit=1),
            ],
        },
    ]

    wide = [
        *balanced,
        {
            "name": "no_state_signature",
            "phases": [
                phase(
                    "branch_reservoir",
                    5.0,
                    state_signature_pruning=False,
                    merge_probe_bbs=64,
                    state_probe_bbs=0,
                    edge_zero_new_limit=1,
                ),
            ],
        },
        {
            "name": "static_potential",
            "phases": [
                phase("branch_reservoir", 5.0, prioritize_static_potential=True, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "exact_occurrence",
            "phases": [
                phase("branch_reservoir", 4.0, relax_root_occurrences=False, edge_zero_new_limit=1),
            ],
        },
    ]

    # Long profile: encode the strategy order that historical dynamic reports
    # showed to have the largest marginal union gains. It is intentionally
    # expensive and should be used when the goal is maximum BB coverage rather
    # than fast iteration.
    max_profile = [
        {
            "name": "diverse_three_stage_finaliso_ctx_full",
            "isr_reservoir_minutes": 5.0,
            "isr_reservoir_max_tasks": 500,
            "isr_reservoir_max_tasks_per_isr": 64,
            "contextual_isr_contexts": 8,
            "phases": [
                phase("branch_reservoir", 5.0, edge_zero_new_limit=1),
                phase("diverse_static_potential", 3.0, prioritize_static_potential=True, edge_zero_new_limit=1, reset_state=True),
                phase("diverse_ordered_constraints", 3.0, ordered_path_constraints=True, edge_zero_new_limit=1, reset_state=True),
                phase("diverse_exact_occurrence", 3.0, relax_root_occurrences=False, edge_zero_new_limit=1, reset_state=True),
                phase("branch_reservoir_final", 20.0, isolate_path_coverage=True, edge_zero_new_limit=1, reset_state=True),
            ],
        },
        {
            "name": "frontier_isr_full",
            "isr_reservoir_minutes": 5.0,
            "isr_reservoir_max_tasks": 500,
            "isr_reservoir_max_tasks_per_isr": 32,
            "phases": [
                phase("branch_reservoir", 30.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "frontier_isr_legacy_short",
            "isr_reservoir_minutes": 3.0,
            "isr_reservoir_max_tasks": 300,
            "isr_reservoir_max_tasks_per_isr": 32,
            "isr_dynamic_frontier_seeding": False,
            "phases": [
                phase("branch_reservoir", 5.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "legacy_state_sig_full",
            "phases": [
                phase("branch_reservoir", 30.0, edge_saturation_pruning=False, dynamic_frontier_seeding=False),
            ],
        },
        {
            "name": "ordered_seq_full",
            "phases": [
                phase("branch_reservoir", 2.5, ordered_path_constraints=True, edge_saturation_pruning=False),
            ],
        },
        {
            "name": "ordered_seq_legacy_short",
            "phases": [
                phase(
                    "branch_reservoir",
                    2.0,
                    ordered_path_constraints=True,
                    edge_saturation_pruning=False,
                    dynamic_frontier_seeding=False,
                ),
            ],
        },
        {
            "name": "contextual_post_full",
            "isr_reservoir_minutes": 5.0,
            "isr_reservoir_max_tasks": 500,
            "isr_reservoir_max_tasks_per_isr": 64,
            "contextual_isr_contexts": 8,
            "contextual_isr_include_reservoir": True,
            "contextual_isr_reservoir_contexts": 24,
            "phases": [
                phase("branch_reservoir", 30.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "edge_sat1_full",
            "phases": [
                phase("branch_reservoir", 30.0, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "edge_sat1_no_frontier_full",
            "phases": [
                phase("branch_reservoir", 30.0, edge_zero_new_limit=1, dynamic_frontier_seeding=False),
            ],
        },
        {
            "name": "stale64_full",
            "phases": [
                phase("branch_reservoir", 30.0, stale_replay_interval=64, edge_zero_new_limit=1),
            ],
        },
        {
            "name": "no_state_signature_full",
            "phases": [
                phase(
                    "branch_reservoir",
                    20.0,
                    state_signature_pruning=False,
                    merge_probe_bbs=64,
                    state_probe_bbs=0,
                    edge_zero_new_limit=1,
                ),
            ],
        },
        {
            "name": "targeted_legacy_state_sig",
            "phases": [
                phase(
                    "branch_reservoir",
                    30.0,
                    edge_saturation_pruning=False,
                    dynamic_frontier_seeding=False,
                    target_guided=True,
                    continue_target_merges=True,
                ),
            ],
        },
        {
            "name": "targeted_edge_frontier",
            "phases": [
                phase(
                    "branch_reservoir",
                    20.0,
                    edge_zero_new_limit=1,
                    max_tasks_per_root=32,
                    stale_replay_interval=64,
                    target_probe_bbs=512,
                    dynamic_frontier_seeding=True,
                    target_guided=True,
                    continue_target_merges=True,
                ),
            ],
        },
        {
            "name": "targeted_edge_nofrontier",
            "phases": [
                phase(
                    "branch_reservoir",
                    20.0,
                    edge_zero_new_limit=1,
                    dynamic_frontier_seeding=False,
                    target_guided=True,
                    continue_target_merges=True,
                ),
            ],
        },
        {
            "name": "targeted_contextual_isr",
            "isr_reservoir_minutes": 3.0,
            "isr_reservoir_max_tasks": 300,
            "isr_reservoir_max_tasks_per_isr": 32,
            "contextual_isr_contexts": 8,
            "contextual_isr_include_reservoir": True,
            "contextual_isr_reservoir_contexts": 32,
            "contextual_isr_minutes": 5.0,
            "phases": [
                phase(
                    "branch_reservoir",
                    8.0,
                    edge_zero_new_limit=1,
                    dynamic_frontier_seeding=True,
                    target_guided=True,
                    continue_target_merges=True,
                ),
            ],
        },
    ]
    return {"quick": quick, "balanced": balanced, "wide": wide, "max": max_profile}


def selected_strategies(profile: str, names: Optional[str]) -> List[Dict[str, object]]:
    profiles = strategy_profiles()
    if profile not in profiles:
        raise ValueError(f"unknown profile: {profile}")
    strategies = profiles[profile]
    if not names:
        return strategies

    wanted = [item.strip() for item in names.split(",") if item.strip()]
    by_name = {strategy["name"]: strategy for strategy in strategies}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown strategies for profile {profile}: {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def scaled_seconds(minutes: float, scale: float) -> float:
    return max(1.0, float(minutes) * max(0.01, scale) * 60.0)


def run_reservoir_phase(
    runner: HistoricalRunner,
    phase_cfg: Dict[str, object],
    args,
    time_scale: float,
) -> Set[int]:
    target_guided = bool(
        phase_cfg.get(
            "target_guided",
            args.target_guided or bool(getattr(args, "target_bbs", set())),
        )
    )
    return runner.run_reservoir_branch_exploration(
        time_limit_seconds=scaled_seconds(float(phase_cfg.get("minutes", args.strategy_minutes)), time_scale),
        replay_instructions=args.replay_instructions,
        replay_timeout=args.replay_timeout_us,
        max_tasks=int(phase_cfg.get("max_tasks", args.max_tasks)),
        max_tasks_per_root=int(phase_cfg.get("max_tasks_per_root", args.max_tasks_per_root)),
        merge_probe_bbs=int(phase_cfg.get("merge_probe_bbs", args.merge_probe_bbs)),
        relax_root_occurrences=bool(phase_cfg.get("relax_root_occurrences", True)),
        ordered_path_constraints=bool(phase_cfg.get("ordered_path_constraints", False)),
        strict_subtree_pruning=bool(phase_cfg.get("strict_subtree_pruning", False)),
        prioritize_static_potential=bool(phase_cfg.get("prioritize_static_potential", False)),
        skip_stale_tasks=bool(phase_cfg.get("skip_stale_tasks", False)),
        defer_stale_tasks=bool(phase_cfg.get("defer_stale_tasks", True)),
        stale_replay_interval=int(phase_cfg.get("stale_replay_interval", args.stale_replay_interval)),
        state_signature_pruning=bool(phase_cfg.get("state_signature_pruning", True)),
        state_probe_bbs=int(phase_cfg.get("state_probe_bbs", args.state_probe_bbs)),
        edge_saturation_pruning=bool(phase_cfg.get("edge_saturation_pruning", True)),
        edge_zero_new_limit=int(phase_cfg.get("edge_zero_new_limit", args.edge_zero_new_limit)),
        dynamic_frontier_seeding=bool(phase_cfg.get("dynamic_frontier_seeding", True)),
        isolate_path_coverage=bool(phase_cfg.get("isolate_path_coverage", False)),
        target_bbs=getattr(args, "target_bbs", set()) if target_guided else None,
        known_coverage_seed=getattr(args, "known_coverage_seed", set()),
        prioritize_target_bbs=target_guided,
        continue_target_merges=bool(phase_cfg.get("continue_target_merges", target_guided and args.continue_target_merges)),
        target_probe_bbs=int(phase_cfg.get("target_probe_bbs", args.target_probe_bbs)),
        phase_name=str(phase_cfg.get("phase_name", "branch_reservoir")),
        reset_state=bool(phase_cfg.get("reset_state", False)),
    )


def parse_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return None


def normalize_constraint_item(item: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not isinstance(item, dict):
        return None
    address = parse_int(item.get("address"))
    value = parse_int(item.get("value"))
    if address is None or value is None:
        return None
    normalized = dict(item)
    normalized["address"] = f"0x{address & 0xFFFFFFFF:08x}"
    normalized["value"] = f"0x{value & 0xFFFFFFFF:08x}"
    for key in ("read_pc", "pc", "constraint_pc", "branch_pc"):
        parsed = parse_int(normalized.get(key))
        if parsed is not None:
            normalized[key] = f"0x{parsed & 0xFFFFFFFF:08x}"
        elif key in normalized:
            normalized[key] = None
    return normalized


def merge_constraint_files(paths: Iterable[Path], output_path: Path) -> int:
    constraints: List[Dict[str, object]] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            continue
        items = data.get("constraints", []) if isinstance(data, dict) else []
        for item in items:
            normalized = normalize_constraint_item(item)
            if not normalized:
                continue
            key = (
                normalized.get("type"),
                normalized.get("read_pc") or normalized.get("pc"),
                normalized.get("address"),
                normalized.get("constraint_pc"),
            )
            if key in seen:
                continue
            seen.add(key)
            constraints.append(normalized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"constraints": constraints}, f, indent=2)
    return len(constraints)


def append_checkpoint(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def greedy_contribution(strategy_reports: List[Dict[str, object]]) -> List[Dict[str, object]]:
    remaining = [
        {
            "name": report["name"],
            "covered": set(report["covered_bb_list"]),
        }
        for report in strategy_reports
    ]
    covered: Set[int] = set()
    order = []
    while True:
        best = None
        for item in remaining:
            gain = len(item["covered"] - covered)
            if gain and (best is None or gain > best[0]):
                best = (gain, item)
        if best is None:
            break
        gain, item = best
        covered.update(item["covered"])
        order.append({
            "name": item["name"],
            "gain": gain,
            "strategy_bbs": len(item["covered"]),
            "union_bbs": len(covered),
        })
    return order


def run_strategy(
    strategy: Dict[str, object],
    firmware: Path,
    output_dir: Path,
    llm_config,
    llm_config_path,
    args,
) -> Dict[str, object]:
    name = str(strategy["name"])
    strategy_dir = output_dir / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    constraint_file = workspace_artifact_path(strategy_dir, firmware, "_lsgemu_constraints.json")
    seed_constraint_file(firmware, constraint_file)
    trace_file = workspace_artifact_path(strategy_dir, firmware, "_reservoir_execution.log")
    report_file = workspace_artifact_path(strategy_dir, firmware, "_reservoir_test_result.json")
    branch_catalog_file = workspace_artifact_path(strategy_dir, firmware, "_branch_catalog.json")
    progress_file = strategy_dir / "progress.json"
    checkpoint_file = strategy_dir / "coverage_checkpoints.jsonl"
    reservoir_state_file = strategy_dir / "reservoir_state.json"
    target_provenance_file = strategy_dir / "target_provenance.jsonl"

    started = time.time()
    with temporary_env({
        "LSGEMU_RESERVOIR_PROGRESS": str(progress_file),
        "LSGEMU_COVERAGE_CHECKPOINT": str(checkpoint_file),
        "LSGEMU_COVERAGE_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
        "LSGEMU_RESERVOIR_STATE": None if args.no_reservoir_state else str(reservoir_state_file),
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
        isr_reservoir = set()
        isr_minutes = float(strategy.get("isr_reservoir_minutes", 0.0))
        if isr_minutes > 0:
            isr_reservoir = runner.context_recovery.isr_reservoir(
                time_limit_seconds=scaled_seconds(isr_minutes, args.time_scale),
                max_instructions=args.isr_reservoir_instructions,
                replay_timeout=args.replay_timeout_us,
                max_tasks=int(strategy.get("isr_reservoir_max_tasks", args.isr_reservoir_max_tasks)),
                max_tasks_per_isr=int(strategy.get("isr_reservoir_max_tasks_per_isr", args.isr_reservoir_max_tasks_per_isr)),
                dynamic_frontier_seeding=bool(strategy.get("isr_dynamic_frontier_seeding", True)),
            )

        reservoir_union: Set[int] = set()
        phase_summaries = []
        previous_global = len(runner.global_coverage)
        for index, phase_cfg in enumerate(strategy.get("phases", []), start=1):
            covered = run_reservoir_phase(runner, phase_cfg, args, args.time_scale)
            reservoir_union.update(covered)
            current_global = len(runner.global_coverage)
            phase_name = str(phase_cfg.get("phase_name", f"reservoir_{index}"))
            phase_summaries.append({
                "phase": phase_name,
                "phase_bbs": len(covered),
                "global_bbs": current_global,
                "new_global_bbs": current_global - previous_global,
                "config": phase_cfg,
            })
            previous_global = current_global

        contextual_isr = set()
        contextual_contexts = int(strategy.get("contextual_isr_contexts", 0))
        if contextual_contexts != 0:
            contextual_isr = runner.context_recovery.contextual_isr(
                max_contexts=contextual_contexts,
                max_isrs=int(strategy.get("contextual_isr_max_isrs", 0)),
                max_instructions=args.contextual_isr_instructions,
                replay_timeout=args.replay_timeout_us,
                time_limit_seconds=(
                    scaled_seconds(float(strategy.get("contextual_isr_minutes", 0.0)), args.time_scale)
                    if float(strategy.get("contextual_isr_minutes", 0.0)) > 0
                    else None
                ),
                include_reservoir_contexts=bool(
                    strategy.get("contextual_isr_include_reservoir", args.contextual_isr_include_reservoir)
                ),
                max_reservoir_contexts=int(
                    strategy.get("contextual_isr_reservoir_contexts", args.contextual_isr_reservoir_contexts)
                ),
            )

        elapsed = time.time() - started
        branch_catalog = runner.save_branch_catalog(branch_catalog_file)
        static_reachability = runner.static_reachability_summary()
        report = runner.save_report(
            report_file,
            execution_time_seconds=elapsed,
            extra={
                "runner": "gateway_ensemble_cached",
                "strategy": name,
                "strategy_config": strategy,
                "constraint_file": str(constraint_file),
                "branch_catalog_file": str(branch_catalog_file),
                "branch_catalog_summary": {
                    "static_conditional_branch_points": branch_catalog["static_conditional_branch_points"],
                    "main_path_branch_events": branch_catalog["main_path_branch_events"],
                    "main_path_branch_points": branch_catalog["main_path_branch_points"],
                    "reservoir_discovered_branch_events": branch_catalog["reservoir_discovered_branch_events"],
                    "reservoir_new_branch_points": branch_catalog["reservoir_new_branch_points"],
                },
                "static_reachability": static_reachability,
                "phase_summaries": phase_summaries,
                "component_bbs": {
                    "baseline": len(baseline),
                    "isr": len(isr),
                    "isr_reservoir": len(isr_reservoir),
                    "reservoir": len(reservoir_union),
                    "contextual_isr": len(contextual_isr),
                },
                "reservoir_state_file": str(reservoir_state_file) if not args.no_reservoir_state else None,
                "target_provenance_file": str(target_provenance_file),
            },
        )

    return {
        "name": name,
        "output_dir": str(strategy_dir),
        "report_file": str(report_file),
        "constraint_file": str(constraint_file),
        "covered_bbs": report["covered_bbs"],
        "coverage_rate": report["coverage_rate"],
        "execution_time_seconds": elapsed,
        "covered_bb_list": report["covered_bb_list"],
        "phase_summaries": phase_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", default=str(DEFAULT_REAL_GATEWAY))
    parser.add_argument("--profile", choices=sorted(strategy_profiles()), default="balanced")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy names from the selected profile")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Multiply all strategy phase budgets")
    parser.add_argument("--strategy-minutes", type=float, default=5.0, help="Fallback minutes for custom phases")
    parser.add_argument("--baseline-instructions", type=int, default=500000)
    parser.add_argument("--baseline-quiescence-bbs", type=int, default=50000, help="0 disables no-new-BB baseline stop")
    parser.add_argument("--max-snapshots", type=int, default=5)
    parser.add_argument("--isr-instructions", type=int, default=10000)
    parser.add_argument("--isr-reservoir-instructions", type=int, default=20000)
    parser.add_argument("--isr-reservoir-max-tasks", type=int, default=500)
    parser.add_argument("--isr-reservoir-max-tasks-per-isr", type=int, default=64)
    parser.add_argument("--contextual-isr-instructions", type=int, default=5000)
    parser.add_argument("--contextual-isr-include-reservoir", action="store_true", help="Inject ISR from reservoir replay snapshots in addition to main-path snapshots")
    parser.add_argument("--contextual-isr-reservoir-contexts", type=int, default=16)
    parser.add_argument("--replay-instructions", type=int, default=15000)
    parser.add_argument("--replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 or negative means unlimited")
    parser.add_argument("--max-tasks-per-root", type=int, default=0, help="0 or negative means unlimited")
    parser.add_argument("--merge-probe-bbs", type=int, default=128)
    parser.add_argument("--stale-replay-interval", type=int, default=16)
    parser.add_argument("--state-probe-bbs", type=int, default=512)
    parser.add_argument("--edge-zero-new-limit", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--stop-after-zero-gain", type=int, default=0, help="Stop after N consecutive zero-gain strategies; 0 disables")
    parser.add_argument("--no-reservoir-state", action="store_true", help="Disable reservoir queue/state checkpointing")
    parser.add_argument("--resume-reservoir-state", action="store_true", help="Resume reservoir queues from per-strategy reservoir_state.json when available")
    parser.add_argument("--reservoir-isr-context-limit", type=int, default=96)
    parser.add_argument("--target-report", default=None, help="Coverage report whose covered BBs should be prioritized as targets")
    parser.add_argument("--target-exclude-report", default=None, help="Coverage report to subtract from target-report")
    parser.add_argument("--target-all-static", action="store_true", help="Use all statically discovered BBs as targets before subtracting target-exclude-report")
    parser.add_argument("--target-guided", action="store_true", help="Enable target-guided scheduling even for phases without target_guided=True")
    parser.add_argument("--continue-target-merges", action="store_true", help="For target-guided phases, keep probing covered merge BBs that can statically reach targets")
    parser.add_argument("--target-probe-bbs", type=int, default=2048, help="Covered merge BB probe budget per replay when continuing toward targets")
    parser.add_argument("--config", default=None, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument(
        "--output-dir",
        default=str(configured_path("LSGEMU_RUN_OUTPUT_DIR", configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4") / ".lsgemu_runs") / "ensemble"),
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))

    firmware = Path(args.firmware).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_config, llm_config_path = load_llm_config(RUNNER_PROJECT_ROOT)
    target_bbs = load_report_coverage(args.target_report)
    known_coverage_seed = load_report_coverage(args.target_exclude_report)
    if args.target_all_static:
        target_bbs.update(PreparedFirmware.from_firmware(firmware, use_ghidra=True).static_bb_set)
    if args.target_exclude_report:
        target_bbs -= known_coverage_seed
    initial_target_bbs = set(target_bbs)
    args.target_bbs = set(initial_target_bbs)
    args.known_coverage_seed = set(known_coverage_seed)
    strategies = selected_strategies(args.profile, args.strategies)
    started = time.time()
    checkpoint_file = output_dir / "ensemble_coverage_checkpoints.jsonl"
    report_file = workspace_artifact_path(output_dir, firmware, "_ensemble_result.json")
    merged_constraint_file = workspace_artifact_path(output_dir, firmware, "_ensemble_constraints.json")

    strategy_reports: List[Dict[str, object]] = []
    union_coverage: Set[int] = set()
    total_bbs = 0
    zero_gain_streak = 0

    for index, strategy in enumerate(strategies, start=1):
        name = str(strategy["name"])
        logging.warning("[Ensemble] strategy %d/%d: %s", index, len(strategies), name)
        before = set(union_coverage)
        args.target_bbs = initial_target_bbs - union_coverage
        strategy_report = run_strategy(strategy, firmware, output_dir, llm_config, llm_config_path, args)
        covered = set(strategy_report["covered_bb_list"])
        union_coverage.update(covered)
        total_bbs = max(total_bbs, int(round(strategy_report["covered_bbs"] / max(strategy_report["coverage_rate"], 0.0001) * 100)))
        gain = len(union_coverage - before)
        strategy_report["union_gain"] = gain
        strategy_report["union_bbs_after"] = len(union_coverage)
        strategy_reports.append(strategy_report)
        zero_gain_streak = zero_gain_streak + 1 if gain == 0 else 0
        append_checkpoint(checkpoint_file, {
            "status": "running",
            "strategy": name,
            "strategy_index": index,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - started,
            "strategy_bbs": strategy_report["covered_bbs"],
            "strategy_gain": gain,
            "union_bbs": len(union_coverage),
            "total_bbs": total_bbs,
            "coverage_rate": (len(union_coverage) / total_bbs * 100) if total_bbs else 0.0,
        })
        if args.stop_after_zero_gain > 0 and zero_gain_streak >= args.stop_after_zero_gain:
            break

    merged_constraints = merge_constraint_files(
        [Path(report["constraint_file"]) for report in strategy_reports],
        merged_constraint_file,
    )

    static_reachability = {}
    if strategy_reports:
        summary_runner = HistoricalRunner(
            firmware,
            max_snapshots=args.max_snapshots,
            use_llm=False,
            constraint_file=str(merged_constraint_file),
        )
        summary_runner.global_coverage = set(union_coverage)
        static_reachability = summary_runner.static_reachability_summary()
        total_bbs = summary_runner.prepared.total_bbs

    occurrence_counter: Dict[int, int] = {}
    for report in strategy_reports:
        for bb in report["covered_bb_list"]:
            occurrence_counter[int(bb)] = occurrence_counter.get(int(bb), 0) + 1
    for report in strategy_reports:
        covered = set(report["covered_bb_list"])
        report["absolute_unique_bbs"] = len([bb for bb in covered if occurrence_counter[int(bb)] == 1])
        # Keep top-level report compact; per-strategy report files contain the full list.
        report.pop("covered_bb_list", None)

    elapsed = time.time() - started
    result = {
        "runner": "gateway_ensemble_cached",
        "firmware": str(firmware),
        "profile": args.profile,
        "time_scale": args.time_scale,
        "total_bbs": total_bbs,
        "covered_bbs": len(union_coverage),
        "coverage_rate": (len(union_coverage) / total_bbs * 100) if total_bbs else 0.0,
        "execution_time_seconds": elapsed,
        "covered_bb_list": sorted(union_coverage),
        "target_report": args.target_report,
        "target_exclude_report": args.target_exclude_report,
        "target_all_static": args.target_all_static,
        "known_coverage_seed_bbs": len(known_coverage_seed),
        "target_bbs": len(initial_target_bbs),
        "target_covered_bbs": len(initial_target_bbs & union_coverage),
        "target_remaining_bbs": len(initial_target_bbs - union_coverage),
        "strategy_reports": strategy_reports,
        "greedy_strategy_order": greedy_contribution([
            {
                "name": report["name"],
                "covered_bb_list": json.load(open(report["report_file"]))["covered_bb_list"],
            }
            for report in strategy_reports
        ]),
        "static_reachability": static_reachability,
        "merged_constraint_file": str(merged_constraint_file),
        "merged_constraints": merged_constraints,
        "checkpoint_file": str(checkpoint_file),
    }
    with report_file.open("w") as f:
        json.dump(result, f, indent=2)
    append_checkpoint(checkpoint_file, {
        "status": "complete",
        "timestamp": time.time(),
        "elapsed_seconds": elapsed,
        "union_bbs": len(union_coverage),
        "total_bbs": total_bbs,
        "coverage_rate": result["coverage_rate"],
    })

    print(f"profile={args.profile}")
    print(f"strategies_run={len(strategy_reports)}")
    print(f"covered_bbs={result['covered_bbs']}")
    print(f"total_bbs={result['total_bbs']}")
    print(f"coverage_rate={result['coverage_rate']:.2f}")
    print(f"entry_plus_vector_reachable_coverage_rate={static_reachability.get('entry_plus_vector_reachable_coverage_rate', 0.0):.2f}")
    print(f"merged_constraints={merged_constraints}")
    print(f"report={report_file}")


if __name__ == "__main__":
    main()
