#!/usr/bin/env python3
"""
Fast Gateway reservoir test.

This script uses HistoricalRunner's static-analysis cache and only runs:
1. baseline
2. ISR exploration
3. reservoir branch exploration

It is intended for iteration on path replay without paying Ghidra cost when the
firmware is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from lsgemu.constraint_utils import merge_learned_constraints_into, seed_constraint_file
    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import HistoricalRunner, DEFAULT_REAL_GATEWAY, PROJECT_ROOT as RUNNER_PROJECT_ROOT
else:
    from lsgemu.constraint_utils import merge_learned_constraints_into, seed_constraint_file
    from lsgemu.deployment_config import apply_config_from_argv, configured_path
    from lsgemu.historical_runner import HistoricalRunner, DEFAULT_REAL_GATEWAY, PROJECT_ROOT as RUNNER_PROJECT_ROOT

apply_config_from_argv()


def load_llm_config(project_root: Path):
    path = configured_path("LSGEMU_LLM_CONFIG", project_root / "LLM.yaml")
    if not path.exists():
        return None, None
    with path.open() as f:
        return yaml.safe_load(f), path


def workspace_artifact_path(output_dir: Path, firmware: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{firmware.stem}{suffix}"


def diverse_pass_configs():
    return [
        {
            "name": "diverse_static_potential",
            "prioritize_static_potential": True,
            "stale_replay_interval": 16,
        },
        {
            "name": "diverse_ordered_constraints",
            "ordered_path_constraints": True,
            "stale_replay_interval": 16,
        },
        {
            "name": "diverse_exact_occurrence",
            "relax_root_occurrences": False,
            "stale_replay_interval": 16,
        },
        {
            "name": "diverse_isolated_path",
            "isolate_path_coverage": True,
            "stale_replay_interval": 16,
        },
        {
            "name": "diverse_stale64",
            "stale_replay_interval": 64,
        },
        {
            "name": "diverse_no_state_signature",
            "state_signature_pruning": False,
            "merge_probe_bbs": 64,
            "state_probe_bbs": 0,
            "stale_replay_interval": 16,
        },
    ]


def targeted_phase_name(base_name: str, cycle_index: int, round_index: int) -> str:
    if cycle_index <= 0:
        return base_name if round_index <= 0 else f"{base_name}_round_{round_index + 1}"
    cycle_name = f"{base_name}_cycle_{cycle_index + 1}"
    return cycle_name if round_index <= 0 else f"{cycle_name}_round_{round_index + 1}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", default=str(DEFAULT_REAL_GATEWAY))
    parser.add_argument("--time", type=int, default=5, help="Reservoir budget in minutes")
    parser.add_argument("--baseline-instructions", type=int, default=500000)
    parser.add_argument("--baseline-quiescence-bbs", type=int, default=50000, help="0 disables no-new-BB baseline stop")
    parser.add_argument("--isr-instructions", type=int, default=10000)
    parser.add_argument("--contextual-isr-contexts", type=int, default=0, help="Main-path branch snapshot contexts for ISR injection; 0 disables")
    parser.add_argument("--contextual-isr-include-reservoir", action="store_true", help="Also use productive reservoir replay snapshots as ISR injection contexts")
    parser.add_argument("--contextual-isr-reservoir-contexts", type=int, default=16, help="Max productive reservoir contexts to include in contextual ISR phases; 0 disables reservoir contexts")
    parser.add_argument("--contextual-isr-max-isrs", type=int, default=0, help="0 or negative means all vector ISRs")
    parser.add_argument("--contextual-isr-instructions", type=int, default=10000)
    parser.add_argument("--contextual-isr-time", type=int, default=0, help="Contextual ISR budget in minutes; 0 means no separate time limit")
    parser.add_argument("--contextual-isr-reservoir-time", type=int, default=0, help="Contextual ISR reservoir budget in minutes; 0 disables")
    parser.add_argument("--contextual-isr-reservoir-instructions", type=int, default=15000)
    parser.add_argument("--contextual-isr-reservoir-max-tasks", type=int, default=500, help="0 or negative means unlimited")
    parser.add_argument("--contextual-isr-reservoir-max-tasks-per-isr", type=int, default=64, help="0 or negative means unlimited")
    parser.add_argument("--no-contextual-isr-reservoir-dynamic-frontier-seeding", action="store_true", help="Disable ISR-local dynamic frontier enqueueing for contextual ISR reservoir")
    parser.add_argument("--contextual-isr-frontier-time", type=int, default=0, help="Optional target-guided contextual ISR frontier pass in minutes; 0 disables")
    parser.add_argument("--contextual-isr-frontier-rounds", type=int, default=1, help="Repeat the contextual ISR frontier pass this many times")
    parser.add_argument("--contextual-isr-frontier-max-targets", type=int, default=0, help="0 or negative means all current frontier BBs for contextual ISR frontier")
    parser.add_argument("--contextual-isr-frontier-min-uncovered-successors", type=int, default=1, help="Only use frontier predecessors with at least this many uncovered successors for contextual ISR frontier")
    parser.add_argument("--contextual-isr-frontier-switch-only", action="store_true", help="Only target TBB/TBH frontier successors in the contextual ISR frontier pass")
    parser.add_argument("--frontier-targeted-time", type=int, default=0, help="Post-pass frontier-targeted reservoir budget in minutes; 0 disables")
    parser.add_argument("--frontier-targeted-rounds", type=int, default=0, help="Repeat frontier-targeted passes this many times; 0 means auto up to --frontier-targeted-max-rounds")
    parser.add_argument("--frontier-targeted-max-rounds", type=int, default=16, help="Auto-mode cap when --frontier-targeted-rounds is 0 or negative")
    parser.add_argument("--frontier-targeted-stale-rounds", type=int, default=2, help="Stop frontier-targeted auto rounds after this many consecutive rounds with no new BBs and no new target BBs")
    parser.add_argument("--frontier-targeted-max-targets", type=int, default=0, help="0 or negative means all immediate uncovered frontier BBs")
    parser.add_argument("--frontier-targeted-min-uncovered-successors", type=int, default=1, help="Only use frontier predecessors with at least this many uncovered successors")
    parser.add_argument("--frontier-targeted-min-progress-bbs", type=int, default=1, help="Rounds below this new-BB and target-discovery count are treated as stale")
    parser.add_argument("--frontier-targeted-switch-only", action="store_true", help="Only target TBB/TBH frontier successors in the post-pass")
    parser.add_argument("--switch-frontier-time", type=int, default=0, help="Optional high-fanout switch-only frontier stage before general frontier targeting, in minutes")
    parser.add_argument("--switch-frontier-rounds", type=int, default=0, help="Repeat the switch-only frontier stage this many times; 0 means auto up to --switch-frontier-max-rounds")
    parser.add_argument("--switch-frontier-max-rounds", type=int, default=8, help="Auto-mode cap when --switch-frontier-rounds is 0 or negative")
    parser.add_argument("--switch-frontier-stale-rounds", type=int, default=1, help="Stop switch-frontier auto rounds after this many consecutive rounds with no new BBs and no new target BBs")
    parser.add_argument("--switch-frontier-min-uncovered-successors", type=int, default=2, help="Only use switch frontier predecessors with at least this many uncovered successors")
    parser.add_argument("--switch-frontier-min-progress-bbs", type=int, default=1, help="Switch rounds below this new-BB and target-discovery count are treated as stale")
    parser.add_argument("--switch-frontier-max-targets", type=int, default=0, help="0 or negative means all switch-only frontier targets")
    parser.add_argument("--frontier-cycle-max-cycles", type=int, default=3, help="Repeat switch/general frontier draining cycles up to this many times in auto mode")
    parser.add_argument("--frontier-cycle-stale-cycles", type=int, default=1, help="Stop auto frontier cycling after this many full cycles with no new BBs and no discovered target BBs")
    parser.add_argument("--isr-reservoir-time", type=int, default=0, help="ISR reservoir budget in minutes; 0 disables")
    parser.add_argument("--isr-reservoir-instructions", type=int, default=20000)
    parser.add_argument("--isr-reservoir-max-tasks", type=int, default=500, help="0 or negative means unlimited")
    parser.add_argument("--isr-reservoir-max-tasks-per-isr", type=int, default=64, help="0 or negative means unlimited")
    parser.add_argument("--no-isr-dynamic-frontier-seeding", action="store_true", help="Disable ISR-local dynamic frontier enqueueing")
    parser.add_argument("--replay-instructions", type=int, default=15000)
    parser.add_argument("--replay-timeout-us", type=int, default=1000000)
    parser.add_argument("--max-tasks", type=int, default=1000, help="0 or negative means unlimited")
    parser.add_argument("--max-tasks-per-root", type=int, default=0, help="0 or negative means unlimited")
    parser.add_argument("--merge-probe-bbs", type=int, default=128, help="Covered BBs to probe after a new subtree merges back")
    parser.add_argument("--exact-root-occurrences", action="store_true", help="Force root branches at their exact recorded occurrence instead of first address hit")
    parser.add_argument("--ordered-path-constraints", action="store_true", help="Apply all path choices as ordered address-level constraints")
    parser.add_argument("--strict-subtree-pruning", action="store_true", help="Use the deepest requested branch when deciding stale path skips")
    parser.add_argument("--prioritize-static-potential", action="store_true", help="Sort child tasks by static reachable uncovered BB potential")
    parser.add_argument("--skip-stale-tasks", action="store_true", help="Skip paths whose current terminal successor is already covered")
    parser.add_argument("--no-defer-stale-tasks", action="store_true", help="Replay stale paths immediately instead of after fresh paths")
    parser.add_argument("--stale-replay-interval", type=int, default=16, help="Replay one deferred stale path after this many fresh path tasks")
    parser.add_argument("--no-state-signature-pruning", action="store_true", help="Disable dynamic state signature pruning after forced branches")
    parser.add_argument("--state-probe-bbs", type=int, default=512, help="Max known-covered BBs with new state to probe after a forced branch")
    parser.add_argument("--no-edge-saturation-pruning", action="store_true", help="Disable pruning of repeatedly unproductive terminal branch edges")
    parser.add_argument("--edge-zero-new-limit", type=int, default=3, help="Zero-new attempts before a stale terminal edge is considered saturated")
    parser.add_argument("--no-dynamic-frontier-seeding", action="store_true", help="Disable enqueueing alternate edges from dynamically observed replay branches")
    parser.add_argument("--isolate-path-coverage", action="store_true", help="Ignore ISR/other phase coverage when pruning main reservoir paths")
    parser.add_argument("--diverse-passes", type=int, default=0, help="Run N additional reservoir passes with diverse scheduling/pruning strategies")
    parser.add_argument("--diverse-pass-time", type=int, default=5, help="Budget in minutes for each diverse reservoir pass")
    parser.add_argument("--diverse-pass-max-tasks", type=int, default=0, help="0 or negative means unlimited per diverse pass")
    parser.add_argument("--final-reservoir-time", type=int, default=0, help="Optional final default reservoir pass after diverse passes, in minutes")
    parser.add_argument("--final-reservoir-max-tasks", type=int, default=0, help="0 or negative means unlimited for the final pass")
    parser.add_argument("--final-isolate-path-coverage", action="store_true", help="Ignore global coverage when seeding/pruning the final reservoir pass")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--config", default=None, help="Deployment YAML/JSON config. Also accepted through LSGEMU_CONFIG_FILE.")
    parser.add_argument("--output-dir", default=str(configured_path("LSGEMU_RUN_OUTPUT_DIR", configured_path("LSGEMU_SOURCE_ROOT", RUNNER_PROJECT_ROOT / "srcv4") / ".lsgemu_runs")))
    parser.add_argument("--constraint-file", default=None)
    parser.add_argument("--constraint-seed-mode", choices=["none", "best", "union"], default="none", help="How to seed a new constraint file from previous runs for the same firmware")
    parser.add_argument("--constraint-seed-activation", choices=["initial", "after-main"], default="after-main", help="When learned constraints become active; after-main preserves baseline path discovery")
    parser.add_argument("--no-merge-learned-constraints", action="store_true", help="Alias for --constraint-seed-mode none")
    parser.add_argument("--constraint-seed-root", action="append", default=[], help="Additional directory to search recursively for learned constraint seeds")
    parser.add_argument("--max-wall-time", type=int, default=0, help="Total script wall-clock budget in minutes; 0 disables")
    parser.add_argument("--wall-time-reserve", type=int, default=60, help="Seconds reserved for final analysis/report writing when --max-wall-time is set")
    parser.add_argument("--targeted-snapshot-retry-budget", type=int, default=None, help="Set LSGEMU_TARGETED_SNAPSHOT_RETRY_BUDGET for each targeted reservoir phase")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.WARNING))
    if args.targeted_snapshot_retry_budget is not None:
        os.environ["LSGEMU_TARGETED_SNAPSHOT_RETRY_BUDGET"] = str(args.targeted_snapshot_retry_budget)

    firmware = Path(args.firmware).resolve()
    llm_config, llm_config_path = load_llm_config(RUNNER_PROJECT_ROOT)
    output_dir = Path(args.output_dir).resolve()
    constraint_file = (
        Path(args.constraint_file).resolve()
        if args.constraint_file
        else workspace_artifact_path(output_dir, firmware, "_lsgemu_constraints.json")
    )
    learned_roots = [Path(item).resolve() for item in args.constraint_seed_root]
    learned_mode = "none" if args.no_merge_learned_constraints or args.constraint_file else args.constraint_seed_mode
    if learned_mode != "none":
        learned_roots.append(output_dir.parent)
    initial_seed_mode = learned_mode if args.constraint_seed_activation == "initial" else "none"
    seeded_constraint_count = seed_constraint_file(
        firmware,
        constraint_file,
        learned_mode=initial_seed_mode,
        learned_roots=learned_roots,
    )
    trace_file = workspace_artifact_path(output_dir, firmware, "_reservoir_execution.log")
    report_file = workspace_artifact_path(output_dir, firmware, "_reservoir_test_result.json")
    branch_catalog_file = workspace_artifact_path(output_dir, firmware, "_branch_catalog.json")

    start = time.time()
    wall_deadline = start + args.max_wall_time * 60 if args.max_wall_time and args.max_wall_time > 0 else None
    wall_budget_exhausted = False

    def remaining_wall_seconds() -> float:
        if wall_deadline is None:
            return float("inf")
        return wall_deadline - time.time() - max(0, args.wall_time_reserve)

    def bounded_time_limit(requested_seconds):
        nonlocal wall_budget_exhausted
        if requested_seconds is None:
            requested = float("inf")
        else:
            requested = max(0.0, float(requested_seconds))
        remaining = remaining_wall_seconds()
        if remaining <= 0:
            wall_budget_exhausted = True
            return 0
        if requested == float("inf"):
            return remaining if wall_deadline is not None else None
        return max(1, int(min(requested, remaining)))

    def can_start_phase(min_seconds: int = 1) -> bool:
        nonlocal wall_budget_exhausted
        if remaining_wall_seconds() < max(1, min_seconds):
            wall_budget_exhausted = True
            return False
        return True

    runner = HistoricalRunner(
        firmware,
        max_snapshots=5,
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
    if learned_mode != "none" and args.constraint_seed_activation == "after-main":
        seeded_constraint_count = merge_learned_constraints_into(
            firmware,
            constraint_file,
            learned_mode=learned_mode,
            learned_roots=learned_roots,
        )
    isr_reservoir = set()
    if args.isr_reservoir_time > 0 and can_start_phase():
        phase_limit = bounded_time_limit(args.isr_reservoir_time * 60)
        isr_reservoir = runner.context_recovery.isr_reservoir(
            time_limit_seconds=phase_limit,
            max_instructions=args.isr_reservoir_instructions,
            replay_timeout=args.replay_timeout_us,
            max_tasks=args.isr_reservoir_max_tasks,
            max_tasks_per_isr=args.isr_reservoir_max_tasks_per_isr,
            dynamic_frontier_seeding=not args.no_isr_dynamic_frontier_seeding,
        )
    contextual_isr = set()
    contextual_isr_reservoir = set()
    switch_frontier_targeted = set()
    switch_frontier_rounds = []
    frontier_targeted = set()
    frontier_target_rounds = []
    frontier_cycles = []
    contextual_isr_frontier_targeted = set()
    contextual_isr_frontier_rounds = []
    reservoir = set()
    reservoir_rounds = []
    diverse_rounds = []
    previous_global = len(runner.global_coverage)
    for round_index in range(args.rounds):
        if not can_start_phase():
            break
        phase_limit = bounded_time_limit(args.time * 60)
        if phase_limit == 0:
            break
        round_coverage = runner.run_reservoir_branch_exploration(
            time_limit_seconds=phase_limit,
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
            phase_name="branch_reservoir" if round_index == 0 else f"branch_reservoir_round_{round_index + 1}",
            reset_state=(round_index == 0),
        )
        reservoir.update(round_coverage)
        current_global = len(runner.global_coverage)
        reservoir_rounds.append({
            "round": round_index + 1,
            "round_bbs": len(round_coverage),
            "global_bbs": current_global,
            "new_global_bbs": current_global - previous_global,
        })
        previous_global = current_global

    configs = diverse_pass_configs()
    for pass_index, config in enumerate(configs[:max(0, args.diverse_passes)]):
        if not can_start_phase():
            break
        phase_name = config["name"]
        phase_limit = bounded_time_limit(args.diverse_pass_time * 60)
        if phase_limit == 0:
            break
        pass_coverage = runner.run_reservoir_branch_exploration(
            time_limit_seconds=phase_limit,
            replay_instructions=args.replay_instructions,
            replay_timeout=args.replay_timeout_us,
            max_tasks=args.diverse_pass_max_tasks,
            max_tasks_per_root=args.max_tasks_per_root,
            merge_probe_bbs=config.get("merge_probe_bbs", args.merge_probe_bbs),
            relax_root_occurrences=config.get("relax_root_occurrences", not args.exact_root_occurrences),
            ordered_path_constraints=config.get("ordered_path_constraints", args.ordered_path_constraints),
            strict_subtree_pruning=args.strict_subtree_pruning,
            prioritize_static_potential=config.get("prioritize_static_potential", args.prioritize_static_potential),
            skip_stale_tasks=args.skip_stale_tasks,
            defer_stale_tasks=not args.no_defer_stale_tasks,
            stale_replay_interval=config.get("stale_replay_interval", args.stale_replay_interval),
            state_signature_pruning=config.get("state_signature_pruning", not args.no_state_signature_pruning),
            state_probe_bbs=config.get("state_probe_bbs", args.state_probe_bbs),
            edge_saturation_pruning=not args.no_edge_saturation_pruning,
            edge_zero_new_limit=args.edge_zero_new_limit,
            dynamic_frontier_seeding=not args.no_dynamic_frontier_seeding,
            isolate_path_coverage=config.get("isolate_path_coverage", args.isolate_path_coverage),
            phase_name=phase_name,
            reset_state=True,
        )
        reservoir.update(pass_coverage)
        current_global = len(runner.global_coverage)
        diverse_rounds.append({
            "round": pass_index + 1,
            "phase": phase_name,
            "round_bbs": len(pass_coverage),
            "global_bbs": current_global,
            "new_global_bbs": current_global - previous_global,
            "config": config,
        })
        previous_global = current_global

    if args.final_reservoir_time > 0 and can_start_phase():
        phase_limit = bounded_time_limit(args.final_reservoir_time * 60)
        final_coverage = runner.run_reservoir_branch_exploration(
            time_limit_seconds=phase_limit,
            replay_instructions=args.replay_instructions,
            replay_timeout=args.replay_timeout_us,
            max_tasks=args.final_reservoir_max_tasks,
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
            isolate_path_coverage=args.final_isolate_path_coverage,
            phase_name="branch_reservoir_final",
            reset_state=True,
        )
        reservoir.update(final_coverage)
        current_global = len(runner.global_coverage)
        diverse_rounds.append({
            "round": len(diverse_rounds) + 1,
            "phase": "branch_reservoir_final",
            "round_bbs": len(final_coverage),
            "global_bbs": current_global,
            "new_global_bbs": current_global - previous_global,
            "config": {"name": "branch_reservoir_final", "time": args.final_reservoir_time},
        })
        previous_global = current_global

    contextual_isr_enabled = args.contextual_isr_contexts != 0 or (
        args.contextual_isr_include_reservoir and args.contextual_isr_reservoir_contexts != 0
    )
    if contextual_isr_enabled and can_start_phase():
        phase_limit = bounded_time_limit(args.contextual_isr_time * 60 if args.contextual_isr_time > 0 else None)
        contextual_isr = runner.context_recovery.contextual_isr(
            max_contexts=args.contextual_isr_contexts,
            max_isrs=args.contextual_isr_max_isrs,
            max_instructions=args.contextual_isr_instructions,
            replay_timeout=args.replay_timeout_us,
            time_limit_seconds=phase_limit,
            include_reservoir_contexts=args.contextual_isr_include_reservoir,
            max_reservoir_contexts=args.contextual_isr_reservoir_contexts,
        )
    if args.contextual_isr_reservoir_time > 0 and contextual_isr_enabled and can_start_phase():
        phase_limit = bounded_time_limit(args.contextual_isr_reservoir_time * 60)
        contextual_isr_reservoir = runner.context_recovery.isr_reservoir(
            time_limit_seconds=phase_limit,
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
    target_probe_bbs = min(args.state_probe_bbs, 512) if args.state_probe_bbs and args.state_probe_bbs > 0 else 512

    def run_frontier_stage(
        *,
        base_name: str,
        cycle_index: int,
        time_minutes: int,
        requested_rounds: int,
        auto_max_rounds: int,
        stale_rounds_limit: int,
        min_progress_bbs: int,
        min_uncovered_successors: int,
        switch_only: bool,
        max_targets: int,
        coverage_accumulator: set,
        round_records: list,
    ):
        initial_targets = set()
        seen_targets = set()
        stale_rounds = 0
        total_new_bbs = 0
        total_target_hits = 0
        total_discovered_targets = 0
        total_new_candidate_targets = 0
        total_new_branch_events = 0
        total_new_snapshot_variants = 0
        total_priority_root_tasks_seeded = 0
        total_dynamic_successor_edges = 0
        executed_rounds = 0
        last_round_new_bbs = 0
        last_round_discovered_targets = 0
        last_round_new_candidate_targets = 0
        last_round_new_branch_events = 0
        last_round_new_snapshot_variants = 0
        last_round_priority_root_tasks_seeded = 0
        last_round_dynamic_successor_edges = 0
        round_limit = requested_rounds if requested_rounds > 0 else max(1, auto_max_rounds)
        for round_index in range(max(1, round_limit)):
            if not can_start_phase():
                break
            current_targets = runner.frontier_target_bbs(
                min_uncovered_successors=max(1, min_uncovered_successors),
                switch_only=switch_only,
                max_targets=max_targets,
            )
            if round_index == 0:
                initial_targets = set(current_targets)
            seen_targets.update(current_targets)
            if not current_targets:
                break
            phase_name = targeted_phase_name(base_name, cycle_index, round_index)
            phase_limit = bounded_time_limit(time_minutes * 60)
            if phase_limit == 0:
                break
            round_coverage = runner.run_reservoir_branch_exploration(
                time_limit_seconds=phase_limit,
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
                target_bbs=current_targets,
                prioritize_target_bbs=True,
                continue_target_merges=True,
                target_probe_bbs=target_probe_bbs,
                phase_name=phase_name,
                reset_state=True,
            )
            coverage_accumulator.update(round_coverage)
            phase_meta = runner.phase_metadata.get(phase_name, {})
            new_bbs = phase_meta.get("new_bbs", 0)
            target_hits = phase_meta.get("target_hit_tasks", 0)
            discovered_targets = phase_meta.get("target_bbs_discovered", 0)
            remembered_branch_events = phase_meta.get("remembered_branch_events", 0)
            remembered_snapshot_variants = phase_meta.get("remembered_branch_root_snapshots", 0)
            priority_root_tasks_seeded = phase_meta.get("priority_root_tasks_seeded", 0)
            dynamic_successor_edges = phase_meta.get("dynamic_successor_edges_added", 0)
            next_targets = set(runner.frontier_target_bbs(
                min_uncovered_successors=max(1, min_uncovered_successors),
                switch_only=switch_only,
                max_targets=max_targets,
            ))
            new_candidate_targets = len(next_targets - seen_targets)
            seen_targets.update(next_targets)
            last_round_new_bbs = new_bbs
            last_round_discovered_targets = discovered_targets
            last_round_new_candidate_targets = new_candidate_targets
            last_round_new_branch_events = remembered_branch_events
            last_round_new_snapshot_variants = remembered_snapshot_variants
            last_round_priority_root_tasks_seeded = priority_root_tasks_seeded
            last_round_dynamic_successor_edges = dynamic_successor_edges
            total_new_bbs += new_bbs
            total_target_hits += target_hits
            total_discovered_targets += discovered_targets
            total_new_candidate_targets += new_candidate_targets
            total_new_branch_events += remembered_branch_events
            total_new_snapshot_variants += remembered_snapshot_variants
            total_priority_root_tasks_seeded += priority_root_tasks_seeded
            total_dynamic_successor_edges += dynamic_successor_edges
            executed_rounds += 1
            round_records.append({
                "cycle": cycle_index + 1,
                "round": round_index + 1,
                "phase": phase_name,
                "candidate_targets": len(current_targets),
                "covered_bbs": len(round_coverage),
                "new_bbs": new_bbs,
                "target_hit_tasks": target_hits,
                "target_bbs_discovered": discovered_targets,
                "new_candidate_targets": new_candidate_targets,
                "remembered_branch_events": remembered_branch_events,
                "remembered_branch_root_snapshots": remembered_snapshot_variants,
                "priority_root_tasks_seeded": priority_root_tasks_seeded,
                "dynamic_successor_edges_added": dynamic_successor_edges,
                "target_remaining_bbs": phase_meta.get("target_remaining_bbs", 0),
                "preferred_target_roots": phase_meta.get("preferred_target_roots", 0),
            })
            progress_threshold = max(1, min_progress_bbs)
            if (
                new_bbs < progress_threshold
                and discovered_targets < progress_threshold
                and new_candidate_targets < progress_threshold
                and remembered_branch_events <= 0
                and remembered_snapshot_variants <= 0
                and priority_root_tasks_seeded <= 0
                and dynamic_successor_edges <= 0
            ):
                stale_rounds += 1
            else:
                stale_rounds = 0
            if stale_rounds >= max(1, stale_rounds_limit):
                break
        remaining_targets = runner.frontier_target_bbs(
            min_uncovered_successors=max(1, min_uncovered_successors),
            switch_only=switch_only,
            max_targets=max_targets,
        )
        return {
            "initial_targets": initial_targets,
            "remaining_targets": set(remaining_targets),
            "rounds_executed": executed_rounds,
            "new_bbs": total_new_bbs,
            "target_hit_tasks": total_target_hits,
            "target_bbs_discovered": total_discovered_targets,
            "new_candidate_targets": total_new_candidate_targets,
            "remembered_branch_events": total_new_branch_events,
            "remembered_branch_root_snapshots": total_new_snapshot_variants,
            "priority_root_tasks_seeded": total_priority_root_tasks_seeded,
            "dynamic_successor_edges_added": total_dynamic_successor_edges,
            "last_round_new_bbs": last_round_new_bbs,
            "last_round_target_bbs_discovered": last_round_discovered_targets,
            "last_round_new_candidate_targets": last_round_new_candidate_targets,
            "last_round_branch_events": last_round_new_branch_events,
            "last_round_snapshot_variants": last_round_new_snapshot_variants,
            "last_round_priority_root_tasks_seeded": last_round_priority_root_tasks_seeded,
            "last_round_dynamic_successor_edges_added": last_round_dynamic_successor_edges,
        }

    switch_frontier_targets = set()
    frontier_targets = set()
    switch_frontier_enabled = args.switch_frontier_time > 0 and not args.frontier_targeted_switch_only
    frontier_targeted_enabled = args.frontier_targeted_time > 0
    frontier_cycle_auto = (
        (switch_frontier_enabled and args.switch_frontier_rounds <= 0)
        or (frontier_targeted_enabled and args.frontier_targeted_rounds <= 0)
    )
    frontier_cycle_limit = 1
    if frontier_cycle_auto and (switch_frontier_enabled or frontier_targeted_enabled):
        frontier_cycle_limit = max(1, args.frontier_cycle_max_cycles)

    frontier_stale_cycles = 0
    for cycle_index in range(frontier_cycle_limit):
        if not can_start_phase():
            break
        cycle_global_before = len(runner.global_coverage)
        cycle_switch_summary = {
            "initial_targets": set(),
            "remaining_targets": set(),
            "rounds_executed": 0,
            "new_bbs": 0,
            "target_hit_tasks": 0,
            "target_bbs_discovered": 0,
            "new_candidate_targets": 0,
            "remembered_branch_events": 0,
            "remembered_branch_root_snapshots": 0,
            "priority_root_tasks_seeded": 0,
            "dynamic_successor_edges_added": 0,
            "last_round_new_bbs": 0,
            "last_round_target_bbs_discovered": 0,
            "last_round_new_candidate_targets": 0,
            "last_round_branch_events": 0,
            "last_round_snapshot_variants": 0,
            "last_round_priority_root_tasks_seeded": 0,
            "last_round_dynamic_successor_edges_added": 0,
        }
        cycle_frontier_summary = {
            "initial_targets": set(),
            "remaining_targets": set(),
            "rounds_executed": 0,
            "new_bbs": 0,
            "target_hit_tasks": 0,
            "target_bbs_discovered": 0,
            "new_candidate_targets": 0,
            "remembered_branch_events": 0,
            "remembered_branch_root_snapshots": 0,
            "priority_root_tasks_seeded": 0,
            "dynamic_successor_edges_added": 0,
            "last_round_new_bbs": 0,
            "last_round_target_bbs_discovered": 0,
            "last_round_new_candidate_targets": 0,
            "last_round_branch_events": 0,
            "last_round_snapshot_variants": 0,
            "last_round_priority_root_tasks_seeded": 0,
            "last_round_dynamic_successor_edges_added": 0,
        }
        if switch_frontier_enabled:
            cycle_switch_summary = run_frontier_stage(
                base_name="switch_frontier_targeted",
                cycle_index=cycle_index,
                time_minutes=args.switch_frontier_time,
                requested_rounds=args.switch_frontier_rounds,
                auto_max_rounds=args.switch_frontier_max_rounds,
                stale_rounds_limit=args.switch_frontier_stale_rounds,
                min_progress_bbs=args.switch_frontier_min_progress_bbs,
                min_uncovered_successors=args.switch_frontier_min_uncovered_successors,
                switch_only=True,
                max_targets=args.switch_frontier_max_targets,
                coverage_accumulator=switch_frontier_targeted,
                round_records=switch_frontier_rounds,
            )
            if cycle_index == 0:
                switch_frontier_targets = set(cycle_switch_summary["initial_targets"])
        if frontier_targeted_enabled:
            cycle_frontier_summary = run_frontier_stage(
                base_name="frontier_targeted",
                cycle_index=cycle_index,
                time_minutes=args.frontier_targeted_time,
                requested_rounds=args.frontier_targeted_rounds,
                auto_max_rounds=args.frontier_targeted_max_rounds,
                stale_rounds_limit=args.frontier_targeted_stale_rounds,
                min_progress_bbs=args.frontier_targeted_min_progress_bbs,
                min_uncovered_successors=args.frontier_targeted_min_uncovered_successors,
                switch_only=args.frontier_targeted_switch_only,
                max_targets=args.frontier_targeted_max_targets,
                coverage_accumulator=frontier_targeted,
                round_records=frontier_target_rounds,
            )
            if cycle_index == 0:
                frontier_targets = set(cycle_frontier_summary["initial_targets"])
        cycle_global_after = len(runner.global_coverage)
        cycle_new_bbs = cycle_global_after - cycle_global_before
        cycle_targets_discovered = (
            cycle_switch_summary["target_bbs_discovered"]
            + cycle_frontier_summary["target_bbs_discovered"]
        )
        cycle_new_candidate_targets = (
            cycle_switch_summary["new_candidate_targets"]
            + cycle_frontier_summary["new_candidate_targets"]
        )
        cycle_new_branch_events = (
            cycle_switch_summary["remembered_branch_events"]
            + cycle_frontier_summary["remembered_branch_events"]
        )
        cycle_new_snapshot_variants = (
            cycle_switch_summary["remembered_branch_root_snapshots"]
            + cycle_frontier_summary["remembered_branch_root_snapshots"]
        )
        cycle_priority_root_tasks_seeded = (
            cycle_switch_summary["priority_root_tasks_seeded"]
            + cycle_frontier_summary["priority_root_tasks_seeded"]
        )
        cycle_dynamic_successor_edges = (
            cycle_switch_summary["dynamic_successor_edges_added"]
            + cycle_frontier_summary["dynamic_successor_edges_added"]
        )
        frontier_cycles.append({
            "cycle": cycle_index + 1,
            "switch_candidate_targets": len(cycle_switch_summary["initial_targets"]),
            "switch_remaining_targets": len(cycle_switch_summary["remaining_targets"]),
            "switch_rounds_executed": cycle_switch_summary["rounds_executed"],
            "switch_new_bbs": cycle_switch_summary["new_bbs"],
            "switch_target_bbs_discovered": cycle_switch_summary["target_bbs_discovered"],
            "switch_new_candidate_targets": cycle_switch_summary["new_candidate_targets"],
            "switch_branch_events": cycle_switch_summary["remembered_branch_events"],
            "switch_snapshot_variants": cycle_switch_summary["remembered_branch_root_snapshots"],
            "switch_priority_root_tasks_seeded": cycle_switch_summary["priority_root_tasks_seeded"],
            "frontier_candidate_targets": len(cycle_frontier_summary["initial_targets"]),
            "frontier_remaining_targets": len(cycle_frontier_summary["remaining_targets"]),
            "frontier_rounds_executed": cycle_frontier_summary["rounds_executed"],
            "frontier_new_bbs": cycle_frontier_summary["new_bbs"],
            "frontier_target_bbs_discovered": cycle_frontier_summary["target_bbs_discovered"],
            "frontier_new_candidate_targets": cycle_frontier_summary["new_candidate_targets"],
            "frontier_branch_events": cycle_frontier_summary["remembered_branch_events"],
            "frontier_snapshot_variants": cycle_frontier_summary["remembered_branch_root_snapshots"],
            "frontier_priority_root_tasks_seeded": cycle_frontier_summary["priority_root_tasks_seeded"],
            "cycle_new_global_bbs": cycle_new_bbs,
            "cycle_target_bbs_discovered": cycle_targets_discovered,
            "cycle_new_candidate_targets": cycle_new_candidate_targets,
            "cycle_branch_events": cycle_new_branch_events,
            "cycle_snapshot_variants": cycle_new_snapshot_variants,
            "cycle_priority_root_tasks_seeded": cycle_priority_root_tasks_seeded,
            "cycle_dynamic_successor_edges_added": cycle_dynamic_successor_edges,
            "switch_tail_new_bbs": cycle_switch_summary["last_round_new_bbs"],
            "switch_tail_target_bbs_discovered": cycle_switch_summary["last_round_target_bbs_discovered"],
            "switch_tail_new_candidate_targets": cycle_switch_summary["last_round_new_candidate_targets"],
            "switch_tail_branch_events": cycle_switch_summary["last_round_branch_events"],
            "switch_tail_snapshot_variants": cycle_switch_summary["last_round_snapshot_variants"],
            "frontier_tail_new_bbs": cycle_frontier_summary["last_round_new_bbs"],
            "frontier_tail_target_bbs_discovered": cycle_frontier_summary["last_round_target_bbs_discovered"],
            "frontier_tail_new_candidate_targets": cycle_frontier_summary["last_round_new_candidate_targets"],
            "frontier_tail_branch_events": cycle_frontier_summary["last_round_branch_events"],
            "frontier_tail_snapshot_variants": cycle_frontier_summary["last_round_snapshot_variants"],
            "global_bbs_after_cycle": cycle_global_after,
        })
        if (
            not frontier_cycle_auto
            or (
                not switch_frontier_enabled
                and not frontier_targeted_enabled
            )
        ):
            break
        if (
            not cycle_switch_summary["initial_targets"]
            and not cycle_frontier_summary["initial_targets"]
        ):
            break
        tail_progress = any((
            cycle_switch_summary["last_round_new_bbs"] > 0,
            cycle_switch_summary["last_round_target_bbs_discovered"] > 0,
            cycle_switch_summary["last_round_new_candidate_targets"] > 0,
            cycle_switch_summary["last_round_branch_events"] > 0,
            cycle_switch_summary["last_round_snapshot_variants"] > 0,
            cycle_switch_summary["last_round_priority_root_tasks_seeded"] > 0,
            cycle_switch_summary["last_round_dynamic_successor_edges_added"] > 0,
            cycle_frontier_summary["last_round_new_bbs"] > 0,
            cycle_frontier_summary["last_round_target_bbs_discovered"] > 0,
            cycle_frontier_summary["last_round_new_candidate_targets"] > 0,
            cycle_frontier_summary["last_round_branch_events"] > 0,
            cycle_frontier_summary["last_round_snapshot_variants"] > 0,
            cycle_frontier_summary["last_round_priority_root_tasks_seeded"] > 0,
            cycle_frontier_summary["last_round_dynamic_successor_edges_added"] > 0,
        ))
        if not tail_progress:
            break
        if (
            cycle_new_bbs <= 0
            and cycle_targets_discovered <= 0
            and cycle_new_candidate_targets <= 0
            and cycle_new_branch_events <= 0
            and cycle_new_snapshot_variants <= 0
            and cycle_priority_root_tasks_seeded <= 0
            and cycle_dynamic_successor_edges <= 0
        ):
            frontier_stale_cycles += 1
        else:
            frontier_stale_cycles = 0
        if frontier_stale_cycles >= max(1, args.frontier_cycle_stale_cycles):
            break

    contextual_isr_frontier_targets = set()
    if args.contextual_isr_frontier_time > 0 and contextual_isr_enabled:
        for round_index in range(max(1, args.contextual_isr_frontier_rounds)):
            if not can_start_phase():
                break
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
            phase_limit = bounded_time_limit(args.contextual_isr_frontier_time * 60)
            if phase_limit == 0:
                break
            round_coverage = runner.context_recovery.isr_reservoir(
                time_limit_seconds=phase_limit,
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

    elapsed = time.time() - start
    branch_catalog = runner.save_branch_catalog(branch_catalog_file)
    static_reachability = runner.static_reachability_summary()
    baseline_meta = runner.phase_metadata.get("baseline", {})
    isr_meta = runner.phase_metadata.get("isr", {})
    contextual_isr_meta = runner.phase_metadata.get("contextual_isr", {})
    isr_reservoir_meta = runner.phase_metadata.get("isr_reservoir", {})
    contextual_isr_reservoir_meta = runner.phase_metadata.get("contextual_isr_reservoir", {})
    contextual_isr_frontier_meta = runner.phase_metadata.get("contextual_isr_frontier_targeted", {})
    switch_frontier_meta = runner.phase_metadata.get("switch_frontier_targeted", {})
    frontier_targeted_meta = runner.phase_metadata.get("frontier_targeted", {})
    switch_frontier_forced_hits = sum(
        runner.phase_metadata.get(item["phase"], {}).get("forced_hit_tasks", 0)
        for item in switch_frontier_rounds
    )
    frontier_targeted_forced_hits = sum(
        runner.phase_metadata.get(item["phase"], {}).get("forced_hit_tasks", 0)
        for item in frontier_target_rounds
    )
    contextual_isr_frontier_forced_hits = sum(
        runner.phase_metadata.get(item["phase"], {}).get("forced_hit_tasks", 0)
        for item in contextual_isr_frontier_rounds
    )
    reservoir_meta = runner.phase_metadata.get("branch_reservoir", {})
    report = runner.save_report(
        report_file,
        execution_time_seconds=elapsed,
        extra={
            "runner": "gateway_reservoir_cached",
            "constraint_file": str(constraint_file),
            "seeded_constraint_count": seeded_constraint_count,
            "constraint_seed_mode": learned_mode,
            "constraint_seed_activation": args.constraint_seed_activation,
            "constraint_seed_roots": [str(path) for path in learned_roots],
            "branch_catalog_file": str(branch_catalog_file),
            "wall_clock_budget": {
                "max_wall_time_minutes": args.max_wall_time,
                "wall_time_reserve_seconds": args.wall_time_reserve,
                "budget_enabled": wall_deadline is not None,
                "budget_exhausted": wall_budget_exhausted,
                "elapsed_seconds": elapsed,
                "targeted_snapshot_retry_budget": args.targeted_snapshot_retry_budget,
            },
            "branch_catalog_summary": {
                "static_conditional_branch_points": branch_catalog["static_conditional_branch_points"],
                "main_path_branch_events": branch_catalog["main_path_branch_events"],
                "main_path_branch_points": branch_catalog["main_path_branch_points"],
                "reservoir_discovered_branch_events": branch_catalog["reservoir_discovered_branch_events"],
                "reservoir_new_branch_points": branch_catalog["reservoir_new_branch_points"],
            },
            "static_reachability": static_reachability,
            "uncovered_analysis_before_frontier": frontier_analysis_before,
            "uncovered_analysis_final": frontier_analysis_after,
            "frontier_targeting": {
                "candidate_targets": len(frontier_targets),
                "max_targets": args.frontier_targeted_max_targets,
                "min_uncovered_successors": args.frontier_targeted_min_uncovered_successors,
                "min_progress_bbs": args.frontier_targeted_min_progress_bbs,
                "switch_only": args.frontier_targeted_switch_only,
                "rounds_requested": args.frontier_targeted_rounds,
                "rounds_executed": len(frontier_target_rounds),
                "cycles_executed": len(frontier_cycles),
                "round_results": frontier_target_rounds,
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
                "min_progress_bbs": args.switch_frontier_min_progress_bbs,
                "rounds_requested": args.switch_frontier_rounds,
                "rounds_executed": len(switch_frontier_rounds),
                "cycles_executed": len(frontier_cycles),
                "round_results": switch_frontier_rounds,
            },
            "frontier_cycles": frontier_cycles,
            "invariant_checks": {
                "branch_catalog_written": branch_catalog_file.exists(),
                "main_path_reached_stable_tail_or_loop": baseline_meta.get("main_path_status") == "stable_loop_or_tail",
                "main_path_stop_reason": baseline_meta.get("run_result", {}).get("stop_reason"),
                "isr_started": isr_meta.get("isr_started") is True,
                "all_vector_isrs_attempted": isr_meta.get("irq_count") == isr_meta.get("explored_irq_count"),
                "contextual_isr_enabled": contextual_isr_enabled,
                "contextual_isr_contexts": contextual_isr_meta.get("contexts_selected", 0),
                "contextual_isr_new_bbs": contextual_isr_meta.get("new_bbs", 0),
                "contextual_isr_reservoir_enabled": args.contextual_isr_reservoir_time > 0 and contextual_isr_enabled,
                "contextual_isr_reservoir_forced_hit_tasks": contextual_isr_reservoir_meta.get("forced_hit_tasks", 0),
                "contextual_isr_frontier_enabled": args.contextual_isr_frontier_time > 0 and contextual_isr_enabled,
                "contextual_isr_frontier_forced_hit_tasks": contextual_isr_frontier_forced_hits or contextual_isr_frontier_meta.get("forced_hit_tasks", 0),
                "switch_frontier_enabled": args.switch_frontier_time > 0 and not args.frontier_targeted_switch_only,
                "switch_frontier_forced_hit_tasks": switch_frontier_forced_hits or switch_frontier_meta.get("forced_hit_tasks", 0),
                "frontier_targeted_enabled": args.frontier_targeted_time > 0,
                "frontier_targeted_forced_hit_tasks": frontier_targeted_forced_hits or frontier_targeted_meta.get("forced_hit_tasks", 0),
                "frontier_cycle_auto_enabled": frontier_cycle_auto,
                "frontier_cycles_executed": len(frontier_cycles),
                "isr_reservoir_enabled": args.isr_reservoir_time > 0,
                "isr_reservoir_forced_hit_tasks": isr_reservoir_meta.get("forced_hit_tasks", 0),
                "new_path_branch_discovery_enabled": True,
                "new_path_branch_points_discovered": reservoir_meta.get("reservoir_new_branch_points", 0),
            },
            "reservoir_rounds": reservoir_rounds,
            "diverse_rounds": diverse_rounds,
        },
    )

    print(f"baseline_bbs={len(baseline)}")
    print(f"isr_bbs={len(isr)}")
    print(f"contextual_isr_bbs={len(contextual_isr)}")
    print(f"isr_reservoir_bbs={len(isr_reservoir)}")
    print(f"contextual_isr_reservoir_bbs={len(contextual_isr_reservoir)}")
    print(f"contextual_isr_frontier_bbs={len(contextual_isr_frontier_targeted)}")
    print(f"switch_frontier_bbs={len(switch_frontier_targeted)}")
    print(f"frontier_targeted_bbs={len(frontier_targeted)}")
    print(f"reservoir_bbs={len(reservoir)}")
    print(f"covered_bbs={report['covered_bbs']}")
    print(f"total_bbs={report['total_bbs']}")
    print(f"coverage_rate={report['coverage_rate']:.2f}")
    print(f"entry_reachable_coverage_rate={static_reachability['entry_static_reachable_coverage_rate']:.2f}")
    print(f"entry_plus_vector_reachable_coverage_rate={static_reachability['entry_plus_vector_reachable_coverage_rate']:.2f}")
    print(f"pending_paths={reservoir_meta.get('pending_paths', 0)}")
    print(f"reservoir_new_branch_points={reservoir_meta.get('reservoir_new_branch_points', 0)}")
    print(f"frontier_target_candidates={len(frontier_targets)}")
    print(f"switch_frontier_candidates={len(switch_frontier_targets)}")
    print(f"switch_frontier_rounds={len(switch_frontier_rounds)}")
    print(f"frontier_target_rounds={len(frontier_target_rounds)}")
    print(f"frontier_cycles={len(frontier_cycles)}")
    print(f"contextual_isr_frontier_candidates={len(contextual_isr_frontier_targets)}")
    print(f"contextual_isr_frontier_rounds={len(contextual_isr_frontier_rounds)}")
    print(f"diverse_passes={len(diverse_rounds)}")
    print(f"wall_budget_exhausted={wall_budget_exhausted}")
    print(f"report={report_file}")


# ---------------------------------------------------------------------------
# cycle2 k.5 §1-6：H1 索引修复配套单测 3 件（挂本文件四处
# run_reservoir_branch_exploration 直调用例所在的产线消费链；stub 级，
# 不跑真实固件——真实固件的值等价由 microbench_v2_r2 canon 比较覆盖）。
# ---------------------------------------------------------------------------
import bisect as _bisect
import unittest
from types import SimpleNamespace as _SimpleNamespace
from unittest import mock as _mock

from lsgemu.scheduler.obligation_discovery import (
    semantic_entry_categories_for_name as _semantic_categories_reference,
)

_SCOPE_SYMBOLS = {
    "main": 0x1000,            # 常规函数
    "$thunk": 0x1100,          # "$" 前缀：必须被排除
    "loop_a": 0x1800,
    "helper": 0x2001,          # 奇地址：表内记 0x2000（& ~1 语义）
    "helper_twin": 0x2001,     # 对齐后同址：排序序 (addr, name) 稳定
    "aeabi_uidiv": 0x3000,
    "not_instruction": 0x9000, # 不在 instruction_lookup：必须被排除
}
_SCOPE_LOOKUP = {0x1000, 0x1800, 0x2001, 0x3000}


def _make_scope_runner():
    runner = HistoricalRunner.__new__(HistoricalRunner)
    runner.prepared = _SimpleNamespace(
        symbols_by_name=dict(_SCOPE_SYMBOLS),
        instruction_lookup=set(_SCOPE_LOOKUP),
    )
    return runner


def _reference_functions(runner):
    """参考重建 canon：逐字复刻修复前 _function_name_for_addr 内的
    每调用地址表物化（cycle2 k.5 前的 :9755 形态）。"""
    return sorted(
        (addr, name) for name, addr in runner._function_starts_by_name().items()
    )


class ReservoirIndexValueEqualityTests(unittest.TestCase):
    """单测 (i)：索引修复对全 scope 与参考重建 canon 逐值等价。"""

    def _sweep(self):
        functions = sorted(_SCOPE_SYMBOLS.items(), key=lambda kv: (kv[1] & ~1, kv[0]))
        anchors = sorted({addr & ~1 for addr in _SCOPE_LOOKUP})
        probes = set()
        for a in anchors:
            probes.update({a - 1, a, a + 1, a + 2})
        probes.update(range(0x0, 0x4000, 3))
        probes.update({0, 1, 0x3FFF, 0x4000})
        return sorted(probes)

    def test_function_name_and_entry_match_reference_rebuild(self):
        runner = _make_scope_runner()
        functions = _reference_functions(runner)
        reference_addrs = [start for start, _name in functions]
        for probe in self._sweep():
            idx = _bisect.bisect_right(reference_addrs, int(probe)) - 1
            expected_name = "" if idx < 0 else functions[idx][1]
            expected_entry = None if idx < 0 else functions[idx][0]
            self.assertEqual(
                runner._function_name_for_addr(probe), expected_name, hex(probe)
            )
            aligned_idx = _bisect.bisect_right(reference_addrs, int(probe) & ~1) - 1
            expected_aligned = (
                None if aligned_idx < 0 else functions[aligned_idx][0]
            )
            self.assertEqual(
                runner._function_entry_for_addr(probe), expected_aligned, hex(probe)
            )

    def test_empty_symbol_table_returns_neutral_values(self):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = _SimpleNamespace(symbols_by_name={}, instruction_lookup=set())
        self.assertEqual(runner._function_name_for_addr(0x1000), "")
        self.assertIsNone(runner._function_entry_for_addr(0x1000))
        self.assertEqual(runner._function_start_addrs(), [])

    def test_semantic_categories_match_reference_and_are_frozen(self):
        corpus = [
            "main", "$thunk", "aeabi_uidivmod", "__libc_init_array",
            "USART1_IRQHandler", "isr_field", "uart_read_cb",
            "can_parse", "firmata_sysex_decode", "vTaskStartScheduler",
            "hal_gpio_write", "_assert_fail", "printf", "uart_available",
            "", "None-ish", "USBReceivePacket",
        ]
        for name in corpus:
            produced = HistoricalRunner._semantic_entry_categories_for_name(name)
            self.assertEqual(
                produced,
                _semantic_categories_reference(name),
                name,
            )
            self.assertIsInstance(produced, frozenset)
        # 记忆表返回不可变 frozenset：调用方拿不到可污染缓存的引用。
        first = HistoricalRunner._semantic_entry_categories_for_name("uart_read_cb")
        second = HistoricalRunner._semantic_entry_categories_for_name("uart_read_cb")
        self.assertIs(first, second)


class SchedulerFeedbackFinalizeGuardTests(unittest.TestCase):
    """单测 (ii)：_scheduler_feedback_summary 的 finalize-only 缓存守卫
    （非 finalize 段必重算）——索引修复不得触碰该语义。"""

    class _Feedback:
        def __init__(self, payload):
            self.payload = payload
            self.calls = 0

        def scheduler_feedback_summary(self):
            self.calls += 1
            return self.payload

    def _make_runner(self, payload):
        runner = HistoricalRunner.__new__(HistoricalRunner)
        feedback = self._Feedback(payload)
        runner._scheduler_component = lambda name, cls: feedback
        return runner, feedback

    def test_non_finalize_stage_recomputes_even_with_poisoned_cache(self):
        runner, feedback = self._make_runner({"marker": "recomputed"})
        runner.current_stage_name = "branch_reservoir_round_1"
        runner.scheduler_feedback = {"marker": "POISONED"}
        runner.scheduler_feedback_cached_stage = "finalize"
        self.assertEqual(runner._scheduler_feedback_summary(), {"marker": "recomputed"})
        self.assertEqual(feedback.calls, 1)

    def test_finalize_stage_returns_cached_dict_without_recompute(self):
        runner, feedback = self._make_runner({"marker": "recomputed"})
        runner.current_stage_name = "finalize"
        runner.scheduler_feedback = {"marker": "cached_finalize"}
        runner.scheduler_feedback_cached_stage = "finalize"
        self.assertEqual(
            runner._scheduler_feedback_summary(), {"marker": "cached_finalize"}
        )
        self.assertEqual(feedback.calls, 0)

    def test_finalize_recompute_marks_stage_but_needs_cache_fill(self):
        runner, feedback = self._make_runner({"marker": "recomputed"})
        runner.current_stage_name = "finalize"
        runner.scheduler_feedback = None
        runner.scheduler_feedback_cached_stage = ""
        self.assertEqual(runner._scheduler_feedback_summary(), {"marker": "recomputed"})
        self.assertEqual(runner.scheduler_feedback_cached_stage, "finalize")
        self.assertEqual(feedback.calls, 1)


class ReservoirIndexReuseTests(unittest.TestCase):
    """单测 (iii)：地址表不出现每调用重建（对象同一性 + 构建计数），
    且与 _sorted_function_starts_cache 同失效点（lockstep）。"""

    def test_no_rebuild_after_warmup_and_object_identity(self):
        runner = _make_scope_runner()
        runner._function_name_for_addr(0x1000)  # 预热建缓存
        starts_before = runner._sorted_function_starts_cache
        addrs_before = runner._function_start_addrs_cache
        rebuild_calls = []
        original = runner._function_starts_by_name

        def counting_source():
            rebuild_calls.append(1)
            return original()

        runner._function_starts_by_name = counting_source
        for probe in range(0x1000, 0x3600, 7):
            runner._function_name_for_addr(probe)
            runner._function_entry_for_addr(probe)
            runner._semantic_entry_categories_for_name("uart_read_cb")
        self.assertEqual(rebuild_calls, [])
        self.assertIs(runner._sorted_function_starts(), starts_before)
        self.assertIs(runner._function_start_addrs(), addrs_before)
        self.assertIsInstance(addrs_before, list)

    def test_address_cache_rebuilds_in_lockstep_with_sorted_cache(self):
        runner = _make_scope_runner()
        runner._function_name_for_addr(0x1000)
        addrs_before = runner._function_start_addrs_cache
        # 单边清掉地址表缓存：下一次查询必须同时重建两张表（同失效点）。
        runner._function_start_addrs_cache = None
        runner._function_name_for_addr(0x1800)
        self.assertIsNot(runner._function_start_addrs_cache, addrs_before)
        self.assertIsNotNone(runner._sorted_function_starts_cache)


class CheckpointWriterH1WiringTests(unittest.TestCase):
    """cycle2 k.5 H1：write_coverage_checkpoint 的 5 字段 + write_timing
    8 戳接线保护（源文本断言，与 test_differential_probe_gate 的
    test_v_checkpoint_writer_wires_probe_summary 同款）。真实写路径的
    落盘证明由 /tmp 微基准脚本对真实固件触发（k.5 报告 §2-B）。"""

    @staticmethod
    def _writer_source():
        import lsgemu.historical_runner as hr_module

        source = Path(hr_module.__file__).read_text(encoding="utf-8")
        start = source.index("def write_coverage_checkpoint(")
        end = source.index("def write_progress(", start)
        return source[start:end]

    def test_five_h1_fields_are_wired_into_checkpoint_dict(self):
        writer = self._writer_source()
        for binding in (
            '"phase_name": phase_name,',
            '"time_limit_seconds": time_limit_seconds,',
            '"setup_elapsed_seconds": setup_elapsed_seconds,',
            '"setup_budget_exhausted_before_loop": setup_budget_exhausted_before_loop,',
            '"setup_budget_bootstrap_used": setup_budget_bootstrap_used,',
        ):
            self.assertIn(binding, writer)

    def test_write_timing_carries_eight_raw_stamps_without_subtraction(self):
        import re

        writer = self._writer_source()
        call_sites = re.findall(r"^\s*\w+ = time\.perf_counter\(\)", writer, re.M)
        self.assertEqual(len(call_sites), 8)
        for key in (
            "entry_perf",
            "covered_lists_post_perf",
            "scheduler_feedback_pre_perf",
            "scheduler_feedback_post_perf",
            "probe_fingerprint_pre_perf",
            "probe_fingerprint_post_perf",
            "append_pre_perf",
            "prev_append_post_perf",
        ):
            self.assertIn(f'"{key}"', writer)
        # H4 条款：写路径不做减法（差值由分析侧读时派生）。
        import re

        self.assertIsNone(re.search(r"wt_\w+\s*-\s*wt_\w+", writer))


if __name__ == "__main__":
    main()
