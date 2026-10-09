"""Semantic-obligation discovery boundary.

This component decides whether semantic evidence may influence scheduling and
provides target-selection adapters.  It does not synthesize coverage and does
not execute replay.  The underlying structural and semantic analyzers remain
in ``HistoricalRunner`` for compatibility with existing callers.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..path_naturalization import normalize_signature as normalize_naturalization_signature
from ..runner_common import _loop_exit_hints_enabled, _normalize_name
from ..runner_config import SEMANTIC_FRONTIER_CATEGORY_PRIORITY


def semantic_entry_categories_for_name(name: object) -> set[str]:
    """Classify a symbol into semantic entry-condition families."""

    normalized = _normalize_name(str(name or ""))
    categories: set[str] = set()
    if not normalized:
        return categories
    if (
        normalized.startswith("aeabi")
        or normalized.startswith("adddf")
        or normalized.startswith("addsf")
        or normalized.startswith("subdf")
        or normalized.startswith("subsf")
        or normalized.startswith("muldf")
        or normalized.startswith("mulsf")
        or normalized.startswith("divdf")
        or normalized.startswith("divsf")
        or normalized.startswith("fixdf")
        or normalized.startswith("fixsf")
        or normalized.startswith("float")
        or normalized.startswith("gnu")
        or normalized.startswith("udiv")
        or normalized.startswith("umod")
    ):
        categories.add("libgcc_runtime_path")
    if any(token in normalized for token in (
        "globaldtors", "globalctors", "staticinitializationanddestruction",
        "libcinitarray", "do_global_dtors_aux", "frame_dummy",
    )):
        categories.add("startup_teardown_path")
    if any(token in normalized for token in ("irqhandler", "isr", "interrupt", "handler")):
        categories.add("interrupt_or_vector_handler")
    if any(token in normalized for token in ("callback", "cb", "hook")):
        categories.add("callback_or_hook")
    if any(token in normalized for token in (
        "parser", "parse", "decode", "processinput", "available", "read",
        "serial", "uart", "usart", "usb", "can", "spi", "i2c",
    )):
        categories.add("stream_or_protocol_input")
    if any(token in normalized for token in (
        "firmata", "sysex", "digitalwrite", "analogwrite", "reportdigital",
    )):
        categories.add("protocol_callback")
    if any(token in normalized for token in (
        "thread", "task", "rtos", "scheduler", "freertos", "osthread",
    )):
        categories.add("rtos_or_task_entry")
    if any(token in normalized for token in (
        "hal", "ll", "dma", "tim", "adc", "gpio", "rcc", "pll",
        "flash", "pwr", "systick",
    )):
        categories.add("hal_or_peripheral_state")
    if any(token in normalized for token in (
        "assert", "error", "fault", "abort", "panic", "default",
    )):
        categories.add("error_or_fault_path")
    if any(token in normalized for token in (
        "print", "println", "write", "flush", "puts", "putchar",
    )):
        categories.add("io_output_path")
    if any(token in normalized for token in (
        "printf", "sprintf", "vfprintf", "malloc", "free", "memcpy",
        "memset", "strlen", "strcmp", "strncmp", "strcpy", "strncpy",
        "strstr", "strchr",
    )):
        categories.add("libc_or_runtime_path")
    return categories


def build_ablation_fallback_policy(args: Any) -> dict[str, object]:
    """Describe the contribution-level fallback contract.

    This is intentionally data-only.  It mirrors the existing experiment
    contract and therefore cannot silently disable a runtime stage.
    """

    active: list[dict[str, object]] = []
    contribution_level = bool(
        getattr(args, "disable_semantic_obligation_stages", False)
        or getattr(args, "disable_scoped_replay_stages", False)
        or getattr(args, "disable_context_event_replay_stages", False)
    )
    if getattr(args, "disable_semantic_obligation_stages", False):
        active.append({
            "name": "without_semantic_obligation",
            "paper_label": "Without Semantic Obligation",
            "ablation_schema_version": 2,
            "removed": [
                "stagnation-to-obligation conversion",
                "semantic obligation types and branch hotsets",
                "semantic target-category scoring",
                "semantic scheduler feedback and feedback-root seeding",
                "semantic obligation-driven candidate generation",
                "semantic entry-prefix probing",
                "semantic anchor-based frontier flush",
                "semantic frontier drain",
                "semantic vector cleanup",
            ],
            "preserved_fallbacks": [
                "baseline reset-entry execution",
                "generic MMIO readiness and global branch-MMIO repair",
                "cross-BB causal tracing and dynamic constraint recovery",
                "generic branch replay without semantic hotsets",
                "CFG frontier/successor replay ranked by structural potential",
                "generic switch and dispatch replay",
                "basic contextual ISR/event replay",
                "direct-call, stream, thread, and deadline-drain replay",
            ],
            "question": "Is semantic-guided obligation discovery necessary?",
        })
    if getattr(args, "disable_scoped_replay_stages", False):
        active.append({
            "name": "without_scoped_replay",
            "paper_label": "Without Scoped Replay",
            "ablation_schema_version": 2,
            "removed": [
                "branch-occurrence scoped replay",
                "branch-specific snapshot and provenance replay",
                "learned read-PC and read-occurrence scoped data repair",
                "target-scoped frontier and dispatch replay",
                "successor-guided targeted replay",
                "target-scoped vector/frontier cleanup",
                "force-free naturalization of scoped path obligations",
            ],
            "preserved_fallbacks": [
                "baseline reset-entry execution",
                "reset-entry task-local address-global MMIO repair",
                "basic contextual ISR/event replay",
                "untargeted direct-call, stream, and thread replay",
                "deadline direct-call replay over entry-reachable targets",
            ],
            "question": "Is evidence-scoped replay necessary?",
        })
    if getattr(args, "disable_context_event_replay_stages", False):
        active.append({
            "name": "without_context_event_recovery",
            "paper_label": "Without Context-aware Event Recovery",
            "removed": [
                "contextual ISR replay",
                "ISR reservoir replay",
                "vector cleanup",
                "ISR-frontier replay",
                "stream-input event replay",
                "RTOS thread-entry replay",
            ],
            "preserved_fallbacks": [
                "baseline reset-entry execution",
                "generic branch replay",
                "branch-MMIO repair",
                "frontier/dispatch replay",
                "direct-call and summary-return replay",
            ],
            "question": "Is event context necessary?",
        })

    semantic_disabled = bool(getattr(args, "disable_semantic_obligation_stages", False))
    scoped_disabled = bool(getattr(args, "disable_scoped_replay_stages", False))
    context_disabled = bool(getattr(args, "disable_context_event_replay_stages", False))
    return {
        "primary_ablation_level": "contribution" if contribution_level else "none",
        "common_contract": [
            "The static valid-BB denominator is unchanged.",
            "The Unicorn execution hook and strict coverage oracle are unchanged.",
            "Disabled modules are replaced only with existing baseline/fallback processing, not with a new coverage source.",
        ],
        "effective_internal_switches": {
            "disable_semantic_obligation_stages": semantic_disabled,
            "disable_scoped_replay_stages": scoped_disabled,
            "disable_context_event_replay_stages": context_disabled,
            "generic_branch_replay_disabled": bool(getattr(args, "disable_branch_reservoir_stages", False)),
            "targeted_frontier_recovery_disabled": bool(getattr(args, "disable_frontier_stages", False)),
            "path_naturalization_disabled": bool(semantic_disabled or scoped_disabled),
            "semantic_frontier_drain_disabled": bool(getattr(args, "disable_semantic_frontier_drain_stage", False)),
            "semantic_stagnation_conversion_disabled": semantic_disabled,
            "semantic_obligation_type_state_disabled": semantic_disabled,
            "semantic_target_scoring_disabled": semantic_disabled,
            "semantic_scheduler_feedback_disabled": semantic_disabled,
            "obligation_driven_candidate_generation_disabled": semantic_disabled,
            "generic_fallback_scheduler_enabled": semantic_disabled,
            "causal_cross_bb_constraint_recovery_enabled": True,
            "learned_pc_occurrence_scoped_constraints_disabled": scoped_disabled,
            "branch_snapshot_provenance_replay_disabled": scoped_disabled,
            "global_mmio_fallback_enabled": scoped_disabled,
            "contextual_isr_replay_disabled": bool(getattr(args, "disable_contextual_isr_stages", False)),
            "direct_stream_thread_replay_disabled": bool(getattr(args, "disable_direct_stream_thread_stages", False)),
            "stream_input_event_replay_disabled": bool(
                context_disabled or bool(getattr(args, "disable_direct_stream_thread_stages", False))
            ),
            "rtos_thread_entry_replay_disabled": bool(
                context_disabled or bool(getattr(args, "disable_direct_stream_thread_stages", False))
            ),
            "direct_call_continuation_disabled": bool(getattr(args, "disable_direct_stream_thread_stages", False)),
        },
        "active": active,
    }


class ObligationDiscovery:
    """Own semantic gating and target discovery, without replay side effects."""

    component_name = "obligation_discovery"

    def __init__(self, runner: Any):
        self.runner = runner

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.runner, "semantic_obligation_enabled", True))

    def category_priority(self, categories: Iterable[str]) -> int:
        if not self.enabled:
            return 0
        category_set = set(categories or []) or {"uncategorized_static_code"}
        return max(
            SEMANTIC_FRONTIER_CATEGORY_PRIORITY.get(str(category), 10)
            for category in category_set
        )

    def categories_for_bb(self, bb_addr: int) -> set[str]:
        if not self.enabled:
            return set()
        categories = semantic_entry_categories_for_name(
            self.runner._function_name_for_addr(int(bb_addr))
        )
        return categories or {"uncategorized_static_code"}

    def priority_for_bb(self, bb_addr: int) -> int:
        return self.category_priority(self.categories_for_bb(int(bb_addr)))

    def configure_emulator_feedback(self, emulator: Any) -> None:
        """Install only semantic-obligation state on an emulator instance."""

        runner = self.runner
        semantic_enabled = self.enabled
        if not semantic_enabled:
            runner.branch_snapshot_hotset.clear()
            runner.deadlock_failed_directions.clear()
        emulator.semantic_obligation_enabled = semantic_enabled
        emulator.enable_loop_intervention = semantic_enabled
        emulator.branch_snapshot_hotset = (
            set(runner.branch_snapshot_hotset) if semantic_enabled else set()
        )
        emulator.runtime_written_pages.update(
            int(page) & ~0xFFF
            for page in (
                getattr(getattr(runner, "emulator", None), "runtime_written_pages", set())
                or set()
            )
        )
        emulator.deadlock_failed_directions = (
            {
                int(loop_head): {
                    int(branch_pc): set(bool(direction) for direction in directions)
                    for branch_pc, directions in branch_map.items()
                }
                for loop_head, branch_map in runner.deadlock_failed_directions.items()
            }
            if semantic_enabled
            else {}
        )
        emulator.loop_exit_iteration_hints = (
            dict(
                getattr(runner, "loop_exit_iteration_hints", {})
                or getattr(getattr(runner, "prepared", None), "loop_exit_iteration_hints", {})
                or {}
            )
            if semantic_enabled and _loop_exit_hints_enabled()
            else {}
        )
        emulator.loop_intervention_threshold_cache.clear()

    def harvest_emulator_feedback(self, emulator: Any) -> bool:
        """Merge semantic hotsets and failed directions from one replay."""

        runner = self.runner
        if not self.enabled:
            runner.branch_snapshot_hotset.clear()
            runner.deadlock_failed_directions.clear()
            return False
        before = len(runner.branch_snapshot_hotset)
        runner.branch_snapshot_hotset.update(
            getattr(emulator, "branch_snapshot_hotset", set())
        )
        for loop_head, branch_map in getattr(
            emulator, "deadlock_failed_directions", {}
        ).items():
            runner_branch_map = runner.deadlock_failed_directions.setdefault(
                int(loop_head), {}
            )
            for branch_pc, directions in branch_map.items():
                runner_branch_map.setdefault(int(branch_pc), set()).update(
                    bool(direction) for direction in directions
                )
        return len(runner.branch_snapshot_hotset) > before

    def register_forced_path_obligation(
        self,
        signature: Iterable[tuple[tuple[int, int], object]],
        *,
        target_bbs: Iterable[int],
        discovered_bbs: Iterable[int],
        discovery_trace_signature: Iterable[
            tuple[tuple[int, int], object]
        ] = tuple(),
        source_phase: str,
    ) -> bool:
        """Register the existing temporal naturalization obligation."""

        if not self.enabled:
            return False
        runner = self.runner
        normalized = normalize_naturalization_signature(
            tuple(signature or tuple())
        )
        obligation = runner._path_naturalization_ledger().register_obligation(
            normalized,
            target_bbs=runner.validate_coverage(target_bbs),
            discovered_bbs=runner.validate_coverage(discovered_bbs),
            discovery_trace_signature=normalize_naturalization_signature(
                tuple(discovery_trace_signature or tuple())
            ),
            source_phase=str(source_phase or ""),
        )
        return obligation is not None
