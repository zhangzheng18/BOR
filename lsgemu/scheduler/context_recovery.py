"""Execution-context recovery boundary.

This component owns the decision and invocation boundary for stream, ISR and
RTOS-thread contexts.  It does not manufacture an entry point: every method
delegates to the existing runner implementation, which requires an
entry-derived snapshot or a validated event context.
"""

from __future__ import annotations

from typing import Any, Optional


class ContextRecovery:
    component_name = "context_recovery"

    def __init__(self, runner: Any):
        self.runner = runner

    def cold_isr(self, *, max_instructions: int, count_coverage: bool) -> set[int]:
        return self.runner.run_isr_exploration(
            max_instructions=max_instructions,
            count_coverage=count_coverage,
        )

    def contextual_isr(
        self,
        *,
        max_contexts: int,
        max_isrs: int,
        max_instructions: int,
        replay_timeout: int,
        time_limit_seconds: Optional[int],
        include_reservoir_contexts: bool = False,
        max_reservoir_contexts: int = 16,
        stage_name: str = "contextual_isr",
    ) -> set[int]:
        return self.runner.run_contextual_isr_exploration(
            max_contexts=max_contexts,
            max_isrs=max_isrs,
            max_instructions=max_instructions,
            replay_timeout=replay_timeout,
            time_limit_seconds=time_limit_seconds,
            include_reservoir_contexts=include_reservoir_contexts,
            max_reservoir_contexts=max_reservoir_contexts,
            phase_name=stage_name,
        )

    def isr_reservoir(
        self,
        *,
        time_limit_seconds: Optional[int] = None,
        max_instructions: int = 20000,
        replay_timeout: int = 1_000_000,
        max_tasks: int = 500,
        max_tasks_per_isr: int = 64,
        dynamic_frontier_seeding: bool = True,
        context_snapshots: bool = False,
        max_contexts: int = 16,
        include_reservoir_contexts: bool = False,
        max_reservoir_contexts: int = 16,
        max_isrs: int = 0,
        target_bbs: Optional[set[int]] = None,
        prioritize_target_bbs: bool = False,
        phase_name: str = "isr_reservoir",
    ) -> set[int]:
        return self.runner.run_isr_reservoir_exploration(
            time_limit_seconds=time_limit_seconds,
            max_instructions=max_instructions,
            replay_timeout=replay_timeout,
            max_tasks=max_tasks,
            max_tasks_per_isr=max_tasks_per_isr,
            dynamic_frontier_seeding=dynamic_frontier_seeding,
            context_snapshots=context_snapshots,
            max_contexts=max_contexts,
            include_reservoir_contexts=include_reservoir_contexts,
            max_reservoir_contexts=max_reservoir_contexts,
            max_isrs=max_isrs,
            target_bbs=target_bbs,
            prioritize_target_bbs=prioritize_target_bbs,
            phase_name=phase_name,
        )

    def stream_input(
        self,
        *,
        time_limit_seconds: Optional[int],
        max_profiles: int,
        max_contexts: int,
        max_seeds: int,
        max_tasks: int,
        max_instructions: int,
        replay_timeout: int,
        no_new_bb_limit: int,
        phase_name: str,
        prefer_stream_summary_contexts: bool = False,
    ) -> set[int]:
        return self.runner.run_stream_input_exploration(
            time_limit_seconds=time_limit_seconds,
            max_profiles=max_profiles,
            max_contexts=max_contexts,
            max_seeds=max_seeds,
            max_tasks=max_tasks,
            max_instructions=max_instructions,
            replay_timeout=replay_timeout,
            no_new_bb_limit=no_new_bb_limit,
            phase_name=phase_name,
            prefer_stream_summary_contexts=prefer_stream_summary_contexts,
        )

    def rtos_thread(
        self,
        *,
        time_limit_seconds: Optional[int],
        max_tasks: int,
        max_instructions: int,
        replay_timeout: int,
        no_new_bb_limit: int,
        target_bbs: Optional[set[int]],
        max_targets: int,
        variants_per_call: int,
        phase_name: str,
    ) -> set[int]:
        return self.runner.run_rtos_thread_entry_exploration(
            time_limit_seconds=time_limit_seconds,
            max_tasks=max_tasks,
            max_instructions=max_instructions,
            replay_timeout=replay_timeout,
            no_new_bb_limit=no_new_bb_limit,
            target_bbs=target_bbs,
            max_targets=max_targets,
            variants_per_call=variants_per_call,
            phase_name=phase_name,
        )

    def summary(self) -> dict[str, object]:
        return {
            "component": self.component_name,
            "entry_derived_contexts": len(
                getattr(self.runner, "reservoir_interrupt_contexts", ()) or ()
            ),
        }
