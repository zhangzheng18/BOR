"""Queue and target-order policy for the interleaved scheduler.

Only deterministic ordering and budget arithmetic live here.  The policy does
not execute a task and does not mutate coverage.  Keeping these operations
pure makes it possible to test scheduler ordering without constructing
Unicorn or a firmware image.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable, Optional


def auto_round_limit(requested_rounds: int, auto_max_rounds: int) -> int:
    return requested_rounds if requested_rounds > 0 else max(1, auto_max_rounds)


def rotate_targets(targets: list[int], round_index: int, limit: int) -> list[int]:
    if not targets:
        return []
    if limit <= 0 or len(targets) <= limit:
        return list(targets)
    offset = (round_index * limit) % len(targets)
    ordered = list(targets[offset:]) + list(targets[:offset])
    return ordered[:limit]


def rotate_target_order(targets: list[int], round_index: int, step: int) -> list[int]:
    if not targets:
        return []
    offset_step = max(1, step)
    offset = (round_index * offset_step) % len(targets)
    return list(targets[offset:]) + list(targets[:offset])


def dedupe_target_order(*target_lists: list[int]) -> list[int]:
    """Preserve the first occurrence, matching the historical scheduler."""

    seen: set[int] = set()
    ordered: list[int] = []
    for targets in target_lists:
        for bb in targets or []:
            if bb in seen:
                continue
            seen.add(bb)
            ordered.append(bb)
    return ordered


def target_head_tail_quota(total: int, head_ratio: float) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    clamped_ratio = min(1.0, max(0.0, head_ratio))
    head = int(total * clamped_ratio)
    if head <= 0:
        head = 1
    if head > total:
        head = total
    return head, max(0, total - head)


def estimate_targeted_frontier_reserve_seconds(
    *,
    total_wallclock_budget_seconds: int,
    switch_frontier_enabled: bool,
    frontier_targeted_enabled: bool,
    frontier_cycle_auto: bool,
    switch_round_seconds: int,
    switch_rounds: int,
    switch_max_rounds: int,
    frontier_round_seconds: int,
    frontier_rounds: int,
    frontier_max_rounds: int,
    frontier_cycle_max_cycles: int,
    frontier_cycle_tail_reserve_seconds: int,
) -> int:
    """Keep the exact legacy frontier reserve calculation in one place."""

    if not (switch_frontier_enabled or frontier_targeted_enabled):
        return 0

    switch_budget = 0
    if switch_frontier_enabled and switch_round_seconds > 0:
        switch_budget = max(0, int(switch_round_seconds)) * max(
            1, auto_round_limit(int(switch_rounds), int(switch_max_rounds))
        )

    frontier_budget = 0
    if frontier_targeted_enabled and frontier_round_seconds > 0:
        frontier_budget = max(0, int(frontier_round_seconds)) * max(
            1, auto_round_limit(int(frontier_rounds), int(frontier_max_rounds))
        )

    per_cycle_budget = switch_budget + frontier_budget
    if per_cycle_budget <= 0:
        return 0

    cycle_limit = max(1, int(frontier_cycle_max_cycles)) if frontier_cycle_auto else 1
    planned_budget = per_cycle_budget * cycle_limit
    first_cycle_floor = per_cycle_budget + max(0, int(frontier_cycle_tail_reserve_seconds))
    if total_wallclock_budget_seconds <= 0:
        return max(0, planned_budget)

    if total_wallclock_budget_seconds <= 600:
        reserve_cap = min(first_cycle_floor, max(36, int(total_wallclock_budget_seconds * 0.16)))
    elif total_wallclock_budget_seconds <= 1800:
        reserve_cap = min(first_cycle_floor, max(120, int(total_wallclock_budget_seconds * 0.12)))
    else:
        reserve_cap = max(first_cycle_floor, int(total_wallclock_budget_seconds * 0.30))
    return max(0, min(planned_budget, reserve_cap))


class QueuePolicy:
    """Facade for deterministic target and task ordering.

    The runner argument is optional for pure use, but allows the policy to
    expose the existing structural prioritizer without duplicating it.
    """

    component_name = "queue_policy"

    def __init__(self, runner: Any = None):
        self.runner = runner

    def prioritized_targets(
        self,
        target_bbs: Iterable[int],
        *,
        switch_only: bool,
        max_targets: int,
        min_uncovered_successors: int,
        nearby_limit: int,
        exclude_targets: Optional[set[int]] = None,
    ) -> list[int]:
        if self.runner is None or max_targets <= 0:
            return []
        targets = set(int(bb) for bb in target_bbs or set())
        if not targets:
            return []
        return self.runner.prioritized_target_bb_list(
            targets,
            min_uncovered_successors=max(1, int(min_uncovered_successors)),
            switch_only=bool(switch_only),
            include_nearby_uncovered=True,
            nearby_limit=max(1, int(nearby_limit)),
            max_targets=int(max_targets),
            exclude_targets=exclude_targets,
            prefer_expandable=True,
        )

    def queue_sizes(self) -> dict[str, int]:
        if self.runner is None:
            return {"ready": 0, "deferred": 0, "scoped_probe": 0}
        return {
            "ready": len(getattr(self.runner, "reservoir_task_queue", ()) or ()),
            "deferred": len(getattr(self.runner, "reservoir_deferred_tasks", ()) or ()),
            "scoped_probe": len(getattr(self.runner, "reservoir_scoped_probe_tasks", ()) or ()),
        }

    @staticmethod
    def pop_first_matching(queue: deque, predicate: Callable[[Any], bool]):
        """Remove the first matching item while preserving all other order."""

        kept = deque()
        matched = None
        while queue:
            item = queue.popleft()
            if matched is None and predicate(item):
                matched = item
                continue
            kept.append(item)
        return matched, kept

    @staticmethod
    def pop_best(queue: deque, score_fn: Callable[[Any], Any]):
        """Remove the highest scored item with stable ordering for all ties."""

        kept = deque()
        best_item = None
        best_score = None
        while queue:
            item = queue.popleft()
            score = score_fn(item)
            if score is None:
                kept.append(item)
                continue
            if best_score is None or score > best_score:
                if best_item is not None:
                    kept.append(best_item)
                best_item = item
                best_score = score
            else:
                kept.append(item)
        return best_item, kept

    @staticmethod
    def pop_scoped_matching(
        queue: deque,
        *,
        path_signature: Callable[[Any], Any],
        context_for_signature: Callable[[Any], Any],
        predicate: Callable[[Any, Any], bool],
    ):
        """Select one scoped task using its separately stored task context."""

        kept = deque()
        matched = None
        while queue:
            item = queue.popleft()
            signature = path_signature(item)
            context = context_for_signature(signature)
            if matched is None and predicate(item, context):
                matched = item
                continue
            kept.append(item)
        return matched, kept
