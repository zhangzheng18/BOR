"""Small contracts shared by the scheduler components.

These types contain orchestration metadata only.  Coverage remains owned by
``HistoricalRunner.validate_coverage`` and ``HistoricalRunner._record_phase``;
the components never award coverage themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class SchedulerRuntime:
    """Callbacks supplied by the interleaved CLI for stage orchestration."""

    runner: Any
    args: Any
    progress_monitor: Any
    remaining_wallclock_seconds: Callable[[], Optional[int]]
    has_wallclock_budget: Callable[[int], bool]
    has_stage_wallclock_budget: Callable[..., bool]
    clamp_enabled_stage_seconds: Callable[..., int]
    short_probe_mode: Callable[[], bool]
    total_wallclock_budget_seconds: int = 0
    targeted_frontier_reserve_seconds: int = 0


def result_size(result: Any) -> int:
    """Return a stable telemetry size without changing the result object."""

    if result is None:
        return 0
    try:
        return len(result)
    except (TypeError, AttributeError):
        return 0
