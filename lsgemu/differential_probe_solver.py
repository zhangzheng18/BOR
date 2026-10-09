#!/usr/bin/env python3
"""Adapter wiring the tracer's differential-probe protocol to the runner.

The tracer owns the escalation decision (expression graph has no record of
the branch); this module owns candidate sourcing and the replay oracle:
each probe forces one candidate MMIO value and replays from the bound
snapshot via ``_replay_mmio_from_snapshot`` + ``_evaluate_branch_replay``
(temp emulators, side-effect isolated).  A probe outcome is the branch
resolution summary; any perturbation that changes it vs. the control replay
proves dependency (existence criterion, design §4.2).

Candidate sources (task B4): runtime MMIO access history (most recent reads
first, latest observed value as baseline) plus static-analysis read records
not already observed.  All budgets are enforced by :class:`DifferentialProber`
plus a process-wide probe budget so repeated naturalization attempts cannot
run away.  Returned witness values re-enter the exact same hypothesis/
replay-validation channel as z3 and LLM slice models.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .analysis.differential_probe import (
    BranchReplayOutcome,
    DifferentialProbeReport,
    DifferentialProbeRequest,
    DifferentialProber,
    ProbeCandidate,
    ProbeOracleUnavailable,
    baseline_unreachable_report,
    classify_branch_replay_outcome,
)
from .dynamic_constraint_recovery import DynamicInputAssignment
from .register_tracer.register_tracer import RegisterTracer
from .runner_models import BranchConstraintCandidate

logger = logging.getLogger(__name__)


@dataclass
class _ReplayContext:
    snapshot_entry: object
    branch_bb: int
    take_branch: bool
    desired_successor: Optional[int]
    replay_instructions: int
    replay_timeout: int


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class DifferentialProbeSolver:
    """Implement the tracer's ``DifferentialProbeFallback`` on the runner."""

    # 公共 MMIO 判定（复审 round2 步骤 7）：直接引用 RegisterTracer 的
    # 定义（本 solver 即该 tracer 的二级升级对端，historical_runner 也
    # 已依赖同一实现），不再维护本模块私有副本。全项目 6 处定义的
    # 收敛不在本轮，本步只保证差分扰动这条新链路不新增第 7 处。
    _is_mmio_address = staticmethod(RegisterTracer._is_mmio_address)

    def __init__(self, runner):
        self.runner = runner
        self.stats: Counter[str] = Counter()
        self.last_report: Optional[DifferentialProbeReport] = None
        self._context: Optional[_ReplayContext] = None
        self._global_probes_used = 0

    # -- lifecycle ------------------------------------------------------
    def bind_replay_context(
        self,
        *,
        snapshot_entry: object,
        branch_bb: int,
        take_branch: bool,
        desired_successor: Optional[int],
        replay_instructions: int,
        replay_timeout: int,
    ) -> None:
        """Attach the replay context for the current naturalization attempt.

        Called by the runner right before asking the tracer for dynamic
        candidate sets; without a bound context the probe never runs (the
        naturalization bootstrap path has no snapshot to replay from).
        """
        self._context = _ReplayContext(
            snapshot_entry=snapshot_entry,
            branch_bb=int(branch_bb) & 0xFFFFFFFF,
            take_branch=bool(take_branch),
            desired_successor=desired_successor,
            replay_instructions=int(replay_instructions),
            replay_timeout=int(replay_timeout),
        )

    def clear_replay_context(self) -> None:
        self._context = None

    @property
    def has_replay_context(self) -> bool:
        """外层（消融臂 generic_dynamic_fallback）是否已绑定重放上下文。

        choke point（_naturalization_dynamic_candidate_sets）只在本属性为
        False 时才就地补绑，保证外层显式绑定优先、消融臂行为不变。
        绑定 snapshot_entry=None（无可用快照）时本属性为 True：上下文已
        绑定、只是无快照可重放，solver 侧照旧 skipped_no_context。
        """
        return self._context is not None

    # -- protocol entry -------------------------------------------------
    def __call__(
        self,
        request: DifferentialProbeRequest,
    ) -> Optional[List[DynamicInputAssignment]]:
        if os.environ.get("LSGEMU_DIFF_PROBE", "1").strip().lower() in {"0", "false", "no", "off"}:
            self.stats["skipped_disabled"] += 1
            return None
        context = self._context
        if context is None or context.snapshot_entry is None:
            self.stats["skipped_no_context"] += 1
            return None
        if self._request_context_mismatch(request, context):
            self.stats["skipped_context_mismatch"] += 1
            return None
        global_budget = _env_int("LSGEMU_DIFF_PROBE_GLOBAL_PROBE_BUDGET", 512)
        if self._global_probes_used >= global_budget:
            self.stats["skipped_global_budget"] += 1
            return None

        self.stats["calls"] += 1
        candidates = self._collect_candidates(int(request.max_candidates))
        if not candidates:
            self.stats["skipped_no_candidates"] += 1
            return None

        def replay_outcome(
            constraint_items,
        ) -> Optional[BranchReplayOutcome]:
            valid, _run_result, restored, replay_events, _snapshots = (
                self.runner._replay_mmio_from_snapshot(
                    context.snapshot_entry,
                    constraint_items,
                    replay_instructions=context.replay_instructions,
                    replay_timeout=context.replay_timeout,
                )
            )
            if not restored:
                return None
            evaluation = self.runner._evaluate_branch_replay(
                replay_events,
                context.branch_bb,
                context.take_branch,
                context.desired_successor,
                valid,
            )
            # 含重复的方向序列（洞 B）：与 _evaluate_branch_replay 同一
            # 过滤口径但不去重，只用于 sequence_changed_same_directions
            # 计数，不进入判据。
            sequence = []
            for event in replay_events or []:
                if int(getattr(event, "address", 0) or 0) != int(
                    context.branch_bb
                ):
                    continue
                sequence.append(
                    "taken"
                    if bool(getattr(event, "original_taken", False))
                    else "not-taken"
                )
            return BranchReplayOutcome(
                branch_seen=bool(evaluation["branch_seen"]),
                failure_reason=evaluation.get("failure_reason"),
                observed_directions=tuple(evaluation["observed_directions"]),
                direction_sequence=tuple(sequence),
            )

        started_at = time.perf_counter()
        baseline_outcome = replay_outcome([])
        if baseline_outcome is None:
            self.stats["baseline_restore_failed"] += 1
            return None
        if not baseline_outcome.branch_seen:
            # 洞 A（复审 round2）：基线重放本身不可达目标分支——扰动后
            # "branch_seen 变 True"只是可达性变化而非方向依赖，按字面
            # 判 dependent 会造出假阳性。全部记 unavailable，不跑扰动。
            report = baseline_unreachable_report(candidates)
            self.last_report = report
            self.stats.update(Counter(report.stats))
            self.stats["elapsed_seconds_total"] += int(
                time.perf_counter() - started_at
            )
            return None

        def evaluate(candidate: ProbeCandidate, value: int):
            self.stats["replays"] += 1
            item = BranchConstraintCandidate(
                constraint_type="mmio",
                address=int(candidate.address) & 0xFFFFFFFF,
                value=int(value) & 0xFFFFFFFF,
                read_pc=(int(candidate.read_pc) & 0xFFFFFFFF) or None,
                source="differential_probe",
            )
            outcome = replay_outcome([item])
            if outcome is None:
                # 重放不可恢复 = oracle 故障，不是"无变化"（round1 发现 3）：
                # 抛给 prober 计入 unresolved_oracle 并保守保留该候选。
                raise ProbeOracleUnavailable(
                    f"replay not restored for site {int(candidate.address) & 0xFFFFFFFF:#x}"
                )
            return outcome

        remaining_budget = max(
            0, global_budget - self._global_probes_used
        )
        prober = DifferentialProber(
            max_probes_per_candidate=int(request.max_probes_per_candidate),
            max_total_probes=min(int(request.max_total_probes), remaining_budget),
            timeout_ms=int(request.timeout_ms),
        )
        report = prober.probe(
            candidates,
            evaluate,
            baseline_outcome,
            classify=classify_branch_replay_outcome,
        )
        self._global_probes_used += int(report.stats.get("probes", 0) or 0)
        self.last_report = report
        self.stats.update(Counter(report.stats))
        self.stats["elapsed_seconds_total"] += int(time.perf_counter() - started_at)

        dependent = [
            verdict for verdict in report.verdicts if verdict.verdict == "dependent"
        ]
        # 与 report.dependent_sites 同口径（复审 round2 步骤 2）：只统计
        # 存在性判据成立的 dependent；保守保留的候选单独计数，两处读出
        # 不再互相矛盾。
        self.stats["dependent_sites"] += len(dependent)
        self.stats["unresolved_candidates"] += len(report.unresolved_candidates)
        if not dependent:
            return None
        # 见证值按单候选逐个返回：各见证在隔离扰动下发现，联合赋值未经
        # 证实，交给重放验证（唯一真值）逐个把关。
        primary = dependent[0]
        candidate = primary.candidate
        self.stats["witness_assignments"] += 1
        width_bits = max(1, min(32, int(candidate.width_bits or 32)))
        mask = (1 << width_bits) - 1 if width_bits < 32 else 0xFFFFFFFF
        return [
            DynamicInputAssignment(
                kind="mmio",
                address=int(candidate.address) & 0xFFFFFFFF,
                read_pc=int(candidate.read_pc) & 0xFFFFFFFF,
                # pc 级 / 全出现语义（复审 round2 步骤 4 最小步）：探测在
                # "全出现强制"下取得证据，见证不再强绑 occurrence=1 的
                # "仅第 1 次出现"语义。None 由 tracer 侧在进入旧 runner
                # 通道前归一为 1（见 register_tracer 接缝注释）。
                occurrence=(
                    int(candidate.occurrence)
                    if candidate.occurrence is not None
                    else None
                ),
                width=width_bits,
                value=int(primary.witness_value or 0) & mask,
                observed_value=int(candidate.baseline_value) & mask,
            )
        ]

    def _request_context_mismatch(
        self,
        request: DifferentialProbeRequest,
        context: _ReplayContext,
    ) -> bool:
        """护栏：请求的分支/方向必须与绑定上下文一致（choke point 常态化绑定后防错配）。

        - 方向：``request.target_taken`` 与 ``context.take_branch`` 等值检查。
        - 分支：``request.branch_pc`` 是**分支指令地址**、``context.branch_bb``
          是**基本块起始**，常态不相等（讨论 round1 (d).2）——直接等值比较
          会把几乎所有合法调用误拒成死臂。正确判式：用 runner 的
          ``_branch_instruction(context.branch_bb)`` 把 bb 还原成分支指令地址
          再比较。拿不到指令地址（runner 无该 helper / 返回 None / 地址为
          0）时退化为只比方向：宁可放行（评估仍以 context 为准，最坏白干
          一轮、见证照旧过重放验证把关），不可误拒。
        """
        if bool(request.target_taken) != bool(context.take_branch):
            return True
        resolver = getattr(self.runner, "_branch_instruction", None)
        if not callable(resolver):
            return False
        try:
            instruction = resolver(int(context.branch_bb) & 0xFFFFFFFF)
            branch_pc = int((instruction or {}).get("address", 0) or 0)
        except Exception:
            return False
        if branch_pc <= 0:
            return False
        return (int(request.branch_pc) & 0xFFFFFFFF) != (branch_pc & 0xFFFFFFFF)

    # -- candidate sourcing (task B4) -----------------------------------
    def _collect_candidates(self, max_candidates: int) -> List[ProbeCandidate]:
        candidates: List[ProbeCandidate] = []
        seen_addresses = set()
        # 运行时候选宽度优先取静态同地址读记录，取不到再退 32
        # （复审 round2 步骤 6；运行时历史元组无宽度字段）。
        static_read_width_bits = self._static_read_width_bits()

        emulator = getattr(self.runner, "emulator", None)
        history = list(getattr(emulator, "mmio_access_history", []) or [])
        # 运行时历史：最近读取优先（失败前最新输入最可能是元凶），
        # 每地址取最新观测值作基线。
        for pc, address, is_read, value in reversed(history):
            try:
                pc = int(pc) & 0xFFFFFFFF
                address = int(address) & 0xFFFFFFFF
                value = int(value) & 0xFFFFFFFF
                is_read = bool(is_read)
            except (TypeError, ValueError):
                continue
            if not is_read or address in seen_addresses:
                continue
            if not self._is_mmio_address(address):
                continue
            seen_addresses.add(address)
            candidates.append(ProbeCandidate(
                address=address,
                baseline_value=value,
                read_pc=pc,
                # pc 级 / 全出现语义（复审 round2 步骤 4 最小步）
                occurrence=None,
                width_bits=static_read_width_bits.get(address, 32),
            ))
            if len(candidates) >= max(1, int(max_candidates)):
                return candidates

        prepared = getattr(self.runner, "prepared", None)
        for item in list(getattr(prepared, "static_mmio_accesses", []) or []):
            if not isinstance(item, dict) or item.get("address") is None:
                continue
            if str(item.get("access_type", "") or "") != "read":
                continue
            try:
                address = int(item.get("address")) & 0xFFFFFFFF
            except (TypeError, ValueError):
                continue
            if address in seen_addresses or not self._is_mmio_address(address):
                continue
            seen_addresses.add(address)
            width_bits = static_read_width_bits.get(address, 32)
            candidates.append(ProbeCandidate(
                address=address,
                baseline_value=0,
                read_pc=int(item.get("pc", 0) or 0) & 0xFFFFFFFF,
                # pc 级 / 全出现语义（复审 round2 步骤 4 最小步）
                occurrence=None,
                width_bits=width_bits,
            ))
            if len(candidates) >= max(1, int(max_candidates)):
                break
        return candidates

    def _static_read_width_bits(self) -> Dict[int, int]:
        """静态读记录的地址 → 位宽表（每地址取首条记录）。

        8/16 位站点按真实位宽截断扰动集，不再按 32 位铺满（省预算、
        更快暴露 unresolved，无召回损失——见复审 round1 发现 6）。
        """
        widths: Dict[int, int] = {}
        prepared = getattr(self.runner, "prepared", None)
        for item in list(getattr(prepared, "static_mmio_accesses", []) or []):
            if not isinstance(item, dict) or item.get("address") is None:
                continue
            if str(item.get("access_type", "") or "") != "read":
                continue
            try:
                address = int(item.get("address")) & 0xFFFFFFFF
            except (TypeError, ValueError):
                continue
            try:
                width_bits = max(1, min(32, 8 * int(item.get("width", 4) or 4)))
            except (TypeError, ValueError):
                width_bits = 32
            widths.setdefault(address, width_bits)
        return widths

    def get_statistics(self) -> Dict[str, int]:
        return dict(self.stats)
