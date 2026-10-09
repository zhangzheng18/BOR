#!/usr/bin/env python3
"""差分扰动定位（设计 §4.2，第二级）。

当动态表达式图无法定位失败的约束（间接跳转、动态代码、图溢出）时，
退化为差分扰动测试：逐个扰动候选输入并重放，观察结果是否变化。

判据是**存在性**而非全称：
    存在任一扰动使结果变化 ⇒ 判定存在依赖；
    只有全部扰动均不改变结果才排除。
这避免掩码场景的假阴性：对 ``(MMIO & 0xf) == target``，落在高位的扰动
不会改变结果，但该输入实际是元凶——同一扰动集合中的低位翻转会暴露它。

单次探测结果为三态（复审 round2 步骤 3）：changed / unchanged /
unavailable。"变化"的判定口径由 ``classify`` 回调定义（分支重放场景用
:class:`classify_branch_replay_outcome` 的三分规则，只比较目标分支自身
的方向解析）；无法归因（分支未到达、oracle 故障）不得当作"无变化"。

扰动值选取具有区分力的取值：极值对（0x0 / 全 1）与基线的逐位翻转
（位序交错，截断时先覆盖字宽两端）。整个探测有界（单候选次数、总次数、
墙钟超时、每候选时间片）且可观测（计数与命中率）。预算耗尽时保守保留
候选（定位阶段的目标是不漏，见设计 §4.4）。

预算自洽（复审 round2 步骤 1）：默认
``max_total_probes == max_candidates × max_probes_per_candidate``（24 == 6 × 4）。
分配保留贪心（候选顺序即优先级，先到先得），per-candidate 上限是真正生效
的旋钮：任何单候选最多消费 ``min(扰动全集, max_probes_per_candidate)`` 次
探测；调大 ``max_total_probes`` / ``max_candidates`` 时应同步调大 per-candidate
上限，否则先到候选吃光总预算、其余候选零探测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Counter, Dict, List, Optional, Sequence, Tuple

# 单次探测结果的三态（复审 round2 步骤 3）：
PROBE_CHANGED = "changed"        # 与基线方向解析不同 ⇒ 依赖成立（存在性判据）
PROBE_UNCHANGED = "unchanged"    # 方向解析与基线一致（未证明也未排除）
PROBE_UNAVAILABLE = "unavailable"  # 无法归因（分支未到达 / oracle 不可用）


class ProbeOracleUnavailable(Exception):
    """重放 oracle 不可用（如快照不可恢复）：该次探测无法评估。

    必须与"无变化"区分（round1 发现 3）：静默当作无变化会掩盖本该发现
    的依赖，且完全不可观测。由 evaluate 侧抛出，prober 统一计入
    ``unresolved_oracle`` 并保守保留该候选。
    """


@dataclass(frozen=True)
class ProbeCandidate:
    """一个待检验的候选输入站点及其基线取值。

    ``occurrence is None`` 表示 pc 级 / 全出现语义（探测强制与见证验证
    覆盖该读点全部出现，复审 round2 步骤 4 最小步；完整 per-occurrence
    候选待后续轮次）。
    """

    address: int
    baseline_value: int = 0
    read_pc: int = 0
    occurrence: Optional[int] = 1
    width_bits: int = 32


@dataclass(frozen=True)
class ProbeVerdict:
    candidate: ProbeCandidate
    # "dependent" | "independent" | "unresolved_budget" | "unresolved_oracle"
    # | "unavailable"
    verdict: str
    witness_value: Optional[int] = None
    probes_used: int = 0
    changed_probes: int = 0
    unavailable_probes: int = 0


@dataclass(frozen=True)
class DifferentialProbeRequest:
    """tracer 交给上层实现的纯数据请求。

    表达式图对该分支没有任何记录（第一级不可用）时由 tracer 物化；
    候选来源与重放 oracle 由上层实现负责（运行时 MMIO 访问历史 +
    静态分析地址清单，见设计 §4.2 与任务书 B4）。
    """

    branch_pc: int
    branch_occurrence: int
    target_taken: bool
    max_candidates: int = 6
    # 预算自洽：6 候选 × 4 次/候选 = 24 总预算（见模块 docstring）。
    max_probes_per_candidate: int = 4
    max_total_probes: int = 24
    timeout_ms: int = 10_000


@dataclass(frozen=True)
class BranchReplayOutcome:
    """分支重放 oracle 的结果摘要（solver 侧构造，供三分规则比较）。

    - ``observed_directions``：去重方向集（既有比较口径，判据输入）；
    - ``direction_sequence``：含重复的方向序列（洞 B：只用于计数
      ``sequence_changed_same_directions``，不升级为判据）；
    - ``branch_seen`` / ``failure_reason``：参与旧口径整体比较
      （:meth:`legacy_key`），successor/coverage 类差异只计数不判依赖。
    """

    branch_seen: bool
    failure_reason: Optional[str] = None
    observed_directions: Tuple[str, ...] = ()
    direction_sequence: Tuple[str, ...] = ()

    def legacy_key(self) -> Tuple[object, ...]:
        """改造前的整体比较口径（branch_seen + failure_reason + 去重方向集）。"""
        return (self.branch_seen, self.failure_reason, self.observed_directions)


def classify_branch_replay_outcome(
    baseline_outcome: BranchReplayOutcome,
    outcome: BranchReplayOutcome,
    stats: Optional[Counter[str]] = None,
) -> str:
    """三分规则（复审 round2 反提案 1 + 洞 A/B；设计 §4.2 判据收窄）。

    - 扰动侧 ``branch_seen == False`` ⇒ ``unavailable``：扰动使执行未到
      目标分支，无法归因，不算变化（诚实归类且省掉一轮验证）；
    - 方向（去重集）与基线不同 ⇒ ``changed``：唯一能证明依赖的情形；
    - 方向相同 ⇒ ``unchanged``；其中旧口径（含 failure_reason 的整体
      比较）会判"变化"的只计数 ``changed_via_successor_only``（衡量收紧
      比较是否丢真依赖的度量，successor/coverage 口径差异），去重集相同
      但含重复序列不同的只计数 ``sequence_changed_same_directions``
      （洞 B：序列不升级为判据，避免 trip-count 变化引入假阳性）。

    基线侧洞 A（``baseline.branch_seen == False`` ⇒ 全部 unavailable、
    不跑扰动）由 solver 在探测前用 :func:`baseline_unreachable_report`
    处理，不经过本函数。
    """
    if not bool(getattr(outcome, "branch_seen", False)):
        if stats is not None:
            stats["unavailable_probes"] += 1
        return PROBE_UNAVAILABLE
    if tuple(getattr(outcome, "observed_directions", ())) != tuple(
        getattr(baseline_outcome, "observed_directions", ())
    ):
        return PROBE_CHANGED
    if stats is not None:
        if outcome.legacy_key() != baseline_outcome.legacy_key():
            stats["changed_via_successor_only"] += 1
        if tuple(outcome.direction_sequence) != tuple(
            baseline_outcome.direction_sequence
        ):
            stats["sequence_changed_same_directions"] += 1
    return PROBE_UNCHANGED


@dataclass(frozen=True)
class DifferentialProbeReport:
    verdicts: Tuple[ProbeVerdict, ...]
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def dependent_sites(self) -> Tuple[ProbeCandidate, ...]:
        """仅含 verdict == "dependent" 的候选（存在性判据成立、有见证值）。

        收窄（复审 round2 步骤 2）：unresolved_* 不再混入。预算/时间片/
        oracle 故障截断的候选没有 witness 值，上层求解器本来也只能按
        "dependent" 过滤（differential_probe_solver.py）；收窄后同一报告
        两处读出语义一致。保守保留的候选见 :attr:`unresolved_candidates`
        （设计 §4.4 的"不漏"由调用方决定是否重试，而不是混进依赖集）。
        """
        return tuple(
            item.candidate
            for item in self.verdicts
            if item.verdict == "dependent"
        )

    @property
    def unresolved_candidates(self) -> Tuple[ProbeCandidate, ...]:
        """未能排除也未能证明依赖的候选（unresolved_* / unavailable）。

        预算耗尽、时间片截断、oracle 故障或目标分支不可达时保守保留，
        供上层观测与重试决策；与 dependent_sites 互斥、并集为非
        independent 候选。
        """
        return tuple(
            item.candidate
            for item in self.verdicts
            if item.verdict not in {"dependent", "independent"}
        )


def baseline_unreachable_report(
    candidates: Sequence[ProbeCandidate],
) -> DifferentialProbeReport:
    """洞 A（复审 round2）：基线重放不可达目标分支时的整体报告。

    基线本身 ``branch_seen == False``（快照不是分支本身、可达性依赖输入）
    时，"扰动后 branch_seen == True 且方向与基线不同"按字面会判
    dependent——但那只是可达性变化，不是对目标约束的方向依赖，会造出
    新的假阳性。全部候选记 ``unavailable``（计数
    ``unavailable_baseline_unreachable``），且不跑任何扰动。
    """
    verdicts = tuple(
        ProbeVerdict(candidate=candidate, verdict="unavailable")
        for candidate in candidates
    )
    return DifferentialProbeReport(
        verdicts=verdicts,
        stats={
            "candidates": len(verdicts),
            "unavailable_baseline_unreachable": len(verdicts),
        },
    )


def perturbation_values(
    baseline: int,
    width_bits: int,
    *,
    limit: Optional[int] = None,
) -> List[int]:
    """按区分度排序的扰动值集合：极值对在前，随后逐位翻转（位序交错）。

    位序交错（bit 0, width-1, 1, width-2, …）保留翻转全集，同时让截断
    优先覆盖字宽两端：朴素顺序（bit 0 起顺序铺满）会把高位翻转挤到预算
    之外，掩码场景下高位依赖永远探不到（复审 round2 发现 1）。
    顺序确定（同输入同输出）；去除与基线相同及重复的值。
    """
    bits = max(1, min(32, int(width_bits)))
    mask = (1 << bits) - 1
    base = int(baseline) & mask
    interleaved: List[int] = []
    low, high = 0, bits - 1
    while low <= high:
        interleaved.append(low)
        if high != low:
            interleaved.append(high)
        low += 1
        high -= 1
    ordered: List[int] = []
    seen = set()
    for value in [0, mask] + [(base ^ (1 << bit)) & mask for bit in interleaved]:
        if value == base or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    if limit is not None:
        return ordered[: max(0, int(limit))]
    return ordered


def _default_classify(
    baseline_outcome: object,
    outcome: object,
    stats: Optional[Counter[str]],
) -> str:
    # 默认口径（非分支重放 outcome）：与基线不同即变化（旧行为）。
    return PROBE_CHANGED if outcome != baseline_outcome else PROBE_UNCHANGED


class DifferentialProber:
    """有界差分扰动探测；outcome 语义由调用方的重放 oracle 定义。"""

    def __init__(
        self,
        *,
        max_probes_per_candidate: int = 4,
        max_total_probes: int = 24,
        timeout_ms: int = 10_000,
    ):
        self.max_probes_per_candidate = max(0, int(max_probes_per_candidate))
        self.max_total_probes = max(0, int(max_total_probes))
        self.timeout_ms = max(0, int(timeout_ms))

    def probe(
        self,
        candidates: Sequence[ProbeCandidate],
        evaluate: Callable[[ProbeCandidate, int], object],
        baseline_outcome: object,
        *,
        classify: Optional[
            Callable[[object, object, Optional[Counter[str]]], str]
        ] = None,
    ) -> DifferentialProbeReport:
        """对每个候选独立做差分扰动。

        ``evaluate(candidate, value)`` 执行一次带强制值的重放并返回可比较
        的结果（不变式：同状态同值 → 同结果）；抛出的任何异常都视为
        oracle 故障（计入 ``unresolved_oracle``），不得静默当作"无变化"。

        ``classify(baseline_outcome, outcome, stats) -> changed/unchanged/
        unavailable`` 定义单次探测的三态判定；缺省为整体不等比较。
        只有 ``changed`` 能证明依赖（存在性判据，短路返回）；
        ``unavailable``（无法归因）与 oracle 故障都不能支持排除，
        全部扰动穷尽且无上述情形才得 ``independent``（设计 §4.4 不漏）。

        时间预算按候选切片（复审 round2 步骤 1）：每个候选分得
        ``timeout_ms / len(candidates)`` 的**独立时间片**（从该候选开始
        时刻起算，且不超过总 deadline）。慢 oracle 只烧掉自己的时间片，
        不会让排在前面的候选饿死后面所有候选；默认 10s / 6 候选 ≈ 1.67s，
        大于单次重放超时（1s），每个候选至少够跑一次探测。
        """
        classify = _default_classify if classify is None else classify
        stats: Counter[str] = Counter()
        deadline = (
            time.monotonic() + self.timeout_ms / 1000.0
            if self.timeout_ms > 0
            else None
        )
        candidate_slice = (
            self.timeout_ms / 1000.0 / max(1, len(candidates))
            if self.timeout_ms > 0
            else None
        )
        verdicts: List[ProbeVerdict] = []
        probes_left = self.max_total_probes
        for candidate in candidates:
            candidate_started = time.monotonic()
            candidate_deadline = (
                min(deadline, candidate_started + candidate_slice)
                if deadline is not None and candidate_slice is not None
                else None
            )
            full_values = perturbation_values(
                candidate.baseline_value,
                candidate.width_bits,
            )
            if 0 <= self.max_probes_per_candidate < len(full_values):
                values = full_values[: self.max_probes_per_candidate]
                sweep_truncated = True
            else:
                values = full_values
                sweep_truncated = False
            probes_used = 0
            changed = 0
            unavailable = 0
            oracle_failures = 0
            verdict = "independent"
            witness: Optional[int] = None
            for value in values:
                if probes_left <= 0:
                    verdict = "unresolved_budget"
                    stats["budget_exhausted"] += 1
                    break
                if (
                    candidate_deadline is not None
                    and time.monotonic() >= candidate_deadline
                ):
                    verdict = "unresolved_budget"
                    stats["timeout_stops"] += 1
                    break
                probes_left -= 1
                probes_used += 1
                stats["probes"] += 1
                try:
                    outcome = evaluate(candidate, value)
                except Exception:
                    # oracle 故障（重放不可恢复、异常）≠ "无变化"
                    # （round1 发现 3）：计数并保守保留，不得据此排除。
                    stats["unresolved_oracle"] += 1
                    oracle_failures += 1
                    continue
                state = classify(baseline_outcome, outcome, stats)
                if state == PROBE_CHANGED:
                    changed += 1
                    stats["changed_probes"] += 1
                    verdict = "dependent"
                    witness = value
                    break  # 存在性判据：任一变化即成立，无需穷尽
                if state == PROBE_UNAVAILABLE:
                    unavailable += 1
                    continue
            if verdict == "independent":
                if sweep_truncated:
                    # 扰动集被截断：未探测的取值里可能存在能暴露依赖的
                    # 扰动，不得据此排除（掩码场景假阴性的来源）。
                    verdict = "unresolved_budget"
                    stats["budget_exhausted"] += 1
                elif oracle_failures:
                    verdict = "unresolved_oracle"
                elif unavailable:
                    # 存在无法归因的探测（如扰动使执行未到目标分支）：
                    # 不能支持排除，保守保留。
                    verdict = "unavailable"
            stats[f"verdict_{verdict}"] += 1
            verdicts.append(ProbeVerdict(
                candidate=candidate,
                verdict=verdict,
                witness_value=witness,
                probes_used=probes_used,
                changed_probes=changed,
                unavailable_probes=unavailable,
            ))
            # 不在首个依赖处停止：定位阶段要产出不漏的候选集（设计 §4.4），
            # 多输入联合作用的场景需要全部依赖输入；预算上限兜底总开销。
        stats["candidates"] = len(verdicts)
        return DifferentialProbeReport(
            verdicts=tuple(verdicts),
            stats=dict(stats),
        )
