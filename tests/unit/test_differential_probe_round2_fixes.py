#!/usr/bin/env python3
"""差分扰动二级定位缺陷修复（2026-09-18 第 1 轮）的回归测试。

对应复审 round2_response §四 步骤 1/2/3/4/6/7；每条测试都在未打补丁的
代码（HEAD 7f2528c）上失败，保护点见各 docstring。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  # 触发包级 bootstrap
from lsgemu.analysis.differential_probe import (
    BranchReplayOutcome,
    DifferentialProbeRequest,
    DifferentialProber,
    ProbeCandidate,
    classify_branch_replay_outcome,
    perturbation_values,
)
from lsgemu.differential_probe_solver import DifferentialProbeSolver
from tests.unit.test_differential_probe import FakeRunner, _bind, _probe_request


def candidate(address, baseline=0, width_bits=32):
    return ProbeCandidate(
        address=address, baseline_value=baseline, width_bits=width_bits
    )


class Step1BudgetSelfConsistencyTests(unittest.TestCase):
    """步骤 1：预算自洽（默认 6 候选 × 4 次/候选 = 24 总预算）+ 位序交错。"""

    def test_default_budget_knobs_are_self_consistent(self):
        """保护：默认配置下 per-candidate 上限是真正生效的旋钮。

        改坏表现：默认 cap（34）大于总预算（24），贪心分配下首个候选
        吃光 24 次预算、cap 永不生效，`max_candidates × cap != total`。
        """
        request = DifferentialProbeRequest(
            branch_pc=0x9000, branch_occurrence=1, target_taken=True
        )
        self.assertEqual(
            request.max_candidates * request.max_probes_per_candidate,
            request.max_total_probes,
        )
        prober_defaults = DifferentialProber()
        self.assertEqual(prober_defaults.max_probes_per_candidate, 4)
        self.assertEqual(prober_defaults.max_total_probes, 24)

    def test_perturbation_order_interleaves_bit_sequence(self):
        """保护：扰动集按位序交错（0,31,1,30,…）生成。

        改坏表现：朴素顺序（bit 0 起顺序铺满）把高位翻转挤到预算之外，
        截断集（limit=6）里不再同时含 bit0 与 bit31。
        """
        values = perturbation_values(0, 32)
        self.assertEqual(
            values[:6],
            [0xFFFFFFFF, 1, 0x80000000, 2, 0x40000000, 4],
        )
        # 8 位：翻转顺序 bit0,bit7,bit1,bit6,…
        values8 = perturbation_values(0x5A, 8)
        self.assertEqual(values8[:2], [0x0, 0xFF])
        self.assertEqual(values8[2:6], [0x5B, 0xDA, 0x58, 0x1A])
        # 截断集仍覆盖字宽两端与极值对
        truncated = perturbation_values(0, 32, limit=6)
        self.assertIn(0x1, truncated)
        self.assertIn(0x80000000, truncated)
        self.assertIn(0xFFFFFFFF, truncated)

    def test_default_budget_distributes_probes_across_candidates(self):
        """「6 候选 × 恒等 oracle」分配：每候选 4 次，而不是首个候选独占 24。

        改坏表现（旧默认 cap=34 > total=24）：分配变为 [24,0,0,0,0,0]，
        `max_candidates=6` 形同虚设。
        """
        candidates = [candidate(0x40000000 + 4 * i) for i in range(6)]

        def evaluate(_candidate, _value):
            return "unchanged"

        report = DifferentialProber().probe(candidates, evaluate, "unchanged")
        self.assertEqual(
            [v.probes_used for v in report.verdicts], [4, 4, 4, 4, 4, 4]
        )
        self.assertEqual(report.stats["probes"], 24)
        self.assertEqual(report.stats["budget_exhausted"], 6)
        self.assertEqual(report.stats.get("timeout_stops", 0), 0)
        self.assertEqual(
            [v.verdict for v in report.verdicts],
            ["unresolved_budget"] * 6,
        )

    def test_per_candidate_time_slice_prevents_starvation(self):
        """保护：per-candidate 时间片（deadline / len(candidates)）生效。

        慢 oracle（每次探测 30ms，时间片 = 400ms/4 = 100ms）只烧掉本候选
        的时间片；改坏表现（无时间片、单一全局 deadline）：首个候选烧光
        全部墙钟预算，分配变为 [14,0,0,0]。
        """
        candidates = [candidate(0x40000000 + 4 * i) for i in range(4)]
        clock = {"now": 1000.0}

        def fake_monotonic():
            return clock["now"]

        def evaluate(_candidate, _value):
            clock["now"] += 0.03  # 每次探测 30ms
            return "unchanged"

        prober = DifferentialProber(
            max_probes_per_candidate=34, max_total_probes=64, timeout_ms=400
        )
        import time as _time
        from unittest.mock import patch

        with patch.object(_time, "monotonic", side_effect=fake_monotonic):
            report = prober.probe(candidates, evaluate, "unchanged")
        # 候选 0-2 各烧掉约一个时间片（4 次探测 ≈ 120ms），候选 3 受总
        # deadline 约束只剩 2 次；没有任何候选被前面的慢候选饿死。
        self.assertEqual(
            [v.probes_used for v in report.verdicts], [4, 4, 4, 2]
        )
        self.assertEqual(report.stats["timeout_stops"], 4)
        self.assertEqual(report.stats.get("budget_exhausted", 0), 0)
        self.assertEqual(report.stats["probes"], 14)


class Step2DependentSitesSemanticsTests(unittest.TestCase):
    """步骤 2：dependent_sites 收窄为只含 dependent；unresolved_candidates。"""

    def test_dependent_sites_only_contains_proven_dependent(self):
        """保护：dependent_sites 只含存在性判据成立的候选。

        改坏表现：unresolved_budget 混入 dependent_sites（旧语义），
        无 witness 值的候选被当作已证明依赖读出。
        """
        dependent = candidate(0x40010000, baseline=0, width_bits=4)
        independent = candidate(0x40010004, baseline=0x3, width_bits=4)
        unresolved = candidate(0x40010008, baseline=0, width_bits=32)

        def evaluate(probe_candidate, value):
            if probe_candidate.address == 0x40010000:
                return value == 0x8  # bit3 翻转改变结果
            return "constant"

        prober = DifferentialProber(max_probes_per_candidate=6, max_total_probes=64)
        report = prober.probe(
            [dependent, independent, unresolved], evaluate, "constant"
        )
        self.assertEqual(
            [v.verdict for v in report.verdicts],
            ["dependent", "independent", "unresolved_budget"],
        )
        self.assertEqual(report.dependent_sites, (dependent,))
        self.assertEqual(report.unresolved_candidates, (unresolved,))
        self.assertNotIn(dependent, report.unresolved_candidates)
        self.assertNotIn(independent, report.dependent_sites)
        self.assertNotIn(independent, report.unresolved_candidates)

    def test_solver_and_report_dependent_sites_agree(self):
        """保护：同一报告两处读出（solver 过滤 vs report 属性）不再矛盾。

        改坏表现：report.dependent_sites 仍按旧口径把 unresolved_budget
        算进去，与 solver 只认 "dependent" 的过滤结果相反。
        """
        def changed(address, value):
            return address == 0x40050000 and value == 0xFFFFFFFF

        runner = FakeRunner(
            history=[
                (0x08000200, 0x40050004, True, 0x0),
                (0x08000100, 0x40050000, True, 0x0),
            ],
            static_accesses=[],
            changed=changed,
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request())
        self.assertIsNotNone(assignments)
        self.assertEqual(assignments[0].address, 0x40050000)
        report = solver.last_report
        # 两处同口径：只有 0x40050000 是已证明依赖；
        # 0x40050004 预算耗尽 → 保守保留在 unresolved_candidates。
        self.assertEqual(
            [c.address for c in report.dependent_sites], [0x40050000]
        )
        self.assertEqual(
            [c.address for c in report.unresolved_candidates], [0x40050004]
        )
        self.assertEqual(solver.stats["dependent_sites"], 1)
        self.assertEqual(solver.stats["unresolved_candidates"], 1)


class ReplayOutcomeFakeRunner:
    """事件流驱动的最小重放面：精确控制三分规则的各比较维度。

    baseline / outcomes 返回 ``(events, covered_bbs)``，events 为
    SimpleNamespace(address, original_taken) 序列；返回 None 表示该次
    重放不可恢复（restored=False）。_evaluate_branch_replay 按
    historical_runner 同款口径从事件流计算 branch_seen / 去重方向集 /
    failure_reason（successor 覆盖取 covered_bbs）。
    """

    def __init__(self, *, history, baseline, outcomes, static_accesses=()):
        from types import SimpleNamespace

        self.emulator = SimpleNamespace(mmio_access_history=list(history))
        self.prepared = SimpleNamespace(
            static_mmio_accesses=list(static_accesses)
        )
        self.baseline = baseline
        self.outcomes = outcomes
        self.replay_calls = []

    def _replay_mmio_from_snapshot(
        self, snapshot_entry, constraint_items, *, replay_instructions, replay_timeout
    ):
        self.replay_calls.append(tuple(constraint_items))
        if not constraint_items:
            restored = self.baseline is not None
            events, covered = self.baseline or ([], set())
            return (set(covered), {}, restored, list(events), [])
        item = constraint_items[0]
        result = self.outcomes(int(item.address), int(item.value))
        if result is None:
            return (set(), {}, False, [], [])
        events, covered = result
        return (set(covered), {}, True, list(events), [])

    def _evaluate_branch_replay(
        self, events, branch_bb, take_branch, desired_successor, covered
    ):
        directions = [
            bool(getattr(event, "original_taken", False))
            for event in events or []
            if int(getattr(event, "address", 0) or 0) == int(branch_bb)
        ]
        unique = []
        for direction in directions:
            if direction not in unique:
                unique.append(direction)
        branch_seen = bool(directions)
        matched = bool(take_branch) in unique
        successor_seen = (
            desired_successor is None or desired_successor in (covered or set())
        )
        failure_reason = None
        if not branch_seen:
            failure_reason = "branch_not_reached"
        elif not matched:
            failure_reason = "wrong_direction"
        elif not successor_seen:
            failure_reason = "successor_not_covered"
        return {
            "success": branch_seen and matched and successor_seen,
            "failure_reason": failure_reason,
            "branch_seen": branch_seen,
            "matched_direction": matched,
            "desired_successor_seen": successor_seen,
            "observed_directions": [
                "taken" if direction else "not-taken" for direction in unique
            ],
            "observed_event_count": len(directions),
        }


def _branch_event(taken):
    from types import SimpleNamespace

    return SimpleNamespace(address=0x9000, original_taken=bool(taken))


HISTORY_ONE_SITE = [(0x08000100, 0x40050000, True, 0x0)]
# 无截断预算：32 位基线 0 的扰动全集 33 个，cap=34 / total=64 不截断。
NO_TRUNCATION_REQUEST = dict(max_probes_per_candidate=34, max_total_probes=64)


class Step3ThreeStateRuleTests(unittest.TestCase):
    """步骤 3：三态 changed/unchanged/unavailable + 三分规则（含洞 A/B）。"""

    def test_mask_scenario_direction_change_via_classifier(self):
        """掩码场景走三分规则：方向（去重集）翻转才算 changed。

        改坏表现：三分规则缺失（旧口径把 failure_reason/整体不等当变化）
        或方向比较错位时，掩码场景的方向见证不再被发现。
        """
        baseline = BranchReplayOutcome(
            branch_seen=True,
            failure_reason="wrong_direction",
            observed_directions=("not-taken",),
            direction_sequence=("not-taken",),
        )
        witness_flip = BranchReplayOutcome(
            branch_seen=True,
            failure_reason=None,
            observed_directions=("taken",),
            direction_sequence=("taken",),
        )
        probe_candidate = ProbeCandidate(
            address=0x40050000, baseline_value=0x53, width_bits=8
        )
        witness = 0x53 ^ 0x4  # bit2 翻转

        def evaluate(_candidate, value):
            return witness_flip if value == witness else baseline

        prober = DifferentialProber(max_probes_per_candidate=34, max_total_probes=64)
        report = prober.probe(
            [probe_candidate],
            evaluate,
            baseline,
            classify=classify_branch_replay_outcome,
        )
        verdict = report.verdicts[0]
        self.assertEqual(verdict.verdict, "dependent")
        self.assertEqual(verdict.witness_value, witness)
        self.assertEqual(report.stats["changed_probes"], 1)

    def test_unrestorable_replay_is_not_no_change(self):
        """重放不可恢复（restored=False）不得被当作"无变化"（发现 3）。

        改坏表现：evaluate 把不可恢复映射回基线 outcome，全套扰动"全无
        变化"→ independent，依赖被静默吞掉且无任何计数。
        """
        runner = ReplayOutcomeFakeRunner(
            history=HISTORY_ONE_SITE,
            baseline=([_branch_event(False)], {0x9100}),
            outcomes=lambda _address, _value: None,  # 全部不可恢复
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request(**NO_TRUNCATION_REQUEST))
        self.assertIsNone(assignments)
        self.assertEqual(solver.stats["unresolved_oracle"], 33)
        self.assertEqual(solver.stats.get("verdict_independent", 0), 0)
        self.assertEqual(
            solver.last_report.verdicts[0].verdict, "unresolved_oracle"
        )
        self.assertIn(
            solver.last_report.verdicts[0].candidate,
            solver.last_report.unresolved_candidates,
        )

    def test_successor_only_change_not_dependent_but_counted(self):
        """successor/coverage 口径差异不进 dependent，只进新计数（问题 B）。

        基线方向已匹配（taken，success）；扰动只改变全局覆盖集使
        failure_reason 变为 successor_not_covered，方向解析与基线一致：
        判据必须仍为 unchanged / independent；旧口径会把它算成 dependent。
        """
        runner = ReplayOutcomeFakeRunner(
            history=HISTORY_ONE_SITE,
            baseline=([_branch_event(True)], {0x9100}),
            # 同方向事件、空覆盖集：successor_not_covered（方向仍 matched）
            outcomes=lambda _address, _value: ([_branch_event(True)], set()),
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request(**NO_TRUNCATION_REQUEST))
        self.assertIsNone(assignments)
        self.assertEqual(
            solver.last_report.verdicts[0].verdict, "independent"
        )
        self.assertEqual(solver.stats["changed_via_successor_only"], 33)
        self.assertEqual(solver.stats.get("verdict_dependent", 0), 0)

    def test_direction_sequence_change_counted_not_promoted(self):
        """洞 B：去重方向集相同、含重复序列不同 ⇒ 只计数不升级判据。

        循环场景基线单次 not-taken、扰动后两次 not-taken：去重集一致，
        序列变化只进 sequence_changed_same_directions，仍判 independent
        （不把 trip-count 变化当方向依赖）。
        """
        runner = ReplayOutcomeFakeRunner(
            history=HISTORY_ONE_SITE,
            baseline=([_branch_event(False)], {0x9100}),
            outcomes=lambda _address, _value: (
                [_branch_event(False), _branch_event(False)],
                {0x9100},
            ),
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request(**NO_TRUNCATION_REQUEST))
        self.assertIsNone(assignments)
        self.assertEqual(
            solver.last_report.verdicts[0].verdict, "independent"
        )
        self.assertEqual(solver.stats["sequence_changed_same_directions"], 33)
        self.assertEqual(solver.stats.get("verdict_dependent", 0), 0)

    def test_baseline_unreachable_marks_all_unavailable_without_probing(self):
        """洞 A：基线 branch_seen == False ⇒ 全部 unavailable，不跑扰动。

        改坏表现：基线不可达时仍跑扰动，"扰动后 branch_seen 变 True"按
        字面判 dependent——可达性变化被误当方向依赖（假阳性）。
        """
        runner = ReplayOutcomeFakeRunner(
            history=HISTORY_ONE_SITE,
            baseline=([], {0x9100}),  # 基线重放可达恢复但未见目标分支
            outcomes=lambda _address, _value: (
                [_branch_event(True)],
                {0x9100},
            ),
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request())
        self.assertIsNone(assignments)
        self.assertEqual(solver.stats["unavailable_baseline_unreachable"], 1)
        self.assertEqual(len(runner.replay_calls), 1)  # 只有控制重放
        self.assertEqual(
            solver.last_report.verdicts[0].verdict, "unavailable"
        )
        self.assertEqual(solver.last_report.stats.get("probes", 0), 0)

    def test_perturbation_branch_not_reached_is_unavailable(self):
        """扰动使执行未到目标分支 ⇒ unavailable，不算变化（反提案 1）。

        改坏表现：旧口径整体比较把 branch_seen 翻转当"变化"→ dependent，
        执行被破坏被记成方向依赖。
        """
        runner = ReplayOutcomeFakeRunner(
            history=HISTORY_ONE_SITE,
            baseline=([_branch_event(False)], {0x9100}),
            outcomes=lambda _address, _value: ([], {0x9100}),
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request(**NO_TRUNCATION_REQUEST))
        self.assertIsNone(assignments)
        self.assertEqual(solver.stats["unavailable_probes"], 33)
        self.assertEqual(
            solver.last_report.verdicts[0].verdict, "unavailable"
        )
        self.assertEqual(
            solver.last_report.verdicts[0].unavailable_probes, 33
        )
        self.assertEqual(solver.stats.get("verdict_dependent", 0), 0)


class Step4PcLevelOccurrenceTests(unittest.TestCase):
    """步骤 4：候选/见证 occurrence=None（pc 级全出现语义，最小步）。"""

    def test_candidates_and_assignment_carry_pc_level_occurrence(self):
        """保护：返回的 assignment 不再强绑 occurrence=1。

        改坏表现：候选/assignment 写死 occurrence=1，探测在"全出现强制"
        语义下取得证据，见证却按"仅第 1 次出现"语义验证（round2 1.1
        核实成立的语义错位）。
        """
        def changed(address, value):
            return address == 0x40050000 and (value & 0xF) == 0x3

        runner = FakeRunner(
            history=[(0x08000100, 0x40050000, True, 0x53)],
            static_accesses=[
                {"pc": 0x08000500, "address": 0x4005000C,
                 "access_type": "read", "width": 4},
            ],
            changed=changed,
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        candidates = solver._collect_candidates(4)
        self.assertGreaterEqual(len(candidates), 2)
        # 运行时与静态来源的候选都是 pc 级（全出现）语义
        for probe_candidate in candidates:
            self.assertIsNone(probe_candidate.occurrence)
        assignments = solver(_probe_request())
        self.assertIsNotNone(assignments)
        self.assertEqual(assignments[0].address, 0x40050000)
        self.assertIsNone(assignments[0].occurrence)
        # 探测强制与见证同一范围：强制项也是 pc 级（read_occurrence=None）
        for call in runner.replay_calls:
            for item in call:
                self.assertIsNone(item.read_occurrence)
                self.assertIsNotNone(item.read_pc)


class Step6RuntimeWidthFromStaticTests(unittest.TestCase):
    """步骤 6：运行时候选宽度优先取静态同地址读记录，取不到退 32。"""

    def test_runtime_candidate_width_prefers_static_record(self):
        """保护：8/16 位站点不再按 32 位铺满扰动（round1 发现 6）。

        改坏表现：运行时历史候选 width_bits 恒为 32——16 位站点生成
        32 位扰动集（高位扰动全部无效，白耗预算）。
        """
        runner = FakeRunner(
            history=[
                (0x08000100, 0x40000008, True, 0x33),
                (0x08000200, 0x4000000C, True, 0x44),
            ],
            static_accesses=[
                {"pc": 0x08000500, "address": 0x40000008,
                 "access_type": "read", "width": 2},   # 16 位
                {"pc": 0x08000600, "address": 0x4000000C,
                 "access_type": "write", "width": 1},  # 写记录不算宽度来源
            ],
            changed=lambda _address, _value: False,
        )
        solver = DifferentialProbeSolver(runner)
        by_address = {c.address: c for c in solver._collect_candidates(8)}
        self.assertEqual(by_address[0x40000008].width_bits, 16)
        self.assertEqual(by_address[0x4000000C].width_bits, 32)  # 无读记录 → 退 32
        values16 = perturbation_values(0x33, 16)
        self.assertEqual(len(values16), 18)  # 极值对 + 16 位翻转
        self.assertTrue(all(0 <= value <= 0xFFFF for value in values16))
        self.assertLess(len(values16), len(perturbation_values(0x33, 32)))


class Step7CommonMmioPredicateTests(unittest.TestCase):
    """步骤 7：solver 引用公共 _is_mmio_address（不新增第 7 处定义）。"""

    def test_is_mmio_address_references_common_definition(self):
        """保护：差分扰动链路的 MMIO 判定与 RegisterTracer 公共实现同源。

        改坏表现：solver 维护私有副本（旧状态）——语义漂移时与 tracer
        的候选口径不一致。
        """
        from lsgemu.register_tracer.register_tracer import RegisterTracer

        self.assertIs(
            DifferentialProbeSolver._is_mmio_address,
            RegisterTracer._is_mmio_address,
        )
        # 行为一致（含边界），实例调用同样走公共定义
        solver = DifferentialProbeSolver(None)
        for address, expected in [
            (0x3FFFFFFF, False), (0x40000000, True), (0x5FFFFFFF, True),
            (0x60000000, False), (0x20000000, False),
            (0xE0000000, True), (0xE00FFFFF, True), (0xE0100000, False),
        ]:
            self.assertEqual(
                DifferentialProbeSolver._is_mmio_address(address), expected
            )
            self.assertEqual(solver._is_mmio_address(address), expected)


if __name__ == "__main__":
    unittest.main()
