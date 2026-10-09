#!/usr/bin/env python3
"""差分扰动定位（设计 §4.2，第二级）的回归测试。

三层：
1. 纯逻辑层（differential_probe.py）：扰动值集合、存在性判据的方向
   （含掩码场景——高位扰动不改变结果仍须判为有依赖）、预算保守性
   （预算耗尽保留候选而非排除）、确定性；
2. tracer 层：表达式图无该分支记录时触发二级定位，返回的 models 与
   z3/LLM 同通道（hypothesis）；有图记录时不触发；
3. 适配器层（differential_probe_solver.py）：候选来源（运行时历史 +
   静态清单）、重放 oracle 有界、见证值 assignment、上下文/开关门控。
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  # 触发包级 bootstrap
from lsgemu.analysis.differential_probe import (
    DifferentialProber,
    ProbeCandidate,
    perturbation_values,
)
from lsgemu.differential_probe_solver import DifferentialProbeSolver
from lsgemu.register_tracer.register_tracer import RegisterTracer

from unicorn.arm_const import UC_ARM_REG_CPSR, UC_ARM_REG_PC


def insn(address, mnemonic, operands):
    return {
        "address": int(address),
        "mnemonic": str(mnemonic),
        "operands": str(operands),
        "size": 2,
    }


class DummyUC:
    def __init__(self):
        self._registers = {UC_ARM_REG_CPSR: 1 << 30, UC_ARM_REG_PC: 0}

    def hook_add(self, *args, **kwargs):
        return 1

    def hook_del(self, handle):
        return None

    def reg_read(self, register):
        return int(self._registers.get(register, 0))


class StubProbeFallback:
    def __init__(self, assignments=None, error=None):
        self.requests = []
        self.assignments = list(assignments or [])
        self.error = error

    def __call__(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return list(self.assignments)


class PerturbationValueTests(unittest.TestCase):
    def test_extremes_first_then_bit_flips(self):
        values = perturbation_values(0x5A, 8)
        self.assertEqual(values[0], 0x0)
        self.assertEqual(values[1], 0xFF)
        self.assertNotIn(0x5A, values)  # 基线本身不探测
        self.assertEqual(len(values), len(set(values)))
        # 逐位翻转全集在场
        for bit in range(8):
            self.assertIn((0x5A ^ (1 << bit)) & 0xFF, values)

    def test_width_masked(self):
        values = perturbation_values(0x1FF, 8)
        self.assertTrue(all(0 <= value <= 0xFF for value in values))
        self.assertIn(0x0, values)
        self.assertNotIn(0xFF, values)  # 0x1FF & 0xFF == 基线，排除

    def test_baseline_zero_yields_pure_bit_set(self):
        values = perturbation_values(0, 4)
        self.assertNotIn(0, values)  # 与基线相同
        self.assertIn(0xF, values)
        for bit in range(4):
            self.assertIn(1 << bit, values)


class DifferentialProberTests(unittest.TestCase):
    def test_mask_scenario_high_bit_perturbations_do_not_hide_dependency(self):
        """掩码场景（任务 B6）：只有特定低位翻转暴露依赖，高位无效。

        结果仅在扰动值等于 ``baseline ^ 0x4`` 时变化——等价于
        ``(MMIO & mask) == target`` 中高位扰动不改变结果的情形。
        存在性判据必须判为「有依赖」，且见证值就是那个低位翻转。
        """
        baseline = 0x53
        candidate = ProbeCandidate(address=0x40050000, baseline_value=baseline, width_bits=8)
        witness = (baseline ^ 0x4) & 0xFF

        def evaluate(_candidate, value):
            return value == witness

        prober = DifferentialProber(max_probes_per_candidate=34, max_total_probes=64)
        report = prober.probe([candidate], evaluate, False)
        verdict = report.verdicts[0]
        self.assertEqual(verdict.verdict, "dependent")
        self.assertEqual(verdict.witness_value, witness)
        # 极值对与 bit0/bit7/bit1/bit6 翻转都不改变结果，直到 bit2 翻转才命中
        # （位序交错顺序 0,7,1,6,2,…：极值对 2 次 + 5 次翻转 = 第 7 次探测命中）
        self.assertEqual(verdict.probes_used, 7)
        self.assertGreaterEqual(verdict.changed_probes, 1)
        self.assertIn(candidate, report.dependent_sites)

    def test_most_perturbations_unchanged_still_dependent(self):
        """多数扰动不变、少数变化 ⇒ 有依赖（存在性而非全称判据）。"""
        candidate = ProbeCandidate(address=0x40050004, baseline_value=0, width_bits=8)

        def evaluate(_candidate, value):
            return value == 0x4  # 仅 bit2 翻转改变结果，其余扰动全部不变

        # 显式放宽预算：默认 per-candidate 上限为 4（预算自洽 6×4=24），
        # 掩码在位序交错下第 6 次才探测到，默认预算内探不到。
        prober = DifferentialProber(max_probes_per_candidate=34, max_total_probes=64)
        report = prober.probe([candidate], evaluate, False)
        self.assertEqual(report.verdicts[0].verdict, "dependent")

    def test_full_sweep_unchanged_proves_independent(self):
        candidate = ProbeCandidate(address=0x40050008, baseline_value=0x11, width_bits=4)

        def evaluate(_candidate, _value):
            return "constant-outcome"

        prober = DifferentialProber(max_probes_per_candidate=34, max_total_probes=64)
        report = prober.probe([candidate], evaluate, "constant-outcome")
        verdict = report.verdicts[0]
        self.assertEqual(verdict.verdict, "independent")
        self.assertIsNone(verdict.witness_value)
        self.assertEqual(len(report.dependent_sites), 0)
        # 全集确实被穷尽（2 极值 + 4 翻转 - 与基线重复的翻转）
        self.assertEqual(
            verdict.probes_used,
            len(perturbation_values(0x11, 4, limit=34)),
        )

    def test_budget_exhaustion_keeps_candidate_conservatively(self):
        """预算截断的扫描不得得出「无依赖」结论（保守保留，设计 §4.4）。

        极值对不改变结果的掩码场景里，只探测 2 次（极值对）的朴素做法
        正是设计警告的假阴性来源；此处必须返回 unresolved_budget。
        """
        candidate = ProbeCandidate(address=0x4005000C, baseline_value=0x53, width_bits=8)
        witness = (0x53 ^ 0x4) & 0xFF

        def evaluate(_candidate, value):
            return value == witness

        prober = DifferentialProber(max_probes_per_candidate=2, max_total_probes=8)
        report = prober.probe([candidate], evaluate, False)
        verdict = report.verdicts[0]
        self.assertEqual(verdict.verdict, "unresolved_budget")
        # 不漏（设计 §4.4）：保守保留改记入 unresolved_candidates
        # （round2 步骤 2 起 dependent_sites 仅含 dependent，与 solver 过滤同口径）
        self.assertIn(candidate, report.unresolved_candidates)
        self.assertNotIn(candidate, report.dependent_sites)
        self.assertEqual(report.stats["budget_exhausted"], 1)

    def test_total_budget_bounds_all_candidates(self):
        candidates = [
            ProbeCandidate(address=0x40050010 + index, baseline_value=0, width_bits=4)
            for index in range(4)
        ]
        calls = []

        def evaluate(candidate, value):
            calls.append((candidate.address, value))
            return "unchanged"

        # 每候选全集 5 个扰动（掩码 + 4 翻转）：总预算 5 恰好穷尽第一个候选，
        # 其余候选只能保守保留。
        prober = DifferentialProber(max_probes_per_candidate=34, max_total_probes=5)
        report = prober.probe(candidates, evaluate, "unchanged")
        self.assertEqual(len(calls), 5)
        self.assertEqual(report.stats["probes"], 5)
        verdicts = [v.verdict for v in report.verdicts]
        self.assertEqual(verdicts[0], "independent")
        self.assertEqual(verdicts.count("unresolved_budget"), 3)

    def test_wall_clock_timeout_keeps_candidate(self):
        candidate = ProbeCandidate(address=0x40050020, baseline_value=0, width_bits=16)

        def evaluate(_candidate, _value):
            time.sleep(0.02)
            return "unchanged"

        prober = DifferentialProber(timeout_ms=1)
        report = prober.probe([candidate], evaluate, "unchanged")
        self.assertEqual(report.verdicts[0].verdict, "unresolved_budget")
        self.assertGreaterEqual(report.stats["timeout_stops"], 1)

    def test_deterministic_probe_order(self):
        candidate = ProbeCandidate(address=0x40050024, baseline_value=0x0F, width_bits=8)
        probe_log = []

        def evaluate(_candidate, value):
            probe_log.append(value)
            return value == 0x0

        prober = DifferentialProber()
        report = prober.probe([candidate], evaluate, False)
        self.assertEqual(report.verdicts[0].verdict, "dependent")
        self.assertEqual(probe_log, perturbation_values(0x0F, 8, limit=34)[: len(probe_log)])


def make_tracer_with_branch_record(probe_fallback=None):
    """构造一个表达式图**已记录** 0x1006 分支的 tracer。"""
    static_bbs = {
        0x1000: [
            insn(0x1000, "LDRB", "r0, [r1]"),
            insn(0x1002, "CMP", "r0, #0x41"),
            insn(0x1004, "BEQ", "0x1010"),
        ]
    }
    tracer = RegisterTracer(
        DummyUC(),
        static_bbs,
        dynamic_graph_enabled=True,
        differential_probe_fallback=probe_fallback,
    )
    graph = tracer.dynamic_graph
    graph.observe_external_read(
        kind="mmio",
        read_pc=0x1000,
        address=0x40000000,
        occurrence=1,
        size=1,
        observed_value=0,
        destination_register="r0",
    )
    for item in static_bbs[0x1000][1:]:
        graph.observe_instruction(item, cpsr=0 if item["mnemonic"] == "BEQ" else None)
    return tracer


def make_tracer_without_branch_record(probe_fallback=None):
    """表达式图为空：任何分支都无记录（一级不可用）。"""
    return RegisterTracer(
        DummyUC(),
        {0x9000: [insn(0x9000, "NOP", "")]},
        dynamic_graph_enabled=True,
        differential_probe_fallback=probe_fallback,
    )


class TracerDifferentialProbeChainTests(unittest.TestCase):
    def test_probe_triggers_when_graph_has_no_branch_record(self):
        from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment

        assignments = [
            DynamicInputAssignment(
                kind="mmio",
                address=0x40050000,
                read_pc=0x08000100,
                occurrence=1,
                width=8,
                value=0x43,
                observed_value=0x53,
            )
        ]
        stub = StubProbeFallback(assignments)
        tracer = make_tracer_without_branch_record(stub)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x9000,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(len(stub.requests), 1)
        request = stub.requests[0]
        self.assertEqual(request.branch_pc, 0x9000)
        self.assertEqual(request.branch_occurrence, 1)
        self.assertTrue(request.target_taken)
        self.assertGreater(request.max_total_probes, 0)
        # 与 z3/LLM 相同的 hypothesis 通道
        self.assertEqual(result.solver_backend, "differential_probe")
        self.assertEqual(result.solver_status, "hypothesis")
        self.assertEqual(result.reason, "differential_probe_models")
        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].strategy, "differential_probe")
        self.assertEqual(result.models[0].assignments[0].value, 0x43)
        self.assertEqual(tracer.differential_probe_stats["calls"], 1)
        self.assertEqual(tracer.differential_probe_stats["success"], 1)

    def test_dead_end_preserved_when_probe_returns_nothing(self):
        stub = StubProbeFallback([])
        tracer = make_tracer_without_branch_record(stub)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x9000,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(result.models, ())
        self.assertEqual(result.reason, "dynamic_branch_predicate_missing")
        self.assertEqual(tracer.differential_probe_stats["no_assignments"], 1)

    def test_dead_end_unchanged_without_fallback(self):
        tracer = make_tracer_without_branch_record(None)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x9000,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(result.reason, "dynamic_branch_predicate_missing")
        self.assertEqual(len(tracer.differential_probe_stats), 0)

    def test_probe_not_triggered_when_graph_covers_branch(self):
        stub = StubProbeFallback()
        tracer = make_tracer_with_branch_record(stub)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x1004,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(len(stub.requests), 0)  # 一级可用，不扰动
        self.assertNotEqual(result.reason, "dynamic_branch_predicate_missing")

    def test_probe_stats_merge_and_clear(self):
        stub = StubProbeFallback([])
        tracer = make_tracer_without_branch_record(stub)
        other = make_tracer_without_branch_record(StubProbeFallback([]))
        other.recover_dynamic_branch_inputs(
            branch_pc=0x9000,
            occurrence=1,
            target_taken=True,
        )
        tracer.merge_from(other)
        self.assertEqual(tracer.differential_probe_stats["calls"], 1)
        tracer.clear()
        self.assertEqual(len(tracer.differential_probe_stats), 0)

    def test_probe_error_is_contained(self):
        stub = StubProbeFallback(error=RuntimeError("boom"))
        tracer = make_tracer_without_branch_record(stub)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=0x9000,
            occurrence=1,
            target_taken=True,
        )
        self.assertEqual(result.reason, "dynamic_branch_predicate_missing")
        self.assertEqual(tracer.differential_probe_stats["errors"], 1)


class FakeRunner:
    """最小重放面：按扰动值模拟分支结果是否变化。"""

    def __init__(self, *, history, static_accesses, changed, control_restored=True):
        self.emulator = SimpleNamespace(mmio_access_history=list(history))
        self.prepared = SimpleNamespace(static_mmio_accesses=list(static_accesses))
        self.changed = changed  # Callable[[address, value], bool]
        self.control_restored = control_restored
        self.replay_calls = []

    def _replay_mmio_from_snapshot(
        self, snapshot_entry, constraint_items, *, replay_instructions, replay_timeout
    ):
        self.replay_calls.append(tuple(constraint_items))
        item = constraint_items[0] if constraint_items else None
        if item is None:
            restored = self.control_restored
            events = ["control"] if self.control_restored else []
            return (set(), {}, restored, events, [])
        # 未改变结果的扰动必须产生与控制重放完全相同的结果摘要；
        # 改变结果的扰动产生不同的分支解析。
        changed = self.changed(int(item.address), int(item.value))
        events = ["evt"] if changed else ["control"]
        return (set(), {}, True, events, [])

    def _evaluate_branch_replay(
        self, events, branch_bb, take_branch, desired_successor, covered
    ):
        if events and events[0] == "evt":
            return {
                "success": True,
                "failure_reason": None,
                "branch_seen": True,
                "matched_direction": True,
                "desired_successor_seen": True,
                "observed_directions": ["taken"],
                "observed_event_count": 1,
            }
        if events:
            # 控制重放与未改变结果的扰动：分支可见但方向不符
            return {
                "success": False,
                "failure_reason": "wrong_direction",
                "branch_seen": True,
                "matched_direction": False,
                "desired_successor_seen": True,
                "observed_directions": ["not-taken"],
                "observed_event_count": 1,
            }
        return {
            "success": False,
            "failure_reason": "branch_not_reached",
            "branch_seen": False,
            "matched_direction": False,
            "desired_successor_seen": False,
            "observed_directions": [],
            "observed_event_count": 0,
        }


def _probe_request(**overrides):
    from lsgemu.analysis.differential_probe import DifferentialProbeRequest

    fields = dict(
        branch_pc=0x9000,
        branch_occurrence=1,
        target_taken=True,
        max_candidates=4,
        max_probes_per_candidate=34,
        max_total_probes=24,
        timeout_ms=60_000,
    )
    fields.update(overrides)
    return DifferentialProbeRequest(**fields)


def _bind(solver, snapshot_entry=object()):
    solver.bind_replay_context(
        snapshot_entry=snapshot_entry,
        branch_bb=0x9000,
        take_branch=True,
        desired_successor=0x9100,
        replay_instructions=200_000,
        replay_timeout=2,
    )


class DifferentialProbeSolverTests(unittest.TestCase):
    def test_mask_scenario_end_to_end_with_witness_assignment(self):
        # 0x40050000 的低 4 位等于 3 才改变结果（掩码场景）；
        # 观测基线 0x53（低 4 位即 3 之外的取值均无效）。
        def changed(address, value):
            return address == 0x40050000 and (value & 0xF) == 0x3

        runner = FakeRunner(
            history=[(0x08000100, 0x40050000, True, 0x53)],
            static_accesses=[],
            changed=changed,
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        assignments = solver(_probe_request())
        self.assertIsNotNone(assignments)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].address, 0x40050000)
        # 位序交错（0,31,…）下 bit31 翻转先于 bit4 被探测：
        # 0x80000053 低 4 位仍为 3，掩码场景高位翻转同样暴露依赖。
        self.assertEqual(assignments[0].value, 0x80000053)
        self.assertEqual(assignments[0].observed_value, 0x53)
        self.assertEqual(assignments[0].kind, "mmio")
        # 探测有界且可观测：1 次控制重放 + 4 次扰动重放
        # （极值对 + bit0 + bit31 翻转即命中）
        self.assertEqual(len(runner.replay_calls), 5)
        self.assertEqual(solver.stats["replays"], 4)
        self.assertEqual(solver.stats["verdict_dependent"], 1)
        self.assertEqual(solver.stats["witness_assignments"], 1)
        self.assertEqual(solver.last_report.verdicts[0].verdict, "dependent")

    def test_candidates_from_runtime_history_then_static(self):
        def changed(_address, _value):
            return False

        runner = FakeRunner(
            history=[
                (0x08000100, 0x40000000, True, 0x11),
                (0x08000200, 0x20000000, True, 0x99),  # 非 MMIO
                (0x08000300, 0x40000004, False, 0x22),  # 写，跳过
                (0x08000400, 0x40000008, True, 0x33),
            ],
            static_accesses=[
                {"pc": 0x08000500, "address": 0x4000000C, "access_type": "read", "width": 4},
                {"pc": 0x08000600, "address": 0x40000000, "access_type": "read", "width": 4},
                {"pc": 0x08000700, "address": None, "access_type": "read", "width": 4},
                {"pc": 0x08000800, "address": 0x40000010, "access_type": "write", "width": 4},
            ],
            changed=changed,
        )
        solver = DifferentialProbeSolver(runner)
        candidates = solver._collect_candidates(8)
        addresses = [candidate.address for candidate in candidates]
        # 运行时读在前（最新优先），静态只补未观测过的读地址
        self.assertEqual(addresses[:2], [0x40000008, 0x40000000])
        self.assertIn(0x4000000C, addresses)
        self.assertNotIn(0x20000000, addresses)
        self.assertNotIn(0x40000004, addresses)
        self.assertNotIn(0x40000010, addresses)
        self.assertEqual(len(addresses), len(set(addresses)))

    def test_no_context_or_disabled_returns_none(self):
        runner = FakeRunner(history=[], static_accesses=[], changed=lambda a, v: False)
        solver = DifferentialProbeSolver(runner)
        self.assertIsNone(solver(_probe_request()))
        self.assertEqual(solver.stats["skipped_no_context"], 1)
        _bind(solver)
        with patch.dict(os.environ, {"LSGEMU_DIFF_PROBE": "0"}):
            self.assertIsNone(solver(_probe_request()))
        self.assertEqual(solver.stats["skipped_disabled"], 1)

    def test_baseline_restore_failure_aborts(self):
        runner = FakeRunner(
            history=[(0x08000100, 0x40050000, True, 0x53)],
            static_accesses=[],
            changed=lambda a, v: False,
            control_restored=False,
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        self.assertIsNone(solver(_probe_request()))
        self.assertEqual(solver.stats["baseline_restore_failed"], 1)
        self.assertEqual(len(runner.replay_calls), 1)  # 只有控制重放

    def test_no_dependency_returns_none_with_full_sweep(self):
        runner = FakeRunner(
            history=[(0x08000100, 0x40050000, True, 0x00)],
            static_accesses=[],
            changed=lambda a, v: False,
        )
        solver = DifferentialProbeSolver(runner)
        _bind(solver)
        self.assertIsNone(solver(_probe_request(max_total_probes=64)))
        self.assertEqual(solver.stats["verdict_independent"], 1)
        self.assertEqual(solver.stats.get("witness_assignments", 0), 0)


if __name__ == "__main__":
    unittest.main()
