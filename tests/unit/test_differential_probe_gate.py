#!/usr/bin/env python3
"""差分扰动二级定位默认臂可达性（方案 A3）与观测落盘的回归测试。

round2 第 3 轮（docs/TASK_probe_gate_A3_20260918.md）四件事的验收：

- (i) choke point 可达性：默认臂（自然化 bootstrap 路径，语义义务开启）
  下探针不再永远 ``skipped_no_context``；
- (ii) 无快照时行为不变：``skipped_no_context`` 递增、死路 reason 与
  空候选集与改动前一致；
- (iii) solver 错配护栏：绑定分支 Y 以分支 X 请求被拒；合法调用
  （branch_pc = 分支指令地址 ≠ branch_bb 的常态场景）不被误拒；
- (iv) 预算熔断：``LSGEMU_DIFF_PROBE_GLOBAL_PROBE_BUDGET=0`` 时全链
  skip、零重放；
- (v) 探针计数落盘：``differential_probe_progress_summary`` 的键结构、
  ``probe_phase_ran`` 两形态与 checkpoint 接线；
- (T5) 探针见证 ``occurrence=None`` 贯通到 ``add_pc_constraint``（pc 级、
  全出现），整数 occurrence 仍走 occurrence 分支（防回归）；
- 消融臂防护：外层（generic_dynamic_fallback）已绑定时 choke point
  不重复绑定、不越权清除。

每条测试的 docstring 标注它保护什么、改坏了哪一处会红。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import lsgemu  # noqa: F401  # 触发包级 bootstrap
from lsgemu.differential_probe_solver import DifferentialProbeSolver
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.register_tracer.register_tracer import RegisterTracer

from unicorn.arm_const import UC_ARM_REG_CPSR, UC_ARM_REG_PC

# 基本块起始与分支指令地址刻意不同（T2 护栏的常态场景：
# request.branch_pc 是分支指令地址、context.branch_bb 是 BB 起始）。
BRANCH_BB = 0x9000
BRANCH_PC = 0x9004
SUCCESSOR_BB = 0x9100


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
    """tracer 探针钩子的可编程替身（见证值直接给定）。"""

    def __init__(self, assignments=None, error=None):
        self.requests = []
        self.assignments = list(assignments or [])
        self.error = error

    def __call__(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return list(self.assignments)


def make_tracer_without_branch_record(probe_fallback=None):
    """表达式图为空：任何分支都无记录（一级不可用）。"""
    return RegisterTracer(
        DummyUC(),
        {BRANCH_BB: [insn(BRANCH_PC, "NOP", "")]},
        dynamic_graph_enabled=True,
        differential_probe_fallback=probe_fallback,
    )


def probe_request(**overrides):
    from lsgemu.analysis.differential_probe import DifferentialProbeRequest

    fields = dict(
        branch_pc=BRANCH_PC,
        branch_occurrence=1,
        target_taken=True,
        max_candidates=4,
        max_probes_per_candidate=8,
        max_total_probes=24,
        timeout_ms=60_000,
    )
    fields.update(overrides)
    return DifferentialProbeRequest(**fields)


class RecordingMMIO:
    """记录约束落入哪条通道（pc 级 / occurrence 级 / 地址全局）。"""

    def __init__(self):
        self.calls = []

    def add_pc_constraint(self, *args, **kwargs):
        self.calls.append(("pc", args, kwargs))

    def add_occurrence_constraint(self, *args, **kwargs):
        self.calls.append(("occurrence", args, kwargs))

    def add_constraint(self, *args, **kwargs):
        self.calls.append(("plain", args, kwargs))


def make_gate_runner(
    *,
    tracer=None,
    snapshot_entries=None,
    history=None,
    changed=None,
):
    """choke point 全链最小替身：object.__new__ + 打桩，不建仿真器。

    ``snapshot_entries=None`` 表示「有可重放快照」（默认给一个），
    传 ``[]`` 表示无快照。``changed`` 是扰动是否改变分支结果的判定
    （掩码场景默认：低 4 位等于 3 才改变）。
    """
    runner = object.__new__(HistoricalRunner)
    runner.register_tracer = tracer
    runner.known_main_branch_events = {}
    runner.differential_probe_solver = DifferentialProbeSolver(runner)
    runner.prepared = SimpleNamespace(
        get_branch_instruction=lambda bb: {
            "address": BRANCH_PC,
            "size": 2,
            "mnemonic": "BNE",
            "operands": "#0x9100",
        },
        static_mmio_accesses=[],
    )
    runner.emulator = SimpleNamespace(
        mmio_access_history=list(
            history
            if history is not None
            else [(0x08000100, 0x40050000, True, 0x53)]
        )
    )
    runner.snapshot_lookups = []

    def _snapshot_entries(bb, max_variants=3):
        runner.snapshot_lookups.append((bb, max_variants))
        return list(snapshot_entries if snapshot_entries is not None else [object()])

    runner._branch_snapshot_replay_entries = _snapshot_entries
    runner._branch_desired_successor_bb = (
        lambda instruction, take_branch: SUCCESSOR_BB
    )
    runner._constraint_compare_pc = lambda pc: pc + 4
    if changed is None:
        changed = lambda address, value: (int(value) & 0xF) == 0x3
    runner.replay_calls = []

    def _replay_mmio_from_snapshot(
        snapshot_entry, constraint_items, *, replay_instructions, replay_timeout
    ):
        runner.replay_calls.append(tuple(constraint_items))
        item = constraint_items[0] if constraint_items else None
        if item is None:
            return (set(), {}, True, ["control"], [])
        perturbed = changed(int(item.address), int(item.value))
        events = ["evt"] if perturbed else ["control"]
        return (set(), {}, True, events, [])

    def _evaluate_branch_replay(
        events, branch_bb, take_branch, desired_successor, covered
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

    runner._replay_mmio_from_snapshot = _replay_mmio_from_snapshot
    runner._evaluate_branch_replay = _evaluate_branch_replay
    return runner


class ChokePointReachabilityTests(unittest.TestCase):
    """(i)/(ii)：choke point 补绑与无快照时的旧行为。"""

    def test_i_default_arm_probe_reachable_through_choke_point(self):
        """保护：默认臂（自然化 bootstrap → _naturalization_dynamic_
        candidate_sets）经 choke point 补绑后探针真正运行、护栏不误拒。
        改坏了会红：choke point 补绑被移除/被条件挡住（未打补丁时
        solver.calls==0 且 skipped_no_context>=1）、或 T2 护栏把合法调用
        （branch_pc=0x9004 ≠ branch_bb=0x9000）误拒成
        skipped_context_mismatch、或 finally 未清理自己绑定的上下文。"""
        tracer = make_tracer_without_branch_record()
        runner = make_gate_runner(
            tracer=tracer,
            snapshot_entries=[object()],
        )
        tracer.differential_probe_fallback = runner.differential_probe_solver
        candidate_sets = runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1),
            True,
            max_models=2,
        )
        solver = runner.differential_probe_solver
        # 探针被真正调用（未打补丁时为 0）
        self.assertGreaterEqual(solver.stats["calls"], 1)
        # 不再死于「无上下文」
        self.assertEqual(solver.stats["skipped_no_context"], 0)
        # T2 护栏没有把合法调用误拒（branch_pc != branch_bb 的常态）
        self.assertEqual(solver.stats["skipped_context_mismatch"], 0)
        # 掩码场景存在依赖 → 见证 → 候选集非空
        self.assertTrue(candidate_sets)
        self.assertEqual(solver.stats["witness_assignments"], 1)
        # choke point 自己绑定的上下文在 finally 里已清理
        self.assertFalse(solver.has_replay_context)
        # 快照自解析发生在与 tracer 请求同源的 branch_bb 上
        self.assertEqual(runner.snapshot_lookups, [(BRANCH_BB, 3)])

    def test_ii_no_snapshot_preserves_old_behaviour(self):
        """保护：无可用快照时 choke point 绑 snapshot_entry=None，solver
        照旧 skipped_no_context、上层仍为死路（reason 保持
        dynamic_branch_predicate_missing、候选集为空）。
        改坏了会红：choke point 不再自解析快照（snapshot_lookups 为空
        ——未打补丁时不发生解析）、或无快照被错误当作可重放上下文
        （skipped_no_context 不再递增 / reason 漂移）。"""
        tracer = make_tracer_without_branch_record()
        runner = make_gate_runner(
            tracer=tracer,
            snapshot_entries=[],
        )
        tracer.differential_probe_fallback = runner.differential_probe_solver
        candidate_sets = runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1),
            True,
            max_models=2,
        )
        solver = runner.differential_probe_solver
        # 无快照 → 旧行为：no_context 早退、零探测、零重放
        self.assertEqual(solver.stats["skipped_no_context"], 1)
        self.assertEqual(solver.stats["calls"], 0)
        self.assertEqual(runner.replay_calls, [])
        # 上层候选集与改动前一致：死路返回、空集
        self.assertEqual(candidate_sets, [])
        self.assertEqual(
            runner.last_dynamic_recovery_result.reason,
            "dynamic_branch_predicate_missing",
        )
        # 未打补丁时 choke point 根本不解析快照 → 该断言红
        self.assertEqual(runner.snapshot_lookups, [(BRANCH_BB, 3)])

    def test_ablation_outer_binding_is_not_rebound_or_cleared(self):
        """保护：外层（消融臂 generic_dynamic_fallback）已绑定上下文时，
        choke point 不重复绑定（补绑条件 has_replay_context 优先）、
        try/finally 不越权清除外层绑定。
        改坏了会红：choke point 无条件重绑（snapshot_lookups 非空、
        外层的快照/方向被覆盖）、或 finally 清掉了不属于自己的上下文
        （调用后 has_replay_context 变 False）。"""
        tracer = make_tracer_without_branch_record()
        runner = make_gate_runner(tracer=tracer, snapshot_entries=[])
        solver = runner.differential_probe_solver
        outer_snapshot = object()
        solver.bind_replay_context(
            snapshot_entry=outer_snapshot,
            branch_bb=BRANCH_BB,
            take_branch=True,
            desired_successor=SUCCESSOR_BB,
            replay_instructions=1000,
            replay_timeout=1000,
        )
        tracer.differential_probe_fallback = solver
        runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1),
            True,
            max_models=2,
        )
        # 外层绑定优先：choke point 没有再自解析、没有重绑
        self.assertEqual(runner.snapshot_lookups, [])
        # 外层绑定仍然在（未被越权清除），且仍持有外层快照
        self.assertTrue(solver.has_replay_context)
        self.assertIs(solver._context.snapshot_entry, outer_snapshot)
        solver.clear_replay_context()


class ContextMismatchGuardTests(unittest.TestCase):
    """(iii)：solver 错配护栏与合法调用不误拒。"""

    def test_iii_mismatched_branch_request_is_rejected(self):
        """保护：绑定分支 Y（branch_bb=0x9000 → 指令地址 0x9004）后以
        分支 X（branch_pc=0x8004）请求 → skipped_context_mismatch 且
        返回 None；方向不一致同样被拒。
        改坏了会红：护栏被移除（未打补丁时 mismatch 键为 0，请求会按
        Y 的上下文继续评估）。"""
        runner = make_gate_runner(snapshot_entries=[object()])
        solver = DifferentialProbeSolver(runner)
        solver.bind_replay_context(
            snapshot_entry=object(),
            branch_bb=BRANCH_BB,
            take_branch=True,
            desired_successor=SUCCESSOR_BB,
            replay_instructions=1000,
            replay_timeout=1000,
        )
        # 分支不一致：请求 0x8004，绑定还原出的指令地址是 0x9004
        self.assertIsNone(solver(probe_request(branch_pc=0x8004)))
        self.assertEqual(solver.stats["skipped_context_mismatch"], 1)
        self.assertEqual(solver.stats["calls"], 0)
        # 方向不一致：branch_pc 一致但 target_taken 相反
        self.assertIsNone(solver(probe_request(target_taken=False)))
        self.assertEqual(solver.stats["skipped_context_mismatch"], 2)
        self.assertEqual(solver.stats["calls"], 0)
        self.assertEqual(runner.replay_calls, [])

    def test_iii_legal_request_with_branch_pc_different_from_bb_is_not_rejected(self):
        """保护：合法调用不被误拒——branch_pc 是分支指令地址（0x9004）、
        branch_bb 是 BB 起始（0x9000），常态不相等；护栏必须经
        _branch_instruction 还原后比较，而非直接等值比较。直接比较
        branch_pc == branch_bb 的写法会在此红（skipped_context_mismatch>=1
        且 witness_assignments==0）。"""
        runner = make_gate_runner(snapshot_entries=[object()])
        solver = DifferentialProbeSolver(runner)
        solver.bind_replay_context(
            snapshot_entry=object(),
            branch_bb=BRANCH_BB,
            take_branch=True,
            desired_successor=SUCCESSOR_BB,
            replay_instructions=1000,
            replay_timeout=1000,
        )
        assignments = solver(probe_request())
        self.assertIsNotNone(assignments)
        self.assertEqual(solver.stats["skipped_context_mismatch"], 0)
        self.assertEqual(solver.stats["witness_assignments"], 1)


class GlobalBudgetCircuitBreakerTests(unittest.TestCase):
    """(iv)：全局预算熔断（全链）。"""

    def test_iv_zero_global_budget_skips_without_replay(self):
        """保护：LSGEMU_DIFF_PROBE_GLOBAL_PROBE_BUDGET=0 时，choke point
        补绑后探针在预算闸门处熔断——skipped_global_budget 递增、零重放。
        改坏了会红：choke point 未接通（未打补丁时死于
        skipped_no_context、skipped_global_budget==0）、或预算闸门被绕过
        （replay_calls 非空）。"""
        tracer = make_tracer_without_branch_record()
        runner = make_gate_runner(
            tracer=tracer,
            snapshot_entries=[object()],
        )
        tracer.differential_probe_fallback = runner.differential_probe_solver
        with patch.dict(
            os.environ, {"LSGEMU_DIFF_PROBE_GLOBAL_PROBE_BUDGET": "0"}
        ):
            candidate_sets = runner._naturalization_dynamic_candidate_sets(
                (BRANCH_BB, 1),
                True,
                max_models=2,
            )
        solver = runner.differential_probe_solver
        self.assertEqual(solver.stats["skipped_global_budget"], 1)
        self.assertEqual(solver.stats["calls"], 0)
        self.assertEqual(runner.replay_calls, [])
        self.assertEqual(candidate_sets, [])


class ProbeProgressCheckpointTests(unittest.TestCase):
    """(v)：探针计数落盘。"""

    def test_v_progress_summary_carries_both_counters_and_phase_state(self):
        """保护：differential_probe_progress_summary 同时携带 tracer 侧
        differential_probe 与 solver 侧 get_statistics 计数，且显式给出
        probe_phase_ran / path_naturalization_skip_reason——用于区分
        「宿主阶段被 skip」（skipped=True → False）与「跑了但 0 工作量」
        （未 skipped → True，计数可为全 0）。
        改坏了会红：helper 被移除（AttributeError）、键缺失、或
        probe_phase_ran 判定颠倒。"""
        tracer = make_tracer_without_branch_record()
        runner = make_gate_runner(
            tracer=tracer,
            snapshot_entries=[],
        )
        tracer.differential_probe_fallback = runner.differential_probe_solver
        # 先产生一份探针计数（无快照 → skipped_no_context）
        runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1), True, max_models=2
        )
        summary = runner.differential_probe_progress_summary()
        self.assertEqual(summary["tracer"].get("calls"), 1)
        self.assertEqual(summary["solver"].get("skipped_no_context"), 1)
        self.assertIn("probe_phase_ran", summary)
        self.assertIn("path_naturalization_skip_reason", summary)
        # 宿主阶段尚未记录 → probe_phase_ran False（区分于「跑了 0 工作量」）
        self.assertFalse(summary["probe_phase_ran"])

        # 宿主阶段被 skip（消融臂形态）：probe_phase_ran 必须为 False，
        # 且 skip_reason 可读——这是与「跑了但 0 工作量」的区分依据。
        runner.phase_metadata = {
            "path_naturalization": {
                "skipped": True,
                "skip_reason": "semantic_obligation_ablation_disabled",
            }
        }
        skipped_summary = runner.differential_probe_progress_summary()
        self.assertFalse(skipped_summary["probe_phase_ran"])
        self.assertEqual(
            skipped_summary["path_naturalization_skip_reason"],
            "semantic_obligation_ablation_disabled",
        )

        # 跑了但 0 工作量（默认臂、计数全 0）：probe_phase_ran 必须为 True。
        runner.phase_metadata = {
            "path_naturalization": {"skipped": False, "elapsed_seconds": 0.5}
        }
        ran_summary = runner.differential_probe_progress_summary()
        self.assertTrue(ran_summary["probe_phase_ran"])
        self.assertIsNone(ran_summary["path_naturalization_skip_reason"])

    def test_v_checkpoint_writer_wires_probe_summary(self):
        """保护：周期 coverage checkpoint（write_coverage_checkpoint）里
        确实接入探针摘要——helper 存在但没接线时（SIGTERM 丢 finalize 报告
        的场景下没有任何落盘点）该断言红。（cycle2 k.5 H1：调用被提升到
        dict 外的局部变量以配 write_timing 戳，接线性质不变。）"""
        source = Path(HistoricalRunner.__module__.replace(".", "/") + ".py")
        if not source.is_file():
            source = Path(sys.modules[HistoricalRunner.__module__].__file__)
        text = source.read_text(encoding="utf-8")
        start = text.index("def write_coverage_checkpoint(")
        end = text.index("def write_progress(", start)
        writer = text[start:end]
        self.assertIn(
            "differential_probe_summary = self.differential_probe_progress_summary()",
            writer,
        )
        self.assertIn('"differential_probe": differential_probe_summary,', writer)


class ProbeWitnessPcLevelOccurrenceTests(unittest.TestCase):
    """T5：探针见证 occurrence=None 贯通到 pc 级验证。"""

    @staticmethod
    def _witness_assignment(occurrence):
        from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment

        return DynamicInputAssignment(
            kind="mmio",
            address=0x40050000,
            read_pc=0x08000100,
            occurrence=occurrence,
            width=8,
            value=0x03,
            observed_value=0x53,
        )

    def test_probe_witness_occurrence_none_reaches_pc_level_constraint(self):
        """保护：探针见证（occurrence=None，pc 级/全出现语义）经 tracer
        接缝（不再归一为 1）→ runner 候选（read_occurrence=None）→
        _apply_temp_mmio_constraints 命中 add_pc_constraint，而不是
        add_occurrence_constraint（仅第 1 次出现）——「探测强制范围 =
        见证验证范围」。
        改坏了会红：tracer 接缝恢复 or 1 归一（assignments[0].occurrence
        变 1）、runner 侧 read_occurrence 退化 max(1, None)（TypeError 或
        occurrence 通道被命中）。"""
        from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment

        stub = StubProbeFallback(
            [DynamicInputAssignment(
                kind="mmio",
                address=0x40050000,
                read_pc=0x08000100,
                occurrence=None,
                width=8,
                value=0x03,
                observed_value=0x53,
            )]
        )
        tracer = make_tracer_without_branch_record(stub)
        result = tracer.recover_dynamic_branch_inputs(
            branch_pc=BRANCH_PC,
            occurrence=1,
            target_taken=True,
        )
        # tracer 接缝保留 None（未打补丁时被归一为 1 → 红）
        self.assertIsNone(result.models[0].assignments[0].occurrence)

        runner = make_gate_runner(tracer=tracer, snapshot_entries=[])
        tracer.differential_probe_fallback = stub
        candidate_sets = runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1),
            True,
            max_models=2,
        )
        self.assertTrue(candidate_sets)
        witness = candidate_sets[0][0]
        self.assertIsNone(witness.read_occurrence)

        runner._scoped_replay_available = lambda: True
        temp_mmio = RecordingMMIO()
        runner._apply_temp_mmio_constraints(temp_mmio, candidate_sets[0])
        # pc 级（全出现），不是 occurrence 级
        self.assertEqual(
            [call[0] for call in temp_mmio.calls],
            ["pc"],
            f"expected pc-level only, got {temp_mmio.calls}",
        )

    def test_integer_occurrence_candidate_keeps_occurrence_channel(self):
        """防回归：occurrence 为整数的普通候选（z3/LLM 通道）仍按出现
        次序走 add_occurrence_constraint——T5 只为探针见证放宽 None，
        不得改变整数语义。改坏了会红：整数 occurrence 被错误转成 None
        （pc 通道被命中）。"""
        from lsgemu.dynamic_constraint_recovery import DynamicInputAssignment

        stub = StubProbeFallback(
            [DynamicInputAssignment(
                kind="mmio",
                address=0x40050000,
                read_pc=0x08000100,
                occurrence=3,
                width=8,
                value=0x03,
                observed_value=0x53,
            )]
        )
        tracer = make_tracer_without_branch_record(stub)
        runner = make_gate_runner(tracer=tracer, snapshot_entries=[])
        candidate_sets = runner._naturalization_dynamic_candidate_sets(
            (BRANCH_BB, 1),
            True,
            max_models=2,
        )
        self.assertTrue(candidate_sets)
        witness = candidate_sets[0][0]
        self.assertEqual(witness.read_occurrence, 3)

        runner._scoped_replay_available = lambda: True
        temp_mmio = RecordingMMIO()
        runner._apply_temp_mmio_constraints(temp_mmio, candidate_sets[0])
        self.assertEqual(
            [call[0] for call in temp_mmio.calls],
            ["occurrence"],
            f"expected occurrence-level only, got {temp_mmio.calls}",
        )
        # 落点带原来的 occurrence 序号（第 3 次出现）
        self.assertEqual(temp_mmio.calls[0][1][2], 3)


if __name__ == "__main__":
    unittest.main()
