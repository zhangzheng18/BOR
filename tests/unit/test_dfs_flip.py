"""Unit tests: flip = solve -> write back -> replay -> self-verify."""


from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import tempfile

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.dfs_anchor_pool import DFSAnchorEntry, DFSAnchorPool
from lsgemu.dfs_flip import (
    DFSFlipGuardrails,
    DFSFlipLedger,
    DFSFlipPlanner,
    DFSFlipTask,
    flip_edge_evidence,
    verify_flip_by_real_execution,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.runner_models import BranchConstraintCandidate
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from lsgemu.dfs_flip_kit import (
    ENTRY_PC,
    FLIP_BRANCH_BB,
    FLIP_BRANCH_PC,
    FLIP_COMPARE_LOOKUP,
    FLIP_FRAGMENT,
    FLIP_STATIC_BBS,
    LOAD_BASE,
    MMIO_ADDR,
    build_flip_emulator,
    build_flip_image,
)

try:
    from unicorn import UC_MODE_MCLASS
except ImportError:
    UC_MODE_MCLASS = 0

def _fake_event(addr: int, occurrence: int, taken: bool):
    return SimpleNamespace(
        address=addr,
        branch_pc=addr,
        target=addr + 0x10,
        fallthrough=addr + 4,
        condition="EQ",
        original_taken=taken,
        order=occurrence,
        depth=occurrence,
        occurrence_index=occurrence,
        alternatives=[],
        original_index=0 if taken else 1,
        original_direction_known=True,
        direction_provenance="unicorn_execution",
    )

class DFSFlipGuardrailTests(unittest.TestCase):
    def test_clean_replay_passes(self):
        emulator = SimpleNamespace(
            intervention_count=0,
            runtime_loop_branch_forces={},
            runtime_loop_branch_force_stats={"installed": 0, "applied": 0, "failed": 0},
            skip_function_stats={"skipped": 0},
            forced_branch_trace=[],
        )
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=emulator,
            run_result={
                "intervention_count": 0,
                "forced_branch_trace_count": 0,
                "forced_branch_choices_configured": 0,
            },
        )
        self.assertTrue(ok)
        self.assertEqual(violations, {})

    def test_each_counter_alone_rejects(self):
        def clean_emulator(**extra):
            base = dict(
                intervention_count=0,
                runtime_loop_branch_forces={},
                runtime_loop_branch_force_stats={"installed": 0, "applied": 0, "failed": 0},
                skip_function_stats={},
            )
            base.update(extra)
            return SimpleNamespace(**base)

        base_run_result = {
            "intervention_count": 0,
            "forced_branch_trace_count": 0,
            "forced_branch_choices_configured": 0,
        }
        cases = [
            ("intervention", clean_emulator(intervention_count=1), {}),
            ("loop_force_installed", clean_emulator(runtime_loop_branch_forces={0x10: {}}), {}),
            ("loop_force_stats_installed", SimpleNamespace(
                intervention_count=0,
                runtime_loop_branch_forces={},
                runtime_loop_branch_force_stats={"installed": 1, "applied": 0, "failed": 0},
                skip_function_stats={},
            ), {}),
            ("skip_function_stats", clean_emulator(skip_function_stats={"summarized": 3}), {}),
            ("forced_trace", clean_emulator(), {"forced_branch_trace_count": 2}),
            ("forced_choices", clean_emulator(), {"forced_branch_choices_configured": 1}),
        ]
        for name, emulator, extra in cases:
            run_result = dict(base_run_result)
            run_result.update(extra)
            ok, violations = DFSFlipGuardrails.evaluate(
                emulator=emulator, run_result=run_result
            )
            self.assertFalse(ok, name)
            self.assertTrue(violations, name)

class DFSFlipVerifyTests(unittest.TestCase):
    def _task(self, suffix=(), direction=True):
        return DFSFlipTask(
            branch_key=(0x08004106, 1),
            target_direction=direction,
            depth=3,
            suffix_signature=tuple(suffix),
        )

    def test_flipped_when_real_execution_took_direction(self):
        events = [
            _fake_event(0x08004000 + i, 1, i % 2 == 0)
            for i in range(3)
        ]
        events.append(_fake_event(0x08004106, 1, True))
        verdict, detail = verify_flip_by_real_execution(self._task(), events)
        self.assertEqual(verdict, "flipped")
        self.assertTrue(detail["matched"] >= 1)

    def test_direction_not_changed_when_replay_took_trunk_side(self):
        events = [_fake_event(0x08004106, 1, False)]
        verdict, _ = verify_flip_by_real_execution(self._task(), events)
        self.assertEqual(verdict, "direction_not_changed")

    def test_prefix_diverged_when_earlier_point_turned(self):
        task = self._task(
            suffix=(((0x08004010, 1), False),), direction=True
        )
        events = [
            _fake_event(0x08004010, 1, True),  # 后缀第一点走了另一边
            _fake_event(0x08004106, 1, True),
        ]
        verdict, _ = verify_flip_by_real_execution(task, events)
        self.assertEqual(verdict, "prefix_diverged")

    def test_not_reached_when_branch_missing(self):
        verdict, _ = verify_flip_by_real_execution(self._task(), [])
        self.assertEqual(verdict, "branch_occurrence_not_reached")

class DFSFlipPlannerTests(unittest.TestCase):
    def test_plan_orders_deep_to_shallow_and_skips_explored(self):
        trunk = [
            ((0x1000, 1), True),
            ((0x1100, 1), False),
            ((0x1200, 1), True),
        ]
        tasks = DFSFlipPlanner.plan(
            trunk,
            anchor_by_index={},
            already_explored={((0x1200, 1), False)},  # 最深点另一边已走过
            trunk_directions={},
        )
        self.assertTrue(tasks)
        depths = [task.depth for task in tasks]
        self.assertEqual(depths, sorted(depths, reverse=True))
        # 已探索的方向不再规划。
        self.assertNotIn((((0x1200, 1), False)), [
            (task.branch_key, task.target_direction) for task in tasks
        ])

    def test_plan_skips_non_binary_choices(self):
        trunk = [((0x1300, 1), 3)]
        tasks = DFSFlipPlanner.plan(trunk, anchor_by_index={}, already_explored=set())
        self.assertEqual(tasks, [])

class DFSFlipLedgerTests(unittest.TestCase):
    def test_success_and_unreachable_records(self):
        ledger = DFSFlipLedger()
        ledger.record_success((0x1000, 1), True, "E1")
        ledger.record_success((0x1000, 1), True, "E1")  # 幂等
        ledger.record_unresolved((0x1100, 1), False, "all_candidates_failed:x")
        payload = ledger.report_payload()
        self.assertEqual(len(payload["explored_direction_edges"]), 1)
        self.assertEqual(payload["explored_direction_edges"][0]["evidence"], "E1")
        # r11 ④：重放失败是 unresolved，不是 unreachable。
        self.assertEqual(len(payload["unresolved_directions"]), 1)
        self.assertEqual(len(payload["unreachable_directions"]), 0)

    def test_no_external_input_is_deduped_and_evidenced(self):
        ledger = DFSFlipLedger()
        evidence = {"cone_classification": {"r1": "internal_ram"}}
        ledger.record_no_external_input((0x2000, 1), True, evidence)
        ledger.record_no_external_input((0x2000, 1), True, evidence)  # 幂等
        payload = ledger.report_payload()
        self.assertEqual(len(payload["no_external_input_directions"]), 1)
        entry = payload["no_external_input_directions"][0]
        self.assertEqual(entry["reason"], "no_external_input_candidates")
        self.assertEqual(entry["evidence"]["cone_classification"], {"r1": "internal_ram"})
        self.assertEqual(
            payload["direction_counts"],
            {"explored": 0, "no_external_input": 1, "unresolved": 0, "proven_unreachable": 0},
        )
        # 已判无候选的边进入负结果缓存，供翻转 pass 跳过重规划。
        self.assertEqual(
            ledger.no_candidate_edge_set(), {((0x2000, 1), True)}
        )

    def test_unreachable_requires_proof(self):
        ledger = DFSFlipLedger()
        # 无证明的「无候选」不得写进不可达清单（口径④）。
        with self.assertRaises(ValueError):
            ledger.record_unreachable((0x2100, 1), False, "no_external_input_candidates")
        ledger.record_unreachable(
            (0x2100, 1),
            False,
            "proven_unreachable:z3_unsat_under_external_inputs",
        )
        payload = ledger.report_payload()
        self.assertEqual(len(payload["unreachable_directions"]), 1)
        self.assertEqual(
            payload["direction_counts"]["proven_unreachable"], 1
        )

    def test_flip_edge_evidence_is_e1(self):
        self.assertEqual(flip_edge_evidence(), "E1")

class DFSFlipReplayAcceptanceTests(unittest.TestCase):
    """验收口径：改输入前走 A、改输入后真实执行走 B，且强制计数全 0。"""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_dfs_flip_")
        self.firmware_path = str(Path(self._temporary.name) / "flip.bin")
        build_flip_image(self.firmware_path)

    def tearDown(self):
        self._temporary.cleanup()

    def _runner_shell(self, emulator: IntelligentEmulator) -> HistoricalRunner:
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = "none"
        runner.max_snapshots = 5
        runner.current_stage_name = "dfs_flip_test"
        runner.snapshot_resource_stats = {}
        runner.prepared = SimpleNamespace(entry_point=ENTRY_PC)
        runner.dfs_anchor_pool = None
        runner.dfs_flip_ledger = None
        # 环境装配走 stub；MMIO 约束执行、emulator.run、护栏与自证全部真实。
        runner._new_temp_emulator = lambda **_kwargs: emulator
        runner._configure_emulator_feedback = lambda *_args, **_kwargs: None
        runner._new_temp_register_tracer = lambda *_args, **_kwargs: None

        def _normalize(raw, execution_started=True):
            result = dict(raw) if isinstance(raw, dict) else {"stop_reason": str(raw)}
            result.setdefault("intervention_count", 0)
            return result

        runner._normalize_replay_run_result = _normalize
        runner.validate_coverage = lambda covered: set(covered or set())

        @contextlib.contextmanager
        def _capture(_emulator):
            yield set()

        runner.capture_coverage = _capture
        runner._finalize_temp_replay = lambda *args, **kwargs: None
        return runner

    def _run_flip_replay(self, runner, candidates):
        return runner._dfs_flip_replay_once(
            anchor=None,
            flip_candidates=candidates,
            replay_instructions=400,
            timeout_seconds=5.0,
        )

    def test_mmio_input_change_flips_direction_without_forcing(self):
        emulator = build_flip_emulator(self.firmware_path)
        runner = self._runner_shell(emulator)

        # 1) 改前：默认外设读值 != 0x20 ⇒ 真实执行走 B（fallthrough）。
        run_before, events_before, violations_before, _covered = self._run_flip_replay(
            runner, []
        )
        self.assertEqual(
            len(violations_before), 0, f"baseline replay has violations: {violations_before}"
        )
        verdict_before, _ = verify_flip_by_real_execution(
            DFSFlipTask(
                branch_key=(FLIP_BRANCH_BB, 1), target_direction=True, depth=0
            ),
            events_before,
        )
        self.assertEqual(
            verdict_before,
            "direction_not_changed",
            f"baseline should take fallthrough (B); events={[(e.address, e.occurrence_index, e.original_taken) for e in events_before[:6]]} stop={run_before.get('stop_reason')}",
        )

        # 2) 改后：回填外部输入候选 0x20 ⇒ 真实执行走 A（taken）。
        candidate = BranchConstraintCandidate(
            constraint_type="mmio",
            address=MMIO_ADDR,
            value=0x20,
            read_pc=ENTRY_PC + 2,
            constraint_pc=ENTRY_PC + 4,
            source="dfs_flip_test",
            input_kind="mmio",
            externally_controllable=True,
        )
        emulator_after = build_flip_emulator(self.firmware_path)
        runner_after = self._runner_shell(emulator_after)
        run_after, events_after, violations_after, covered_after = self._run_flip_replay(
            runner_after, [candidate]
        )
        self.assertEqual(
            len(violations_after), 0, f"flip replay has violations: {violations_after}"
        )
        verdict_after, _ = verify_flip_by_real_execution(
            DFSFlipTask(
                branch_key=(FLIP_BRANCH_BB, 1), target_direction=True, depth=0
            ),
            events_after,
        )
        self.assertEqual(
            verdict_after,
            "flipped",
            f"constrained replay should take branch (A); events={[(e.address, e.occurrence_index, e.original_taken) for e in events_after[:6]]} stop={run_after.get('stop_reason')}",
        )
        # 3) 硬护栏：两次重放的强制计数全 0。
        for run_result in (run_before, run_after):
            self.assertEqual(run_result.get("forced_branch_trace_count"), 0)
            self.assertEqual(run_result.get("forced_branch_choices_configured"), 0)
            self.assertEqual(run_result.get("intervention_count"), 0)
        self.assertEqual(getattr(emulator_after, "intervention_count", 0), 0)
        self.assertTrue(all(
            int(count or 0) == 0
            for count in dict(
                getattr(emulator_after, "skip_function_stats", None) or {}
            ).values()
        ))

class DFSFlipPassKnownCoverageTests(unittest.TestCase):
    """r10 ①：flip pass 的 known_coverage 绑定与语义（无需 unicorn）。"""

    def _shell(self) -> HistoricalRunner:
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.current_stage_name = "dfs_flip_test"
        runner.snapshot_resource_stats = {}
        runner.dfs_anchor_pool = None
        runner.dfs_flip_ledger = None
        runner.known_main_branch_events = {}
        runner.reservoir_explored_edges = set()
        runner.validate_coverage = lambda covered: set(covered or set())
        return runner

    def _eligible_anchor(self, signature, next_key) -> DFSAnchorEntry:
        # 直接构造合格锚点（绕过 consider 的血统校验，专测翻转消费侧）。
        return DFSAnchorEntry(
            anchor_key=(tuple(signature), tuple(next_key)),
            depth=len(signature),
            sequence=1,
            entry=object(),
            prefix_constraint_tables={"runner_scoped_constraints": []},
            flip_eligible=True,
        )

    def test_flip_new_covered_only_counts_bbs_outside_known_coverage(self):
        runner = self._shell()
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16)
        anchor = self._eligible_anchor(
            (((0x1000, 1), True),), (0x1100, 1)
        )
        pool.anchors[anchor.anchor_key] = anchor
        runner.dfs_anchor_pool = pool
        # 锚点 next_branch_key 的主干方向（False）进入 trunk_directions。
        runner.known_main_branch_events[(0x1100, 1)] = _fake_event(
            0x1100, 1, False
        )
        replay_events = [
            _fake_event(0x1000, 1, True),
            _fake_event(0x1100, 1, True),
        ]

        def _candidates(branch_key, target_direction, **_kwargs):
            return [SimpleNamespace(value=0x20)]

        def _replay_once(*, anchor, flip_candidates, **_kwargs):
            return (
                {"stop_reason": "max_instructions"},
                list(replay_events),
                {},
                {0xAA, 0xBB},
            )

        runner._naturalization_bootstrap_candidates = _candidates
        runner._dfs_flip_replay_once = _replay_once

        payload = runner._dfs_execute_flip_pass(known_coverage={0xAA})
        self.assertNotIn("error", payload)
        self.assertGreater(payload["attempt_stats"]["planned"], 0)
        self.assertGreater(payload["attempt_stats"]["attempts"], 0)
        self.assertGreater(len(payload["explored_direction_edges"]), 0)
        # 语义：0xAA 属于翻转前已知覆盖，重放再经过不算翻转新增。
        self.assertEqual(payload["flip_new_covered_bbs"], [0xBB])

    def test_known_coverage_defaults_to_empty_set(self):
        runner = self._shell()
        pool = DFSAnchorPool(stride=4, hot_limit=8, total_limit=16)
        anchor = self._eligible_anchor(
            (((0x1000, 1), True),), (0x1100, 1)
        )
        pool.anchors[anchor.anchor_key] = anchor
        runner.dfs_anchor_pool = pool
        runner.known_main_branch_events[(0x1100, 1)] = _fake_event(
            0x1100, 1, False
        )

        def _candidates(branch_key, target_direction, **_kwargs):
            return []

        runner._naturalization_bootstrap_candidates = _candidates
        payload = runner._dfs_execute_flip_pass(known_coverage=None)
        self.assertNotIn("error", payload)
        self.assertEqual(payload["flip_new_covered_bbs"], [])
        # r11 ④：无候选进 no_external_input_directions（带证据），不可达清单保持空。
        self.assertGreater(len(payload["no_external_input_directions"]), 0)
        self.assertEqual(len(payload["unreachable_directions"]), 0)
        for entry in payload["no_external_input_directions"]:
            self.assertEqual(entry["reason"], "no_external_input_candidates")
            self.assertIn("evidence", entry)

class DFSFlipPassEmptyQueueRegressionTests(unittest.TestCase):
    """r10 ① 回归：空队列 phase 的 flip pass 不得再抛 UnboundLocalError。

    r9D 现场（targeted_frontier_cycles，队列排干）：任务循环零迭代 ⇒
    ``run_reservoir_branch_exploration`` 的局部 ``known_coverage`` 从未绑定，
    调用翻转 pass 时在实参求值处（宽 except 内）只剩异常字符串。此处驱动
    真实方法走完空队列路径，断言 trunk report 的 flip 段干净可解释。
    """

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="lsgemu_dfs_reg_")
        self.firmware_path = str(Path(self._temporary.name) / "flip.bin")
        build_flip_image(self.firmware_path)
        self.report_path = str(
            Path(self._temporary.name) / "dfs_trunk_report.json"
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _runner_shell(self, emulator: IntelligentEmulator) -> HistoricalRunner:
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.constraint_file = None
        runner.runtime_branch_mmio_file_mode = "none"
        runner.max_snapshots = 5
        runner.current_stage_name = "targeted_frontier_cycles"
        runner.snapshot_resource_stats = {}
        runner.prepared = SimpleNamespace(
            entry_point=ENTRY_PC,
            static_successors={},
            reachable_from=None,
            static_bb_set=set(FLIP_STATIC_BBS),
        )
        runner.dfs_anchor_pool = None
        runner.dfs_flip_ledger = None
        runner.emulator = emulator
        runner.validate_coverage = lambda covered: set(covered or set())
        # 三个 reservoir 队列全空：任务循环零迭代（r9D 现场形态）。
        runner.reservoir_task_queue = []
        runner.reservoir_deferred_tasks = []
        runner.reservoir_scoped_probe_tasks = []
        runner.reservoir_initialized = True
        runner.known_main_branch_events = {}
        runner.known_branch_root_snapshots = {}
        runner._known_branch_root_snapshot_variant_count = lambda: 0
        runner.reservoir_explored_edges = set()
        runner.reservoir_explored_paths = set()
        runner.reservoir_incomplete_paths = set()
        runner.reservoir_pending_paths = set()
        runner.reservoir_discovered_branch_keys = set()
        runner.reservoir_new_branch_points = set()
        runner.reservoir_interrupt_contexts = []
        runner.interrupt_contexts = []
        runner.reservoir_root_budget_used = {}
        runner.reservoir_roots_started = {}
        runner.reservoir_roots_started_cumulative = {}
        runner.reservoir_saturated_edges = {}
        runner.reservoir_state_signatures = set()
        runner.reservoir_prefix_snapshots = {}
        runner.reservoir_state_file = None
        runner.remembered_root_snapshots = {}
        runner.reservoir_explored_edge_stats = {}
        runner.dynamic_successors = {}
        runner.branch_mmio_pending_root_keys = set()
        runner.branch_mmio_harvested_root_keys = set()
        runner.branch_snapshot_hotset = set()
        runner.scoped_branch_constraints = {}
        runner.global_coverage = set()
        runner.phase_coverage = {}
        runner.phase_metadata = {}
        runner.coverage_by_evidence = {}
        return runner

    def test_empty_queue_phase_reports_planned_zero_reason_not_error(self):
        emulator = build_flip_emulator(self.firmware_path)
        runner = self._runner_shell(emulator)
        with patch.dict(
            os.environ,
            {
                "LSGEMU_DFS_FLIP_ENABLED": "1",
                "LSGEMU_DFS_TRUNK_REPORT": self.report_path,
            },
        ):
            runner.run_reservoir_branch_exploration(
                phase_name="targeted_frontier_cycles",
            )
            report = json.loads(Path(self.report_path).read_text())
        flip_payload = report.get("flip") or {}
        self.assertNotIn("error", flip_payload, str(flip_payload))
        self.assertNotIn("error_traceback", flip_payload)
        self.assertIn("planned_zero_reason", flip_payload)
        self.assertIn(
            flip_payload["planned_zero_reason"],
            {"no_anchor_pool", "no_anchors", "empty_trunk_prefix"},
        )

if __name__ == "__main__":
    unittest.main()
