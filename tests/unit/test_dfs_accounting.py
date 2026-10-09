#!/usr/bin/env python3
"""r7 P3 单测：字段与记账（边集证据三元组 / R6-D2 守卫 / 快进计数 / 改名完备）。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.evidence_contract import _intervention_counter_from_mapping
from lsgemu.historical_runner import HistoricalRunner, _evidence_rank


class ExploredEdgeEvidenceTests(unittest.TestCase):
    def _runner(self) -> HistoricalRunner:
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.reservoir_explored_edges = set()
        return runner

    def test_record_edge_dedups_and_upgrades_evidence(self):
        runner = self._runner()
        runner._record_reservoir_explored_edge((0x1000, 1), True, "E2")
        self.assertEqual(len(runner.reservoir_explored_edges), 1)
        # 同边弱证据不覆盖
        runner._record_reservoir_explored_edge((0x1000, 1), True, "E3")
        self.assertEqual(len(runner.reservoir_explored_edges), 1)
        self.assertIn(((0x1000, 1), True, "E2"), runner.reservoir_explored_edges)
        # 同边强证据替换
        runner._record_reservoir_explored_edge((0x1000, 1), True, "E1")
        self.assertEqual(len(runner.reservoir_explored_edges), 1)
        self.assertIn(((0x1000, 1), True, "E1"), runner.reservoir_explored_edges)
        # 不同方向是不同边
        runner._record_reservoir_explored_edge((0x1000, 1), False, "E2")
        self.assertEqual(len(runner.reservoir_explored_edges), 2)

    def test_evidence_rank_order(self):
        self.assertGreater(_evidence_rank("E0"), _evidence_rank("E1"))
        self.assertGreater(_evidence_rank("E1"), _evidence_rank("E2"))
        self.assertGreater(_evidence_rank("E2"), _evidence_rank("E3"))
        self.assertGreater(_evidence_rank("E3"), _evidence_rank("unknown"))
        self.assertEqual(_evidence_rank(None), _evidence_rank("unknown"))

    def test_flip_planner_accepts_triple_edges(self):
        runner = self._runner()
        runner._record_reservoir_explored_edge((0x1200, 1), False, "E1")
        from lsgemu.dfs_flip import DFSFlipPlanner

        trunk = [
            ((0x1000, 1), True),
            ((0x1100, 1), False),
            ((0x1200, 1), True),
        ]
        tasks = DFSFlipPlanner.plan(
            trunk,
            anchor_by_index={},
            already_explored={
                (edge[0], edge[1]) for edge in runner.reservoir_explored_edges
            },
            trunk_directions={},
        )
        planned = {(task.branch_key, task.target_direction) for task in tasks}
        self.assertNotIn(((0x1200, 1), False), planned)


class RuntimeLoopForceGuardTests(unittest.TestCase):
    """R6-D2：三处循环强转 installer 默认关，守卫在函数体最前。"""

    BARE_SNAPSHOT = SimpleNamespace(
        bb_instructions=[
            {"address": "0x08004106", "mnemonic": "BEQ", "operands": "0x08004200"}
        ]
    )

    def test_guard_blocks_before_any_state_access(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE", None)
            # 裸实例无任何属性：守卫在最前 ⇒ 直接 False，不触碰属性。
            self.assertFalse(
                emulator._install_runtime_self_loop_branch_force(
                    self.BARE_SNAPSHOT, 0x08004100, {"address": 0x10, "value": 1}
                )
            )
            self.assertFalse(
                emulator._install_runtime_loop_branch_force(
                    [0x08004100, 0x08004106], 0x08004100, 0x40000000, 1
                )
            )
            self.assertFalse(
                emulator._install_runtime_loop_branch_force_at(
                    0x08004106, True, 0x08004100, 0x40000000, 1
                )
            )

    def test_env_on_passes_guard(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        with patch.dict(
            os.environ, {"LSGEMU_ENABLE_RUNTIME_LOOP_BRANCH_FORCE": "1"}
        ):
            # 越过守卫后裸实例必然在 instruction_to_bb 等属性上抛错——
            # 这恰好证明守卫确实放行且位于函数体最前。
            with self.assertRaises(AttributeError):
                emulator._install_runtime_self_loop_branch_force(
                    self.BARE_SNAPSHOT, 0x08004100, {"address": 0x10, "value": 1}
                )


class LoopFastForwardCounterTests(unittest.TestCase):
    def test_counter_records_total_and_by_handler(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.loop_fast_forward_emulation = {"total": 0, "by_handler": {}}
        self.assertTrue(emulator._record_loop_fast_forward("finite_memory_copy_loop"))
        self.assertTrue(emulator._record_loop_fast_forward("finite_memory_copy_loop"))
        self.assertTrue(emulator._record_loop_fast_forward("finite_byte_copy_loop"))
        stats = emulator.loop_fast_forward_emulation
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["by_handler"]["finite_memory_copy_loop"], 2)
        self.assertEqual(stats["by_handler"]["finite_byte_copy_loop"], 1)

    def test_intervention_counter_aggregates_fast_forward(self):
        # r9 裁定：快进族不再入干预集合（工程优化，覆盖照常计入）。
        counts = _intervention_counter_from_mapping(
            {"loop_fast_forward_emulation": {"total": 3, "by_handler": {}}}
        )
        self.assertNotIn("loop_fast_forward_emulation", counts)

    def test_intervention_labels_split_and_residual(self):
        # r9 裁定：家族标签拆账；loop_intervention 只留残差（诊断口径）。
        counts = _intervention_counter_from_mapping(
            {
                "intervention_count": 4,
                "intervention_event_labels": {
                    "loop_fast_forward_emulation": 2,
                    "loop_mmio_adjust": 1,
                    "loop_intervention": 1,
                },
            }
        )
        self.assertEqual(counts.get("loop_intervention"), 1)
        self.assertNotIn("loop_fast_forward_emulation", counts)
        self.assertNotIn("loop_mmio_adjust", counts)
        self.assertNotIn("loop_wait_handled", counts)

    def test_intervention_labels_absorb_full_count(self):
        # 全部事件都归家族时，loop_intervention 残差为 0 ⇒ 无干预理由。
        counts = _intervention_counter_from_mapping(
            {
                "intervention_count": 3,
                "intervention_event_labels": {
                    "loop_wait_handled": 2,
                    "loop_mmio_adjust": 1,
                },
            }
        )
        self.assertNotIn("loop_intervention", counts)
        self.assertNotIn("loop_wait_handled", counts)

    def test_legacy_intervention_count_without_labels(self):
        # 旧生产者（无标签）：行为不变，intervention_count 整体计为 loop_intervention。
        counts = _intervention_counter_from_mapping({"intervention_count": 2})
        self.assertEqual(counts.get("loop_intervention"), 2)

    def test_unresolved_limit_trips_are_blocking(self):
        counts = _intervention_counter_from_mapping(
            {"loop_unresolved_limit_trips": 2}
        )
        self.assertEqual(counts.get("loop_unresolved_limit"), 2)

    def test_preflight_repair_is_environment_fact(self):
        # r9 裁定：execution_preflight_repair = 环境事实（豁免）。
        counts = _intervention_counter_from_mapping(
            {
                "execution_preflight_stats": {
                    "passed": 1,
                    "thumb_state_corrected": 2,
                }
            }
        )
        self.assertNotIn("execution_preflight_repair", counts)


class FieldRenameContractTests(unittest.TestCase):
    """改名完备性：旧名清零、新名在场（phase metadata 无其他读者，R6-D5）。"""

    SOURCE = Path(
        __import__("lsgemu.historical_runner", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")

    def test_old_metadata_keys_removed(self):
        self.assertNotIn("forced_directions=", self.SOURCE)
        self.assertNotIn("forced_branch_edges=", self.SOURCE)

    def test_new_metadata_keys_present(self):
        self.assertIn("reservoir_tasks_run=tasks_run", self.SOURCE)
        self.assertIn("explored_direction_edges=len(", self.SOURCE)

    def test_dfs_trunk_report_writer_exists(self):
        self.assertIn("def _write_dfs_trunk_report", self.SOURCE)


if __name__ == "__main__":
    unittest.main()
