#!/usr/bin/env python3
"""r15 D1 单测：确定性快转在 force-free 翻转重放中的豁免与边界。

- 护栏只统计方向性强制：``loop_fast_forward_emulation`` 家族豁免；
  ``loop_intervention`` 家族、裸计数（无可依标签）、forced trace/choices
  照旧拒绝（反例——护栏没有被放宽）。
- 快转资格判据 ``_flip_fast_forward_eligible``：主跑路径不受影响；
  MMIO 环 / 未成形环不快转。
- 快转事件逐条记账（``loop_fast_forward_events`` 封顶）与开关函数。
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.dfs_flip import (
    DFSFlipGuardrails,
    dfs_flip_fast_forward_enabled,
)


def _bare_emulator(**attrs) -> IntelligentEmulator:
    emulator = IntelligentEmulator.__new__(IntelligentEmulator)
    for name, value in attrs.items():
        setattr(emulator, name, value)
    return emulator


class FlipFastForwardGuardrailTests(unittest.TestCase):
    """D1：快转家族豁免 + 方向性强制照旧拒绝。"""

    def test_exempts_fast_forward_family(self):
        # 3 次快转：intervention_count=3 但全部带 loop_fast_forward_emulation
        # 标签（r9 口径：计数↔标签 1:1）⇒ 方向性强制 = 0，护栏放行。
        emulator = SimpleNamespace(
            intervention_count=3,
            intervention_event_labels=Counter(
                {"loop_fast_forward_emulation": 3}
            ),
        )
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=emulator, run_result={}
        )
        self.assertTrue(ok)
        self.assertEqual(violations, {})

    def test_rejects_directional_intervention_mixed_with_fast_forward(self):
        # 反例：3 次干预里 1 次是 loop_intervention（本地约束——方向性强制）
        # ⇒ 仍拒绝，且只按方向性子集计数。
        emulator = SimpleNamespace(
            intervention_count=3,
            intervention_event_labels=Counter(
                {"loop_fast_forward_emulation": 2, "loop_intervention": 1}
            ),
        )
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=emulator, run_result={}
        )
        self.assertFalse(ok)
        self.assertEqual(violations["intervention_count"], 1)
        self.assertEqual(
            violations["intervention_count_detail"],
            {"total": 3, "fast_forward_exempt": 2},
        )

    def test_rejects_bare_count_without_labels(self):
        # 无标签可依（旧模拟器/异常路径）⇒ 全额计入——fail-closed。
        emulator = SimpleNamespace(intervention_count=1)
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=emulator, run_result={}
        )
        self.assertFalse(ok)
        self.assertEqual(violations["intervention_count"], 1)

    def test_forced_branch_trace_still_rejected(self):
        # 快转豁免不得波及其它判据：forced trace 照旧拒绝。
        emulator = SimpleNamespace(
            intervention_count=1,
            intervention_event_labels=Counter(
                {"loop_fast_forward_emulation": 1}
            ),
        )
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=emulator,
            run_result={"forced_branch_trace_count": 2},
        )
        self.assertFalse(ok)
        self.assertEqual(violations["forced_branch_trace_count"], 2)

    def test_run_result_only_evaluation_stays_strict(self):
        # emulator=None（纯 run_result 判定）无标签可依：任何计数都拒绝。
        ok, violations = DFSFlipGuardrails.evaluate(
            emulator=None, run_result={"intervention_count": 2}
        )
        self.assertFalse(ok)
        self.assertEqual(violations["intervention_count"], 2)


class FlipFastForwardEligibilityTests(unittest.TestCase):
    """D1 边界：资格判据（成形环 + 纯内存 + 仅翻转路径）。"""

    @staticmethod
    def _loop_info(**attrs) -> SimpleNamespace:
        base = {"iteration_count": 2, "has_mmio_access": False}
        base.update(attrs)
        return SimpleNamespace(**base)

    def _emulator(self, loop_info=None) -> IntelligentEmulator:
        return _bare_emulator(
            dfs_flip_deterministic_fast_forward=True,
            enable_loop_intervention=False,
            early_byte_copy_fast_forward=True,
            loop_classifier=SimpleNamespace(
                loop_heads={0x08005062: loop_info or self._loop_info()}
            ),
        )

    def test_eligible_for_pure_memory_loop(self):
        self.assertTrue(self._emulator()._flip_fast_forward_eligible(0x08005062))

    def test_not_eligible_when_flip_flag_off(self):
        emulator = self._emulator()
        emulator.dfs_flip_deterministic_fast_forward = False
        self.assertFalse(emulator._flip_fast_forward_eligible(0x08005062))

    def test_not_eligible_on_main_intervention_path(self):
        # 主跑路径（enable_loop_intervention=True）不走翻转豁免——行为不变。
        emulator = self._emulator()
        emulator.enable_loop_intervention = True
        self.assertFalse(emulator._flip_fast_forward_eligible(0x08005062))

    def test_not_eligible_for_unformed_loop(self):
        emulator = self._emulator(loop_info=self._loop_info(iteration_count=0))
        self.assertFalse(emulator._flip_fast_forward_eligible(0x08005062))

    def test_not_eligible_for_mmio_loop(self):
        # 触 MMIO 的环：外设副作用不可 O(1) 物化 ⇒ 一律逐条真实执行。
        emulator = self._emulator(loop_info=self._loop_info(has_mmio_access=True))
        self.assertFalse(emulator._flip_fast_forward_eligible(0x08005062))

    def test_not_eligible_for_unknown_loop_head(self):
        emulator = self._emulator()
        self.assertFalse(emulator._flip_fast_forward_eligible(0x08005064))


class FlipFastForwardEventLedgerTests(unittest.TestCase):
    """快转事件逐条记账（审计证据：哪些环被 O(1) 物化）。"""

    def test_events_recorded_with_loop_head(self):
        emulator = _bare_emulator(
            loop_fast_forward_emulation={"total": 0, "by_handler": {}},
            loop_fast_forward_events=[],
        )
        self.assertTrue(
            emulator._record_loop_fast_forward(
                "finite_memory_initialization_loop", loop_head=0x08005062
            )
        )
        self.assertEqual(
            emulator.loop_fast_forward_events,
            [(0x08005062, "finite_memory_initialization_loop")],
        )
        # 不带 loop_head（兼容旧调用）：只计数，不记事件。
        self.assertTrue(emulator._record_loop_fast_forward("finite_memory_copy_loop"))
        self.assertEqual(len(emulator.loop_fast_forward_events), 1)
        stats = emulator.loop_fast_forward_emulation
        self.assertEqual(stats["total"], 2)

    def test_events_capped_at_64(self):
        emulator = _bare_emulator(
            loop_fast_forward_emulation={"total": 0, "by_handler": {}},
            loop_fast_forward_events=[],
        )
        for index in range(80):
            emulator._record_loop_fast_forward("h", loop_head=0x1000 + index * 4)
        self.assertEqual(len(emulator.loop_fast_forward_events), 64)
        self.assertEqual(emulator.loop_fast_forward_emulation["total"], 80)

    def test_flag_default_on_env_off(self):
        self.assertTrue(dfs_flip_fast_forward_enabled())
        with patch.dict(os.environ, {"LSGEMU_DFS_FLIP_FAST_FORWARD": "0"}):
            self.assertFalse(dfs_flip_fast_forward_enabled())


if __name__ == "__main__":
    unittest.main()
