#!/usr/bin/env python3
"""cycle3 k.5 C7（P1-C/P1-D）：进度/报告面键——零执行语义变更。

FINAL_PLAN_CYCLE3 §4 C7 断言面：
1. ``natural_supported_bbs`` 刷新键在场且来源标 canonical_validated；
2. run-result 面 ``instruction_count`` 携带 ``is_bb_counted=True``
   （该计数器按规范化 BB 自增——报告计量坑的显式化）；
3. U2 修复版判据五联词的机读面（真实 phase_metadata 形态：顶层
   ``no_new_bb_tail`` + 嵌套 ``run_result``）；
4. K4 地址+符号双写（偶基 + |1 双口径，成员判定在偶基）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.historical_runner import u2_fixed_criterion
from lsgemu.run_elfmultifuzz_campaign import build_k4_symbol_watch


class NaturalSupportedRefreshKeyTests(unittest.TestCase):
    def test_report_face_carries_refresh_provenance_key(self):
        # 键在场（报告面 build_report 产物的字面键集，源码级防回归）。
        import inspect

        from lsgemu import historical_runner as hr

        source = inspect.getsource(hr.HistoricalRunner.build_report)
        self.assertIn('"natural_supported_bbs": len(naturally_supported)', source)
        self.assertIn(
            '"natural_supported_bbs_refreshed_from": "canonical_validated"',
            source,
        )
        # naturally_supported 即 canonical_validated（刷新语义的赋值源）。
        self.assertIn(
            "naturally_supported = canonical_validated", source
        )


class IsBbCountedFlagTests(unittest.TestCase):
    def test_last_run_result_flags_bb_units(self):
        # 真 producer：IntelligentEmulator.last_run_result（即 phase_metadata
        # 嵌套 run_result 的来源字典）跑一次小执行后必须携带单位旗标。
        import tempfile

        import lsgemu  # noqa: F401  (unicorn 绑定补丁)
        from lsgemu.dfs_flip_kit import (
            ENTRY_PC,
            build_flip_emulator,
            build_flip_image,
        )

        with tempfile.TemporaryDirectory(prefix="lsgemu_k5_c7_") as tmp:
            firmware_path = str(Path(tmp) / "flip.bin")
            build_flip_image(firmware_path)
            emulator = build_flip_emulator(firmware_path)
            try:
                emulator.run(entry_point=ENTRY_PC, max_instructions=200)
                run_result = emulator.last_run_result
                self.assertIn("instruction_count", run_result)
                self.assertIs(run_result["is_bb_counted"], True)
            finally:
                try:
                    emulator.uc.close()
                except Exception:
                    pass


class U2FixedCriterionTests(unittest.TestCase):
    def test_five_conjuncts_from_real_phase_meta_shape(self):
        # 真 armP3 报告形态：顶层 no_new_bb_tail + 嵌套 run_result。
        phase_meta = {
            "no_new_bb_tail": 0,
            "run_result": {
                "instruction_count": 108628,
                "stop_reason": "fatal_sink_terminal",
                "is_bb_counted": True,
            },
        }
        face = u2_fixed_criterion(phase_meta)
        self.assertIs(face["no_new_bb_tail_zero"], True)
        self.assertIs(face["fatal_sink_terminal"], True)
        self.assertEqual(face["instruction_count"], 108628)
        self.assertIs(face["is_bb_counted"], True)

    def test_nonzero_tail_and_other_stop_reason_read_honestly(self):
        face = u2_fixed_criterion({
            "no_new_bb_tail": 3,
            "run_result": {"instruction_count": 10, "stop_reason": "timeout_reached"},
        })
        self.assertIs(face["no_new_bb_tail_zero"], False)
        self.assertIs(face["fatal_sink_terminal"], False)
        self.assertIs(face["is_bb_counted"], False)

    def test_none_meta_is_total_zero(self):
        face = u2_fixed_criterion(None)
        self.assertEqual(face["instruction_count"], 0)
        self.assertIs(face["fatal_sink_terminal"], False)


class K4SymbolWatchTests(unittest.TestCase):
    SPEC = (
        "AP_Vehicle::setup@0x08077981,"
        "Copter::init_ardupilot@0x0801BBC1,"
        "SVC_Handler@0x0812ED34,"
        "TIM5_vector@0x08135AB9"
    )

    def test_dual_write_address_and_symbol_with_even_base(self):
        covered = {0x08077980}  # 只覆盖第一个（偶基）
        validated = set()
        watch = build_k4_symbol_watch(self.SPEC, covered, validated)
        self.assertTrue(watch["enabled"])
        self.assertEqual(len(watch["entries"]), 4)
        first = watch["entries"][0]
        # 双写：符号 + 偶基地址 + |1 地址；成员判定在偶基（|1 输入被剥位）。
        self.assertEqual(first["symbol"], "AP_Vehicle::setup")
        self.assertEqual(first["address"], "0x08077980")
        self.assertEqual(first["address_thumb"], "0x08077981")
        self.assertIs(first["covered"], True)
        self.assertIs(first["validated"], False)
        self.assertEqual(watch["covered_count"], 1)
        self.assertEqual(watch["validated_count"], 0)
        for entry in watch["entries"][1:]:
            self.assertIs(entry["covered"], False)

    def test_disabled_when_spec_empty(self):
        watch = build_k4_symbol_watch(None, {1}, {1})
        self.assertIs(watch["enabled"], False)
        self.assertEqual(watch["entries"], [])


if __name__ == "__main__":
    unittest.main()
