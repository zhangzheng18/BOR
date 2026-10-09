#!/usr/bin/env python3
"""Regression tests for replay-safe persistence of learned loop exits."""

import json
from pathlib import Path
import tempfile
import unittest

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.analysis.stateful_mmio_handler import StatefulMMIOHandler
from lsgemu.mmio_handler.enhanced_mmio_handler import EnhancedMMIOHandler


class OccurrenceScopedLoopConstraintTests(unittest.TestCase):
    def test_internal_unicorn_fragment_is_not_a_static_branch_transition(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.static_bbs = {
            0x080057FC: [],
            0x08005806: [],
        }
        emulator.instruction_to_bb = {
            0x080057FC: 0x080057FC,
            0x080057FE: 0x080057FC,
            0x08005800: 0x080057FC,
            0x08005804: 0x080057FC,
            0x08005806: 0x08005806,
        }

        self.assertTrue(
            emulator._is_internal_unicorn_bb_fragment(
                0x08005800,
                0x080057FC,
                0x080057FC,
            )
        )
        self.assertFalse(
            emulator._is_internal_unicorn_bb_fragment(
                0x080057FC,
                0x080057FC,
                0x080057FC,
            )
        )
        self.assertFalse(
            emulator._is_internal_unicorn_bb_fragment(
                0x08005806,
                0x08005806,
                0x080057FC,
            )
        )

    def test_internal_unicorn_fragment_does_not_advance_runtime_state(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.instruction_count = 17
        emulator.internal_unicorn_bb_fragments = 0
        emulator._external_rom_summary_target = None
        emulator.static_bbs = {0x080057FC: []}
        emulator.instruction_to_bb = {
            0x080057FC: 0x080057FC,
            0x08005800: 0x080057FC,
        }
        emulator.bb_history = [0x080057FC]
        emulator._ensure_dynamic_basic_block = lambda _address: 0x080057FC
        time_events = []
        emulator.time_handler = type(
            "TimeHandlerProbe",
            (),
            {
                "tick": lambda self: time_events.append("tick"),
                "handle_function_return": (
                    lambda self, _uc, _address: time_events.append("return")
                ),
            },
        )()

        emulator.bb_hook(None, 0x08005800, 2, None)

        self.assertEqual(17, emulator.instruction_count)
        self.assertEqual([0x080057FC], emulator.bb_history)
        self.assertEqual(1, emulator.internal_unicorn_bb_fragments)
        self.assertEqual([], time_events)

    def test_current_input_occurrence_uses_exact_read_site(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.input_occurrence_counts = {
            (0x08001000, 0x40002000): 51,
        }
        self.assertEqual(
            51,
            emulator._current_input_read_occurrence(0x08001000, 0x40002000),
        )
        self.assertIsNone(
            emulator._current_input_read_occurrence(0x08001002, 0x40002000)
        )

    def test_occurrence_persistence_replaces_legacy_broad_loop_rule(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_occurrence_loop_") as tmp:
            path = Path(tmp) / "constraints.json"
            path.write_text(json.dumps({
                "constraints": [{
                    "type": "mmio",
                    "read_pc": "0x08001000",
                    "address": "0x40002000",
                    "value": "0x00000001",
                    "constraint_pc": "0x08001002",
                    "description": "Local loop-exit MMIO constraint from loop @ 0x08001000",
                    "added_by": "intelligent_emulator",
                    "iteration": 0,
                }]
            }), encoding="utf-8")

            emulator = IntelligentEmulator.__new__(IntelligentEmulator)
            emulator.constraint_json_path = str(path)
            emulator.iteration = 0
            emulator.enable_dynamic_memory_constraints = False
            emulator._save_constraint_to_json({
                "type": "mmio",
                "read_pc": 0x08001000,
                "address": 0x40002000,
                "value": 2,
                "constraint_pc": 0x08001002,
                "read_occurrence": 51,
                "description": "Local loop-exit MMIO constraint from loop @ 0x08001000",
            })

            items = json.loads(path.read_text(encoding="utf-8"))["constraints"]
            self.assertEqual(1, len(items))
            self.assertEqual(51, items[0]["read_occurrence"])

            primary = StatefulMMIOHandler(constraint_json_path=str(path))
            self.assertNotIn((0x08001000, 0x40002000), primary.static_constraints)

            overlay = EnhancedMMIOHandler(
                object(),
                str(path),
                branch_mmio_file_mode="scoped",
            )
            self.assertNotIn((0x08001000, 0x40002000), overlay.pc_constraints)
            self.assertEqual(
                2,
                overlay.occurrence_constraints[
                    (0x08001000, 0x40002000, 51)
                ],
            )


if __name__ == "__main__":
    unittest.main()
