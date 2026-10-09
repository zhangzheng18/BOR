#!/usr/bin/env python3
"""Ensure the ARM-only execution backend rejects other ELF architectures."""

from __future__ import annotations

from pathlib import Path
import unittest

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator


class ArchitectureContractTests(unittest.TestCase):
    def test_non_arm_elf_is_rejected_before_execution(self):
        candidate = Path("/bin/true")
        if not candidate.exists():
            self.skipTest("host does not provide /bin/true")
        with self.assertRaisesRegex(ValueError, "supports 32-bit ARM/Cortex-M only"):
            IntelligentEmulator(
                firmware_path=str(candidate),
                static_bbs={},
                constraint_json_path=None,
                max_snapshots=1,
                llm_config_path=None,
            )


if __name__ == "__main__":
    unittest.main()
