#!/usr/bin/env python3
"""Contracts for deterministic MMIO inference and its centralized fallback."""

from __future__ import annotations

import unittest

from lsgemu.analysis.intelligent_mmio_inferencer import IntelligentMMIOInferencer


class _UnexpectedAnalyzer:
    def __getattr__(self, name):
        raise AssertionError(f"local inferencer attempted LLM access through {name}")


class MMIOInferencerBoundaryTests(unittest.TestCase):
    def test_unresolved_local_rule_returns_to_centralized_fallback(self):
        inferencer = IntelligentMMIOInferencer(llm_analyzer=_UnexpectedAnalyzer())

        result = inferencer._infer_value(
            0x40001000,
            0x08000020,
            "ORR r3, r3, r4",
            [0x08000020],
            {0x08000020: []},
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
