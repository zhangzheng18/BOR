#!/usr/bin/env python3
"""Contracts for bounded loop-classifier diagnostic windows."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from lsgemu.analysis.intelligent_loop_classifier import IntelligentLoopClassifier


class LoopHistoryRetentionTests(unittest.TestCase):
    def test_long_execution_retains_every_analysis_window_and_total_counts(self):
        with patch.dict(
            os.environ,
            {
                "LSGEMU_LOOP_BB_HISTORY_LIMIT": "256",
                "LSGEMU_LOOP_REGISTER_HISTORY_LIMIT": "16",
            },
        ):
            classifier = IntelligentLoopClassifier(static_bbs={})

        for index in range(900):
            classifier.record_execution(
                0x1000 + index * 4,
                {"r0": index, "r1": index + 1},
            )

        self.assertEqual(900, classifier.total_bb_observations)
        self.assertEqual(900, classifier.total_register_snapshots)
        self.assertLessEqual(len(classifier.bb_history), 512)
        self.assertGreaterEqual(len(classifier.bb_history), 256)
        self.assertLessEqual(len(classifier.register_snapshots), 32)
        self.assertGreaterEqual(len(classifier.register_snapshots), 16)
        self.assertEqual(0x1000 + 899 * 4, classifier.bb_history[-1])
        self.assertEqual(899, classifier.register_snapshots[-1]["r0"])
        self.assertGreater(classifier.bb_history_entries_discarded, 0)
        self.assertGreater(classifier.register_snapshots_discarded, 0)

        stats = classifier.get_statistics()["history_retention"]
        self.assertEqual(900, stats["bb_total"])
        self.assertEqual(900, stats["register_total"])

    def test_recent_cycle_detection_survives_window_compaction(self):
        with patch.dict(os.environ, {"LSGEMU_LOOP_BB_HISTORY_LIMIT": "256"}):
            classifier = IntelligentLoopClassifier(static_bbs={})

        for index in range(700):
            classifier.record_execution(0x2000 + index * 4)
        for bb in (0x3000, 0x3010, 0x3000, 0x3010):
            classifier.record_execution(bb)

        self.assertEqual([0x3000, 0x3010], classifier._find_recent_repeated_cycle())


if __name__ == "__main__":
    unittest.main()
