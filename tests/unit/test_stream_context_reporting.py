#!/usr/bin/env python3
"""Regression tests for stream replay context report serialization."""

from collections import Counter
import unittest

from lsgemu.historical_runner import HistoricalRunner


class StreamContextReportingTests(unittest.TestCase):
    def make_runner(self) -> HistoricalRunner:
        runner = object.__new__(HistoricalRunner)
        runner.stream_context_zero_new_counts = Counter()
        return runner

    def test_current_evidence_key_does_not_parse_e3_as_an_address(self):
        runner = self.make_runner()
        key = (
            "stream_api_summary",
            "uart_read",
            "root",
            "E3",
            0x08001234,
            5,
            17,
            3,
            tuple(),
            tuple(),
        )
        runner.stream_context_zero_new_counts[key] = 4

        record = runner._stream_context_key_record(key, 2)

        self.assertEqual("E3", record["evidence_class"])
        self.assertEqual("0x08001234", record["context_bb"])
        self.assertEqual(5, record["occurrence_index"])
        self.assertEqual(17, record["capture_order"])
        self.assertEqual(3, record["depth"])
        self.assertEqual(4, record["persistent_zero_new_count"])

    def test_pre_evidence_and_legacy_keys_remain_readable(self):
        runner = self.make_runner()
        pre_evidence = (
            "buffer",
            "rx_ring",
            "prefix",
            0x08002000,
            2,
            9,
            4,
            tuple(),
            tuple(),
        )
        legacy = ("buffer", "rx_ring", 0x08003000, 1, 7, 2)

        pre_record = runner._stream_context_key_record(pre_evidence, 1)
        legacy_record = runner._stream_context_key_record(legacy, 1)

        self.assertIsNone(pre_record["evidence_class"])
        self.assertEqual("0x08002000", pre_record["context_bb"])
        self.assertEqual("legacy", legacy_record["origin"])
        self.assertEqual("0x08003000", legacy_record["context_bb"])


if __name__ == "__main__":
    unittest.main()
