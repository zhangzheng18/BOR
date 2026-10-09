#!/usr/bin/env python3
"""Contracts for serial campaign resource and I/O telemetry."""

from __future__ import annotations

import unittest

from experiments.run_lsgemu_serial_eval import parse_proc_io, public_resource_usage


class SerialResourceSamplingTests(unittest.TestCase):
    def test_proc_io_parser_keeps_supported_nonnegative_counters(self):
        parsed = parse_proc_io(
            "rchar: 100\n"
            "wchar: 200\n"
            "syscr: 7\n"
            "read_bytes: 4096\n"
            "write_bytes: -1\n"
            "cancelled_write_bytes: 12\n"
        )

        self.assertEqual({
            "rchar": 100,
            "wchar": 200,
            "read_bytes": 4096,
            "write_bytes": 0,
        }, parsed)

    def test_private_sampler_state_is_not_persisted(self):
        public = public_resource_usage({
            "samples": 3,
            "process_tree_write_bytes_lower_bound": 8192,
            "_io_last_by_pid": {10: {"write_bytes": 8192}},
        })

        self.assertEqual(3, public["samples"])
        self.assertEqual(8192, public["process_tree_write_bytes_lower_bound"])
        self.assertNotIn("_io_last_by_pid", public)


if __name__ == "__main__":
    unittest.main()
