#!/usr/bin/env python3
"""Contracts for bounded-memory artifact persistence and publication."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lsgemu.artifact_io import atomic_json_dump, atomic_publish_file


class ArtifactIOTests(unittest.TestCase):
    def test_streamed_compact_json_round_trips(self):
        payload = {
            "records": [{"index": index, "value": "x" * 32} for index in range(128)],
            "nested": {"enabled": True},
        }
        with tempfile.TemporaryDirectory(prefix="lsgemu_artifact_") as tmpdir:
            path = Path(tmpdir) / "report.json"
            atomic_json_dump(payload, path, indent=None, durable=False)
            encoded = path.read_text(encoding="utf-8")

        self.assertEqual(payload, json.loads(encoded))
        self.assertTrue(encoded.endswith("\n"))
        self.assertNotIn("\n  ", encoded)

    def test_publish_uses_hardlink_on_same_filesystem(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_artifact_") as tmpdir:
            root = Path(tmpdir)
            source = root / "attempt" / "report.json"
            destination = root / "stable" / "report.json"
            source.parent.mkdir()
            source.write_bytes(b"immutable-report")

            atomic_publish_file(source, destination, durable=False)

            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertTrue(source.samefile(destination))


if __name__ == "__main__":
    unittest.main()
