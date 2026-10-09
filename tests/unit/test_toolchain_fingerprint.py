#!/usr/bin/env python3
"""Contracts for executable availability and identity collection."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lsgemu.toolchain_fingerprint import executable_identity


class ToolchainFingerprintTests(unittest.TestCase):
    def test_known_help_marker_can_validate_nonzero_probe(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_tool_identity_") as tmpdir:
            executable = Path(tmpdir) / "tool"
            executable.write_text(
                "#!/bin/sh\nprintf 'usage: tool [-postscript name]\\n'\nexit 1\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)

            identity = executable_identity(
                executable,
                ("-h",),
                success_output_markers=("[-postscript",),
            )

        self.assertTrue(identity["available"])
        self.assertFalse(identity["probe_succeeded"])
        self.assertTrue(identity["success_output_marker_matched"])
        self.assertEqual(1, identity["returncode"])

    def test_path_command_records_resolved_executable_file(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_tool_identity_") as tmpdir:
            executable = Path(tmpdir) / "path-tool"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            with patch.dict(os.environ, {"PATH": tmpdir}, clear=False):
                identity = executable_identity("path-tool")

        self.assertTrue(identity["available"])
        self.assertEqual(str(executable.resolve()), identity["path"])
        self.assertTrue(identity["exists"])
        self.assertTrue(identity["sha256"])


if __name__ == "__main__":
    unittest.main()
