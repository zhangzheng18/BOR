#!/usr/bin/env python3
"""Regression tests for config-before-Unicorn runtime selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lsgemu.runtime_bootstrap import _sha256_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNICORN_BINDINGS = Path(
    os.environ.get("LSGEMU_UNICORN_BINDINGS")
    or (PROJECT_ROOT / "software" / "unicorn" / "bindings" / "python")
)
UNICORN_LIBRARY = PROJECT_ROOT / ".local" / "unicorn-hookfix" / "lib" / "libunicorn.so.2"


class RuntimeBootstrapTests(unittest.TestCase):
    def test_sha256_path_is_stable_for_runtime_library(self):
        digest = _sha256_path(UNICORN_LIBRARY)
        self.assertIsNotNone(digest)
        self.assertEqual(64, len(digest or ""))
        self.assertEqual(digest, _sha256_path(UNICORN_LIBRARY))

    @unittest.skipUnless(
        UNICORN_BINDINGS.is_dir(),
        "requires a local Unicorn 2.x Python bindings checkout "
        "(set LSGEMU_UNICORN_BINDINGS or place it under <root>/software/unicorn/)",
    )
    def test_command_line_config_is_applied_before_unicorn_import(self):
        config = """
schema: lsgemu.deployment_config.v1
project:
  source_root: {root}
  root: {root}
python:
  pythonpath:
    - {bindings}
unicorn:
  use_system_unicorn: 0
  local_python_bindings: {bindings}
  local_build_dir: {lib_dir}
  shared_library: {library}
  ld_library_path_entry: {lib_dir}
""".format(
            root=str(PROJECT_ROOT),
            bindings=str(UNICORN_BINDINGS),
            lib_dir=str(UNICORN_LIBRARY.parent),
            library=str(UNICORN_LIBRARY),
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as stream:
            stream.write(config)
            config_path = stream.name
        try:
            code = """
import json
import sys
sys.argv = ['probe', '--config', sys.argv[1]]
import lsgemu
import unicorn
info = lsgemu.RUNTIME_BOOTSTRAP_INFO
print(json.dumps({
    'config': lsgemu.DEPLOYMENT_CONFIG.get('_config_file'),
    'binding': info.get('unicorn_python_module'),
    'native': info.get('unicorn_native_library'),
    'binding_matches': info.get('unicorn_python_binding_matches'),
    'native_matches': info.get('unicorn_native_library_matches'),
    'identity_matches': info.get('unicorn_native_library_identity_matches'),
}))
"""
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            env.pop("LSGEMU_CONFIG_FILE", None)
            env.pop("LIBUNICORN_PATH", None)
            env.pop("LSGEMU_UNICORN_SHARED_LIB", None)
            completed = subprocess.run(
                [sys.executable, "-c", code, config_path],
                cwd=str(PROJECT_ROOT),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(str(Path(config_path).resolve()), payload["config"])
        self.assertEqual(str(UNICORN_LIBRARY.resolve()), payload["native"])
        self.assertTrue(payload["binding_matches"])
        self.assertTrue(payload["native_matches"])
        self.assertTrue(payload["identity_matches"])


if __name__ == "__main__":
    unittest.main()
