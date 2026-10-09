#!/usr/bin/env python3
"""
LSGEmu package.

The repo historically mixed package-style imports with direct script execution.
This file makes the package boundary explicit for the modernized scripts.
"""

# ``lsgemu`` is imported before ``lsgemu.lsgemu`` can process its command-line
# arguments.  Apply the optional ``--config`` selection at the package boundary
# first; otherwise importing this package would load Unicorn with the default
# library and a later config could no longer replace the native handle.
from .deployment_config import apply_config_from_argv

try:
    DEPLOYMENT_CONFIG = apply_config_from_argv()
except Exception:
    # Keep import errors reportable by the actual entrypoint.  In particular,
    # argparse should still be able to render a useful error for a bad config.
    DEPLOYMENT_CONFIG = {}

from .runtime_bootstrap import bootstrap_runtime_dependencies

RUNTIME_BOOTSTRAP_INFO = bootstrap_runtime_dependencies()

__all__ = [
    "HistoricalRunner",
    "PreparedFirmware",
    "DEPLOYMENT_CONFIG",
    "RUNTIME_BOOTSTRAP_INFO",
]


def __getattr__(name):
    if name in {"HistoricalRunner", "PreparedFirmware"}:
        from .historical_runner import HistoricalRunner, PreparedFirmware

        exports = {
            "HistoricalRunner": HistoricalRunner,
            "PreparedFirmware": PreparedFirmware,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
