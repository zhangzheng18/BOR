#!/usr/bin/env python3
"""
Runtime dependency bootstrap for local LSGEmu workflows.

This keeps third-party compatibility shims in one place:
1. Prefer workspace-local `.vendor` packages when present.
2. In the fuzzware venv, prefer the locally checked-out Unicorn 2.x bindings.
"""

from __future__ import annotations

import os
import sys
import importlib
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

from .deployment_config import configured_path, load_and_apply_deployment_config, source_root


logger = logging.getLogger(__name__)


DEPLOYMENT_CONFIG = load_and_apply_deployment_config()

SRCV4_ROOT = configured_path("LSGEMU_SOURCE_ROOT", source_root())
PROJECT_ROOT = configured_path("LSGEMU_PROJECT_ROOT", SRCV4_ROOT.parent)
VENDOR_DIR = SRCV4_ROOT / ".vendor"
LOCAL_UNICORN_BINDINGS = configured_path(
    "LSGEMU_UNICORN_PYTHON_BINDINGS",
    PROJECT_ROOT / "unicorn" / "bindings" / "python",
)
LOCAL_UNICORN_BUILD = configured_path("LIBUNICORN_PATH", PROJECT_ROOT / "unicorn" / "build")
LOCAL_UNICORN_SHARED_LIB = configured_path(
    "LSGEMU_UNICORN_SHARED_LIB",
    LOCAL_UNICORN_BUILD / "libunicorn.so.2",
)


def _prepend_sys_path(path: Path) -> bool:
    if not path.exists():
        return False

    path_str = str(path)
    if path_str in sys.path:
        return False

    sys.path.insert(0, path_str)
    return True


def _sha256_path(path: Optional[Path]) -> Optional[str]:
    """Return a file digest for runtime identity diagnostics."""
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return None
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _loaded_unicorn_status(expected_binding: Optional[Path], expected_library: Optional[Path]) -> Dict[str, object]:
    """Describe the Unicorn implementation actually loaded in this process.

    ``sys.path`` only describes what a future import would select.  Once a
    caller has imported ``unicorn``, changing ``sys.path`` cannot replace that
    module safely.  Recording the loaded module and native handle makes this
    distinction explicit in reports and prevents a path-injection claim from
    being mistaken for a runtime guarantee.
    """
    module = sys.modules.get("unicorn")
    if module is None:
        return {
            "loaded": False,
            "python_module": None,
            "python_version": None,
            "native_library": None,
            "native_library_sha256": None,
            "expected_native_library": str(expected_library.resolve())
            if expected_library is not None
            else None,
            "expected_native_library_sha256": _sha256_path(expected_library),
            "python_binding_matches": None,
            "native_library_matches": None,
            "native_library_identity_matches": None,
            "import_conflict": False,
        }

    module_file = getattr(module, "__file__", None)
    module_path = Path(module_file).resolve() if module_file else None
    version = getattr(module, "__version__", None)
    native_library = None
    try:
        unicorn_py3 = importlib.import_module("unicorn.unicorn_py3.unicorn")
        native_library = getattr(getattr(unicorn_py3, "uclib", None), "_name", None)
    except Exception:
        # Unicorn 1.x and alternate bindings do not expose unicorn_py3.
        native_library = None
    native_path = Path(native_library).resolve() if native_library else None
    binding_matches = None
    if expected_binding is not None and module_path is not None:
        try:
            binding_matches = expected_binding.resolve() in module_path.parents
        except OSError:
            binding_matches = False
    native_matches = None
    if expected_library is not None and native_path is not None:
        try:
            native_matches = native_path == expected_library.resolve()
        except OSError:
            native_matches = False
    expected_digest = _sha256_path(expected_library)
    loaded_digest = _sha256_path(native_path)
    identity_matches = None
    if expected_digest is not None and loaded_digest is not None:
        identity_matches = expected_digest == loaded_digest
    import_conflict = bool(
        binding_matches is False
        or native_matches is False
        or identity_matches is False
    )
    return {
        "loaded": True,
        "python_module": str(module_path) if module_path else None,
        "python_version": str(version) if version is not None else None,
        "native_library": str(native_path) if native_path else None,
        "native_library_sha256": loaded_digest,
        "expected_native_library": str(expected_library.resolve())
        if expected_library is not None
        else None,
        "expected_native_library_sha256": expected_digest,
        "python_binding_matches": binding_matches,
        "native_library_matches": native_matches,
        "native_library_identity_matches": identity_matches,
        "import_conflict": import_conflict,
    }


def bootstrap_runtime_dependencies() -> Dict[str, object]:
    """Inject local dependency overrides and return a small runtime report."""
    vendor_added = _prepend_sys_path(VENDOR_DIR)

    active_venv = os.environ.get("VIRTUAL_ENV", "")
    prefer_local_unicorn = (
        os.environ.get("LSGEMU_USE_SYSTEM_UNICORN") != "1"
        and LOCAL_UNICORN_BINDINGS.exists()
    )
    unicorn_added = False
    if prefer_local_unicorn:
        unicorn_added = _prepend_sys_path(LOCAL_UNICORN_BINDINGS)
        if LOCAL_UNICORN_SHARED_LIB.exists():
            os.environ.setdefault("LIBUNICORN_PATH", str(LOCAL_UNICORN_BUILD))

    # Import once after path/library setup so all formal LSGEmu entrypoints use
    # the same binding/native pair.  If a parent imported Unicorn first, keep
    # that module and report the conflict instead of mutating sys.modules.
    try:
        importlib.import_module("unicorn")
    except Exception:
        pass

    configured_libunicorn_path = os.environ.get("LIBUNICORN_PATH", "")

    loaded_status = _loaded_unicorn_status(
        LOCAL_UNICORN_BINDINGS if prefer_local_unicorn else None,
        LOCAL_UNICORN_SHARED_LIB if prefer_local_unicorn else None,
    )
    if loaded_status.get("import_conflict"):
        message = (
            "Unicorn runtime identity mismatch: "
            f"binding={loaded_status.get('python_module')!r}, "
            f"native={loaded_status.get('native_library')!r}, "
            f"expected={loaded_status.get('expected_native_library')!r}"
        )
        if os.environ.get("LSGEMU_REQUIRE_UNICORN_MATCH", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(message)
        logger.warning(message)
    return {
        "vendor_dir": str(VENDOR_DIR),
        "vendor_available": VENDOR_DIR.exists(),
        "vendor_injected": vendor_added or str(VENDOR_DIR) in sys.path,
        "prefer_local_unicorn": prefer_local_unicorn,
        "local_unicorn_bindings": str(LOCAL_UNICORN_BINDINGS),
        "local_unicorn_available": LOCAL_UNICORN_BINDINGS.exists(),
        "local_unicorn_injected": unicorn_added or str(LOCAL_UNICORN_BINDINGS) in sys.path,
        "local_unicorn_build": str(LOCAL_UNICORN_BUILD),
        "local_unicorn_shared_lib": str(LOCAL_UNICORN_SHARED_LIB),
        "local_unicorn_shared_lib_available": LOCAL_UNICORN_SHARED_LIB.exists(),
        "configured_libunicorn_path": configured_libunicorn_path,
        "local_unicorn_lib_configured": (
            bool(configured_libunicorn_path)
            and Path(configured_libunicorn_path).resolve() == LOCAL_UNICORN_BUILD.resolve()
        ),
        "active_venv": active_venv,
        **{
            f"unicorn_{key}": value
            for key, value in loaded_status.items()
        },
    }
