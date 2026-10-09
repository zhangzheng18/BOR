#!/usr/bin/env python3
"""Deployment configuration loader for the packaged LSGEmu tree.

The original research tree relied on many machine-local environment variables
and a few hardcoded absolute paths.  This module centralizes those values in a
YAML/JSON deployment config and applies them before heavy modules import
Unicorn, Ghidra helpers, or campaign runners.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Optional[Path]:
    env_path = os.environ.get("LSGEMU_CONFIG_FILE")
    if env_path:
        return Path(env_path).expanduser()
    root = source_root()
    for candidate in (
        root / "configs" / "lsgemu_config.yaml",
        root / "configs" / "lsgemu_config.yml",
        root / "configs" / "lsgemu_config.example.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def config_arg_from_argv(argv: Optional[Iterable[str]] = None) -> Optional[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(args):
        if arg == "--config" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _seed_default_env(config_path: Optional[Path]) -> Dict[str, str]:
    root = source_root()
    config_dir = config_path.expanduser().resolve().parent if config_path else root / "configs"
    defaults = {
        "LSGEMU_SRC_FINAL_ROOT": str(root),
        "LSGEMU_SOURCE_ROOT": str(root),
        "LSGEMU_SRCV4_ROOT": str(root),
        "LSGEMU_CONFIG_DIR": str(config_dir),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return defaults


def _expand_env_text(value: str) -> str:
    text = value
    for _ in range(16):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1)
            fallback = match.group(2)
            if name in os.environ:
                changed = True
                return os.environ[name]
            if fallback is not None:
                changed = True
                return _expand_env_text(fallback)
            return ""

        new_text = ENV_PATTERN.sub(replace, text)
        if new_text == text and not changed:
            return new_text
        text = new_text
    return text


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_text(value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_deployment_config(path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser() if path else default_config_path()
    _seed_default_env(config_path)
    if not config_path or not config_path.exists():
        return {}
    resolved = config_path.resolve()
    with resolved.open() as f:
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError(f"PyYAML is required to read {resolved}: {exc}") from exc
            payload = yaml.safe_load(f) or {}
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"deployment config must be a mapping: {resolved}")
    payload = expand_env(payload)
    payload["_config_file"] = str(resolved)
    return payload


def _set_env(name: str, value: Any, *, override: bool = False) -> bool:
    if value is None:
        return False
    text = str(value)
    if text == "":
        return False
    if not override and os.environ.get(name):
        return False
    os.environ[name] = text
    return True


def _prepend_env_path(name: str, path: str | Path | None) -> bool:
    if not path:
        return False
    path_text = str(path)
    if not path_text:
        return False
    current = os.environ.get(name, "")
    entries = [item for item in current.split(os.pathsep) if item]
    if path_text in entries:
        return False
    os.environ[name] = path_text if not current else f"{path_text}{os.pathsep}{current}"
    return True


def _prepend_sys_path(path: str | Path | None) -> bool:
    if not path:
        return False
    path_text = str(path)
    if not path_text or path_text in sys.path:
        return False
    sys.path.insert(0, path_text)
    return True


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _apply_dataset_env(datasets: Mapping[str, Any], *, override: bool) -> Dict[str, str]:
    mapping = {
        "elfmultifuzz_root": "ELFMULTIFUZZ_ROOT",
        "p2im_real_tests_root": "P2IM_REAL_TESTS_ROOT",
        "manual_mcu_root": "MANUAL_MCU_ROOT",
        "manual_mcu_selected_root": "MANUAL_MCU_SELECTED_ROOT",
        "fuzzware_soat_root": "FUZZWARE_SOAT_ROOT",
        "multifuzz_soat_root": "MULTIFUZZ_SOAT_ROOT",
    }
    applied: Dict[str, str] = {}
    for config_key, env_key in mapping.items():
        if _set_env(env_key, datasets.get(config_key), override=override):
            applied[env_key] = os.environ[env_key]
    if os.environ.get("ELFMULTIFUZZ_ROOT"):
        _set_env("LSGEMU_VALID_BB_ROOT", os.environ["ELFMULTIFUZZ_ROOT"], override=override)
        applied.setdefault("LSGEMU_VALID_BB_ROOT", os.environ.get("LSGEMU_VALID_BB_ROOT", ""))
    return applied


def apply_deployment_config(config: Mapping[str, Any], *, override_env: bool = False) -> Dict[str, str]:
    applied: Dict[str, str] = {}
    config_file = config.get("_config_file")
    if config_file and _set_env("LSGEMU_CONFIG_FILE", config_file, override=override_env):
        applied["LSGEMU_CONFIG_FILE"] = os.environ["LSGEMU_CONFIG_FILE"]

    project = _mapping(config.get("project"))
    source = project.get("source_root") or project.get("src_final_root") or project.get("srcv4_root")
    upstream = project.get("root") or project.get("upstream_root")
    for env_key, value in (
        ("LSGEMU_SOURCE_ROOT", source),
        ("LSGEMU_SRC_FINAL_ROOT", source),
        ("LSGEMU_SRCV4_ROOT", project.get("srcv4_root") or source),
        ("LSGEMU_PROJECT_ROOT", upstream or source),
        ("LSGEMU_STATIC_CACHE_DIR", project.get("cache_dir")),
        ("LSGEMU_RUN_OUTPUT_DIR", project.get("run_output_dir")),
        ("TMPDIR", project.get("temp_dir")),
    ):
        if _set_env(env_key, value, override=override_env):
            applied[env_key] = os.environ[env_key]

    python_cfg = _mapping(config.get("python"))
    for item in python_cfg.get("pythonpath") or []:
        if item:
            _prepend_sys_path(item)
            _prepend_env_path("PYTHONPATH", item)
    if python_cfg.get("virtualenv"):
        _set_env("VIRTUAL_ENV", python_cfg.get("virtualenv"), override=override_env)

    ghidra = _mapping(config.get("ghidra"))
    for env_key, value in (
        ("GHIDRA_INSTALL_DIR", ghidra.get("install_dir")),
        ("GHIDRA_ANALYZE_HEADLESS", ghidra.get("analyze_headless")),
        ("JAVA_HOME", ghidra.get("java_home")),
        ("XDG_CONFIG_HOME", ghidra.get("xdg_config_home")),
    ):
        if _set_env(env_key, value, override=override_env):
            applied[env_key] = os.environ[env_key]

    unicorn = _mapping(config.get("unicorn"))
    if _set_env("LSGEMU_USE_SYSTEM_UNICORN", unicorn.get("use_system_unicorn"), override=override_env):
        applied["LSGEMU_USE_SYSTEM_UNICORN"] = os.environ["LSGEMU_USE_SYSTEM_UNICORN"]
    if _set_env("LSGEMU_UNICORN_PYTHON_BINDINGS", unicorn.get("local_python_bindings"), override=override_env):
        applied["LSGEMU_UNICORN_PYTHON_BINDINGS"] = os.environ["LSGEMU_UNICORN_PYTHON_BINDINGS"]
    if _set_env("LIBUNICORN_PATH", unicorn.get("local_build_dir"), override=override_env):
        applied["LIBUNICORN_PATH"] = os.environ["LIBUNICORN_PATH"]
    if _set_env("LSGEMU_UNICORN_SHARED_LIB", unicorn.get("shared_library"), override=override_env):
        applied["LSGEMU_UNICORN_SHARED_LIB"] = os.environ["LSGEMU_UNICORN_SHARED_LIB"]
    _prepend_env_path("LD_LIBRARY_PATH", unicorn.get("ld_library_path_entry"))

    llm = _mapping(config.get("llm"))
    for env_key, value in (
        ("LSGEMU_LLM_CONFIG", llm.get("config_file")),
        ("LSGEMU_LLM_MODEL", llm.get("model")),
        ("OPENAI_BASE_URL", llm.get("base_url")),
    ):
        if _set_env(env_key, value, override=override_env):
            applied[env_key] = os.environ[env_key]

    svd = _mapping(config.get("svd"))
    paths = [str(item) for item in (svd.get("paths") or []) if item]
    if paths and _set_env("LSGEMU_SVD_PATHS", os.pathsep.join(paths), override=override_env):
        applied["LSGEMU_SVD_PATHS"] = os.environ["LSGEMU_SVD_PATHS"]

    applied.update(_apply_dataset_env(_mapping(config.get("datasets")), override=override_env))

    run_profile = _mapping(config.get("run_profile"))
    environment = _mapping(run_profile.get("environment"))
    for key, value in sorted(environment.items()):
        key_text = str(key)
        if not key_text.startswith("LSGEMU_"):
            continue
        if _set_env(key_text, value, override=override_env):
            applied[key_text] = os.environ[key_text]

    return applied


def load_and_apply_deployment_config(
    path: str | Path | None = None,
    *,
    override_env: bool = False,
) -> Dict[str, Any]:
    config = load_deployment_config(path)
    applied = apply_deployment_config(config, override_env=override_env) if config else {}
    config["_applied_environment"] = applied
    return config


def apply_config_from_argv(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    config_path = config_arg_from_argv(argv)
    return load_and_apply_deployment_config(config_path)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def configured_path(env_key: str, fallback: str | Path) -> Path:
    """Resolve an env-configured path to an absolute location.

    Campaign child processes run with a cwd different from the repo root
    (e.g. srcv4/), so a relative env value that does not exist under the
    current cwd is retried against the repo root it was exported from —
    otherwise LLM configs and run roots silently fail to load in children.
    Relative fallbacks keep their historical cwd anchoring.
    """
    raw = os.environ.get(env_key)
    if raw is None:
        return Path(str(fallback)).expanduser().resolve()
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_anchored = expanded.resolve()
    if cwd_anchored.exists():
        return cwd_anchored
    repo_anchored = (_REPO_ROOT / expanded).resolve()
    return repo_anchored if repo_anchored.exists() else cwd_anchored
