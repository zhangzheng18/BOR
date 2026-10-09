#!/usr/bin/env python3
"""Normalize firmware inputs before they enter the LSGEmu analysis pipeline.

Native ELF inputs and explicitly specified BIN conversions retain their
historical behavior.  A raw BIN supplied without an explicit
:class:`BinConversionSpec` is first screened by :mod:`lsgemu.vector_table_locator`;
when a confident Cortex-M vector table is found, the scan-derived load base
and reset entry drive BintoElf wrapping automatically (zero-parameter BIN
usage).  Images without a confident vector table keep the legacy raw-BIN
loader, preserving the static-analysis denominator of existing raw campaigns.
The returned provenance keeps the raw and generated identities separate so
reports do not silently attribute generated-ELF properties to the original
image.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Mapping, Optional, Tuple

from .artifact_io import atomic_write_bytes
from .toolchain_fingerprint import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINTOELF_ROOT = Path(
    os.environ.get("BINTOELF_ROOT", "/opt/artifact/BintoElf")
)
DEFAULT_BINTOELF_CACHE = Path(
    os.environ.get(
        "LSGEMU_BINTOELF_CACHE_DIR",
        str(PROJECT_ROOT / ".lsgemu_cache" / "bin2elf"),
    )
)
INPUT_PROVENANCE_SCHEMA = "lsgemu.firmware_input.v1"
CONVERSION_IDENTITY_SCHEMA = "lsgemu.bintoelf_conversion.v1"


class FirmwareInputError(ValueError):
    """Raised when a firmware input cannot be normalized safely."""


@dataclass(frozen=True)
class BinConversionSpec:
    """Explicit raw-BIN layout required to construct an analysis ELF."""

    arch: str
    bits: int
    endian: str
    base_address: int
    entry: Optional[int] = None
    entry_offset: Optional[int] = None
    thumb: Optional[bool] = None
    physical_address: Optional[int] = None
    permissions: str = "r-x"
    segment_align: int = 0x1000
    bss_size: int = 0

    def resolved_entry(self) -> int:
        if self.entry is not None and self.entry_offset is not None:
            raise FirmwareInputError(
                "BIN conversion accepts either an absolute entry or an entry offset, not both"
            )
        base = int(self.base_address)
        entry = (
            int(self.entry)
            if self.entry is not None
            else base + int(self.entry_offset)
            if self.entry_offset is not None
            else base
        )
        if self.thumb is True:
            entry |= 1
        elif self.thumb is False and entry & 1:
            raise FirmwareInputError(
                "an odd ARM32 entry encodes Thumb state and conflicts with --no-bin-thumb"
            )
        return entry


@dataclass(frozen=True)
class ResolvedFirmwareInput:
    """The path consumed by LSGEmu plus auditable source provenance."""

    source_path: Path
    analysis_path: Path
    provenance: Mapping[str, object]
    execution_thumb_override: Optional[bool] = None

    @property
    def converted(self) -> bool:
        return self.source_path != self.analysis_path


@dataclass(frozen=True)
class _BintoElfAPI:
    module: ModuleType
    root: Path
    version: str
    source_identity: Mapping[str, object]


_BINTOELF_MODULE_CACHE: Dict[Tuple[str, str], ModuleType] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_paths(root: str | Path) -> Tuple[Path, Path, Path]:
    requested = Path(root).expanduser().resolve()
    package_dir = requested / "bin2elf"
    project_root = requested
    if not (package_dir / "__init__.py").is_file():
        if requested.name == "bin2elf" and (requested / "__init__.py").is_file():
            package_dir = requested
            project_root = requested.parent
        else:
            raise FirmwareInputError(
                f"BintoElf package not found under {requested}; expected bin2elf/__init__.py"
            )
    init_path = package_dir / "__init__.py"
    core_path = package_dir / "core.py"
    if not core_path.is_file():
        raise FirmwareInputError(f"BintoElf core module is missing: {core_path}")
    return project_root, init_path, core_path


def _load_bintoelf(root: str | Path) -> _BintoElfAPI:
    project_root, init_path, core_path = _package_paths(root)
    init_sha256 = sha256_file(init_path)
    core_sha256 = sha256_file(core_path)
    source_identity: Dict[str, object] = {
        "root": str(project_root),
        "init_sha256": init_sha256,
        "core_sha256": core_sha256,
    }
    source_identity["source_sha256"] = _json_identity({
        "init_sha256": init_sha256,
        "core_sha256": core_sha256,
    })
    cache_key = (str(project_root), str(source_identity["source_sha256"]))
    module = _BINTOELF_MODULE_CACHE.get(cache_key)
    if module is None:
        module_identity = hashlib.sha256(
            (str(project_root) + str(source_identity["source_sha256"])).encode("utf-8")
        ).hexdigest()
        module_name = f"_lsgemu_bintoelf_{module_identity[:16]}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(init_path.parent)],
        )
        if spec is None or spec.loader is None:
            raise FirmwareInputError(f"cannot load BintoElf package from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            for loaded_name in list(sys.modules):
                if loaded_name == module_name or loaded_name.startswith(module_name + "."):
                    sys.modules.pop(loaded_name, None)
            raise FirmwareInputError(f"failed to import BintoElf from {project_root}: {exc}") from exc
        _BINTOELF_MODULE_CACHE[cache_key] = module

    required = ("ElfConfig", "build_elf", "extract_binary", "inspect_elf")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise FirmwareInputError(
            f"BintoElf API is incomplete under {project_root}: missing {', '.join(missing)}"
        )
    version = str(getattr(module, "__version__", "unknown"))
    source_identity["version"] = version
    return _BintoElfAPI(module, project_root, version, source_identity)


def _file_record(path: Path, *, digest: Optional[str] = None) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": digest or sha256_file(path),
    }


def _native_input(path: Path, *, source_kind: str) -> ResolvedFirmwareInput:
    record = _file_record(path)
    provenance = {
        "schema": INPUT_PROVENANCE_SCHEMA,
        "source_kind": source_kind,
        "adapter": "none",
        "transformed": False,
        "source": dict(record),
        "analysis": dict(record),
    }
    if source_kind == "legacy_raw_bin":
        provenance["layout_authority"] = "legacy_lsgemu_heuristics_or_cli_overrides"
    return ResolvedFirmwareInput(path, path, provenance)


def _detect_raw_bin_vector_layout(payload: bytes) -> Optional[Dict[str, object]]:
    """对 raw BIN 做向量表全扫描，置信命中时返回布局建议。

    返回 None 表示未找到置信向量表（调用方保持 legacy raw 行为）。扫描
    会尝试常见 Cortex-M flash 基址并用 Reset 指针扇区对齐值补充候选，
    例如 ardupilot 应用镜像自动得到 load_base=0x08004000、
    reset=Reset 向量本身。
    """
    from .vector_table_locator import scan_vector_table

    candidates = scan_vector_table(payload)
    if not candidates:
        return None
    top = candidates[0]
    if not top.confident:
        return None
    return {
        "method": "full_scan",
        "score": float(top.score),
        "signals": list(top.signals),
        "offset": int(top.offset),
        "load_base": int(top.load_base),
        "initial_sp": int(top.initial_sp),
        "reset_pc": int(top.reset_pc),
    }


def _safe_cache_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "firmware"


def _normalized_conversion(
    api: _BintoElfAPI,
    payload_size: int,
    spec: BinConversionSpec,
) -> Tuple[object, Dict[str, object], bool]:
    entry = spec.resolved_entry()
    try:
        config = api.module.ElfConfig(
            arch=spec.arch,
            bits=spec.bits,
            endian=spec.endian,
            base_address=spec.base_address,
            physical_address=spec.physical_address,
            entry=entry,
            permissions=spec.permissions,
            segment_align=spec.segment_align,
            bss_size=spec.bss_size,
        )
        normalized = config.normalized(payload_size)
    except Exception as exc:
        raise FirmwareInputError(f"invalid BintoElf conversion specification: {exc}") from exc

    architecture = normalized.architecture
    if architecture.family != "arm" or int(architecture.bits) != 32:
        raise FirmwareInputError(
            "BintoElf can represent this target, but LSGEmu execution currently supports "
            f"only ARM32; requested {architecture.name}/{architecture.bits}"
        )
    if "x" not in str(normalized.permissions):
        raise FirmwareInputError(
            "LSGEmu requires the wrapped firmware segment to be executable; "
            f"permissions={normalized.permissions!r}"
        )

    execution_thumb = bool(int(normalized.entry) & 1)
    canonical = {
        "architecture": str(architecture.name),
        "family": str(architecture.family),
        "bits": int(architecture.bits),
        "endian": str(normalized.endian),
        "base_address": int(normalized.base_address),
        "physical_address": int(normalized.physical_address),
        "entry": int(normalized.entry),
        "execution_mode": "thumb" if execution_thumb else "arm",
        "permissions": str(normalized.permissions),
        "segment_align": int(normalized.segment_align),
        "bss_size": int(normalized.bss_size),
        "e_type": int(normalized.e_type),
        "e_flags": int(normalized.e_flags),
    }
    return config, canonical, execution_thumb


def _validate_generated_elf(
    api: _BintoElfAPI,
    elf_bytes: bytes,
    payload: bytes,
    *,
    canonical: Mapping[str, object],
    conversion_identity: str,
) -> Mapping[str, object]:
    try:
        inspected = api.module.inspect_elf(elf_bytes, strict=True)
        restored = api.module.extract_binary(elf_bytes)
    except Exception as exc:
        raise FirmwareInputError(f"BintoElf generated an ELF that failed validation: {exc}") from exc
    if restored != payload:
        raise FirmwareInputError("BintoElf round-trip validation did not reproduce the source BIN")

    metadata = dict(inspected.get("metadata") or {})
    checks = {
        "family": canonical["family"],
        "bits": canonical["bits"],
        "endian": canonical["endian"],
        "entry": canonical["entry"],
    }
    for key, expected in checks.items():
        if inspected.get(key) != expected:
            raise FirmwareInputError(
                f"BintoElf validation mismatch for {key}: {inspected.get(key)!r} != {expected!r}"
            )
    program_headers = [
        item for item in inspected.get("program_headers", []) if int(item.get("type", 0)) == 1
    ]
    if len(program_headers) != 1:
        raise FirmwareInputError(
            f"BintoElf integration requires exactly one PT_LOAD segment, got {len(program_headers)}"
        )
    load = program_headers[0]
    if int(load.get("vaddr", -1)) != int(canonical["base_address"]):
        raise FirmwareInputError("BintoElf PT_LOAD virtual address does not match --bin-base")
    if int(load.get("paddr", -1)) != int(canonical["physical_address"]):
        raise FirmwareInputError("BintoElf PT_LOAD physical address does not match --bin-paddr")
    expected_metadata = {
        "payload_sha256": _sha256_bytes(payload),
        "payload_size": len(payload),
        "lsgemu_conversion_identity": conversion_identity,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise FirmwareInputError(
                f"BintoElf metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    return inspected


def resolve_firmware_input(
    firmware_path: str | Path,
    *,
    input_format: str = "auto",
    bin_spec: Optional[BinConversionSpec] = None,
    bintoelf_root: str | Path = DEFAULT_BINTOELF_ROOT,
    cache_root: str | Path = DEFAULT_BINTOELF_CACHE,
) -> ResolvedFirmwareInput:
    """Resolve an ELF/raw input without changing the exploration pipeline.

    ``input_format='auto'`` keeps native ELF handling and legacy raw-BIN
    handling for images without a confident Cortex-M vector table.  A raw
    BIN whose vector table is located confidently by
    :func:`lsgemu.vector_table_locator.scan_vector_table` is wrapped through
    BintoElf using the scan-derived base/entry even without an explicit
    ``bin_spec`` (zero-parameter BIN usage).  Explicit ``bin_spec`` values
    always win and skip the scan.  ``input_format='raw'`` forces the legacy
    loader unconditionally.
    """

    source = Path(firmware_path).expanduser().resolve()
    if not source.is_file():
        raise FirmwareInputError(f"firmware input does not exist or is not a file: {source}")
    mode = str(input_format or "auto").strip().lower()
    if mode not in {"auto", "elf", "raw", "bin"}:
        raise FirmwareInputError(f"unsupported firmware input format: {input_format!r}")
    with source.open("rb") as handle:
        magic = handle.read(4)
    is_elf = magic == b"\x7fELF"

    if mode == "elf":
        if not is_elf:
            raise FirmwareInputError("--firmware-format elf requires an ELF input")
        if bin_spec is not None:
            raise FirmwareInputError("--bin-* conversion options cannot be applied to an ELF input")
        return _native_input(source, source_kind="elf")
    if mode == "raw":
        if is_elf:
            raise FirmwareInputError("--firmware-format raw cannot be applied to an ELF input")
        if bin_spec is not None:
            raise FirmwareInputError(
                "--firmware-format raw preserves the legacy loader and cannot be combined with --bin-*"
            )
        return _native_input(source, source_kind="legacy_raw_bin")
    if is_elf:
        if mode == "bin":
            raise FirmwareInputError("--firmware-format bin refuses to wrap an existing ELF as raw bytes")
        if bin_spec is not None:
            raise FirmwareInputError("--bin-* conversion options cannot be applied to an ELF input")
        return _native_input(source, source_kind="elf")
    vector_layout: Optional[Dict[str, object]] = None
    payload: Optional[bytes] = None
    if bin_spec is None:
        # 无显式布局：先尝试向量表自动定位（BintoElf 之前），置信命中才
        # 自动构造转换参数，避免"基址猜错→向量表全错位"。用户显式给出
        # --bin-base/--bin-entry 时 bin_spec 非 None，不会进入本分支。
        if mode in {"auto", "bin"}:
            try:
                payload = source.read_bytes()
                vector_layout = _detect_raw_bin_vector_layout(payload)
            except Exception as exc:
                raise FirmwareInputError(
                    f"automatic vector-table localization failed for {source}: {exc}"
                ) from exc
        if vector_layout is None:
            if mode == "bin":
                raise FirmwareInputError(
                    "BIN conversion requires explicit --bin-arch, --bin-bits, "
                    "--bin-endian and --bin-base (automatic Cortex-M vector-table "
                    "localization found no confident candidate)"
                )
            return _native_input(source, source_kind="legacy_raw_bin")
        bin_spec = BinConversionSpec(
            arch="arm",
            bits=32,
            endian="little",
            base_address=int(vector_layout["load_base"]),
            entry=int(vector_layout["reset_pc"]),
            thumb=True,
        )

    if payload is None:
        payload = source.read_bytes()
    source_sha256 = _sha256_bytes(payload)
    api = _load_bintoelf(bintoelf_root)
    base_config, canonical, execution_thumb = _normalized_conversion(
        api,
        len(payload),
        bin_spec,
    )
    conversion_payload: Dict[str, object] = {
        "schema": CONVERSION_IDENTITY_SCHEMA,
        "source_sha256": source_sha256,
        "source_size": len(payload),
        "bintoelf_version": api.version,
        "bintoelf_source_sha256": api.source_identity["source_sha256"],
        "configuration": canonical,
    }
    if vector_layout is not None:
        # 布局决策参与 identity：同一 BIN 的定位结果确定，缓存键保持稳定，
        # 而不同布局建议会生成不同的 ELF。
        conversion_payload["vector_table_detection"] = dict(vector_layout)
    conversion_identity = _json_identity(conversion_payload)

    try:
        config = api.module.ElfConfig(
            arch=base_config.arch,
            bits=base_config.bits,
            endian=base_config.endian,
            base_address=base_config.base_address,
            physical_address=base_config.physical_address,
            entry=base_config.entry,
            permissions=base_config.permissions,
            segment_align=base_config.segment_align,
            bss_size=base_config.bss_size,
            e_type=base_config.e_type,
            e_flags=base_config.e_flags,
            osabi=base_config.osabi,
            abiversion=base_config.abiversion,
            file_offset=base_config.file_offset,
            include_metadata=True,
            include_sections=True,
            allow_entry_outside=base_config.allow_entry_outside,
            metadata={
                "consumer": "LSGEmu",
                "lsgemu_conversion_schema": CONVERSION_IDENTITY_SCHEMA,
                "lsgemu_conversion_identity": conversion_identity,
            },
            max_file_size=base_config.max_file_size,
        )
        elf_bytes = api.module.build_elf(payload, config)
    except Exception as exc:
        raise FirmwareInputError(f"BintoElf conversion failed: {exc}") from exc

    inspected = _validate_generated_elf(
        api,
        elf_bytes,
        payload,
        canonical=canonical,
        conversion_identity=conversion_identity,
    )
    generated_sha256 = _sha256_bytes(elf_bytes)
    output = (
        Path(cache_root).expanduser().resolve()
        / conversion_identity
        / f"{_safe_cache_stem(source)}.elf"
    )
    cache_hit = False
    if output.is_file():
        try:
            cache_hit = output.stat().st_size == len(elf_bytes) and sha256_file(output) == generated_sha256
        except OSError:
            cache_hit = False
    if not cache_hit:
        atomic_write_bytes(output, elf_bytes)

    provenance = {
        "schema": INPUT_PROVENANCE_SCHEMA,
        "source_kind": "raw_bin",
        "adapter": "BintoElf",
        "transformed": True,
        "source": {
            "path": str(source),
            "size": len(payload),
            "sha256": source_sha256,
        },
        "analysis": {
            "path": str(output),
            "size": len(elf_bytes),
            "sha256": generated_sha256,
            "format": "ELF",
        },
        "conversion": {
            "schema": CONVERSION_IDENTITY_SCHEMA,
            "identity_hash": conversion_identity,
            "cache_hit": cache_hit,
            "cache_root": str(Path(cache_root).expanduser().resolve()),
            "configuration": canonical,
            "bintoelf": dict(api.source_identity),
            "validated_metadata": dict(inspected.get("metadata") or {}),
            "round_trip_validated": True,
            **(
                {"vector_table_detection": dict(vector_layout)}
                if vector_layout is not None
                else {}
            ),
        },
        "execution_mode": "thumb" if execution_thumb else "arm",
        "layout_authority": (
            "vector_table_scan" if vector_layout is not None else "explicit_bin_spec"
        ),
        "static_denominator_policy": "disable_external_valid_bb_auto_match",
        "static_analysis_caveat": (
            "A wrapped raw BIN has no original symbols, relocations, section boundaries, "
            "or trustworthy code/data partition. Its static-BB denominator is not "
            "automatically comparable with one derived from an original linker ELF."
        ),
    }
    return ResolvedFirmwareInput(
        source_path=source,
        analysis_path=output,
        provenance=provenance,
        execution_thumb_override=execution_thumb,
    )


__all__ = [
    "BinConversionSpec",
    "CONVERSION_IDENTITY_SCHEMA",
    "DEFAULT_BINTOELF_CACHE",
    "DEFAULT_BINTOELF_ROOT",
    "FirmwareInputError",
    "INPUT_PROVENANCE_SCHEMA",
    "ResolvedFirmwareInput",
    "resolve_firmware_input",
]
