#!/usr/bin/env python3
"""Crash-consistent persistence primitives for runtime and evaluation artifacts."""

from __future__ import annotations

import errno
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - LSGEmu targets Linux, keep import portable.
    fcntl = None


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(
    path: str | Path,
    data: bytes | bytearray | memoryview,
    *,
    mode: Optional[int] = None,
    durable: bool = True,
) -> Path:
    """Replace *path* atomically with bytes written in the same directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes(data)
    existing_mode: Optional[int] = None
    try:
        existing_mode = destination.stat().st_mode & 0o777
    except OSError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        effective_mode = mode if mode is not None else (existing_mode or 0o644)
        os.fchmod(fd, int(effective_mode) & 0o777)
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while persisting artifact")
            offset += written
        if durable:
            os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, destination)
        if durable:
            _fsync_directory(destination.parent)
        return destination
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: Optional[int] = None,
    durable: bool = True,
) -> Path:
    return atomic_write_bytes(
        path,
        str(text).encode(encoding),
        mode=mode,
        durable=durable,
    )


def atomic_copy_file(
    source: str | Path,
    destination: str | Path,
    *,
    durable: bool = True,
) -> Path:
    """Publish a file copy atomically without loading the artifact into memory."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_mode = source_path.stat().st_mode & 0o777
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=str(destination_path.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, source_mode or 0o644)
        with source_path.open("rb") as source_handle:
            while True:
                chunk = source_handle.read(1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    written = os.write(fd, chunk[offset:])
                    if written <= 0:
                        raise OSError("short write while copying artifact")
                    offset += written
        if durable:
            os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, destination_path)
        if durable:
            _fsync_directory(destination_path.parent)
        return destination_path
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_publish_file(
    source: str | Path,
    destination: str | Path,
    *,
    durable: bool = True,
) -> Path:
    """Publish an immutable artifact without duplicating it when possible.

    Evaluation attempts and their stable aliases live on the same filesystem
    in the normal layout.  A hard link keeps both names crash-consistent while
    avoiding a second full read/write of reports and logs.  Filesystems that do
    not support hard links transparently fall back to ``atomic_copy_file``.
    Callers must treat the source as immutable after publication.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_path.resolve() == destination_path.resolve():
            return destination_path
    except OSError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".link.tmp",
        dir=str(destination_path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        if durable:
            source_fd = os.open(str(source_path), os.O_RDONLY)
            try:
                os.fsync(source_fd)
            finally:
                os.close(source_fd)
        os.link(source_path, temporary)
        os.replace(temporary, destination_path)
        if durable:
            _fsync_directory(destination_path.parent)
        return destination_path
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        if exc.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
        }:
            raise
        return atomic_copy_file(
            source_path,
            destination_path,
            durable=durable,
        )


def atomic_json_dump(
    payload: Any,
    path: str | Path,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    durable: bool = True,
) -> Path:
    """Atomically stream JSON to disk without constructing a second full copy.

    Large evaluation reports can contain tens of megabytes of phase metadata.
    ``json.dumps`` followed by ``encode`` temporarily retained both a Unicode
    string and a byte buffer.  Streaming preserves the on-disk JSON contract
    while bounding finalization memory to the encoder's chunk size.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: Optional[int] = None
    try:
        existing_mode = destination.stat().st_mode & 0o777
    except OSError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, existing_mode or 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(
                payload,
                handle,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
                separators=(",", ":") if indent is None else None,
            )
            handle.write("\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if durable:
            _fsync_directory(destination.parent)
        return destination
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_pickle_dump(
    payload: Any,
    path: str | Path,
    *,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    durable: bool = True,
) -> Path:
    return atomic_write_bytes(
        path,
        pickle.dumps(payload, protocol=protocol),
        durable=durable,
    )


def append_jsonl(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    durable: bool = False,
) -> Path:
    """Append one complete JSON record with thread/process serialization."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = (
        json.dumps(payload, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    ).encode("utf-8")
    fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        offset = 0
        while offset < len(record):
            written = os.write(fd, record[offset:])
            if written <= 0:
                raise OSError("short write while appending JSONL artifact")
            offset += written
        if durable:
            os.fsync(fd)
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
    return destination
