#!/usr/bin/env python3
"""Immutable, byte-compatible snapshot memory with page interning."""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
import tempfile
import threading
import time
import weakref
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional


class SnapshotCaptureError(RuntimeError):
    """A required CPU or memory component could not be captured faithfully."""

    def __init__(self, component: str, detail: str, *, address: int | None = None):
        self.component = str(component)
        self.detail = str(detail)
        self.address = address
        location = f" @ 0x{int(address) & 0xFFFFFFFF:08x}" if address is not None else ""
        super().__init__(f"snapshot capture failed for {self.component}{location}: {self.detail}")


class SnapshotIntegrityError(RuntimeError):
    """A retained snapshot no longer satisfies its capture contract."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def snapshot_storage_directory(
    explicit: str | Path | None = None,
) -> Optional[Path]:
    """Return the configured per-run directory for cold snapshot payloads.

    Disk backing is deliberately opt-in at this low-level API.  Entry points
    call ``configure_snapshot_storage_for_run`` after choosing their output
    directory, while unit tests and library users retain the historical
    in-memory default unless they explicitly configure a directory.
    """

    if explicit is None:
        raw = str(os.environ.get("LSGEMU_SNAPSHOT_STORAGE_DIR") or "").strip()
        if not raw:
            return None
        if not _env_flag("LSGEMU_SNAPSHOT_DISK_BACKING", True):
            return None
    else:
        raw = str(explicit).strip()
        if not raw:
            return None
    return Path(raw).expanduser().resolve()


def configure_snapshot_storage_for_run(
    output_dir: str | Path,
    *,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """Configure an isolated payload directory for one emulator run.

    The caller may disable this feature explicitly with
    ``LSGEMU_SNAPSHOT_DISK_BACKING=0``.  Otherwise a unique child directory
    is selected so repeated runs never append to one another's payload file.
    The function only sets environment configuration; actual files are opened
    lazily when the first snapshot is captured.
    """

    if "LSGEMU_SNAPSHOT_DISK_BACKING" in os.environ and not _env_flag(
        "LSGEMU_SNAPSHOT_DISK_BACKING"
    ):
        return None
    output = Path(output_dir).expanduser().resolve()
    token = str(run_id or os.environ.get("LSGEMU_ATTEMPT_ID") or "").strip()
    # Environment variables cannot contain NUL bytes; the resolved output
    # path and token are already unambiguous with this separator.
    run_key = f"{output}|{token}"
    current = snapshot_storage_directory()
    current_is_auto = _env_flag("LSGEMU_SNAPSHOT_STORAGE_AUTO", False)
    current_run_key = os.environ.get("LSGEMU_SNAPSHOT_STORAGE_RUN_KEY", "")

    if current is not None and not current_is_auto:
        # A caller-supplied storage directory is an explicit shared-storage
        # contract. Respect it; changing it silently would be surprising for
        # library users who intentionally manage the backing directory.
        configured = current
    elif (
        current is not None
        and current_is_auto
        and current_run_key == run_key
    ):
        # Multiple entrypoint layers can configure the same run. Reuse the
        # already selected child rather than creating two independent stores.
        configured = current
    else:
        if not token:
            token = f"pid-{os.getpid()}-{time.time_ns()}"
            run_key = f"{output}|{token}"
        configured = output / ".snapshot_store" / token
        os.environ["LSGEMU_SNAPSHOT_STORAGE_DIR"] = str(configured)
        os.environ["LSGEMU_SNAPSHOT_STORAGE_AUTO"] = "1"
        os.environ["LSGEMU_SNAPSHOT_STORAGE_RUN_KEY"] = run_key
    os.environ.setdefault("LSGEMU_SNAPSHOT_DISK_BACKING", "1")
    try:
        configured.mkdir(parents=True, exist_ok=True)
    except OSError:
        # The concrete store will transparently fall back to memory if the
        # directory cannot be created.  Do not fail firmware execution here.
        return None
    return configured


class _AppendOnlyFileStore:
    """Small process-local append/read store for immutable snapshot payloads."""

    def __init__(self, directory: str | Path, prefix: str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(
            prefix=f"{prefix}-{os.getpid()}-",
            suffix=".bin",
            dir=str(self.directory),
        )
        self.path = Path(path)
        self._fd = int(fd)
        self._offset = 0
        self._lock = threading.RLock()
        self._closed = False
        self.bytes_written = 0
        self.bytes_read = 0
        self.append_operations = 0
        self.read_operations = 0
        self.write_syscalls = 0
        self.read_syscalls = 0

    def append(self, payload: bytes | bytearray | memoryview) -> tuple[int, int]:
        data = memoryview(payload)
        size = len(data)
        with self._lock:
            if self._closed or self._fd < 0:
                raise OSError("snapshot payload store is closed")
            offset = int(self._offset)
            os.lseek(self._fd, offset, os.SEEK_SET)
            written = 0
            while written < size:
                count = os.write(self._fd, data[written:])
                self.write_syscalls += 1
                if count <= 0:
                    raise OSError("short write to snapshot payload store")
                written += count
            self._offset += size
            self.bytes_written += size
            self.append_operations += 1
            return offset, size

    def read(self, offset: int, size: int) -> bytes:
        offset = int(offset)
        size = int(size)
        if offset < 0 or size < 0:
            raise ValueError("invalid snapshot payload range")
        with self._lock:
            if self._closed or self._fd < 0:
                raise OSError("snapshot payload store is closed")
            if size == 0:
                self.read_operations += 1
                return b""

            current = offset
            remaining = size
            first = os.pread(self._fd, remaining, current)
            self.read_syscalls += 1
            if not first:
                raise SnapshotIntegrityError(
                    "snapshot payload store ended before the requested range"
                )
            if len(first) == size:
                result = first
                self.bytes_read += size
                self.read_operations += 1
                return result

            # Regular-file pread normally satisfies the complete request. Keep
            # a short-read fallback for interrupted or unusual filesystems, but
            # avoid the bytearray -> bytes copy on the normal restore path.
            chunks = [first]
            current += len(first)
            remaining -= len(first)
            while remaining:
                chunk = os.pread(self._fd, remaining, current)
                self.read_syscalls += 1
                if not chunk:
                    raise SnapshotIntegrityError(
                        "snapshot payload store ended before the requested range"
                    )
                chunks.append(chunk)
                current += len(chunk)
                remaining -= len(chunk)
            result = b"".join(chunks)
            self.bytes_read += len(result)
            self.read_operations += 1
            return result

    def flush(self) -> None:
        with self._lock:
            if not self._closed and self._fd >= 0:
                os.fsync(self._fd)

    def close(self, *, durable: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            if durable and self._fd >= 0:
                try:
                    os.fsync(self._fd)
                except OSError:
                    pass
            if self._fd >= 0:
                os.close(self._fd)
            self._fd = -1
            self._closed = True

    def statistics(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "directory": str(self.directory),
            "bytes_written": int(self.bytes_written),
            "bytes_read": int(self.bytes_read),
            "append_operations": int(self.append_operations),
            "read_operations": int(self.read_operations),
            "write_syscalls": int(self.write_syscalls),
            "read_syscalls": int(self.read_syscalls),
            "file_size": int(self._offset),
            "closed": bool(self._closed),
        }

    def __del__(self):  # pragma: no cover - interpreter shutdown ordering varies.
        try:
            self.close()
        except Exception:
            pass


class SnapshotBlobStore(_AppendOnlyFileStore):
    """Shared append-only file for serialized external replay state."""

    # Blobs and emulator owners keep the store alive for exactly as long as it
    # can still be read. The registry must not extend that lifetime across
    # repeated in-process runs or retain an otherwise unused file descriptor.
    _registry: weakref.WeakValueDictionary[str, "SnapshotBlobStore"] = (
        weakref.WeakValueDictionary()
    )
    _registry_lock = threading.RLock()

    def __init__(self, directory: str | Path) -> None:
        super().__init__(directory, "snapshot-state")
        # Identical serialized states are common across replay engines. Keep a
        # compact content-address index so repeated captures reuse one immutable
        # file range instead of appending another copy. The index is process
        # local and disappears with the store; snapshot semantics do not depend
        # on it.
        self._payload_index: dict[tuple[int, str], tuple[int, int]] = {}
        self._deduplicated_reuses = 0
        # Statistics for the strict-collision validation path in append():
        # it increments _stats["index_read_failures"], which must exist even
        # though the base _AppendOnlyFileStore does not define a _stats dict.
        self._stats: Counter[str] = Counter()

    def append(self, payload: bytes | bytearray | memoryview) -> tuple[int, int]:
        data = memoryview(payload)
        digest = hashlib.sha256(data).hexdigest()
        key = (len(data), digest)
        with self._lock:
            if self._closed or self._fd < 0:
                raise OSError("snapshot payload store is closed")
            existing = self._payload_index.get(key)
            if existing is not None:
                strict = _env_flag("LSGEMU_SNAPSHOT_DISK_STRICT_COLLISION", False)
                if not strict:
                    self._deduplicated_reuses += 1
                    return existing
                try:
                    if self._read_without_stats(*existing) == data.tobytes():
                        self._deduplicated_reuses += 1
                        return existing
                except (OSError, SnapshotIntegrityError):
                    self._stats["index_read_failures"] += 1
            location = super().append(data)
            self._payload_index[key] = location
            return location

    def _read_without_stats(self, offset: int, size: int) -> bytes:
        """Read an indexed payload for optional collision validation."""
        if self._closed or self._fd < 0:
            raise OSError("snapshot payload store is closed")
        result = os.pread(self._fd, int(size), int(offset))
        if len(result) != int(size):
            raise SnapshotIntegrityError("indexed snapshot payload is truncated")
        return result

    @classmethod
    def from_environment(cls) -> Optional["SnapshotBlobStore"]:
        directory = snapshot_storage_directory()
        if directory is None:
            return None
        key = str(directory)
        with cls._registry_lock:
            store = cls._registry.get(key)
            if store is None or store._closed:
                store = cls(directory)
                cls._registry[key] = store
            return store

    def get_statistics(self) -> dict[str, object]:
        result = self.statistics()
        result["kind"] = "external_state"
        result["deduplicated_reuses"] = int(self._deduplicated_reuses)
        result["indexed_payloads"] = len(self._payload_index)
        return result


class SnapshotStateBlob(Mapping[str, object]):
    """Lazy, immutable storage for nested replay-model state.

    External model state contains diagnostic histories and nested peripheral
    objects.  Keeping a separately deep-copied Python object graph in every
    retained snapshot makes the graph the dominant resident-memory consumer.
    The serialized payload is an exact representation of the old mapping; it
    is decoded only when a replay actually restores the snapshot.  Small or
    unpickleable mappings transparently fall back to an ordinary dict.
    """

    __slots__ = (
        "_payload",
        "_compressed",
        "_raw_size",
        "_decoded",
        "_store",
        "_file_offset",
        "_file_size",
        "_payload_sha256",
    )

    def __init__(
        self,
        payload: Optional[bytes],
        *,
        compressed: bool,
        raw_size: int,
        store: Optional[SnapshotBlobStore] = None,
        file_offset: int = 0,
        file_size: int = 0,
    ) -> None:
        self._payload = bytes(payload) if payload is not None else None
        self._compressed = bool(compressed)
        self._raw_size = max(0, int(raw_size))
        self._decoded: Optional[dict[str, object]] = None
        self._store = store
        self._file_offset = max(0, int(file_offset))
        self._file_size = max(
            0,
            int(file_size or (len(self._payload) if self._payload is not None else 0)),
        )
        serialized = self._payload
        self._payload_sha256 = (
            hashlib.sha256(serialized).hexdigest() if serialized is not None else ""
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        *,
        min_raw_bytes: int = 4096,
        compression_level: int = 1,
        store: Optional[SnapshotBlobStore] = None,
    ) -> "SnapshotStateBlob | dict[str, object]":
        value = dict(mapping or {})
        try:
            raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Snapshot capture historically accepted arbitrary Python values.
            # Preserve that compatibility rather than making serialization a
            # new reason for a valid execution path to fail.
            return copy.deepcopy(value)
        if len(raw) < max(0, int(min_raw_bytes)):
            return copy.deepcopy(value)
        try:
            level = min(9, max(0, int(compression_level)))
            compressed = zlib.compress(raw, level) if level > 0 else raw
        except Exception:
            compressed = raw
        if len(compressed) < len(raw):
            serialized = compressed
            is_compressed = True
        else:
            serialized = raw
            is_compressed = False
        if store is None:
            store = SnapshotBlobStore.from_environment()
        if store is not None:
            try:
                offset, size = store.append(serialized)
                blob = cls(
                    None,
                    compressed=is_compressed,
                    raw_size=len(raw),
                    store=store,
                    file_offset=offset,
                    file_size=size,
                )
                blob._payload_sha256 = hashlib.sha256(serialized).hexdigest()
                return blob
            except (OSError, ValueError):
                # Disk backing is a storage optimization.  A transient I/O
                # failure must preserve the historical in-memory behavior.
                pass
        return cls(serialized, compressed=is_compressed, raw_size=len(raw))

    @classmethod
    def from_serialized(
        cls,
        payload: bytes,
        compressed: bool,
        raw_size: int,
    ) -> "SnapshotStateBlob":
        return cls(payload, compressed=compressed, raw_size=raw_size)

    @property
    def raw_size(self) -> int:
        return self._raw_size

    @property
    def stored_size(self) -> int:
        if self._store is not None:
            return int(self._file_size)
        return len(self._payload or b"")

    @property
    def is_disk_backed(self) -> bool:
        return self._store is not None

    @property
    def storage_path(self) -> str:
        return str(getattr(self._store, "path", "")) if self._store else ""

    @property
    def is_compressed(self) -> bool:
        return self._compressed

    @property
    def is_materialized(self) -> bool:
        """Whether the decoded object graph is currently retained.

        A blob normally keeps only its serialized payload.  The flag is
        exposed for lifecycle diagnostics and contract tests; callers should
        not depend on materialization as part of snapshot identity.
        """
        return self._decoded is not None

    def materialize(self) -> dict[str, object]:
        if self._decoded is not None:
            return self._decoded
        if self._payload is None and self._store is None:
            self._decoded = {}
            return self._decoded
        try:
            serialized = self._serialized_payload()
            raw = zlib.decompress(serialized) if self._compressed else serialized
            value = pickle.loads(raw)
        except Exception as exc:
            raise SnapshotIntegrityError(
                f"unable to decode retained replay state: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise SnapshotIntegrityError(
                "retained replay state did not decode to a mapping"
            )
        self._decoded = value
        return value

    def _serialized_payload(self) -> bytes:
        if self._store is not None:
            serialized = self._store.read(self._file_offset, self._file_size)
        else:
            serialized = self._payload or b""
        if self._payload_sha256 and hashlib.sha256(serialized).hexdigest() != self._payload_sha256:
            raise SnapshotIntegrityError("snapshot payload digest mismatch")
        return serialized

    def release_materialized(self) -> None:
        """Drop the transient decoded graph while retaining the payload.

        Replays may visit the same retained snapshot repeatedly.  Keeping the
        decoded dictionary after the first visit would turn lazy storage back
        into one full Python object graph per visited snapshot.  Snapshot
        ownership remains unchanged because the serialized payload is the
        authoritative representation.
        """
        self._decoded = None

    def __getitem__(self, key: str) -> object:
        return self.materialize()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.materialize())

    def __len__(self) -> int:
        return len(self.materialize())

    def __reduce__(self):
        if self._payload is None:
            return (
                self.from_serialized,
                (self._serialized_payload(), self._compressed, self._raw_size),
            )
        return (
            self.from_serialized,
            (self._payload, self._compressed, self._raw_size),
        )


class _InternedPage:
    __slots__ = (
        "_data",
        "_store",
        "_offset",
        "_length",
        "digest",
        "__weakref__",
    )

    def __init__(
        self,
        data: Optional[bytes],
        digest: str,
        *,
        store: Optional[_AppendOnlyFileStore] = None,
        offset: int = 0,
        length: int = 0,
    ):
        self._data = bytes(data) if data is not None else None
        self._store = store
        self._offset = max(0, int(offset))
        self._length = max(
            0,
            int(length or (len(self._data) if self._data is not None else 0)),
        )
        self.digest = str(digest)

    @property
    def data(self) -> bytes:
        if self._data is not None:
            return self._data
        if self._store is None:
            raise SnapshotIntegrityError("interned page has no backing storage")
        return self._store.read(self._offset, self._length)

    @property
    def size(self) -> int:
        return int(self._length)

    @property
    def is_disk_backed(self) -> bool:
        return self._store is not None


class PagedMemory:
    """Read-only memory image supporting the bytes operations used by LSGEmu."""

    __slots__ = ("_pages", "_size", "page_size", "sha256")

    def __init__(
        self,
        pages: tuple[_InternedPage, ...],
        size: int,
        page_size: int,
        sha256: str,
    ):
        self._pages = pages
        self._size = max(0, int(size))
        self.page_size = max(1, int(page_size))
        self.sha256 = str(sha256)

    @classmethod
    def from_bytes(cls, data: bytes, page_size: int = 4096) -> "PagedMemory":
        return SnapshotPageStore(page_size=page_size).intern_region(data)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __bytes__(self) -> bytes:
        if not self._pages:
            return b""
        # A zero or unchanged RAM page may occur hundreds of times in one
        # region while referring to one interned object. Materialize each
        # object once per restore so optional disk backing does not issue the
        # same pread repeatedly.
        materialized: dict[int, bytes] = {}
        chunks = []
        for page in self._pages:
            identity = id(page)
            page_data = materialized.get(identity)
            if page_data is None:
                page_data = page.data
                materialized[identity] = page_data
            chunks.append(page_data)
        return b"".join(chunks)[:self._size]

    def __getitem__(self, key):
        if isinstance(key, int):
            index = key + self._size if key < 0 else key
            if index < 0 or index >= self._size:
                raise IndexError("PagedMemory index out of range")
            page_index, page_offset = divmod(index, self.page_size)
            return self._pages[page_index].data[page_offset]
        if not isinstance(key, slice):
            raise TypeError(f"PagedMemory indices must be integers or slices, not {type(key).__name__}")
        start, stop, step = key.indices(self._size)
        if step != 1:
            return bytes(self)[key]
        if stop <= start:
            return b""
        first_page = start // self.page_size
        last_page = (stop - 1) // self.page_size
        chunks = []
        materialized: dict[int, bytes] = {}
        for page_index in range(first_page, last_page + 1):
            page_start = page_index * self.page_size
            local_start = max(0, start - page_start)
            page = self._pages[page_index]
            identity = id(page)
            page_data = materialized.get(identity)
            if page_data is None:
                page_data = page.data
                materialized[identity] = page_data
            local_stop = min(len(page_data), stop - page_start)
            chunks.append(page_data[local_start:local_stop])
        return b"".join(chunks)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PagedMemory):
            if self._size != other._size or self.sha256 != other.sha256:
                return False
            return all(
                left.data == right.data
                for left, right in zip(self._pages, other._pages)
            )
        if isinstance(other, (bytes, bytearray, memoryview)):
            return bytes(self) == bytes(other)
        return NotImplemented

    def __reduce__(self):
        return (self.from_bytes, (bytes(self), self.page_size))

    def iter_page_bytes(self) -> Iterator[bytes]:
        for page in self._pages:
            yield page.data

    def iter_page_objects(self) -> Iterator[_InternedPage]:
        """Iterate stable page identities without materializing page bytes."""
        return iter(self._pages)

    def __repr__(self) -> str:
        return f"PagedMemory(size={self._size}, pages={len(self._pages)}, sha256='{self.sha256[:12]}...')"


class SnapshotPageStore:
    """Weak page interning shared by replay emulators for one prepared firmware."""

    def __init__(
        self,
        page_size: int = 4096,
        storage_dir: str | Path | None = None,
    ):
        self.page_size = max(256, int(page_size))
        self._pages: weakref.WeakValueDictionary[str, _InternedPage] = weakref.WeakValueDictionary()
        # Weak page objects are allowed to disappear when no retained snapshot
        # references them. Keep only the compact immutable disk location so a
        # later capture of the same page can resurrect it without appending a
        # duplicate 4 KiB payload.
        self._disk_page_index: dict[str, tuple[int, int, str]] = {}
        self._lock = threading.RLock()
        self._stats: Counter[str] = Counter()
        self._file_store: Optional[_AppendOnlyFileStore] = None
        self.configure_disk_backing(storage_dir)

    def __getstate__(self) -> dict[str, int]:
        # Static-analysis caches may contain PreparedFirmware. Runtime pages,
        # locks, and cumulative counters are process-local and must not be
        # serialized into that cache.
        return {"page_size": int(self.page_size)}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(page_size=int((state or {}).get("page_size", 4096)))

    def configure_disk_backing(
        self,
        storage_dir: str | Path | None = None,
    ) -> Optional[Path]:
        """Enable page spilling without changing already captured pages."""
        if self._file_store is not None:
            return self._file_store.directory
        if storage_dir is None and not _env_flag(
            "LSGEMU_SNAPSHOT_PAGE_DISK_BACKING", True
        ):
            return None
        directory = snapshot_storage_directory(storage_dir)
        if directory is None:
            return None
        try:
            self._file_store = _AppendOnlyFileStore(directory, "snapshot-pages")
            self._stats["disk_backing_enabled"] = 1
            return directory
        except OSError:
            self._file_store = None
            self._stats["disk_backing_failures"] += 1
            return None

    def _page_for_key(
        self,
        key: str,
        data: bytes,
        digest: str,
        *,
        strict_collision_check: bool,
    ) -> tuple[Optional[_InternedPage], bool, str]:
        """Return an identical live/indexed page and whether *key* is occupied."""
        existing = self._pages.get(key)
        if existing is not None:
            if (
                self._file_store is not None
                and not strict_collision_check
                and existing.size == len(data)
                and existing.digest == digest
            ):
                return existing, True, "hash_identity"
            if existing.data == data:
                return existing, True, "byte_identity"
            return None, True, "collision"

        indexed = self._disk_page_index.get(key)
        if indexed is None or self._file_store is None:
            return None, False, "missing"
        offset, length, indexed_digest = indexed
        if int(length) != len(data) or str(indexed_digest) != digest:
            return None, True, "collision"
        if strict_collision_check:
            try:
                if self._file_store.read(offset, length) != data:
                    return None, True, "collision"
            except (OSError, SnapshotIntegrityError):
                self._stats["disk_index_read_failures"] += 1
                return None, True, "collision"
        page = _InternedPage(
            None,
            digest,
            store=self._file_store,
            offset=offset,
            length=length,
        )
        self._pages[key] = page
        return page, True, "disk_index"

    def _intern_page(self, data: bytes) -> _InternedPage:
        digest = hashlib.sha256(data).hexdigest()
        primary_key = f"{len(data)}:{digest}"
        strict_collision_check = _env_flag(
            "LSGEMU_SNAPSHOT_DISK_STRICT_COLLISION", False
        )
        with self._lock:
            key = primary_key
            collision_suffix = hashlib.sha512(data).hexdigest()
            collision_index = 0
            while True:
                existing, occupied, reuse_kind = self._page_for_key(
                    key,
                    data,
                    digest,
                    strict_collision_check=strict_collision_check,
                )
                if existing is not None:
                    self._stats["page_reuses"] += 1
                    if reuse_kind == "hash_identity":
                        self._stats["hash_identity_reuses"] += 1
                    elif reuse_kind == "disk_index":
                        self._stats["disk_index_reuses"] += 1
                    return existing
                if not occupied:
                    break
                self._stats["hash_collisions"] += 1
                key = f"{primary_key}:{collision_suffix}"
                if collision_index:
                    key = f"{key}:{collision_index}"
                collision_index += 1

            if self._file_store is not None:
                try:
                    offset, length = self._file_store.append(data)
                    page = _InternedPage(
                        None,
                        digest,
                        store=self._file_store,
                        offset=offset,
                        length=length,
                    )
                    self._disk_page_index[key] = (offset, length, digest)
                    self._stats["disk_pages_created"] += 1
                except (OSError, ValueError):
                    # Preserve the old behavior if the optional disk store is
                    # temporarily unavailable.
                    page = _InternedPage(bytes(data), digest)
                    self._stats["disk_fallback_pages"] += 1
            else:
                page = _InternedPage(bytes(data), digest)
            self._pages[key] = page
            self._stats["unique_pages_created"] += 1
            self._stats["unique_bytes_created"] += len(data)
            return page

    def intern_region(self, data: bytes | bytearray | memoryview) -> PagedMemory:
        raw = bytes(data)
        pages = tuple(
            self._intern_page(raw[offset:offset + self.page_size])
            for offset in range(0, len(raw), self.page_size)
        )
        with self._lock:
            self._stats["regions_created"] += 1
            self._stats["logical_bytes_captured"] += len(raw)
            self._stats["logical_pages_captured"] += len(pages)
        return PagedMemory(
            pages=pages,
            size=len(raw),
            page_size=self.page_size,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def get_statistics(self) -> dict[str, int]:
        with self._lock:
            live_pages = list(self._pages.values())
            stats = dict(self._stats)
        stats.update({
            "page_size": self.page_size,
            "live_unique_pages": len(live_pages),
            "live_unique_bytes": sum(page.size for page in live_pages),
            "indexed_unique_pages": len(self._disk_page_index),
            "indexed_unique_bytes": sum(
                int(location[1]) for location in self._disk_page_index.values()
            ),
            "disk_backed": bool(self._file_store is not None),
        })
        if self._file_store is not None:
            stats["disk_store"] = self._file_store.statistics()
        logical = int(stats.get("logical_bytes_captured", 0) or 0)
        unique = int(stats.get("unique_bytes_created", 0) or 0)
        stats["cumulative_bytes_avoided"] = max(0, logical - unique)
        return stats

    def close(self, *, durable: bool = False) -> None:
        """Close the page file after every retained PagedMemory is released."""
        if self._file_store is not None:
            self._file_store.close(durable=durable)


class InternedSnapshotMapping(dict):
    """A read-only dict used by retained snapshots."""

    is_snapshot_mapping = True

    def _immutable(self, *_args, **_kwargs):
        raise TypeError("snapshot mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __reduce__(self):
        # Serialized caches/checkpoints need the values, not the process-local
        # interning identity.
        return (dict, (dict(self),))


class InternedSnapshotSet(frozenset):
    """A read-only set whose identical instances can be weakly interned."""

    def __reduce__(self):
        # Do not serialize the metadata-store cache identity.
        return (frozenset, (tuple(self),))


class SnapshotMetadataStore:
    """Weakly intern flat snapshot mappings across replay emulators.

    Registers, MMIO values, and input occurrence counters contain immutable
    scalar values.  Thousands of branch snapshots often carry identical maps;
    sharing those maps preserves every replay state while removing duplicate
    Python hash tables.
    """

    def __init__(self):
        self._mappings: weakref.WeakValueDictionary[
            str, InternedSnapshotMapping
        ] = weakref.WeakValueDictionary()
        self._sets: weakref.WeakValueDictionary[
            str, InternedSnapshotSet
        ] = weakref.WeakValueDictionary()
        self._lock = threading.RLock()
        self._stats: Counter[str] = Counter()

    @staticmethod
    def _digest(mapping: Mapping[object, object]) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(mapping.items(), key=lambda item: repr(item[0])):
            digest.update(repr(key).encode("utf-8", errors="backslashreplace"))
            digest.update(b"\0")
            digest.update(repr(value).encode("utf-8", errors="backslashreplace"))
            digest.update(b"\0")
        return digest.hexdigest()

    def intern_mapping(
        self,
        mapping: Mapping[object, object] | None,
    ) -> InternedSnapshotMapping:
        raw = dict(mapping or {})
        digest = self._digest(raw)
        key = f"{len(raw)}:{digest}"
        with self._lock:
            existing = self._mappings.get(key)
            if existing is not None and existing == raw:
                self._stats["mapping_reuses"] += 1
                self._stats["entries_avoided"] += len(raw)
                return existing
            if existing is not None:
                self._stats["hash_collisions"] += 1
                key = f"{key}:{hashlib.sha512(repr(raw).encode()).hexdigest()}"
                collision_existing = self._mappings.get(key)
                if collision_existing is not None and collision_existing == raw:
                    self._stats["mapping_reuses"] += 1
                    self._stats["entries_avoided"] += len(raw)
                    return collision_existing
            interned = InternedSnapshotMapping(raw)
            self._mappings[key] = interned
            self._stats["unique_mappings_created"] += 1
            self._stats["unique_entries_created"] += len(raw)
            return interned

    @staticmethod
    def _set_digest(values: Iterable[object]) -> tuple[str, frozenset]:
        normalized = frozenset(values or ())
        digest = hashlib.sha256()
        for value in sorted(normalized, key=repr):
            digest.update(repr(value).encode("utf-8", errors="backslashreplace"))
            digest.update(b"\0")
        key = f"{len(normalized)}:{digest.hexdigest()}"
        return key, normalized

    def intern_set(self, values: Iterable[object] | None) -> InternedSnapshotSet:
        """Share an immutable set while retaining ordinary set semantics.

        Snapshot dirty-page sets are read-only after capture.  A weak cache is
        important here: it removes duplicate set objects without retaining a
        set after the last snapshot that references it is released.
        """
        key, normalized = self._set_digest(values or ())
        with self._lock:
            existing = self._sets.get(key)
            if existing is not None and existing == normalized:
                self._stats["set_reuses"] += 1
                self._stats["set_entries_avoided"] += len(normalized)
                return existing
            if existing is not None:
                self._stats["set_hash_collisions"] += 1
                key = f"{key}:{hashlib.sha512(repr(normalized).encode()).hexdigest()}"
                collision_existing = self._sets.get(key)
                if collision_existing is not None and collision_existing == normalized:
                    self._stats["set_reuses"] += 1
                    self._stats["set_entries_avoided"] += len(normalized)
                    return collision_existing
            interned = InternedSnapshotSet(normalized)
            self._sets[key] = interned
            self._stats["unique_sets_created"] += 1
            self._stats["unique_set_entries_created"] += len(normalized)
            return interned

    def get_statistics(self) -> dict[str, int]:
        with self._lock:
            live = list(self._mappings.values())
            live_sets = list(self._sets.values())
            stats = dict(self._stats)
        stats.update({
            "live_unique_mappings": len(live),
            "live_unique_entries": sum(len(mapping) for mapping in live),
            "live_unique_sets": len(live_sets),
            "live_unique_set_entries": sum(len(values) for values in live_sets),
        })
        return stats

    def __getstate__(self) -> dict[str, object]:
        return {}

    def __setstate__(self, _state: dict[str, object]) -> None:
        self.__init__()
