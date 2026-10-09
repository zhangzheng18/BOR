#!/usr/bin/env python3
"""Corpus management for LSGEmu fuzzing campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set

try:
    from .utils import load_seed_files, safe_fragment
except ImportError:
    from utils import load_seed_files, safe_fragment

try:
    from lsgemu.artifact_io import atomic_copy_file, atomic_json_dump, atomic_write_bytes
except ImportError:  # pragma: no cover - supports running fuzzengine standalone.
    atomic_copy_file = atomic_json_dump = atomic_write_bytes = None


def seed_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SeedRecord:
    """One crash-campaign seed plus replay-context metadata."""

    seed_id: str
    path: str
    sha256: str
    size: int
    parent_id: Optional[str] = None
    generation: int = 0
    origin: str = "initial"
    created_at: float = field(default_factory=time.time)
    runs: int = 0
    best_new_bbs: int = 0
    best_covered_bbs: int = 0
    best_valid_covered_bbs: int = 0
    crash_buckets: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SeedRecord":
        return cls(
            seed_id=str(payload["seed_id"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size=int(payload.get("size") or 0),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") else None,
            generation=int(payload.get("generation") or 0),
            origin=str(payload.get("origin") or "initial"),
            created_at=float(payload.get("created_at") or time.time()),
            runs=int(payload.get("runs") or 0),
            best_new_bbs=int(payload.get("best_new_bbs") or 0),
            best_covered_bbs=int(payload.get("best_covered_bbs") or 0),
            best_valid_covered_bbs=int(payload.get("best_valid_covered_bbs") or 0),
            crash_buckets=list(payload.get("crash_buckets") or []),
            metadata=dict(payload.get("metadata") or {}),
        )


class CorpusStore:
    """Persistent seed corpus for crash discovery.

    Coverage remains replay context only. It is retained to keep seeds that reach
    different firmware states, not as the campaign objective.
    """

    def __init__(self, root: object):
        self.root = Path(root)
        self.queue_dir = self.root / "queue"
        self.interesting_dir = self.root / "interesting"
        self.import_dir = self.root / "imports"
        self.meta_path = self.root / "corpus.json"
        for directory in (self.queue_dir, self.interesting_dir, self.import_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, SeedRecord] = {}
        self.hash_index: Dict[str, str] = {}
        self.global_coverage: Set[int] = set()
        self.load()

    def load(self) -> None:
        if not self.meta_path.exists():
            return
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.records.clear()
        self.hash_index.clear()
        for item in payload.get("records") or []:
            if not isinstance(item, dict):
                continue
            record = SeedRecord.from_dict(item)
            self.records[record.seed_id] = record
            self.hash_index[record.sha256] = record.seed_id
        self.global_coverage = {
            int(value)
            for value in payload.get("global_coverage") or []
            if isinstance(value, int)
        }

    def save(self) -> None:
        payload = {
            "schema": "lsgemu.fuzzengine.corpus.v1",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "records": [record.to_dict() for record in self.records.values()],
            "global_coverage": sorted(self.global_coverage),
        }
        if atomic_json_dump is not None:
            atomic_json_dump(payload, self.meta_path, indent=2, ensure_ascii=False)
        else:
            self.meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_seed(
        self,
        data: bytes,
        *,
        parent_id: Optional[str] = None,
        generation: int = 0,
        origin: str = "generated",
        metadata: Optional[Dict[str, object]] = None,
        force_queue: bool = True,
    ) -> SeedRecord:
        digest = seed_sha256(data)
        existing_id = self.hash_index.get(digest)
        if existing_id and existing_id in self.records:
            return self.records[existing_id]

        seed_id = f"id_{len(self.records):06d}_{digest[:12]}"
        directory = self.queue_dir if force_queue else self.import_dir
        path = directory / f"{safe_fragment(seed_id, default='seed', max_len=96)}.seed"
        if atomic_write_bytes is not None:
            atomic_write_bytes(path, data, durable=False)
        else:
            path.write_bytes(data)
        record = SeedRecord(
            seed_id=seed_id,
            path=str(path),
            sha256=digest,
            size=len(data),
            parent_id=parent_id,
            generation=generation,
            origin=origin,
            metadata=dict(metadata or {}),
        )
        self.records[seed_id] = record
        self.hash_index[digest] = seed_id
        return record

    def import_paths(self, paths: Sequence[object]) -> List[SeedRecord]:
        records: List[SeedRecord] = []
        for path in load_seed_files(paths):
            try:
                data = path.read_bytes()
            except Exception:
                continue
            records.append(self.add_seed(
                data,
                origin="imported",
                metadata={"source_path": str(path)},
                force_queue=True,
            ))
        if not records and not self.records:
            records.append(self.add_seed(b"", origin="empty_seed"))
        return records

    def mark_result(
        self,
        record: SeedRecord,
        *,
        covered_bbs: Optional[Iterable[int]] = None,
        covered_bbs_count: int = 0,
        valid_covered_bbs: int = 0,
        crash_bucket: Optional[str] = None,
    ) -> int:
        record.runs += 1
        record.best_covered_bbs = max(record.best_covered_bbs, int(covered_bbs_count or 0))
        record.best_valid_covered_bbs = max(record.best_valid_covered_bbs, int(valid_covered_bbs or 0))
        if crash_bucket and crash_bucket not in record.crash_buckets:
            record.crash_buckets.append(crash_bucket)

        new_bbs = 0
        if covered_bbs is not None:
            normalized = {int(bb) & ~1 for bb in covered_bbs}
            novel = normalized - self.global_coverage
            new_bbs = len(novel)
            if novel:
                self.global_coverage.update(novel)
                record.best_new_bbs = max(record.best_new_bbs, new_bbs)
                self._copy_interesting(record, new_bbs)
        return new_bbs

    def _copy_interesting(self, record: SeedRecord, new_bbs: int) -> None:
        src = Path(record.path)
        if not src.exists():
            return
        dst = self.interesting_dir / f"{safe_fragment(record.seed_id, default='seed', max_len=96)}_new{int(new_bbs)}.seed"
        if not dst.exists():
            if atomic_copy_file is not None:
                atomic_copy_file(src, dst, durable=False)
            else:
                dst.write_bytes(src.read_bytes())

    def pick(self, index: int) -> SeedRecord:
        if not self.records:
            raise RuntimeError("corpus is empty")
        records = list(self.records.values())
        records.sort(key=lambda item: (item.runs, -item.best_new_bbs, item.generation, item.seed_id))
        return records[index % len(records)]
