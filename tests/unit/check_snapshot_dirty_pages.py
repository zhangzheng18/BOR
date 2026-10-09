#!/usr/bin/env python3
"""Regression test for snapshot dirty-page provenance propagation."""

from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.analysis.branch_snapshot_manager import BranchSnapshotManager
from lsgemu.analysis.unique_bb_snapshot_manager import UniqueBBSnapshotManager


class DummyUC:
    def __init__(self):
        self.registers = {}
        self.memory = {}

    def reg_read(self, reg_id):
        return self.registers.get(reg_id, 0)

    def reg_write(self, reg_id, value):
        self.registers[reg_id] = int(value)

    def mem_read(self, address, size):
        return self.memory.get((int(address), int(size)), b"\x00" * int(size))

    def mem_write(self, address, data):
        self.memory[(int(address), len(data))] = bytes(data)


def main() -> int:
    uc = DummyUC()
    branch_mgr = BranchSnapshotManager()
    branch_mgr.set_dirty_page_provider(lambda: {0x20001000, 0x20002000})
    snapshot = branch_mgr.save_snapshot(
        uc,
        0x08001000,
        0x08001010,
        0x08001014,
        "BNE",
        mmio_state={},
    )
    dirty_pages_ok = snapshot.dirty_pages == {0x20001000, 0x20002000}

    unique_mgr = UniqueBBSnapshotManager(max_snapshots=2)
    unique_snap = unique_mgr.create_snapshot(
        uc,
        0x08002000,
        dirty_pages={0x20003000},
    )
    unique_dirty_ok = unique_snap.dirty_pages == {0x20003000}
    anchor_ok = unique_mgr.lightweight_anchors and 0x20003000 in unique_mgr.lightweight_anchors[-1].dirty_page_hashes
    print(json.dumps({
        "branch_dirty_pages": [f"0x{x:08x}" for x in sorted(snapshot.dirty_pages)],
        "unique_dirty_pages": [f"0x{x:08x}" for x in sorted(unique_snap.dirty_pages)],
        "anchor_dirty_pages": [f"0x{x:08x}" for x in sorted(unique_mgr.lightweight_anchors[-1].dirty_page_hashes.keys())] if unique_mgr.lightweight_anchors else [],
        "checks": {
            "dirty_pages_ok": dirty_pages_ok,
            "unique_dirty_ok": unique_dirty_ok,
            "anchor_ok": anchor_ok,
        },
    }, indent=2))
    return 0 if dirty_pages_ok and unique_dirty_ok and anchor_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
