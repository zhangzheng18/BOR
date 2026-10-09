#!/usr/bin/env python3
"""Deterministic checksum hypotheses over already observed input events.

The routines in this module do not claim that a firmware predicate is a
checksum.  They generate bounded, reproducible candidate assignments for
common wire formats; concrete force-free replay remains the acceptance oracle.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


U32_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class ChecksumInputSite:
    site_index: int
    kind: str
    address: int
    read_pc: int
    occurrence: int
    width: int
    observed_value: int
    event_index: int
    trace_event_id: int = 0

    @property
    def byte_width(self) -> int:
        return max(1, min(4, (int(self.width) + 7) // 8))

    def observed_bytes(self) -> bytes:
        mask = (1 << (self.byte_width * 8)) - 1
        return (int(self.observed_value) & mask).to_bytes(
            self.byte_width,
            "little",
        )


@dataclass(frozen=True)
class ChecksumAssignment:
    site_index: int
    value: int
    observed_value: int


@dataclass(frozen=True)
class ChecksumHypothesis:
    strategy: str
    algorithm: str
    field_location: str
    assignments: Tuple[ChecksumAssignment, ...]
    checksum_width: int
    evidence_complete: bool = False


def _xor_value(data: bytes, width: int) -> int:
    mask = (1 << (width * 8)) - 1
    value = 0
    if width == 1:
        for byte in data:
            value ^= int(byte)
        return value & mask
    for offset in range(0, len(data), width):
        chunk = data[offset : offset + width]
        value ^= int.from_bytes(chunk.ljust(width, b"\x00"), "little")
    return value & mask


def _sum_value(data: bytes, width: int) -> int:
    mask = (1 << (width * 8)) - 1
    if width == 1:
        return sum(int(byte) for byte in data) & mask
    return sum(
        int.from_bytes(data[offset : offset + width].ljust(width, b"\x00"), "little")
        for offset in range(0, len(data), width)
    ) & mask


def _crc8(data: bytes, *, polynomial: int, init: int = 0) -> int:
    crc = int(init) & 0xFF
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            crc = ((crc << 1) ^ polynomial) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc & 0xFF


def _crc8_reflected(data: bytes, *, polynomial: int, init: int = 0) -> int:
    crc = int(init) & 0xFF
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            crc = ((crc >> 1) ^ polynomial) & 0xFF if crc & 1 else (crc >> 1) & 0xFF
    return crc & 0xFF


def _crc16(
    data: bytes,
    *,
    polynomial: int,
    init: int,
    reflected: bool,
    xorout: int = 0,
) -> int:
    crc = int(init) & 0xFFFF
    for byte in data:
        if reflected:
            crc ^= int(byte)
            for _ in range(8):
                crc = ((crc >> 1) ^ polynomial) & 0xFFFF if crc & 1 else (crc >> 1) & 0xFFFF
        else:
            crc ^= int(byte) << 8
            for _ in range(8):
                crc = ((crc << 1) ^ polynomial) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return (crc ^ int(xorout)) & 0xFFFF


def _algorithms(width: int) -> Sequence[Tuple[str, Callable[[bytes], int]]]:
    if width == 1:
        return (
            ("xor8", lambda data: _xor_value(data, 1)),
            ("sum8", lambda data: _sum_value(data, 1)),
            ("sum8_twos_complement", lambda data: (-_sum_value(data, 1)) & 0xFF),
            ("crc8_atm", lambda data: _crc8(data, polynomial=0x07, init=0x00)),
            ("crc8_atm_ff", lambda data: _crc8(data, polynomial=0x07, init=0xFF)),
            ("crc8_maxim", lambda data: _crc8_reflected(data, polynomial=0x8C, init=0x00)),
        )
    if width == 2:
        return (
            ("crc16_ccitt_false", lambda data: _crc16(data, polynomial=0x1021, init=0xFFFF, reflected=False)),
            ("crc16_modbus", lambda data: _crc16(data, polynomial=0xA001, init=0xFFFF, reflected=True)),
            ("crc16_ibm", lambda data: _crc16(data, polynomial=0xA001, init=0x0000, reflected=True)),
            ("crc16_xmodem", lambda data: _crc16(data, polynomial=0x1021, init=0x0000, reflected=False)),
            ("crc16_x25", lambda data: _crc16(data, polynomial=0x8408, init=0xFFFF, reflected=True, xorout=0xFFFF)),
            ("xor16", lambda data: _xor_value(data, 2)),
            ("sum16", lambda data: _sum_value(data, 2)),
            ("sum16_twos_complement", lambda data: (-_sum_value(data, 2)) & 0xFFFF),
        )
    if width == 4:
        return (
            ("crc32_ieee", lambda data: binascii.crc32(data) & U32_MASK),
            ("xor32", lambda data: _xor_value(data, 4)),
            ("sum32", lambda data: _sum_value(data, 4)),
            ("sum32_twos_complement", lambda data: (-_sum_value(data, 4)) & U32_MASK),
        )
    return ()


def _split_field_sites(
    sites: Sequence[ChecksumInputSite],
    width: int,
    location: str,
) -> Optional[Tuple[Sequence[ChecksumInputSite], Sequence[ChecksumInputSite]]]:
    if not sites or width <= 0:
        return None
    if location == "suffix":
        selected: List[ChecksumInputSite] = []
        total = 0
        for site in reversed(sites):
            selected.append(site)
            total += site.byte_width
            if total >= width:
                break
        if total != width or len(selected) >= len(sites):
            return None
        field_sites = list(reversed(selected))
        payload_sites = list(sites[: len(sites) - len(field_sites)])
    else:
        selected = []
        total = 0
        for site in sites:
            selected.append(site)
            total += site.byte_width
            if total >= width:
                break
        if total != width or len(selected) >= len(sites):
            return None
        field_sites = selected
        payload_sites = list(sites[len(field_sites) :])
    if not payload_sites:
        return None
    return payload_sites, field_sites


def _assign_field_bytes(
    field_sites: Sequence[ChecksumInputSite],
    field_bytes: bytes,
) -> Tuple[ChecksumAssignment, ...]:
    assignments: List[ChecksumAssignment] = []
    cursor = 0
    for site in field_sites:
        chunk = field_bytes[cursor : cursor + site.byte_width]
        if len(chunk) != site.byte_width:
            return tuple()
        value = int.from_bytes(chunk, "little")
        cursor += site.byte_width
        if value == (int(site.observed_value) & ((1 << (site.byte_width * 8)) - 1)):
            continue
        assignments.append(ChecksumAssignment(
            site_index=int(site.site_index),
            value=value,
            observed_value=int(site.observed_value),
        ))
    return tuple(assignments)


def generate_checksum_hypotheses(
    inputs: Iterable[ChecksumInputSite],
    *,
    max_hypotheses: int = 16,
    max_sites: int = 256,
    max_payload_bytes: int = 1024,
    selection_offset: int = 0,
) -> Tuple[ChecksumHypothesis, ...]:
    """Generate bounded checksum-field candidates at message prefix/suffix."""
    ordered = sorted(
        list(inputs or ()),
        key=lambda item: (
            int(item.trace_event_id or 0) or int(item.event_index),
            int(item.event_index),
            int(item.site_index),
        ),
    )[: max(2, int(max_sites))]
    if len(ordered) < 2:
        return tuple()

    hypotheses: List[ChecksumHypothesis] = []
    seen: set[Tuple[Tuple[int, int], ...]] = set()
    for width in (1, 2, 4):
        for location in ("suffix", "prefix"):
            split = _split_field_sites(ordered, width, location)
            if split is None:
                continue
            payload_sites, field_sites = split
            payload = b"".join(site.observed_bytes() for site in payload_sites)
            if not payload or len(payload) > max(1, int(max_payload_bytes)):
                continue
            for algorithm, calculator in _algorithms(width):
                checksum = int(calculator(payload)) & ((1 << (width * 8)) - 1)
                for byte_order in ("little", "big"):
                    assignments = _assign_field_bytes(
                        field_sites,
                        checksum.to_bytes(width, byte_order),
                    )
                    signature = tuple(
                        (int(item.site_index), int(item.value))
                        for item in assignments
                    )
                    if not assignments or signature in seen:
                        continue
                    seen.add(signature)
                    hypotheses.append(ChecksumHypothesis(
                        strategy=f"checksum_field_{location}_{byte_order}",
                        algorithm=algorithm,
                        field_location=location,
                        assignments=assignments,
                        checksum_width=width,
                        evidence_complete=False,
                    ))
                    if len(hypotheses) >= max(32, int(max_hypotheses) * 16):
                        break

    buckets: Dict[Tuple[int, str], List[ChecksumHypothesis]] = {}
    for hypothesis in hypotheses:
        buckets.setdefault(
            (int(hypothesis.checksum_width), str(hypothesis.field_location)),
            [],
        ).append(hypothesis)
    scheduled: List[ChecksumHypothesis] = []
    bucket_order = [
        (1, "suffix"),
        (2, "suffix"),
        (4, "suffix"),
        (1, "prefix"),
        (2, "prefix"),
        (4, "prefix"),
    ]
    index = 0
    while True:
        added = False
        for key in bucket_order:
            items = buckets.get(key, [])
            if index >= len(items):
                continue
            scheduled.append(items[index])
            added = True
        if not added:
            break
        index += 1
    if not scheduled:
        return tuple()
    offset = max(0, int(selection_offset or 0)) % len(scheduled)
    rotated = scheduled[offset:] + scheduled[:offset]
    return tuple(rotated[: max(1, int(max_hypotheses))])
