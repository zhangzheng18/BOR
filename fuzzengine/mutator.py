#!/usr/bin/env python3
"""Deterministic byte-level mutators for firmware stream-input fuzzing."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence


INTERESTING_8 = [0x00, 0x01, 0x02, 0x7F, 0x80, 0xFE, 0xFF, ord("\n"), ord("\r")]
INTERESTING_16 = [0x0000, 0x0001, 0x007F, 0x0080, 0x00FF, 0x0100, 0x7FFF, 0x8000, 0xFFFF]
INTERESTING_32 = [
    0x00000000,
    0x00000001,
    0x0000007F,
    0x00000080,
    0x000000FF,
    0x00000100,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
]
TOKENS = [b"\x00", b"\xff", b"\r\n", b"AT", b"GET ", b"POST ", b"AAAA", b"%x", b"%s", b"../"]


@dataclass
class MutationConfig:
    max_size: int = 4096
    min_size: int = 0
    havoc_ops: int = 8
    dictionary: Sequence[bytes] = ()


class ByteMutator:
    """Small AFL-style mutator intended for replay-oriented LSGEmu campaigns."""

    def __init__(self, seed: int = 0, config: Optional[MutationConfig] = None):
        self.random = random.Random(int(seed))
        self.config = config or MutationConfig()
        self.tokens = list(TOKENS) + [bytes(item) for item in self.config.dictionary]

    def mutate(self, data: bytes) -> bytes:
        if not data and self.config.min_size > 0:
            data = b"\x00" * self.config.min_size
        mode = self.random.choice([
            "bitflip",
            "byteflip",
            "interesting",
            "arith",
            "insert",
            "delete",
            "splice_token",
            "havoc",
        ])
        if mode == "bitflip":
            out = self._bitflip(data)
        elif mode == "byteflip":
            out = self._byteflip(data)
        elif mode == "interesting":
            out = self._interesting(data)
        elif mode == "arith":
            out = self._arith(data)
        elif mode == "insert":
            out = self._insert(data)
        elif mode == "delete":
            out = self._delete(data)
        elif mode == "splice_token":
            out = self._splice_token(data)
        else:
            out = data
            for _ in range(max(1, int(self.config.havoc_ops))):
                out = self.mutate(out)
        return self._clamp(out)

    def deterministic_cases(self, data: bytes, *, limit: int = 64) -> List[bytes]:
        """Generate a small deterministic queue for fresh imported seeds."""
        cases: List[bytes] = []
        size = len(data)
        for offset in range(min(size, max(0, limit))):
            buf = bytearray(data)
            buf[offset] ^= 0xFF
            cases.append(self._clamp(bytes(buf)))
            if len(cases) >= limit:
                return cases
            buf = bytearray(data)
            buf[offset] = 0
            cases.append(self._clamp(bytes(buf)))
            if len(cases) >= limit:
                return cases
        for token in self.tokens:
            cases.append(self._clamp(data + token))
            if len(cases) >= limit:
                return cases
        if not data:
            for token in self.tokens:
                cases.append(self._clamp(token))
                if len(cases) >= limit:
                    break
        return cases

    def _rand_offset(self, data: bytes, allow_end: bool = False) -> int:
        upper = len(data) if allow_end else max(0, len(data) - 1)
        return self.random.randint(0, upper) if upper > 0 else 0

    def _bitflip(self, data: bytes) -> bytes:
        if not data:
            return b"\x01"
        buf = bytearray(data)
        offset = self._rand_offset(buf)
        buf[offset] ^= 1 << self.random.randrange(8)
        return bytes(buf)

    def _byteflip(self, data: bytes) -> bytes:
        if not data:
            return bytes([self.random.randrange(256)])
        buf = bytearray(data)
        offset = self._rand_offset(buf)
        buf[offset] = self.random.randrange(256)
        return bytes(buf)

    def _interesting(self, data: bytes) -> bytes:
        if not data:
            return bytes([self.random.choice(INTERESTING_8)])
        width = self.random.choice([1, 2, 4])
        if len(data) < width:
            width = 1
        offset = self.random.randint(0, max(0, len(data) - width))
        if width == 1:
            value = self.random.choice(INTERESTING_8)
        elif width == 2:
            value = self.random.choice(INTERESTING_16)
        else:
            value = self.random.choice(INTERESTING_32)
        buf = bytearray(data)
        buf[offset:offset + width] = int(value).to_bytes(width, "little", signed=False)
        return bytes(buf)

    def _arith(self, data: bytes) -> bytes:
        if not data:
            return b"\x01"
        width = 1 if len(data) < 2 else self.random.choice([1, 2, 4 if len(data) >= 4 else 2])
        offset = self.random.randint(0, max(0, len(data) - width))
        delta = self.random.choice([-32, -16, -1, 1, 16, 32])
        value = int.from_bytes(data[offset:offset + width], "little", signed=False)
        value = (value + delta) & ((1 << (8 * width)) - 1)
        buf = bytearray(data)
        buf[offset:offset + width] = value.to_bytes(width, "little", signed=False)
        return bytes(buf)

    def _insert(self, data: bytes) -> bytes:
        token = self.random.choice(self.tokens) if self.tokens else bytes([self.random.randrange(256)])
        if self.random.random() < 0.5:
            token = bytes(self.random.randrange(256) for _ in range(self.random.randint(1, 8)))
        offset = self._rand_offset(data, allow_end=True)
        return data[:offset] + token + data[offset:]

    def _delete(self, data: bytes) -> bytes:
        if len(data) <= self.config.min_size:
            return data
        start = self._rand_offset(data)
        max_len = max(1, min(32, len(data) - start))
        length = self.random.randint(1, max_len)
        return data[:start] + data[start + length:]

    def _splice_token(self, data: bytes) -> bytes:
        token = self.random.choice(self.tokens) if self.tokens else b"\x00"
        if not data:
            return token
        offset = self._rand_offset(data, allow_end=True)
        replace = self.random.randint(0, min(len(token), max(0, len(data) - offset)))
        return data[:offset] + token + data[offset + replace:]

    def _clamp(self, data: bytes) -> bytes:
        if len(data) > self.config.max_size:
            data = data[: self.config.max_size]
        if len(data) < self.config.min_size:
            data = data + b"\x00" * (self.config.min_size - len(data))
        return data


def parse_dictionary(items: Optional[Iterable[str]]) -> List[bytes]:
    out: List[bytes] = []
    for item in items or []:
        text = str(item)
        if text.startswith("hex:"):
            try:
                out.append(bytes.fromhex(text[4:]))
            except ValueError:
                continue
        else:
            out.append(text.encode("utf-8", errors="replace"))
    return out
