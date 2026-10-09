"""Memory-efficient, read-only storage for cached CFG reachability sets.

The scheduler consumes transitive reachability as a set of basic-block
addresses.  Retaining one Python ``set`` per starting block is expensive for
large firmware images because every address is stored repeatedly.  This
module stores addresses in packed native arrays and materializes a temporary
set only inside the scheduler call that needs native set operations.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Set as AbstractSet
import sys
from typing import Iterator, Optional


class PackedReachabilitySet(AbstractSet[int]):
    """Read-only set view backed by a packed 32- or 64-bit address array.

    The view deliberately returns ordinary ``set`` objects for binary set
    operations.  Existing scheduler code therefore keeps its established
    semantics, while the long-lived cache uses four bytes per Cortex-M
    address instead of one Python object and hash-table slot per membership.
    """

    __slots__ = ("_addresses",)

    def __init__(self, addresses: array) -> None:
        self._addresses = addresses

    @classmethod
    def from_nodes(
        cls,
        nodes: Iterable[int],
    ) -> Optional["PackedReachabilitySet"]:
        """Pack addresses without changing the represented membership."""
        # Keep a one-shot iterator available for the wider-type fallback;
        # normal scheduler inputs are sets and avoid this tuple allocation.
        values = nodes if hasattr(nodes, "__len__") else tuple(nodes)
        try:
            return cls(array("I", values))
        except (OverflowError, TypeError, ValueError):
            try:
                return cls(array("Q", values))
            except (OverflowError, TypeError, ValueError):
                return None

    @property
    def storage_bytes(self) -> int:
        """Approximate retained size of the packed array and its payload."""
        return int(sys.getsizeof(self._addresses))

    def __contains__(self, value: object) -> bool:
        try:
            return value in self._addresses
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        yield from self._addresses

    def __len__(self) -> int:
        return len(self._addresses)

    def materialize(self) -> set[int]:
        """Return the equivalent ordinary set when a mutable set is needed."""
        return set(self._addresses)

    def __and__(self, other: Iterable[int]) -> set[int]:
        if isinstance(other, (set, frozenset)):
            # ``set.intersection`` consumes the packed array in C and avoids
            # executing a Python membership loop for every basic block.
            return other.intersection(self._addresses)
        return set(self._addresses).intersection(other)

    def __rand__(self, other: Iterable[int]) -> set[int]:
        return self.__and__(other)

    def __sub__(self, other: Iterable[int]) -> set[int]:
        return set(self._addresses).difference(other)

    def __rsub__(self, other: Iterable[int]) -> set[int]:
        return set(other).difference(self._addresses)

    def __or__(self, other: Iterable[int]) -> set[int]:
        return set(self._addresses).union(other)

    def __ror__(self, other: Iterable[int]) -> set[int]:
        return set(other).union(self._addresses)

    def __xor__(self, other: Iterable[int]) -> set[int]:
        return set(self._addresses).symmetric_difference(other)

    def __rxor__(self, other: Iterable[int]) -> set[int]:
        return set(other).symmetric_difference(self._addresses)

    def __repr__(self) -> str:
        return repr(self.materialize())


# Kept as a private-era compatibility alias for focused tests and external
# analysis scripts that imported the original helper name.
ReachabilitySetView = PackedReachabilitySet
