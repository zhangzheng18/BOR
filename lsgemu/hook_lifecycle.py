"""Shared hook ownership helpers for Unicorn-backed LSGEmu components.

The runner creates hooks from several cooperating components.  Keeping the
owner lookup here lets those components share one lifecycle without importing
the concrete emulator class or changing their public constructors.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Optional
import weakref


_LOCK = RLock()
_WEAK_OWNERS: "weakref.WeakKeyDictionary[object, object]" = weakref.WeakKeyDictionary()
_FALLBACK_OWNERS: dict[int, tuple[object, object]] = {}


def _owner_value(value: object) -> Optional[object]:
    """Resolve a weak owner reference and discard dead references."""
    if isinstance(value, weakref.ReferenceType):
        return value()
    return value


def _weak_owner(owner: object) -> object:
    try:
        return weakref.ref(owner)
    except TypeError:
        return owner


def register_hook_owner(uc: object, owner: object) -> None:
    """Associate one Unicorn object with its emulator-level hook owner."""
    if uc is None or owner is None:
        return
    with _LOCK:
        try:
            _WEAK_OWNERS[uc] = _weak_owner(owner)
            _FALLBACK_OWNERS.pop(id(uc), None)
        except TypeError:
            # A few extension/test doubles are not weak-referenceable.  Keep
            # an identity-checked fallback so object-id reuse is harmless.
            _FALLBACK_OWNERS[id(uc)] = (uc, _weak_owner(owner))


def unregister_hook_owner(uc: object, owner: Optional[object] = None) -> None:
    """Remove an association, optionally only when it still belongs to owner."""
    if uc is None:
        return
    with _LOCK:
        try:
            registered = _owner_value(_WEAK_OWNERS.get(uc))
        except TypeError:
            registered = None
        if registered is not None and (owner is None or registered is owner):
            try:
                _WEAK_OWNERS.pop(uc, None)
            except TypeError:
                pass

        fallback = _FALLBACK_OWNERS.get(id(uc))
        if fallback is None or fallback[0] is not uc:
            return
        registered = _owner_value(fallback[1])
        if owner is None or registered is owner:
            _FALLBACK_OWNERS.pop(id(uc), None)


def get_hook_owner(uc: object) -> Optional[object]:
    """Return the live owner for ``uc`` if one has been registered."""
    if uc is None:
        return None
    with _LOCK:
        try:
            value = _WEAK_OWNERS.get(uc)
        except TypeError:
            value = None
        owner = _owner_value(value) if value is not None else None
        if owner is not None:
            return owner
        if value is not None:
            try:
                _WEAK_OWNERS.pop(uc, None)
            except TypeError:
                pass

        fallback = _FALLBACK_OWNERS.get(id(uc))
        if fallback is None:
            return None
        if fallback[0] is not uc:
            _FALLBACK_OWNERS.pop(id(uc), None)
            return None
        owner = _owner_value(fallback[1])
        if owner is None:
            _FALLBACK_OWNERS.pop(id(uc), None)
        return owner


def managed_hook_add(uc: object, *args: Any, **kwargs: Any):
    """Add a hook through the registered owner when available."""
    owner = get_hook_owner(uc)
    add = getattr(owner, "_add_owned_hook", None) if owner is not None else None
    if callable(add):
        return add(*args, **kwargs)
    return uc.hook_add(*args, **kwargs)


def managed_hook_del(uc: object, handle: object) -> None:
    """Delete a hook through the registered owner when available."""
    owner = get_hook_owner(uc)
    remove = getattr(owner, "_remove_owned_hook", None) if owner is not None else None
    if callable(remove):
        remove(handle)
        return
    uc.hook_del(handle)


def managed_emu_start(uc: object, *args: Any, **kwargs: Any):
    """Start emulation through the owner boundary when one is registered."""
    owner = get_hook_owner(uc)
    start = getattr(owner, "_managed_emu_start", None) if owner is not None else None
    if callable(start):
        return start(*args, **kwargs)
    return uc.emu_start(*args, **kwargs)


def managed_mem_map(
    uc: object,
    address: int,
    size: int,
    perms: Optional[int] = None,
) -> Any:
    """Map memory through the owning emulator's safe-point boundary.

    Components such as the ISR and auxiliary MMIO models may not know about
    the concrete emulator class.  Routing their dynamic maps through this
    helper prevents them from mutating Unicorn's address-space topology from a
    native callback.  Unowned Unicorn instances retain the historical direct
    API behavior.
    """
    owner = get_hook_owner(uc)
    mapper = getattr(owner, "_managed_mem_map", None) if owner is not None else None
    if callable(mapper):
        return mapper(int(address), int(size), perms=perms)
    if perms is None:
        return uc.mem_map(int(address), int(size))
    return uc.mem_map(int(address), int(size), int(perms))


__all__ = [
    "get_hook_owner",
    "managed_emu_start",
    "managed_mem_map",
    "managed_hook_add",
    "managed_hook_del",
    "register_hook_owner",
    "unregister_hook_owner",
]
