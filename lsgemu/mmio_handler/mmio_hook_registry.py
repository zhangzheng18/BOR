"""Registry for primary MMIO handlers attached to Unicorn instances.

EnhancedMMIOHandler is still used by replay stages as a constraint/state
container.  When an IntelligentEmulator already owns the MMIO hooks for the
same Unicorn instance, EnhancedMMIOHandler should bridge into that primary
handler instead of installing a second read/write hook.
"""

from __future__ import annotations

from threading import RLock
from typing import Dict, Optional
import weakref


_LOCK = RLock()
_PRIMARY_MMIO_HANDLERS: "weakref.WeakKeyDictionary[object, object]" = weakref.WeakKeyDictionary()
# Some extension types do not support weak references. Keep an identity-checked
# fallback so object-id reuse can never return another Unicorn instance's
# handler. Correct close/unregister still removes these strong references.
_PRIMARY_MMIO_FALLBACK: Dict[int, tuple[object, object]] = {}


def register_primary_mmio_handler(uc: object, handler: object) -> None:
    if uc is None or handler is None:
        return
    with _LOCK:
        try:
            _PRIMARY_MMIO_HANDLERS[uc] = handler
            _PRIMARY_MMIO_FALLBACK.pop(id(uc), None)
        except TypeError:
            _PRIMARY_MMIO_FALLBACK[id(uc)] = (uc, handler)


def unregister_primary_mmio_handler(uc: object, handler: Optional[object] = None) -> None:
    if uc is None:
        return
    with _LOCK:
        try:
            registered = _PRIMARY_MMIO_HANDLERS.get(uc)
        except TypeError:
            registered = None
        if registered is not None:
            if handler is None or registered is handler:
                try:
                    _PRIMARY_MMIO_HANDLERS.pop(uc, None)
                except TypeError:
                    pass

        fallback = _PRIMARY_MMIO_FALLBACK.get(id(uc))
        if fallback is None or fallback[0] is not uc:
            return
        if handler is None or fallback[1] is handler:
            _PRIMARY_MMIO_FALLBACK.pop(id(uc), None)


def get_primary_mmio_handler(uc: object) -> Optional[object]:
    if uc is None:
        return None
    with _LOCK:
        try:
            handler = _PRIMARY_MMIO_HANDLERS.get(uc)
        except TypeError:
            handler = None
        if handler is not None:
            return handler
        fallback = _PRIMARY_MMIO_FALLBACK.get(id(uc))
        if fallback is None:
            return None
        if fallback[0] is not uc:
            _PRIMARY_MMIO_FALLBACK.pop(id(uc), None)
            return None
        return fallback[1]
