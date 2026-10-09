#!/usr/bin/env python3
"""Function-name semantic classification helpers for direct-call replay."""

from __future__ import annotations

import re
from typing import List, Tuple

from .runner_common import _normalize_name


def _summary_symbol_tokens(symbol: object) -> Tuple[str, str, List[str]]:
    raw = str(symbol or "").strip().lower()
    normalized = _normalize_name(raw)
    tokens = [part for part in re.split(r"[^a-z0-9]+", raw) if part]
    return raw, normalized, tokens


def _is_blocking_summary_symbol(symbol: object) -> bool:
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not raw:
        return False
    exact_tokens = {
        "wait",
        "delay",
        "sleep",
        "timeout",
        "poll",
        "mutex",
        "sema",
        "sem",
        "ztimer",
        "lock",
        "unlock",
    }
    if any(token in exact_tokens for token in tokens):
        return True
    substrings = (
        "waiton",
        "untiltimeout",
        "semwait",
        "semawait",
        "mutexlock",
        "mutexunlock",
        "lockacquire",
        "lockrelease",
        "delayms",
        "delayus",
        "sleepms",
    )
    return any(token in normalized for token in substrings)


def _is_safe_io_summary_symbol(symbol: object) -> bool:
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not raw:
        return False
    exact_normalized = {
        "digitalwrite",
        "digitaliowrite",
        "halgpiowritepin",
        "pinmode",
        "analogwrite",
        "shiftout",
        "tone",
        "notone",
        "digitalread",
        "digitalioread",
        "halgpioreadpin",
        "adcreadvalue",
    }
    if normalized in exact_normalized:
        return True
    prefixes = (
        "liquidcrystal",
    )
    if any(normalized.startswith(prefix) for prefix in prefixes):
        return True
    substrings = (
        "digitalwrite",
        "digitaliowrite",
        "halgpiowritepin",
        "analogwrite",
        "writebyte",
        "writebytes",
        "i2cwrite",
        "spisend",
        "spitransmit",
        "uartwrite",
        "uarttransmit",
        "serialwrite",
        "halmemwrite",
        "hali2cmemwrite",
        "hali2cmastertransmit",
        "haluarttransmit",
        "halspitransmit",
        "setpin",
        "writepin",
        "reportdigital",
        "reportanalog",
        "senddata",
        "write8bits",
        "write4bits",
        "pulseenable",
    )
    if any(part in normalized for part in substrings):
        return True
    if (
        ("i2c" in normalized and "write" in normalized)
        or ("spi" in normalized and ("write" in normalized or "send" in normalized or "transmit" in normalized))
        or ("uart" in normalized and ("write" in normalized or "send" in normalized or "transmit" in normalized))
        or "serialbaud" in normalized
        or "serialformat" in normalized
        or "serialirqhandler" in normalized
        or "pinmappinout" in normalized
        or "gpioinit" in normalized
        or "i2cinit" in normalized
        or "spiinit" in normalized
        or "uartinit" in normalized
        or "aquire" in normalized
        or "acquire" in normalized
        or "release" in normalized
    ):
        return True
    if any(token in {"strobe", "blink"} for token in tokens):
        return True
    return False


def _is_bus_write_summary_symbol(symbol: object) -> bool:
    _raw, normalized, _tokens = _summary_symbol_tokens(symbol)
    if not normalized:
        return False
    substrings = (
        "writebyte",
        "writebytes",
        "i2cwrite",
        "spisend",
        "spitransmit",
        "uartwrite",
        "uarttransmit",
        "serialwrite",
        "halmemwrite",
        "hali2cmemwrite",
        "hali2cmastertransmit",
        "haluarttransmit",
        "halspitransmit",
    )
    return any(part in normalized for part in substrings)


def _is_time_summary_symbol(symbol: object) -> bool:
    _raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not normalized:
        return False
    if normalized in {
        "millis",
        "micros",
        "getcurrentmicro",
        "getcurrentmilli",
        "halgettick",
    }:
        return True
    if "getcurrentmicro" in normalized or "getcurrentmilli" in normalized:
        return True
    return any(token in {"millis", "micros"} for token in tokens)


def _is_configuration_query_summary_symbol(symbol: object) -> bool:
    _raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not normalized:
        return False
    if normalized in {"ispinconfigured", "isconfigured", "getgpioport", "getgpiopin"}:
        return True
    if any(token in {"isconfigured", "ispinconfigured"} for token in tokens):
        return True
    return any(part in normalized for part in ("ispinconfigured", "getgpioport", "getgpiopin"))


def _is_buffered_input_summary_symbol(symbol: object) -> bool:
    """Input helpers that usually return data through caller-owned buffers."""
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not normalized:
        return False
    if raw in {"read", "_read", "_read_r", "__sread"}:
        return True
    if _is_blocking_summary_symbol(symbol):
        return False
    if any(token in {"available", "peek", "poll", "isready", "isavailable", "getstate", "getstatus"} for token in tokens):
        return False
    if any(part in normalized for part in (
            "directserial4read",
            "mbed6stream4read",
            "mbed3i2c4read",
            "haluartreceive",
            "halspimasterreceive",
            "hali2cmasterreceive",
    )):
        return True
    substrings = (
        "memread",
        "readmem",
        "readbyte",
        "readbytes",
        "readdata",
        "readraw",
        "readaccel",
        "readgyro",
        "readtemp",
        "readtemperature",
        "spitransfer",
        "i2cread",
        "uartread",
        "serialread",
        "haladcgetvalue",
    )
    return any(part in normalized for part in substrings)


def _is_status_return_buffered_read_symbol(symbol: object) -> bool:
    """HAL-style reads return status in R0 and place data in an output buffer."""
    _raw, normalized, _tokens = _summary_symbol_tokens(symbol)
    if not normalized:
        return False
    if "adcgetvalue" in normalized:
        return False
    if not any(prefix in normalized for prefix in ("hali2c", "halspi", "haluart")):
        return "mbed3i2c4read" in normalized or ("i2c4read" in normalized and "mbed" in normalized)
    return any(part in normalized for part in ("read", "receive"))


def _summary_symbol_priority(symbol: object) -> int:
    """Prioritize callsite summaries that can unlock state/input paths."""
    if _is_configuration_query_summary_symbol(symbol):
        return 25
    if _is_blocking_summary_symbol(symbol):
        return 92
    if _is_buffered_input_summary_symbol(symbol):
        return 75
    if _is_input_state_summary_symbol(symbol):
        return 95
    if _is_time_summary_symbol(symbol):
        return 82
    if _is_bus_write_summary_symbol(symbol):
        return 70
    if _is_safe_io_summary_symbol(symbol):
        return 20
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not raw:
        return 10
    if normalized in {"libcinitarray", "systeminit", "init"}:
        return 5
    if any(token in {"init", "setup"} for token in tokens):
        return 15
    if any(part in normalized for part in ("write", "print", "puts", "putchar")):
        return 20
    return 40


def _is_low_value_summary_symbol(symbol: object) -> bool:
    """Identify summaries that often resume boilerplate but rarely change state."""
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not raw:
        return False
    if _is_configuration_query_summary_symbol(symbol):
        return True
    if _is_time_summary_symbol(symbol) or _is_blocking_summary_symbol(symbol) or _is_input_state_summary_symbol(symbol):
        return False
    if normalized in {"libcinitarray", "systeminit", "init"}:
        return True
    if any(token in {"init"} for token in tokens):
        return True
    if _is_bus_write_summary_symbol(symbol):
        return False
    if _is_safe_io_summary_symbol(symbol):
        return True
    if any(part in normalized for part in ("write", "print", "puts", "putchar")):
        return True
    return False


def _is_input_state_summary_symbol(symbol: object) -> bool:
    raw, normalized, tokens = _summary_symbol_tokens(symbol)
    if not raw:
        return False
    if raw in {"serial_readable", "serial_getc", "fgetc", "getchar"}:
        return True
    if any(part in normalized for part in (
        "readable",
        "available",
        "isready",
        "isavailable",
        "mbed9mbedgetc",
        "stream4getc",
        "serial5getc",
        "serialbase10basegetc",
    )):
        return True
    exact_tokens = {
        "available",
        "read",
        "peek",
        "recv",
        "receive",
        "poll",
    }
    if any(token in exact_tokens for token in tokens):
        return True
    substrings = (
        "readdata",
        "readraw",
        "readbyte",
        "readbytes",
        "readaccel",
        "readgyro",
        "readtemp",
        "readtemperature",
        "readthermo",
        "readthermocouple",
        "digitalread",
        "readadc",
        "adcread",
        "adcreadvalue",
        "spitransfer",
        "i2cread",
        "uartread",
        "serialread",
        "serialavailable",
        "isconfigured",
        "ispinconfigured",
        "isready",
        "isavailable",
        "getstate",
        "getstatus",
    )
    return any(part in normalized for part in substrings)


def _summary_symbol_enabled(symbol: object) -> bool:
    return (
        _is_blocking_summary_symbol(symbol)
        or _is_safe_io_summary_symbol(symbol)
        or _is_time_summary_symbol(symbol)
        or _is_buffered_input_summary_symbol(symbol)
        or _is_input_state_summary_symbol(symbol)
    )
