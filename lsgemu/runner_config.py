#!/usr/bin/env python3
"""Runner-level default paths and semantic scheduling constants."""

from __future__ import annotations

from .deployment_config import configured_path
from .runner_common import PROJECT_ROOT


DEFAULT_REAL_TESTS_ROOT = configured_path("P2IM_REAL_TESTS_ROOT", PROJECT_ROOT / "testcase" / "real_tests")
DEFAULT_REAL_GATEWAY = DEFAULT_REAL_TESTS_ROOT / "P2IM.Gateway.elf"
DEFAULT_REAL_CNC = DEFAULT_REAL_TESTS_ROOT / "P2IM.CNC.elf"
DEFAULT_SAMPLE_GATEWAY = PROJECT_ROOT / "sample_firmware" / "P2IM.Gateway.elf"


SEMANTIC_FRONTIER_CATEGORY_PRIORITY = {
    "protocol_callback": 90,
    "callback_or_hook": 85,
    "stream_or_protocol_input": 80,
    "interrupt_or_vector_handler": 70,
    "hal_or_peripheral_state": 60,
    "rtos_or_task_entry": 55,
    "uncategorized_static_code": 25,
    "io_output_path": 8,
    "libc_or_runtime_path": 5,
    "startup_teardown_path": 3,
    "libgcc_runtime_path": 2,
    "error_or_fault_path": 0,
}


EXTENDED_STATIC_ROOT_CATEGORIES = {
    "rtos_or_task_entry",
    "callback_or_hook",
    "protocol_callback",
    "stream_or_protocol_input",
    "interrupt_or_vector_handler",
    "startup_teardown_path",
}
