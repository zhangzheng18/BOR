#!/usr/bin/env python3
"""
Smoke-test RegisterTracer memory provenance across propagation and branch snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from lsgemu.register_tracer.register_tracer import RegisterSource, RegisterTracer


class DummyUC:
    pass


def main() -> int:
    static_bbs = {
        0x08001000: [
            {"address": 0x08001000, "mnemonic": "ANDS", "operands": "R3, R1, #0x1", "size": 2},
            {"address": 0x08001002, "mnemonic": "CMP", "operands": "R3, #0", "size": 2},
            {"address": 0x08001004, "mnemonic": "BNE", "operands": "0x08001020", "size": 2},
        ]
    }

    tracer = RegisterTracer(DummyUC(), static_bbs)
    original_source = RegisterSource(
        register="r1",
        source_type="memory",
        source_pc=0x080000F0,
        instruction="LDR R1, [R4, #0x20]",
        memory_address=0x20000020,
        operation="load",
        chain=("RAM[0x20000020]@0x080000f0",),
        call_depth=0,
    )

    tracer._set_memory_source(0x20000020, 4, original_source)
    recovered_source = tracer._get_memory_source(0x20000020, 4)
    recovered_ok = (
        recovered_source is not None
        and recovered_source.source_type == "memory"
        and recovered_source.memory_address == 0x20000020
        and recovered_source.mmio_address is None
    )

    if recovered_source is not None:
        tracer._set_register_source(
            "r1",
            recovered_source,
            "LDR R1, [R4, #0x20]",
            source_type="memory",
            memory_address=0x20000020,
        )

    tracer._propagate_register_sources(static_bbs[0x08001000][0])
    propagated_source = tracer.register_sources.get("r3")
    propagated_ok = (
        propagated_source is not None
        and propagated_source.source_type == "memory"
        and propagated_source.memory_address == 0x20000020
        and propagated_source.origin_source_pc == 0x080000F0
    )

    traced_source = tracer.trace_register_back("r3")
    trace_ok = (
        traced_source is not None
        and traced_source.memory_address == 0x20000020
        and traced_source.mmio_address is None
    )

    tracer._record_branch_dependency_snapshot(0x08001002)
    snapshots = tracer.branch_dependency_snapshots.get(0x08001004, [])
    snapshot_items = snapshots[-1]["register_sources"] if snapshots else {}
    snapshot_source = snapshot_items.get("r3")
    snapshot_ok = (
        snapshot_source is not None
        and snapshot_source.get("memory_address") == 0x20000020
        and snapshot_source.get("mmio_address") is None
        and snapshot_source.get("source_type") == "memory"
        and snapshot_source.get("origin_source_pc") == 0x080000F0
    )

    dependency_result = tracer.analyze_branch_dependencies(0x08001004)
    memory_dependencies = [
        item
        for item in dependency_result.get("dependencies", [])
        if item.get("type") == "memory"
    ]
    analyze_ok = any(
        int(item.get("address", 0)) == 0x20000020
        and int(item.get("source_pc", 0)) == 0x080000F0
        for item in memory_dependencies
    )
    provenance_result = tracer.analyze_branch_mmio_dependency(0x08001004)
    call_stack_ok = "call_stack_samples" in provenance_result

    call_static = {
        0x08003000: [
            {"address": 0x08003000, "mnemonic": "BL", "operands": "0x08004000", "size": 4},
            {"address": 0x08003004, "mnemonic": "CMP", "operands": "R0, #1", "size": 2},
            {"address": 0x08003006, "mnemonic": "BNE", "operands": "0x08003020", "size": 2},
        ]
    }
    call_tracer = RegisterTracer(DummyUC(), call_static)
    call_source = RegisterSource(
        register="r0",
        source_type="memory",
        source_pc=0x08002FF0,
        instruction="LDR R0, [R7,#0x8]",
        memory_address=0x20000088,
        operation="load",
        chain=("RAM[0x20000088]@0x08002ff0",),
        call_depth=0,
    )
    call_tracer.register_sources["r0"] = call_source
    call_tracer._record_call_frame(0x08003000, 4, call_static[0x08003000][0])
    call_tracer.register_sources.pop("r0", None)
    call_tracer._retire_call_frames(0x08003004)
    call_return_source = call_tracer.register_sources.get("r0")
    call_return_ok = (
        call_return_source is not None
        and call_return_source.memory_address == 0x20000088
        and call_return_source.operation == "call_return"
    )
    call_tracer._record_branch_dependency_snapshot(0x08003004)
    call_dependency = call_tracer.analyze_branch_dependencies(0x08003006)
    call_dependency_ok = any(
        item.get("register") == "r0" and item.get("type") == "memory" and int(item.get("address", 0)) == 0x20000088
        for item in call_dependency.get("dependencies", [])
    )

    composite_static = {
        0x08005000: [
            {"address": 0x08005000, "mnemonic": "ORRS", "operands": "R3, R1, R2", "size": 2},
            {"address": 0x08005002, "mnemonic": "TST", "operands": "R3, #0x3", "size": 2},
            {"address": 0x08005004, "mnemonic": "BNE", "operands": "0x08005020", "size": 2},
        ]
    }
    composite_tracer = RegisterTracer(DummyUC(), composite_static)
    composite_tracer.register_sources["r1"] = RegisterSource(
        register="r1",
        source_type="memory",
        source_pc=0x08004FF0,
        instruction="LDR R1, [R7,#0x10]",
        memory_address=0x20000100,
        operation="load",
        chain=("RAM[0x20000100]@0x08004ff0",),
        call_depth=0,
    )
    composite_tracer.register_sources["r2"] = RegisterSource(
        register="r2",
        source_type="mmio",
        source_pc=0x08004FF4,
        instruction="LDR R2, [R6,#0x4]",
        mmio_address=0x40021004,
        operation="load",
        chain=("MMIO[0x40021004]@0x08004ff4",),
        call_depth=0,
    )
    composite_tracer._propagate_register_sources(composite_static[0x08005000][0])
    composite_source = composite_tracer.register_sources.get("r3")
    composite_source_ok = (
        composite_source is not None
        and composite_source.source_type == "composite"
        and len(composite_source.dependency_sites) == 2
    )
    composite_tracer._record_branch_dependency_snapshot(0x08005002)
    composite_dependency = composite_tracer.analyze_branch_dependencies(0x08005004)
    composite_addresses = {
        (item.get("type"), int(item.get("address", 0)))
        for item in composite_dependency.get("dependencies", [])
    }
    composite_dependency_ok = {
        ("memory", 0x20000100),
        ("mmio", 0x40021004),
    }.issubset(composite_addresses)

    stack_static = {
        0x08006000: [
            {"address": 0x08006000, "mnemonic": "CMP", "operands": "R4, #0", "size": 2},
            {"address": 0x08006002, "mnemonic": "BNE", "operands": "0x08006020", "size": 2},
        ]
    }
    stack_tracer = RegisterTracer(DummyUC(), stack_static)
    stack_source = RegisterSource(
        register="r4",
        source_type="memory",
        source_pc=0x08005FF0,
        instruction="LDR R4, [SP,#0x18]",
        memory_address=0x20001018,
        memory_role="stack_frame",
        memory_base=0x20001000,
        memory_offset=0x18,
        memory_access_pc=0x08005FF0,
        memory_size=4,
        operation="load",
        chain=("RAM[sp+0x18]@0x08005ff0",),
        call_depth=2,
    )
    stack_tracer.register_sources["r4"] = stack_source
    stack_tracer._record_branch_dependency_snapshot(0x08006000)
    stack_dependency = stack_tracer.analyze_branch_dependencies(0x08006002)
    stack_dependency_ok = any(
        item.get("type") == "memory"
        and int(item.get("address", 0)) == 0x20001018
        and item.get("memory_role") == "stack_frame"
        and int(item.get("memory_offset", 0)) == 0x18
        for item in stack_dependency.get("dependencies", [])
    )

    recursive_tracer = RegisterTracer(DummyUC(), stack_static)
    recursive_tracer.register_sources["r0"] = RegisterSource(
        register="r0",
        source_type="mmio",
        source_pc=0x08006100,
        instruction="LDR R0, [R6,#0x0]",
        mmio_address=0x40023000,
        dependency_sites=(("mmio", 0x40023000, 0x08006100, 0x08006100),),
        operation="load",
        chain=("MMIO[0x40023000]@0x08006100",),
    )
    recursive_tracer.register_sources["r1"] = RegisterSource(
        register="r1",
        source_type="register",
        source_pc=0x08006102,
        instruction="MOV R1, R0",
        operation="MOV",
        chain=("MOV R1, R0",),
    )
    recursive_source = recursive_tracer.trace_register_back("r1")
    recursive_trace_ok = (
        recursive_source is not None
        and recursive_source.mmio_address == 0x40023000
    )

    byte_merge_tracer = RegisterTracer(DummyUC(), composite_static)
    byte_merge_tracer.memory_sources[0x20000300] = RegisterSource(
        register="r1",
        source_type="memory",
        source_pc=0x08007000,
        instruction="STRB R1, [R7,#0x0]",
        memory_address=0x20000200,
        dependency_sites=(("memory", 0x20000200, 0x08007000, 0x08007000),),
        operation="store",
        chain=("RAM[0x20000200]@0x08007000",),
    )
    byte_merge_tracer.memory_sources[0x20000301] = RegisterSource(
        register="r2",
        source_type="mmio",
        source_pc=0x08007004,
        instruction="LDRB R2, [R6,#0x0]",
        mmio_address=0x40021010,
        dependency_sites=(("mmio", 0x40021010, 0x08007004, 0x08007004),),
        operation="load",
        chain=("MMIO[0x40021010]@0x08007004",),
    )
    merged_source = byte_merge_tracer._get_memory_source(0x20000300, 2)
    merged_memory_ok = (
        merged_source is not None
        and merged_source.source_type == "composite"
        and {site[0:2] for site in merged_source.dependency_sites}
        >= {("memory", 0x20000200), ("mmio", 0x40021010)}
    )

    snapshot_limit_static = {
        0x08008000: [
            {"address": 0x08008000, "mnemonic": "CMP", "operands": "R1, #0", "size": 2},
            {"address": 0x08008002, "mnemonic": "BNE", "operands": "0x08008020", "size": 2},
        ]
    }
    snapshot_limit_tracer = RegisterTracer(DummyUC(), snapshot_limit_static)
    for index in range(24):
        snapshot_limit_tracer.register_sources["r1"] = RegisterSource(
            register="r1",
            source_type="mmio",
            source_pc=0x08008100 + index,
            instruction=f"LDR R1, [R0,#0x{index:x}]",
            mmio_address=0x40022000 + index * 4,
            dependency_sites=(("mmio", 0x40022000 + index * 4, 0x08008100 + index, 0x08008100 + index),),
            operation="load",
            chain=(f"MMIO[{index}]@0x08008100",),
        )
        snapshot_limit_tracer._record_branch_dependency_snapshot(0x08008000)
    adaptive_snapshots = snapshot_limit_tracer.branch_dependency_snapshots.get(0x08008002, [])
    adaptive_snapshot_ok = len(adaptive_snapshots) > 16

    dotted_static = {
        0x08002000: [
            {"address": 0x08002000, "mnemonic": "LDRSB.W", "operands": "R3, [R1,#0x0]", "size": 4},
            {"address": 0x08002004, "mnemonic": "CMP.W", "operands": "R3, #0xffffffff", "size": 4},
            {"address": 0x08002008, "mnemonic": "BEQ", "operands": "0x08002020", "size": 2},
        ]
    }
    dotted_tracer = RegisterTracer(DummyUC(), dotted_static)
    stale_source = RegisterSource(
        register="r3",
        source_type="memory",
        source_pc=0x08001000,
        instruction="LDR R3, [SP,#0x4]",
        memory_address=0x20000004,
        mmio_address=0x40021018,
        operation="load",
        chain=("MMIO[0x40021018]@0x08001000", "AND", "store", "load"),
        call_depth=1,
    )
    dotted_tracer.register_sources["r3"] = stale_source
    dotted_tracer._propagate_register_sources(dotted_static[0x08002000][0])
    dotted_cleared_source = dotted_tracer.register_sources.get("r3")
    dotted_trace_ok = dotted_cleared_source is None
    dotted_compare_ok = (
        dotted_tracer._find_compare_before_branch(0x08002008) is not None
        and dotted_tracer._build_compare_lookup().get(0x08002008, {}).get("address") == 0x08002004
    )

    result = {
        "recovered_ok": recovered_ok,
        "propagated_ok": propagated_ok,
        "trace_ok": trace_ok,
        "snapshot_ok": snapshot_ok,
        "analyze_ok": analyze_ok,
        "dotted_trace_ok": dotted_trace_ok,
        "dotted_compare_ok": dotted_compare_ok,
        "call_stack_ok": call_stack_ok,
        "call_return_ok": call_return_ok,
        "call_dependency_ok": call_dependency_ok,
        "composite_source_ok": composite_source_ok,
        "composite_dependency_ok": composite_dependency_ok,
        "stack_dependency_ok": stack_dependency_ok,
        "recursive_trace_ok": recursive_trace_ok,
        "merged_memory_ok": merged_memory_ok,
        "adaptive_snapshot_ok": adaptive_snapshot_ok,
        "dependency_result": dependency_result,
        "call_dependency": call_dependency,
        "composite_dependency": composite_dependency,
        "stack_dependency": stack_dependency,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all([
        recovered_ok,
        propagated_ok,
        trace_ok,
        snapshot_ok,
        analyze_ok,
        dotted_trace_ok,
        dotted_compare_ok,
        call_stack_ok,
        call_return_ok,
        call_dependency_ok,
        composite_source_ok,
        composite_dependency_ok,
        stack_dependency_ok,
        recursive_trace_ok,
        merged_memory_ok,
        adaptive_snapshot_ok,
    ]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
