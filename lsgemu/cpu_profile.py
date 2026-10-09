#!/usr/bin/env python3
"""Cortex-M CPU 型号自动匹配（unicorn ctl_set_cpu_model 精确切换）。

背景：IntelligentEmulator 的 MCLASS 模式（UC_MODE_MCLASS）此前把 CPU 写死
成 cortex-m33。虽然 M33 的 v8-M 主线编码兼容绝大多数 v7-M Thumb 代码，
但型号不匹配会带来语义漂移（如周期计数、个别系统寄存器行为）。实际
固件型号可从两路信号推断：

1. ELF ``e_flags``（EF_ARM_EABI_VER / EABI 浮点标志）——只能把候选约束到
   "M0+/M3 族（无 FPU）" 或 "M4/M7/M33 族（带 FPU）"，无法唯一确定型号；
2. 固件名中的 MCU 型号（STM32F4→M4、STM32H7→M7 等）——能唯一确定核心。

因此实际优先级为：文件名 MCU 表匹配（精确）> e_flags 族约束（取代表
型号）> 兜底 M33（当前行为）。两路信号矛盾时（如名称指向无 FPU 核但
e_flags 带 hard-float）保留名称结论并打 warning——名称是更强的信号。

浮点语义说明（覆盖率场景无实际影响，注释备查）：
- M4F = VFPv4-SP（fpv4-sp-d16），M33 = FPv5-SP（fpv5-sp-d16），
  FPv5-SP 是 VFPv4-SP 的超集且二进制编码兼容；
- 本项目实测 M33 模式可执行 M4F 固件的 SP 浮点指令（vmov.f32/vadd.f32），
  仅 f64 双精度缺失，而 M4F 固件本身不含 f64 指令；
- 因此 M4/M7/M33 之间切换对普通与 SP 浮点代码的覆盖率统计没有影响。

本模块只依赖标准库；unicorn 常量按需 lazy import（旧绑定可能缺失
UC_CPU_ARM_CORTEX_* 常量，缺失时 cpu_model_id 为 None，调用方回退默认）。

M0 语义提示为预防性告警：实测本机 unicorn 2.1.4（qemu cortex-m0）仍执行
部分 Thumb-2 编码（如 movw），v6-M 限制并非严格掩码；告警不改变执行行为，
只在真遇到 UC_ERR_INSN_INVALID 时提示排查方向。
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

FORCE_CPU_MODEL_ENV = "LSGEMU_FORCE_CPU_MODEL"

# MCLASS 模式下 unicorn 的默认 CPU（cpu.c: mode & UC_MODE_MCLASS
# -> UC_CPU_ARM_CORTEX_M33），即项目此前的写死行为，作为兜底。
CORTEX_M_FALLBACK_CONSTANT = "UC_CPU_ARM_CORTEX_M33"

# ---------------------------------------------------------------------------
# ELF e_flags 解析（ELF ARM ABI）
# ---------------------------------------------------------------------------
EF_ARM_EABI_MASK = 0xFF000000
EF_ARM_EABI_VER4 = 0x04000000
EF_ARM_EABI_VER5 = 0x05000000
# EABI5 浮点约定标志（与旧名 EF_ARM_SOFT_FLOAT/EF_ARM_VFP_FLOAT 同值）。
EF_ARM_ABI_FLOAT_SOFT = 0x00000200
EF_ARM_ABI_FLOAT_HARD = 0x00000400

# 可扩展的 MCU 型号 -> unicorn CPU 常量名映射表。
# 键为"去除非字母数字+大写"后的型号子串，匹配时按键长度降序尝试
# （长键优先，避免 STM32F4 抢先命中 STM32F4xx 之外的误配）。
# unicorn 2.1.4 的 arm_const 没有 UC_CPU_ARM_CORTEX_M0P，M0+ 映射到 M0
# （qemu cortex-m0 实现完整 v6-M ISA）。
MCU_NAME_TO_CPU: Dict[str, str] = {
    # ARMv7E-M / M4F 族
    "STM32F4": "UC_CPU_ARM_CORTEX_M4",
    "STM32F3": "UC_CPU_ARM_CORTEX_M4",
    "STM32L4": "UC_CPU_ARM_CORTEX_M4",
    "STM32G4": "UC_CPU_ARM_CORTEX_M4",
    "STM32WB": "UC_CPU_ARM_CORTEX_M4",
    "AT32F4": "UC_CPU_ARM_CORTEX_M4",
    "GD32F4": "UC_CPU_ARM_CORTEX_M4",
    "NRF52": "UC_CPU_ARM_CORTEX_M4",
    # ARMv7-M / M7 族
    "STM32H7": "UC_CPU_ARM_CORTEX_M7",
    "STM32F7": "UC_CPU_ARM_CORTEX_M7",
    "MIMXRT": "UC_CPU_ARM_CORTEX_M7",  # i.MX RT 系列
    "SAME7": "UC_CPU_ARM_CORTEX_M7",
    # ARMv7-M / M3 族
    "STM32F1": "UC_CPU_ARM_CORTEX_M3",
    "STM32F2": "UC_CPU_ARM_CORTEX_M3",
    "STM32L1": "UC_CPU_ARM_CORTEX_M3",
    # ARMv8-M / M33 族
    "STM32H5": "UC_CPU_ARM_CORTEX_M33",
    "STM32L5": "UC_CPU_ARM_CORTEX_M33",
    "STM32U5": "UC_CPU_ARM_CORTEX_M33",
    "RP2350": "UC_CPU_ARM_CORTEX_M33",
    "LPC55": "UC_CPU_ARM_CORTEX_M33",
    # ARMv6-M / M0(+)
    "STM32F0": "UC_CPU_ARM_CORTEX_M0",
    "STM32G0": "UC_CPU_ARM_CORTEX_M0",
    "STM32L0": "UC_CPU_ARM_CORTEX_M0",
    "NRF51": "UC_CPU_ARM_CORTEX_M0",
}

# 无 FPU 的核心（用于 e_flags 矛盾检查与 M0 语义提示）。
_CORTEX_NO_FPU_CONSTANTS = frozenset({
    "UC_CPU_ARM_CORTEX_M0",
    "UC_CPU_ARM_CORTEX_M0P",
    "UC_CPU_ARM_CORTEX_M3",
})
# ARMv6-M（M0/M0+）核心：无 Thumb-2、无硬件除法。
_CORTEX_V6M_CONSTANTS = frozenset({
    "UC_CPU_ARM_CORTEX_M0",
    "UC_CPU_ARM_CORTEX_M0P",
})


@dataclass(frozen=True)
class CpuProfile:
    """推断出的 Cortex-M CPU 型号。

    Attributes:
        unicorn_constant_name: unicorn CPU 常量名（如 ``UC_CPU_ARM_CORTEX_M4``）。
        cpu_model_id: 该常量在当前 unicorn 绑定中的数值（旧绑定缺失常量时
            为 None，调用方应回退 MCLASS 默认 cortex-m33）。
        source: ``"mcu_name"``（文件名 MCU 表命中）| ``"elf_flags"``
            （e_flags 族约束代表型号）| ``"fallback"``（兜底 M33）|
            ``"env_force"``（LSGEMU_FORCE_CPU_MODEL 强制）。
        reason: 人类可读的决策依据（含 e_flags 原始值，便于审计）。
    """

    unicorn_constant_name: str
    cpu_model_id: Optional[int]
    source: str
    reason: str


def unicorn_cpu_model_id(unicorn_constant_name: str) -> Optional[int]:
    """把常量名解析成当前 unicorn 绑定里的数值；绑定缺失该常量时返回 None。"""
    try:
        from unicorn import arm_const  # lazy：本模块不强制依赖 unicorn
    except Exception:
        return None
    value = getattr(arm_const, str(unicorn_constant_name), None)
    return int(value) if isinstance(value, int) else None


def pretty_cpu_name(unicorn_constant_name: str) -> str:
    """``UC_CPU_ARM_CORTEX_M4`` -> ``cortex-m4``。"""
    return str(unicorn_constant_name).replace("UC_CPU_ARM_", "").replace("_M", "-m").lower()


def cpu_profile_to_dict(profile: Optional[CpuProfile]) -> Optional[Dict[str, object]]:
    """报告用 JSON 视图；``None``（未启用 MCLASS/推断关闭）保持 null。"""
    if profile is None:
        return None
    return {
        "unicorn_constant_name": profile.unicorn_constant_name,
        "cpu_model": pretty_cpu_name(profile.unicorn_constant_name),
        "cpu_model_id": (
            int(profile.cpu_model_id) if profile.cpu_model_id is not None else None
        ),
        "cpu_model_id_hex": (
            f"{int(profile.cpu_model_id):#x}" if profile.cpu_model_id is not None else None
        ),
        "source": profile.source,
        "reason": profile.reason,
    }


def parse_elf_arm_flags(path: str | Path) -> Optional[Dict[str, object]]:
    """最小化解析 ELF 头，返回 ARM e_flags 视图；非 ELF/解析失败返回 None。

    只读 ELF 文件头（52/64 字节），不引入 pyelftools 依赖。
    """
    try:
        header = Path(path).expanduser().read_bytes()[:64]
    except OSError:
        return None
    if len(header) < 52 or header[:4] != b"\x7fELF":
        return None
    elf_class = header[4]
    endian = "<" if header[5] == 1 else ">"
    if elf_class == 1:
        # ELF32: e_entry@0x18(I32), e_flags@0x24(I32)
        e_entry = struct.unpack_from(endian + "I", header, 0x18)[0]
        e_flags = struct.unpack_from(endian + "I", header, 0x24)[0]
        bits = 32
    elif elf_class == 2:
        if len(header) < 64:
            return None
        # ELF64: e_entry@0x18(I64), e_flags@0x30(I32)
        e_entry = struct.unpack_from(endian + "Q", header, 0x18)[0]
        e_flags = struct.unpack_from(endian + "I", header, 0x30)[0]
        bits = 64
    else:
        return None
    eabi_version = (int(e_flags) & EF_ARM_EABI_MASK) >> 24
    if e_flags & EF_ARM_ABI_FLOAT_HARD:
        float_abi = "hard"
    elif e_flags & EF_ARM_ABI_FLOAT_SOFT:
        float_abi = "soft"
    else:
        float_abi = "none"
    return {
        "e_flags": int(e_flags),
        "e_flags_hex": f"{int(e_flags):#x}",
        "eabi_version": int(eabi_version),
        "float_abi": float_abi,
        "e_entry": int(e_entry),
        "elf_class": bits,
    }


def _normalize_mcu_name(value: str) -> str:
    return "".join(ch.upper() for ch in str(value) if ch.isalnum())


def _match_mcu_name(normalized_name: str) -> Optional[Tuple[str, str]]:
    """在去符号大写后的固件名里按"长键优先"查 MCU 表。"""
    for key in sorted(MCU_NAME_TO_CPU, key=len, reverse=True):
        if key in normalized_name:
            return key, MCU_NAME_TO_CPU[key]
    return None


def _describe_flags(flags_info: Optional[Dict[str, object]]) -> str:
    if not flags_info:
        return "e_flags 不可读（非 ELF 或头部解析失败）"
    return (
        f"e_flags={flags_info['e_flags_hex']}"
        f"(EABIv{flags_info['eabi_version']}, float_abi={flags_info['float_abi']})"
    )


def _warn_flags_conflict(
    profile_constant: str, flags_info: Optional[Dict[str, object]], name_source: str
) -> None:
    """名称结论与 e_flags 浮点约定矛盾时告警（保留名称结论）。"""
    if not flags_info or flags_info.get("float_abi") != "hard":
        return
    if profile_constant not in _CORTEX_NO_FPU_CONSTANTS:
        return
    logger.warning(
        "CPU 型号信号矛盾：%s 匹配到 %s（无 FPU），但 ELF e_flags=%s 标记 hard-float。"
        "保留名称结论——e_flags 只反映编译器的浮点调用约定，可能来自同名前缀的"
        "不同派生型号",
        name_source,
        pretty_cpu_name(profile_constant),
        flags_info.get("e_flags_hex"),
    )


def infer_cortex_m_cpu(
    elf_path: str | Path,
    firmware_name: Optional[str] = None,
) -> CpuProfile:
    """从 ELF e_flags 与固件名推断 Cortex-M CPU 型号。

    优先级：文件名 MCU 表（唯一确定型号）> e_flags 族约束（取代表型号）
    > 兜底 M33。详见模块 docstring 的信号强弱讨论。
    """
    path = Path(elf_path) if elf_path else None
    flags_info = parse_elf_arm_flags(path) if path is not None else None
    name_source = str(firmware_name) if firmware_name else (
        str(path.name) if path is not None else ""
    )
    normalized = _normalize_mcu_name(name_source)
    flags_desc = _describe_flags(flags_info)

    match = _match_mcu_name(normalized) if normalized else None
    if match is not None:
        key, constant = match
        _warn_flags_conflict(constant, flags_info, name_source)
        return CpuProfile(
            unicorn_constant_name=constant,
            cpu_model_id=unicorn_cpu_model_id(constant),
            source="mcu_name",
            reason=f"固件名 '{name_source}' 匹配 MCU 表键 '{key}'；{flags_desc}",
        )

    if flags_info is not None:
        eabi = int(flags_info["eabi_version"])
        float_abi = str(flags_info["float_abi"])
        if eabi >= 5 and float_abi == "hard":
            constant = "UC_CPU_ARM_CORTEX_M4"
            return CpuProfile(
                unicorn_constant_name=constant,
                cpu_model_id=unicorn_cpu_model_id(constant),
                source="elf_flags",
                reason=(
                    f"{flags_desc}：EABI5+hard-float 指向 M4/M7/M33 带核 FPU 族，"
                    "e_flags 无法进一步区分，取代表型号 M4"
                ),
            )
        if eabi >= 4:
            constant = "UC_CPU_ARM_CORTEX_M3"
            return CpuProfile(
                unicorn_constant_name=constant,
                cpu_model_id=unicorn_cpu_model_id(constant),
                source="elf_flags",
                reason=(
                    f"{flags_desc}：无 hard-float 标志指向 M0+/M3 无核 FPU 族，"
                    "e_flags 无法进一步区分，取代表型号 M3（v7-M 指令集覆盖 v6-M，"
                    "向上兼容多数 M0+ 镜像）"
                ),
            )

    return CpuProfile(
        unicorn_constant_name=CORTEX_M_FALLBACK_CONSTANT,
        cpu_model_id=unicorn_cpu_model_id(CORTEX_M_FALLBACK_CONSTANT),
        source="fallback",
        reason=f"固件名与 e_flags 均无法定位 MCU 型号（{flags_desc}），保持 MCLASS 默认 cortex-m33",
    )


def _normalize_forced_model_token(token: str) -> Optional[str]:
    """把 LSGEMU_FORCE_CPU_MODEL 的取值归一化为 unicorn 常量名。

    接受 ``cortex-m4`` / ``M4`` / ``UC_CPU_ARM_CORTEX_M4`` 三种写法；
    无法识别返回 None。
    """
    value = str(token).strip().upper().replace("-", "_")
    if not value:
        return None
    if value.startswith("UC_CPU_ARM_"):
        candidate = value
    else:
        candidate = f"UC_CPU_ARM_{value}"
    if not candidate.startswith("UC_CPU_ARM_CORTEX_"):
        candidate = candidate.replace("UC_CPU_ARM_", "UC_CPU_ARM_CORTEX_", 1)
    return candidate if candidate in set().union(
        set(MCU_NAME_TO_CPU.values()), {CORTEX_M_FALLBACK_CONSTANT, "UC_CPU_ARM_CORTEX_M0P"}
    ) else None


def resolve_cortex_m_cpu(
    elf_path: str | Path,
    firmware_name: Optional[str] = None,
) -> CpuProfile:
    """带环境变量覆盖的完整解析：LSGEMU_FORCE_CPU_MODEL 优先，其次推断。"""
    import os

    forced = os.environ.get(FORCE_CPU_MODEL_ENV, "").strip()
    if forced:
        constant = _normalize_forced_model_token(forced)
        if constant is not None:
            return CpuProfile(
                unicorn_constant_name=constant,
                cpu_model_id=unicorn_cpu_model_id(constant),
                source="env_force",
                reason=f"LSGEMU_FORCE_CPU_MODEL={forced} 显式指定",
            )
        logger.warning(
            "LSGEMU_FORCE_CPU_MODEL=%r 无法识别（期望 cortex-m4 / M7 / "
            "UC_CPU_ARM_CORTEX_M33 等写法），忽略并继续推断流程",
            forced,
        )
    return infer_cortex_m_cpu(elf_path, firmware_name=firmware_name)


def is_armv6m_profile(profile: CpuProfile) -> bool:
    """是否 ARMv6-M（M0/M0+）：无 Thumb-2、无硬件除法。"""
    return profile.unicorn_constant_name in _CORTEX_V6M_CONSTANTS


def count_thumb2_halfword_prefixes(path: str | Path) -> int:
    """统计疑似 32 位 Thumb-2 指令首半字的个数（M0 语义提示用，启发式）。

    32 位 Thumb 指令的首半字高 5 位为 0b11101/11110/11111
    （即 ``(hw & 0xF800) >= 0xE800``）。ELF 只扫描可执行 PT_LOAD 段；
    raw BIN 扫描整个文件。数据段中的偶发前缀会少量误报，因此本计数
    仅用于"推断为 M0 但固件疑似含 Thumb-2"的告警，不参与任何决策。
    """
    try:
        payload = Path(path).expanduser().read_bytes()
    except OSError:
        return 0
    windows: List[bytes] = [payload]
    if payload[:4] == b"\x7fELF" and len(payload) >= 52:
        endian = "<" if payload[5] == 1 else ">"
        elf_class = payload[4]
        if elf_class == 1:
            phoff = struct.unpack_from(endian + "I", payload, 0x1C)[0]
            phentsize = struct.unpack_from(endian + "H", payload, 0x2A)[0]
            phnum = struct.unpack_from(endian + "H", payload, 0x2C)[0]
            unpack_filesz = lambda offset: struct.unpack_from(endian + "I", payload, offset)[0]
            executable_flag = 0x1  # PF_X
        else:
            phoff = struct.unpack_from(endian + "Q", payload, 0x20)[0]
            phentsize = struct.unpack_from(endian + "H", payload, 0x36)[0]
            phnum = struct.unpack_from(endian + "H", payload, 0x38)[0]
            unpack_filesz = lambda offset: struct.unpack_from(endian + "Q", payload, offset)[0]
            executable_flag = 0x1
        windows = []
        for index in range(int(phnum)):
            entry_base = int(phoff) + index * int(phentsize)
            if entry_base + int(phentsize) > len(payload):
                break
            p_type = struct.unpack_from(endian + "I", payload, entry_base)[0]
            if p_type != 1:  # PT_LOAD
                continue
            if elf_class == 1:
                # ELF32 phdr: type,offset,vaddr,paddr,filesz,memsz,flags,align
                p_offset = struct.unpack_from(endian + "I", payload, entry_base + 4)[0]
                p_flags = struct.unpack_from(endian + "I", payload, entry_base + 24)[0]
                filesz_offset = entry_base + 16
            else:
                # ELF64 phdr: type,flags,offset,vaddr,paddr,filesz,memsz,align
                p_offset = struct.unpack_from(endian + "Q", payload, entry_base + 8)[0]
                p_flags = struct.unpack_from(endian + "I", payload, entry_base + 4)[0]
                filesz_offset = entry_base + 32
            if not (p_flags & executable_flag):
                continue
            start = max(0, int(p_offset))
            end = min(len(payload), start + int(unpack_filesz(filesz_offset)))
            if end > start:
                windows.append(payload[start:end])
    count = 0
    for window in windows:
        for offset in range(0, len(window) - 1, 2):
            if (struct.unpack_from("<H", window, offset)[0] & 0xF800) >= 0xE800:
                count += 1
    return count


__all__ = [
    "CORTEX_M_FALLBACK_CONSTANT",
    "CpuProfile",
    "FORCE_CPU_MODEL_ENV",
    "MCU_NAME_TO_CPU",
    "count_thumb2_halfword_prefixes",
    "cpu_profile_to_dict",
    "infer_cortex_m_cpu",
    "is_armv6m_profile",
    "parse_elf_arm_flags",
    "pretty_cpu_name",
    "resolve_cortex_m_cpu",
    "unicorn_cpu_model_id",
]
