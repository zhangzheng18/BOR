#!/usr/bin/env python3
"""angr 从入口可达的基本块（BB）基线。

背景：LSGEmu 覆盖率分母历史上取 Ghidra 全量静态 BB（含大量从入口不可达
的代码，例如未引用库函数/死代码/数据误判）。对无人机固件（如 ardupilot
STM32F427）该分母显著偏大，导致覆盖率被低估。本模块用 angr CFGEmulated
从 ELF 入口（e_entry，通常是 Reset_Handler）做符号化可达性分析，得到
"真实可达 BB 集合"作为可选分母：

1. ``compute_reachable_bbs`` —— 核心分析函数，返回 :class:`ReachabilityReport`。
2. ``python -m lsgemu.angr_reachability`` —— CLI，输出 JSON 报告与
   ``--bb-file-format`` 纯十六进制行文件（与数据集侧 valid_basic_blocks.txt
   同构，可直接被 ``_load_valid_bbs`` 风格的解析器读取）。
3. ``load_angr_reachable_bb_set`` —— 管线集成入口：
   ``LSGEMU_ANGR_REACHABILITY=1`` 开启后按 ``<stem>_angr_reachable.txt``
   文件名约定懒加载（见 PreparedFirmware.angr_reachable_bb_set）。

angr 依赖重（import 约 10 秒），全部函数内 lazy import，模块本身不拖累
现有测试启动。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .artifact_io import atomic_json_dump, atomic_write_text


logger = logging.getLogger(__name__)

REACHABILITY_SCHEMA = "lsgemu.angr_reachability.v1"

# CFGEmulated 需要在符号执行里逐状态展开，复杂度随固件体积超线性增长；
# 大于该阈值的 ELF 自动降级 CFGFast（线性扫描+数据流，快但可能过近似，
# report 的 method/degrade_reason 字段会如实标注）。
CFG_EMULATED_SIZE_LIMIT = 5 * 1024 * 1024

# 管线集成约定（PreparedFirmware 懒加载）。
ANGR_REACHABILITY_ENV = "LSGEMU_ANGR_REACHABILITY"
ANGR_REACHABLE_FILE_ENV = "LSGEMU_ANGR_REACHABLE_FILE"
ANGR_REACHABLE_SUFFIX = "_angr_reachable.txt"

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


class AngrReachabilityError(RuntimeError):
    """angr 可达性分析无法完成（angr 缺失/输入非法/全部方法失败）。"""


class AngrReachabilityTimeout(RuntimeError):
    """SIGALRM 兜底超时（防 CFGEmulated 死循环）。"""


@dataclass(frozen=True)
class ReachabilityReport:
    """angr 从入口可达 BB 基线。

    Attributes:
        elf_path: 被分析的 ELF（解析后的绝对路径）。
        entry: 分析入口（默认 e_entry；显式传入时为传入值），32 位截断。
        bb_addrs: 图可达 BB 起始地址（升序、去重、32 位、排除 simprocedure）。
        node_count: CFG 图节点总数（含 simprocedure 节点，供对照）。
        method: ``"cfg_emulated"`` 或 ``"cfg_fast"``（大固件/超时降级）。
        degrade_reason: ``None`` 或降级原因（elf 大小超限 / emulated 超时等）。
        elapsed_seconds: 端到端耗时（含 angr.Project 构造）。
        angr_version: 参与分析的 angr 版本号。
    """

    elf_path: str
    entry: int
    bb_addrs: List[int]
    node_count: int
    elapsed_seconds: float
    entry_source: str = "e_entry"
    method: str = "cfg_emulated"
    degrade_reason: Optional[str] = None
    angr_version: str = "unknown"

    @property
    def bb_count(self) -> int:
        return len(self.bb_addrs)


@contextmanager
def _alarm_timeout(seconds: float):
    """SIGALRM 兜底超时；仅在主线程生效（非主线程无法安装信号处理器）。

    angr 分析是纯 Python 循环，信号处理器抛出的异常会在字节码边界打断
    死循环；不需要杀进程。
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - 触发路径由真实固件覆盖
        raise AngrReachabilityTimeout(
            f"angr analysis exceeded {seconds:g}s wall-clock budget"
        )

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _collect_bb_addrs(cfg) -> Tuple[List[int], int]:
    """从 CFG 图提取 BB 起始地址（排除 simprocedure）与节点总数。

    地址做 32 位截断并清除 Thumb 位（angr 图节点偶有带 bit0 的跳转目标；
    LSGEmu 的静态/动态 BB 地址口径均为偶数指令地址）。
    """
    bb_addrs = {
        int(node.addr) & 0xFFFFFFFE
        for node in cfg.graph.nodes
        if not bool(getattr(node, "is_simprocedure", False))
    }
    return sorted(bb_addrs), int(cfg.graph.number_of_nodes())


def compute_reachable_bbs(
    elf_path: str | Path,
    entry: Optional[int] = None,
    timeout_seconds: float = 300,
    max_steps: int = 50000,
    force_cfg_fast: bool = False,
) -> ReachabilityReport:
    """计算从 ELF 入口可达的 BB 集合。

    Args:
        elf_path: ARM/Cortex-M 固件 ELF。
        entry: 分析入口；``None`` 用 angr ``project.entry``（即 e_entry）。
            显式传入时按 32 位截断（Thumb 位原样保留，angr 自行处理）。
        timeout_seconds: CFG 分析壁钟预算（SIGALRM 兜底，仅主线程生效）。
        max_steps: CFGEmulated 单一起始状态的最大步数（防路径爆炸）。
        force_cfg_fast: 无条件使用 CFGFast（诊断用）。

    Returns:
        ReachabilityReport，见 dataclass 文档。

    Raises:
        AngrReachabilityError: 输入不存在 / angr 不可用 / 两种 CFG 方法均失败。
    """
    started = time.monotonic()
    try:
        import angr  # lazy：angr import 慢且依赖重，不拖累模块导入方
    except Exception as exc:  # pragma: no cover - 环境缺失分支
        raise AngrReachabilityError(f"angr is not importable: {exc}") from exc

    path = Path(elf_path).expanduser().resolve()
    if not path.is_file():
        raise AngrReachabilityError(f"ELF input does not exist or is not a file: {path}")

    elf_size = path.stat().st_size
    method = "cfg_emulated"
    degrade_reason: Optional[str] = None
    if force_cfg_fast:
        method = "cfg_fast"
    elif elf_size > CFG_EMULATED_SIZE_LIMIT:
        method = "cfg_fast"
        degrade_reason = f"elf_size_{elf_size}_gt_{CFG_EMULATED_SIZE_LIMIT}"

    # BintoElf 生成的 ELF 可能缺 v7-M e_flags，angr 会误选 A-profile ArchARMEL，
    # 导致 arm_spotter 在 MSR/MRS(Cortex-M) 时查 primask 寄存器报
    # "Register primask does not exist!"。对 Cortex-M 固件强制 cortexm arch。
    project = angr.Project(str(path), auto_load_libs=False, arch="cortexm")
    if entry is not None:
        entry_addr = int(entry) & 0xFFFFFFFF
        entry_source = "explicit"
    else:
        entry_addr = int(project.entry) & 0xFFFFFFFF
        entry_source = "e_entry"

    cfg = None
    if method == "cfg_emulated":
        # resolve_indirect_jumps=False：裸机固件的函数指针/跳转表缺少重定位
        # 与堆布局信息，angr 的间接跳转解析器会产出 None 目标并在建图时抛
        # "ValueError: None cannot be a node"（Heat_Press.elf 实测）。关闭后
        # 只沿直接调用/分支展开——对"从入口可达基线"是保守欠近似（宁可少
        # 计也不误报），且实测 F427 从 reset 恰好得到先前人工测量的 639 BB。
        emulated_kwargs: Dict[str, object] = {
            "normalize": True,
            "max_steps": max(1, int(max_steps)),
            "resolve_indirect_jumps": False,
        }
        if entry is not None:
            emulated_kwargs["starts"] = [entry_addr]
        try:
            with _alarm_timeout(timeout_seconds):
                cfg = project.analyses.CFGEmulated(**emulated_kwargs)
        except AngrReachabilityTimeout as exc:
            method = "cfg_fast"
            degrade_reason = f"cfg_emulated_timeout_after_{timeout_seconds:g}s ({exc})"
        except Exception as exc:
            # CFGEmulated 对裸机固件的个别指令/重定位敏感；失败时降级
            # CFGFast 而不是让调用方拿不到基线。
            method = "cfg_fast"
            degrade_reason = f"cfg_emulated_failed: {type(exc).__name__}: {exc}"
    if cfg is None:
        try:
            with _alarm_timeout(timeout_seconds):
                cfg = project.analyses.CFGFast(normalize=True)
        except AngrReachabilityTimeout as exc:
            raise AngrReachabilityError(
                f"both CFG methods exceeded {timeout_seconds:g}s for {path}"
            ) from exc

    bb_addrs, node_count = _collect_bb_addrs(cfg)
    return ReachabilityReport(
        elf_path=str(path),
        entry=entry_addr,
        bb_addrs=bb_addrs,
        node_count=node_count,
        elapsed_seconds=time.monotonic() - started,
        entry_source=entry_source,
        method=method,
        degrade_reason=degrade_reason,
        angr_version=str(getattr(angr, "__version__", "unknown")),
    )


def report_to_dict(report: ReachabilityReport) -> Dict[str, object]:
    """JSON 可序列化视图（BB 地址为 0x%08x 十六进制字符串列表）。"""
    return {
        "schema": REACHABILITY_SCHEMA,
        "elf_path": report.elf_path,
        "entry": int(report.entry),
        "entry_hex": f"0x{int(report.entry) & 0xFFFFFFFF:08x}",
        "entry_source": report.entry_source,
        "method": report.method,
        "degrade_reason": report.degrade_reason,
        "angr_version": report.angr_version,
        "node_count": int(report.node_count),
        "bb_count": int(report.bb_count),
        "elapsed_seconds": float(report.elapsed_seconds),
        "bb_addrs": [f"0x{addr:08x}" for addr in report.bb_addrs],
    }


def write_bb_file(report: ReachabilityReport, output_path: str | Path) -> Path:
    """写出与 valid_basic_blocks.txt 同构的纯十六进制行文件。

    每行一个 BB 起始地址（小写、无 0x 前缀、8 位十六进制），与数据集侧
    ``_load_valid_bbs``（``int(line, 16)``）的解析口径一致。
    """
    lines = [f"{addr:08x}" for addr in report.bb_addrs]
    output = Path(output_path).expanduser().resolve()
    atomic_write_text(output, "\n".join(lines) + ("\n" if lines else ""))
    return output


def angr_reachable_bb_file_candidates(firmware_path: str | Path) -> List[Path]:
    """可达 BB 文件的候选路径（按优先级排序）。

    1. ``LSGEMU_ANGR_REACHABLE_FILE`` 显式指定；
    2. 与被分析固件同目录的 ``<stem>_angr_reachable.txt``（约定名）。
    """
    firmware = Path(firmware_path)
    candidates: List[Path] = []
    explicit = os.environ.get(ANGR_REACHABLE_FILE_ENV)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(firmware.parent / f"{firmware.stem}{ANGR_REACHABLE_SUFFIX}")
    return candidates


def angr_reachability_enabled() -> bool:
    """``LSGEMU_ANGR_REACHABILITY=1`` 环境开关是否开启。"""
    value = os.environ.get(ANGR_REACHABILITY_ENV) or ""
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def parse_reachable_bb_lines(text: str) -> Set[int]:
    """解析纯十六进制行文本（兼容 0x 前缀与空行），返回 32 位对齐地址集合。"""
    addrs: Set[int] = set()
    for line in text.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            addrs.add(int(token, 16) & 0xFFFFFFFF)
        except ValueError:
            logger.debug("跳过无法解析的可达 BB 行: %r", token)
    return addrs


def load_angr_reachable_bb_set(
    firmware_path: str | Path,
) -> Tuple[Optional[Path], Optional[Set[int]]]:
    """管线集成入口：开关开启且文件存在时加载可达 BB 集合。

    Returns:
        (文件路径, 地址集合)；开关关闭或文件缺失时为 ``(None, None)``，
        报告字段据此输出 null（向后兼容）。
    """
    if not angr_reachability_enabled():
        return None, None
    for candidate in angr_reachable_bb_file_candidates(firmware_path):
        if not candidate.is_file():
            continue
        try:
            addrs = parse_reachable_bb_lines(candidate.read_text())
        except OSError as exc:
            logger.warning("读取 angr 可达 BB 文件失败 %s: %s", candidate, exc)
            continue
        if not addrs:
            logger.warning("angr 可达 BB 文件为空，忽略: %s", candidate)
            continue
        return candidate, addrs
    logger.warning(
        "r40 P4 降级：LSGEMU_ANGR_REACHABILITY=1 但未找到 %s（候选: %s）——"
        "denominator=unavailable，reachable_* 报告字段将为 null（不许静默冒充数字）",
        ANGR_REACHABLE_SUFFIX,
        ", ".join(str(p) for p in angr_reachable_bb_file_candidates(firmware_path)),
    )
    return None, None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lsgemu.angr_reachability",
        description=(
            "用 angr 计算从 ELF 入口可达的 BB 基线；"
            "输出 JSON 报告，并可选写出与 valid_basic_blocks.txt 同构的十六进制行文件。"
        ),
    )
    parser.add_argument("elf", help="ARM/Cortex-M 固件 ELF 路径")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON 报告输出路径（缺省打印到 stdout 摘要，不写文件）",
    )
    parser.add_argument(
        "--bb-file-format",
        default=None,
        metavar="PATH",
        help="纯十六进制行 BB 文件输出路径（与 valid_basic_blocks.txt 同构，供 LSGEMU_ANGR_REACHABILITY 管线加载）",
    )
    parser.add_argument(
        "--entry",
        type=lambda value: int(value, 0),
        default=None,
        help="分析入口（默认 ELF e_entry），如 0x08004fb5",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="CFG 分析壁钟预算秒数（默认 300，SIGALRM 兜底）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="CFGEmulated 单一起始状态最大步数（默认 50000）",
    )
    parser.add_argument(
        "--force-cfg-fast",
        action="store_true",
        help="跳过 CFGEmulated 直接 CFGFast（诊断用）",
    )
    args = parser.parse_args(argv)

    report = compute_reachable_bbs(
        args.elf,
        entry=args.entry,
        timeout_seconds=args.timeout,
        max_steps=args.max_steps,
        force_cfg_fast=bool(args.force_cfg_fast),
    )

    if args.bb_file_format:
        bb_path = write_bb_file(report, args.bb_file_format)
        print(f"[angr_reachability] BB 文件: {bb_path} ({report.bb_count} 行)")
    payload = report_to_dict(report)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        atomic_json_dump(payload, output_path, indent=2)
        print(f"[angr_reachability] JSON 报告: {output_path}")

    print(f"[angr_reachability] ELF: {report.elf_path}")
    print(
        f"[angr_reachability] 入口: 0x{report.entry:08x} ({report.entry_source}) | "
        f"方法: {report.method} | angr {report.angr_version}"
    )
    if report.degrade_reason:
        print(f"[angr_reachability] 降级原因: {report.degrade_reason}")
    print(
        f"[angr_reachability] 可达 BB: {report.bb_count} "
        f"(CFG 节点总数 {report.node_count}) | 耗时 {report.elapsed_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
