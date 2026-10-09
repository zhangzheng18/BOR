#!/usr/bin/env python3
"""
LLM代码分析器

用于分析最近的BB指令序列，识别导致死循环的根本原因

核心功能：
1. 分析最近N个BB的指令序列
2. 识别错误处理路径（Error_Handler等）
3. 识别关键的分支判断
4. 推断需要模拟的内存地址和值
5. 生成新的约束配置
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re
from bisect import bisect_right

try:
    from ..runtime_bootstrap import bootstrap_runtime_dependencies
    from ..llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        create_openai_compatible_client,
        extract_response_text,
        parse_json_object,
    )
except ImportError:
    from lsgemu.runtime_bootstrap import bootstrap_runtime_dependencies
    from lsgemu.llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        create_openai_compatible_client,
        extract_response_text,
        parse_json_object,
    )

bootstrap_runtime_dependencies()

logger = logging.getLogger(__name__)


@dataclass
class CodeAnalysisResult:
    """
    代码分析结果
    """
    # 是否需要回退
    need_rollback: bool = False
    rollback_levels: int = 0  # 需要回退几级

    # 识别出的问题
    problem_type: str = ""  # "error_handler" / "null_pointer" / "uninitialized_memory" / "branch_condition"
    problem_description: str = ""

    # 关键指令位置
    critical_pc: Optional[int] = None
    critical_instruction: Optional[str] = None

    # 推断的约束
    suggested_constraints: List[Dict] = None  # [{"type": "memory", "address": 0x20008120, "value": 0x20008120}, ...]

    # 置信度
    confidence: float = 0.0

    # 是否已经确认这是终止错误路径
    terminal_path: bool = False

    # 用于运行时记住这次尝试的是哪个控制分支、哪个方向
    control_branch_pc: Optional[int] = None
    desired_branch_taken: Optional[bool] = None

    # 如唯一BB快照不足，可直接恢复到可安全重放的分支BB
    restore_branch_bb: Optional[int] = None

    # 分析理由
    reason: str = ""

    def __post_init__(self):
        if self.suggested_constraints is None:
            self.suggested_constraints = []

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "need_rollback": self.need_rollback,
            "rollback_levels": self.rollback_levels,
            "problem_type": self.problem_type,
            "problem_description": self.problem_description,
            "critical_pc": f"0x{self.critical_pc:08x}" if self.critical_pc else None,
            "critical_instruction": self.critical_instruction,
            "suggested_constraints": self.suggested_constraints,
            "confidence": self.confidence,
            "terminal_path": self.terminal_path,
            "control_branch_pc": f"0x{self.control_branch_pc:08x}" if self.control_branch_pc is not None else None,
            "desired_branch_taken": self.desired_branch_taken,
            "restore_branch_bb": f"0x{self.restore_branch_bb:08x}" if self.restore_branch_bb is not None else None,
            "reason": self.reason
        }


class LLMCodeAnalyzer:
    """
    LLM代码分析器

    分析最近的BB指令序列，识别问题并生成约束
    """

    _missing_config_warned: Set[str] = set()
    _empty_config_warned: bool = False
    _openai_missing_warned: bool = False
    _client_init_failure_warned: Set[str] = set()

    def __init__(
        self,
        config_path='LLM.yaml',
        static_bbs: Optional[Dict[int, List[Dict]]] = None,
        thumb_mode: Optional[bool] = None,
    ):
        """
        初始化

        Args:
            config_path: LLM配置文件路径
        """
        self.config = self._load_config(config_path)
        self.client = None
        self.static_bbs = static_bbs or {}
        self.thumb_mode = thumb_mode
        self.instruction_index = self._build_instruction_index()
        self.instruction_to_bb = self._build_instruction_to_bb()
        self.target_reference_index = self._build_target_reference_index()
        self.sorted_bb_addresses = sorted(self.static_bbs)
        self.predecessor_index = self._build_predecessor_index()
        self._init_client()
        self.analysis_cache: Dict[Tuple[object, ...], CodeAnalysisResult] = {}
        try:
            self.max_llm_calls_per_loop = max(
                0,
                int(os.environ.get("LSGEMU_MAX_LLM_CALLS_PER_LOOP", "1")),
            )
        except ValueError:
            self.max_llm_calls_per_loop = 1
        self.llm_call_counts_by_loop: Dict[int, int] = {}
        self.fast_loop_analysis = (
            os.environ.get("LSGEMU_FAST_LOOP_ANALYSIS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # 统计
        self.total_analyses = 0
        self.successful_analyses = 0

    def _analysis_cache_key(self, snapshots: List, current_bb: int) -> Tuple[object, ...]:
        tail_bbs: List[int] = []
        for snapshot in snapshots[-4:]:
            try:
                tail_bbs.append(int(getattr(snapshot, "bb_address", 0) or 0) & 0xFFFFFFFF)
            except Exception:
                tail_bbs.append(0)
        current_snapshot = snapshots[-1] if snapshots else None
        mmio_signature: List[Tuple[int, int, bool]] = []
        for item in list(getattr(current_snapshot, "mmio_access_history", []) or [])[-4:]:
            try:
                pc, addr, is_read, _value = item
                mmio_signature.append((int(pc) & 0xFFFFFFFF, int(addr) & 0xFFFFFFFF, bool(is_read)))
            except Exception:
                continue
        return (
            int(current_bb) & 0xFFFFFFFF,
            tuple(tail_bbs),
            tuple(mmio_signature),
            self._describe_current_pattern(current_snapshot, current_bb),
        )

    def _build_instruction_index(self) -> Dict[int, Dict]:
        lookup: Dict[int, Dict] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                address = self._parse_hex(insn.get("address"))
                if address is not None:
                    lookup[address] = insn
        return lookup

    def _build_instruction_to_bb(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                address = self._parse_hex(insn.get("address"))
                if address is not None:
                    lookup[address] = bb_addr
        return lookup

    def add_dynamic_basic_block(self, bb_addr: int, instructions: List[Dict]) -> None:
        """Expose runtime-decoded code to local constraint validation helpers."""
        if not instructions:
            return
        bb_addr = int(bb_addr) & ~1
        self.static_bbs[bb_addr] = instructions
        for insn in instructions:
            address = self._parse_hex(insn.get("address"))
            if address is None:
                continue
            self.instruction_index[address] = insn
            self.instruction_to_bb[address] = bb_addr
        if bb_addr not in self.sorted_bb_addresses:
            self.sorted_bb_addresses.append(bb_addr)
            self.sorted_bb_addresses.sort()
        self.analysis_cache.clear()

    def _build_target_reference_index(self) -> Dict[int, List[Dict[str, object]]]:
        refs: Dict[int, List[Dict[str, object]]] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
                if not (
                    mnemonic.startswith("B")
                    or mnemonic in {"CBZ", "CBNZ", "TBB", "TBH"}
                ):
                    continue
                target = self._parse_branch_target(str(insn.get("operands", "")))
                if target is None:
                    continue
                target_bb = self._resolve_bb_start(target)
                if target_bb is None:
                    continue
                refs.setdefault(target_bb, []).append({
                    "bb": bb_addr,
                    "address": self._parse_hex(insn.get("address")) or bb_addr,
                    "mnemonic": mnemonic,
                    "operands": str(insn.get("operands", "")),
                })
        return refs

    def _build_predecessor_index(self) -> Dict[int, List[Dict[str, object]]]:
        predecessors: Dict[int, List[Dict[str, object]]] = {}

        for target_bb, refs in self.target_reference_index.items():
            predecessors.setdefault(target_bb, []).extend({
                **ref,
                "kind": "explicit",
            } for ref in refs)

        for bb_addr, instructions in self.static_bbs.items():
            if not instructions:
                continue
            last_insn = instructions[-1]
            mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic"))
            if not self._bb_can_fallthrough(mnemonic):
                continue
            next_bb = self._next_sequential_bb(bb_addr)
            if next_bb is None:
                continue
            predecessors.setdefault(next_bb, []).append({
                "bb": bb_addr,
                "address": self._parse_hex(last_insn.get("address")) or bb_addr,
                "mnemonic": mnemonic,
                "operands": str(last_insn.get("operands", "")),
                "kind": "fallthrough",
            })

        return predecessors

    def _load_config(self, config_path: str) -> Dict:
        """加载LLM配置"""
        if not config_path:
            return {}
        try:
            # 支持 ~ 与 $VAR 展开；优先使用 resolve 后的绝对路径。
            # campaign 子进程的 cwd 不是仓库根，相对路径必须额外按仓库根
            # （lsgemu/analysis/ 向上两级）兜底，否则永远找不到配置文件。
            expanded = os.path.expandvars(os.path.expanduser(str(config_path)))
            repo_root = Path(__file__).resolve().parents[2]
            candidates = [Path(expanded)]
            if not candidates[0].is_absolute():
                candidates.append(repo_root / expanded)
                candidates.append(repo_root.parent / expanded)
            possible_paths: List[str] = []
            seen_paths: Set[str] = set()
            for candidate in candidates:
                resolved = str(candidate.resolve())
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    possible_paths.append(resolved)

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path) as f:
                        config = yaml.safe_load(f)
                        logger.info(f"加载LLM配置: {path}")
                        return config.get('llm', {})

            config_key = str(config_path)
            if config_key not in self._missing_config_warned:
                logger.error(
                    "配置文件不存在: %s (尝试过的路径: %s)",
                    config_path,
                    ", ".join(possible_paths),
                )
                self._missing_config_warned.add(config_key)
            return {}

        except Exception as e:
            config_key = f"{config_path}:{e}"
            if config_key not in self._client_init_failure_warned:
                logger.error(f"加载配置失败: {e}")
                self._client_init_failure_warned.add(config_key)
            return {}

    def _init_client(self):
        """初始化LLM客户端"""
        if not self.config:
            if not type(self)._empty_config_warned:
                logger.warning("LLM配置为空，使用启发式模式")
                type(self)._empty_config_warned = True
            return

        client, model, backend = create_openai_compatible_client(
            self.config,
            logger=logger,
            warn_key="code_analyzer",
            default_model="qwen-plus",
        )
        self.client = client
        self.llm_model = model
        if client is not None:
            logger.info(f"LLM客户端初始化成功: {self.llm_model} backend={backend}")

    def analyze_deadlock(self,
                        recent_snapshots: List,
                        current_bb: int,
                        loop_count: int,
                        branch_snapshots: Optional[Dict[int, object]] = None,
                        excluded_branch_directions: Optional[Dict[int, set]] = None) -> CodeAnalysisResult:
        """
        分析死循环

        Args:
            recent_snapshots: 最近的快照列表（包含BB指令序列）
            current_bb: 当前卡死的BB地址
            loop_count: 循环次数

        Returns:
            分析结果
        """
        self.total_analyses += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"LLM代码分析")
        logger.info(f"{'='*80}")
        logger.info(f"当前BB: 0x{current_bb:08x}")
        logger.info(f"循环次数: {loop_count}")
        logger.info(f"可用快照: {len(recent_snapshots)}")

        snapshots = self._normalize_snapshot_order(recent_snapshots)
        cache_key = self._analysis_cache_key(snapshots, current_bb)
        cached = self.analysis_cache.get(cache_key)
        if cached is not None:
            logger.info(
                "复用循环分析缓存: current_bb=0x%08x confidence=%.2f type=%s",
                current_bb,
                cached.confidence,
                cached.problem_type,
            )
            return cached

        # 先跑本地启发式。常见的 Error_Handler / B self 不需要付费调用 LLM，
        # replay 阶段会反复遇到这些模式，先本地解决能显著降低路径探索开销。
        result = self._heuristic_analyze(
            snapshots,
            current_bb,
            loop_count,
            branch_snapshots=branch_snapshots,
            excluded_branch_directions=excluded_branch_directions,
        )

        # 启发式不确定时，LLM才作为保底。
        loop_llm_calls = self.llm_call_counts_by_loop.get(int(current_bb), 0)
        may_call_llm = (
            bool(self.client)
            and self.max_llm_calls_per_loop > 0
            and loop_llm_calls < self.max_llm_calls_per_loop
        )
        if self.fast_loop_analysis and result.confidence <= 0.3 and loop_llm_calls > 0:
            may_call_llm = False
        if result.confidence < 0.6 and may_call_llm:
            self.llm_call_counts_by_loop[int(current_bb)] = loop_llm_calls + 1
            llm_result = self._call_llm_api(snapshots, current_bb, loop_count)
            if llm_result.confidence >= result.confidence:
                result = llm_result
            if result.confidence < 0.3:
                logger.warning("LLM分析失败或置信度太低，保留启发式分析结果")
        elif result.confidence < 0.6 and self.client:
            logger.info(
                "跳过重复LLM循环分析: current_bb=0x%08x calls=%d limit=%d confidence=%.2f",
                current_bb,
                loop_llm_calls,
                self.max_llm_calls_per_loop,
                result.confidence,
            )

        if result.confidence >= 0.6:
            self.successful_analyses += 1
            logger.info(f"✅ 分析成功: {result.problem_type}")
            logger.info(f"   置信度: {result.confidence:.2f}")
            logger.info(f"   建议: {result.problem_description}")
        else:
            logger.warning(f"⚠️ 分析置信度较低: {result.confidence:.2f}")

        self.analysis_cache[cache_key] = result
        return result

    def _call_llm_api(self, recent_snapshots: List, current_bb: int, loop_count: int) -> CodeAnalysisResult:
        """
        调用LLM API进行分析

        Args:
            recent_snapshots: 最近的快照列表
            current_bb: 当前BB地址
            loop_count: 循环次数

        Returns:
            分析结果
        """
        try:
            # 构建prompt
            prompt = self._build_analysis_prompt(recent_snapshots, current_bb, loop_count)

            response = self._call_llm_json(
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # 解析响应
            result = self._parse_llm_response(response)
            return self._post_validate_result(result, recent_snapshots, current_bb)

        except Exception as e:
            logger.warning(f"LLM API调用失败: {e}")
            return self._heuristic_analyze(recent_snapshots, current_bb, loop_count)

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """You are an expert ARM Cortex-M firmware deadlock analyst.

Your job is to diagnose why execution got stuck and propose the smallest concrete fix.

        Important rules:
1. Distinguish fatal sinks from wait loops.
   - Fatal sink: current BB is usually a single unconditional branch to itself.
   - Wait loop: current BB usually loads memory/MMIO, compares/tests it, then branches back to itself.
2. Do not focus only on the sink block. Identify the branch/load/test that actually controls progress.
3. If the current BB is a sink like _Error_Handler/_exit, never put the sink PC itself into read_pc.
4. read_pc must be the exact PC of a real load instruction (LDR/LDRB/LDRH/...) that reads the constrained memory/MMIO value.
5. For self-loops, your goal is to make execution LEAVE the loop by making the back-edge branch NOT taken.
   - If the loop-back branch is BEQ, the compared values must become NOT equal.
   - If the loop-back branch is BNE, the compared values must become equal.
6. Only suggest constraints when you can name a concrete address and value that will change control flow.
7. All addresses and values in the JSON must be strings like \"0x08001234\" or \"0x00000001\".
8. Return exactly one JSON object. No markdown, no code fence, no extra prose.

Required JSON schema:
{
  "need_rollback": true,
  "rollback_levels": 1,
  "problem_type": "error_handler|deadlock|polling|branch_condition|uninitialized_memory|unknown",
  "problem_description": "short summary",
  "critical_pc": "0x00000000",
  "critical_instruction": "instruction text",
  "suggested_constraints": [
    {
      "type": "memory|mmio",
      "read_pc": "0x00000000",
      "address": "0x00000000",
      "value": "0x00000000",
      "constraint_pc": "0x00000000",
      "description": "why this value exits the loop"
    }
  ],
  "confidence": 0.0,
  "terminal_path": false,
  "reason": "detailed reasoning"
}"""

    def _build_analysis_prompt(self, recent_snapshots: List, current_bb: int, loop_count: int) -> str:
        """
        构建分析prompt

        Args:
            recent_snapshots: 最近的快照列表
            current_bb: 当前BB地址
            loop_count: 循环次数

        Returns:
            prompt字符串
        """
        prompt_parts = []
        snapshots = self._normalize_snapshot_order(recent_snapshots)
        current_snapshot = snapshots[-1] if snapshots else None
        pattern_summary = self._describe_current_pattern(current_snapshot, current_bb)
        static_refs = self._describe_static_references(current_bb)
        heuristic_hint = self._build_heuristic_hint(current_snapshot)

        prompt_parts.append("## Deadlock Information")
        prompt_parts.append(f"Current BB: 0x{current_bb:08x}")
        prompt_parts.append(f"Loop count: {loop_count}")
        prompt_parts.append(f"Available snapshots: {len(snapshots)}")
        prompt_parts.append(f"Current pattern: {pattern_summary}")
        if static_refs:
            prompt_parts.append("Static incoming control-flow references:")
            prompt_parts.extend(f"- {item}" for item in static_refs)
        static_contexts = self._describe_static_predecessor_contexts(current_bb)
        if static_contexts:
            prompt_parts.append("Static predecessor contexts around the sink/loop:")
            prompt_parts.extend(static_contexts)
        if heuristic_hint:
            prompt_parts.append(f"Local heuristic hint: {heuristic_hint}")
        prompt_parts.append("")

        prompt_parts.append("## Recent Basic Blocks (from oldest to newest)")
        prompt_parts.append("")

        for i, snapshot in enumerate(snapshots):
            prompt_parts.append(f"### BB #{i+1}: 0x{snapshot.bb_address:08x}")
            prompt_parts.append(f"Instruction count: {snapshot.instruction_count}")
            prompt_parts.append(f"Entry PC: 0x{snapshot.pc:08x}")

            # 显示寄存器状态
            if snapshot.cpu_state:
                prompt_parts.append("Key registers:")
                for reg, val in self._important_registers(snapshot.cpu_state):
                    prompt_parts.append(f"  {reg} = 0x{val:08x}")

            if snapshot.mmio_access_history:
                prompt_parts.append("Recent MMIO accesses:")
                for pc, addr, is_read, value in snapshot.mmio_access_history[-6:]:
                    access_type = "READ" if is_read else "WRITE"
                    prompt_parts.append(f"  {access_type} pc=0x{pc:08x} addr=0x{addr:08x} value=0x{value:08x}")

            # 显示指令序列
            if snapshot.bb_instructions:
                prompt_parts.append("Instructions:")
                for insn in snapshot.bb_instructions[:12]:
                    mnemonic = insn.get('mnemonic', '')
                    operands = insn.get('operands', '')
                    address = self._parse_hex(insn.get("address"))
                    if address is not None:
                        prompt_parts.append(f"  0x{address:08x}: {mnemonic:8s} {operands}")
                    else:
                        prompt_parts.append(f"  {mnemonic:8s} {operands}")

            prompt_parts.append("")

        prompt_parts.append("## Question")
        prompt_parts.append("Analyze the trace and answer these questions in one JSON object:")
        prompt_parts.append("1. Is the current BB a fatal sink or a wait/self-loop?")
        prompt_parts.append("2. Which branch/load/test instruction is the real progress blocker?")
        prompt_parts.append("3. If this is a self-loop, your constraint must make the loop-back branch NOT taken. If a concrete memory/MMIO constraint can do that, give the exact address/value/read_pc/constraint_pc.")
        prompt_parts.append("4. If no concrete constraint is justifiable, leave suggested_constraints empty but still explain the root cause.")
        prompt_parts.append("")
        prompt_parts.append("Remember:")
        prompt_parts.append("- addresses/values must be hex strings")
        prompt_parts.append("- read_pc must be a real load instruction, never the sink/self-loop branch itself")
        prompt_parts.append("- if this is only an error path and no concrete load can be named, keep suggested_constraints empty")
        prompt_parts.append("- return JSON only")

        return "\n".join(prompt_parts)

    def _call_llm_json(self, messages: List[Dict[str, str]]):
        """调用 LLM，优先要求 JSON 输出，兼容不支持 response_format 的服务端。"""
        repair_prompt = (
            "Return exactly one JSON object. No markdown, no code fence, no extra prose. "
            "`reason` and `problem_description` must be one-line plain text. "
            "All addresses/values must remain hex strings."
        )
        return call_llm_json(
            client=self.client,
            model=self.config.get("model", "qwen-plus"),
            messages=messages,
            # 推理模型思维链与最终答案共用 max_tokens 预算，未显式配置时
            # 默认必须足够大，否则 content 会被截断成空。
            max_tokens=self.config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
            temperature=min(float(self.config.get("temperature", 0.2) or 0.2), 0.2),
            repair_prompt=repair_prompt,
            parse_response=self._parse_json_response,
            logger=logger,
            warn_key=f"codeanalyzer:{self.config.get('model', 'qwen-plus')}",
        )

    def _parse_json_response(self, text: str) -> Dict:
        """解析模型返回，兼容 markdown/json/yaml 风格的额外包裹。"""
        return parse_json_object(text)

    def _normalize_constraint(self, constraint: Dict) -> Optional[Dict]:
        if not isinstance(constraint, dict):
            return None

        constraint_type = str(constraint.get("type", "")).lower()
        if constraint_type not in {"memory", "mmio"}:
            return None

        address = self._parse_hex(constraint.get("address"))
        value = self._parse_hex(constraint.get("value"))
        if address is None or value is None:
            return None

        normalized = {
            "type": constraint_type,
            "address": address & 0xFFFFFFFF,
            "value": value & 0xFFFFFFFF,
            "description": str(constraint.get("description", "")),
        }
        read_pc = self._parse_hex(constraint.get("read_pc"))
        constraint_pc = self._parse_hex(constraint.get("constraint_pc"))
        if constraint_type == "mmio":
            if not (0x40000000 <= normalized["address"] < 0x60000000):
                logger.debug("拒绝非MMIO地址的mmio约束: %s", constraint)
                return None
            if read_pc is None:
                logger.debug("拒绝缺少read_pc的mmio约束: %s", constraint)
                return None

        if read_pc is not None:
            insn = self.instruction_index.get(read_pc)
            if insn is None or not self._is_load_instruction(insn):
                logger.debug("拒绝read_pc不是实际load指令的约束: %s", constraint)
                return None
            normalized["read_pc"] = read_pc & 0xFFFFFFFF
        elif constraint_type == "memory":
            normalized["read_pc"] = None

        if constraint_pc is not None:
            constraint_insn = self.instruction_index.get(constraint_pc)
            if constraint_insn and self._is_fatal_sink_instruction(constraint_insn):
                logger.debug("拒绝constraint_pc指向fatal sink的约束: %s", constraint)
                return None
            normalized["constraint_pc"] = constraint_pc & 0xFFFFFFFF
        return normalized

    def normalize_constraint(self, constraint: Dict) -> Optional[Dict]:
        """公开的约束归一化入口，供运行时加载/落地时复用。"""
        return self._normalize_constraint(constraint)

    def _clamp_confidence(self, value) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = 0.0
        return max(0.0, min(1.0, parsed))

    def _parse_llm_response(self, response) -> CodeAnalysisResult:
        """
        解析LLM响应

        Args:
            response: LLM响应

        Returns:
            分析结果
        """
        try:
            content = extract_response_text(response)
            logger.debug(f"LLM响应内容: {content[:1000]}")
            data = self._parse_json_response(content)
            constraints = []
            for constraint in data.get("suggested_constraints", []) or []:
                normalized = self._normalize_constraint(constraint)
                if normalized:
                    constraints.append(normalized)

            return CodeAnalysisResult(
                need_rollback=bool(data.get("need_rollback", False)),
                rollback_levels=max(0, int(self._parse_hex(data.get("rollback_levels")) or 0)),
                problem_type=str(data.get("problem_type", "unknown")),
                problem_description=str(data.get("problem_description", "")),
                critical_pc=self._parse_hex(data.get("critical_pc")),
                critical_instruction=str(data.get("critical_instruction", "")),
                suggested_constraints=constraints,
                confidence=self._clamp_confidence(data.get("confidence", 0.0)),
                terminal_path=bool(data.get("terminal_path", False)),
                reason=str(data.get("reason", "")),
            )

        except Exception as e:
            logger.warning(f"解析LLM响应失败: {e}")
            return CodeAnalysisResult(confidence=0.0, reason=f"Parse error: {e}")

    def _post_validate_result(
        self,
        result: CodeAnalysisResult,
        recent_snapshots: List,
        current_bb: int,
    ) -> CodeAnalysisResult:
        """用本地语义检查兜底，防止LLM把“继续满足回边条件”误认为“退出循环”."""
        snapshots = self._normalize_snapshot_order(recent_snapshots)
        current_snapshot = snapshots[-1] if snapshots else None
        heuristic = self._analyze_self_loop_constraint(current_snapshot)

        if heuristic.confidence < 0.8 or not heuristic.suggested_constraints:
            return result
        if not result.suggested_constraints:
            return result

        llm_constraint = result.suggested_constraints[0]
        expected_constraint = heuristic.suggested_constraints[0]
        if (
            llm_constraint.get("type") == expected_constraint.get("type")
            and llm_constraint.get("address") == expected_constraint.get("address")
            and llm_constraint.get("value") == expected_constraint.get("value")
        ):
            return result

        logger.warning(
            "LLM结果与本地自环求解冲突，使用本地求解覆盖: current_bb=0x%08x llm=%s expected=%s",
            current_bb,
            llm_constraint,
            expected_constraint,
        )
        heuristic.reason = (
            f"{heuristic.reason}; local validator overrode an inconsistent LLM proposal"
        )
        return heuristic

    def _heuristic_analyze(
        self,
        recent_snapshots: List,
        current_bb: int,
        loop_count: int,
        branch_snapshots: Optional[Dict[int, object]] = None,
        excluded_branch_directions: Optional[Dict[int, set]] = None,
    ) -> CodeAnalysisResult:
        """
        启发式分析（fallback）

        Args:
            recent_snapshots: 最近的快照列表
            current_bb: 当前BB地址
            loop_count: 循环次数

        Returns:
            分析结果
        """
        logger.info("使用启发式分析...")
        result = CodeAnalysisResult()
        snapshots = self._normalize_snapshot_order(recent_snapshots)

        if not snapshots:
            result.confidence = 0.0
            result.reason = "No snapshots available"
            return result

        current_snapshot = snapshots[-1]

        compare_loop_result = self._analyze_self_loop_constraint(current_snapshot)
        if compare_loop_result.confidence >= 0.8:
            logger.info("✓ 检测到可直接求解的自环比较/等待循环")
            return compare_loop_result

        sink_result = self._analyze_fatal_sink(
            snapshots,
            current_bb,
            branch_snapshots=branch_snapshots,
            excluded_branch_directions=excluded_branch_directions,
        )
        if sink_result.confidence >= compare_loop_result.confidence:
            result = sink_result
        else:
            result = compare_loop_result

        if result.confidence < 0.6:
            fallback_result = self._analyze_older_memory_hints(snapshots)
            if fallback_result.confidence > result.confidence:
                result = fallback_result

        if result.confidence == 0.0:
            result.reason = "Heuristic analysis found no clear pattern"
            logger.warning("启发式分析未找到明确模式")

        return result

    def _normalize_snapshot_order(self, snapshots: List) -> List:
        ordered = list(snapshots or [])
        if len(ordered) >= 2:
            first_count = getattr(ordered[0], "instruction_count", 0)
            last_count = getattr(ordered[-1], "instruction_count", 0)
            if first_count > last_count:
                ordered.reverse()
        return ordered

    def _important_registers(self, cpu_state: Dict[str, int]) -> List[Tuple[str, int]]:
        order = ["r0", "r1", "r2", "r3", "r4", "r5", "sp", "lr"]
        result = []
        for reg in order:
            if reg in cpu_state:
                result.append((reg, cpu_state[reg]))
        return result

    def _normalize_mnemonic(self, mnemonic) -> str:
        raw = str(mnemonic or "").strip().upper()
        if not raw:
            return ""
        parts = [part for part in raw.split(".") if part]
        if len(parts) >= 2 and parts[0] == "B":
            condition = parts[1]
            if condition not in {"N", "W", "NW"}:
                return f"B{condition}"
            return "B"
        return parts[0]

    def _split_operands(self, operands: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for ch in str(operands or ""):
            if ch == "[":
                depth += 1
            elif ch == "]" and depth > 0:
                depth -= 1

            if ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _reg_name(self, token: str) -> Optional[str]:
        token = str(token or "").strip().lower()
        if re.fullmatch(r"r(?:[0-9]|1[0-2])", token):
            return token
        if token in {"sp", "lr", "pc"}:
            return token
        return None

    def _parse_immediate(self, token: str) -> Optional[int]:
        token = str(token or "").strip()
        if token.startswith("#"):
            token = token[1:]
        token = token.strip()
        if self._reg_name(token) is not None:
            return None
        if not re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|\d+)", token):
            return None
        return self._parse_hex(token)

    def _parse_branch_target(self, operands: str) -> Optional[int]:
        matches = re.findall(r"0x[0-9a-fA-F]+", str(operands or ""))
        if not matches:
            return None
        try:
            return int(matches[-1], 16)
        except ValueError:
            return None

    def _resolve_bb_start(self, address: Optional[int]) -> Optional[int]:
        if address is None:
            return None
        if address in self.static_bbs:
            return address
        return self.instruction_to_bb.get(address)

    def _next_sequential_bb(self, bb_addr: int) -> Optional[int]:
        if not self.sorted_bb_addresses:
            return None
        index = bisect_right(self.sorted_bb_addresses, bb_addr)
        if index >= len(self.sorted_bb_addresses):
            return None
        return self.sorted_bb_addresses[index]

    def _bb_can_fallthrough(self, mnemonic: str) -> bool:
        if mnemonic in {"B", "BX", "POP"}:
            return False
        return True

    def _describe_current_pattern(self, snapshot, current_bb: int) -> str:
        if snapshot is None or not snapshot.bb_instructions:
            return f"no snapshot for 0x{current_bb:08x}"

        last_insn = snapshot.bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic"))
        target = self._parse_branch_target(last_insn.get("operands", ""))
        target_bb = self._resolve_bb_start(target)

        if mnemonic == "B" and target_bb == snapshot.bb_address:
            return "single unconditional self-loop (fatal sink / error handler candidate)"
        if mnemonic in {"BEQ", "BNE", "BGT", "BLT", "BGE", "BLE", "BHI", "BLO", "BHS", "BLS", "CBZ", "CBNZ"} and target_bb == snapshot.bb_address:
            return "conditional self-loop (wait loop / state-not-progressing candidate)"
        return f"terminal instruction {mnemonic} {last_insn.get('operands', '')}".strip()

    def _describe_static_references(self, current_bb: int) -> List[str]:
        refs = self._incoming_refs_to_bb(current_bb)
        descriptions = []
        for ref in refs[:6]:
            descriptions.append(
                f"0x{ref['address']:08x}: {ref['mnemonic']} {ref['operands']} (from BB 0x{ref['bb']:08x})"
            )
        return descriptions

    def _describe_static_predecessor_contexts(self, current_bb: int, max_refs: int = 4) -> List[str]:
        contexts: List[str] = []
        seen = set()
        direct_refs = self._incoming_refs_to_bb(current_bb)

        for ref in direct_refs[:max_refs]:
            caller_bb = ref["bb"]
            caller_context = self._render_bb_context(caller_bb, highlight_address=ref["address"])
            if caller_context and ("direct", caller_bb) not in seen:
                contexts.append(
                    f"- Direct incoming BB 0x{caller_bb:08x} into 0x{current_bb:08x}:\n{caller_context}"
                )
                seen.add(("direct", caller_bb))

            for upstream in self.target_reference_index.get(caller_bb, [])[:2]:
                upstream_bb = upstream["bb"]
                if not self._is_conditional_branch_mnemonic(str(upstream.get("mnemonic", ""))):
                    continue
                key = ("upstream", upstream_bb, caller_bb)
                if key in seen:
                    continue
                upstream_context = self._render_bb_context(
                    upstream_bb,
                    highlight_address=upstream["address"],
                )
                if upstream_context:
                    contexts.append(
                        f"- Upstream conditional BB 0x{upstream_bb:08x} reaching caller BB 0x{caller_bb:08x}:\n"
                        f"{upstream_context}"
                    )
                    seen.add(key)

        return contexts

    def _build_heuristic_hint(self, current_snapshot) -> str:
        if current_snapshot is None:
            return ""
        hint = self._analyze_self_loop_constraint(current_snapshot)
        if hint.confidence >= 0.8 and hint.suggested_constraints:
            constraint = hint.suggested_constraints[0]
            read_pc = constraint.get("read_pc")
            constraint_pc = constraint.get("constraint_pc")
            return (
                f"local solver thinks this loop can be exited by setting "
                f"{constraint['type']}[0x{constraint['address']:08x}] = 0x{constraint['value']:08x} "
                f"(read_pc={self._format_optional_hex(read_pc)}, constraint_pc={self._format_optional_hex(constraint_pc)})"
            )
        return hint.reason or ""

    def _is_self_loop_branch(self, snapshot) -> bool:
        if snapshot is None or not snapshot.bb_instructions:
            return False
        last_insn = snapshot.bb_instructions[-1]
        target = self._resolve_bb_start(self._parse_branch_target(last_insn.get("operands", "")))
        return target == snapshot.bb_address

    def _is_fatal_sink_snapshot(self, snapshot) -> bool:
        if snapshot is None or not snapshot.bb_instructions:
            return False
        last_insn = snapshot.bb_instructions[-1]
        mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic"))
        return mnemonic == "B" and self._is_self_loop_branch(snapshot) and len(snapshot.bb_instructions) <= 2

    def _is_fatal_sink_instruction(self, insn: Dict) -> bool:
        address = self._parse_hex(insn.get("address"))
        if address is None:
            return False
        bb_addr = self.instruction_to_bb.get(address)
        if bb_addr is None:
            return False
        instructions = self.static_bbs.get(bb_addr, [])
        if not instructions:
            return False
        return self._is_fatal_sink_snapshot(type("Snapshot", (), {
            "bb_instructions": instructions,
            "bb_address": bb_addr,
        })())

    def _analyze_fatal_sink(
        self,
        snapshots: List,
        current_bb: int,
        branch_snapshots: Optional[Dict[int, object]] = None,
        excluded_branch_directions: Optional[Dict[int, set]] = None,
    ) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        if not snapshots:
            return result

        current_snapshot = snapshots[-1]
        if not self._is_fatal_sink_snapshot(current_snapshot):
            return result

        result.need_rollback = True
        result.rollback_levels = 1
        result.problem_type = "error_handler"
        result.problem_description = "Current BB is a single unconditional self-loop fatal sink"
        result.confidence = 0.75
        result.terminal_path = True

        recent_recoverable = self._recover_recent_path_constraint(
            snapshots,
            current_bb,
            excluded_branch_directions=excluded_branch_directions,
        )
        if recent_recoverable.confidence >= 0.8 and recent_recoverable.suggested_constraints:
            return recent_recoverable

        for prev_snapshot in reversed(snapshots[:-1]):
            for insn in reversed(prev_snapshot.bb_instructions or []):
                target = self._resolve_bb_start(self._parse_branch_target(insn.get("operands", "")))
                if target == current_bb:
                    result.critical_pc = self._parse_hex(insn.get("address")) or prev_snapshot.bb_address
                    result.critical_instruction = self._format_instruction(insn)
                    result.reason = (
                        f"Reached fatal sink 0x{current_bb:08x} from recent BB 0x{prev_snapshot.bb_address:08x} "
                        f"via {result.critical_instruction}"
                    )
                    upstream_ref, chain = self._find_actionable_predecessor_ref(prev_snapshot.bb_address, snapshots)
                    if upstream_ref is not None:
                        upstream_context = self._resolve_branch_context_snapshot(
                            snapshots,
                            upstream_ref,
                            branch_snapshots=branch_snapshots,
                        )
                        recoverable = self._recover_fatal_sink_constraint(
                            snapshots,
                            current_bb,
                            upstream_ref,
                            chain,
                            branch_snapshots=branch_snapshots,
                            excluded_branch_directions=excluded_branch_directions,
                        )
                        if recoverable.confidence >= 0.8 and recoverable.suggested_constraints:
                            return recoverable
                        if upstream_context is None:
                            logger.info(
                                "忽略缺少动态上下文的静态上游分支 @ 0x%08x",
                                self._parse_hex(upstream_ref.get("address")) or 0,
                            )
                            if self._is_wrapper_to_terminal_sink(prev_snapshot.bb_address, current_bb):
                                result.reason = (
                                    f"Reached fatal sink 0x{current_bb:08x} via wrapper BB 0x{prev_snapshot.bb_address:08x}; "
                                    f"this path terminates in a sink without a recoverable load/test constraint"
                                )
                            return result
                        result.critical_pc = upstream_ref["address"]
                        result.critical_instruction = self._format_ref_instruction(upstream_ref)
                        result.problem_description = "Fatal sink is controlled by an upstream conditional branch"
                        result.confidence = 0.82
                        result.reason = (
                            f"Reached fatal sink 0x{current_bb:08x} via caller BB 0x{prev_snapshot.bb_address:08x}; "
                            f"upstream condition is {result.critical_instruction}"
                        )
                        if chain:
                            result.reason += f" (chain: {self._format_bb_chain(chain)})"
                    elif self._is_wrapper_to_terminal_sink(prev_snapshot.bb_address, current_bb):
                        result.reason = (
                            f"Reached fatal sink 0x{current_bb:08x} via wrapper BB 0x{prev_snapshot.bb_address:08x}; "
                            f"this path terminates in a sink without a recoverable load/test constraint"
                        )
                    return result

        static_refs = self._describe_static_references(current_bb)
        for ref in self._incoming_refs_to_bb(current_bb):
            upstream_ref, chain = self._find_actionable_predecessor_ref(ref["bb"], snapshots)
            if upstream_ref is None:
                continue
            upstream_context = self._resolve_branch_context_snapshot(
                snapshots,
                upstream_ref,
                branch_snapshots=branch_snapshots,
            )
            recoverable = self._recover_fatal_sink_constraint(
                snapshots,
                current_bb,
                upstream_ref,
                chain,
                branch_snapshots=branch_snapshots,
                excluded_branch_directions=excluded_branch_directions,
            )
            if recoverable.confidence >= 0.8 and recoverable.suggested_constraints:
                return recoverable
            if upstream_context is None:
                continue
            result.critical_pc = upstream_ref["address"]
            result.critical_instruction = self._format_ref_instruction(upstream_ref)
            result.problem_description = "Fatal sink is controlled by an upstream conditional branch"
            result.confidence = 0.8
            result.reason = (
                f"Static sink predecessor analysis traced 0x{current_bb:08x} back to "
                f"{result.critical_instruction}"
            )
            if chain:
                result.reason += f" (chain: {self._format_bb_chain(chain)})"
            return result

        if static_refs:
            result.reason = "Fatal sink with static incoming refs: " + "; ".join(static_refs[:3])
        return result

    def _recover_recent_path_constraint(
        self,
        snapshots: List,
        current_bb: int,
        excluded_branch_directions: Optional[Dict[int, set]] = None,
    ) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        ordered = self._normalize_snapshot_order(snapshots)
        if len(ordered) < 2:
            return result

        for index in range(len(ordered) - 2, -1, -1):
            snapshot = ordered[index]
            branch_ref = self._find_controlling_conditional_in_bb(snapshot.bb_address)
            if branch_ref is None:
                continue

            actual_taken = self._infer_observed_branch_taken(branch_ref, ordered, index)
            if actual_taken is None:
                continue
            branch_pc = self._parse_hex(branch_ref.get("address"))
            desired_taken = not actual_taken
            if (
                branch_pc is not None
                and excluded_branch_directions
                and desired_taken in excluded_branch_directions.get(branch_pc, set())
            ):
                continue

            recoverable = self._analyze_branch_constraint_snapshot(
                snapshot,
                branch_ref,
                desired_taken=desired_taken,
            )
            if recoverable.confidence < 0.8 or not recoverable.suggested_constraints:
                continue

            recoverable.problem_type = "error_handler"
            recoverable.problem_description = "Fatal sink is controlled by a recent-path branch constraint"
            recoverable.terminal_path = False
            recoverable.rollback_levels = max(1, len(ordered) - 1 - index)
            recoverable.reason = (
                f"Recent path into fatal sink 0x{current_bb:08x} passed through "
                f"{self._format_ref_instruction(branch_ref)} and then BB 0x{ordered[index + 1].bb_address:08x}; "
                f"{recoverable.reason}"
            )
            return recoverable

        return result

    def _recover_fatal_sink_constraint(
        self,
        snapshots: List,
        current_bb: int,
        upstream_ref: Dict[str, object],
        chain: List[int],
        branch_snapshots: Optional[Dict[int, object]] = None,
        excluded_branch_directions: Optional[Dict[int, set]] = None,
    ) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        sink_taken = self._infer_sink_path_taken(upstream_ref, chain)
        if sink_taken is None:
            return result
        branch_pc = self._parse_hex(upstream_ref.get("address"))
        desired_taken = not sink_taken
        if (
            branch_pc is not None
            and excluded_branch_directions
            and desired_taken in excluded_branch_directions.get(branch_pc, set())
        ):
            return result

        branch_snapshot = self._resolve_branch_context_snapshot(
            snapshots,
            upstream_ref,
            branch_snapshots=branch_snapshots,
        )
        if branch_snapshot is None:
            logger.info(
                "上游分支恢复失败: 缺少分支上下文快照 @ 0x%08x",
                branch_pc or 0,
            )
            return result

        recoverable = self._analyze_branch_constraint_snapshot(
            branch_snapshot,
            upstream_ref,
            desired_taken=desired_taken,
        )
        if recoverable.confidence < 0.8 or not recoverable.suggested_constraints:
            logger.info(
                "上游分支恢复失败: 无法从上下文求出具体约束 @ 0x%08x (bb=0x%08x, regs=%s)",
                branch_pc or 0,
                self._parse_hex(upstream_ref.get("bb")) or 0,
                ", ".join(sorted((getattr(branch_snapshot, "cpu_state", {}) or {}).keys())),
            )
            return result

        recoverable.problem_type = "error_handler"
        recoverable.problem_description = "Fatal sink is controlled by a recoverable upstream branch constraint"
        recoverable.terminal_path = False

        rollback_levels = self._compute_rollback_levels_to_bb(
            snapshots,
            upstream_ref["bb"],
            upstream_ref["address"],
        )
        if rollback_levels is not None:
            recoverable.rollback_levels = max(1, rollback_levels)
        elif getattr(branch_snapshot, "_branch_snapshot_origin", False):
            recoverable.rollback_levels = 0
            recoverable.restore_branch_bb = upstream_ref["bb"]

        prefix = (
            f"Fatal sink 0x{current_bb:08x} is avoidable via "
            f"{self._format_ref_instruction(upstream_ref)}"
        )
        if chain:
            prefix += f" (chain: {self._format_bb_chain(chain)})"
        recoverable.reason = f"{prefix}; {recoverable.reason}"
        return recoverable

    def _resolve_branch_context_snapshot(
        self,
        snapshots: List,
        branch_ref: Dict[str, object],
        branch_snapshots: Optional[Dict[int, object]] = None,
    ):
        branch_pc = self._parse_hex(branch_ref.get("address"))
        branch_bb = self._parse_hex(branch_ref.get("bb"))

        for snapshot in reversed(snapshots or []):
            if branch_bb is not None and getattr(snapshot, "bb_address", None) == branch_bb:
                return snapshot
            if branch_pc is not None and self._snapshot_contains_instruction(snapshot, branch_pc):
                return snapshot

        if branch_bb is not None and branch_snapshots:
            branch_snapshot = branch_snapshots.get(branch_bb)
            if branch_snapshot is not None:
                return self._materialize_branch_snapshot(branch_snapshot)
        return None

    def _materialize_branch_snapshot(self, branch_snapshot):
        branch_bb = self._parse_hex(getattr(branch_snapshot, "address", None))
        if branch_bb is None:
            return None

        memory_regions = {}
        memory_base = getattr(branch_snapshot, "memory_base", None)
        memory_size = getattr(branch_snapshot, "memory_size", None)
        memory_data = getattr(branch_snapshot, "memory_data", None)
        if memory_base is not None and memory_size and memory_data is not None:
            memory_regions[(memory_base, memory_size)] = bytes(memory_data)

        snapshot = type("BranchContextSnapshot", (), {})()
        snapshot.bb_address = branch_bb
        snapshot.timestamp = 0.0
        snapshot.instruction_count = 0
        snapshot.cpu_state = dict(getattr(branch_snapshot, "registers", {}) or {})
        snapshot.pc = snapshot.cpu_state.get("pc", branch_bb)
        snapshot.flags = getattr(branch_snapshot, "cpsr", 0)
        snapshot.memory_regions = memory_regions
        snapshot.mmio_values = {}
        snapshot.mmio_access_history = []
        snapshot.execution_path = []
        snapshot.loop_counters = {}
        snapshot.bb_instructions = list(self.static_bbs.get(branch_bb, []))
        snapshot._branch_snapshot_origin = True
        return snapshot

    def _snapshot_contains_instruction(self, snapshot, address: int) -> bool:
        for insn in getattr(snapshot, "bb_instructions", []) or []:
            if self._parse_hex(insn.get("address")) == address:
                return True
        return False

    def _infer_sink_path_taken(self, branch_ref: Dict[str, object], chain: List[int]) -> Optional[bool]:
        branch_bb = self._parse_hex(branch_ref.get("bb"))
        if branch_bb is None:
            return None

        next_on_sink = None
        if chain:
            try:
                index = chain.index(branch_bb)
            except ValueError:
                index = -1
            if index >= 0 and index + 1 < len(chain):
                next_on_sink = chain[index + 1]

        branch_target = self._resolve_bb_start(
            self._parse_branch_target(str(branch_ref.get("operands", "")))
        )
        if next_on_sink is None:
            if branch_target is not None:
                return True
            return None
        if branch_target == next_on_sink:
            return True

        fallthrough_bb = self._next_sequential_bb(branch_bb)
        if fallthrough_bb == next_on_sink:
            return False
        return None

    def _compute_rollback_levels_to_bb(
        self,
        snapshots: List,
        branch_bb: int,
        branch_pc: Optional[int] = None,
    ) -> Optional[int]:
        ordered = self._normalize_snapshot_order(snapshots)
        for index, snapshot in enumerate(ordered):
            if getattr(snapshot, "bb_address", None) == branch_bb:
                return max(0, len(ordered) - 1 - index)
            if branch_pc is not None and self._snapshot_contains_instruction(snapshot, branch_pc):
                return max(0, len(ordered) - 1 - index)
        return None

    def _infer_observed_branch_taken(
        self,
        branch_ref: Dict[str, object],
        ordered_snapshots: List,
        snapshot_index: int,
    ) -> Optional[bool]:
        if snapshot_index + 1 >= len(ordered_snapshots):
            return None

        next_bb = getattr(ordered_snapshots[snapshot_index + 1], "bb_address", None)
        if next_bb is None:
            return None

        branch_target = self._resolve_bb_start(
            self._parse_branch_target(str(branch_ref.get("operands", "")))
        )
        if branch_target == next_bb:
            return True

        branch_bb = self._parse_hex(branch_ref.get("bb"))
        if branch_bb is None:
            return None
        fallthrough_bb = self._next_sequential_bb(branch_bb)
        if fallthrough_bb == next_bb:
            return False
        return None

    def _find_instruction_index(self, instructions: List[Dict], address: int) -> Optional[int]:
        for index, insn in enumerate(instructions or []):
            if self._parse_hex(insn.get("address")) == address:
                return index
        return None

    def _find_compare_instruction_before(self, instructions: List[Dict], branch_index: int) -> Optional[Dict]:
        if branch_index <= 0:
            return None
        for insn in reversed((instructions or [])[:branch_index]):
            if self._normalize_mnemonic(insn.get("mnemonic")) in {"CMP", "CMN", "TST", "TEQ"}:
                return insn
        return None

    def _analyze_branch_constraint_snapshot(
        self,
        snapshot,
        branch_ref: Dict[str, object],
        desired_taken: bool,
    ) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        instructions = getattr(snapshot, "bb_instructions", []) or []
        branch_pc = self._parse_hex(branch_ref.get("address"))
        if branch_pc is None or not instructions:
            return result

        branch_index = self._find_instruction_index(instructions, branch_pc)
        if branch_index is None:
            return result

        branch_insn = instructions[branch_index]
        branch_mnemonic = self._normalize_mnemonic(branch_insn.get("mnemonic"))
        reg_values, reg_sources = self._evaluate_snapshot_registers(snapshot, end_index=branch_index)

        source = None
        desired_value = None
        constraint_pc = branch_pc
        reason = ""

        if branch_mnemonic in {"CBZ", "CBNZ"}:
            parts = self._split_operands(branch_insn.get("operands", ""))
            source_reg = self._reg_name(parts[0]) if parts else None
            source = reg_sources.get(source_reg) if source_reg else None
            desired_value = self._infer_cbz_cbnz_value(
                branch_mnemonic,
                desired_taken,
                source_reg,
                reg_values,
                source.get("width", 4) if source else 4,
            )
            reason = (
                f"{branch_mnemonic} at 0x{branch_pc:08x} controls this path; "
                f"set {source_reg} to {'zero' if desired_value == 0 else 'non-zero'}"
            )
        else:
            compare_insn = self._find_compare_instruction_before(instructions, branch_index)
            if compare_insn is None:
                return result
            compare_mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic"))
            parts = self._split_operands(compare_insn.get("operands", ""))
            if len(parts) < 2:
                return result

            source_reg, source = self._select_constraint_source(parts, reg_sources)
            if source is None:
                return result

            desired_value = self._infer_branch_condition_value(
                branch_mnemonic,
                compare_mnemonic,
                source_reg,
                parts,
                reg_values,
                source.get("width", 4),
                desired_taken,
            )
            constraint_pc = self._parse_hex(compare_insn.get("address")) or branch_pc
            reason = (
                f"{branch_mnemonic} at 0x{branch_pc:08x} depends on "
                f"{self._format_instruction(compare_insn)}"
            )

        if source is None or desired_value is None:
            logger.info(
                "分支约束求解失败 @ 0x%08x: mnemonic=%s source=%s desired=%s",
                branch_pc,
                branch_mnemonic,
                source,
                desired_value,
            )
            return result

        desired_value &= self._width_mask(source.get("width", 4))
        result.need_rollback = True
        result.rollback_levels = 1
        result.problem_type = source["type"]
        result.problem_description = "Recovered a concrete load-controlled branch constraint"
        result.critical_pc = constraint_pc
        result.critical_instruction = self._format_ref_instruction(branch_ref)
        result.control_branch_pc = branch_pc
        result.desired_branch_taken = bool(desired_taken)
        result.suggested_constraints.append({
            "type": source["type"],
            "read_pc": source["read_pc"],
            "address": source["address"],
            "value": desired_value,
            "constraint_pc": constraint_pc,
            "description": reason,
        })
        result.confidence = 0.9
        result.reason = (
            f"{reason}; set {source['type']}[0x{source['address']:08x}] = 0x{desired_value:08x} "
            f"so the branch becomes {'taken' if desired_taken else 'not taken'}"
        )
        return result

    def _find_actionable_predecessor_ref(
        self,
        target_bb: int,
        snapshots: List,
        max_depth: int = 6,
        visited: Optional[set] = None,
    ) -> Tuple[Optional[Dict[str, object]], List[int]]:
        if visited is None:
            visited = set()
        if max_depth < 0 or target_bb in visited:
            return None, []
        visited.add(target_bb)

        in_bb_ref = self._find_controlling_conditional_in_bb(target_bb)
        if in_bb_ref is not None:
            return in_bb_ref, [target_bb]

        recent_bbs = {getattr(snapshot, "bb_address", None) for snapshot in snapshots or []}
        refs = self._incoming_refs_to_bb(target_bb)
        ranked = sorted(
            refs,
            key=lambda ref: (
                0 if ref["bb"] in recent_bbs else 1,
                0 if ref.get("kind") == "fallthrough" else 1,
                0 if self._is_conditional_branch_mnemonic(str(ref.get("mnemonic", ""))) else 1,
                -int(ref.get("address") or 0),
            ),
        )
        for ref in ranked:
            if self._is_conditional_branch_mnemonic(str(ref.get("mnemonic", ""))):
                return ref, [ref["bb"], target_bb]

            candidate, chain = self._find_actionable_predecessor_ref(
                ref["bb"],
                snapshots,
                max_depth=max_depth - 1,
                visited=visited,
            )
            if candidate is not None:
                return candidate, chain + [target_bb]

        return None, []

    def _analyze_self_loop_constraint(self, snapshot) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        if snapshot is None or not snapshot.bb_instructions:
            return result

        instructions = snapshot.bb_instructions
        last_insn = instructions[-1]
        branch_mnemonic = self._normalize_mnemonic(last_insn.get("mnemonic"))
        if branch_mnemonic not in {
            "BEQ", "BNE", "BGT", "BLT", "BGE", "BLE", "BHI", "BLO", "BHS", "BLS", "CBZ", "CBNZ"
        }:
            return result
        if not self._is_self_loop_branch(snapshot):
            return result

        reg_values, reg_sources = self._evaluate_snapshot_registers(snapshot)
        branch_pc = self._parse_hex(last_insn.get("address")) or snapshot.bb_address

        if branch_mnemonic in {"CBZ", "CBNZ"}:
            source_reg = self._reg_name(self._split_operands(last_insn.get("operands", ""))[0]) if self._split_operands(last_insn.get("operands", "")) else None
            source = reg_sources.get(source_reg) if source_reg else None
            desired = 1 if branch_mnemonic == "CBZ" else 0
            compare_pc = branch_pc
            reason = f"Self-loop {branch_mnemonic} exits when {source_reg} becomes {'non-zero' if desired else 'zero'}"
        else:
            compare_insn = None
            for insn in reversed(instructions[:-1]):
                if self._normalize_mnemonic(insn.get("mnemonic")) in {"CMP", "CMN", "TST", "TEQ"}:
                    compare_insn = insn
                    break
            if compare_insn is None:
                return result

            compare_mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic"))
            parts = self._split_operands(compare_insn.get("operands", ""))
            if len(parts) < 2:
                return result

            # The loaded value is usually on the left (`cmp r3, #imm`), but
            # firmware table scans also use forms such as `cmp r3, r0` where
            # either side may be the load-derived operand after a register
            # move/extension. Reuse the general branch solver so self-loop
            # handling is symmetric with predecessor branch recovery.
            source_reg, source = self._select_constraint_source(parts, reg_sources)
            desired = self._infer_branch_condition_value(
                branch_mnemonic,
                compare_mnemonic,
                source_reg,
                parts,
                reg_values,
                source.get("width", 4) if source else 4,
                False,
            )
            compare_pc = self._parse_hex(compare_insn.get("address")) or branch_pc
            reason = (
                f"Self-loop {branch_mnemonic} at 0x{branch_pc:08x} depends on "
                f"{self._format_instruction(compare_insn)}"
            )

        if source is None or desired is None:
            result.need_rollback = True
            result.rollback_levels = 1
            result.problem_type = "branch_condition"
            result.problem_description = "Detected self-looping branch but no concrete load source/value was derived"
            result.confidence = 0.45
            result.reason = reason
            return result

        desired &= self._width_mask(source.get("width", 4))
        result.need_rollback = True
        result.rollback_levels = 1
        result.problem_type = source["type"]
        result.problem_description = "Detected self-loop controlled by loaded state; synthesized exit constraint"
        result.critical_pc = compare_pc
        result.critical_instruction = self._format_instruction(last_insn)
        result.suggested_constraints.append({
            "type": source["type"],
            "read_pc": source["read_pc"],
            "address": source["address"],
            "value": desired,
            "constraint_pc": compare_pc,
            "description": reason,
        })
        result.confidence = 0.9
        result.reason = (
            f"{reason}; set {source['type']}[0x{source['address']:08x}] = 0x{desired:08x} "
            f"to break the self-loop"
        )
        return result

    def _analyze_older_memory_hints(self, snapshots: List) -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        for snapshot in reversed(snapshots[:-1]):
            for insn in snapshot.bb_instructions or []:
                mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
                operands = str(insn.get("operands", ""))
                if mnemonic == "LDR" and "=" in operands:
                    result.suggested_constraints.append({
                        "type": "memory",
                        "address": 0x20008000,
                        "value": 0x20008000,
                        "description": f"Initialize memory loaded by: {mnemonic} {operands}",
                    })
                    result.need_rollback = True
                    result.rollback_levels = 2
                    result.problem_type = "uninitialized_memory"
                    result.problem_description = "Found earlier literal load that may require initialization"
                    result.confidence = 0.65
                    result.reason = f"Found potential uninitialized memory load in 0x{snapshot.bb_address:08x}: {operands}"
                    return result
        return result

    def _evaluate_snapshot_registers(
        self,
        snapshot,
        end_index: Optional[int] = None,
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        reg_values = {
            self._reg_name(reg): value & 0xFFFFFFFF
            for reg, value in (snapshot.cpu_state or {}).items()
            if self._reg_name(reg) is not None
        }
        reg_sources: Dict[str, Dict[str, int]] = {}
        instructions = list(snapshot.bb_instructions or [])
        if end_index is not None:
            if end_index < 0:
                instructions = []
            else:
                instructions = instructions[:end_index + 1]

        for insn in instructions:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
            parts = self._split_operands(insn.get("operands", ""))

            if mnemonic in {"BL", "BLX"}:
                # Calls clobber ARM volatile registers.  Keeping an r0-r3
                # memory source alive across a call wrongly attributes return
                # value checks to pre-call object/pointer loads.
                for reg in ("r0", "r1", "r2", "r3", "r12", "lr"):
                    reg_values.pop(reg, None)
                    reg_sources.pop(reg, None)
                continue

            if mnemonic in {"BX", "BXJ"}:
                # A register branch is a control-flow boundary in the linear
                # local model.  Do not carry volatile data dependencies across
                # it into a later sorted-BB compare.
                for reg in ("r0", "r1", "r2", "r3", "r12", "lr"):
                    reg_values.pop(reg, None)
                    reg_sources.pop(reg, None)
                continue

            if mnemonic in {"MOV", "MOVS", "MOVW"} and len(parts) >= 2:
                dest = self._reg_name(parts[0])
                imm = self._parse_immediate(parts[1])
                src = self._reg_name(parts[1])
                if dest and imm is not None:
                    reg_values[dest] = imm & 0xFFFFFFFF
                    reg_sources.pop(dest, None)
                elif dest and src and src in reg_values:
                    reg_values[dest] = reg_values[src]
                    if src in reg_sources:
                        reg_sources[dest] = dict(reg_sources[src])
                continue

            if mnemonic == "MOVT" and len(parts) >= 2:
                dest = self._reg_name(parts[0])
                imm = self._parse_immediate(parts[1])
                if dest and imm is not None:
                    low = reg_values.get(dest, 0) & 0xFFFF
                    reg_values[dest] = ((imm & 0xFFFF) << 16) | low
                    reg_sources.pop(dest, None)
                continue

            if mnemonic in {"UXTB", "UXTH"} and len(parts) >= 2:
                dest = self._reg_name(parts[0])
                src = self._reg_name(parts[1])
                if dest and src and src in reg_values:
                    mask = 0xFF if mnemonic == "UXTB" else 0xFFFF
                    reg_values[dest] = reg_values[src] & mask
                    if src in reg_sources:
                        reg_sources[dest] = dict(reg_sources[src])
                        reg_sources[dest]["width"] = min(reg_sources[dest].get("width", 4), 1 if mnemonic == "UXTB" else 2)
                continue

            if mnemonic in {"ADD", "ADDS", "SUB", "SUBS"} and len(parts) >= 3:
                dest = self._reg_name(parts[0])
                left = self._reg_name(parts[1])
                imm = self._parse_immediate(parts[2])
                right = self._reg_name(parts[2])
                if dest and left and left in reg_values:
                    if imm is not None:
                        if mnemonic.startswith("ADD"):
                            reg_values[dest] = (reg_values[left] + imm) & 0xFFFFFFFF
                        else:
                            reg_values[dest] = (reg_values[left] - imm) & 0xFFFFFFFF
                        reg_sources.pop(dest, None)
                    elif right and right in reg_values:
                        if mnemonic.startswith("ADD"):
                            reg_values[dest] = (reg_values[left] + reg_values[right]) & 0xFFFFFFFF
                        else:
                            reg_values[dest] = (reg_values[left] - reg_values[right]) & 0xFFFFFFFF
                        reg_sources.pop(dest, None)
                continue

            if mnemonic in {"AND", "ANDS"} and len(parts) >= 3:
                dest = self._reg_name(parts[0])
                left = self._reg_name(parts[1])
                imm = self._parse_immediate(parts[2])
                if dest and left and left in reg_values and imm is not None:
                    reg_values[dest] = reg_values[left] & imm
                    if left in reg_sources:
                        reg_sources[dest] = dict(reg_sources[left])
                continue

            if mnemonic in {"LDR", "LDR.W", "LDRB", "LDRH", "LDRH.W"} and len(parts) >= 2:
                dest = self._reg_name(parts[0])
                is_pc_relative = self._memory_operand_base(parts[1]) == "pc"
                addr = self._resolve_memory_operand(parts[1], reg_values, insn=insn)
                if dest and addr is not None:
                    width = 1 if "LDRB" in mnemonic else 2 if "LDRH" in mnemonic else 4
                    value = self._read_snapshot_value(snapshot, addr, width)
                    if value is not None:
                        reg_values[dest] = value & self._width_mask(width)
                    else:
                        reg_values.pop(dest, None)
                    # PC-relative LDRs are literal-pool constants, not external
                    # state reads. If the literal is available, use it to carry
                    # the pointer/value forward; if not, avoid generating a
                    # bogus constraint on the code address itself.
                    if is_pc_relative:
                        reg_sources.pop(dest, None)
                    else:
                        reg_sources[dest] = {
                            "type": "mmio" if 0x40000000 <= addr < 0x60000000 else "memory",
                            "address": addr & 0xFFFFFFFF,
                            "read_pc": self._parse_hex(insn.get("address")) or snapshot.bb_address,
                            "width": width,
                        }
                continue

        return reg_values, reg_sources

    def _select_constraint_source(
        self,
        compare_parts: List[str],
        reg_sources: Dict[str, Dict[str, int]],
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        if not compare_parts:
            return None, None

        left_reg = self._reg_name(compare_parts[0])
        left_source = reg_sources.get(left_reg) if left_reg else None
        if left_source is not None:
            return left_reg, left_source

        if len(compare_parts) >= 2:
            right_reg = self._reg_name(compare_parts[1])
            right_source = reg_sources.get(right_reg) if right_reg else None
            if right_source is not None:
                return right_reg, right_source
        return None, None

    def _memory_operand_base(self, operand: str) -> Optional[str]:
        text = str(operand or "").strip()
        match = re.match(r"\[\s*([^\],]+)\s*(?:,\s*([^\]]+))?\]", text)
        if not match:
            return None

        base = self._reg_name(match.group(1))
        return base

    def _architectural_pc_for_instruction(self, insn: Optional[Dict]) -> Optional[int]:
        address = self._parse_hex((insn or {}).get("address"))
        if address is None:
            return None
        if self.thumb_mode is None:
            size = self._parse_hex((insn or {}).get("size"))
            thumb_mode = bool(size == 2)
        else:
            thumb_mode = bool(self.thumb_mode)
        if thumb_mode:
            return ((address + 4) & ~0x3) & 0xFFFFFFFF
        return (address + 8) & 0xFFFFFFFF

    def resolve_memory_operand_for_instruction(
        self,
        insn: Dict,
        operand: str,
        reg_values: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        return self._resolve_memory_operand(operand, reg_values or {}, insn=insn)

    def _resolve_memory_operand(
        self,
        operand: str,
        reg_values: Dict[str, int],
        insn: Optional[Dict] = None,
    ) -> Optional[int]:
        text = str(operand or "").strip()
        match = re.match(r"\[\s*([^\],]+)\s*(?:,\s*([^\]]+))?\]", text)
        if not match:
            return None

        base = self._reg_name(match.group(1))
        offset_text = (match.group(2) or "").strip()
        if base is None:
            return None

        if base == "pc" and insn is not None:
            pc_value = self._architectural_pc_for_instruction(insn)
            if pc_value is None:
                return None
            address = pc_value
        elif base in reg_values:
            address = reg_values[base]
        else:
            return None

        if not offset_text:
            return address & 0xFFFFFFFF

        offset_reg = self._reg_name(offset_text)
        if offset_reg and offset_reg in reg_values:
            return (address + reg_values[offset_reg]) & 0xFFFFFFFF

        offset = self._parse_immediate(offset_text)
        if offset is None:
            return None
        return (address + offset) & 0xFFFFFFFF

    def _read_snapshot_value(self, snapshot, address: int, width: int) -> Optional[int]:
        if 0x40000000 <= address < 0x60000000 and address in (snapshot.mmio_values or {}):
            return snapshot.mmio_values[address]

        for (start, size), data in (snapshot.memory_regions or {}).items():
            end = start + size
            if start <= address and address + width <= end:
                offset = address - start
                return int.from_bytes(data[offset:offset + width], "little")
        return None

    def _infer_self_loop_exit_value(
        self,
        branch_mnemonic: str,
        compare_mnemonic: str,
        rhs_operand: str,
        reg_values: Dict[str, int],
        width: int,
    ) -> Optional[int]:
        mask = self._width_mask(width)
        rhs_reg = self._reg_name(rhs_operand)
        rhs_value = reg_values.get(rhs_reg) if rhs_reg else self._parse_immediate(rhs_operand)

        if compare_mnemonic in {"CMP", "CMN"}:
            if rhs_value is None:
                return None
            rhs_value &= mask
            if branch_mnemonic == "BEQ":
                return (rhs_value + 1) & mask
            if branch_mnemonic == "BNE":
                return rhs_value
            if branch_mnemonic in {"BGT", "BHI"}:
                return rhs_value
            if branch_mnemonic in {"BLT", "BLO"}:
                return rhs_value
            if branch_mnemonic in {"BGE", "BHS"}:
                return (rhs_value - 1) & mask
            if branch_mnemonic in {"BLE", "BLS"}:
                return (rhs_value + 1) & mask
            return None

        if compare_mnemonic == "TST":
            rhs_value = self._parse_immediate(rhs_operand) if rhs_value is None else rhs_value
            if rhs_value is None:
                return None
            rhs_value &= mask
            if branch_mnemonic == "BEQ":
                return rhs_value
            if branch_mnemonic == "BNE":
                return 0
        return None

    def _infer_cbz_cbnz_value(
        self,
        branch_mnemonic: str,
        desired_taken: bool,
        source_reg: Optional[str],
        reg_values: Dict[str, int],
        width: int,
    ) -> Optional[int]:
        mask = self._width_mask(width)
        want_zero = desired_taken if branch_mnemonic == "CBZ" else not desired_taken
        if want_zero:
            return 0

        current = reg_values.get(source_reg or "", 0) & mask
        return current if current != 0 else 1

    def _infer_branch_condition_value(
        self,
        branch_mnemonic: str,
        compare_mnemonic: str,
        source_reg: Optional[str],
        compare_parts: List[str],
        reg_values: Dict[str, int],
        width: int,
        desired_taken: bool,
    ) -> Optional[int]:
        mask = self._width_mask(width)
        source_reg = source_reg or ""
        source_on_left = self._reg_name(compare_parts[0]) == source_reg
        rhs_token = compare_parts[1] if source_on_left and len(compare_parts) >= 2 else compare_parts[0]
        rhs_reg = self._reg_name(rhs_token)
        rhs_value = reg_values.get(rhs_reg) if rhs_reg else self._parse_immediate(rhs_token)
        lhs_current = reg_values.get(source_reg)

        if compare_mnemonic == "CMP":
            if rhs_value is None:
                return None
            rhs_value &= mask
            if branch_mnemonic == "BEQ":
                return rhs_value if desired_taken else (rhs_value + 1) & mask
            if branch_mnemonic == "BNE":
                return (rhs_value + 1) & mask if desired_taken else rhs_value
            if not source_on_left:
                if branch_mnemonic in {"BHI", "BGT"}:
                    if desired_taken:
                        return None if rhs_value == 0 else (rhs_value - 1) & mask
                    return rhs_value
                if branch_mnemonic in {"BHS", "BGE"}:
                    if desired_taken:
                        return rhs_value
                    return None if rhs_value == mask else (rhs_value + 1) & mask
                if branch_mnemonic in {"BLO", "BLT"}:
                    if desired_taken:
                        return None if rhs_value == mask else (rhs_value + 1) & mask
                    return rhs_value
                if branch_mnemonic in {"BLS", "BLE"}:
                    if desired_taken:
                        return rhs_value
                    return None if rhs_value == 0 else (rhs_value - 1) & mask
                return None
            if branch_mnemonic in {"BHI", "BGT"}:
                return (rhs_value + 1) & mask if desired_taken else rhs_value
            if branch_mnemonic in {"BHS", "BGE"}:
                return rhs_value if desired_taken else (rhs_value - 1) & mask
            if branch_mnemonic in {"BLO", "BLT"}:
                return (rhs_value - 1) & mask if desired_taken else rhs_value
            if branch_mnemonic in {"BLS", "BLE"}:
                return rhs_value if desired_taken else (rhs_value + 1) & mask
            return None

        if compare_mnemonic == "TEQ":
            if rhs_value is None:
                return None
            rhs_value &= mask
            if branch_mnemonic == "BEQ":
                return rhs_value if desired_taken else (rhs_value + 1) & mask
            if branch_mnemonic == "BNE":
                return (rhs_value + 1) & mask if desired_taken else rhs_value
            return None

        if compare_mnemonic == "TST":
            if rhs_value is None:
                return None
            rhs_value &= mask
            want_zero = desired_taken if branch_mnemonic == "BEQ" else not desired_taken
            if want_zero:
                return 0
            if rhs_value != 0:
                return rhs_value
            current = lhs_current if lhs_current is not None else 1
            return (current | 1) & mask

        if compare_mnemonic == "CMN":
            if rhs_value is None:
                return None
            rhs_value &= mask
            zero_value = (-rhs_value) & mask
            if branch_mnemonic == "BEQ":
                return zero_value if desired_taken else (zero_value + 1) & mask
            if branch_mnemonic == "BNE":
                return (zero_value + 1) & mask if desired_taken else zero_value
        return None

    def _incoming_refs_to_bb(self, bb_addr: int) -> List[Dict[str, object]]:
        refs: List[Dict[str, object]] = []
        for ref in self.predecessor_index.get(bb_addr, []):
            if ref["bb"] == bb_addr and self._normalize_mnemonic(ref.get("mnemonic")) == "B":
                target = self._parse_branch_target(str(ref.get("operands", "")))
                if self._resolve_bb_start(target) == bb_addr:
                    continue
            refs.append(ref)
        return refs

    def _find_controlling_conditional_in_bb(self, bb_addr: int) -> Optional[Dict[str, object]]:
        instructions = self.static_bbs.get(bb_addr, [])
        if not instructions:
            return None
        for insn in reversed(instructions):
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
            if self._is_conditional_branch_mnemonic(mnemonic):
                return {
                    "bb": bb_addr,
                    "address": self._parse_hex(insn.get("address")) or bb_addr,
                    "mnemonic": mnemonic,
                    "operands": str(insn.get("operands", "")),
                    "kind": "intra_bb",
                }
        return None

    def _format_ref_instruction(self, ref: Dict[str, object]) -> str:
        return f"0x{ref['address']:08x}: {ref['mnemonic']} {ref['operands']}".rstrip()

    def _format_bb_chain(self, chain: List[int]) -> str:
        return " -> ".join(f"0x{bb:08x}" for bb in chain)

    def _is_wrapper_to_terminal_sink(self, bb_addr: int, sink_bb: int) -> bool:
        instructions = self.static_bbs.get(bb_addr, [])
        if not instructions:
            return False

        calls_sink = False
        nested_calls = 0
        for insn in instructions:
            mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
            target = self._resolve_bb_start(self._parse_branch_target(str(insn.get("operands", ""))))
            if mnemonic in {"BL", "BLX"}:
                nested_calls += 1
                if target == sink_bb:
                    calls_sink = True
        return calls_sink and nested_calls >= 2

    def _width_mask(self, width: int) -> int:
        if width <= 1:
            return 0xFF
        if width == 2:
            return 0xFFFF
        return 0xFFFFFFFF

    def _is_conditional_branch_mnemonic(self, mnemonic: str) -> bool:
        normalized = self._normalize_mnemonic(mnemonic)
        return normalized in {
            "BEQ", "BNE", "BGT", "BLT", "BGE", "BLE",
            "BHI", "BLO", "BHS", "BLS", "BMI", "BPL",
            "BVS", "BVC", "CBZ", "CBNZ",
        }

    def _is_load_instruction(self, insn: Dict) -> bool:
        mnemonic = self._normalize_mnemonic(insn.get("mnemonic"))
        return mnemonic.startswith("LDR") or mnemonic in {"LDM", "LDMIA", "POP"}

    def _render_bb_context(self, bb_addr: int, highlight_address: Optional[int] = None, limit: int = 12) -> str:
        instructions = self.static_bbs.get(bb_addr, [])
        if not instructions:
            return ""

        lines: List[str] = []
        for insn in instructions[:limit]:
            address = self._parse_hex(insn.get("address"))
            prefix = ">>" if highlight_address is not None and address == highlight_address else "  "
            if address is not None:
                lines.append(
                    f"{prefix} 0x{address:08x}: {insn.get('mnemonic', '')} {insn.get('operands', '')}".rstrip()
                )
            else:
                lines.append(f"{prefix} {insn.get('mnemonic', '')} {insn.get('operands', '')}".rstrip())
        return "\n".join(lines)

    def _format_instruction(self, insn: Optional[Dict]) -> str:
        if not insn:
            return ""
        address = self._parse_hex(insn.get("address"))
        prefix = f"0x{address:08x}: " if address is not None else ""
        mnemonic = insn.get("mnemonic", "")
        operands = insn.get("operands", "")
        return f"{prefix}{mnemonic} {operands}".strip()

    def _format_optional_hex(self, value) -> Optional[str]:
        parsed = self._parse_hex(value)
        if parsed is None:
            return None
        return f"0x{parsed:08x}"

    def _parse_hex(self, value) -> Optional[int]:
        """解析十六进制值"""
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            match = re.search(r"-?0x[0-9a-fA-F]+|-?\d+", value)
            if not match:
                return None
            token = match.group(0)
            if token.startswith("0x") or token.startswith("0X") or token.startswith("-0x") or token.startswith("-0X"):
                return int(token, 16)
            try:
                return int(token)
            except ValueError:
                return None
        return None

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        success_rate = (self.successful_analyses / self.total_analyses * 100
                       if self.total_analyses > 0 else 0)

        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate": f"{success_rate:.1f}%",
            "llm_enabled": self.client is not None
        }
