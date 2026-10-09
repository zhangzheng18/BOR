#!/usr/bin/env python3
"""
LLM Guide - 大模型约束推断模块

功能:
1. 分析分支条件
2. 使用 LLM 推断合理的 MMIO 约束
3. 生成能让路径继续执行的约束值
"""

import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, Optional, List, Tuple, Set
import json
import yaml

try:
    from ..runtime_bootstrap import bootstrap_runtime_dependencies
    from ..llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        create_openai_compatible_client,
        extract_json_string_field,
        extract_response_reasoning,
        extract_response_text,
        extract_unquoted_field_text,
        parse_json_object,
        repair_json_like_text,
        salvage_named_fields,
    )
    from ..artifact_io import atomic_json_dump
except ImportError:
    from lsgemu.runtime_bootstrap import bootstrap_runtime_dependencies
    from lsgemu.llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        create_openai_compatible_client,
        extract_json_string_field,
        extract_response_reasoning,
        extract_response_text,
        extract_unquoted_field_text,
        parse_json_object,
        repair_json_like_text,
        salvage_named_fields,
    )
    from lsgemu.artifact_io import atomic_json_dump

bootstrap_runtime_dependencies()

logger = logging.getLogger(__name__)


class LLMGuide:
    """LLM 约束推断器"""

    _openai_missing_warned: bool = False
    _client_init_failure_warned: Set[str] = set()

    def __init__(
        self,
        static_bbs,
        use_llm: bool = False,
        llm_config: Optional[Dict] = None,
        instruction_lookup: Optional[Dict[int, Dict]] = None,
        compare_lookup: Optional[Dict[int, Dict]] = None,
    ):
        """
        初始化

        Args:
            static_bbs: 静态基本块
            use_llm: 是否使用真实的 LLM（默认使用规则）
            llm_config: LLM配置（从LLM.yaml加载）
            instruction_lookup: 指令地址索引（可选）
            compare_lookup: 分支对应比较指令索引（可选）
        """
        self.static_bbs = static_bbs
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.instruction_lookup = instruction_lookup or self._build_instruction_lookup()
        self.compare_lookup = compare_lookup or self._build_compare_lookup()
        self.instruction_to_bb = self._build_instruction_to_bb()
        self.inference_cache: Dict[Tuple[int, str, bool, int], int] = {}
        self.inference_cache_meta: Dict[Tuple[int, str, bool, int], Dict[str, Any]] = {}
        self.last_inference_metadata: Optional[Dict[str, Any]] = None

        # 推断历史
        self.inference_history: List[Dict] = []

        # 增量 journal：每条推断完成即追加落盘（JSONL，一行一条），
        # 使 SIGTERM/SIGKILL/native 崩溃都不会丢失推断历史。
        # finalize 阶段的 save_inference_history 语义保持不变。
        self._inference_journal_path: Optional[str] = None
        self._inference_journal_handle = None
        self._inference_journal_failed = False
        self._inference_journal_session_id = self._journal_session_id()

        # max_tokens：从配置读取（推理模型的思维链与最终答案共用预算，
        # 默认值必须足够大，否则 content 会被截断成空）
        self.llm_max_tokens = DEFAULT_LLM_MAX_TOKENS
        cfg = dict(llm_config) if isinstance(llm_config, dict) else {}
        nested = cfg.get("llm")
        if isinstance(nested, dict):
            cfg = nested
        try:
            configured_max_tokens = int(cfg.get("max_tokens") or 0)
        except (TypeError, ValueError):
            configured_max_tokens = 0
        if configured_max_tokens > 0:
            self.llm_max_tokens = configured_max_tokens

        # 初始化LLM客户端
        if use_llm and llm_config:
            self._init_llm_client()

    def _build_instruction_lookup(self) -> Dict[int, Dict]:
        """构建指令地址索引"""
        lookup: Dict[int, Dict] = {}
        for instructions in self.static_bbs.values():
            for insn in instructions:
                lookup[insn['address']] = insn
        return lookup

    def _build_compare_lookup(self) -> Dict[int, Dict]:
        """构建分支到比较指令的索引"""
        lookup: Dict[int, Dict] = {}
        for instructions in self.static_bbs.values():
            if not instructions:
                continue

            branch_insn = instructions[-1]
            for previous in reversed(instructions[max(0, len(instructions) - 11):-1]):
                if self._normalize_mnemonic(previous['mnemonic']) in ['CMP', 'CMN', 'TST', 'TEQ']:
                    lookup[branch_insn['address']] = previous
                    break

        return lookup

    def _build_instruction_to_bb(self) -> Dict[int, int]:
        """构建指令地址到基本块起始地址的索引"""
        lookup: Dict[int, int] = {}
        for bb_addr, instructions in self.static_bbs.items():
            for insn in instructions:
                lookup[insn['address']] = bb_addr
        return lookup

    def _init_llm_client(self):
        """初始化LLM客户端"""
        client, model, backend = create_openai_compatible_client(
            self.llm_config,
            logger=logger,
            warn_key="llmguide",
            default_model="qwen-plus",
        )
        self.llm_client = client
        self.llm_model = model
        if client is None:
            self.use_llm = False
            return
        logger.info(f"[LLMGuide] LLM客户端初始化成功: {self.llm_model} backend={backend}")

    def infer_constraint(self, branch_pc: int, branch_condition: str,
                        target_direction: bool, mmio_addr: int) -> Optional[int]:
        """
        推断约束值

        Args:
            branch_pc: 分支指令 PC
            branch_condition: 分支条件 (BEQ, BNE, BGT, etc.)
            target_direction: 目标方向 (True=taken, False=not-taken)
            mmio_addr: MMIO 地址

        Returns:
            推断的 MMIO 值
        """
        condition = self._normalize_mnemonic(branch_condition)
        cache_key = (branch_pc, condition, target_direction, mmio_addr)
        if cache_key in self.inference_cache:
            value = self.inference_cache[cache_key]
            self.last_inference_metadata = self._metadata_for_cache_hit(
                branch_pc,
                condition,
                target_direction,
                mmio_addr,
                value,
                self.inference_cache_meta.get(cache_key),
            )
            logger.debug(f"[LLMGuide] 缓存命中: {hex(branch_pc)} {condition} -> {hex(value)}")
            return value

        force_llm = (
            self.use_llm
            and os.environ.get("LSGEMU_FORCE_LLM_BRANCH_INFERENCE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
            and not self._llm_call_budget_exhausted()
        )
        history_start = len(self.inference_history)
        if force_llm:
            value = self._infer_with_llm(branch_pc, condition, target_direction, mmio_addr)
        elif self._should_use_rule_first(branch_pc, condition):
            value = self._infer_with_rules(branch_pc, condition, target_direction, mmio_addr)
        elif self.use_llm and not self._llm_call_budget_exhausted():
            value = self._infer_with_llm(branch_pc, condition, target_direction, mmio_addr)
        else:
            value = self._infer_with_rules(branch_pc, condition, target_direction, mmio_addr)

        history_delta = self.inference_history[history_start:]
        inference_meta = self._summarize_inference_records(
            history_delta,
            branch_pc,
            condition,
            target_direction,
            mmio_addr,
            value,
        )
        self.last_inference_metadata = inference_meta
        if value is not None:
            self.inference_cache[cache_key] = value
            self.inference_cache_meta[cache_key] = inference_meta
        return value

    def infer_rule_constraint(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        mmio_addr: int,
    ) -> Optional[int]:
        """Return the deterministic local candidate without invoking an LLM.

        Path naturalization uses this API before bounded local solving.  An LLM
        alternative is requested only after concrete replay feedback shows that
        deterministic candidates were consumed but insufficient.
        """
        return self._infer_with_rules(
            branch_pc,
            self._normalize_mnemonic(branch_condition),
            bool(target_direction),
            mmio_addr,
        )

    def _llm_call_budget_exhausted(self) -> bool:
        try:
            limit = int(float(os.environ.get("LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS") or 0))
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            return False
        used = 0
        for item in self.inference_history:
            if not isinstance(item, dict):
                continue
            method = str(item.get("method") or "")
            if method in {"llm", "llm_rejected", "llm_error"} or method.startswith(
                "llm_compound"
            ):
                used += 1
            elif method.startswith("llm_slice") and method != "llm_slice_skipped":
                used += 1
        return used >= limit

    def _summarize_inference_records(
        self,
        records: List[Dict],
        branch_pc: int,
        condition: str,
        target_direction: bool,
        mmio_addr: int,
        value: Optional[int],
    ) -> Dict[str, Any]:
        methods = [str(item.get("method") or "unknown") for item in records if isinstance(item, dict)]
        method_counts: Dict[str, int] = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        if "llm_compound" in method_counts:
            inference_method = "llm_compound"
        elif "llm" in method_counts:
            inference_method = "llm"
        elif "llm_rejected" in method_counts and "rules" in method_counts:
            inference_method = "llm_rejected_then_rules"
        elif "llm_error" in method_counts and "rules" in method_counts:
            inference_method = "llm_error_then_rules"
        elif methods:
            inference_method = methods[-1]
        else:
            inference_method = "unknown"

        def record_root_cause(item: Dict[str, Any]) -> Optional[str]:
            value = item.get("root_cause")
            if value is None:
                value = item.get("llm_root_cause")
            return str(value) if value else None

        def record_recommended_action(item: Dict[str, Any]) -> Optional[str]:
            value = item.get("recommended_action")
            if value is None:
                value = item.get("llm_recommended_action")
            return str(value) if value else None

        root_causes = [
            value
            for item in records
            if isinstance(item, dict)
            for value in [record_root_cause(item)]
            if value
        ]
        recommended_actions = [
            value
            for item in records
            if isinstance(item, dict)
            for value in [record_recommended_action(item)]
            if value
        ]

        return {
            "branch_pc": hex(branch_pc),
            "condition": condition,
            "target_direction": bool(target_direction),
            "mmio_addr": hex(mmio_addr),
            "inferred_value": hex(value & 0xFFFFFFFF) if value is not None else None,
            "inference_method": inference_method,
            "method_counts": method_counts,
            "history_entries": len(methods),
            "actual_llm_calls": sum(
                method_counts.get(name, 0)
                for name in (
                    "llm",
                    "llm_rejected",
                    "llm_error",
                    "llm_compound",
                    "llm_compound_error",
                )
            ),
            "llm_successful_values": method_counts.get("llm", 0),
            "llm_rejected": method_counts.get("llm_rejected", 0),
            "llm_errors": method_counts.get("llm_error", 0),
            "rule_inference_entries": method_counts.get("rules", 0),
            "root_cause_counts": dict(
                Counter(root_causes)
            ),
            "recommended_action_counts": dict(
                Counter(recommended_actions)
            ),
            "latest_root_cause": next(
                (
                    record_root_cause(item)
                    for item in reversed(records)
                    if isinstance(item, dict) and record_root_cause(item)
                ),
                None,
            ),
            "latest_recommended_action": next(
                (
                    record_recommended_action(item)
                    for item in reversed(records)
                    if isinstance(item, dict) and record_recommended_action(item)
                ),
                None,
            ),
            "cache_hit": False,
        }

    def _metadata_for_cache_hit(
        self,
        branch_pc: int,
        condition: str,
        target_direction: bool,
        mmio_addr: int,
        value: int,
        cached_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        meta = dict(cached_meta or {})
        original_method = str(meta.get("inference_method") or "unknown")
        meta.update({
            "branch_pc": hex(branch_pc),
            "condition": condition,
            "target_direction": bool(target_direction),
            "mmio_addr": hex(mmio_addr),
            "inferred_value": hex(value & 0xFFFFFFFF),
            "inference_method": original_method,
            "cached_inference_method": original_method,
            "history_entries": 0,
            "actual_llm_calls": 0,
            "llm_successful_values": 0,
            "llm_rejected": 0,
            "llm_errors": 0,
            "rule_inference_entries": 0,
            "cache_hit": True,
        })
        return meta

    def _should_use_rule_first(self, branch_pc: int, condition: str) -> bool:
        """Use deterministic local semantics for simple flag-setting patterns."""
        if condition in {"CBZ", "CBNZ"}:
            return True
        compare_insn = self._find_compare_instruction(branch_pc)
        if not compare_insn:
            return False
        mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
        masked_compare = self._find_masked_register_compare(branch_pc, compare_insn)
        if mnemonic in {"CMP", "CMN"} and self._extract_compare_value(compare_insn) is not None:
            return True
        if masked_compare is not None and condition in {"BCS", "BHS", "BCC", "BLO", "BLS", "BHI"}:
            return True
        if mnemonic == "TST" and self._extract_bitmask(compare_insn) is not None:
            return condition in {"BEQ", "BNE"}
        if mnemonic == "TEQ" and self._extract_compare_value(compare_insn) is not None:
            return condition in {"BEQ", "BNE"}
        return False

    def _infer_with_rules_core(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
    ) -> Dict[str, Any]:
        """Compute the deterministic local branch solution without recording history."""
        compare_insn = self._find_compare_instruction(branch_pc)
        condition = self._normalize_mnemonic(branch_condition)
        compare_mnemonic = self._normalize_mnemonic(compare_insn['mnemonic']) if compare_insn else None
        compare_value = None
        bitmask = None
        masked_compare = self._find_masked_register_compare(branch_pc, compare_insn) if compare_insn else None

        if condition == 'CBZ':
            value = 0 if target_direction else 1
        elif condition == 'CBNZ':
            value = 1 if target_direction else 0
        elif masked_compare is not None:
            value = self._infer_masked_register_compare_value(masked_compare, condition, target_direction)
        elif compare_insn and compare_mnemonic == 'TST':
            bitmask = self._extract_bitmask(compare_insn)
            if bitmask is not None:
                non_zero_value = self._least_significant_bit(bitmask)
                if condition == 'BNE':
                    value = non_zero_value if target_direction else 0
                elif condition == 'BEQ':
                    value = 0 if target_direction else non_zero_value
                else:
                    value = non_zero_value if target_direction else 0
            else:
                value = self._fallback_unknown_tst_value(condition, target_direction)
        elif compare_insn and compare_mnemonic == 'TEQ':
            compare_value = self._extract_compare_value(compare_insn)
            if compare_value is None:
                value = 1 if target_direction else 0
            elif condition == 'BEQ':
                value = compare_value if target_direction else (compare_value ^ 1)
            elif condition == 'BNE':
                value = (compare_value ^ 1) if target_direction else compare_value
            else:
                value = compare_value if target_direction else (compare_value ^ 1)
        elif compare_insn:
            compare_value = self._extract_compare_value(compare_insn)
            if compare_value is None:
                compare_value = 0

            if condition == 'BEQ':
                value = compare_value if target_direction else (compare_value + 1)
            elif condition == 'BNE':
                value = (compare_value + 1) if target_direction else compare_value
            elif condition in ['BGT', 'BHI']:
                value = (compare_value + 1) if target_direction else compare_value
            elif condition in ['BLT', 'BLO']:
                value = (compare_value - 1) if target_direction else compare_value
            elif condition in ['BGE', 'BHS']:
                value = compare_value if target_direction else (compare_value - 1)
            elif condition in ['BLE', 'BLS']:
                value = compare_value if target_direction else (compare_value + 1)
            else:
                value = 1 if target_direction else 0
        else:
            value = 1 if target_direction else 0

        value = self._sanitize_mmio_value(value)
        return {
            "value": value,
            "condition": condition,
            "compare_insn": compare_insn,
            "compare_mnemonic": compare_mnemonic,
            "compare_value": compare_value,
            "bitmask": bitmask,
            "masked_compare": masked_compare,
            "branch_context": self._get_branch_context(branch_pc),
        }

    def _infer_with_rules(self, branch_pc: int, branch_condition: str,
                         target_direction: bool, mmio_addr: int) -> Optional[int]:
        """
        基于规则的推断（快速，不需要 LLM）

        规则:
        - BEQ (相等): taken=0, not-taken=1
        - BNE (不等): taken=1, not-taken=0
        - BGT (大于): taken=0x100, not-taken=0
        - BLT (小于): taken=0, not-taken=0x100
        - BHI (无符号大于): taken=0x100, not-taken=0
        - BLO (无符号小于): taken=0, not-taken=0x100
        """
        core = self._infer_with_rules_core(branch_pc, branch_condition, target_direction)
        compare_insn = core["compare_insn"]
        condition = core["condition"]
        compare_value = core["compare_value"]
        bitmask = core["bitmask"]
        masked_compare = core["masked_compare"]
        value = core["value"]
        branch_context = core["branch_context"]

        # 记录推断
        self._record_inference({
            'branch_pc': hex(branch_pc),
            'condition': condition,
            'target_direction': target_direction,
            'mmio_addr': hex(mmio_addr),
            'compare_instruction': self._format_instruction(compare_insn) if compare_insn else None,
            'compare_value': hex(compare_value) if compare_value is not None else None,
            'bitmask': hex(bitmask) if bitmask is not None else None,
            'masked_compare': masked_compare,
            'branch_context': branch_context,
            'inferred_value': hex(value),
            'method': 'rules'
        })

        logger.debug(f"[LLMGuide] 推断: {condition} @ {hex(branch_pc)} -> MMIO[{hex(mmio_addr)}] = {hex(value)}")

        return value

    def _infer_with_llm(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        mmio_addr: int,
        *,
        rule_candidate: Optional[int] = None,
        failure_context: Optional[Dict[str, Any]] = None,
        avoid_values: Optional[Iterable[int]] = None,
        prompt_purpose: str = "initial",
        allow_replay_validation_override: bool = False,
    ) -> Optional[int]:
        """
        基于 LLM 的推断
        """
        rule_core = self._infer_with_rules_core(branch_pc, branch_condition, target_direction)
        compare_insn = rule_core["compare_insn"]
        branch_context = rule_core["branch_context"]
        compare_info = self._format_instruction(compare_insn) if compare_insn else "未找到"
        compare_mnemonic = self._normalize_mnemonic(compare_insn['mnemonic']) if compare_insn else None
        compare_value = rule_core["compare_value"]
        bitmask = rule_core["bitmask"]
        if compare_insn and compare_mnemonic == 'TST' and bitmask is None:
            bitmask = self._extract_bitmask(compare_insn)
        if rule_candidate is None:
            rule_candidate = rule_core["value"]
        avoid_list = []
        for item in avoid_values or []:
            try:
                avoid_list.append(int(item) & 0xFFFFFFFF)
            except Exception:
                continue
        avoid_text = ", ".join(f"0x{item:08x}" for item in sorted(set(avoid_list))) or "无"
        failure_text = self._format_failure_context(failure_context)
        value_hints = self._semantic_value_hints(
            branch_condition,
            target_direction,
            compare_insn,
            rule_candidate,
            avoid_list,
        )
        goal_summary = self._describe_branch_goal(branch_condition, target_direction, compare_insn)

        try:
            prompt = f"""你是一个世界级的固件分析和嵌入式系统专家。

任务：分析下面的 ARM Cortex-M 汇编分支，推断一个合理的 32 位无符号 MMIO 值，使分支按照目标方向执行。
你不能强行修改 PC，只能通过设置 MMIO[{hex(mmio_addr)}] 的读取结果来影响后续比较/测试和分支。

分支信息：
- 分支 PC: {hex(branch_pc)}
- 分支指令: {branch_condition}
- 目标方向: {'taken (跳转)' if target_direction else 'not-taken (不跳转)'}
- 依赖的 MMIO 地址: {hex(mmio_addr)}
- 比较/测试指令: {compare_info}
- 本地语义目标: {goal_summary}
- 本地规则候选值: {hex(rule_candidate & 0xFFFFFFFF) if rule_candidate is not None else '无'}
- 应尽量避免重复的失败值: {avoid_text}
- 比较立即数: {hex(compare_value) if compare_value is not None else '未知/非立即数'}
- 位掩码: {hex(bitmask) if bitmask is not None else '无'}
- 调用目的: {prompt_purpose}
- 上次 replay/候选反馈: {failure_text}
- 满足分支语义的值提示: {value_hints}

分支前后的汇编代码上下文：
```asm
{branch_context}
```

分析要求：
1. 先识别最近的 load/compare/test/branch 链条，推断这段代码在等待什么硬件状态，例如 PLL 就绪、串口发送完成、SPI 传输完成、GPIO 状态等。
2. 明确这是数值比较还是位测试：
   - `TST Rn, #mask` 表示测试 `(value & mask)` 是否为 0。
   - `TEQ Rn, #imm` 表示测试 `(value ^ imm)` 是否为 0，本质上是“value 是否等于 imm”。
   - `CMP/CMN` 需要根据比较常量和分支条件判断大小关系/相等关系。
3. 只能输出一个 32 位无符号值；不要返回负数，不要超过 `0xffffffff`。
4. 优先返回最简单、最小、最可解释的值，例如 `0x0`、`0x1`、某个 bit mask、比较常量或常量加一。
5. 如果本地规则候选已经是该分支语义的唯一解，必须返回同一个值，并在 `needs_state_fix` 说明 replay 失败更可能是 read_pc/状态/路径前缀问题。
6. 如果存在多个满足分支语义的值，且失败反馈显示旧值已被读取但没有改变分支结果，可以返回另一个满足语义的值。
7. 如果失败反馈显示 MMIO 地址/值已命中但分支仍未变化，优先判断 root cause：
   - 读 PC 不一致或候选没有在分支前命中：`wrong_read_pc` + `retry_closer_snapshot` 或 `retry_root_provenance`。
   - 同一 BB 可达但寄存器/RAM/调用栈状态不一致：`state_prefix_mismatch` + `retry_root_provenance`。
   - 需要先写控制寄存器再读状态寄存器，或状态依赖访问序列：`needs_transaction_model` + `infer_transaction_sequence`。
   - 比较寄存器来自 RAM/栈/软件状态而不是该 MMIO：`not_mmio_dependency` + `inspect_non_mmio_dependency`。
   - 本地规则值是唯一满足语义的值：`unique_semantic_value`，不要为了变化而返回错误值。
8. `analysis` 必须明确说明“为什么这个值会让该分支朝目标方向执行”，以及 replay 失败时更可能调度哪个动作。
9. `analysis` 和 `why_value_satisfies_branch` 必须是一行纯文本，不要包含未转义的双引号，不要换行。

只按 JSON 返回，不要返回其他内容：
{{ 
  "analysis": "简要说明推断依据",
  "mmio_value": "0xXXXXXXXX",
  "confidence": 0.0,
  "why_value_satisfies_branch": "一句话说明",
  "needs_state_fix": false,
  "root_cause": "try_alternative_value|wrong_read_pc|state_prefix_mismatch|needs_transaction_model|not_mmio_dependency|unique_semantic_value|unknown",
  "recommended_action": "use_value|retry_closer_snapshot|retry_root_provenance|infer_transaction_sequence|skip_value_change|inspect_non_mmio_dependency"
}}"""

            repair_prompt = f"""你上一次的输出没有严格满足 JSON 要求。请只返回一个单行 JSON 对象，不要 markdown，不要代码块，不要额外解释。

硬性要求：
1. `analysis` 和 `why_value_satisfies_branch` 只能是单行纯文本，禁止未转义双引号。
2. `mmio_value` 必须是形如 `0x1234abcd` 的 32 位无符号十六进制字符串。
3. 如果不确定，也必须返回一个最简单且可解释的值，例如 `0x0`、`0x1`、bit mask 或比较常量。
4. `root_cause` 和 `recommended_action` 只能从指定枚举里选择；如果无法判断，写 `unknown` / `use_value`。

返回格式：
{{"analysis":"...","mmio_value":"0xXXXXXXXX","confidence":0.0,"why_value_satisfies_branch":"...","needs_state_fix":false,"root_cause":"unknown","recommended_action":"use_value"}}"""

            response = self._call_llm_json(prompt, repair_prompt=repair_prompt)
            result_str = extract_response_text(response)
            if not result_str:
                # 推理模型预算耗尽时 content 为空且思维链里可能没有完整的
                # 平衡 JSON；截断的思维链通常已写出结论值，最后从
                # reasoning_content 里按字段名打捞一次，避免白白回退规则。
                salvaged = self._salvage_response_fields(
                    extract_response_reasoning(response)
                )
                if salvaged:
                    result_str = json.dumps(salvaged, ensure_ascii=False)
                    logger.info(
                        "[LLMGuide] content为空，已从截断的reasoning_content打捞出字段: %s",
                        sorted(salvaged.keys()),
                    )
            logger.debug(f"[LLMGuide] LLM 回复:\n{result_str}")

            result = self._parse_json_response(result_str)
            value = self._parse_mmio_value(result.get('mmio_value'))
            if value is None:
                raise ValueError(f"LLM返回的mmio_value无效: {result.get('mmio_value')!r}")
            if value < 0 or value > 0xFFFFFFFF:
                raise ValueError(f"LLM返回的mmio_value超出32位无符号范围: {result.get('mmio_value')!r}")

            value = self._sanitize_mmio_value(value)
            analysis = result.get('analysis', '')
            root_cause = self._normalize_llm_enum(
                result.get("root_cause"),
                {
                    "try_alternative_value",
                    "wrong_read_pc",
                    "state_prefix_mismatch",
                    "needs_transaction_model",
                    "not_mmio_dependency",
                    "unique_semantic_value",
                    "unknown",
                },
                "unknown",
            )
            recommended_action = self._normalize_llm_enum(
                result.get("recommended_action"),
                {
                    "use_value",
                    "retry_closer_snapshot",
                    "retry_root_provenance",
                    "infer_transaction_sequence",
                    "skip_value_change",
                    "inspect_non_mmio_dependency",
                },
                "use_value",
            )
            local_validation = self._validate_inferred_value(
                value,
                branch_condition,
                target_direction,
                compare_insn,
            )
            if compare_mnemonic == 'TST' and bitmask is None:
                value = self._fallback_unknown_tst_value(branch_condition, target_direction)
                local_validation = None
            if local_validation is False and not allow_replay_validation_override:
                self._record_inference({
                    'branch_pc': hex(branch_pc),
                    'condition': branch_condition.upper(),
                    'target_direction': target_direction,
                    'mmio_addr': hex(mmio_addr),
                    'compare_instruction': compare_info if compare_insn else None,
                    'compare_value': hex(compare_value) if compare_value is not None else None,
                    'bitmask': hex(bitmask) if bitmask is not None else None,
                    'branch_context': branch_context,
                    'llm_analysis': analysis,
                    'llm_confidence': result.get('confidence'),
                    'llm_why_value_satisfies_branch': result.get('why_value_satisfies_branch'),
                    'llm_needs_state_fix': result.get('needs_state_fix'),
                    'llm_root_cause': root_cause,
                    'llm_recommended_action': recommended_action,
                    'local_validation': local_validation,
                    'llm_full_response': result_str,
                    'inferred_value': hex(value),
                    'rule_candidate': hex(rule_candidate & 0xFFFFFFFF) if rule_candidate is not None else None,
                    'avoid_values': [hex(item) for item in sorted(set(avoid_list))],
                    'failure_context': failure_context,
                    'prompt_purpose': prompt_purpose,
                    'allow_replay_validation_override': bool(
                        allow_replay_validation_override
                    ),
                    'method': 'llm_rejected'
                })
                raise ValueError(
                    f"LLM返回值未通过本地语义校验: value={hex(value)} "
                    f"branch={branch_condition} target={target_direction}"
                )

            # 记录推断
            self._record_inference({
                'branch_pc': hex(branch_pc),
                'condition': branch_condition.upper(),
                'target_direction': target_direction,
                'mmio_addr': hex(mmio_addr),
                'compare_instruction': compare_info if compare_insn else None,
                'compare_value': hex(compare_value) if compare_value is not None else None,
                'bitmask': hex(bitmask) if bitmask is not None else None,
                'branch_context': branch_context,
                'llm_analysis': analysis,
                'llm_confidence': result.get('confidence'),
                'llm_why_value_satisfies_branch': result.get('why_value_satisfies_branch'),
                'llm_needs_state_fix': result.get('needs_state_fix'),
                'llm_root_cause': root_cause,
                'llm_recommended_action': recommended_action,
                'local_validation': local_validation,
                'llm_full_response': result_str,
                'inferred_value': hex(value),
                'rule_candidate': hex(rule_candidate & 0xFFFFFFFF) if rule_candidate is not None else None,
                'avoid_values': [hex(item) for item in sorted(set(avoid_list))],
                'failure_context': failure_context,
                'prompt_purpose': prompt_purpose,
                'allow_replay_validation_override': bool(
                    allow_replay_validation_override
                ),
                'requires_force_free_replay_validation': bool(
                    local_validation is False
                    and allow_replay_validation_override
                ),
                'method': 'llm'
            })

            logger.info(
                f"[LLMGuide] LLM推断: {branch_condition} @ {hex(branch_pc)} "
                f"-> MMIO[{hex(mmio_addr)}] = {hex(value)} (分析: {analysis})"
            )
            return value

        except Exception as e:
            self._record_inference({
                'branch_pc': hex(branch_pc),
                'condition': branch_condition.upper(),
                'target_direction': target_direction,
                'mmio_addr': hex(mmio_addr),
                'compare_instruction': compare_info if compare_insn else None,
                'compare_value': hex(compare_value) if compare_value is not None else None,
                'bitmask': hex(bitmask) if bitmask is not None else None,
                'branch_context': branch_context,
                'llm_error': str(e),
                'rule_candidate': hex(rule_candidate & 0xFFFFFFFF) if rule_candidate is not None else None,
                'avoid_values': [hex(item) for item in sorted(set(avoid_list))],
                'failure_context': failure_context,
                'prompt_purpose': prompt_purpose,
                'allow_replay_validation_override': bool(
                    allow_replay_validation_override
                ),
                'method': 'llm_error'
            })
            logger.warning(f"[LLMGuide] LLM推断失败: {e}，回退到规则推断")
            return self._infer_with_rules(branch_pc, branch_condition, target_direction, mmio_addr)

    def infer_alternative_constraint(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        mmio_addr: int,
        *,
        previous_value: Optional[int] = None,
        failure_context: Optional[Dict[str, Any]] = None,
        avoid_values: Optional[Iterable[int]] = None,
        allow_replay_validation_override: bool = False,
    ) -> Optional[int]:
        """Ask the LLM for a replay-safe alternative without using the rule cache."""
        if not self.use_llm or self._llm_call_budget_exhausted():
            self.last_inference_metadata = {
                "branch_pc": hex(branch_pc),
                "condition": self._normalize_mnemonic(branch_condition),
                "target_direction": bool(target_direction),
                "mmio_addr": hex(mmio_addr),
                "inferred_value": None,
                "inference_method": "llm_alternative_skipped",
                "history_entries": 0,
                "actual_llm_calls": 0,
                "llm_successful_values": 0,
                "llm_rejected": 0,
                "llm_errors": 0,
                "rule_inference_entries": 0,
                "cache_hit": False,
            }
            return None

        avoid = list(avoid_values or [])
        if previous_value is not None:
            avoid.append(int(previous_value) & 0xFFFFFFFF)
        history_start = len(self.inference_history)
        value = self._infer_with_llm(
            branch_pc,
            branch_condition,
            target_direction,
            mmio_addr,
            rule_candidate=previous_value,
            failure_context=failure_context,
            avoid_values=avoid,
            prompt_purpose="alternative_after_rule_or_replay_feedback",
            allow_replay_validation_override=allow_replay_validation_override,
        )
        history_delta = self.inference_history[history_start:]
        self.last_inference_metadata = self._summarize_inference_records(
            history_delta,
            branch_pc,
            self._normalize_mnemonic(branch_condition),
            target_direction,
            mmio_addr,
            value,
        )
        has_successful_llm_value = any(
            isinstance(item, dict) and str(item.get("method") or "") == "llm"
            for item in history_delta
        )
        if value is not None and has_successful_llm_value:
            self.last_inference_metadata["inference_method"] = "llm_alternative"
            return value
        return None

    def infer_compound_alternative_constraints(
        self,
        branch_pc: int,
        branch_condition: str,
        target_direction: bool,
        sites: Iterable[Dict[str, Any]],
        *,
        failure_context: Optional[Dict[str, Any]] = None,
        max_assignments: int = 4,
    ) -> List[Dict[str, int]]:
        """Generate a bounded multi-input hypothesis for replay validation.

        This is deliberately a *hypothesis* API, not a solver API.  The model
        may choose only from the caller-provided, runtime-observed input sites;
        it cannot invent an address, read PC, occurrence, branch direction, or
        control-flow edge.  The caller must still perform concrete force-free
        replay before accepting the result.
        """
        if not self.use_llm or self._llm_call_budget_exhausted():
            return []

        normalized_sites: List[Dict[str, Any]] = []
        for index, raw_site in enumerate(list(sites or [])):
            if not isinstance(raw_site, dict):
                continue
            if raw_site.get("externally_controllable") is False:
                continue
            try:
                address = int(raw_site.get("address")) & 0xFFFFFFFF
            except (TypeError, ValueError):
                continue
            try:
                width = max(1, min(32, int(raw_site.get("width") or 32)))
            except (TypeError, ValueError):
                width = 32
            mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
            try:
                current_value = int(raw_site.get("current_value", 0) or 0) & mask
            except (TypeError, ValueError):
                current_value = 0
            failed_values: List[int] = []
            for raw_value in list(raw_site.get("failed_values", []) or []):
                parsed = self._parse_mmio_value(raw_value)
                if parsed is not None and 0 <= parsed <= mask:
                    failed_values.append(int(parsed))
            normalized_sites.append({
                "site_index": len(normalized_sites),
                "source_index": index,
                "kind": str(raw_site.get("kind") or raw_site.get("type") or "input").lower(),
                "address": address,
                "read_pc": raw_site.get("read_pc"),
                "occurrence": raw_site.get("occurrence"),
                "width": width,
                "current_value": current_value,
                "failed_values": sorted(set(failed_values)),
                "dependency_register": str(raw_site.get("dependency_register") or ""),
                "dependency_expression": str(raw_site.get("dependency_expression") or ""),
                "dependency_group": str(raw_site.get("dependency_group") or ""),
            })
        if len(normalized_sites) < 2:
            return []

        limit = max(1, min(8, int(max_assignments or 4)))
        site_text = json.dumps(normalized_sites, ensure_ascii=False, sort_keys=True)
        failure_text = self._format_failure_context(failure_context)
        condition = self._normalize_mnemonic(branch_condition)
        try:
            prompt = f"""你是负责嵌入式固件具体回放的分析器。

目标：为 ARM Cortex-M 分支 {hex(int(branch_pc) & 0xffffffff)} ({condition}) 生成一个有限的多输入候选，使其朝 {'taken' if target_direction else 'not-taken'} 方向执行。

这不是控制流强制，也不是最终结论。你只能从下面已经由动态执行观察到的 input sites 中选择，不能新增地址、read_pc、occurrence 或修改寄存器/PC。多字节字段、checksum、协议前缀和状态事务可以同时修改多个 site，但最多修改 {limit} 个。

允许的 sites：
{site_text}

最近的具体 replay 反馈：{failure_text}

规则：
1. 只返回 site_index 和 value；site_index 必须来自列表。
2. value 必须是无符号整数或十六进制，且符合该 site 的 width。
3. 不要返回列表中未出现的 site，也不要返回“强制分支”“修改 PC”等操作。
4. 尽量少改 site；保留协议字段之间可能存在的 checksum/长度关系。
5. 这是候选假设，必须由后续无强制具体回放验证。

严格返回单行 JSON：
{{"assignments":[{{"site_index":0,"value":"0x00"}}],"analysis":"...","confidence":0.0}}"""
            repair_prompt = """只返回单行 JSON。assignments 必须是对象数组，每个对象只包含 site_index 和 value；不要使用未提供的 site_index，不要输出 markdown 或解释文本。"""
            response = self._call_llm_json(prompt, repair_prompt=repair_prompt)
            result_str = extract_response_text(response)
            result = self._parse_json_response(result_str)
            raw_assignments = result.get("assignments")
            if isinstance(raw_assignments, str):
                raw_assignments = self._parse_json_response(raw_assignments).get("assignments")
            if not isinstance(raw_assignments, list):
                raise ValueError("assignments must be a JSON array")

            by_index = {
                int(item["site_index"]): item for item in normalized_sites
            }
            assignments: List[Dict[str, int]] = []
            seen_indices: Set[int] = set()
            for raw_assignment in raw_assignments:
                if not isinstance(raw_assignment, dict):
                    raise ValueError("assignment must be an object")
                try:
                    site_index = int(raw_assignment.get("site_index"))
                except (TypeError, ValueError):
                    raise ValueError("invalid site_index")
                if site_index not in by_index or site_index in seen_indices:
                    raise ValueError("site_index is not unique or not allowed")
                site = by_index[site_index]
                value = self._parse_mmio_value(raw_assignment.get("value"))
                if value is None:
                    raise ValueError("invalid assignment value")
                width = int(site["width"])
                mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
                if value < 0 or value > mask:
                    raise ValueError("assignment value exceeds site width")
                value &= mask
                seen_indices.add(site_index)
                if value == int(site["current_value"]):
                    continue
                if value in set(site["failed_values"]):
                    continue
                assignments.append({"site_index": site_index, "value": value})
                if len(assignments) > limit:
                    raise ValueError("too many assignments")
            if not assignments:
                raise ValueError("no changed, non-failed assignments")

            self._record_inference({
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": condition,
                "target_direction": bool(target_direction),
                "assignments": assignments,
                "llm_analysis": str(result.get("analysis") or ""),
                "llm_confidence": result.get("confidence"),
                "failure_context": failure_context,
                "requires_force_free_replay_validation": True,
                "method": "llm_compound",
            })
            self.last_inference_metadata = {
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": condition,
                "target_direction": bool(target_direction),
                "inference_method": "llm_compound",
                "assignments": assignments,
                "actual_llm_calls": 1,
                "requires_force_free_replay_validation": True,
            }
            return assignments
        except Exception as exc:
            self._record_inference({
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": condition,
                "target_direction": bool(target_direction),
                "failure_context": failure_context,
                "llm_error": str(exc),
                "method": "llm_compound_error",
            })
            logger.warning("[LLMGuide] 复合输入假设无效: %s", exc)
            return []

    def infer_slice_assignments(
        self,
        *,
        branch_pc: int,
        condition: str,
        target_taken: bool,
        instruction_lines: List[str],
        input_sites: List[Dict[str, Any]],
        coupled_input_count: int,
        relation_kind: str = "",
        compare_op: str = "",
        compare_pc: int = 0,
        constraint_note: str = "",
        same_variable_constraints: int = 0,
        prior_failure: Optional[Dict[str, Any]] = None,
        previous_attempt: Optional[Dict[str, Any]] = None,
        max_assignments: int = 4,
    ) -> Optional[List[Dict[str, int]]]:
        """Solve one exported instruction slice for MMIO input values.

        Like ``infer_compound_alternative_constraints`` this is a *hypothesis*
        API: the model may only assign values to the caller-provided observed
        input sites, and every assignment still passes the caller's force-free
        replay validation.  The difference is the input: a trimmed
        instruction-level slice of the failing constraint (design §5.2's
        "大模型理解" tier), reached only after the symbolic solver failed.
        """
        normalized_condition = self._normalize_mnemonic(condition)
        if (
            not self.use_llm
            or self._llm_call_budget_exhausted()
            or self._llm_slice_budget_exhausted()
        ):
            self._record_inference({
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": normalized_condition,
                "target_direction": bool(target_taken),
                "slice_instruction_count": len(instruction_lines or []),
                "coupled_input_count": int(coupled_input_count or 0),
                "inference_method": "llm_slice_skipped",
                "prompt_purpose": (
                    "slice_retry_after_replay_failure"
                    if previous_attempt is not None
                    else "slice_initial"
                ),
                "method": "llm_slice_skipped",
            })
            return None

        normalized_sites: List[Dict[str, Any]] = []
        for raw_site in list(input_sites or []):
            if not isinstance(raw_site, dict):
                continue
            try:
                address = int(raw_site.get("address")) & 0xFFFFFFFF
                read_pc = int(raw_site.get("read_pc") or 0) & 0xFFFFFFFF
                occurrence = max(1, int(raw_site.get("occurrence") or 1))
            except (TypeError, ValueError):
                continue
            try:
                width = max(1, min(32, int(raw_site.get("width") or 32)))
            except (TypeError, ValueError):
                width = 32
            mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
            try:
                current_value = int(raw_site.get("current_value", 0) or 0) & mask
            except (TypeError, ValueError):
                current_value = 0
            failed_values: List[int] = []
            for raw_value in list(raw_site.get("failed_values", []) or []):
                parsed = self._parse_mmio_value(raw_value)
                if parsed is not None and 0 <= parsed <= mask:
                    failed_values.append(int(parsed))
            normalized_sites.append({
                "site_index": len(normalized_sites),
                "kind": str(raw_site.get("kind") or "input").lower(),
                "address": f"0x{address:08x}",
                "read_pc": f"0x{read_pc:08x}",
                "occurrence": occurrence,
                "width": width,
                "current_value": f"0x{current_value:0{max(2, (width + 3) // 4)}x}",
                "failed_values": [f"0x{value:0{max(2, (width + 3) // 4)}x}" for value in sorted(set(failed_values))],
            })
        if not normalized_sites:
            return None

        limit = max(1, min(8, int(max_assignments or 4)))
        coupled = len(normalized_sites)
        coupling_text = (
            f"本约束只有 {coupled} 个输入来源（单变量求解）：只需确定 site 0 的取值。"
            if coupled <= 1
            else f"本约束有 {coupled} 个输入来源需要同时联合确定（耦合输入），"
                 f"最多修改 {min(limit, coupled)} 个 site；它们共同决定比较结果。"
        )
        asm_text = "\n".join(str(line) for line in (instruction_lines or [])) or "（切片为空）"
        note_text = str(constraint_note or "").strip() or "无"
        prior_text = self._format_failure_context(prior_failure)
        sites_text = json.dumps(normalized_sites, ensure_ascii=False, sort_keys=True)
        retry_text = self._format_slice_previous_attempt(previous_attempt)
        purpose = (
            "slice_retry_after_replay_failure"
            if previous_attempt is not None
            else "slice_initial_after_symbolic_failure"
        )

        try:
            prompt = f"""你是世界级的固件逆向与约束求解专家。

背景：符号约束求解器（z3）对下面的分支约束求解失败了（原因见「符号求解失败摘要」）。
现在把这条约束的**指令级切片**交给你阅读：这是从出错分支沿寄存器定义—使用链反向
裁减出的指令序列，只保留数据依赖与同一变量的全部约束，控制依赖已剔除。你的任务是
给出各 MMIO 输入的取值，使最末的目标分支按期望方向执行。

目标分支：{normalized_condition} @ {hex(int(branch_pc) & 0xffffffff)}，期望方向 = {'taken (跳转)' if target_taken else 'not-taken (不跳转)'}
比较指令类型：{str(compare_op or 'CMP')}
关系类型：{str(relation_kind or 'unknown')}
切片内保留的同一变量先前约束条数：{int(same_variable_constraints or 0)}（这些约束必须继续成立）
符号求解失败摘要：{prior_text}
切片预算说明：{note_text}
输入耦合判定：{coupling_text}

可赋值的输入 sites（只能从中选择，不能发明新地址/read_pc/occurrence）：
{sites_text}

上次尝试与重放反馈：{retry_text}

指令级切片（按执行顺序；行尾 ; 后为标注）：
```asm
{asm_text}
```

要求：
1. 只返回 site_index 与 value；site_index 必须来自上面的列表，每个 site 至多一次。
2. value 必须是无符号整数或 0x 十六进制，且不超过该 site 的 width 位宽。
3. 不要返回与 current_value 相同的值，也不要返回 failed_values 里已失败的值。
4. 必须同时满足切片中保留的同一变量全部先前约束，而不是只满足最后一条比较。
5. 优先选择最简单、最可解释的值（位掩码位、比较常量、常量±1）。
6. 如果是重试调用：必须给出与上次不同的替代值，并在 analysis 里说明为什么新值更可能让重放成功。
7. 这是候选假设：最终由无强制具体重放验证，不能靠修改 PC 或寄存器强行通过。

只按 JSON 返回，不要返回其他内容：
{{"assignments":[{{"site_index":0,"value":"0x00"}}],"analysis":"一行说明依据","confidence":0.0,"why_values_satisfy_branch":"一行说明为什么这些值让目标分支按期望方向执行"}}"""

            repair_prompt = """只返回单行 JSON。assignments 必须是对象数组，每个对象只含 site_index 和 value；site_index 必须来自提供的列表；不要 markdown，不要解释文本。"""

            response = self._call_llm_json(prompt, repair_prompt=repair_prompt)
            result_str = extract_response_text(response)
            if not result_str:
                salvaged = self._salvage_response_fields(
                    extract_response_reasoning(response)
                )
                if salvaged and salvaged.get("assignments"):
                    result_str = json.dumps(salvaged, ensure_ascii=False)
            result = self._parse_json_response(result_str)
            raw_assignments = result.get("assignments")
            if isinstance(raw_assignments, str):
                raw_assignments = self._parse_json_response(raw_assignments).get("assignments")
            if not isinstance(raw_assignments, list):
                raise ValueError("assignments must be a JSON array")

            by_index = {int(item["site_index"]): item for item in normalized_sites}
            assignments: List[Dict[str, int]] = []
            seen_indices: Set[int] = set()
            for raw_assignment in raw_assignments:
                if not isinstance(raw_assignment, dict):
                    raise ValueError("assignment must be an object")
                try:
                    site_index = int(raw_assignment.get("site_index"))
                except (TypeError, ValueError):
                    raise ValueError("invalid site_index")
                if site_index not in by_index or site_index in seen_indices:
                    raise ValueError("site_index is not unique or not allowed")
                site = by_index[site_index]
                value = self._parse_mmio_value(raw_assignment.get("value"))
                if value is None:
                    raise ValueError("invalid assignment value")
                width = int(site["width"])
                mask = (1 << width) - 1 if width < 32 else 0xFFFFFFFF
                if value < 0 or value > mask:
                    raise ValueError("assignment value exceeds site width")
                value &= mask
                seen_indices.add(site_index)
                current = self._parse_mmio_value(site["current_value"])
                failed = {
                    parsed for parsed in (
                        self._parse_mmio_value(item) for item in site["failed_values"]
                    ) if parsed is not None
                }
                if current is not None and value == int(current):
                    continue
                if value in failed:
                    continue
                assignments.append({"site_index": site_index, "value": value})
                if len(assignments) > limit:
                    raise ValueError("too many assignments")
            if not assignments:
                raise ValueError("no changed, non-failed assignments")

            self._record_inference({
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": normalized_condition,
                "target_direction": bool(target_taken),
                "assignments": assignments,
                "slice_instruction_count": len(instruction_lines or []),
                "coupled_input_count": int(coupled_input_count or 0),
                "relation_kind": str(relation_kind or ""),
                "same_variable_constraints": int(same_variable_constraints or 0),
                "constraint_note": str(constraint_note or ""),
                "llm_analysis": str(result.get("analysis") or ""),
                "llm_confidence": result.get("confidence"),
                "prior_failure": prior_failure,
                "previous_attempt": previous_attempt,
                "prompt_purpose": purpose,
                "requires_force_free_replay_validation": True,
                "method": "llm_slice",
            })
            self.last_inference_metadata = {
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": normalized_condition,
                "target_direction": bool(target_taken),
                "inference_method": "llm_slice",
                "assignments": assignments,
                "actual_llm_calls": 1,
                "requires_force_free_replay_validation": True,
            }
            return assignments
        except Exception as exc:
            self._record_inference({
                "branch_pc": hex(int(branch_pc) & 0xFFFFFFFF),
                "condition": normalized_condition,
                "target_direction": bool(target_taken),
                "llm_error": str(exc),
                "slice_instruction_count": len(instruction_lines or []),
                "coupled_input_count": int(coupled_input_count or 0),
                "prior_failure": prior_failure,
                "previous_attempt": previous_attempt,
                "prompt_purpose": purpose,
                "method": "llm_slice_error",
            })
            logger.warning("[LLMGuide] 切片求解无效，放弃本次升级: %s", exc)
            return None

    def _format_slice_previous_attempt(
        self,
        previous_attempt: Optional[Dict[str, Any]],
    ) -> str:
        """Compress the prior slice attempt + real replay outcome for retries."""
        if not isinstance(previous_attempt, dict) or not previous_attempt:
            return "无（首次调用）"
        parts: List[str] = []
        assignments = previous_attempt.get("assignments")
        if isinstance(assignments, list) and assignments:
            rendered = []
            for item in assignments:
                if not isinstance(item, dict):
                    continue
                value = self._parse_mmio_value(item.get("value"))
                value_text = (
                    f"0x{int(value) & 0xFFFFFFFF:08x}"
                    if value is not None
                    else str(item.get("value"))
                )
                rendered.append(
                    f"site_index={item.get('site_index')} value={value_text}"
                )
            if rendered:
                parts.append("上次给出的值: " + "; ".join(rendered))
        replay = previous_attempt.get("replay_outcome")
        if isinstance(replay, dict) and replay:
            parts.append("重放的真实结果: " + self._format_failure_context(replay))
        elif replay:
            parts.append(f"重放的真实结果: {str(replay)[:200]}")
        parts.append("请给出与上次不同的替代值，并说明理由")
        return "；".join(parts) if parts else "无（首次调用）"

    def _llm_slice_budget_exhausted(self) -> bool:
        """Dedicated bounded budget for the slice tier (no unbounded retries)."""
        try:
            limit = int(float(os.environ.get("LSGEMU_MAX_LLM_SLICE_CALLS") or 64))
        except (TypeError, ValueError):
            limit = 64
        if limit <= 0:
            return False
        used = sum(
            1
            for item in self.inference_history
            if isinstance(item, dict)
            and str(item.get("method") or "").startswith("llm_slice")
            and str(item.get("method") or "") != "llm_slice_skipped"
        )
        return used >= limit

    def _find_compare_instruction(self, branch_pc: int) -> Optional[Dict]:
        """查找分支前的比较指令"""
        if branch_pc in self.compare_lookup:
            return self.compare_lookup[branch_pc]

        instructions = self._get_bb_instructions(branch_pc)
        if not instructions:
            return None

        for i, insn in enumerate(instructions):
            if insn['address'] == branch_pc:
                for j in range(i - 1, max(-1, i - 11), -1):
                    prev_insn = instructions[j]
                    if self._normalize_mnemonic(prev_insn['mnemonic']) in ['CMP', 'CMN', 'TST', 'TEQ']:
                        return prev_insn
        return None

    def _normalize_mnemonic(self, mnemonic: str) -> str:
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

    def _get_bb_instructions(self, pc: int) -> List[Dict]:
        """获取 PC 所在基本块的指令列表"""
        bb_addr = self.instruction_to_bb.get(pc)
        if bb_addr is not None:
            return self.static_bbs.get(bb_addr, [])

        # 兼容外部传入非索引地址的旧调用方
        for start, instructions in self.static_bbs.items():
            if instructions and start <= pc <= instructions[-1]['address']:
                return instructions
        return []

    def _get_branch_context(self, branch_pc: int, before: int = 15, after: int = 2) -> str:
        """获取分支附近汇编上下文"""
        instructions = self._get_bb_instructions(branch_pc)
        if not instructions:
            return ""

        for i, insn in enumerate(instructions):
            if insn['address'] == branch_pc:
                start = max(0, i - before)
                end = min(len(instructions), i + after + 1)
                return "\n".join(self._format_instruction(item, include_address=True) for item in instructions[start:end])
        return ""

    def _format_instruction(self, insn: Dict, include_address: bool = False) -> str:
        """格式化单条指令"""
        if include_address:
            return f"0x{insn['address']:08x}: {insn['mnemonic']} {insn['operands']}".strip()
        return f"{insn['mnemonic']} {insn['operands']}".strip()

    def _extract_compare_value(self, compare_insn: Dict) -> Optional[int]:
        """从比较指令中提取比较值"""
        try:
            operands = compare_insn['operands'].split(',')
            if len(operands) >= 2:
                right_op = operands[1].strip()

                # 解析立即数
                if right_op.startswith('#'):
                    value_str = right_op[1:]
                    if value_str.startswith('0x'):
                        return int(value_str, 16)
                    else:
                        return int(value_str)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass

        return None

    def _extract_bitmask(self, compare_insn: Dict) -> Optional[int]:
        """从 TST 指令中提取位掩码"""
        return self._extract_compare_value(compare_insn)

    def _find_masked_register_compare(self, branch_pc: int, compare_insn: Optional[Dict]) -> Optional[Dict[str, object]]:
        """识别 `AND reg, reg, #mask` 后紧跟 `CMP/CMN reg, reg` 的模式。"""
        if not compare_insn:
            return None

        operands = [item.strip() for item in str(compare_insn.get('operands', '')).split(',')]
        if len(operands) < 2:
            return None

        left_reg = operands[0].lower()
        right_reg = operands[1].lower()
        instructions = self._get_bb_instructions(branch_pc)
        compare_index = None
        for index, insn in enumerate(instructions):
            if int(insn.get('address', 0) or 0) == int(compare_insn.get('address', 0) or 0):
                compare_index = index
                break
        if compare_index is None:
            return None

        for prev_insn in reversed(instructions[max(0, compare_index - 4):compare_index]):
            mnemonic = self._normalize_mnemonic(prev_insn.get('mnemonic', ''))
            if mnemonic not in {'AND', 'ANDS'}:
                continue
            prev_operands = [item.strip() for item in str(prev_insn.get('operands', '')).split(',')]
            if len(prev_operands) < 3:
                continue
            dest_reg = prev_operands[0].lower()
            mask = self._parse_immediate_operand(prev_operands[2])
            if mask is None:
                continue
            if dest_reg == right_reg:
                return {
                    'masked_reg': right_reg,
                    'other_reg': left_reg,
                    'masked_side': 'right',
                    'mask': mask & 0xFFFFFFFF,
                    'instruction': self._format_instruction(prev_insn),
                }
            if dest_reg == left_reg:
                return {
                    'masked_reg': left_reg,
                    'other_reg': right_reg,
                    'masked_side': 'left',
                    'mask': mask & 0xFFFFFFFF,
                    'instruction': self._format_instruction(prev_insn),
                }
        return None

    def _parse_immediate_operand(self, operand: str) -> Optional[int]:
        text = str(operand or '').strip()
        if not text.startswith('#'):
            return None
        text = text[1:]
        try:
            return int(text, 16) if text.lower().startswith('0x') else int(text)
        except ValueError:
            return None

    def _infer_masked_register_compare_value(
        self,
        masked_compare: Dict[str, object],
        branch_condition: str,
        target_direction: bool,
    ) -> int:
        """对 `AND #mask -> CMP reg, reg` 模式给出稳定的极值解。"""
        condition = self._normalize_mnemonic(branch_condition)
        mask = int(masked_compare.get('mask', 0)) & 0xFFFFFFFF
        compare_operand = self._parse_immediate_operand(str(masked_compare.get('other_reg', '') or ''))
        if compare_operand is not None:
            return self._infer_masked_immediate_compare_value(
                mask,
                compare_operand,
                condition,
                target_direction,
            )
        max_value = mask if mask else 0xFFFFFFFF
        min_value = 0x0
        masked_side = masked_compare.get('masked_side')

        def branch_wants_masked_greater() -> Optional[bool]:
            if masked_side == 'right':
                if condition in {'BCS', 'BHS'}:
                    return not target_direction
                if condition in {'BCC', 'BLO'}:
                    return target_direction
                if condition == 'BHI':
                    return not target_direction
                if condition == 'BLS':
                    return target_direction
            elif masked_side == 'left':
                if condition in {'BCS', 'BHS'}:
                    return target_direction
                if condition in {'BCC', 'BLO'}:
                    return not target_direction
                if condition == 'BHI':
                    return target_direction
                if condition == 'BLS':
                    return not target_direction
            return None

        wants_masked_greater = branch_wants_masked_greater()
        if wants_masked_greater is None:
            return max_value if target_direction else min_value
        return max_value if wants_masked_greater else min_value

    def _infer_masked_immediate_compare_value(
        self,
        mask: int,
        compare_value: int,
        branch_condition: str,
        target_direction: bool,
    ) -> int:
        """对 `(value & mask) ? imm` 模式构造能真实满足条件的最小值。"""
        condition = self._normalize_mnemonic(branch_condition)
        mask &= 0xFFFFFFFF
        compare_value &= 0xFFFFFFFF
        masked_target = compare_value & mask
        masked_domain_max = mask

        if condition == 'BEQ':
            if target_direction:
                return masked_target
            if masked_target != 0:
                return 0
            return self._least_significant_bit(mask)

        if condition == 'BNE':
            if target_direction:
                if masked_target != 0:
                    return 0
                return self._least_significant_bit(mask)
            return masked_target

        if condition in {'BHI'}:
            threshold = masked_target
            if target_direction:
                return self._masked_next_greater_value(threshold, mask)
            return threshold

        if condition in {'BHS', 'BCS'}:
            threshold = masked_target
            if target_direction:
                return threshold
            return self._masked_next_lower_value(threshold, mask)

        if condition in {'BLO', 'BCC'}:
            threshold = masked_target
            if target_direction:
                return self._masked_next_lower_value(threshold, mask)
            return threshold

        if condition == 'BLS':
            threshold = masked_target
            if target_direction:
                return threshold
            return self._masked_next_greater_value(threshold, mask)

        if condition == 'BGT':
            threshold = self._signed32(masked_target)
            if target_direction:
                return self._masked_next_signed_greater_value(threshold, mask)
            return masked_target

        if condition == 'BGE':
            threshold = self._signed32(masked_target)
            if target_direction:
                return masked_target
            return self._masked_next_signed_lower_value(threshold, mask)

        if condition == 'BLT':
            threshold = self._signed32(masked_target)
            if target_direction:
                return self._masked_next_signed_lower_value(threshold, mask)
            return masked_target

        if condition == 'BLE':
            threshold = self._signed32(masked_target)
            if target_direction:
                return masked_target
            return self._masked_next_signed_greater_value(threshold, mask)

        if target_direction:
            return masked_target
        if masked_target != 0:
            return 0
        return self._least_significant_bit(mask)

    def _sanitize_mmio_value(self, value: int) -> int:
        """约束为 32 位无符号值"""
        return max(0, min(0xFFFFFFFF, int(value)))

    @staticmethod
    def _normalize_llm_enum(value: Any, allowed: Set[str], default: str) -> str:
        text = str(value or "").strip().lower()
        if text in allowed:
            return text
        return default

    @staticmethod
    def _least_significant_bit(value: int) -> int:
        value = int(value) & 0xFFFFFFFF
        return value & -value if value else 1

    @staticmethod
    def _least_significant_zero_bit(mask: int) -> int:
        mask = int(mask) & 0xFFFFFFFF
        for bit in range(32):
            candidate = 1 << bit
            if (mask & candidate) == 0:
                return candidate
        return 0xFFFFFFFF

    @staticmethod
    def _masked_next_greater_value(current: int, mask: int) -> int:
        current &= mask & 0xFFFFFFFF
        mask &= 0xFFFFFFFF
        candidates = [1 << bit for bit in range(32) if mask & (1 << bit)]
        for candidate in candidates:
            if current + candidate <= mask:
                return (current + candidate) & 0xFFFFFFFF
        return current

    @staticmethod
    def _masked_next_lower_value(current: int, mask: int) -> int:
        current &= mask & 0xFFFFFFFF
        mask &= 0xFFFFFFFF
        if current == 0:
            return 0
        candidates = [1 << bit for bit in range(32) if current & (1 << bit)]
        if not candidates:
            return 0
        return (current - min(candidates)) & 0xFFFFFFFF

    def _masked_next_signed_greater_value(self, threshold: int, mask: int) -> int:
        mask &= 0xFFFFFFFF
        best = None
        for bit in range(32):
            candidate = 1 << bit
            if not (mask & candidate):
                continue
            trial = candidate & mask
            if self._signed32(trial) > threshold:
                if best is None or self._signed32(trial) < self._signed32(best):
                    best = trial
        return best if best is not None else (mask & 0xFFFFFFFF)

    def _masked_next_signed_lower_value(self, threshold: int, mask: int) -> int:
        mask &= 0xFFFFFFFF
        candidates = [0]
        for bit in range(32):
            candidate = 1 << bit
            if mask & candidate:
                candidates.append(candidate & mask)
        best = None
        for trial in candidates:
            if self._signed32(trial) < threshold:
                if best is None or self._signed32(trial) > self._signed32(best):
                    best = trial
        return best if best is not None else 0

    def _fallback_unknown_tst_value(self, branch_condition: str, target_direction: bool) -> int:
        """TST reg, reg 无法拿到立即数掩码时，给出尽量稳妥的默认值。"""
        condition = self._normalize_mnemonic(branch_condition)
        wants_non_zero = (
            (condition == 'BNE' and target_direction)
            or (condition == 'BEQ' and not target_direction)
        )
        return 0xFFFFFFFF if wants_non_zero else 0x0

    def _signed32(self, value: int) -> int:
        value &= 0xFFFFFFFF
        return value if value < 0x80000000 else value - 0x100000000

    def _describe_branch_goal(
        self,
        branch_condition: str,
        target_direction: bool,
        compare_insn: Optional[Dict],
    ) -> str:
        condition = self._normalize_mnemonic(branch_condition)
        direction = "taken" if target_direction else "not-taken"
        if condition == 'CBZ':
            return f"让寄存器值在 {direction} 场景下 {'等于 0' if target_direction else '不等于 0'}"
        if condition == 'CBNZ':
            return f"让寄存器值在 {direction} 场景下 {'不等于 0' if target_direction else '等于 0'}"

        if not compare_insn:
            return f"让分支 {condition} 按 {direction} 方向执行"

        mnemonic = self._normalize_mnemonic(compare_insn['mnemonic'])
        immediate = self._extract_compare_value(compare_insn)
        if mnemonic == 'TST':
            if immediate is None:
                return f"TST 后让 {condition} 按 {direction} 方向执行"
            zero_text = f"(value & {hex(immediate)}) == 0"
            non_zero_text = f"(value & {hex(immediate)}) != 0"
            if condition == 'BEQ':
                return f"目标是让 {zero_text if target_direction else non_zero_text}"
            if condition == 'BNE':
                return f"目标是让 {non_zero_text if target_direction else zero_text}"
        if mnemonic == 'TEQ' and immediate is not None:
            equal_text = f"value == {hex(immediate)}"
            not_equal_text = f"value != {hex(immediate)}"
            if condition == 'BEQ':
                return f"目标是让 {equal_text if target_direction else not_equal_text}"
            if condition == 'BNE':
                return f"目标是让 {not_equal_text if target_direction else equal_text}"
        if immediate is not None:
            return f"目标是让 value 与 {hex(immediate)} 的关系满足 {condition} 的 {direction} 方向"
        return f"让分支 {condition} 按 {direction} 方向执行"

    def _format_failure_context(self, failure_context: Optional[Dict[str, Any]]) -> str:
        """Compress replay feedback into one prompt-safe line."""
        if not isinstance(failure_context, dict) or not failure_context:
            return "无"
        interesting_keys = [
            "outcome",
            "failure_reason",
            "strategy",
            "control_failure_reason",
            "result_failure_reason",
            "branch_seen",
            "observed_directions",
            "matched_constraint_reads",
            "matched_address_reads",
            "all_constraint_reads_matched",
            "stop_reason",
            "read_pc",
            "read_occurrence",
            "dependency_register",
            "dependency_expression",
            "dependency_group",
            "dynamic_solver_status",
            "dynamic_solver_reason",
            "dynamic_fallback_eligible",
            "dynamic_omitted_path_predicates",
            "dynamic_branch_record_match",
        ]
        parts: List[str] = []
        for key in interesting_keys:
            if key not in failure_context:
                continue
            value = failure_context.get(key)
            if isinstance(value, (dict, list, tuple, set)):
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                text = str(value)
            text = re.sub(r"\s+", " ", text).replace('"', "'")
            if len(text) > 96:
                text = text[:93] + "..."
            parts.append(f"{key}={text}")
        return "; ".join(parts) if parts else "无"

    def _semantic_value_hints(
        self,
        branch_condition: str,
        target_direction: bool,
        compare_insn: Optional[Dict],
        rule_candidate: Optional[int],
        avoid_values: List[int],
    ) -> str:
        """Describe the satisfying value set to keep LLM alternatives semantic."""
        condition = self._normalize_mnemonic(branch_condition)
        rule_text = f"规则候选 0x{int(rule_candidate) & 0xFFFFFFFF:08x}" if rule_candidate is not None else "无规则候选"
        avoid = {int(value) & 0xFFFFFFFF for value in avoid_values or []}
        alternative = None

        if not compare_insn:
            return f"{rule_text}; 未识别比较指令，若无更强证据不要偏离规则候选"

        mnemonic = self._normalize_mnemonic(compare_insn.get("mnemonic", ""))
        if condition in {"CBZ", "CBNZ"}:
            return f"{rule_text}; CBZ/CBNZ 只有零值或非零值两类，优先 0x0/0x1"

        if mnemonic == "TST":
            bitmask = self._extract_bitmask(compare_insn)
            if bitmask is None:
                return f"{rule_text}; TST 掩码未知，零值表示所有测试位为 0，0xffffffff 表示至少一位通常非零"
            bitmask &= 0xFFFFFFFF
            wants_non_zero = (
                (condition == "BNE" and target_direction)
                or (condition == "BEQ" and not target_direction)
            )
            if wants_non_zero:
                candidates = [
                    self._least_significant_bit(bitmask),
                    bitmask,
                    (int(rule_candidate) | bitmask) & 0xFFFFFFFF if rule_candidate is not None else bitmask,
                ]
                for candidate in candidates:
                    if candidate not in avoid:
                        alternative = candidate
                        break
                return (
                    f"{rule_text}; 满足条件需要 (value & {hex(bitmask)}) != 0，"
                    f"可选值包括最低有效位或完整掩码，备选 {hex(alternative) if alternative is not None else '无'}"
                )
            return f"{rule_text}; 满足条件需要 (value & {hex(bitmask)}) == 0，通常 0x0 是最小且最稳妥的唯一语义类"

        compare_value = self._extract_compare_value(compare_insn)
        if compare_value is None:
            return f"{rule_text}; 比较常量未知，若没有 replay 证据不要编造复杂值"
        compare_value &= 0xFFFFFFFF
        if mnemonic == "TEQ":
            if (condition == "BEQ" and target_direction) or (condition == "BNE" and not target_direction):
                return f"{rule_text}; TEQ 零结果要求 value == {hex(compare_value)}，这是唯一精确值"
            alternative = (compare_value ^ 1) & 0xFFFFFFFF
            return f"{rule_text}; TEQ 非零结果要求 value != {hex(compare_value)}，最小备选 {hex(alternative)}"

        if condition in {"BEQ"} and target_direction:
            return f"{rule_text}; 相等分支要求 value == {hex(compare_value)}，这是唯一精确值"
        if condition in {"BNE"} and not target_direction:
            return f"{rule_text}; 不等分支的 not-taken 要求 value == {hex(compare_value)}，这是唯一精确值"
        if condition in {"BEQ"} and not target_direction:
            alternative = 0 if compare_value != 0 else 1
            return f"{rule_text}; not-taken 要求 value != {hex(compare_value)}，最小备选 {hex(alternative)}"
        if condition in {"BNE"} and target_direction:
            alternative = 0 if compare_value != 0 else 1
            return f"{rule_text}; taken 要求 value != {hex(compare_value)}，最小备选 {hex(alternative)}"
        return f"{rule_text}; 需要满足 {condition} 与 {hex(compare_value)} 的大小关系，优先比较常量相邻值"

    def _validate_inferred_value(
        self,
        value: int,
        branch_condition: str,
        target_direction: bool,
        compare_insn: Optional[Dict],
    ) -> Optional[bool]:
        condition = self._normalize_mnemonic(branch_condition)
        if value < 0 or value > 0xFFFFFFFF:
            return False

        unsigned_value = value & 0xFFFFFFFF
        if condition == 'CBZ':
            return (unsigned_value == 0) == target_direction
        if condition == 'CBNZ':
            return (unsigned_value != 0) == target_direction
        if not compare_insn:
            return None

        mnemonic = self._normalize_mnemonic(compare_insn['mnemonic'])
        if mnemonic == 'TST':
            bitmask = self._extract_bitmask(compare_insn)
            if bitmask is None:
                return None
            zero_result = (unsigned_value & bitmask) == 0
            if condition == 'BEQ':
                actual_taken = zero_result
            elif condition == 'BNE':
                actual_taken = not zero_result
            else:
                return None
            return actual_taken == target_direction

        compare_value = self._extract_compare_value(compare_insn)
        masked_compare = self._find_masked_register_compare(
            int(compare_insn.get('address', 0) or 0),
            compare_insn,
        )
        if masked_compare is not None and compare_value is not None:
            masked_value = unsigned_value & (int(masked_compare.get('mask', 0)) & 0xFFFFFFFF)
            left_unsigned = masked_value & 0xFFFFFFFF
            right_unsigned = compare_value & 0xFFFFFFFF
            left_signed = self._signed32(left_unsigned)
            right_signed = self._signed32(right_unsigned)
        else:
            left_unsigned = unsigned_value
            right_unsigned = (compare_value & 0xFFFFFFFF) if compare_value is not None else 0
            left_signed = self._signed32(left_unsigned)
            right_signed = self._signed32(right_unsigned)

        if compare_value is None:
            return None

        if mnemonic == 'TEQ':
            zero_result = (left_unsigned ^ right_unsigned) == 0
            if condition == 'BEQ':
                actual_taken = zero_result
            elif condition == 'BNE':
                actual_taken = not zero_result
            else:
                return None
            return actual_taken == target_direction

        actual_taken = None
        if condition == 'BEQ':
            actual_taken = left_unsigned == right_unsigned
        elif condition == 'BNE':
            actual_taken = left_unsigned != right_unsigned
        elif condition == 'BGT':
            actual_taken = left_signed > right_signed
        elif condition == 'BGE':
            actual_taken = left_signed >= right_signed
        elif condition == 'BLT':
            actual_taken = left_signed < right_signed
        elif condition == 'BLE':
            actual_taken = left_signed <= right_signed
        elif condition in ['BHI']:
            actual_taken = left_unsigned > right_unsigned
        elif condition in ['BHS', 'BCS']:
            actual_taken = left_unsigned >= right_unsigned
        elif condition in ['BLO', 'BCC']:
            actual_taken = left_unsigned < right_unsigned
        elif condition == 'BLS':
            actual_taken = left_unsigned <= right_unsigned

        if actual_taken is None:
            return None
        return actual_taken == target_direction

    def _parse_mmio_value(self, raw_value) -> Optional[int]:
        """解析并验证 LLM 返回的 MMIO 值"""
        if raw_value is None:
            return None

        if isinstance(raw_value, int):
            return raw_value

        value_text = str(raw_value).strip()
        match = re.search(r'-?0x[0-9a-fA-F]+|-?\d+', value_text)
        if not match:
            return None

        token = match.group(0)
        try:
            return int(token, 16) if token.lower().startswith('-0x') or token.lower().startswith('0x') else int(token)
        except ValueError:
            return None

    def _parse_json_response(self, result_str: str) -> Dict:
        """解析 JSON 回复，兼容模型额外包裹文本的情况"""
        try:
            return parse_json_object(result_str)
        except Exception:
            salvaged = self._salvage_response_fields(result_str)
            if salvaged:
                return salvaged
            raise

    def _salvage_response_fields(self, result_str: str) -> Optional[Dict]:
        """在 JSON 不规范时，尽量提取关键字段，避免白白回退规则。"""
        data = salvage_named_fields(
            result_str,
            numeric_fields=("mmio_value", "required_value", "value"),
            text_fields=("analysis", "why_value_satisfies_branch"),
        )
        if not data:
            mmio_match = re.search(r'MMIO\[[^\]]+\]\s*=\s*(0x[0-9a-fA-F]+|\d+)', result_str, re.IGNORECASE)
            if not mmio_match:
                return None
            data = {"mmio_value": mmio_match.group(1)}
        if "mmio_value" not in data:
            for alias in ("required_value", "value"):
                if alias in data:
                    data["mmio_value"] = data[alias]
                    break
        if "mmio_value" not in data:
            return None

        confidence_match = re.search(
            r'["\']?confidence["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            result_str,
            re.IGNORECASE,
        )
        if confidence_match:
            try:
                data["confidence"] = float(confidence_match.group(1))
            except ValueError:
                pass

        return data

    def _extract_unquoted_field_text(self, result_str: str, field_name: str) -> Optional[str]:
        return extract_unquoted_field_text(result_str, field_name)

    def _repair_json_like_text(self, text: str) -> Optional[Dict]:
        return repair_json_like_text(text)

    def _extract_json_string_field(self, result_str: str, field_name: str) -> Optional[str]:
        return extract_json_string_field(result_str, field_name)

    def _call_llm_json(self, prompt: str, repair_prompt: Optional[str] = None):
        """调用 LLM，优先请求 JSON 模式，兼容不支持 response_format 的服务端"""
        return call_llm_json(
            client=self.llm_client,
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.llm_max_tokens,
            temperature=0.0,
            repair_prompt=repair_prompt,
            parse_response=self._parse_json_response,
            logger=logger,
            warn_key=f"llmguide:{self.llm_model}",
        )

    def infer_multiple_constraints(self, branches: List[Dict]) -> Dict[int, int]:
        """
        批量推断多个分支的约束

        Args:
            branches: [{'pc': ..., 'condition': ..., 'direction': ..., 'mmio': ...}, ...]

        Returns:
            {mmio_addr: value}
        """
        constraints = {}

        for branch in branches:
            value = self.infer_constraint(
                branch['pc'],
                branch['condition'],
                branch['direction'],
                branch['mmio']
            )

            if value is not None:
                # 最新值覆盖旧值
                constraints[branch['mmio']] = value

        return constraints

    def get_inference_history(self) -> List[Dict]:
        """获取推断历史"""
        return self.inference_history

    @staticmethod
    def _journal_session_id() -> str:
        return f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    def enable_inference_journal(self, journal_path, *, session_id: Optional[str] = None) -> bool:
        """开启增量推断 journal（JSONL，一行一条，追加写）。

        每条推断完成即写入并 flush 到操作系统缓冲，因此 SIGTERM/SIGKILL/
        native 崩溃都不会丢失已完成的推断；finalize 阶段的
        save_inference_history 仍然独立产出完整 JSON。
        journal 打开或写入失败只告警一次并自动停用，绝不影响推断主流程。
        """
        if self._inference_journal_handle is not None:
            self.close_inference_journal()
        path = str(journal_path)
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            handle = open(path, "a", encoding="utf-8")
        except OSError as exc:
            self._inference_journal_failed = True
            logger.warning(f"[LLMGuide] 推断journal打开失败（停用增量落盘）: {path}: {exc}")
            return False
        self._inference_journal_path = path
        self._inference_journal_handle = handle
        self._inference_journal_failed = False
        if session_id:
            self._inference_journal_session_id = str(session_id)
        header = {
            "journal_event": "session_start",
            "session_id": self._inference_journal_session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
            "use_llm": bool(self.use_llm),
            "llm_model": getattr(self, "llm_model", None),
            "records_before_journal": len(self.inference_history),
        }
        self._write_journal_line(header)
        return True

    def _write_journal_line(self, payload: Dict[str, Any]) -> None:
        handle = self._inference_journal_handle
        if handle is None or self._inference_journal_failed:
            return
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            # flush 到 OS 缓冲即可保证进程死亡后数据仍在；无需每条 fsync。
            handle.flush()
        except Exception as exc:
            self._inference_journal_failed = True
            logger.warning(
                f"[LLMGuide] 推断journal写入失败（停用增量落盘）: "
                f"{self._inference_journal_path}: {exc}"
            )
            try:
                handle.close()
            except Exception:
                pass
            self._inference_journal_handle = None

    def _record_inference(self, record: Dict[str, Any]) -> None:
        """推断历史的唯一写入口：内存列表 + 增量 journal 双写。"""
        self.inference_history.append(record)
        if self._inference_journal_handle is None:
            return
        line = dict(record)
        line["seq"] = len(self.inference_history) - 1
        line["session_id"] = self._inference_journal_session_id
        self._write_journal_line(line)

    def close_inference_journal(self) -> None:
        """关闭增量 journal（幂等；finalize 后可显式调用）。"""
        handle = self._inference_journal_handle
        if handle is None:
            return
        self._inference_journal_handle = None
        try:
            handle.flush()
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    def save_inference_history(self, output_file: str):
        """保存推断历史到文件"""
        try:
            atomic_json_dump(self.inference_history, output_file, indent=2, ensure_ascii=False)
            logger.info(f"[LLMGuide] 保存了 {len(self.inference_history)} 条推断历史")
        except Exception as e:
            logger.error(f"[LLMGuide] 保存失败: {e}")

    def clear(self):
        """清空历史"""
        if self._inference_journal_handle is not None:
            self._write_journal_line({
                "journal_event": "clear",
                "session_id": self._inference_journal_session_id,
                "cleared_records": len(self.inference_history),
            })
        self.inference_history.clear()
