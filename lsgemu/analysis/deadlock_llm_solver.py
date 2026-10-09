#!/usr/bin/env python3
"""
死循环LLM求解器 - 集成到IntelligentEmulator
"""

import json
import re
import logging
import os
from typing import Optional, Dict, List, Tuple

try:
    from ..runtime_bootstrap import bootstrap_runtime_dependencies
    from ..llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        extract_response_text,
        parse_json_object,
        salvage_named_fields,
    )
except ImportError:
    from lsgemu.runtime_bootstrap import bootstrap_runtime_dependencies
    from lsgemu.llm_json_utils import (
        DEFAULT_LLM_MAX_TOKENS,
        call_llm_json,
        extract_response_text,
        parse_json_object,
        salvage_named_fields,
    )

bootstrap_runtime_dependencies()

logger = logging.getLogger(__name__)


class DeadlockLLMSolver:
    """使用LLM求解死循环约束"""

    def __init__(self, llm_client, llm_model, static_bbs):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.static_bbs = static_bbs
        self.inference_history: List[Dict] = []
        self.solve_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Dict[int, int]]] = {}
        self.call_counts_by_loop: Dict[int, int] = {}
        self.last_solution_metadata: Optional[Dict[str, object]] = None
        try:
            self.max_calls_per_loop = max(
                0,
                int(os.environ.get("LSGEMU_MAX_DEADLOCK_LLM_CALLS_PER_LOOP", "1")),
            )
        except ValueError:
            self.max_calls_per_loop = 1

    def _call_llm_json(self, prompt: str):
        repair_prompt = (
            "只返回单行 JSON 对象，不要 markdown，不要解释。"
            " `mmio_address` 必须从候选集合中选一个十六进制地址；"
            " `required_value` 必须是 32 位无符号十六进制字符串；"
            " `status_mask` 如果未知则为 null；"
            " `reason` 只能是一行短句。"
        )
        return call_llm_json(
            client=self.llm_client,
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            # 推理模型思维链与最终答案共用 max_tokens 预算，过小会截断成空。
            max_tokens=DEFAULT_LLM_MAX_TOKENS,
            temperature=0.0,
            repair_prompt=repair_prompt,
            parse_response=self._parse_json_response,
            logger=logger,
            warn_key=f"deadlock:{self.llm_model}",
        )

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        try:
            return parse_json_object(text)
        except Exception:
            salvaged = salvage_named_fields(
                text,
                numeric_fields=("mmio_address", "required_value", "value", "status_mask", "confidence"),
                text_fields=("reason", "semantic_kind"),
            )
            if salvaged and "required_value" not in salvaged and "value" in salvaged:
                salvaged["required_value"] = salvaged["value"]
            return salvaged

    def _parse_u32(self, raw_value) -> Optional[int]:
        if raw_value is None:
            return None
        if isinstance(raw_value, int):
            value = raw_value
        else:
            match = re.search(r"0x[0-9a-fA-F]+|\d+", str(raw_value).strip())
            if not match:
                return None
            token = match.group(0)
            value = int(token, 16) if token.lower().startswith("0x") else int(token)
        if value < 0 or value > 0xFFFFFFFF:
            return None
        return value

    @staticmethod
    def _parse_float(raw_value, default: float = 0.0) -> float:
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            match = re.search(r"\d+(?:\.\d+)?", str(raw_value))
            if not match:
                return default
            value = float(match.group(0))
        return max(0.0, min(1.0, value))

    def solve_deadlock(
        self,
        loop_address: int,
        execution_count: int,
        mmio_addresses: set,
        loop_bbs: Optional[List[int]] = None,
        mmio_access_history: Optional[List[Tuple[int, int, bool, int]]] = None,
    ) -> Optional[Dict[int, int]]:
        """
        求解死循环约束

        Args:
            loop_address: 循环地址
            execution_count: 执行次数
            mmio_addresses: 循环中访问的MMIO地址

        Returns:
            {mmio_addr: value} 或 None
        """
        if not mmio_addresses:
            logger.warning(f"[DeadlockSolver] 循环 {hex(loop_address)} 没有MMIO访问")
            return None
        cache_key = (
            int(loop_address) & 0xFFFFFFFF,
            tuple(sorted(int(addr) & 0xFFFFFFFF for addr in mmio_addresses)),
        )
        if cache_key in self.solve_cache:
            cached = self.solve_cache[cache_key]
            logger.debug(
                "[DeadlockSolver] 缓存命中: loop=%s result=%s",
                hex(loop_address),
                cached,
            )
            return dict(cached) if cached else None
        calls = self.call_counts_by_loop.get(int(loop_address), 0)
        if calls >= self.max_calls_per_loop:
            logger.info(
                "[DeadlockSolver] 跳过重复LLM求解: loop=%s calls=%d limit=%d",
                hex(loop_address),
                calls,
                self.max_calls_per_loop,
            )
            self.solve_cache[cache_key] = None
            return None
        self.call_counts_by_loop[int(loop_address)] = calls + 1

        # 获取完整循环体汇编。多BB轮询经常是 caller -> MMIO helper -> caller
        # continuation，只给 loop_head 会让 LLM 看到 `bx lr` 之类的无效片段。
        context_bbs = []
        seen_bbs = set()
        for bb in list(loop_bbs or []) + [loop_address]:
            try:
                bb = int(bb)
            except Exception:
                continue
            if bb in seen_bbs:
                continue
            if bb not in self.static_bbs:
                continue
            seen_bbs.add(bb)
            context_bbs.append(bb)
        if not context_bbs:
            self.solve_cache[cache_key] = None
            return None

        mmio_read_pcs: Dict[int, List[int]] = {}
        for item in mmio_access_history or []:
            try:
                pc, addr, is_read, _value = item
            except Exception:
                continue
            if not is_read:
                continue
            addr = int(addr) & 0xFFFFFFFF
            if addr not in mmio_addresses:
                continue
            mmio_read_pcs.setdefault(int(pc) & 0xFFFFFFFF, []).append(addr)

        recent_mmio_lines: List[str] = []
        for item in list(mmio_access_history or [])[-24:]:
            try:
                pc, addr, is_read, value = item
            except Exception:
                continue
            direction = "R" if is_read else "W"
            candidate = " candidate" if (int(addr) & 0xFFFFFFFF) in mmio_addresses else ""
            recent_mmio_lines.append(
                f"  {direction} pc=0x{int(pc) & 0xffffffff:08x} "
                f"addr=0x{int(addr) & 0xffffffff:08x} "
                f"value=0x{int(value) & 0xffffffff:08x}{candidate}"
            )

        asm_lines = []
        for bb in context_bbs[:8]:
            asm_lines.append(f"BB 0x{bb:08x}:")
            for insn in self.static_bbs.get(bb, [])[:24]:
                try:
                    pc = int(insn.get("address", 0) or 0) & 0xFFFFFFFF
                except Exception:
                    pc = 0
                marker = ""
                if pc in mmio_read_pcs:
                    marker = " ; MMIO_READ " + ",".join(
                        f"0x{addr:08x}" for addr in sorted(set(mmio_read_pcs[pc]))
                    )
                asm_lines.append(
                    f"  0x{pc:08x}: {insn.get('mnemonic', '')} {insn.get('operands', '')}{marker}".rstrip()
                )
        asm_code = "\n".join(asm_lines)

        # 构建提示词
        mmio_list = ", ".join([hex(addr) for addr in mmio_addresses])

        prompt = f"""你是 Cortex-M/ARM 固件等待循环分析器。目标：只通过一个 MMIO 读取值，让循环退出。

Loop:
- bb: {hex(loop_address)}
- iterations: {execution_count}
- candidate_mmio: {mmio_list}
- loop_bbs: {", ".join(hex(bb) for bb in context_bbs)}

ASM:
{asm_code}

Recent MMIO access sequence:
{chr(10).join(recent_mmio_lines) if recent_mmio_lines else "  <none>"}

Rules:
1. 只能从 candidate_mmio 中选一个地址。
2. 如果存在比较/测试后回跳，必须让回跳条件不成立；注意 MMIO 读取、比较和回跳可能分布在不同 BB 或 helper 调用返回后。
3. `required_value` 必须是 32 位无符号值，优先最小最简单值：0x0, 0x1, bit mask, compare immediate。
4. 如果能识别位测试或状态寄存器，返回 `status_mask`；例如 TST #0x20 就返回 0x20。未知则返回 null。
5. `semantic_kind` 只能从 polling_ready, polling_busy_clear, ack_after_write, data_available, unknown 中选择。
6. `reason` 只能用一行短句说明“为什么这个值会退出循环”。
7. 只返回 JSON，不要 markdown，不要代码块，不要额外文字。

Return:
{{"mmio_address":"0x40021000","required_value":"0x02000000","status_mask":"0x02000000","semantic_kind":"polling_ready","confidence":0.8,"reason":"..."}}
"""

        try:
            response = self._call_llm_json(prompt)

            result = extract_response_text(response)

            data = self._parse_json_response(result)
            if not data:
                raise ValueError(f"无法解析JSON: {result!r}")

            mmio_addr = self._parse_u32(data.get("mmio_address"))
            value = self._parse_u32(data.get("required_value"))
            if mmio_addr is None or value is None:
                raise ValueError(f"LLM返回了无效地址/数值: {data}")
            if mmio_addr not in mmio_addresses:
                raise ValueError(f"LLM返回了不在候选集合中的MMIO地址: {hex(mmio_addr)}")
            status_mask = self._parse_u32(data.get("status_mask"))
            semantic_kind = str(data.get("semantic_kind", "unknown") or "unknown")
            confidence = self._parse_float(data.get("confidence"), default=0.0)
            self.last_solution_metadata = {
                "mmio_address": mmio_addr,
                "required_value": value,
                "status_mask": status_mask,
                "semantic_kind": semantic_kind,
                "confidence": confidence,
                "reason": str(data.get("reason", "")),
            }

            logger.info(f"[DeadlockSolver] LLM求解成功:")
            logger.info(f"  MMIO[{hex(mmio_addr)}] = {hex(value)}")
            logger.info(f"  原因: {data.get('reason', 'N/A')}")
            self.inference_history.append({
                "loop_address": hex(loop_address),
                "execution_count": int(execution_count),
                "mmio_candidates": [hex(addr) for addr in sorted(mmio_addresses)],
                "llm_full_response": result,
                "mmio_address": hex(mmio_addr),
                "required_value": hex(value),
                "status_mask": hex(status_mask) if status_mask is not None else None,
                "semantic_kind": semantic_kind,
                "confidence": confidence,
                "reason": str(data.get("reason", "")),
                "method": "llm",
            })

            solved = {mmio_addr: value}
            self.solve_cache[cache_key] = solved
            return solved

        except Exception as e:
            self.last_solution_metadata = None
            logger.error(f"[DeadlockSolver] LLM求解失败: {e}")
            self.inference_history.append({
                "loop_address": hex(loop_address),
                "execution_count": int(execution_count),
                "mmio_candidates": [hex(addr) for addr in sorted(mmio_addresses)],
                "error": str(e),
                "method": "llm_error",
            })

        self.solve_cache[cache_key] = None
        return None
