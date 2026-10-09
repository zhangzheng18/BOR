"""
简单等待循环处理器

处理类似这样的死循环：
    LDRH.W  R3, [this,#0x13E]
    CMP     R3, R2
    BEQ     loc_8008C04

解决方案：往 this+0x13E 写入不等于R2的值
"""

import logging
import os
from unicorn.arm_const import *

logger = logging.getLogger(__name__)


class SimpleWaitLoopHandler:
    """简单等待循环处理器"""

    def __init__(self):
        # 已知的简单等待循环
        self.wait_loops = {
            0x08008c04: {
                'description': 'LDRH [this,#0x13E]; CMP R3,R2; BEQ',
                'handled': False
            }
        }
        try:
            self.progressive_iterations = max(
                0,
                int(os.environ.get("LSGEMU_SIMPLE_WAIT_PROGRESSIVE_ITERATIONS", "32")),
            )
        except ValueError:
            self.progressive_iterations = 32
        self.handle_counts = {}

    def _candidate_this_pointers(self, uc):
        """Return plausible object pointers for member-field wait loops."""
        register_order = [
            UC_ARM_REG_R0,
            UC_ARM_REG_R1,
            UC_ARM_REG_R2,
            UC_ARM_REG_R3,
            UC_ARM_REG_R4,
            UC_ARM_REG_R5,
            UC_ARM_REG_R6,
            UC_ARM_REG_R7,
            UC_ARM_REG_R8,
            UC_ARM_REG_R9,
            UC_ARM_REG_R10,
            UC_ARM_REG_R11,
            UC_ARM_REG_R12,
            UC_ARM_REG_SP,
        ]
        candidates = []
        seen = set()
        for reg in register_order:
            try:
                value = int(uc.reg_read(reg)) & 0xFFFFFFFF
            except Exception:
                continue
            if value == 0 or value in seen:
                continue
            seen.add(value)
            candidates.append(value)
        return candidates

    def handle_wait_loop(self, uc, loop_head):
        """
        处理简单等待循环

        策略：
        修改内存值，而不是寄存器
        因为循环会重新从内存加载

        指令分析：
        LDRH.W  R3, [this,#0x13E]  # 从内存加载
        UXTH    R3, R3
        UXTB    R2, R4
        CMP     R3, R2
        BEQ     loc_8008C04

        解决方案：往内存 [this+0x13E] 写入不等于R4的值

        Args:
            uc: Unicorn实例
            loop_head: 循环头地址
        """
        if loop_head not in self.wait_loops:
            return False

        # 标记为已处理（用于统计）
        self.wait_loops[loop_head]['handled'] = True
        handle_count = self.handle_counts.get(loop_head, 0) + 1
        self.handle_counts[loop_head] = handle_count

        try:
            r4 = int(uc.reg_read(UC_ARM_REG_R4)) & 0xFFFFFFFF
            expected_value = r4 & 0xFF
            preferred_candidates = []
            fallback_candidates = []

            for this_ptr in self._candidate_this_pointers(uc):
                target_addr = (this_ptr + 0x13E) & 0xFFFFFFFF
                try:
                    current_value = uc.mem_read(target_addr, 2)
                    current_value = int.from_bytes(current_value, 'little')
                except Exception:
                    continue
                candidate = (this_ptr, target_addr, current_value)
                if current_value == expected_value or current_value == (expected_value & 0xFFFF):
                    preferred_candidates.append(candidate)
                else:
                    fallback_candidates.append(candidate)

            for this_ptr, target_addr, current_value in preferred_candidates + fallback_candidates:
                try:
                    # 比较实际是 UXTH(load) == UXTB(R4)。前若干次写
                    # UXTB(R4)+1，让固件自然推进扫描并产生更多真实分支；
                    # 超过上限后写 0xffff，避免该等待循环无界拖慢 replay。
                    if handle_count <= self.progressive_iterations:
                        new_value = ((expected_value) + 1) & 0xFFFF
                    else:
                        new_value = 0xFFFF
                    uc.mem_write(target_addr, new_value.to_bytes(2, 'little'))

                    logger.info(f"✓ 处理简单等待循环 @ 0x{loop_head:08x}")
                    logger.info(f"  this={this_ptr:08x}, [this+0x13E]={current_value:04x} -> {new_value:04x}")
                    logger.info(f"  R4={r4:08x}")

                    return True
                except Exception:
                    continue

            logger.warning("无法找到可写的this指针")
            return False

        except Exception as e:
            logger.error(f"处理简单等待循环失败: {e}")
            return False

    def is_wait_loop(self, loop_head):
        """判断是否是已知的简单等待循环"""
        return loop_head in self.wait_loops

    def get_statistics(self):
        """获取统计信息"""
        handled_count = sum(1 for info in self.wait_loops.values() if info['handled'])
        return {
            'total_wait_loops': len(self.wait_loops),
            'handled_wait_loops': handled_count
        }
