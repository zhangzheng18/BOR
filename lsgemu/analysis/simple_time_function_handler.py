"""
简单的时间函数处理器 - 全局软时钟版本

核心思想：
1. 维护一个全局软时钟（每个BB自增）
2. 任何时间函数调用，直接返回软时钟值
3. 简单、通用、高效

备注：之前的方案（手动指定+每次递增）也可以工作，但这个方案更优雅
"""

import logging
from unicorn.arm_const import UC_ARM_REG_R0

logger = logging.getLogger(__name__)


class SimpleTimeFunctionHandler:
    """简单的时间函数处理器 - 全局软时钟"""

    def __init__(self):
        # 全局软时钟 - 每个BB自增
        self.soft_tick = 0

        # 统计信息
        self.call_count = 0

        # 已知的时间函数及其返回指令地址
        self.time_functions = {
            0x08003764: 0x0800376a,  # millis: 返回指令在0x0800376a
            0x08008b00: 0x08008b06,  # GetCurrentMilli: 返回指令在0x08008b06
        }

    def tick(self):
        """
        软时钟自增

        在每个BB执行时调用
        """
        self.soft_tick += 1

    def is_time_function_return(self, address):
        """
        判断当前地址是否是时间函数的返回指令

        Args:
            address: 当前PC地址

        Returns:
            int: 如果是时间函数返回，返回函数地址；否则返回None
        """
        for func_addr, return_addr in self.time_functions.items():
            if address == return_addr:
                return func_addr
        return None

    def handle_function_return(self, uc, address):
        """
        处理函数返回

        在函数返回指令执行时修改R0寄存器的值
        直接返回全局软时钟值

        Args:
            uc: Unicorn实例
            address: 当前PC地址
        """
        func_addr = self.is_time_function_return(address)
        if func_addr is None:
            return

        # 直接返回软时钟值
        uc.reg_write(UC_ARM_REG_R0, self.soft_tick)

        self.call_count += 1
        logger.debug(f"时间函数 0x{func_addr:08x} 返回 {self.soft_tick}")

        return self.soft_tick

    def get_statistics(self):
        """获取统计信息"""
        return {
            'soft_tick': self.soft_tick,
            'time_function_calls': self.call_count,
            'tracked_functions': len(self.time_functions)
        }
