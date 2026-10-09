# 死循环问题分析

## 问题

### 第一次测试结果
- 覆盖率: 15.79% (627/3972 BBs)
- 成功控制: 2个死循环（0x80057fc, 0x8003764）
- **仍有死循环**: 3个新循环（18000+次）

## 原因分析

### 新死循环代码
```assembly
# BB @ 0x80036c2 (执行18857次)
0x80036c0: movs r3, #0x0        ; i = 0
0x80036c2: cmp r3, #0x3b        ; while (i < 59)
0x80036c6: ldr r2, [0x080036d4] ; r2 = string_base
0x80036c8: ldrsb r2, [r2,r3]    ; r2 = string[i]
0x80036ca: cmp r0, r2           ; if (r0 == string[i])
0x80036cc: beq 0x080036bc       ;   break
0x80036ce: adds r3, #0x1        ; i++
0x80036d0: b 0x80036c2          ; continue
```

**这是`strchr`函数！** 在字符串中查找字符。

### 为什么LLM没解决？

**1. 循环类型不匹配**
- 旧策略: 只对`polling`类型干预
- 这个循环: `initialization`或`unknown`类型
- 结果: 不触发干预

**2. 没有MMIO访问**
- polling循环: 访问MMIO寄存器
- strchr循环: 只访问普通内存
- LLM求解器: 专门处理MMIO约束

**3. 干预条件**
```python
# 旧代码
if loop_type == LoopType.POLLING and iteration >= 100:
    return True  # 只对polling干预
```

## 解决方案

### 修改1: 扩大干预范围
```python
# 新代码
if iteration >= 100:
    return True  # 对所有类型干预
```

### 修改2: 添加强制停止
```python
elif action == "skip":
    logger.warning("未知循环类型，强制停止执行")
    self.uc.emu_stop()
```

## 预期效果

### 修改前
- polling循环: ✅ 干预（LLM求解）
- 其他循环: ❌ 不干预（继续执行18000+次）

### 修改后
- polling循环: ✅ 干预（LLM求解）
- 其他循环: ✅ 干预（强制停止）

## 测试中

**进程**: 3618953
**预期**: 
- strchr循环在100次时被停止
- 覆盖率可能下降（因为停止了执行）
- 但不会卡死18000+次

## 更好的方案

### 方案1: 智能跳过
对于非polling循环，不是停止，而是：
1. 识别循环出口条件
2. 强制满足出口条件
3. 跳出循环继续执行

### 方案2: 符号执行
对于strchr这种循环：
1. 识别为字符串查找
2. 直接返回"未找到"
3. 继续执行

### 方案3: 提高阈值
对于非polling循环，使用更高的阈值（如1000次）
