# LSGEmu 当前状态

## 清理完成 ✅

### 目录结构
```
srcv4/lsgemu/
├── lsgemu.py              # 主程序（唯一入口）
├── analysis/              # 本地分析子集
├── mmio_handler/          # MMIO约束管理
├── llm_guide/             # LLM约束推断
├── register_tracer/       # 寄存器追踪
├── isr_explorer/          # 中断探索
├── README.md              # 使用文档
└── STATUS.md              # 本文档
```

### 已删除
- ✅ 所有旧版本（lsgemu_simple.py等）
- ✅ 所有日志文件
- ✅ 所有无用文件夹（advanced/deadlock_detector等）
- ✅ 所有markdown文档（除README和STATUS）

## 核心功能

### 1. 智能循环检测 ✅
- 依赖: lsgemu/analysis/intelligent_emulator.py
- 功能: 自动检测死循环（>100次）
- 集成: LLM求解器

### 2. LLM约束求解 ✅
- 测试: 3个用例全部通过
- 能力: 理解汇编，推断MMIO约束
- 集成: IntelligentEmulator

### 3. 执行日志 ✅
- 记录所有BB执行
- 统计执行次数
- 发现死循环

## 依赖关系

### srcv4内部
- lsgemu.py → mmio_handler
- lsgemu.py → llm_guide
- lsgemu.py → register_tracer
- lsgemu.py → isr_explorer

### 本地分析依赖（已复制）
- FirmwareAnalyzer（静态分析）
- IntelligentEmulator（智能仿真）

## 使用方法

```bash
cd /opt/artifact/lsgemu
python lsgemu.py firmware.elf --time 20
```

## 测试状态

**进程**: 3587213 (运行中)
**固件**: P2IM.Gateway.elf
**预期**: 覆盖率从2.47%提升到40-60%

## 下一步

1. ⏳ 等待测试完成
2. 📊 验证LLM求解器效果
3. 🔄 批量测试所有固件
4. 📝 更新文档

## 关键改进

1. ✅ 代码清理完成
2. ✅ 只保留核心文件
3. ✅ 单一入口点（lsgemu.py）
4. ✅ 完整文档（README.md）
