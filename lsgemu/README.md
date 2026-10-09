# LSGEmu 使用说明

LSGEmu 是一个面向 ARM Cortex-M/MCU 固件的 Unicorn 重托管与路径探索工具。它的目标不是强行跳 PC，而是在真实执行过程中记录分支、MMIO、寄存器依赖、快照和中断上下文，然后生成可回放的局部约束，继续探索更多有效基本块。

## 核心源码范围

你提到的五个目录是核心能力模块，但不是全部核心源码：

- `analysis/`：固件解析、静态分析、智能仿真器、循环分类、静态约束、状态化 MMIO、快照管理。
- `register_tracer/`：运行时追踪寄存器和内存来源，把分支条件回溯到 MMIO/RAM 读取点。
- `mmio_handler/`：MMIO hook、PC 作用域约束、约束文件加载、运行时访问记录。
- `llm_guide/`：Branch-MMIO 约束推断，包含规则推断、qwen-plus JSON 调用、LLM second-opinion 候选和本地语义校验。
- `isr_explorer/`：中断向量解析、ISR 上下文构造、ISR 覆盖探索。

完整运行链还依赖这些文件：

- `historical_runner.py`：主编排器，负责 baseline、ISR、Branch-MMIO、reservoir、frontier、dispatch、thread-entry、checkpoint、coverage merge。
- `lsgemu.py`：单固件统一入口的轻量封装。
- `runtime_bootstrap.py`：运行环境修正，优先加载可用的 Unicorn/OpenAI 依赖。
- `constraint_utils.py`：约束格式和规范化工具。
- `llm_json_utils.py`：LLM JSON 调用、修复和容错解析。
- `run_elfmultifuzz_*.py` / `run_*campaign*.py`：批量实验入口。
- `stat_*.py` / `merge_*.py` / `summarize_*.py`：覆盖率统计和实验汇总。

一般来说，`test_*` 是回归/冒烟测试，`validate_*` 是安全候选或厂商固件的验证 harness，`analyze_*` 是诊断脚本，不属于主工具的核心运行路径。

## 工作流程

1. 静态准备：解析 ELF/bin，恢复基本块、分支、比较指令、MMIO 访问和入口/向量信息。
2. Baseline 执行：从入口真实执行，记录覆盖 BB、MMIO 历史、寄存器依赖、分支事件和快照。
3. 循环处理：识别等待循环，优先用静态/运行时规则推断 PC 作用域 MMIO 约束。
4. Branch-MMIO 探索：对真实遇到的分支，回溯其依赖的 MMIO/RAM 读取点，构造候选约束并从快照或入口回放。
5. 中断/上下文探索：在合适的执行上下文下触发 ISR、线程入口、dispatch 和 frontier 任务。
6. replay 验证：候选只有在 control replay 不能达到目标、result replay 能达到目标或新增有效 BB 时才被接受。
7. 结果落盘：输出覆盖报告、约束文件、分支 catalog、LLM history、progress/checkpoint 和 reservoir 状态。

## LLM 的真实作用

LLM 不是可信执行核心，也不会直接决定覆盖率。当前逻辑是：

- 规则可确定时，先用本地语义规则求值，例如 `TST mask + BNE/BEQ`、`CMP imm + BEQ/BNE/BHI/BLS`。
- LLM 只产生候选 MMIO 值，必须通过 32 位范围校验、本地分支语义校验和 replay/control 校验。
- 如果规则候选 replay 失败，`LLMGuide.infer_alternative_constraint()` 会把规则候选、失败上下文、读取 PC、分支上下文和可避免值一起发给 qwen-plus，让它给出 second-opinion 候选。
- 如果某个分支语义只有唯一正确值，LLM 应该返回同值并提示更可能是 read PC、状态前缀、快照或路径问题，而不是编造新值。

相关环境变量：

```bash
export LSGEMU_FORCE_LLM_BRANCH_INFERENCE=1
export LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS=20
export LSGEMU_BRANCH_MMIO_LLM_SECOND_OPINION=1
export LSGEMU_BRANCH_MMIO_LLM_SECOND_OPINION_BUDGET=8
```

建议默认只开启 second-opinion，不建议无限制强制 LLM。强制 LLM 适合调试提示词和 attribution，不适合作为最终覆盖率主策略。

## Direct Call 策略

`direct_call_continuation` 和 `direct_call_summary_return` 是两条不同的路径：

- `direct_call_continuation`：恢复到入口派生的 callsite 快照后，真正执行 callee 和后续返回路径。
- `direct_call_summary_return`：只有在 concrete callee continuation 已经证明该 callsite 持续阻塞、超时或零收益时，才允许跳过 callee，从 return 后继续执行。

当前默认策略是先 concrete、后 summary。也就是说，summary-return 现在是 fallback，不再是默认的积极捷径。

如果需要复现实验中更宽松的旧行为，可以显式开启：

```bash
export LSGEMU_DIRECT_CALL_SUMMARY_ALLOWLIST_EAGER=1
```

## 环境

推荐使用当前机器上的 anaconda Python，因为它同时有 `unicorn` 和新版 `openai`：

```bash
/opt/artifact/anaconda3/bin/python -c "import unicorn, openai; print(unicorn.__version__, openai.__version__)"
```

LLM 配置文件默认读取：

```text
/opt/artifact/artifact/LLM.yaml
```

示例格式：

```yaml
llm:
  model: qwen-plus
  api_key: "..."
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

如果不用 LLM，工具仍会走规则推断和 replay 验证。

## 单固件运行

默认入口会运行完整 interleaved pipeline，默认时间预算是 60 分钟，baseline 不再设置指令数上限：

```bash
cd /opt/artifact
/opt/artifact/anaconda3/bin/python -m lsgemu.lsgemu /path/to/firmware.elf
```

Gateway 示例：

```bash
/opt/artifact/anaconda3/bin/python -m lsgemu.lsgemu \
  /opt/artifact/artifact/testcase/real_tests/P2IM.Gateway.elf \
  --time 60
```

`--max-instructions 0` 表示 baseline 只受 wallclock、停滞检测和内部安全条件约束；这是默认值。旧的四阶段轻量入口仍可用于快速对比：

```bash
/opt/artifact/anaconda3/bin/python -m lsgemu.lsgemu \
  /opt/artifact/artifact/testcase/real_tests/P2IM.Gateway.elf \
  --mode simple --time 20 --max-instructions 500000
```

批量 elfmultifuzz 运行：

```bash
/opt/artifact/anaconda3/bin/python lsgemu/run_elfmultifuzz_interleaved_strict_campaign.py \
  --root /opt/artifact/benchmarks/elfmultifuzz \
  --output-root /tmp/lsgemu_elfmultifuzz_run \
  --minutes 120 \
  --jobs 1
```

当前机器不建议高并发运行，历史上并发过高会触发 IO/swap 压力。优先 `--jobs 1`。

## 常见输出

单个运行目录通常包含：

- `*_interleaved_report.json`：主覆盖和阶段统计。
- `*_strict_checkpoint_union_report.json`：从 checkpoint/progress 合并出的严格覆盖报告。
- `*_lsgemu_constraints.json`：持久化约束，重点看 `read_pc,address,value,constraint_pc`。
- `*_branch_catalog.json`：动态分支事件、分支 occurrence、候选路径。
- `*_llm_history.json`：规则/LLM 推断历史、prompt 相关上下文、返回值、本地校验结果。
- `*_coverage_progress.jsonl`：时间序列覆盖增长，用于画覆盖曲线。
- `reservoir_state.json` / `progress.json`：reservoir 队列和可恢复状态。

覆盖率统计：

```bash
/opt/artifact/anaconda3/bin/python lsgemu/stat_elfmultifuzz_valid_coverage.py \
  --root /tmp/lsgemu_elfmultifuzz_run
```

## 测试

语法检查：

```bash
python3 -m py_compile \
  lsgemu/llm_guide/llm_guide.py \
  lsgemu/historical_runner.py \
  lsgemu/mmio_handler/enhanced_mmio_handler.py \
  lsgemu/register_tracer/register_tracer.py
```

本地语义测试：

```bash
python3 lsgemu/test_branch_constraint_validation.py
python3 lsgemu/test_branch_mmio_pruning.py
python3 lsgemu/test_register_tracer_memory_provenance.py
python3 -m unittest lsgemu.test_local_constraint_recovery
python3 -m unittest lsgemu.test_path_naturalization
```

qwen-plus live 测试：

```bash
/opt/artifact/anaconda3/bin/python lsgemu/test_qwen_live.py
```

## 调试建议

- 覆盖率低时，先看 `*_interleaved_report.json` 的 `phase_metadata`，确认卡在 baseline、loop、branch-mmio、ISR 还是 reservoir。
- 如果 LLM 返回值没有效果，先看 `*_llm_history.json` 的 `local_validation`，再看 `attempt_records` 的 `constraint_feedback`。若 `matched_constraint_reads=0`，问题通常不是值，而是 `read_pc`、快照前缀或路径没有走到读取点。
- 如果 `control_already_satisfies_target` 很多，说明候选分支在 control replay 已经可达，不应计入新增约束。
- 如果 `no_effect_same_outcome` 且 `all_constraint_reads_matched=true`，才值得让 LLM 或规则生成另一个满足同一分支语义的值。
- `path_naturalization.v5` 会在 `phase_metadata.path_naturalization` 中记录动态跨 BB 输入切片、联合 occurrence 候选、前缀保持的输入值精化、多 discovery context、复合候选、occurrence retarget、局部 solver 时间和失败分类；不要和 v1-v4 的自然化率混算。
- 约束尽量使用 `<read_pc, address, value, constraint_pc>` 四元组，避免全局地址约束污染其他路径。

## 当前边界

LSGEmu 能较好处理直接 MMIO 等待循环、局部分支约束、中断上下文和动态路径 replay。仍然困难的情况包括：复杂 RTOS 调度状态、长期跨函数数据依赖、外设事务阶段机、真实物理输入条件、以及需要持久化 NVM/flash 状态的路径。LLM 可以帮助解释和补充候选，但不能替代 replay 验证。
