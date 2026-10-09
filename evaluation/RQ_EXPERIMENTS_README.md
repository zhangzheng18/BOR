# LSGEmu RQ1/RQ2/RQ3 Experiments

This directory contains the one-command evaluation entrypoint for the paper's
first three research questions.

## Default Paper Run

Run the 23-firmware `elfmultifuzz` corpus:

```bash
python3 evaluation/run_rq_experiments.py --rq all
```

The current deployment config uses
`/opt/artifact/benchmarks/elfmultifuzz`, which
should contain 23 ELF firmware images.  The default budgets are upper bounds,
not forced runtimes.  RQ1 uses a 24h wallclock upper bound because LSGEmu
normally reaches quiescence before that limit.  RQ2 uses equal-budget 60m
ablations.  RQ3 audits hypothesis quality from the generated reports and runs a
bounded focused LLM attribution probe only when both `LLM.yaml` and its named
credential are available. `--rq3-llm-mode force` fails before execution when
the credential is missing instead of labeling heuristic fallback as LLM output.

Long budgets still use a bounded entry baseline.  The baseline starts from the
firmware entry point and counts only Unicorn-observed valid BBs, but it has a
wallclock cap so an Arduino/PLC-style main loop cannot consume the entire 24h
budget before the diagnostic replay pipeline starts.  Progress JSONL includes
`active_emulator_observed_valid_bbs` as a diagnostic-only live counter while a
phase is still running; final coverage remains phase-end committed coverage.

Use controlled subprocess parallelism on a large machine:

```bash
python3 evaluation/run_rq_experiments.py --rq all --jobs 0
python3 evaluation/run_rq_experiments.py --rq all --jobs 4
python3 evaluation/run_rq_experiments.py --rq all --jobs 8
```

Each job is a separate LSGEmu process with isolated output, `run.log`, and
report files.  The default remains `--jobs 1` to avoid accidental I/O or swap
pressure.  `--jobs 0` selects an automatic safe value, currently capped at 4.
Requests above the default safety cap of 8 are clamped unless
`--allow-high-jobs` is explicitly set.

When `--jobs > 1`, the runner uses a guarded subprocess scheduler instead of a
plain worker pool.  `--jobs` is only the maximum number of concurrently running
LSGEmu subprocesses.  The scheduler itself is one extra Python process, so
`--jobs 8` can appear as 9 Python processes: 1 parent scheduler plus up to 8
firmware emulation subprocesses.  The parent is not counted as a firmware job.

The guarded scheduler starts conservatively and ramps up:

```text
--scheduler-initial-jobs 2
--scheduler-ramp-seconds 300
--scheduler-min-free-memory-gib 96
--scheduler-per-job-memory-gib 32
--scheduler-kill-below-free-memory-gib 48
--scheduler-max-swap-used-gib 8
--scheduler-kill-above-swap-used-gib 16
```

This matters because LSGEmu memory can grow during long runs as snapshots,
static-analysis artifacts, Unicorn mappings, coverage JSONL, and replay
diagnostics accumulate.  The scheduler therefore checks `MemAvailable`, swap
usage, and observed per-process RSS before starting another target.  If memory
falls below the emergency guard, it terminates the largest running subprocess
and marks that row as `memory_guard_terminated` instead of mixing it into normal
coverage data.

Progress JSONL defaults to count-only coverage records:

```text
--progress-coverage-list-mode none
```

This prevents long runs from repeatedly serializing thousands of BB addresses
into every interval record.  Final LSGEmu reports still contain the authoritative
coverage lists.  Use `--progress-coverage-list-mode final`, `sample`,
`on_change`, or `all` only when a specific diagnostic needs progress-time BB
lists.

For this workload, 30--40 processes is usually too aggressive even on a 96-core
server: the bottleneck is often static-analysis/cache I/O, JSONL/report writes,
temporary emulator memory pressure, and Python/Unicorn process startup rather
than raw CPU cores.

For a long paper run on the current machine, prefer:

```bash
python3 evaluation/run_rq_experiments.py --rq all --jobs 4
```

If 30--60 minutes of monitoring shows stable `MemAvailable`, no swap activity,
and low disk wait, `--jobs 8` is acceptable as an upper bound because the guarded
scheduler may still run fewer jobs when memory pressure rises.

Outputs are written under:

```text
evaluation/results/rq1_rq2_rq3_<timestamp>/
```

## Auditing Interrupted RQ1 Runs

Long RQ1 runs can be resource-censored or crash before writing a final report.
Those rows must not be treated as completed coverage.  Audit an existing run
without launching new emulation:

```bash
python3 evaluation/run_rq_experiments.py \
  --rq rq1 \
  --audit-rq1-root evaluation/results/rq1_rq2_rq3_20260624_204606
```

This generates:

```text
RQ1_coverage_reachability/rq1_outcome_audit.csv
RQ1_coverage_reachability/rq1_outcome_audit.md
RQ1_coverage_reachability/rq1_paper_completed_only.csv
RQ1_coverage_reachability/rq1_rerun_manifest.json
```

Use `rq1_paper_completed_only.csv` for final RQ1 coverage tables.  Use
`last_observed_*` fields only for censored-run diagnostics and progress curves.
Rows marked `resource_censored`, `segmentation_fault`, `stuck_zero_coverage`,
`timeout_censored`, or `still_running_or_orphaned` need rerun/debug handling
before they can support final coverage claims.

## Debug Run

Use short budgets and command inspection:

```bash
python3 evaluation/run_rq_experiments.py --rq all --target-suite p2im-three --quick
python3 evaluation/run_rq_experiments.py --rq all --target-suite p2im-three --dry-run
```

## Individual RQs

```bash
python3 evaluation/run_rq_experiments.py --rq rq1 --target-suite elfmultifuzz-23
python3 evaluation/run_rq_experiments.py --rq rq2 --target CNC --target Gateway --target PLC
python3 evaluation/run_rq_experiments.py --rq rq3 --rq3-llm-mode force --rq3-llm-call-budget 8
```

RQ2 default paper ablations are:

```text
full
no_semantic_obligation_v2
no_scoped_replay_v2
no_context_event_replay
```

These map to the paper-level mechanisms:

```text
no_semantic_obligation_v2: removes semantic obligation discovery and uses structural fallback scheduling
no_scoped_replay_v2: removes occurrence/provenance-scoped replay and keeps address-global MMIO fallback
no_context_event_replay: removes contextual ISR, stream-input event, and RTOS-thread replay
```

Legacy implementation-level diagnostics are available with
`--include-fine-ablations`; they are not the paper's contribution ablations:

```text
no_branch_reservoir
no_frontier
no_contextual_isr
no_direct_stream_thread
```

To run a subset:

```bash
python3 evaluation/run_rq_experiments.py --rq rq2 \
  --ablation no_semantic_obligation_v2 \
  --ablation no_scoped_replay_v2
```

The `full` variant is always included when RQ2 runs, because ablation deltas need
an equal-budget full baseline.

For the formal 23-firmware campaign, use
`experiments/run_lsgemu_serial_eval.py`; it additionally enforces attempt,
source-tree, toolchain, firmware, and denominator identities and starts exactly
one firmware process at a time.

## Outputs

RQ1:

```text
RQ1_coverage_reachability/rq1_summary.csv
RQ1_coverage_reachability/rq1_progress.csv
RQ1_coverage_reachability/rq1_time_to_thresholds.csv
RQ1_coverage_reachability/rq1_summary.md
```

RQ2:

```text
RQ2_mechanism_diagnostics/rq2_raw_rows.csv
RQ2_mechanism_diagnostics/rq2_progress.csv
RQ2_mechanism_diagnostics/rq2_ablation_deltas.csv
RQ2_mechanism_diagnostics/rq2_summary.md
```

RQ3:

```text
RQ3_hypothesis_quality_false_progress/rq3_payload.json
RQ3_hypothesis_quality_false_progress/rq3_by_source.csv
RQ3_hypothesis_quality_false_progress/rq3_case_rows.csv
RQ3_hypothesis_quality_false_progress/rq3_summary.md
```

Each target run also stores:

```text
command.json
run.log
run_result.json
*_interleaved_report.json
*_coverage_progress.jsonl
*_llm_history.json
```

## Counting Contract

The script does not compute coverage itself.  It extracts the authoritative
values from LSGEmu reports.  A BB is counted only if the report marks coverage as
`dynamic_unicorn_execution` and the BB belongs to `valid_basic_blocks.txt` for
valid coverage.  Static CFG recovery, branch catalogs, constraints, LLM output,
and scheduler diagnostics are only used as evidence or scheduling inputs.

RQ3 intentionally separates:

- Generated hypotheses.
- Replayed hypotheses.
- Replay-accepted hypotheses.
- Persisted/scoped constraints.
- LLM history entries and LLM-derived replay impact.
- Environment-feasibility accepted/rejected candidates.

LLM history entries alone are not treated as coverage improvement.
