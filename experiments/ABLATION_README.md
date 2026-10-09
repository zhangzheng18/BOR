# LSGEmu Ablation Entry Points

The main paper ablations are contribution-level, not implementation-module
toggles.  They run the same interleaved scheduler as `lsgemu/lsgemu.py`; the
static valid-BB denominator, Unicorn execution hook, replay validation, and
strict coverage oracle stay unchanged.

## Main Contribution Ablations

- `lsgemu_no_semantic_obligation.py`
  - Paper label: `Without Semantic Obligation`.
  - Question: is semantic-guided obligation discovery necessary?
  - Ablation schema: version 2. Direct and serial entry points write it under
    `no_semantic_obligation_v2`, so legacy reports cannot be silently reused.
  - Removes stagnation-to-obligation conversion, typed semantic category
    state, semantic branch hotsets and failed-direction feedback, semantic
    target scoring, scheduler feedback and feedback-root seeding,
    obligation-driven candidate generation, semantic entry/flush/drain stages,
    and semantic vector cleanup.
  - Cross-BB causal tracing and dynamic constraint recovery remain enabled;
    they are used by the category-blind branch/MMIO fallback instead of being
    selected by semantic obligations.
  - Preserves baseline reset-entry execution with generic MMIO readiness,
    ordinary branch reservoir replay, structurally ranked CFG frontier and
    switch/dispatch replay, global branch-MMIO fallback, contextual event
    mechanisms, and direct-call/stream/thread/deadline-drain replay.  These
    preserved downstream mechanisms may use their own concrete observed
    contexts, but receive no semantic categories, hotsets, or scheduler targets
    from the removed obligation-discovery layer.

- `lsgemu_no_scoped_replay.py`
  - Paper label: `Without Scoped Replay`.
  - Question: is evidence-scoped replay necessary?
  - Ablation schema: version 2. Direct and serial entry points write it under
    `no_scoped_replay_v2`. Legacy A2 reports used branch-specific snapshots and
    scoped constraints and must not be pooled with this version.
  - Removes branch-specific snapshot/provenance replay, learned read-PC and
    read-occurrence constraints, branch-occurrence replay, target-scoped
    frontier/dispatch replay, successor-guided targeted replay, target-scoped
    vector/frontier cleanup, and force-free naturalization of obligations
    discovered by scoped replay.
  - Preserves baseline reset-entry execution, a task-local address-global MMIO
    fallback replayed from reset entry, basic contextual ISR/event replay,
    untargeted direct-call/stream/thread replay, and deadline direct-call
    fallback over entry-reachable targets. Dynamic memory-read repairs have no
    sound address-global counterpart and are therefore absent from A2.

- `lsgemu_no_context_event_replay.py`
  - Paper label: `Without Context-aware Event Recovery`.
  - Question: is event context necessary?
  - Removes contextual ISR replay, ISR reservoir replay, vector cleanup,
    ISR-frontier replay, stream-input event replay, and RTOS thread-entry replay.
  - Preserves baseline reset-entry execution, generic branch replay,
    branch-MMIO repair, frontier/dispatch replay, and direct-call/summary-return
    replay.

Each report records the active policy under
`phase_metadata.ablation_config.fallback_policy`.  `fallback_policy.active`
uses the paper labels above, and `fallback_policy.effective_internal_switches`
records the lower-level stage switches derived from the contribution-level
ablation.

## Evidence Integrity Fields

The full and ablation entry points emit the same evidence fields:

- `coverage_by_evidence`: E0 reset-entry, E1 force-free replay under consumed
  environment facts, E2 control-forced replay, and E3 execution-context
  recovery. Total coverage continues to include all four classes.
- `coverage_evidence_summary`: the discovery union and its witness-backed,
  E2-only, E3-only, E2/E3-shared, and unclassified partitions. The reported
  partitions are mutually exclusive; raw `coverage_by_evidence` sets may
  overlap because one BB can be observed by several replay classes.
  `valid_witness_coverage_rate`
  is the strict force-free lower bound over the unchanged valid-BB denominator.
- `path_naturalization_summary`: forced-path obligations, externalized MMIO or
  declared external-input facts, value-domain and compound candidate sets,
  force-free attempts, occurrence retargets, inherited discovery-prefix traces,
  selected upstream repair edges, prefix-preserving input refinements, distinct
  witnessed discovery contexts, prefix advances, promotions, local-solver and
  phase-local LLM costs, and categorized failure reasons. Reports with causal
  refinement use schema `path_naturalization.v5`; v1-v4 reports must not be
  pooled into the same naturalization-rate calculation.
- `snapshot_provenance`: effective and capture-time evidence counts for root,
  direct-call, stream, reservoir-prefix, and interrupt-context snapshots.
  `legacy_or_unknown_entries` should be zero in newly generated reports.
- `snapshot_resources`: retained references versus unique snapshot objects,
  primary page-store bytes/pages, configured limits, and eviction/truncation
  counters. `LSGEMU_BRANCH_SNAPSHOT_CURRENT_LIMIT` and
  `LSGEMU_DIRECT_CALL_SNAPSHOT_VARIANTS_PER_ADDRESS` default to `0`, preserving
  the historical unbounded behavior unless a campaign explicitly enables a
  host-memory policy.
- `strict_real_entry_replayable`: true only when every counted BB has E0/E1
  support and no counted phase violates the reset-entry replay contract.
- `causal_input_model.all_emulator_aggregate.unretained_event_ids`: the gap
  between each emulator's final event-id watermark and retained bounded event
  records. Snapshot rewind can contribute to this gap, so it is deliberately
  not labeled as event-buffer truncation.

Failed naturalization never removes E2/E3 coverage. It records why a candidate
could not be reproduced naturally so missing peripheral, input, transaction,
or prefix-state models can be analyzed separately from exploration coverage.
The frozen v3 defaults are four causal sources, four values per source, four
provenance-supported compound candidates, and eight concrete attempts per path.
They are exposed as `--path-naturalization-max-sources-per-edge`,
`--path-naturalization-values-per-source`,
`--path-naturalization-max-compound-candidates`, and
`--path-naturalization-max-attempts-per-path`.
The old `--path-naturalization-max-facts-per-edge` spelling remains a command-line
alias only for compatibility with existing launch scripts.

## Fingerprinted Serial Campaign

Use a new output root for every paper round. The four commands remain strictly
serial: a child process must exit before the next ELF starts.

```bash
python3 experiments/run_lsgemu_serial_eval.py 1 --time 1440 \
  --output-root .lsgemu_runs/evaluation_serial_round3
python3 experiments/run_lsgemu_serial_eval.py 2 --time 1440 \
  --output-root .lsgemu_runs/evaluation_serial_round3
python3 experiments/run_lsgemu_serial_eval.py 3 --time 1440 \
  --output-root .lsgemu_runs/evaluation_serial_round3
python3 experiments/run_lsgemu_serial_eval.py 4 --time 1440 \
  --output-root .lsgemu_runs/evaluation_serial_round3
```

The driver records a campaign fingerprint over the mode, arguments,
`LSGEMU_*` environment, and SHA-256 of execution/configuration sources. Each
job also records the firmware SHA-256. Credential-like `LSGEMU_*` values are
stored only as SHA-256 identities. External `--config` and `--run-profile`
files are fingerprinted by content and rechecked before every child. Options
owned by the campaign (`--firmware`, time, output directory, mode, and the three
ablation switches) cannot be overridden through child passthrough arguments.
An existing report is skipped only when
its status file matches the campaign, firmware, source-tree, runtime-toolchain,
and full campaign-toolchain identities; legacy reports without this metadata
are rerun instead of being silently mixed into a new campaign. Firmware-specific
valid-BB/toolchain state is recorded separately under
`prepared_toolchain_identity_hash`, `valid_bb_denominator_hash`, and
`static_cache_identity_hash`. The driver also
recomputes the source fingerprint before each child and aborts if execution or
configuration, LLM, credential, or environment identity changed during a
campaign. While each child runs, the
parent samples the Linux `/proc` process tree once per second and stores peak
aggregate RSS, root-process high-water RSS, maximum tree size, and minimum host
memory/swap availability in `resource_usage` inside the job status and campaign
summary. Every child starts in its own process group; timeout or interruption
terminates the whole group before the serial driver can start another ELF. A
job is successful only when this invocation rewrites a complete report, so a
stale report left by an older run cannot mask a failed child.

Each invocation writes to an attempt-specific directory and tags progress,
status, and final artifacts with the same attempt ID. The analyzer accepts a
causal full/ablation pair only when both final reports are complete and their
firmware, denominator, source-tree, runtime-toolchain, full campaign-toolchain,
and available static-cache identities agree. Failed runs remain visible through
same-attempt durable checkpoint lower bounds; old unscoped artifacts are
descriptive only.

Mode 2 writes under `no_semantic_obligation_v2`, and mode 3 writes under
`no_scoped_replay_v2`. The corresponding legacy directories remain readable
for historical analysis but are not pooled with the strict v2 ablations.
