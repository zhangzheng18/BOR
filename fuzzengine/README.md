# fuzzengine

This package adds a lightweight crash-discovery loop on top of LSGEmu. It is
not intended to be the primary coverage-improvement mechanism. LSGEmu still
owns MCU rehosting, MMIO/IRQ/path replay, and coverage exploration; fuzzengine
uses concrete seeds and replay evidence to find, classify, and package crash
candidates.

It has two layers:

- Campaign crash discovery: manage a seed corpus, mutate stream-input seeds,
  run LSGEmu in supervised subprocesses, bucket crash candidates, and replay
  new crash buckets for determinism.
- Crash triage: classify LSGEmu run results, Unicorn runtime events, and native
  subprocess failures into firmware candidates, hangs, model artifacts,
  tooling/runtime failures, and likely trigger sources.

The campaign is intentionally conservative. It reports crash candidates and
evidence packages; CVE-grade or safety-impact claims still require deterministic
replay and model-fidelity review.

## Crash Discovery Campaign

Recommended two-stage run:

```bash
# Stage 1: run a long LSGEmu simulation once and keep the profile artifacts.
python3 lsgemu/lsgemu.py /path/to/firmware.elf \
  --time 1440 \
  --mode interleaved \
  --output-dir /tmp/lsgemu_24h_profile

# Stage 2: build a reusable crash-fuzzing profile from the long-run report.
python3 -m fuzzengine.profile \
  --report /tmp/lsgemu_24h_profile/firmware_interleaved_report.json \
  --out /tmp/lsgemu_24h_profile/fuzz_profile.json

# Stage 3: run crash-focused short replay/fuzzing using the profile.
python3 -m fuzzengine.campaign \
  --firmware /path/to/firmware.elf \
  --profile /tmp/lsgemu_24h_profile/fuzz_profile.json \
  --work-dir /tmp/lsgemu_crash_fuzz \
  --seed /path/to/seeds \
  --iterations 1000 \
  --lsgemu-path lsgemu/lsgemu.py
```

`--profile` can also point directly to an LSGEmu `*_interleaved_report.json`;
the campaign will build the profile in memory.

Single-stage fallback run:

```bash
python3 -m fuzzengine.campaign \
  --firmware /path/to/firmware.elf \
  --work-dir /tmp/lsgemu_fuzz_work \
  --seed /path/to/seeds \
  --iterations 100 \
  --time-minutes 2 \
  --timeout-seconds 420 \
  --lsgemu-path lsgemu/lsgemu.py
```

Useful options:

- `--profile`: fuzzengine profile JSON or a long-run LSGEmu
  `*_interleaved_report.json`. Enables short crash-focused replay settings.
- `--seed`: seed file or directory. Repeat it for multiple seed roots.
- `--dictionary`: protocol token. Use `hex:414243` for raw bytes.
- `--run-profile`: LSGEmu YAML/JSON run profile.
- `--crash-config`: crash detector JSON config.
- `--extra-arg`: one raw argument forwarded to LSGEmu. Repeat for option/value pairs.
- `--replay-crashes`: deterministic replay count for each new crash bucket.

Work directory layout:

- `corpus/queue`: all imported and generated seeds.
- `corpus/interesting`: seeds that produced new BB coverage. This is retained
  as replay context, not as the primary objective.
- `lsgemu_runs`: per-seed LSGEmu reports and coverage progress.
- `findings`: crash/hang/artifact buckets with seed, run result, and triage report.
- `replays`: replay validation attempts for new crash buckets.
- `progress.jsonl`: one line per campaign iteration.
- `summary.json`: final campaign summary.

The campaign injects each seed through `LSGEMU_STREAM_EXTRA_SEEDS_HEX` and
enables `LSGEMU_CRASH_TRIAGE`, `LSGEMU_RUNTIME_CRASH_MONITOR`, and phase BB-list
recording by default.

When a profile is supplied, the campaign records `profile_context` in
`summary.json`, applies the profile's recommended short replay arguments, and
sets `LSGEMU_FUZZ_PROFILE_REPORT`/`LSGEMU_FUZZ_PROFILE_SHA256` for provenance.
The current implementation reuses the long-run report as guidance and
provenance; it does not yet restore Unicorn snapshots directly from the 24h run.

## Post-Simulation Security Test

`security_campaign` is the post-simulation entry point.  It consumes the
completed LSGEmu report, inventories observed input/peripheral/IRQ/indirect-
control/sink surfaces, reuses the report as a guidance profile, runs the
existing isolated crash campaign, and writes one conservative security report.
It does not change the emulator's coverage oracle or turn a crash into a
confirmed vulnerability.

```bash
python3 -m fuzzengine.security_campaign \
  --simulation-report /tmp/lsgemu_24h_profile/firmware_interleaved_report.json \
  --firmware /path/to/firmware.elf \
  --work-dir /tmp/lsgemu_security/firmware \
  --seed /path/to/seeds \
  --iterations 1000 \
  --time-minutes 0.25 \
  --replay-crashes 3
```

The command refuses to mix a report and ELF with different identities unless
`--allow-identity-mismatch` is explicitly supplied.  It produces:

- `security_report.json` and `SECURITY_REPORT.md`: attack-surface inventory,
  campaign summary, finding buckets, replay stability, and missing evidence;
- `security_profile.json`: the guidance-only profile derived from the simulation
  report;
- the existing `findings/` and `replays/` evidence packages.

Use `--analyze-only` to build the inventory and assessment from an existing
campaign directory without running more firmware.  The finding classes follow
the conservative vocabulary `reachable_bug_needs_impact`,
`model_or_harness_artifact`, `tooling_or_runtime_failure`, and
`unreachable_under_current_model`; the automated report never emits a strict
confirmed-vulnerability result.

## Trigger Source Classes

Every `CrashReport` includes `trigger_source`, `trigger_confidence`, and
`trigger_evidence`:

- `external_input`: the seed was actually consumed by a modeled stream/API input
  path, such as UART/network/protocol bytes. These are usually the easiest to
  reproduce on real devices.
- `mmio_peripheral_environment`: the crash is tied mainly to MMIO status/data
  reads or writes. Treat this as peripheral/environment stress evidence unless
  a real external stimulus can be mapped to that MMIO state.
- `irq_or_interrupt_environment`: the crash depends mainly on interrupt timing
  or ISR context injection.
- `external_input_plus_peripheral_state`: both external input and MMIO/IRQ state
  contributed. This is common for protocol paths gated by device-ready flags.
- `model_or_harness_artifact`: the stop is more likely caused by incomplete
  memory/MMIO/ISR/modeling.
- `tooling_or_runtime_failure`: Python/Unicorn/native process failure.
- `unknown`: insufficient evidence to attribute the trigger.

This distinction matters for validation. External-input crashes can often be
replayed with protocol traffic on hardware; MMIO/IRQ-triggered crashes are
closer to robustness or stress testing until the physical stimulus and timing
are identified.

Crash classes:
- `firmware_crash`: invalid memory write/fetch outside configured firmware/MMIO mappings, configured crash point, fault handler PC, stack guard/abort, PC outside executable ranges.
- `hang`: timeout or instruction-limit style hangs, depending on config.
- `model_or_harness_artifact`: unproven dynamic code, invalid mapping, restore/state mismatch, unmapped MMIO/configured memory, invalid instruction outside configured code.
- `tooling_or_runtime_failure`: native emulator process signal such as SIGSEGV.

## How Crashes Are Found

Use two layers together:

- Runtime discovery: install `RuntimeCrashMonitor` on a Unicorn instance. It hooks invalid memory events and code execution, records PC/LR/SP/xPSR, access address, trace tail, configured crash points, and fault handlers, then stops emulation.
- Post-hoc triage: run `CrashDetector` over LSGEmu JSON reports, direct-call/debug records, or subprocess return codes. This catches crashes already written into historical reports and separates host failures from guest failures.

Example report scan:

```bash
python3 -m fuzzengine.crash_detector \
  --lsgemu-report /path/to/firmware_interleaved_report.json \
  --config fuzzengine/crash_detector_config.example.json \
  --out-dir /tmp/lsgemu_crash_triage
```

Example runtime integration:

```python
from fuzzengine import CrashDetector, CrashStore, RuntimeCrashMonitor

monitor = RuntimeCrashMonitor(uc, detector_config)
monitor.install()
try:
    uc.emu_start(start_pc, end_pc, timeout=timeout_us)
finally:
    run_result = monitor.as_record()
    monitor.uninstall()

report = CrashDetector(detector_config).detect(run_result, seed_bytes=seed)
if report.is_crash or report.is_hang:
    CrashStore("/tmp/workdir").record(report, seed_bytes=seed)
```

Example subprocess supervision for native Unicorn/runtime crashes:

```bash
python3 -m fuzzengine.subprocess_runner \
  --config fuzzengine/crash_detector_config.example.json \
  --timeout 300 \
  --firmware /path/to/firmware.elf \
  --out /tmp/replay_supervised.json \
  -- python3 lsgemu/lsgemu.py /path/to/firmware.elf --time 5
```

## Distinguishing Crash Origins

The detector uses `source_layer`, `category`, `trigger_source`,
`evidence_level`, and `requires_replay_validation`:

- Firmware crash candidate: guest PC reaches a real fault handler/crash point, or an invalid access goes outside configured code/RAM/MMIO ranges. This still needs deterministic replay before it is a strict vulnerability result.
- Model or harness artifact: guest touches unmapped MMIO, unmapped configured RAM/Flash, unproven dynamic code, or an invalid instruction outside configured code. These usually mean the memory map, Thumb state, dynamic-code policy, ISR stack, or peripheral model needs improvement.
- Tooling/runtime failure: the emulator subprocess exits by SIGSEGV/SIGABRT/SIGBUS/SIGILL. This is a Unicorn/host failure unless a replay in a fresh child process consistently reaches the same guest state before the native crash.

## Verification Policy

Automated triage finds candidates. CVE-grade or safety-impact claims require
replay validation: same seed/input, same firmware hash, stable PC/LR/SP/access,
trace tail, and a model-fidelity check that the crash is not caused by missing
MMIO, missing memory mapping, artificial ISR state, or an impossible shortcut.

The detector intentionally separates model artifacts from firmware crashes so
that incomplete peripheral modeling does not inflate vulnerability evidence.

## Regression Tests

```bash
python3 fuzzengine/test_crash_detector.py
python3 fuzzengine/test_fuzz_campaign.py
python3 fuzzengine/test_security_campaign.py
```
