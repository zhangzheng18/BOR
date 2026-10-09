# LSGEmu — Anonymous Artifact

This repository is the artifact accompanying our submission. It contains the
source code of the system, the scripts used to set up and run the experiments,
and the firmware benchmarks used in the evaluation. No paper sources, internal
notes, or documentation are included.

LSGEmu is an emulation-based firmware analysis system for ARM Cortex-M devices.
It rehosts a firmware image inside a modified Unicorn engine, models the
peripheral (MMIO) surface well enough to make on-chip firmware run off-chip,
explores execution coverage under an explicit evidence contract, and — when a
target branch is observed but not reached — recovers the input constraints of
the relevant input occurrence, synthesizes candidate input values, and replays
them from a snapshot to verify that newly covered code is reachable through
legitimate external inputs rather than through engine-level intervention.

---

## 1. Layout

```
.
├── lsgemu/                 Core system: rehosting, MMIO inference, replay
│                           scheduler, ISR exploration, register provenance,
│                           constraint recovery, coverage accounting, campaign
│                           runners, validation helpers, unit tests.
├── fuzzengine/             Post-simulation security testing: seed campaigns,
│                           crash discovery and classification, subprocess
│                           monitoring.
├── evaluation/             One-command experiment harness (RQ1/RQ2/RQ3),
│                           ablation definitions, result analysis.
├── experiments/            Ablation entrypoints and comparison runners.
├── scripts/                Categorized executable wrappers:
│                             run/        single-firmware / interleaved runners
│                             campaigns/  corpus-level campaign launchers
│                             validation/ post-run validation utilities
│                             analysis/   result-analysis helpers
├── configs/                Deployment configuration and examples.
├── tests/
│   ├── unit/               Unit tests for the core modules (773 tests), plus
│   │                       standalone `check_*.py` verification scripts.
│   └── analysis/           Analysis-level tests (accounting, snapshots).
├── benchmarks/
│   ├── elfmultifuzz/       23 ELF targets (P2IM, Pretender, uEmu, WYCINWYC,
│   │                       HALucinator, RIOT/MultiFuzz).
│   ├── drone_firmware/     24 flight-controller firmware images
│   │                       (ArduPilot, PX4, Betaflight, INAV).
│   └── reachability/       Static reachability baselines and denominators:
│                           angr CFGFast basic-block enumerations, angr
│                           CFGEmulated reachable sets, and per-firmware
│                           valid-BB lists, plus the index that maps every
│                           baseline to the exact firmware SHA-256.
├── .local/unicorn-hookfix/lib/
│                           Bundled patched Unicorn 2.x shared library
│                           (libunicorn.so.2) used by the configuration and
│                           by the test suite.
├── Requirements.txt        Python dependencies.
├── LLM.yaml                LLM endpoint template (no credentials; the key is
│                           read from an environment variable).
├── LLM.deepseek.yaml       Second endpoint template used by the tests.
└── configure_paths.sh      Rewrites the anonymous path placeholder to your
                            checkout location.
```

The configuration files ship with an anonymous placeholder prefix
(`/opt/artifact`) instead of absolute paths from the authors' machine.

## 2. Setup

```bash
bash configure_paths.sh                      # point configs at this checkout
python3 -m venv .venv && source .venv/bin/activate
pip install -r Requirements.txt
```

Requirements: Linux x86-64, Python 3.10.

* **Core** — `capstone`, `pyelftools`, `PyYAML`, `numpy`, `requests`, `tqdm`.
* **Emulation engine** — the system runs against a patched **Unicorn 2.x**
  (ARM Cortex-M class). The patched shared library is bundled at
  `.local/unicorn-hookfix/lib/libunicorn.so.2`; export `LIBUNICORN_PATH` and
  `LD_LIBRARY_PATH` to that directory:
  ```bash
  export LIBUNICORN_PATH=$PWD/.local/unicorn-hookfix/lib
  export LD_LIBRARY_PATH=$LIBUNICORN_PATH:$LD_LIBRARY_PATH
  ```
  The Python bindings must be a Unicorn **2.x** installation
  (`pip install unicorn==2.1.4`), or point `LSGEMU_UNICORN_BINDINGS` at a local
  bindings checkout. With pip-installed bindings the shipped library is still
  used through `LIBUNICORN_PATH`; set `unicorn.use_system_unicorn: 1` in
  `configs/lsgemu_config.yaml` to use a system Unicorn instead.
* **Ghidra headless — required by the default pipeline.** Static basic-block
  and MMIO-access extraction uses `analyzeHeadless`. Point the configuration at
  a local Ghidra installation:
  ```yaml
  ghidra:
    enabled: true
    install_dir: /path/to/ghidra
    analyze_headless: /path/to/ghidra/support/analyzeHeadless
  ```
  or export `GHIDRA_INSTALL_DIR` / `GHIDRA_ANALYZE_HEADLESS`. A Capstone-based
  analyzer exists for standalone use (`python3 -m lsgemu.analysis.firmware_analyzer
  --capstone`), but the end-to-end runner expects Ghidra.
* **Optional** — `angr`/`claripy` (reachability baselines), `z3-solver`
  (constraint-guided candidate generation), `pandas`/`matplotlib` (result
  analysis). `openai`/`httpx` are only needed for the optional LLM-guided
  mode: without an API key the system logs `LLM配置缺少api_key，禁用真实LLM调用`
  and runs the pipeline LLM-free.

External binaries (`readelf`, `nm`, `objdump`, `strings`, `java`) are used by
some analysis helpers and are documented inline in
`configs/lsgemu_config.example.yaml`.

### Running the tests

```bash
# Unicorn 2.x bindings and library, then:
PYTHONPATH=$PWD python3 -m pytest -q tests/unit         # 773 passed, 28 skipped
PYTHONPATH=$PWD python3 -m pytest -q tests/analysis     # 29 tests
```

The suite requires Unicorn **2.x** Python bindings; with the pip-installed
1.x bindings collection fails with `cannot import name 'UC_ARM_REG_XPSR'`.
Tests that need an external Unicorn bindings checkout or a full firmware run
are skipped when that environment is absent.

### Verified quick start

The package was verified after an anonymous-path rewrite on a clean copy:
`P2IM/CNC` with a one-minute budget executes the full stage sequence
(`baseline` → `contextual_isr` → `interleaved` → `targeted_frontier_cycles` →
`vector_only_cleanup` → `run_complete`) and writes
`CNC_interleaved_report.json`, `CNC_branch_catalog.json`,
`CNC_lsgemu_constraints.json` and `CNC_llm_history.json` under
`project.run_output_dir`.

## 3. Run one firmware

```bash
# wrapper form
scripts/run/lsgemu benchmarks/elfmultifuzz/P2IM/Gateway/Gateway.elf --time 60

# module form
python3 -m lsgemu.lsgemu --config configs/lsgemu_config.yaml \
    benchmarks/elfmultifuzz/P2IM/Gateway/Gateway.elf --time 60
```

Reports are written under `project.run_output_dir` from the configuration.
A raw `.bin` image can be converted on the fly through the BintoElf adapter
(`--firmware-format bin --bin-arch arm --bin-bits 32 --bin-base 0x08000000 ...`).

## 4. Reproducing the experiments

### 4.1 Coverage / ablation harness

```bash
# full evaluation over the bundled corpus
python3 evaluation/run_rq_experiments.py --rq all --jobs 4

# equal-budget ablation sweep (RQ2): contribution-level arms
python3 evaluation/run_rq_experiments.py --rq rq2 \
    --rq2-time 240 --ablation full --ablation no_constraint_guided_candidates
```

Ablation arms are declared in `evaluation/run_rq_experiments.py`
(`ABLATIONS`), each mapping to a command-line switch of the runner:

| Arm | Switch | What is removed |
|---|---|---|
| `full` | — | nothing (reference configuration) |
| `no_constraint_guided_candidates` | `--disable-constraint-guided-candidates` | symbolic constraint solving for candidate *values*. The located input occurrence, its recovered range, snapshot/context/prefix repair and the replay validation are all kept; candidates come from a fixed generic policy (boundary values, seed mutation, seeded random sampling) that never reads the target predicate |
| `no_semantic_obligation_v2` | `--disable-semantic-obligation-stages` | semantic obligation stages |
| `no_scoped_replay_v2` | `--disable-scoped-replay-stages` | scoped replay stages |
| `no_context_event_replay` | `--disable-context-event-replay-stages` | context-aware event replay |
| `full_llm_off` | (env) `LSGEMU_MAX_LLM_BRANCH_INFERENCE_CALLS=0` | LLM-guided branch inference, kept as a paired control for the full configuration |

### 4.2 Crash-discovery campaign (fuzzengine)

```bash
python3 -m fuzzengine.security_campaign \
    --simulation-report <firmware>_interleaved_report.json \
    --firmware <firmware>.elf \
    --work-dir /tmp/lsgemu_security/<firmware> \
    --seed <seed_dir> --iterations 1000
```

`fuzzengine` reports input-bound firmware candidates separately from model
artifacts and native emulator failures; it is an evidence tool, not a
vulnerability claim.

### 4.3 Static reachability baselines

`benchmarks/reachability/` stores, for every firmware image:

* `<firmware>_angr_cfgfast.txt` / `.json` — basic blocks enumerated by
  `angr`'s `CFGFast` (`normalize=True`), the static denominator used in the
  coverage tables. Each JSON records the method, the `angr` version, the
  elapsed time and the SHA-256 of the exact ELF that was analyzed.
* `angr_reachable/<firmware>_valid_bb.txt` / `_reach.json` — blocks found by
  `CFGEmulated` from the reset vector. These are a **lower bound** (symbolic
  exploration stops at the first unmodelled peripheral interaction) and are
  *not* usable as a denominator.
* `angr_reachable/INDEX_cfgfast_20261009.{csv,md}` — the table that maps every
  baseline file to its firmware, its block count and its SHA-256.
* `<firmware>_valid_bb.txt` — the per-instruction valid-BB list used by the
  campaign metadata.

## 5. Benchmarks

* **`benchmarks/elfmultifuzz/` — 23 targets**: P2IM (CNC, Console, Drone,
  Gateway, Heat_Press, PLC, Reflow_Oven, Robot, Soldering_Iron,
  Steering_Control), Pretender (RF_Door_Lock, Thermostat), uEmu (3Dprinter,
  GPSTracker, LiteOS_IoT, uTasker_MODBUS, uTasker_USB, Zephyr_SocketCAN),
  WYCINWYC (XML_Parser), HALucinator (6LoWPAN_Receiver), MultiFuzz/RIOT
  (ccn-lite-relay, filesystem, gnrc_networking).
* **`benchmarks/drone_firmware/` — 24 flight-controller images**: ArduPilot
  (CubeOrange, KakuteF7, MatekH743, Pixhawk1, Pixhawk4, Pixracer), PX4
  (fmu-v2, v3, v4pro, v5, v5x, v6c, v6x), Betaflight (F405, F411, H743,
  RP2350), INAV (ANYFCF7, BETAFPVF435, MATEKF405, MATEKF722, MATEKH743,
  OMNIBUSF4, PIXRACER).
* Third-party firmware images originate from the corresponding public
  benchmark suites and flight-stack projects; they are redistributed here only
  to make the evaluation reproducible.

## 6. Notes

* Paths: after `bash configure_paths.sh`, all configuration files reference
  your checkout. `MANIFEST.tsv` lists every file shipped with the artifact.
* Anonymity: absolute paths, account names and host names from the authors'
  environment were replaced or removed; there is no git history and no
  identifying metadata in the package.
* No documentation, paper sources, run outputs or cached artifacts are
  included by design — only source, experiment scripts and benchmarks.
