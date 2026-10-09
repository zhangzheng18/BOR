# LSGEmu ElfMultiFuzz Coverage Report

- Original all-2h output root: `/tmp/lsgemu_all_2h_after_frontier_rr_20260525_002155`
- Low-coverage rerun output root: `/tmp/lsgemu_lowcov_rerun_2h_jobs4_20260525_235301`
- 3Dprinter post-fix 2h output root: `/tmp/lsgemu_3dprinter_full_fix_20260526_134802`
- Report refreshed: `2026-05-26`
- Campaigns completed: original all-2h `True`, low-coverage rerun `True`, 3Dprinter post-fix 2h `True`
- Firmware cases: `23`
- Current best weighted valid BB coverage: `67990/102900 = 66.07%`
- Current best mean per-firmware coverage: `71.92%`
- Current best median per-firmware coverage: `72.17%`
- Current best cases >=60%: `17/23`
- Current best cases <60%: `6`

## Strict Candidate Validation Addendum

This coverage report measures general firmware execution coverage. It is not
used by itself as CVE-grade proof. CVE-grade strict validation is counted only
when a candidate-specific emulation run reaches the relevant dangerous behavior
or sensitive-output sink and records concrete sink data.

Latest strict triage validation command:

```bash
source /opt/artifact/.virtualenvs/fuzzware/bin/activate
python3 /opt/artifact/lsgemu/validate_mcu_triage_suite.py \
  --output-json /tmp/mcu_security_validation_data.json \
  --output-md /tmp/mcu_security_validation_report.md
```

Latest strict validation result:

| Item | Status | Strict Evidence |
|---|---|---|
| C-02 | `emulated_sensitive_output_triggered` | `Security Key: %s` output helper `0x34308` receives modeled PSK `CVE-Test-Key-12345` from the CPCL configuration-page builder `0x34404`. |
| U-01 | `emulated_sink_triggered` | Route helper commands reach `popen` with unquoted shell metacharacters in 2/4 emulation cases. |

Current strictly emulation-validated items: `2`

Additional anti-false-positive revalidation:

```bash
source /opt/artifact/.virtualenvs/fuzzware/bin/activate
python3 /opt/artifact/lsgemu/revalidate_strict_cves.py \
  --output-json /opt/artifact/reports/strict_cve_revalidation.json \
  --output-md /opt/artifact/reports/STRICT_CVE_REVALIDATION.md
```

Revalidation result: `2` strict items (`U-01`, `C-02`). `U-01` passed
safe-input negative checks and malicious-input positive checks. `C-02` passed
dynamic-key A/B checks and a negative gate-disabled check.

Remaining device-entry/protocol-unverified items: `6` (`C-01`, `C-03`,
`C-04`, `U-02`, `U-03`, `U-04`).

Latest validation artifacts:

- `/tmp/mcu_security_validation_data.json`
- `/tmp/mcu_security_validation_report.md`
- `reports/CVE_U01_Honeywell_LNX3_route_command_injection/`
- `reports/CVE_C02_Honeywell_LNX3_wifi_key_plaintext_disclosure/`

## Coverage Distribution

| Range | Cases |
|---|---:|
| <10% | 0 |
| 10-30% | 0 |
| 30-60% | 6 |
| 60-80% | 8 |
| >=80% | 9 |

## Family Summary

| Family | Cases | Valid Covered | Valid Total | Weighted Rate | Mean Rate |
|---|---:|---:|---:|---:|---:|
| HALucinator | 1 | 4563 | 6977 | 65.40% | 65.40% |
| MultiFuzz | 3 | 9399 | 21552 | 43.61% | 51.42% |
| P2IM | 10 | 23900 | 29126 | 82.06% | 80.28% |
| Pretender | 2 | 6137 | 7993 | 76.78% | 76.11% |
| WYCINWYC | 1 | 5914 | 9376 | 63.08% | 63.08% |
| uEmu | 6 | 18077 | 27876 | 64.85% | 69.40% |

## Low-Coverage Analysis

Detailed analysis and fix notes are in
`/opt/artifact/reports/BASELINE_LOW_COVERAGE_ANALYSIS.md`.

The remaining `<60%` cases after best-result merge are:

| Firmware | Best Source | Valid Covered | Valid Total | Rate | Reachable Rate | Diagnosis |
|---|---|---:|---:|---:|---:|---|
| MultiFuzz/riot-ccn-lite-relay | lowcov rerun | 4398 | 12675 | 34.70% | 70.72% | large static-unreachable denominator plus RTOS/message/callback state gap |
| P2IM/Console | lowcov rerun | 966 | 2251 | 42.91% | 81.56% | reachable portion mostly covered; denominator includes code outside current entry/vector model |
| P2IM/Heat_Press | original all-2h | 871 | 1837 | 47.41% | 89.52% | reachable portion mostly covered; missing IRQ/timer/peripheral states |
| uEmu/3Dprinter | post-fix 2h | 4087 | 8045 | 50.80% | 53.41% | real entry/IRQ/persistent-state, dynamic dispatch, and call-like frontier gap; persisted-constraint validation bug fixed |
| MultiFuzz/riot-gnrc_networking | lowcov rerun | 3364 | 6448 | 52.17% | 59.42% | real RIOT scheduler/message state and frontier conversion gap |
| uEmu/Zepyhr_SocketCan | original all-2h | 3207 | 5943 | 53.96% | 97.11% | reachable portion saturated; denominator/vector/task-entry modeling dominates |

## Per-Firmware Results

| Firmware | Status | Valid Covered | Valid Total | Rate | Reachable Rate | Elapsed | Exit | Resource |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MultiFuzz/riot-ccn-lite-relay | completed_strict_checkpoint_union | 4398 | 12675 | 34.70% | 70.72% | 7182s | 0 | False |
| P2IM/Console | completed_partial_strict_resource_union | 966 | 2251 | 42.91% | 81.56% | 848s | 1 | False |
| P2IM/Heat_Press | completed_partial_strict_resource_union | 871 | 1837 | 47.41% | 89.52% | 1513s | 1 | False |
| uEmu/3Dprinter | completed | 4087 | 8045 | 50.80% | 53.41% | 7181s | 0 | False |
| MultiFuzz/riot-gnrc_networking | completed_strict_checkpoint_union | 3364 | 6448 | 52.17% | 59.42% | 3600s | 0 | False |
| uEmu/Zepyhr_SocketCan | completed_partial_strict_resource_union | 3207 | 5943 | 53.96% | 97.11% | 4274s | 1 | True |
| WYCINWYC/XML_Parser | completed_partial_strict_resource_union | 5914 | 9376 | 63.08% | 71.51% | 5700s | 1 | False |
| HALucinator/6LoWPAN_Receiver | completed | 4563 | 6977 | 65.40% | 96.88% | 7201s | 0 | None |
| MultiFuzz/riot-filesystem | completed_partial_strict_resource_union | 1637 | 2429 | 67.39% | 99.03% | 2649s | 1 | False |
| uEmu/GPSTracker | completed_partial_strict_resource_union | 2952 | 4194 | 70.39% | 78.24% | 5564s | 1 | False |
| P2IM/Reflow_Oven | completed_partial_strict_resource_union | 2081 | 2947 | 70.61% | 75.43% | 1712s | 1 | False |
| Pretender/RF_Door_Lock | completed_partial_strict_resource_union | 2396 | 3320 | 72.17% | 80.13% | 1073s | 1 | True |
| uEmu/LiteOS_IoT | completed_partial_strict_resource_union | 1888 | 2423 | 77.92% | 96.77% | 2747s | 1 | False |
| P2IM/Gateway | completed_partial_strict_resource_union | 3932 | 4921 | 79.90% | 97.59% | 2222s | 1 | False |
| Pretender/Thermostat | completed_partial_strict_resource_union | 3741 | 4673 | 80.06% | 87.45% | 5205s | 1 | False |
| uEmu/utasker_USB | completed_partial_strict_resource_union | 2800 | 3491 | 80.21% | 98.90% | 3628s | 1 | False |
| uEmu/utasker_MODBUS | completed_partial_strict_resource_union | 3143 | 3780 | 83.15% | 91.37% | 848s | 1 | False |
| P2IM/Soldering_Iron | completed_partial_strict_resource_union | 3247 | 3656 | 88.81% | 97.38% | 2542s | 1 | True |
| P2IM/Steering_Control | completed_partial_strict_resource_union | 1683 | 1835 | 91.72% | 100.00% | 2808s | 1 | False |
| P2IM/CNC | completed_partial_strict_resource_union | 3396 | 3614 | 93.97% | 99.21% | 1887s | 1 | False |
| P2IM/Robot | completed_partial_strict_resource_union | 2877 | 3034 | 94.83% | 89.74% | 900s | 1 | True |
| P2IM/PLC | completed_partial_strict_resource_union | 2214 | 2303 | 96.14% | 99.82% | 1980s | 1 | False |
| P2IM/Drone | completed_partial_strict_resource_union | 2633 | 2728 | 96.52% | 99.32% | 1733s | 1 | False |

## Remaining Rerun Set (<60%)

| Firmware | Current Rate | Main Symptom |
|---|---:|---|
| MultiFuzz/riot-ccn-lite-relay | 34.70% | static_unreachable_denominator,entry_or_irq_state_prefix,frontier_branch_not_converted,call_like_return_path |
| P2IM/Console | 42.91% | denominator/entry-vector model; reachable rate already 81.56% |
| P2IM/Heat_Press | 47.41% | low dynamic/frontier conversion |
| uEmu/3Dprinter | 50.80% | entry_or_irq_state_prefix,frontier_branch_not_converted,dynamic_dispatch_switch,call_like_return_path,missing_entry_derived_snapshots |
| MultiFuzz/riot-gnrc_networking | 52.17% | entry_or_irq_state_prefix,frontier_branch_not_converted,call_like_return_path |
| uEmu/Zepyhr_SocketCan | 53.96% | denominator/vector/task-entry model; reachable rate already 97.11% |

## Rerun Command Policy

- Rerun only `<60%` cases to avoid wasting CPU on already high-covered firmware.
- Use `jobs=4` on this 96-core / 314GiB machine. Higher is possible, but 4 is safer because each Unicorn replay is wallclock-sensitive.
- Enable crash retry and resource-failure retry; previous run disabled crash retry, so early SIGSEGV cases did not get compensation plans.
- Enable full-duration low-coverage retry with one compensation plan for cases that complete but remain below target.
