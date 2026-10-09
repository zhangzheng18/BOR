# angr CFGFast 静态枚举 BB 索引（2026-10-09）

目录：`/opt/artifact/benchmarks/reachability/angr_reachable`；每固件 `<固件>_angr_cfgfast.txt`（十六进制 BB 地址）+ `.json`（元数据）

| 固件 | CFGFast BB | 表里 angr 分母 | 差 | 差% | 方法 | angr | ELF sha256(前12) |
|---|---|---|---|---|---|---|---|
| px4_fmu-v5x_STM32F765 | 139,937 | 136128 | +3,809 | +2.8% | cfg_fast | 9.2.14 | 51e79a9eac69 |
| px4_fmu-v3_STM32F427 | 128,654 | 132635 | -3,981 | -3.0% | cfg_fast | 9.2.14 | 5a6def995f4d |
| px4_fmu-v4pro_STM32F469 | 127,275 | 131973 | -4,698 | -3.6% | cfg_fast | 9.2.14 | 6d231eff2f01 |
| px4_fmu-v6x_STM32H753 | 126,688 | 131909 | -5,221 | -4.0% | cfg_fast | 9.2.14 | 2f09702ca2e8 |
| px4_fmu-v5_STM32F765 | 125,241 | 131320 | -6,079 | -4.6% | cfg_fast | 9.2.14 | 0f00f334622f |
| px4_fmu-v6c_STM32H743 | 120,246 | 124756 | -4,510 | -3.6% | cfg_fast | 9.2.14 | 1868b6571f46 |
| ardupilot_CubeOrange_STM32H757 | 117,077 | 119347 | -2,270 | -1.9% | cfg_fast | 9.2.14 | 9ef06cafaceb |
| ardupilot_Pixhawk1_STM32F427 | 116,014 | 118513 | -2,499 | -2.1% | cfg_fast | 9.2.14 | 35874d5d4b77 |
| ardupilot_MatekH743_STM32H743 | 114,002 | 116748 | -2,746 | -2.4% | cfg_fast | 9.2.14 | 6ec4508fe08c |
| ardupilot_Pixhawk4_STM32F765 | 113,224 | 116791 | -3,567 | -3.1% | cfg_fast | 9.2.14 | 94f8ccb198e0 |
| ardupilot_Pixracer_STM32F405 | 112,262 | 116436 | -4,174 | -3.6% | cfg_fast | 9.2.14 | 5deeab535090 |
| px4_fmu-v2_STM32F427 | 72,137 | 74539 | -2,402 | -3.2% | cfg_fast | 9.2.14 | cb1b0c7ea1fe |
| ardupilot_KakuteF7_STM32F745 | 64,642 | 66920 | -2,278 | -3.4% | cfg_fast | 9.2.14 | 459c0ffb8343 |
| betaflight_STM32H743_STM32H743 | 43,930 | 45688 | -1,758 | -3.8% | cfg_fast | 9.2.14 | accb5bdf9311 |
| inav_MATEKH743_STM32H743 | 42,178 | 43738 | -1,560 | -3.6% | cfg_fast | 9.2.14 | e6659bc55ec8 |
| betaflight_STM32F405_STM32F405 | 39,333 | 41264 | -1,931 | -4.7% | cfg_fast | 9.2.14 | 085ff97d37bd |
| inav_ANYFCF7_STM32F745 | 38,809 | 40551 | -1,742 | -4.3% | cfg_fast | 9.2.14 | 95208186809e |
| inav_MATEKF405_STM32F405 | 38,785 | 40287 | -1,502 | -3.7% | cfg_fast | 9.2.14 | 18ddc7d63788 |
| inav_OMNIBUSF4_STM32F405 | 37,824 | 39333 | -1,509 | -3.8% | cfg_fast | 9.2.14 | a66ca4adda1b |
| inav_PIXRACER_STM32F405 | 37,022 | 38940 | -1,918 | -4.9% | cfg_fast | 9.2.14 | f1f1bbd51be0 |
| inav_BETAFPVF435_AT32F435 | 36,952 | 38373 | -1,421 | -3.7% | cfg_fast | 9.2.14 | eccdd00f1585 |
| betaflight_RP2350_RP2350B | 34,039 | — |  |  | cfg_fast | 9.2.14 | 6c3c9cb67551 |
| inav_MATEKF722_STM32F722 | 32,611 | 33506 | -895 | -2.7% | cfg_fast | 9.2.14 | 9e4f3676d939 |
| betaflight_STM32F411_STM32F411 | 29,443 | 30717 | -1,274 | -4.1% | cfg_fast | 9.2.14 | e7c1401f244a |

## 口径说明

- **本目录的 `*_angr_cfgfast.txt`** = angr `CFGFast(normalize=True)` 的**全部基本块**（静态枚举，非可达性），
  即用户历史表格中 “angr 分母” 一列的可复现出处；生成命令见 `LSGEmu/scripts/gen_angr_cfgfast_all_20261009.sh`。
- **`px4_fmu-v5x_STM32F765`** 因 angr 9.2.14 的 `CFGFast._collect_data_references` 崩溃（`irsb=None`），
  改用 `data_references=False` 绕过后得到 139,937（唯一高于历史表的样本，+2.8%）。
- 与历史表的系统性偏差：多数固件低 **2%–5%**，归因于 angr 版本、间接跳转/跳转表解析设置与 Thumb 对齐差异。
- 与 **`*_valid_bb.txt`（本目录同名的另一批文件）** 区分：那是 `CFGEmulated` 从入口出发的**符号化可达集**（Pixhawk1=639），
  属**下界**，不可当分母。
