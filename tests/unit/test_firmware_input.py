#!/usr/bin/env python3
"""Contracts for the optional BintoElf firmware-input adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lsgemu.analysis.intelligent_emulator import IntelligentEmulator
from lsgemu.firmware_input import (
    BinConversionSpec,
    DEFAULT_BINTOELF_ROOT,
    FirmwareInputError,
    resolve_firmware_input,
)
from lsgemu.historical_runner import HistoricalRunner
from lsgemu.prepared_firmware import PreparedFirmware
from lsgemu.toolchain_fingerprint import sha256_file


class FirmwareInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = Path(DEFAULT_BINTOELF_ROOT) / "bin2elf" / "__init__.py"
        if not package.is_file():
            raise unittest.SkipTest(f"BintoElf test dependency is unavailable: {package}")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lsgemu_bintoelf_test_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "sample firmware.bin"
        self.payload = bytes.fromhex("00bf00bf704700bf")
        self.source.write_bytes(self.payload)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.temporary.cleanup()

    def thumb_spec(self, **overrides) -> BinConversionSpec:
        values = {
            "arch": "arm",
            "bits": 32,
            "endian": "little",
            "base_address": 0x08000000,
            "entry_offset": 0,
            "thumb": True,
        }
        values.update(overrides)
        return BinConversionSpec(**values)

    def resolve(self, spec=None, **kwargs):
        return resolve_firmware_input(
            self.source,
            bin_spec=spec,
            cache_root=self.cache,
            bintoelf_root=DEFAULT_BINTOELF_ROOT,
            **kwargs,
        )

    def test_explicit_bin_conversion_is_deterministic_and_preserves_source(self):
        first = self.resolve(self.thumb_spec())
        second = self.resolve(self.thumb_spec())

        self.assertTrue(first.converted)
        self.assertFalse(first.provenance["conversion"]["cache_hit"])
        self.assertTrue(second.provenance["conversion"]["cache_hit"])
        self.assertEqual(first.analysis_path, second.analysis_path)
        self.assertEqual("sample_firmware.elf", first.analysis_path.name)
        self.assertEqual(self.payload, self.source.read_bytes())
        self.assertEqual(b"\x7fELF", first.analysis_path.read_bytes()[:4])
        self.assertEqual(
            first.provenance["analysis"]["sha256"],
            sha256_file(first.analysis_path),
        )
        self.assertTrue(first.provenance["conversion"]["round_trip_validated"])

    def test_thumb_entry_and_execution_mode_are_explicit(self):
        resolved = self.resolve(self.thumb_spec())
        configuration = resolved.provenance["conversion"]["configuration"]
        self.assertEqual(0x08000001, configuration["entry"])
        self.assertEqual("thumb", configuration["execution_mode"])
        self.assertIs(resolved.execution_thumb_override, True)

        arm = self.resolve(
            self.thumb_spec(
                entry_offset=None,
                entry=0x08000000,
                thumb=False,
            )
        )
        self.assertEqual("arm", arm.provenance["execution_mode"])
        self.assertIs(arm.execution_thumb_override, False)
        self.assertNotEqual(resolved.analysis_path, arm.analysis_path)

    def test_corrupt_cache_entry_is_replaced(self):
        first = self.resolve(self.thumb_spec())
        expected_sha = first.provenance["analysis"]["sha256"]
        first.analysis_path.write_bytes(b"corrupt")

        repaired = self.resolve(self.thumb_spec())
        self.assertFalse(repaired.provenance["conversion"]["cache_hit"])
        self.assertEqual(expected_sha, sha256_file(repaired.analysis_path))

    def test_native_elf_and_legacy_raw_inputs_remain_unmodified(self):
        legacy = self.resolve()
        self.assertFalse(legacy.converted)
        self.assertEqual("legacy_raw_bin", legacy.provenance["source_kind"])

        generated = self.resolve(self.thumb_spec())
        native = resolve_firmware_input(generated.analysis_path)
        self.assertFalse(native.converted)
        self.assertEqual("elf", native.provenance["source_kind"])

    def test_non_arm32_targets_are_rejected_before_lsgemu(self):
        for arch, bits in (("mips", 32), ("aarch64", 64)):
            with self.subTest(arch=arch, bits=bits):
                with self.assertRaisesRegex(FirmwareInputError, "only ARM32"):
                    self.resolve(
                        BinConversionSpec(
                            arch=arch,
                            bits=bits,
                            endian="little",
                            base_address=0,
                            entry=0,
                        )
                    )

    def test_ambiguous_or_non_executable_layout_is_rejected(self):
        with self.assertRaisesRegex(FirmwareInputError, "conflicts"):
            self.resolve(
                self.thumb_spec(
                    entry_offset=None,
                    entry=0x08000001,
                    thumb=False,
                )
            )
        with self.assertRaisesRegex(FirmwareInputError, "executable"):
            self.resolve(self.thumb_spec(permissions="rw-"))

    def test_provenance_is_bound_to_source_and_analysis_hashes(self):
        resolved = self.resolve(self.thumb_spec())
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(
            firmware_path=resolved.analysis_path,
            firmware_sha256=sha256_file(resolved.analysis_path),
        )
        runner.firmware_input_provenance = dict(resolved.provenance)
        validated = runner._validated_firmware_input_provenance()
        self.assertEqual(
            resolved.provenance["source"]["sha256"],
            validated["source"]["sha256"],
        )

        self.source.write_bytes(self.payload + b"\x00")
        with self.assertRaisesRegex(ValueError, "source size"):
            runner._validated_firmware_input_provenance()

    def test_untransformed_provenance_cannot_claim_another_source(self):
        generated = self.resolve(self.thumb_spec())
        native = resolve_firmware_input(generated.analysis_path)
        runner = HistoricalRunner.__new__(HistoricalRunner)
        runner.prepared = SimpleNamespace(
            firmware_path=native.analysis_path,
            firmware_sha256=sha256_file(native.analysis_path),
        )
        provenance = dict(native.provenance)
        provenance["source"] = dict(provenance["source"])
        provenance["source"]["path"] = str(self.source)
        runner.firmware_input_provenance = provenance
        with self.assertRaisesRegex(ValueError, "identical source and analysis paths"):
            runner._validated_firmware_input_provenance()

    def test_execution_mode_override_precedes_elf_default(self):
        emulator = IntelligentEmulator.__new__(IntelligentEmulator)
        emulator.arch_info = SimpleNamespace(entry_point=0x08000001)
        emulator.is_elf_firmware = True
        emulator.execution_thumb_override = False
        self.assertFalse(emulator._infer_thumb_execution_mode())
        emulator.execution_thumb_override = True
        self.assertTrue(emulator._infer_thumb_execution_mode())

        prepared = PreparedFirmware.__new__(PreparedFirmware)
        prepared.result = SimpleNamespace(
            arch_info=SimpleNamespace(entry_point=0x08000000)
        )
        prepared.firmware_path = Path("wrapped.elf")
        prepared.execution_thumb_override = False
        self.assertFalse(prepared._infer_static_thumb_mode())

    def test_explicit_empty_valid_hint_set_blocks_name_based_lookup(self):
        result = SimpleNamespace(basic_blocks=[], instructions=[])
        with patch(
            "lsgemu.prepared_firmware._resolve_valid_bb_metadata",
            side_effect=AssertionError("name-based valid-BB lookup must stay disabled"),
        ):
            views = PreparedFirmware._build_static_views(
                result,
                Path("PLC.elf"),
                valid_hint_set=set(),
            )
        self.assertEqual(set(), views.static_bb_set)


if __name__ == "__main__":
    unittest.main()
