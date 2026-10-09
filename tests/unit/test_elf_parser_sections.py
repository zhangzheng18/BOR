#!/usr/bin/env python3
"""Regression tests for the lazy ELF section-header API."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from lsgemu.analysis.file_parser.elf_parser import ELFParser


def _minimal_arm_elf32() -> bytes:
    section_names = b"\0.text\0.shstrtab\0"
    section_table_offset = 0x80
    section_entry_size = 40
    image = bytearray(section_table_offset + 3 * section_entry_size)
    image[:16] = (
        b"\x7fELF"
        + bytes((ELFParser.ELFCLASS32, ELFParser.ELFDATA2LSB, 1, 0))
        + bytes(8)
    )
    image[16:52] = struct.pack(
        "<HHIIIIIHHHHHH",
        2,
        40,
        1,
        0x08000001,
        0,
        section_table_offset,
        0,
        52,
        32,
        0,
        section_entry_size,
        3,
        2,
    )
    image[0x40:0x44] = b"\x00\xbf\x00\xbf"
    image[0x50:0x50 + len(section_names)] = section_names
    image[section_table_offset + section_entry_size:section_table_offset + 2 * section_entry_size] = struct.pack(
        "<IIIIIIIIII",
        1,
        1,
        0x6,
        0x08000000,
        0x40,
        4,
        0,
        0,
        2,
        0,
    )
    image[section_table_offset + 2 * section_entry_size:section_table_offset + 3 * section_entry_size] = struct.pack(
        "<IIIIIIIIII",
        7,
        3,
        0,
        0,
        0x50,
        len(section_names),
        0,
        0,
        1,
        0,
    )
    return bytes(image)


class ELFParserSectionTests(unittest.TestCase):
    def test_sections_are_named_and_get_section_by_name_uses_same_parser(self):
        with tempfile.TemporaryDirectory(prefix="lsgemu_elf_sections_") as tmpdir:
            elf_path = Path(tmpdir) / "minimal.elf"
            elf_path.write_bytes(_minimal_arm_elf32())
            parser = ELFParser(str(elf_path))

            parser.parse()
            sections = parser.get_sections()

            self.assertEqual(["", ".text", ".shstrtab"], [item["name"] for item in sections])
            self.assertEqual(0x08000000, sections[1]["sh_addr"])
            self.assertEqual(
                {"offset": 0x40, "vaddr": 0x08000000, "size": 4},
                parser.get_section_by_name(".text"),
            )
            self.assertIsNone(parser.get_section_by_name(".missing"))


if __name__ == "__main__":
    unittest.main()
