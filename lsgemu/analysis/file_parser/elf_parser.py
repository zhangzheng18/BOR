#!/usr/bin/env python3
"""
ELF文件解析器

支持解析ELF文件头，识别架构、大小端、位宽等信息
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .architecture import (
    Architecture, Endianness, ArchInfo,
    identify_architecture_from_elf_machine,
    get_machine_name
)

logger = logging.getLogger(__name__)


class ELFParser:
    """ELF文件解析器"""

    # ELF魔数
    ELF_MAGIC = b'\x7fELF'

    # ELF头偏移
    EI_CLASS = 4      # 文件类别 (32/64位)
    EI_DATA = 5       # 数据编码 (大小端)
    EI_VERSION = 6    # 文件版本
    EI_OSABI = 7      # OS/ABI标识

    # EI_CLASS值
    ELFCLASS32 = 1    # 32位
    ELFCLASS64 = 2    # 64位

    # EI_DATA值
    ELFDATA2LSB = 1   # 小端
    ELFDATA2MSB = 2   # 大端

    def __init__(self, file_path: str):
        """
        初始化ELF解析器

        Args:
            file_path: ELF文件路径
        """
        self.file_path = Path(file_path)
        self.elf_header: Optional[Dict[str, Any]] = None
        self.arch_info: Optional[ArchInfo] = None
        self.segments: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []
        self._sections_parsed = False

        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    def is_elf(self) -> bool:
        """检查是否是ELF文件"""
        try:
            with open(self.file_path, 'rb') as f:
                magic = f.read(4)
                return magic == self.ELF_MAGIC
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return False

    def parse(self) -> ArchInfo:
        """
        解析ELF文件

        Returns:
            ArchInfo对象，包含架构信息
        """
        if not self.is_elf():
            raise ValueError(f"不是有效的ELF文件: {self.file_path}")

        with open(self.file_path, 'rb') as f:
            # 读取ELF标识（前16字节）
            e_ident = f.read(16)
            if len(e_ident) != 16:
                raise ValueError(f"ELF 标识不完整: {self.file_path}")

            # 解析基本信息
            ei_class = e_ident[self.EI_CLASS]
            ei_data = e_ident[self.EI_DATA]
            ei_version = e_ident[self.EI_VERSION]
            ei_osabi = e_ident[self.EI_OSABI]

            # 确定位宽
            if ei_class == self.ELFCLASS32:
                bits = 32
                is_64bit = False
            elif ei_class == self.ELFCLASS64:
                bits = 64
                is_64bit = True
            else:
                raise ValueError(f"未知的ELF类别: {ei_class}")

            # 确定字节序
            if ei_data == self.ELFDATA2LSB:
                endianness = Endianness.LITTLE
                endian_char = '<'
            elif ei_data == self.ELFDATA2MSB:
                endianness = Endianness.BIG
                endian_char = '>'
            else:
                raise ValueError(f"未知的字节序: {ei_data}")

            # 根据位宽和字节序读取ELF头的其余部分
            if is_64bit:
                # 64位ELF头格式
                fmt = f'{endian_char}HHIQQQIHHHHHH'
                header_size = struct.calcsize(fmt)
                header_data = f.read(header_size)
                if len(header_data) != header_size:
                    raise ValueError(f"ELF64 文件头不完整: {self.file_path}")

                (e_type, e_machine, e_version,
                 e_entry, e_phoff, e_shoff,
                 e_flags, e_ehsize, e_phentsize, e_phnum,
                 e_shentsize, e_shnum, e_shstrndx) = struct.unpack(fmt, header_data)
            else:
                # 32位ELF头格式
                fmt = f'{endian_char}HHIIIIIHHHHHH'
                header_size = struct.calcsize(fmt)
                header_data = f.read(header_size)
                if len(header_data) != header_size:
                    raise ValueError(f"ELF32 文件头不完整: {self.file_path}")

                (e_type, e_machine, e_version,
                 e_entry, e_phoff, e_shoff,
                 e_flags, e_ehsize, e_phentsize, e_phnum,
                 e_shentsize, e_shnum, e_shstrndx) = struct.unpack(fmt, header_data)

            # 保存ELF头信息
            self.elf_header = {
                'ei_class': ei_class,
                'ei_data': ei_data,
                'ei_version': ei_version,
                'ei_osabi': ei_osabi,
                'e_type': e_type,
                'e_machine': e_machine,
                'e_version': e_version,
                'e_entry': e_entry,
                'e_phoff': e_phoff,
                'e_shoff': e_shoff,
                'e_flags': e_flags,
                'e_ehsize': e_ehsize,
                'e_phentsize': e_phentsize,
                'e_phnum': e_phnum,
                'e_shentsize': e_shentsize,
                'e_shnum': e_shnum,
                'e_shstrndx': e_shstrndx,
            }
            self.sections = []
            self._sections_parsed = False
            self.segments = self._parse_program_headers(
                f,
                endian_char=endian_char,
                is_64bit=is_64bit,
                e_phoff=e_phoff,
                e_phentsize=e_phentsize,
                e_phnum=e_phnum,
            )

            # 识别架构
            architecture = identify_architecture_from_elf_machine(e_machine, ei_class)
            machine_name = get_machine_name(e_machine)

            load_segments = [
                segment for segment in self.segments
                if segment.get('p_type') == 1
                and max(int(segment.get('p_filesz', 0) or 0), int(segment.get('p_memsz', 0) or 0)) > 0
            ]
            if load_segments:
                base_addr = min(int(segment['p_vaddr']) for segment in load_segments)
                code_end = max(
                    int(segment['p_vaddr']) + max(
                        int(segment.get('p_filesz', 0) or 0),
                        int(segment.get('p_memsz', 0) or 0),
                    )
                    for segment in load_segments
                )
                code_size = max(0, code_end - base_addr)
            else:
                base_addr = e_entry & 0xFFFF0000
                code_size = 0

            # 创建架构信息对象
            self.arch_info = ArchInfo(
                architecture=architecture,
                endianness=endianness,
                bits=bits,
                machine_type=machine_name,
                machine_code=e_machine,
                entry_point=e_entry,
                base_addr=base_addr,
                code_size=code_size,
            )

            logger.info(f"ELF解析成功: {self.file_path.name}")
            logger.info(f"  架构: {architecture.value}")
            logger.info(f"  字节序: {endianness.value}")
            logger.info(f"  位宽: {bits}位")
            logger.info(f"  机器类型: {machine_name}")
            logger.info(f"  入口点: 0x{e_entry:08x}")
            logger.info(f"  加载基址: 0x{base_addr:08x}")

            return self.arch_info

    def _parse_program_headers(
        self,
        f,
        *,
        endian_char: str,
        is_64bit: bool,
        e_phoff: int,
        e_phentsize: int,
        e_phnum: int,
    ) -> List[Dict[str, Any]]:
        """Parse ELF program headers enough to recover PT_LOAD layout."""
        if not e_phoff or not e_phnum:
            return []

        segments: List[Dict[str, Any]] = []
        if is_64bit:
            fmt = f'{endian_char}IIQQQQQQ'
            names = (
                'p_type', 'p_flags', 'p_offset', 'p_vaddr', 'p_paddr',
                'p_filesz', 'p_memsz', 'p_align',
            )
        else:
            fmt = f'{endian_char}IIIIIIII'
            names = (
                'p_type', 'p_offset', 'p_vaddr', 'p_paddr', 'p_filesz',
                'p_memsz', 'p_flags', 'p_align',
            )

        expected_size = struct.calcsize(fmt)
        for index in range(e_phnum):
            try:
                f.seek(e_phoff + index * e_phentsize)
                data = f.read(expected_size)
                if len(data) != expected_size:
                    break
                values = struct.unpack(fmt, data)
                segments.append(dict(zip(names, values)))
            except Exception as exc:
                logger.debug(f"解析程序头失败 #{index}: {exc}")
                break
        if self.elf_header is not None:
            self.elf_header['program_headers'] = segments
        return segments

    def _parse_section_headers(
        self,
        f,
        *,
        endian_char: str,
        is_64bit: bool,
        e_shoff: int,
        e_shentsize: int,
        e_shnum: int,
        e_shstrndx: int,
    ) -> List[Dict[str, Any]]:
        """Parse ELF section headers and resolve names from ``.shstrtab``."""

        if not e_shoff or not e_shentsize:
            return []
        if is_64bit:
            fmt = f'{endian_char}IIQQQQIIQQ'
        else:
            fmt = f'{endian_char}IIIIIIIIII'
        names = (
            'sh_name', 'sh_type', 'sh_flags', 'sh_addr', 'sh_offset',
            'sh_size', 'sh_link', 'sh_info', 'sh_addralign', 'sh_entsize',
        )
        expected_size = struct.calcsize(fmt)
        if int(e_shentsize) < expected_size:
            logger.warning(
                "ELF section entry too small: declared=%d required=%d",
                e_shentsize,
                expected_size,
            )
            return []

        f.seek(0, 2)
        file_size = f.tell()
        available_entries = max(0, (file_size - int(e_shoff)) // int(e_shentsize))
        if available_entries <= 0:
            return []

        def read_header(index: int) -> Optional[Dict[str, Any]]:
            if index < 0 or index >= available_entries:
                return None
            f.seek(int(e_shoff) + index * int(e_shentsize))
            raw = f.read(expected_size)
            if len(raw) != expected_size:
                return None
            section = dict(zip(names, struct.unpack(fmt, raw)))
            section['index'] = int(index)
            section['name'] = ''
            return section

        first = read_header(0)
        section_count = int(e_shnum)
        if section_count == 0 and first is not None:
            # ELF extended numbering stores the real count in section zero.
            section_count = int(first.get('sh_size', 0) or 0)
        section_count = min(max(0, section_count), available_entries)
        if section_count <= 0:
            return []

        sections: List[Dict[str, Any]] = []
        for index in range(section_count):
            section = first if index == 0 and first is not None else read_header(index)
            if section is None:
                break
            sections.append(section)

        string_table_index = int(e_shstrndx)
        if string_table_index == 0xFFFF and sections:
            string_table_index = int(sections[0].get('sh_link', 0) or 0)
        string_table = b''
        if 0 <= string_table_index < len(sections):
            table = sections[string_table_index]
            table_offset = int(table.get('sh_offset', 0) or 0)
            table_size = int(table.get('sh_size', 0) or 0)
            if table_offset >= 0 and table_size > 0 and table_offset + table_size <= file_size:
                f.seek(table_offset)
                string_table = f.read(table_size)

        if string_table:
            for section in sections:
                name_offset = int(section.get('sh_name', 0) or 0)
                if name_offset < 0 or name_offset >= len(string_table):
                    continue
                end = string_table.find(b'\0', name_offset)
                if end < 0:
                    end = len(string_table)
                section['name'] = string_table[name_offset:end].decode(
                    'utf-8', errors='replace'
                )
        return sections

    def get_entry_point(self) -> Optional[int]:
        """获取入口点地址"""
        if self.elf_header:
            return self.elf_header['e_entry']
        return None

    def get_sections(self) -> list:
        """Return parsed ELF section headers without affecting load behavior."""

        if self.elf_header is None:
            self.parse()
        if self._sections_parsed:
            return [dict(section) for section in self.sections]
        header = self.elf_header or {}
        endian_char = '<' if header.get('ei_data') == self.ELFDATA2LSB else '>'
        with open(self.file_path, 'rb') as f:
            self.sections = self._parse_section_headers(
                f,
                endian_char=endian_char,
                is_64bit=header.get('ei_class') == self.ELFCLASS64,
                e_shoff=int(header.get('e_shoff', 0) or 0),
                e_shentsize=int(header.get('e_shentsize', 0) or 0),
                e_shnum=int(header.get('e_shnum', 0) or 0),
                e_shstrndx=int(header.get('e_shstrndx', 0) or 0),
            )
        self._sections_parsed = True
        return [dict(section) for section in self.sections]

    def get_segments(self) -> list:
        """获取程序头段信息。"""
        if not self.segments and self.elf_header is None:
            self.parse()
        return list(self.segments)

    def print_header(self):
        """打印ELF头信息"""
        if not self.elf_header:
            print("ELF头未解析")
            return

        print("=" * 60)
        print("ELF头信息")
        print("=" * 60)

        h = self.elf_header

        print(f"类别:       {'64位' if h['ei_class'] == 2 else '32位'}")
        print(f"字节序:     {'小端' if h['ei_data'] == 1 else '大端'}")
        print(f"版本:       {h['ei_version']}")
        print(f"OS/ABI:     {h['ei_osabi']}")
        print(f"类型:       {h['e_type']}")
        print(f"机器:       0x{h['e_machine']:04x}")
        print(f"入口点:     0x{h['e_entry']:08x}")
        print(f"程序头偏移: 0x{h['e_phoff']:08x}")
        print(f"节头偏移:   0x{h['e_shoff']:08x}")
        print(f"标志:       0x{h['e_flags']:08x}")
        print(f"程序头数量: {h['e_phnum']}")
        print(f"节头数量:   {h['e_shnum']}")
        print("=" * 60)

    def get_section_by_name(self, section_name: str) -> Optional[Dict]:
        """获取指定名称的段信息"""
        if not self.is_elf():
            return None
        try:
            for section in self.get_sections():
                if section.get('name') == section_name:
                    return {
                        'offset': section['sh_offset'],
                        'vaddr': section['sh_addr'],
                        'size': section['sh_size']
                    }
            return None
        except Exception as e:
            logger.debug(f"获取段{section_name}失败: {e}")
            return None
