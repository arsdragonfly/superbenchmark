# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Verify real AMDGPU Wave32 descriptor bits rather than only ELF target labels."""

import struct


def section_bytes(data, section):
    """Reject truncated ELF sections before examining their contents."""
    start, size = section[4:6]
    if start + size > len(data):
        raise ValueError('ELF section extends beyond the object')
    return data[start:start + size]


def kernel_flags(data):
    """Return unique gfx1250 kernel descriptor flags from an ELF64 object."""
    if data[:6] != b'\x7fELF\x02\x01' or len(data) < 64:
        raise ValueError('Expected a little-endian ELF64 object')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    if header[2] != 224 or header[7] & 0xff not in (0x49, 0xeb):
        raise ValueError('Expected a gfx1250 AMDGPU object')
    offset, stride, count = header[6], header[11], header[12]
    if stride != 64 or count == 0 or offset + stride * count > len(data):
        raise ValueError('Invalid ELF section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + index * stride) for index in range(count)]
    flags = {}
    for section in sections:
        if section[1] in (2, 11):
            flags.update(symbol_flags(data, section, sections))
    return flags


def symbol_flags(data, table, sections):
    """Read descriptor symbols from a static or dynamic symbol table."""
    symbols = section_bytes(data, table)
    if table[9] != 24 or len(symbols) % 24 or table[6] >= len(sections):
        raise ValueError('Invalid ELF symbol table')
    strings = section_bytes(data, sections[table[6]])
    flags = {}
    for offset in range(0, len(symbols), 24):
        name, _, _, index, address, size = struct.unpack_from('<IBBHQQ', symbols, offset)
        if name >= len(strings):
            raise ValueError('Invalid ELF symbol name')
        symbol = strings[name:strings.index(b'\0', name)]
        if not symbol.endswith(b'.kd'):
            continue
        if index >= len(sections) or size not in (0, 64):
            raise ValueError('Invalid kernel descriptor symbol')
        section = sections[index]
        contents = section_bytes(data, section)
        location = address - section[3]
        if location < 0 or location + 64 > len(contents):
            raise ValueError('Kernel descriptor extends beyond its section')
        flags[index, address] = struct.unpack_from('<H', contents, location + 56)[0]
    return flags


def check_cache(root):
    """Fail on empty caches, missing descriptors, or cleared Wave32 bits."""
    objects = sorted(root.rglob('*.obj'))
    if not objects:
        raise ValueError(f'No translated objects under {root}')
    descriptors = 0
    for path in objects:
        flags = kernel_flags(path.read_bytes())
        if any(not value & 0x400 for value in flags.values()):
            raise ValueError(f'Translator cleared a Wave32 descriptor: {path}')
        descriptors += len(flags)
    if not descriptors:
        raise ValueError(f'No kernel descriptors under {root}')
    return {'objects': len(objects), 'descriptors': descriptors, 'result': 'PASS'}
