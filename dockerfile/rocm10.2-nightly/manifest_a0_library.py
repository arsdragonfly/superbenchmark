# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate and fingerprint a coherent A0 library generated from pinned sources."""

import hashlib
import json
from pathlib import Path
import re
import struct
import sys


root, output, revision = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not re.fullmatch('[0-9a-f]{40}', revision):
    raise ValueError('Expected a full library source revision')
masters = [path for name in ('TensileLibrary_lazy_gfx1250.dat', 'TensileLibrary_lazy_gfx1250.dat.zlib')
           if (path := root / name).is_file() and path.stat().st_size > 0]
if len(masters) != 1:
    raise RuntimeError('Expected exactly one nonempty A0 master library')
objects = sorted(path for path in root.iterdir() if path.suffix in ('.co', '.hsaco'))
if not objects:
    raise RuntimeError('No A0 code objects were generated')
for path in objects:
    data = path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 224:
        raise RuntimeError(f'Not an uncompressed AMDGPU ELF64 object: {path}')
    if struct.unpack_from('<I', data, 48)[0] & 0xFF != 0x49:
        raise RuntimeError(f'Wrong compiler target for the current runtime mode: {path}')
manifest = {
    'source_revision': revision, 'architecture_alias': 'gfx1250v0',
    'compiler_target': 'gfx1250', 'tuning_architecture': 'gfx1250-strict',
    'master': masters[0].name, 'code_objects': len(objects), 'helper_cache_disabled': True,
    'sha256': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir()) if path.is_file()
    },
}
output.write_text(json.dumps(manifest, indent=2) + '\n')
print(f"A0_LIBRARY_FILES={len(manifest['sha256'])} CODE_OBJECTS={len(objects)}")
