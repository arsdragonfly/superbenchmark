# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Check pinned SDK provenance, A0 assets and unsigned-address code generation."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sysconfig


root = Path('/opt/base-provenance')
sdk = Path(os.environ['ROCM_PATH'])
site = Path(sysconfig.get_path('purelib'))
expected = json.loads((root / 'expected-manifest.json').read_text())
actual = json.loads((sdk / 'share/therock/therock_manifest.json').read_text())
if actual != expected:
    raise RuntimeError('SDK source provenance differs from the selected nightly')
for package in ('rocm', 'rocm-sdk-core', 'rocm-sdk-devel', 'rocm-sdk-libraries'):
    if version(package) != '10.2.0a20261018':
        raise RuntimeError(f'Unexpected package version: {package}')
if version('torch') != '2.13.0+rocm10.2.0a20261018':
    raise RuntimeError('Unexpected PyTorch version')
manifest = json.loads((root / 'a0-library-manifest.json').read_text())
if manifest['source_revision'] != '8a29d2e82b93f5065c7fe5630248d3689f0c3f06':
    raise RuntimeError('Wrong A0 library source revision')
for view in ('_rocm_sdk_libraries', '_rocm_sdk_devel'):
    for name, expected_hash in manifest['sha256'].items():
        path = site / view / 'lib/hipblaslt/library/gfx1250v0' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Wrong A0 asset: {path}')
for selector in (0, 1):
    result = subprocess.run(
        [str(sdk / 'lib/llvm/bin/clang'), '-target', 'amdgcn-amd-amdhsa',
         '-mcpu=gfx1250', '-nogpulib', '-O3', '-S', '-x', 'ir',
         '-mllvm', f'-global-isel={selector}', '-mllvm', '-global-isel-abort=1',
         str(root / 'unsigned-offset.ll'), '-o', '-'],
        check=True, capture_output=True, text=True)
    assembly = result.stdout
    (root / f'unsigned-offset-isel-{selector}.s').write_text(assembly)
    address = re.search(r'v_lshl_add_u64 v\[(\d+):(\d+)\]', assembly)
    if not address:
        raise RuntimeError('Missing full-width unsigned address calculation')
    low, high = address.groups()
    if not re.search(rf'v_(?:dual_)?mov_b32(?:_e32)? v{high}, 0\b', assembly):
        raise RuntimeError('Unsigned index was not zero-extended')
    if not re.search(rf'global_load_b32 v\d+, v\[{low}:{high}\], off', assembly):
        raise RuntimeError('Global memory operation did not use the full address')
subprocess.run(['sha256sum', '--check', str(root / 'fixed-triton-library.sha256')], check=True)
report = {
    'result': 'CPU_PROVENANCE_AND_COMPILER_CHECKS_PASS',
    'gpu_qualification': 'Required separately',
    'a0_files_per_view': len(manifest['sha256']),
    'sources': {item['submodule_name']: item['pin_sha'] for item in actual['submodules']},
    'therock': actual['the_rock_commit'],
}
(root / 'base-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
