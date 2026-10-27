# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the matched nightly, vendor A0 assets and each independent compiler."""

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

from packaging.requirements import Requirement
from validate_translation_cache import check_cache


ROOT = Path('/opt/base-provenance')


def run(*command):
    """Keep failures explicit throughout compiler qualification."""
    subprocess.run(command, check=True)


def inventory_assets(site):
    """Require A0 master tables and coherent files in both runtime layouts."""
    files = {}
    for library in ('hipblaslt', 'hipsparselt'):
        relative = Path(f'lib/{library}/library/gfx1250v0')
        source = site / '_rocm_sdk_libraries' / relative
        master = source / 'TensileLibrary_lazy_gfx1250.dat.zlib'
        if not master.is_file() or master.stat().st_size == 0:
            raise RuntimeError(f'Missing vendor A0 master: {master}')
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = site / '_rocm_sdk_devel' / relative / path.name
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Inconsistent vendor A0 asset: {target}')
            files[str(relative / path.name)] = digest
    if not any(name.endswith(('.co', '.hsaco')) for name in files):
        raise RuntimeError('No A0 device code objects found')
    return files


def check_clang(sdk):
    """Exercise both instruction selectors independently of Triton."""
    for selector in (0, 1):
        output = ROOT / f'unsigned-offset-isel-{selector}.s'
        run(str(sdk / 'lib/llvm/bin/clang'), '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx1250',
            '-nogpulib', '-O3', '-S', '-x', 'ir', '-mllvm', f'-global-isel={selector}',
            '-mllvm', '-global-isel-abort=1', str(ROOT / 'unsigned-offset.ll'), '-o', str(output))
        text = output.read_text()
        address = re.search(r'v_lshl_add_u64 v\[(\d+):(\d+)\]', text)
        if not address:
            raise RuntimeError('Missing widened unsigned address')
        low, high = address.groups()
        if not re.search(rf'v_(?:dual_)?mov_b32(?:_e32)? v{high}, 0\b', text):
            raise RuntimeError('Unsigned index was not zero-extended')
        if not re.search(rf'global_load_b32 v\d+, v\[{low}:{high}\], off', text):
            raise RuntimeError('Load does not use the full address')


def check_serving(role):
    """Check source-pinned frameworks, their native files and installed launch profiles."""
    if version('amd-aiter') != '0.1.26.dev0+g2cb3d1758':
        raise RuntimeError('Unexpected AITER build')
    site = Path(sysconfig.get_path('purelib'))
    for relative in ('include/ck_tile/core.hpp', 'example/ck_tile/01_fmha/generate.py'):
        a = Path('/opt/aiter/3rdparty/composable_kernel') / relative
        b = site / 'aiter_meta/3rdparty/composable_kernel' / relative
        if not a.is_file() or a.read_bytes() != b.read_bytes():
            raise RuntimeError(f'Missing pinned CK JIT source: {relative}')
    for module in {'vllm': ('vllm._C',), 'sglang': ('sglang', 'sgl_kernel')}[role]:
        importlib.import_module(module)
    for package in ('triton', 'flydsl'):
        run('sha256sum', '--check', str(ROOT / f'fixed-{package}-installed.sha256'))
    run(sys.executable, str(ROOT / 'flydsl_unsigned_offset.py'))
    if role == 'vllm':
        check_vllm_patches()
    profiles = ('vllm-triton',) if role == 'vllm' else ('sglang-qwen397', 'sglang-compile')
    for profile in profiles:
        run('/usr/local/bin/sb-serve', '--dry-run', profile, '--help')


def check_vllm_patches():
    """Verify both AMD patches reached all installed wheel modules."""
    hashes = json.loads((ROOT / 'amd-vllm/installed.json').read_text())
    if len(hashes) != 12:
        raise RuntimeError('AMD vLLM patches missing')
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Patched module changed: {name}')


def check_pins(sdk):
    """Compare actual SDK and Python versions to the frozen input manifest."""
    manifest = json.loads((sdk / 'share/therock/therock_manifest.json').read_text())
    expected = json.loads((ROOT / 'expected-manifest.json').read_text())
    pins = {x['submodule_name']: x['pin_sha'] for x in manifest['submodules']}
    if any(manifest[k] != expected[k] for k in ('rocm_package_version', 'the_rock_commit')):
        raise RuntimeError('SDK version or build source mismatch')
    if any(pins[k] != v for k, v in expected['submodules'].items()):
        raise RuntimeError('SDK component source mismatch')
    for line in (ROOT / 'requirements-base.lock').read_text().splitlines():
        requirement = Requirement(line)
        if not requirement.specifier.contains(version(requirement.name), prereleases=True):
            raise RuntimeError(f'Package pin changed: {line}')
    for flag in ('TRITON_HIP_USE_ASYNC_COPY', 'TRITON_HIP_USE_EXPERT_SCHEDULING'):
        if os.environ.get(flag) != '0':
            raise RuntimeError(f'Required A0 mitigation is not set: {flag}')
    return expected


def main():
    """Save CPU-only validation evidence without pretending to qualify GPU inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-assets', action='store_true')
    parser.add_argument('--role', choices=('base', 'vllm', 'sglang'), default='base')
    args = parser.parse_args()
    sdk = Path(os.environ['ROCM_PATH'])
    expected = check_pins(sdk)
    assets = inventory_assets(Path(sysconfig.get_path('purelib')))
    asset_file = ROOT / 'vendor-a0-sha256.json'
    if args.record_assets:
        if asset_file.exists():
            raise RuntimeError('Refusing to overwrite the base asset inventory')
        asset_file.write_text(json.dumps(assets, indent=2) + '\n')
    elif assets != json.loads(asset_file.read_text()):
        raise RuntimeError('Overlay changed vendor A0 assets')
    check_clang(sdk)
    run('sha256sum', '--check', str(ROOT / 'fixed-triton-library.sha256'))
    run(sys.executable, str(ROOT / 'triton_unsigned_offset.py'), '--expect-fixed')
    run(str(ROOT / 'hiprtc_unsigned_offset'), str(ROOT / 'hiprtc-offset.hsaco'))
    text = subprocess.check_output([str(sdk / 'lib/llvm/bin/llvm-objdump'), '--disassemble',
                                    '--mcpu=gfx1250', str(ROOT / 'hiprtc-offset.hsaco')], text=True)
    if not re.search(r'global_load_b32.*v\[\d+:\d+\], off', text):
        raise RuntimeError('HIPRTC unsigned-offset check failed')
    cache = check_cache(sdk / 'share/rocjitsu/translations')
    import torch
    import torchvision
    if torch.arange(16).sum().item() != 120:
        raise RuntimeError('CPU tensor check failed')
    if torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]),
                           torch.tensor([0.9, 0.8]), 0.5).tolist() != [0]:
        raise RuntimeError('Native NMS check failed')
    if args.role != 'base':
        check_serving(args.role)
    run(sys.executable, '-m', 'pip', 'check')
    report = {'result': 'PASS', 'role': args.role, 'vendor_a0_files': len(assets),
              'cache': cache, 'gpu_validation': 'NOT_RUN', 'sources': expected}
    (ROOT / f'validation-{args.role}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
