# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Check the serving stack, real compiler paths and stock translator on a CPU host."""

import argparse
import hashlib
import importlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from validate_translation_cache import check_cache


ROOT = Path('/opt/base-provenance')


def run(*command):
    """Propagate validation command failures instead of reporting a partial pass."""
    subprocess.run(command, check=True)


def check_packages(role):
    """Check all pinned installed packages, allowing role-specific dependency downgrades."""
    locks = [ROOT / 'requirements-base.lock', ROOT / 'constraints-base.txt']
    if role in ('vllm', 'sglang'):
        locks.append(ROOT / 'constraints-app.txt')
    expected = {}
    for path in locks:
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                requirement = Requirement(line)
                expected[canonicalize_name(requirement.name)] = requirement
    installed = {canonicalize_name(d.metadata['Name']): d.version for d in distributions()}
    if role in ('vllm', 'sglang') and set(installed) != set(expected):
        raise RuntimeError(
            f'Package inventory changed: missing={sorted(set(expected) - set(installed))}, '
            f'unpinned={sorted(set(installed) - set(expected))}')
    for name, requirement in expected.items():
        if name in installed and not requirement.specifier.contains(installed[name], prereleases=True):
            raise RuntimeError(f'Package mismatch: expected {requirement}, found {installed[name]}')
    for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'apex', 'rocm-sdk-core', 'rocm-sdk-devel'):
        version(name)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f'Expected Python 3.12, found {sys.version}')
    return installed


def check_manifest(sdk):
    """Pin the SDK version and its compiler, runtime and math-library source revisions."""
    actual = json.loads((sdk / 'share/therock/therock_manifest.json').read_text())
    expected = json.loads((ROOT / 'expected-manifest.json').read_text())
    for key in ('rocm_package_version', 'the_rock_commit'):
        if actual[key] != expected[key]:
            raise RuntimeError(f'Unexpected SDK {key}: {actual[key]}')
    sources = {item['submodule_name']: item['pin_sha'] for item in actual['submodules']}
    for name, commit in expected['submodules'].items():
        if sources.get(name) != commit:
            raise RuntimeError(f'Unexpected SDK source: {name}')
    (ROOT / 'therock_manifest.json').write_text(json.dumps(actual, indent=2) + '\n')


def check_clang(sdk):
    """Exercise both SDK instruction selectors with the unsigned-offset regression."""
    for selector in (0, 1):
        output = ROOT / f'unsigned-offset-isel-{selector}.s'
        run(str(sdk / 'lib/llvm/bin/clang'), '-target', 'amdgcn-amd-amdhsa',
            '-mcpu=gfx1250', '-nogpulib', '-O3', '-S', '-x', 'ir',
            '-mllvm', f'-global-isel={selector}', '-mllvm', '-global-isel-abort=1',
            str(ROOT / 'unsigned-offset.ll'), '-o', str(output))
        assembly = output.read_text()
        address = re.search(r'v_lshl_add_u64 v\[(\d+):(\d+)\]', assembly)
        if not address:
            raise RuntimeError(f'Clang selector {selector}: missing widened address')
        low, high = address.groups()
        if not re.search(rf'v_(?:dual_)?mov_b32(?:_e32)? v{high}, 0\b', assembly):
            raise RuntimeError('Unsigned index was not zero-extended')
        if not re.search(rf'global_load_b32 v\d+, v\[{low}:{high}\], off', assembly):
            raise RuntimeError('Load did not use the full unsigned address')


def check_hiprtc_and_translator(sdk):
    """Generate a redistributable fixture from source and check its translated Wave32 descriptor."""
    code = ROOT / 'hiprtc-offset.hsaco'
    run(str(ROOT / 'hiprtc_unsigned_offset'), str(code))
    assembly = subprocess.check_output(
        [str(sdk / 'lib/llvm/bin/llvm-objdump'), '--disassemble', '--mcpu=gfx1250', str(code)], text=True)
    (ROOT / 'hiprtc-offset.s').write_text(assembly)
    if not re.search(r'global_load_b32.*v\[\d+:\d+\], off', assembly):
        raise RuntimeError('HIPRTC used a raw signed offset')
    store = ROOT / 'generated-translations'
    run(sys.executable, str(sdk / 'bin/rocjitsu-pretranslate.py'), '--target', 'gfx1250',
        '--fail-on-skipped', '--bundler', str(sdk / 'lib/llvm/bin/clang-offload-bundler'),
        '--store-root', str(store), '--report-dir', str(ROOT / 'translator-report'), str(code))
    return {
        'generated_fixture': check_cache(store),
        'packaged_cache': check_cache(sdk / 'share/rocjitsu/translations'),
        'translator_sha256': hashlib.sha256(
            (sdk / 'lib/librocjitsu_gfx1250_b0_to_a0.so.0').read_bytes()).hexdigest(),
    }


def cpu_imports(role):
    """Avoid Apex's optional AITER import once AITER's GPU discovery is installed."""
    modules = ('torchaudio', 'apex') if role == 'sdk' else ('torchaudio',)
    return modules + {'vllm': ('vllm._C',), 'sglang': ('sglang', 'sgl_kernel')}.get(role, ())


def check_framework(role):
    """Import CPU-safe native extensions; AITER GPU discovery remains a separate runtime test."""
    import torch
    import torchvision

    if torch.arange(16).sum().item() != 120:
        raise RuntimeError('CPU tensor check failed')
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
    if torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() != [0]:
        raise RuntimeError('Native torchvision NMS check failed')
    for module in cpu_imports(role):
        importlib.import_module(module)


def check_a0_library(site):
    """Require a coherent source-matched master and all generated device assets in both views."""
    manifest = json.loads((ROOT / 'a0/a0-library-manifest.json').read_text())
    expected = json.loads((ROOT / 'expected-manifest.json').read_text())['submodules']['rocm-libraries']
    if manifest['source_revision'] != expected or manifest['architecture_alias'] != 'gfx1250v0':
        raise RuntimeError('A0 library does not match the installed SDK source')
    if len(manifest['sha256']) != 271 or manifest['code_objects'] != 135:
        raise RuntimeError('A0 library inventory differs from the tested generic profile')
    if manifest['master'] not in manifest['sha256']:
        raise RuntimeError('Missing A0 master selection table')
    for view in ('_rocm_sdk_libraries', '_rocm_sdk_devel'):
        base = site / view / 'lib/hipblaslt/library/gfx1250v0'
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'A0 library asset changed: {view}/{name}')
    run('sha256sum', '--check', str(ROOT / 'hipblaslt-host.sha256'))


def check_runtime_repairs(role):
    """Check source-matched A0 assets, CK runtime sources and actual installed patches."""
    site = Path(sysconfig.get_path('purelib'))
    check_a0_library(site)
    if (ROOT / 'aiter-ck-source.txt').read_text().strip() != '96afae1d54fc79c26857de7e1683b0dde1ff3c6f':
        raise RuntimeError('Unexpected AITER CK source')
    check_ck_sources(site)
    if role == 'vllm':
        hashes = json.loads((ROOT / 'amd-vllm/installed.json').read_text())
        if len(hashes) != 12:
            raise RuntimeError('Missing AMD vLLM patches')
        for path, digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Patched vLLM module changed: {path}')
    return {'a0_files_per_view': 271, 'a0_code_objects': 135, 'ck_runtime_assets': 'PASS',
            'amd_vllm_patched_modules': 12 if role == 'vllm' else 0}


def check_ck_sources(site):
    """Require the packaged headers/generator to match the pinned checkout."""
    checkout = Path('/opt/aiter/3rdparty/composable_kernel')
    installed = site / 'aiter_meta/3rdparty/composable_kernel'
    for relative in ('include/ck_tile/core.hpp', 'example/ck_tile/01_fmha/generate.py'):
        original = (checkout / relative).read_bytes()
        if not original or (installed / relative).read_bytes() != original:
            raise RuntimeError(f'Missing or inconsistent CK runtime source: {relative}')
    includes = os.environ.get('CPATH', '').split(':')
    if any(str(checkout / subdir) not in includes for subdir in ('include', 'library/include')):
        raise RuntimeError('CK header lookup is not configured for runtime JIT')
    for flag in ('TRITON_HIP_USE_EXPERT_SCHEDULING', 'TRITON_HIP_USE_ASYNC_COPY'):
        if os.environ.get(flag) != '0':
            raise RuntimeError(f'Required A0 compiler workaround is not enabled: {flag}')


def check_launch_profiles(role):
    """Exercise the installed launcher without importing GPU-dependent serving CLIs."""
    profiles = ('vllm-triton',) if role == 'vllm' else ('sglang-qwen397', 'sglang-compile')
    for profile in profiles:
        result = subprocess.check_output(
            ['/usr/local/bin/sb-serve', '--dry-run', profile, '--help'], text=True)
        plan = json.loads(result)
        if plan['profile'] != profile or '--attention-backend' not in plan['command']:
            raise RuntimeError(f'Invalid installed serving profile: {profile}')
        if plan['environment']['TRITON_HIP_USE_EXPERT_SCHEDULING'] != '0':
            raise RuntimeError('Launcher did not apply the expert-scheduling workaround')
    return list(profiles)


def main():
    """Save an explicit CPU-only report after all required checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('sdk', 'base', 'vllm', 'sglang'))
    role = parser.parse_args().role
    packages = check_packages(role)
    run(sys.executable, '-m', 'pip', 'check')
    sdk = Path(os.environ['ROCM_PATH'])
    check_manifest(sdk)
    check_clang(sdk)
    translator = check_hiprtc_and_translator(sdk)
    check_framework(role)
    repairs = check_runtime_repairs(role) if role != 'sdk' else None
    if role != 'sdk':
        run('sha256sum', '--check', str(ROOT / 'fixed-triton-installed.sha256'))
        run(sys.executable, str(ROOT / 'triton_unsigned_offset.py'), '--expect-fixed')
    if role in ('vllm', 'sglang'):
        for name in ('amd-aiter', 'flydsl', role):
            version(name)
        run('sha256sum', '--check', str(ROOT / 'fixed-flydsl-installed.sha256'))
        run(sys.executable, str(ROOT / 'flydsl_unsigned_offset.py'))
        profiles = check_launch_profiles(role)
    else:
        profiles = []
    report = {'result': 'PASS', 'role': role, 'packages': packages, 'translator': translator,
              'runtime_repairs': repairs,
              'launch_profiles': profiles,
              'cpu_framework_imports': list(cpu_imports(role)),
              'device_dependent_imports': 'NOT_TESTED: AITER and Apex-with-AITER need GPU discovery',
              'gpu_inference': 'NOT_TESTED: CPU build and code-generation checks only'}
    (ROOT / f'validation-{role}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
