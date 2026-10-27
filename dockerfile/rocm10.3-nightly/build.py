# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build the four October 26 nightly images on a CPU-only Docker builder."""

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RECIPES = 'dockerfile/rocm10.3-nightly'
SDK_VERSION = '10.3.0a20261026'
AITER = '2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0'
LLVM = 'superbench-serving-triton-llvm:5f07f818-3b2c3b02'
OUTPUTS = {
    'base': 'sb-rocm103-base:20261026-py314',
    'superbench': 'sb-rocm103-superbench:20261026',
    'vllm': 'sb-rocm103-vllm:20261026',
    'sglang': 'sb-rocm103-sglang:20261026',
}


def build(recipe, tag, arguments=(), target=None):
    """Assemble a Docker command with no shell interpolation."""
    command = ['docker', 'buildx', 'build', '--load', '--progress=plain', '-f', recipe, '-t', tag]
    if target:
        command += ['--target', target]
    for argument in arguments:
        command += ['--build-arg', argument]
    return command + ['.']


def plan(args):
    """Keep every dependency explicit, sharing the two independent compiler artifacts."""
    llvm = args.triton_llvm_image or LLVM
    if not args.triton_llvm_image:
        yield build('dockerfile/rocm10.2-nightly-base.dockerfile', llvm,
                    (f'LLVM_BUILD_JOBS={args.jobs}',), 'llvm-source')
    versions = []
    if any(role in args.images for role in ('base', 'superbench')):
        versions.append('314')
    if any(role in args.images for role in ('vllm', 'sglang')):
        versions.append('312')
    for py in versions:
        sdk = f'sb-rocm103-sdk:20261026-py{py}'
        base = f'sb-rocm103-base:20261026-py{py}'
        sdk_args = []
        if py == '312':
            sdk_args = [
                'UBUNTU_IMAGE=ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac',
                'PYTHON_VERSION=3.12', 'DEVICE_EXTRAS=devel,libraries,device-gfx1250',
                'TORCH_DEVICE_EXTRAS=device-gfx1250']
        yield build(f'{RECIPES}/sdk.dockerfile', sdk, sdk_args)
        yield build(f'{RECIPES}/base.dockerfile', base,
                    (f'SDK_IMAGE={sdk}', f'LLVM_IMAGE={llvm}', f'BUILD_JOBS={args.jobs}',
                     f'SB_SOURCE_REVISION={args.source_revision}'))
        if py == '314' and 'superbench' not in args.images:
            continue
        flydsl = getattr(args, f'flydsl{py}_image') or f'sb-rocm103-flydsl:20261026-py{py}'
        if not getattr(args, f'flydsl{py}_image'):
            yield build('dockerfile/rocm10.2-serving/flydsl.dockerfile', flydsl,
                        (f'BASE_IMAGE={base}', f'BUILD_JOBS={args.jobs}'))
        if py == '314':
            yield build('dockerfile/rocm10.2-nightly.dockerfile', OUTPUTS['superbench'], (
                f'BASE_IMAGE={base}', f'FLYDSL_WHEEL_IMAGE={flydsl}', f'SB_SOURCE_REVISION={args.source_revision}',
                f'ROCM_VERSION={SDK_VERSION}', f'TORCH_VERSION=2.13.0+rocm{SDK_VERSION}',
                f'TRITON_VERSION=3.8.0+gita3c5afb7.rocm{SDK_VERSION}',
                'ROCM_SYSTEMS_COMMIT=39fd0ea559a5970e42afd87c30f275744e00b8d1',
                'ROCM_TESTS_SHA256=bf13fff67ffcc231bc004400ddb8740437e1d4037cc20aaf379ca6a59890d206',
                f'AITER_COMMIT={AITER}', 'AITER_PACKAGE_VERSION=0.1.26.dev0+g2cb3d1758',
                'FLYDSL_COMMIT=1f3511677662afc58bcf141dec812ccf05334fd8',
                f'NUM_MAKE_JOBS={args.jobs}', 'AMDGPU_TARGETS=gfx1250'))
            yield ['docker', 'run', '--rm', '--network=none', OUTPUTS['superbench'],
                   'python3', '/opt/base-provenance/check_base.py', '--role', 'superbench']
        else:
            yield from serving_plan(args, base, flydsl)


def serving_plan(args, base, flydsl):
    """Build the two serving roles against one matched Python 3.12 foundation."""
    common = 'sb-rocm103-common:20261026'
    yield build(f'{RECIPES}/common.dockerfile', common, (f'BASE_IMAGE={base}', f'BUILD_JOBS={args.jobs}'))
    for role in ('vllm', 'sglang'):
        if role not in args.images:
            continue
        app = f'sb-rocm103-{role}-build:20261026'
        options = [f'BASE_IMAGE={common}', f'BUILD_JOBS={args.jobs}',
                   f'CONSTRAINTS_FILE={RECIPES}/constraints-{role}.txt']
        if role == 'vllm':
            options.append(f'VLLM_VERSION=0.31.0+rocm{SDK_VERSION}.llvmfix')
        yield build(f'dockerfile/rocm10.2-serving/{role}.dockerfile', app, options)
        yield build(f'{RECIPES}/qualify.dockerfile', OUTPUTS[role], (
            f'APP_IMAGE={app}', f'FLYDSL_IMAGE={flydsl}', f'ROLE={role}',
            f'SB_SOURCE_REVISION={args.source_revision}'))
        yield ['docker', 'run', '--rm', '--network=none', OUTPUTS[role],
               'python3', '/opt/base-provenance/check_base.py', '--role', role]


def main():
    """Build selected images; source identity is explicit for archived build contexts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', choices=tuple(OUTPUTS), nargs='+', default=list(OUTPUTS))
    parser.add_argument('--jobs', type=int, default=24)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--triton-llvm-image')
    parser.add_argument('--flydsl312-image')
    parser.add_argument('--flydsl314-image')
    args = parser.parse_args()
    if args.jobs < 1 or not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        parser.error('Positive jobs and a full source commit are required')
    for command in plan(args):
        print('RUN ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
