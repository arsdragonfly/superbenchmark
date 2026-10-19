# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build the pinned Python 3.12 serving images without private build dependencies."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RECIPES = 'dockerfile/rocm10.2-serving'
NIGHTLY = '10.2.0a20261015'
LLVM = 'superbench-serving-triton-llvm:5f07f818-3b2c3b02'
FLYDSL = 'superbench-serving-flydsl:0.4.0-llvmfix'
BASE = f'superbench-serving-sdk:{NIGHTLY}-py3.12'
COMMON = f'superbench-serving-build:{NIGHTLY}-py3.12'
FIXED = f'superbench-serving-base:{NIGHTLY}-py3.12-llvmfix'
VERSIONS = {'vllm': '0.31.0', 'sglang': '0.5.18'}


def positive(value):
    """Reject invalid build parallelism before invoking Docker."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Build jobs must be positive')
    return number


def build_command(recipe, tag, arguments=(), target=None):
    """Use the repository root as the context for shared compiler helpers."""
    command = ['docker', 'buildx', 'build', '--load', '--progress=plain', '-f', recipe, '-t', tag]
    if target:
        command += ['--target', target]
    for argument in arguments:
        command += ['--build-arg', argument]
    return command + ['.']


def commands(args):
    """Yield every prerequisite in order; cached compiler overrides are opt-in."""
    llvm = args.triton_llvm_image or LLVM
    flydsl = args.flydsl_wheel_image or FLYDSL
    if not args.triton_llvm_image:
        yield build_command(
            'dockerfile/rocm10.2-nightly-base.dockerfile', llvm,
            (f'LLVM_BUILD_JOBS={args.llvm_jobs}',), target='llvm-source')
    yield build_command(f'{RECIPES}/base.dockerfile', BASE)
    yield build_command(
        f'{RECIPES}/common.dockerfile', COMMON, (f'BASE_IMAGE={BASE}', f'BUILD_JOBS={args.jobs}'))
    yield build_command(
        f'{RECIPES}/triton.dockerfile', FIXED,
        (f'BASE_IMAGE={COMMON}', f'LLVM_IMAGE={llvm}', f'BUILD_JOBS={args.jobs}'))
    if not args.flydsl_wheel_image:
        yield build_command(
            f'{RECIPES}/flydsl.dockerfile', flydsl,
            (f'BASE_IMAGE={FIXED}', f'BUILD_JOBS={args.llvm_jobs}'))
    for role in VERSIONS if args.app == 'all' else (args.app,):
        intermediate = f'superbench-{role}-build:{VERSIONS[role]}-rocm{NIGHTLY}'
        final = f'superbench-{role}:{VERSIONS[role]}-rocm{NIGHTLY}-llvmfix-a0fix'
        yield build_command(
            f'{RECIPES}/{role}.dockerfile', intermediate,
            (f'BASE_IMAGE={FIXED}', f'BUILD_JOBS={args.jobs}'))
        yield build_command(
            f'{RECIPES}/qualify.dockerfile', final,
            (f'APP_IMAGE={intermediate}', f'FLYDSL_IMAGE={flydsl}', f'ROLE={role}'))
        yield ['docker', 'run', '--rm', '--network=none', final,
               'python3', '/opt/base-provenance/validate.py', role]


def main():
    """Build both applications by default, stopping at the first failed command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', choices=('all', *VERSIONS), default='all')
    parser.add_argument('--jobs', type=positive, default=24)
    parser.add_argument('--llvm-jobs', type=positive, default=32)
    parser.add_argument('--triton-llvm-image', help='Optional qualified image containing /opt/triton-llvm.')
    parser.add_argument('--flydsl-wheel-image', help='Optional qualified Python 3.12 image containing /wheels.')
    args = parser.parse_args()
    for command in commands(args):
        print('RUN ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
