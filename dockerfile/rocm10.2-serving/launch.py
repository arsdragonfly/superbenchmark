#!/opt/venv/bin/python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Launch an explicit MI455X A0 serving profile; raw framework commands remain available."""

import argparse
import json
import os
from pathlib import Path
import sys


PROFILES = {
    'vllm-triton': ('vllm', ['vllm', 'serve'], ['--attention-backend', 'TRITON_ATTN']),
    'sglang-qwen397': (
        'sglang', [sys.executable, '-m', 'sglang.launch_server'],
        ['--attention-backend', 'triton', '--mm-attention-backend', 'triton_attn']),
    'sglang-compile': (
        'sglang', [sys.executable, '-m', 'sglang.launch_server'],
        ['--attention-backend', 'triton', '--enable-torch-compile']),
}


def launch_plan(profile, arguments, role):
    """Keep tested profile controls explicit and reject ambiguous overrides."""
    framework, command, flags = PROFILES[profile]
    if role != framework:
        raise ValueError(f'{profile} requires the {framework} image, not {role}')
    reserved = {flag for flag in flags if flag.startswith('--')}
    reserved.add('--attention-config')
    if profile == 'sglang-compile':
        reserved.update(('--cuda-graph-backend-decode', '--disable-cuda-graph', '--disable-torch-compile'))
    for argument in arguments:
        if argument.split('=', 1)[0] in reserved:
            raise ValueError(f'{argument} conflicts with the selected profile; use the raw framework CLI to override')
    environment = {'TRITON_HIP_USE_ASYNC_COPY': '0', 'TRITON_HIP_USE_EXPERT_SCHEDULING': '0'}
    if profile == 'sglang-compile':
        environment.update(TORCH_BLAS_PREFER_HIPBLASLT='0', ROCBLAS_USE_HIPBLASLT='0',
                           SGLANG_TORCH_COMPILE_MODE='max-autotune-no-cudagraphs')
    return [*command, *arguments, *flags], environment


def main():
    """Exec the framework directly so signals and exit codes are preserved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dry-run', action='store_true', help='Print the command and profile environment without GPU work.')
    parser.add_argument('profile', choices=tuple(PROFILES))
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Framework arguments, including the model.')
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not arguments:
        parser.error('Supply framework arguments, including the model or --help')
    role = Path('/opt/base-provenance/serving-role').read_text().strip()
    try:
        command, environment = launch_plan(args.profile, arguments, role)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'profile': args.profile, 'command': command, 'environment': environment}), flush=True)
    if not args.dry_run:
        os.execvpe(command[0], command, {**os.environ, **environment})


if __name__ == '__main__':
    main()
