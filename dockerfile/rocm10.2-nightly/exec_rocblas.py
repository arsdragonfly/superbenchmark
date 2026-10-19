# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run SuperBench with the qualified ordinary-PyTorch rocBLAS backend."""

import argparse
import json
import os
import runpy
import shutil
import sys


def main():
    """Keep backend selection in the interpreter that executes SuperBench."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.validate_only and not args.command:
        parser.error('SuperBench arguments are required')
    os.environ['TORCH_BLAS_PREFER_HIPBLASLT'] = '0'
    os.environ['ROCBLAS_USE_HIPBLASLT'] = '0'
    import torch

    if not torch.version.hip:
        raise RuntimeError('The rocBLAS profile requires HIP-enabled PyTorch')
    torch.backends.cuda.preferred_blas_library('cublas')
    selected = str(torch.backends.cuda.preferred_blas_library())
    if selected != '_BlasBackend.Cublas':
        raise RuntimeError(f'PyTorch did not select rocBLAS: {selected}')
    print('SB_BLAS_BACKEND=' + json.dumps({
        'backend': 'rocblas', 'torch_selector': selected,
        'torch_version': torch.__version__, 'hip_version': torch.version.hip,
        'scope': 'Ordinary PyTorch GEMMs; explicit hipBLASLt and TE FP8 calls retain their own backend',
    }), flush=True)
    if args.validate_only:
        return
    executable = shutil.which('sb')
    if executable is None:
        raise RuntimeError('SuperBench CLI is not installed')
    sys.argv = [executable, *args.command]
    runpy.run_path(executable, run_name='__main__')


if __name__ == '__main__':
    main()
