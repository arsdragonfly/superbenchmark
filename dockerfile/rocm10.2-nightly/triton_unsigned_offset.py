# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compile unsigned-address regression kernels without requiring a GPU."""

import argparse
import os
import re
from pathlib import Path

os.environ['AMDGCN_USE_BUFFER_OPS'] = '0'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '0'

# Configure address lowering before Triton's backend is imported.
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--expect-fixed', action='store_true')
args = parser.parse_args()


@triton.jit
def read_u32(src, out, offset, BLOCK: tl.constexpr):
    """Load through an unsigned element offset."""
    lane = tl.arange(0, BLOCK)
    index = lane.to(tl.uint32) + offset
    value = tl.load(src + index)
    tl.store(out + lane, value)


@triton.jit
def read_byte_u32(src, out, offset, BLOCK: tl.constexpr):
    """Load through an unsigned byte offset."""
    lane = tl.arange(0, BLOCK)
    index = (lane * 4).to(tl.uint32) + offset
    pointer = (src.to(tl.pointer_type(tl.int8)) + index).to(tl.pointer_type(tl.float32))
    tl.store(out + lane, tl.load(pointer))


@triton.jit
def store_u32(out, offset, BLOCK: tl.constexpr):
    """Store through an unsigned element offset."""
    lane = tl.arange(0, BLOCK)
    index = lane.to(tl.uint32) + offset
    tl.store(out + index, lane.to(tl.float32))


@triton.jit
def atomic_u32(out, offset, BLOCK: tl.constexpr):
    """Perform an atomic update through an unsigned element offset."""
    lane = tl.arange(0, BLOCK)
    index = lane.to(tl.uint32) + offset
    tl.atomic_add(out + index, 1.0, sem='relaxed')


output = Path(os.environ.get('PROBE_OUTPUT', '/tmp/triton-offset'))
output.mkdir(parents=True, exist_ok=True)
for function, signature, instruction in (
    (read_u32, {'src': '*fp32', 'out': '*fp32', 'offset': 'u32'}, 'global_load'),
    (read_byte_u32, {'src': '*fp32', 'out': '*fp32', 'offset': 'u32'}, 'global_load'),
    (store_u32, {'out': '*fp32', 'offset': 'u32'}, 'global_store'),
    (atomic_u32, {'out': '*fp32', 'offset': 'u32'}, 'global_atomic'),
):
    source = ASTSource(function, signature=signature, constexprs={'BLOCK': 128})
    kernel = triton.compile(source, target=GPUTarget('hip', 'gfx1250', 32))
    for name in ('ttir', 'llir', 'amdgcn'):
        (output / f'{function.__name__}.{name}').write_text(kernel.asm[name])
    assembly = kernel.asm['amdgcn']
    if args.expect_fixed:
        operations = [line for line in assembly.splitlines() if instruction in line]
        if not any(re.search(r'v\[\d+:\d+\]', line) and ', off' in line for line in operations):
            raise RuntimeError(f'{function.__name__}: missing full 64-bit global address\n{assembly}')
        if not re.search(r'v_(?:lshl_add_u64|add_nc_u64|add_co_ci_u32)', assembly):
            raise RuntimeError(f'{function.__name__}: unsigned address was not widened\n{assembly}')
        print(f'Triton unsigned-offset regression passed: {function.__name__}')
    else:
        print(f'Compiled {function.__name__}; assembly saved without asserting correctness')
