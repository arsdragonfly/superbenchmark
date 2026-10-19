# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise FlyDSL's actual MLIR/ROCDL code generator without a GPU."""

from pathlib import Path
import re

from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text


SOURCE = """
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @read_u32(%src: !llvm.ptr<1>, %out: !llvm.ptr<1>, %offset: i32) kernel {
      %lane = gpu.thread_id x
      %lane32 = arith.index_cast %lane : index to i32
      %index = arith.addi %lane32, %offset : i32
      %wide = arith.extui %index : i32 to i64
      %p = llvm.getelementptr %src[%wide] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %value = llvm.load %p : !llvm.ptr<1> -> f32
      %lane64 = arith.extui %lane32 : i32 to i64
      %q = llvm.getelementptr %out[%lane64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %value, %q : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
"""


def main():
    """Require zero-extended addressing through the independent FlyDSL LLVM."""
    backend = RocmBackend(RocmBackend.make_target('gfx1250'))
    with _create_mlir_context() as context:
        module = ir.Module.parse(SOURCE, context=context)
        fragments = backend.pipeline_fragments(compile_hints={})
        if 'format=fatbin' not in fragments[-1]:
            raise RuntimeError(f'Unexpected FlyDSL compiler pipeline: {fragments}')
        fragments[-1] = fragments[-1].replace('format=fatbin', 'format=isa')
        PassManager.parse('builtin.module(' + ','.join(fragments) + ')', context).run(module.operation)
        assembly = _extract_isa_text(str(module))
    Path('/opt/base-provenance/flydsl-offset.s').write_text(assembly)
    operations = [line for line in assembly.splitlines() if 'global_load' in line]
    if not any(re.search(r'v\[\d+:\d+\]', line) and ', off' in line for line in operations):
        raise RuntimeError(f'FlyDSL did not use a full unsigned address\n{assembly}')
    if 'v_lshl_add_u64' not in assembly:
        raise RuntimeError(f'FlyDSL did not widen the unsigned address\n{assembly}')
    print('FlyDSL unsigned-offset code generation PASS')


if __name__ == '__main__':
    main()
