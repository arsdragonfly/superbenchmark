# Rebuilding the experimental vLLM and SGLang images

These recipes preserve the **October 15, 2026 ROCm 10.2 nightly serving stack**.
They are separate from the October 18 Python 3.14 SuperBench/Transformer Engine
profile. The serving images use **Python 3.12** and target **gfx1250**; their
AMD Quark and SGLang dependencies do not support that Python 3.14 stack.

## Build both images

Requirements: Linux x86-64, Docker with Buildx/BuildKit, Python 3, network access
to the public package/source repositories, and permission to run Docker.
A GPU, ACR credentials, private compiler images, and files from a previous
session are **not required**. Budget hundreds of GiB of disk space, substantial
RAM, and several hours for the first LLVM/MLIR and native-extension builds.
Lower the job counts for a smaller builder.

From this experimental branch:

```bash
python3 dockerfile/rocm10.2-serving/build.py --jobs 24 --llvm-jobs 32
```

The script finds the repository root from its own location, streams build
output, loads the resulting images into Docker, and stops on any build or
validation failure. It builds all prerequisites automatically. Use
`--app vllm` or `--app sglang` to build only one application and its prerequisites.
No SuperBench native submodules are needed for this serving-only build.

Output tags:

```text
superbench-vllm:0.31.0-rocm10.2.0a20261015-llvmfix-a0fix
superbench-sglang:0.5.18-rocm10.2.0a20261015-llvmfix-a0fix
```

The command does not publish anything. These are source-rebuilt images, not a
claim that their OCI digests equal the previously published experimental images.

## Complete build chain

The [build script](../dockerfile/rocm10.2-serving/build.py) wires these steps:

1. Build Triton's LLVM from the shared
   [nightly base recipe's `llvm-source` stage](../dockerfile/rocm10.2-nightly-base.dockerfile).
   Only that CPU compiler stage is built, not the October 18 SDK.
2. Build the [Python 3.12 SDK](../dockerfile/rocm10.2-serving/base.dockerfile).
3. Build [common serving dependencies and AITER](../dockerfile/rocm10.2-serving/common.dockerfile).
4. Rebuild [Triton and Triton kernels](../dockerfile/rocm10.2-serving/triton.dockerfile)
   against the corrected LLVM.
5. Build [FlyDSL and its independent LLVM/MLIR compiler](../dockerfile/rocm10.2-serving/flydsl.dockerfile).
6. Build [vLLM](../dockerfile/rocm10.2-serving/vllm.dockerfile) and/or
   [SGLang and its HIP kernels](../dockerfile/rocm10.2-serving/sglang.dockerfile).
7. Apply the [final compiler qualification layer](../dockerfile/rocm10.2-serving/qualify.dockerfile),
   which replaces the dependency's FlyDSL wheel with the repaired wheel,
   protects it against replacement, and validates the final installed stack.

The context for every Docker build is the repository root, not the individual
recipe directory. The driver handles this automatically. Shared helpers and
the compact LLVM patch remain in
[rocm10.2-nightly](../dockerfile/rocm10.2-nightly), rather than duplicated here.

### Optional compiler caches

Only when already built and qualified, compiler artifact images can be reused:

```bash
python3 dockerfile/rocm10.2-serving/build.py \
  --triton-llvm-image superbench-serving-triton-llvm:5f07f818-3b2c3b02 \
  --flydsl-wheel-image superbench-serving-flydsl:0.4.0-llvmfix
```

The first must contain `/opt/triton-llvm`; the second must contain the corrected
**Python 3.12 FlyDSL 0.4.0** wheel under `/wheels` and compiler evidence under
`/opt/provenance`. The October 18 Python 3.14 FlyDSL 0.3.4.1 wheel is not compatible.
Overrides skip only the compiler rebuilds, not the final regression tests.

## Frozen inputs

| Component | Pin |
|---|---|
| ROCm / PyTorch | `10.2.0a20261015` / `2.13.0+rocm10.2.0a20261015` |
| TheRock | `98a3c880016a5d7fae72dda6c5a60356a4fcce2b` |
| rocm-systems | `b2d4b1ec26326e45153e36c7cb2eb94458b72188` |
| rocm-libraries | `3dc098eb2db3c8329b7d447bab37021d529bc23d` |
| SDK LLVM | `bdc35841c97ff51400273b91d7bc05c96e19c2d9` |
| Triton / its LLVM | `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25` / `5f07f818b51b786b0a87b4e514882600ecba112f` |
| FlyDSL / its LLVM | `1f3511677662afc58bcf141dec812ccf05334fd8` / `e2a39f504fee836e4def9581bed817ecc327b9dc` |
| AITER | `4eb7dcac7e5f5abf566246534809644f73f07b49` |
| vLLM 0.31.0 | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| SGLang 0.5.18 | `478b0b00627bfba44e273aaad6ebefd8decb72d8` |
| Rust toolchain | `1.95.0` |

Triton and FlyDSL each receive LLVM unsigned-offset correction
`3b2c3b02c0374cba15e30739b71780ad820e6485`. Their versions are retained for
compatibility, but the rebuilt wheels and native libraries are hash-pinned
inside the image. `PIP_CONSTRAINT` protects the fixed compiler packages.
Do not unset it or reinstall unpatched public compiler wheels.

The [core requirements](../dockerfile/rocm10.2-serving/requirements-base.lock)
and the [base](../dockerfile/rocm10.2-serving/constraints-base.txt),
[vLLM](../dockerfile/rocm10.2-serving/constraints-vllm.txt), and
[SGLang](../dockerfile/rocm10.2-serving/constraints-sglang.txt) dependency
constraints pin the versions recorded from the previously built images.
Application-specific constraints preserve their required NumPy/fsspec
downgrades. The final validator checks the installed inventory against them.

This is a pinned-source/package rebuild, **not a hermetic or bit-for-bit
reproducible build**. Ubuntu package repositories and bootstrap downloads are
external inputs. AMD nightly artifacts are subject to retention: preserve
the images/wheels if long-term offline rebuilds are needed. Missing pinned
packages fail the build rather than silently selecting a newer nightly.

## Validation and limitations

Build-time checks run without GPU devices:

* SDK version/source pins, package constraints and `pip check`.
* Native CPU tensor/NMS and framework-extension imports.
  AITER and Apex-with-AITER perform GPU discovery, so the CPU build checks
  their installed versions/dependencies rather than faking a GPU import.
* Both SDK LLVM instruction selectors, HIPRTC/COMGR, four Triton
  load/store/atomic tests, and FlyDSL's real MLIR/ROCDL lowering path.
* The stock translator processes a kernel compiled from checked-in HIP source;
  both its output and the packaged cache are checked for intact Wave32
  descriptor bits. Empty or descriptorless caches fail validation.
* Fixed compiler native-library hashes are checked after application installs.

Captured runtime binaries, credentials, host logs, and private fixtures are not
build inputs. The source-generated translator fixture replaces the old
session-only exact-binary replay; it is not a claim of equivalent GPU coverage.

Repository contract tests:

```bash
python3 -m unittest -v tests.test_rocm102_serving
```

### Validation performed on October 19

Both applications were rebuilt on a CPU-only host and passed fresh-container
validation with **238 pinned installed packages for vLLM** and **191 for SGLang**.
The generated translator test checked one kernel descriptor; the packaged
translation cache checked **112 objects / 336 descriptors**.

The build command used these previously qualified compiler artifacts to avoid
recompiling both LLVM trees:

```bash
python3 dockerfile/rocm10.2-serving/build.py --jobs 24 --llvm-jobs 32 \
  --triton-llvm-image superbench-triton-llvm:5f07f818-3b2c3b02-compact \
  --flydsl-wheel-image superbench-flydsl:0.4.0-llvmfix
```

Result: **passed**, including both application builds and their final validators.
The final explicit AITER source-version pin was also rebuilt and verified.
Both source-compiler Dockerfiles passed `docker buildx build --check`; the LLVM
source builds themselves were not repeated during this validation.
Without the optional overrides, the documented default command builds those
compiler artifacts from the checked-in recipes.

Additional exact checks:

```bash
python3 -m unittest -q tests.test_rocm102_serving tests.test_rocm102_dockerfiles
flake8 dockerfile/rocm10.2-serving/*.py tests/test_rocm102_serving.py
git diff --check
```

Results: **19 tests passed**, **lint passed**, and **whitespace check passed**.
The rebuilt images remained local to the CPU builder; no ACR tags were changed.

These recipes intentionally preserve the serving profiles: AITER/Triton,
`TRITON_HIP_USE_ASYNC_COPY=0`, and `NCCL_MIN_NCHANNELS=64`. vLLM excludes the
unqualified TileLang compiler; SGLang excludes Petit/Wave/diffusion backends.
They do **not** inherit the October 18 SuperBench image's generic A0 hipBLASLt
fallback library or its 54-case qualification.

**Real model inference remains separately qualified.** On October 19, the
existing October 15 vLLM image failed Qwen2.5-1.5B initialization with
`HIPBLAS_STATUS_INVALID_VALUE` during hipBLASLt algorithm selection. The
original HSA-exception comparison is unfinished. Checking these build recipes
into the branch does not fix that runtime issue. No GPU tests should run on
a shared node without confirming availability.
