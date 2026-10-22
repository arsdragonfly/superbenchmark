# Rebuilding the experimental vLLM and SGLang images

These recipes preserve the **October 15, 2026 ROCm 10.2 nightly serving stack**.
They are separate from the October 18 Python 3.14 SuperBench/Transformer Engine
profile. The serving images use **Python 3.12** and target **gfx1250**; their
AMD Quark and SGLang dependencies do not support that Python 3.14 stack.

The recipes now incorporate the October 20 retest repairs. Raw framework
commands remain available; the installed `sb-serve` command selects explicit
tested profiles rather than globally changing every workload's BLAS or
attention backend.

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
3. Generate a coherent [generic A0 hipBLASLt library](../dockerfile/rocm10.2-serving/a0-library.dockerfile)
   from the matching October 15 library source. Build
   [common serving dependencies and AITER](../dockerfile/rocm10.2-serving/common.dockerfile),
   installing the A0 assets in both SDK views and restoring pinned CK runtime
   headers/generators omitted by the minimal AITER wheel.
4. Rebuild [Triton and Triton kernels](../dockerfile/rocm10.2-serving/triton.dockerfile)
   against the corrected LLVM.
5. Build [FlyDSL and its independent LLVM/MLIR compiler](../dockerfile/rocm10.2-serving/flydsl.dockerfile).
6. Build [vLLM](../dockerfile/rocm10.2-serving/vllm.dockerfile) and/or
   [SGLang and its HIP kernels](../dockerfile/rocm10.2-serving/sglang.dockerfile).
   vLLM applies AMD's FLA and Qwen enablement patches before building its
   wheel, then verifies all 12 patched installed Python modules.
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

## Integrated repairs and launch profiles

Common image defaults now include both:

```bash
TRITON_HIP_USE_EXPERT_SCHEDULING=0
TRITON_HIP_USE_ASYNC_COPY=0
```

The same pair is also set by the SuperBench smoke runner. This does not
retroactively modify already-published images or their recorded smoke results.

The hipBLASLt repair is **not** a BLAS-backend or precision substitution.
The nightly supplied A0 code objects without a master selection table.
The new source-build stage selects unmodified generic `gfx1250-strict` logic,
uses the supported `--architecture gfx1250 --gfx1250v0` alias, and installs
the coherent 271-file library (135 ELF objects) in both the canonical and
development SDK views. It preserves the host hipBLASLt DSO and verifies
its hash. The source revision must match the installed ROCm libraries.

The vLLM wheel also contains:

* [ROCm/vllm#1359](https://github.com/ROCm/vllm/pull/1359): FLA Triton changes.
* [ROCm/vllm#1354](https://github.com/ROCm/vllm/pull/1354): Qwen3.8/gfx1250
  MoE and GEMM dispatch changes. The backport retains this source pin's
  existing `K % 256` compatibility guard.

Those changes alone do not fix vLLM's native `ROCM_ATTN` path: its pinned
gfx1250 WMMA helper explicitly traps. Use the Triton profile below.
Similarly, restoring AITER's CK sources makes runtime JIT possible but does
not make unsupported CK vision-attention kernels valid for gfx1250.

| Profile | Additional behavior | October 20 evidence |
|---|---|---|
| `vllm-triton` | `--attention-backend TRITON_ATTN`; no forced eager mode or BLAS override | Qwen2.5 eager and graph serving both passed |
| `sglang-qwen397` | Text `--attention-backend triton`, vision `--mm-attention-backend triton_attn` | Single-GPU Qwen397 original 16-token first request and complete sanity answers passed |
| `sglang-compile` | Triton text attention, `--enable-torch-compile`, ordinary PyTorch GEMMs through rocBLAS | Two fresh-cache runs passed autotuning, 52 graph shapes, and inference |

Only `sglang-compile` sets `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`ROCBLAS_USE_HIPBLASLT=0`, and
`SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.
It does **not** disable the outer SGLang CUDA graphs or torch.compile.
Explicit AITER GEMMs retain their own implementation.
This remains a workaround for the observed hipBLASLt/graph-capture interaction,
not a claim that its underlying cause is fixed.

All profiles force both Triton flags off and reject conflicting profile-owned
CLI options. Use the raw `vllm` or `python3 -m sglang.launch_server` commands for
deliberately different experiments. Compiler cache defaults use a new
`a0-retest` namespace; use empty, separate caches when comparing configurations.
There is no automatic server startup or change to the image's default shell.

### Running the profiles

On an idle MI455X A0 node, set an absolute model cache path. The examples
expose the API only on host loopback and retain private shared memory:

```bash
HF_CACHE="${HOME}/.cache/huggingface/hub"
mkdir -p "${HF_CACHE}"
GPU_ARGS=(--rm --device=/dev/kfd --device=/dev/dri
  --group-add "$(stat -c %g /dev/kfd)"
  --group-add "$(stat -c %g /dev/dri/renderD128)"
  --ipc=private --shm-size=16g --ulimit memlock=-1:-1
  -e HIP_VISIBLE_DEVICES=0 -e HSA_HOTSWAP_ENABLE=1 -e HSA_TOOLS_LIB=
  -e HF_HUB_CACHE=/hf-cache -v "${HF_CACHE}:/hf-cache"
  -p 127.0.0.1:8000:8000)
```

Run **one** of these at a time:

```bash
docker run "${GPU_ARGS[@]}" \
  superbench-vllm:0.31.0-rocm10.2.0a20261015-llvmfix-a0fix \
  sb-serve vllm-triton Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --max-model-len 4096 --gpu-memory-utilization 0.80 \
  --host 0.0.0.0 --port 8000
# Add --enforce-eager to reproduce the separately tested eager mode.

docker run "${GPU_ARGS[@]}" \
  superbench-sglang:0.5.18-rocm10.2.0a20261015-llvmfix-a0fix \
  sb-serve sglang-qwen397 --model-path amd/Qwen3.5-397B-A17B-MXFP4 \
  --revision edf0958bc3734dda98a9d191cc7a0a83c4f42821 \
  --tp 1 --mem-fraction-static 0.90 --context-length 32768 \
  --host 0.0.0.0 --port 8000

docker run "${GPU_ARGS[@]}" \
  superbench-sglang:0.5.18-rocm10.2.0a20261015-llvmfix-a0fix \
  sb-serve sglang-compile --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --tp 1 --mem-fraction-static 0.90 --context-length 32768 \
  --trust-remote-code --enable-metrics --host 0.0.0.0 --port 8000
```

Inspect a profile without a GPU using, for example,
`docker run --rm <vllm-image> sb-serve --dry-run vllm-triton Qwen/Qwen2.5-1.5B-Instruct`.
It prints the argument array and profile-specific environment, without secrets
from the inherited environment. Readiness alone is not sufficient qualification:
verify actual completions and kernel health. Qwen397 needs approximately
239 GB of downloaded weights and enough output tokens for its reasoning.

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
| AITER's CK runtime sources | `96afae1d54fc79c26857de7e1683b0dde1ff3c6f` |
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
* A0 master/library hashes in both SDK views, unchanged host hipBLASLt,
  pinned CK headers/generators and their include paths, both Triton flags,
  and the actual installed AMD-patched vLLM modules.

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

### Integrated recipe validation on October 22

The same build command above, using the two qualified compiler cache overrides,
rebuilt **both applications successfully**, including the new A0 library
generated from source. Fresh-container validators passed with the same
238/191 pinned package inventories. They additionally checked the installed
A0 master/assets, CK runtime sources and header paths, both Triton flags,
AMD-patched modules, and all three installed launch profiles.

All 12 installed vLLM modules changed by the AMD patches are byte-identical
to the October 20 GPU-tested candidate. The regenerated A0 library's executable
code and kernel descriptor/metadata sections also match the tested objects;
ELF linking metadata differs, so this is not a claim of identical image digests.

```bash
python3 -B -m unittest -q \
  tests.test_rocm102_serving tests.test_rocm102_smoke_runner tests.test_rocm102_dockerfiles
flake8 dockerfile/rocm10.2-serving/*.py dockerfile/rocm10.2-nightly/run_smoke.py \
  tests/test_rocm102_serving.py tests/test_rocm102_smoke_runner.py
git diff --check
```

Results: **35 tests passed**, **lint passed**, and **whitespace check passed**.
No GPU workloads were run during this recipe integration and no ACR release
tags were changed. The hardware evidence remains the explicitly scoped
October 20 retests, not a new full GPU qualification of the rebuilt images.

These recipes preserve AITER/Triton and `NCCL_MIN_NCHANNELS=64`. vLLM excludes the
unqualified TileLang compiler; SGLang excludes Petit/Wave/diffusion backends.
They rebuild the generic A0 library from their **own October 15 source pin**,
not the October 18 SuperBench library, and do not inherit its 54-case qualification.

The October 20 GPU results above were obtained with isolated experimental
images on four MI455X A0 devices, kernel `6.17.0-42-generic`, and eight-backport
amdgpu srcversion `DD338B73DA3807A0184A715`. They establish tested configurations,
not full-size performance or long-duration stability. Raw/default serving
paths still have the native-attention and capture limitations described above.
Rebuilt OCI images require their own GPU qualification before promotion.
No GPU tests should run on a shared node without confirming availability.
