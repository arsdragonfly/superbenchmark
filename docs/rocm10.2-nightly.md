# Experimental MI455X A0: ROCm 10.2 nightly

This profile provides a reproducible ROCm/PyTorch foundation, a full SuperBench
overlay, and a **54-case functional smoke test**. It is not a throughput,
long-stress, fabric, or full-size-model qualification.

The nightly was selected on October 18, 2026 and is deliberately frozen rather
than following a moving `latest` tag:

| Component | Pin |
|---|---|
| ROCm SDK | `10.2.0a20261018` |
| Python / PyTorch | Python 3.14 / `2.13.0+rocm10.2.0a20261018` |
| TheRock | `ac008259bb0d7be08d4a2ce36dab2e46383c9e63` |
| rocm-systems | `2a8fffb7d64c59f83fe0d2efa22ed31d87d8adbf` |
| rocm-libraries | `8a29d2e82b93f5065c7fe5630248d3689f0c3f06` |
| SDK LLVM | `bdc35841c97ff51400273b91d7bc05c96e19c2d9` |
| Triton | `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25` |
| Transformer Engine | `e7835ed1b134f56e58c575fba2455471064cab9b` (2.15.0) |
| AITER | `89a47b84ac4b576339a50047e290c383c2377389` |
| Megatron Core | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` (0.19.2) |
| FlyDSL | `9dda109684ce26b97d2504503378df5266fb41b0` (0.3.4.1) |

## Image layers

* [Nightly base](../dockerfile/rocm10.2-nightly-base.dockerfile):
  the all-device SDK and matched PyTorch family, fixed Triton, coherent generic
  A0 hipBLASLt device libraries, and ahead-of-time RCCL translation.
* [SuperBench overlay](../dockerfile/rocm10.2-nightly.dockerfile):
  SuperBench, native clients, DOCA/OpenMPI userspace, RVS/TransferBench, TE,
  AITER, Megatron and fixed FlyDSL. Host kernel modules are not installed.

The base dependency lock is in
[requirements-base.lock](../dockerfile/rocm10.2-nightly/requirements-base.lock).
The signed, SHA-256-pinned RVS payload uses the existing nightly SDK libraries;
its shared-library dependencies are checked explicitly. The nightly native APT
index does not provide a Release signature, so the overlay neither disables
signature verification nor installs a second ROCm runtime. The ROCm client
archive is SHA-256 checked and its source manifest is compared with the SDK
before use.

### Necessary A0 repairs

1. The SDK compiler contains the LLVM unsigned-offset correction, but the
   vendor Triton wheel still fails the regression test. Triton's own LLVM
   `5f07f818...` is rebuilt with upstream fix
   `3b2c3b02c0374cba15e30739b71780ad820e6485`.
2. FlyDSL has another independent LLVM, `e2a39f50...`; its build carries the same
   correction. Fixing only the SDK compiler is insufficient.
3. The nightly lacks A0 hipBLASLt master mappings. The base builds the matching
   generic `gfx1250-strict` logic with the supported `gfx1250v0` build alias,
   including mixed E4M3/E5M2 types. It does not relabel B0 mappings.
4. Cold RCCL startup can spend several minutes in CPU-side ROCJitsu translation.
   The packaged `rocjitsu-pretranslate.py` tool builds trusted, read-only AOT
   entries once. Matching translator binaries are checked before exposing the
   cache in both SDK loader locations. Translation remains enabled.
5. TE's basic A0 GEMM library does not provide fused FP32 weight/bias gradients.
   An [opt-in patch](../third_party/TransformerEngine/rocm_fp32_bgrad.patch)
   computes that FP32 bias gradient separately without reducing precision.

TE FlashAttention, CK fused attention and AOTriton fused attention are disabled
at build and runtime. The profile enables `NVTE_ROCM_UNFUSED_FP32_BGRAD=1`.
Ordinary PyTorch GEMMs use the explicit rocBLAS profile; TE FP8 and native
hipBLASLt calls retain their own backend.

AITER performs native GPU discovery/JIT work when imported. CPU image builds
check its installed distribution and dependencies, not a simulated GPU import;
AITER and Megatron runtime imports are checked on the GPU node.

## Build from source

Run from a checkout of the experimental branch. Only the native submodules
used by this overlay are required:

```bash
git submodule update --init third_party/fio third_party/perftest

docker buildx build --load \
  -f dockerfile/rocm10.2-nightly-base.dockerfile \
  -t sb-rocm102-base:20261018 .

docker buildx build --load \
  -f dockerfile/rocm10.2-nightly.dockerfile \
  --build-arg BASE_IMAGE=sb-rocm102-base:20261018 \
  --build-arg SB_SOURCE_REVISION="$(git rev-parse HEAD)" \
  --build-arg AMDGPU_TARGETS=gfx1250 \
  -t sb-rocm102-smoke:20261018 .
```

LLVM source builds need substantial CPU, memory and disk space. Defaults build
all compiler prerequisites from pinned sources. `SDK_IMAGE`,
`TRITON_LLVM_IMAGE`, and `FLYDSL_WHEEL_IMAGE` are optional local-cache overrides
for already-qualified artifacts, not required private build dependencies.

Use the published immutable image digests for exact run reproduction. Source
recipes also record package/source provenance, but are not a claim of
bit-for-bit reproducibility across different system toolchains.

## Run the smoke profile

Run the host-side helper on an idle, four-GPU MI455X node with Docker, Python,
PyYAML and permission to read the kernel journal and GPU process holders:

```bash
IMAGE_ID="$(docker image inspect --format '{{.Id}}' sb-rocm102-smoke:20261018)"
python3 dockerfile/rocm10.2-nightly/run_smoke.py \
  --image "${IMAGE_ID}" \
  --output-dir outputs/mi455x-rocm102-smoke
```

For an ACR image, first authenticate with `az acr login --name mi455xacr`, pull
the published `repository@sha256:...` reference, and pass that reference as
`--image`. Floating tags are rejected by the runner.

The tested host baseline is Linux `6.17.0-42-generic` with the eight-backport
amdgpu module reporting srcversion `DD338B73DA3807A0184A715`. The image does not
change the host driver, firmware or boot configuration. A deliberately
different host baseline must be supplied explicitly with
`--expected-driver-srcversion`; that does not inherit the original host's
qualification.

The runner verifies that the host YAML matches the copy inside the image,
executes one case at a time, isolates local GPU processes, uses four workers for
distributed cases, and checks every rank's result. It uses explicit loopback
rendezvous and 8 GiB private shared memory. GPU-copy cases alone receive
`SYS_NICE`. Kernel faults or persistent blocked tasks stop further submissions.
The helper never reboots the machine automatically.

`--only <case> ...` runs an explicit subset into a new output directory. A
failed case, missing rank result, invalid metric or failed RCCL correctness
check is not converted into a pass.

## Minimum work and coverage

The [YAML](../superbench/config/amd_mi455x_rocm102_smoke.yaml) keeps all 54
previously selected case/precision combinations:

| Family | Warmup / measured work | Smoke size |
|---|---|---|
| Kernel launch | 0 / 1, interval 0 | One launch per timer mode |
| GPU copy | 0 / 1, correctness enabled | 4 KiB / 64 KiB, all six copy directions |
| rocBLAS / hipBLASLt | 0 / 1 | One 64x64x64 shape |
| Five RCCL collectives | 0 / 1, one validation pass | 16 KiB |
| Matmul / sharding | 0 / 1 per mode | 64x64x64; all sharding modes |
| Compute/communication overlap | 0 / 1 outer step per kernel | 32-sized dimensions, ratio 1 |
| Eager inference | 0 / 1 | One layer, 16-sized dimensions |
| Graph inference | 1 / 1 replay | One initialization warmup, then actual graph capture |
| Model cases | 0 / 1 training step | Small batches and inputs; stop/log checks every step |

The overlap implementation still performs **100 collectives per outer step**;
the YAML does not pretend that this internal loop is configurable.

Transformer smoke cases retain their preset widths/head counts but use one
block and sequence length 16. CNNs retain their named architectures but use
32x32 images and batch size 2 so final-layer BatchNorm remains valid. These are
smoke instances, not full-size-model performance results.

Three directly related harness fixes make these tests meaningful:

* Explicit rocBLAS iteration counts now reach the native client. Omitted
  counts retain the client's previous defaults.
* The PyTorch inference graph flag now captures/replays a graph rather than
  silently executing eager mode.
* GPT2's Hugging Face `Conv1D` projections are converted to TE Linear with the
  correct transposed weight layout. An unaligned layer no longer prevents
  conversion of later siblings.

Excluded: unsafe KFD HQD debugfs, host-NUMA/large-allocation characterization,
destructive storage, unqualified RDMA/fabric, CUDA-only backends, and separate
Megatron/DeepSpeed training jobs. The built-in monitor stays disabled because
of the Python 3.14 forkserver/RLock incompatibility; host-side health monitoring
remains active.
