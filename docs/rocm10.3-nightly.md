# October 26 ROCm 10.3 nightly refresh

This refresh preserves the separately qualified October 15/18 recipes and image
digests. It builds new **experimental, CPU-validated** base, SuperBench, vLLM and
SGLang images. Earlier GPU qualification does not automatically transfer to
these rebuilt images.

## Published images: October 27, 2026

All four were built on `hbv3-build`, CPU-validated, pushed to
`mi455xacr.azurecr.io`, and pulled back by digest to verify identity.
The common tag is `10.3.0a20261026-mi455x-a0-experimental`.
The previous qualified tags were not overwritten.

| Image repository | Immutable digest |
|---|---|
| `rocm-pytorch-base` | `sha256:c58c92d66616c1ced4999964a809504c6ff5c69df06bcc9cc9bc2a3d72ba86a0` |
| `superbench-rocm-pytorch` | `sha256:a7638a75932c9d8b3771fba355a73193736e03df99c42433ca4617d15ff40673` |
| `superbench-vllm` | `sha256:3ae8846dc34c3936ebf9f8ef6ff06d6a99c521e1bbb2f1ed2f54f5015342ded8` |
| `superbench-sglang` | `sha256:ed5ce2c32d20b86689cb84c29e08a91db6cfa969aa4a0475735396cfddd4c35e` |

For example:

```bash
docker pull mi455xacr.azurecr.io/superbench-rocm-pytorch:10.3.0a20261026-mi455x-a0-experimental
```

The image source is `32a127a7b0f012580095830ac81323b2adceb6d7` for the base
and serving images, and `f2e35e22c474c096a1749bb0fc2e46a6b2f89820` for
SuperBench. The latter adds the missing SCM build plugin and an early AITER
version assertion; it does not change the framework/SDK source pins.
Both revisions are on
`experiments/mi455x-rocm10.2-nightly-smoke-20261018` (the existing branch name
is retained despite the SDK moving to 10.3).

**GPU qualification: NOT RUN for all four new images.** The October 18/20
hardware results apply to the older images, not these digests.

## What the newer nightly incorporates

The selected complete set is `10.3.0a20261026`, with PyTorch
`2.13.0+rocm10.3.0a20261026` and matched torchvision/torchaudio/Apex wheels.
Python 3.14 is retained for SuperBench and Python 3.12 for serving.
Torch 2.15 is also published, but a framework-version jump is not necessary
to obtain these runtime fixes.

The actual SDK manifest pins:

| Component | Revision |
|---|---|
| TheRock | `b6b24b2b8b40aeb6ddd1262283cc1c87442e4f50` |
| rocm-systems | `39fd0ea559a5970e42afd87c30f275744e00b8d1` |
| rocm-libraries | `565c4c748eaf0b1198bf5912198cc1bc531ecb18` |
| SDK LLVM | `2792f8ceeb5aa6e188c1ce2b735923b2406a8e59` |
| Triton | `a3c5afb73ecd9e70046a983a8dd18fd676a30438` |
| AITER (source-built) | `2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0` |

The published gfx1250 wheel now includes the hipBLASLt and hipSPARSELt
`gfx1250v0` master tables and mappings. This incorporates
[ROCm/rocm-systems#13780](https://github.com/ROCm/rocm-systems/pull/13780);
we no longer replace the vendor library with our old generic-only build.
Both SDK views are verified and their vendor assets are fingerprinted.

Other included changes:

| Component | Included fix |
|---|---|
| RCCL | [ROCm/rocm-systems#13590](https://github.com/ROCm/rocm-systems/pull/13590): asynchronous error-query ownership race |
| TensileLite | [ROCm/rocm-libraries#13913](https://github.com/ROCm/rocm-libraries/pull/13913): GSU atomic scope and TDM synchronization |
| CK | [ROCm/rocm-libraries#13808](https://github.com/ROCm/rocm-libraries/pull/13808): TDM LDS-padding correction |
| TensileLite | [ROCm/rocm-libraries#13266](https://github.com/ROCm/rocm-libraries/pull/13266): exact integer tile division |
| Triton | [ROCm/triton#979](https://github.com/ROCm/triton/pull/979): expert scheduling defaults off |
| AITER | [ROCm/aiter#6629](https://github.com/ROCm/aiter/pull/6629): large-output GEMM store addressing |
| AITER | [ROCm/aiter#6663](https://github.com/ROCm/aiter/pull/6663): grouped-MoE tail and bias addressing |
| AITER | [ROCm/aiter#6348](https://github.com/ROCm/aiter/pull/6348): RMSNorm tail/reduction corrections |

AITER is updated separately in the application images: a newer SDK alone
does not update that package. Inclusion of a fix is not proof that a
particular ADO reproducer now passes.

The matched Triton now defaults expert scheduling off, but still uses its
independent LLVM `5f07f818...`; the recipe rebuilds it with the unsigned-address
fix rather than assuming the SDK LLVM update repairs it. The actual vendor
wheel failed our `read_u32` code-generation regression during this refresh.
Existing corrected
FlyDSL 0.4.0 compiler sources are retained for compatibility. The AITER package
version is explicitly stamped `0.1.26.dev0+g2cb3d1758`; the source hash, not
a release-version claim, identifies this build.

Both `TRITON_HIP_USE_EXPERT_SCHEDULING=0` and `TRITON_HIP_USE_ASYNC_COPY=0`
remain explicit. The [serving launch profiles](rocm10.2-serving.md) remain
available through `sb-serve`; the vLLM Triton attention and SGLang compile
rocBLAS workarounds are not removed by this refresh.

## Build all four images

Run from a checkout on an x86-64 Linux Docker/Buildx CPU builder:

```bash
git submodule update --init third_party/fio third_party/perftest
python3 dockerfile/rocm10.3-nightly/build.py \
  --source-revision "$(git rev-parse HEAD)" --jobs 24
```

The complete [build plan](../dockerfile/rocm10.3-nightly/build.py) creates all
compiler prerequisites by default. The two existing application Dockerfiles
are parameterized to accept refreshed base/constraints while retaining their
historical defaults. The new
[SDK](../dockerfile/rocm10.3-nightly/sdk.dockerfile) and
[base](../dockerfile/rocm10.3-nightly/base.dockerfile) recipes feed the shared
[full overlay](../dockerfile/rocm10.2-nightly.dockerfile); its older filename
does not select the SDK when the refresh driver supplies the new arguments.
The full overlay retains DOCA/OpenMPI userspace,
RVS/TransferBench, native clients, TE with fused attention disabled, AITER,
Megatron and repaired FlyDSL.

Optional compatible local compiler caches:

```bash
python3 dockerfile/rocm10.3-nightly/build.py \
  --source-revision "$(git rev-parse HEAD)" --jobs 24 \
  --triton-llvm-image superbench-triton-llvm:5f07f818-3b2c3b02-compact \
  --flydsl312-image superbench-flydsl:0.4.0-llvmfix
```

`--flydsl314-image` may reuse a **Python 3.14 FlyDSL 0.4.0** artifact; do not
substitute the old 0.3.4.1 wheel or a Python 3.12 wheel. Building both LLVM/MLIR
trees from scratch needs considerable disk/RAM. `--images base superbench`
or `--images vllm sglang` selects an explicit subset with prerequisites.
Publication is separate from building; the script never overwrites ACR tags.

This is pinned-source/package reproducibility, not bit-for-bit reproducibility:
Ubuntu packages and external bootstrapping infrastructure can change.
Missing nightly artifacts fail rather than silently selecting another date.

## Validation and publication scope

Required CPU checks include actual SDK provenance, vendor A0 master tables
and both-layout file consistency, LLVM/Triton/HIPRTC/FlyDSL unsigned-address
regressions, packaged Wave32 translation-cache checks, native CPU tensor/NMS
checks, package consistency, and installed serving profile/module checks.
GPU-dependent AITER imports and actual inference require separate hardware
qualification. Do not interpret a successful CPU image build as a new
54-case SuperBench GPU pass or resolution of MODE2/fabric/graph-capture bugs.

No host driver, firmware or kernel changes are part of these recipes.

### Recorded CPU results

| Check | Result |
|---|---|
| SDK source/package pins and `python3 -m pip check`, all four images | PASS |
| Vendor A0 assets, both SDK layouts | PASS: 568 hipBLASLt/hipSPARSELt files |
| Packaged RCCL translation cache | PASS: 114 objects, 395 Wave32 descriptors |
| SDK LLVM, both instruction selectors | PASS |
| Rebuilt Triton unsigned load, byte load, store and atomic code generation | PASS |
| HIPRTC unsigned-address code generation | PASS |
| Patched FlyDSL MLIR/ROCDL code generation, all three application images | PASS |
| TE CPU import, pinned AITER/Megatron metadata, full-overlay CK JIT assets | PASS |
| Serving imports, 12 patched vLLM module hashes and explicit launch profiles | PASS |
| Focused unit tests, local checkout and final SuperBench image | 44 PASS |
| Refresh Python helpers/tests, flake8 | PASS |

The per-role compiler/package checks can be repeated without GPUs:

```bash
docker run --rm --network=none sb-rocm103-base:20261026-py314 \
  python3 /opt/base-provenance/check_base.py --role base
docker run --rm --network=none sb-rocm103-superbench:20261026 \
  python3 /opt/base-provenance/check_base.py --role superbench
docker run --rm --network=none sb-rocm103-vllm:20261026 \
  python3 /opt/base-provenance/check_base.py --role vllm
docker run --rm --network=none sb-rocm103-sglang:20261026 \
  python3 /opt/base-provenance/check_base.py --role sglang
```

All four commands passed on the builder. Substitute the corresponding ACR
image reference to check a pulled image. Successful builds retain their
provenance under `/opt/base-provenance`; rerunning the validator in a
disposable container prints a fresh report.

The focused source tests and lint commands passed:

```bash
python3 -B -m unittest -q tests.test_rocm103_nightly tests.test_rocm102_serving \
  tests.test_rocm102_dockerfiles tests.test_rocm102_smoke_runner
flake8 dockerfile/rocm10.3-nightly/*.py tests/test_rocm103_nightly.py
```

To repeat all 44 tests inside the built SuperBench image, mount the
source-only `.dockerignore` test fixture (Docker does not copy that file):

```bash
docker run --rm --network=none \
  --mount "type=bind,src=$(pwd)/.dockerignore,dst=/opt/superbench/.dockerignore,readonly" \
  sb-rocm103-superbench:20261026 \
  python3 -B -m unittest -q tests.test_rocm103_nightly tests.test_rocm102_serving \
  tests.test_rocm102_dockerfiles tests.test_rocm102_smoke_runner
```
