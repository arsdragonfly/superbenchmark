ARG BASE_IMAGE=superbench-serving-sdk:10.2.0a20261015-py3.12
ARG A0_IMAGE=superbench-serving-a0-library:10.2.0a20261015
FROM ${A0_IMAGE} AS a0
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG AITER_COMMIT=4eb7dcac7e5f5abf566246534809644f73f07b49
ARG BUILD_JOBS=24
RUN sha256sum /opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/libhipblaslt.so.1 \
        /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhipblaslt.so.1 \
        > /opt/base-provenance/hipblaslt-host.sha256
COPY --from=a0 /gfx1250v0/ /opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1250v0/
COPY --from=a0 /gfx1250v0/ /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/hipblaslt/library/gfx1250v0/
COPY --from=a0 /provenance/ /opt/base-provenance/a0/
RUN sha256sum --check /opt/base-provenance/hipblaslt-host.sha256
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnuma-dev libssl-dev pkg-config protobuf-compiler libibverbs-dev \
        librdmacm-dev libaio-dev libcurl4-openssl-dev && apt-get clean
RUN python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "amdsmi.pth").write_text("_rocm_sdk_core/share/amd_smi\n")' && \
    python3 -c 'import amdsmi; print(amdsmi.__file__)'
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.95.0 && rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:${PATH} \
    CARGO_BUILD_JOBS=16 \
    MAX_JOBS=24 \
    PYTORCH_ROCM_ARCH=gfx1250 \
    GPU_ARCHS=gfx1250 \
    AITER_USE_SYSTEM_TRITON=1 \
    TRITON_HIP_USE_ASYNC_COPY=0 \
    TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
    NCCL_MIN_NCHANNELS=64 \
    TRITON_CACHE_DIR=/root/.cache/triton-rocm102-20261015-a0-retest \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor-rocm102-20261015-a0-retest \
    MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-rocm102-20261015-llvmfix
RUN python3 -m pip freeze | \
        grep -Ei '^(torch|torchvision|torchaudio|triton|apex|rocm|rocm[-_][^=]+|amd[-_](torch|torchvision)[-_][^=]+)==' \
        > /opt/base-provenance/core-constraints.txt && \
    python3 -m pip install -c /opt/base-provenance/constraints-base.txt \
        cmake==3.31.10 setuptools==79.0.1 setuptools-scm setuptools-rust \
        scikit-build-core pybind11==3.0.4 ninja wheel packaging && \
    git init /opt/aiter && \
    git -C /opt/aiter remote add origin https://github.com/ROCm/aiter.git && \
    git -C /opt/aiter fetch --depth 256 --tags origin "${AITER_COMMIT}" && \
    git -C /opt/aiter checkout --detach FETCH_HEAD && \
    git -C /opt/aiter submodule update --init --recursive --depth 1
RUN python3 -m pip install -c /opt/base-provenance/constraints-base.txt -r /opt/aiter/requirements.txt && \
    cd /opt/aiter && \
    PIP_CONSTRAINT=/opt/base-provenance/constraints-base.txt \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AITER=0.1.26.dev97+g4eb7dcac7 \
    PREBUILD_KERNELS=0 ENABLE_CK=0 MAX_JOBS="${BUILD_JOBS}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /opt/aiter-wheels . && \
    python3 -m pip install --no-deps /opt/aiter-wheels/*.whl && python3 -m pip check
# ENABLE_CK=0 avoids building unsupported kernels, but runtime JIT still needs
# the pinned CK headers and code generators (e.g. MoE routing and vision).
RUN test "$(git -C /opt/aiter/3rdparty/composable_kernel rev-parse HEAD)" = 96afae1d54fc79c26857de7e1683b0dde1ff3c6f && \
    test -f /opt/aiter/3rdparty/composable_kernel/include/ck_tile/core.hpp && \
    test -f /opt/aiter/3rdparty/composable_kernel/example/ck_tile/01_fmha/generate.py && \
    mkdir -p /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel && \
    cp -a /opt/aiter/3rdparty/composable_kernel/. \
        /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel/ && \
    git -C /opt/aiter/3rdparty/composable_kernel rev-parse HEAD > /opt/base-provenance/aiter-ck-source.txt
ENV CPATH=/opt/aiter/3rdparty/composable_kernel/include:/opt/aiter/3rdparty/composable_kernel/library/include
LABEL com.superbench.aiter.commit="4eb7dcac7e5f5abf566246534809644f73f07b49" \
    com.superbench.target="gfx1250" \
    com.superbench.a0-workarounds="TRITON_HIP_USE_ASYNC_COPY=0,TRITON_HIP_USE_EXPERT_SCHEDULING=0,NCCL_MIN_NCHANNELS=64" \
    com.superbench.hipblaslt.profile="gfx1250v0-generic-grid"
