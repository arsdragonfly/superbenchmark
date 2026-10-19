ARG BASE_IMAGE=superbench-serving-sdk:10.2.0a20261015-py3.12
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG AITER_COMMIT=4eb7dcac7e5f5abf566246534809644f73f07b49
ARG BUILD_JOBS=24
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
    NCCL_MIN_NCHANNELS=64 \
    TRITON_CACHE_DIR=/root/.cache/triton-rocm102-20261015-llvmfix \
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
LABEL com.superbench.aiter.commit="4eb7dcac7e5f5abf566246534809644f73f07b49" \
    com.superbench.target="gfx1250" \
    com.superbench.a0-workarounds="TRITON_HIP_USE_ASYNC_COPY=0,NCCL_MIN_NCHANNELS=64"
