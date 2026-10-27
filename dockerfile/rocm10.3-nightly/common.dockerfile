ARG BASE_IMAGE=sb-rocm103-base:20261026-py312
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=24
COPY dockerfile/rocm10.3-nightly/constraints-base.txt /opt/base-provenance/constraints-base.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnuma-dev libssl-dev pkg-config protobuf-compiler libibverbs-dev librdmacm-dev \
        libaio-dev libcurl4-openssl-dev && apt-get clean && \
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.95.0 && rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:${PATH} CARGO_BUILD_JOBS=16 MAX_JOBS=24 \
    PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 AITER_USE_SYSTEM_TRITON=1 \
    NCCL_MIN_NCHANNELS=64 TRITON_CACHE_DIR=/root/.cache/triton-rocm103-20261026 \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor-rocm103-20261026
RUN python3 -m pip install -c /opt/base-provenance/constraints-base.txt \
        cmake==3.31.10 setuptools==79.0.1 setuptools-scm setuptools-rust scikit-build-core \
        pybind11==3.0.4 ninja wheel packaging && \
    git init /opt/aiter && git -C /opt/aiter remote add origin https://github.com/ROCm/aiter.git && \
    git -C /opt/aiter fetch --depth 1 origin 2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0 && \
    git -C /opt/aiter checkout --detach FETCH_HEAD && \
    git -C /opt/aiter submodule update --init --recursive --depth 1 && \
    python3 -m pip install -c /opt/base-provenance/constraints-base.txt -r /opt/aiter/requirements.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AITER=0.1.26.dev0+g2cb3d1758 \
        PREBUILD_KERNELS=0 ENABLE_CK=0 MAX_JOBS="${BUILD_JOBS}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /opt/aiter-wheels /opt/aiter && \
    python3 -m pip install --no-deps /opt/aiter-wheels/*.whl && python3 -m pip check && \
    test "$(git -C /opt/aiter/3rdparty/composable_kernel rev-parse HEAD)" = 96afae1d54fc79c26857de7e1683b0dde1ff3c6f && \
    mkdir -p /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel && \
    cp -a /opt/aiter/3rdparty/composable_kernel/. \
        /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel/
ENV CPATH=/opt/aiter/3rdparty/composable_kernel/include:/opt/aiter/3rdparty/composable_kernel/library/include
LABEL com.superbench.aiter.commit="2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0"
