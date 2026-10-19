ARG BASE_IMAGE=superbench-serving-base:10.2.0a20261015-py3.12-llvmfix
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG SGLANG_COMMIT=478b0b00627bfba44e273aaad6ebefd8decb72d8
ARG BUILD_JOBS=24
ENV SGLANG_USE_AITER=1 \
    SGLANG_USE_AITER_UNIFIED_ATTN=1
COPY dockerfile/rocm10.2-serving/constraints-sglang.txt /opt/base-provenance/constraints-app.txt
RUN git init /opt/sglang && \
    git -C /opt/sglang remote add origin https://github.com/sgl-project/sglang.git && \
    git -C /opt/sglang fetch --depth 1 origin "${SGLANG_COMMIT}" && \
    git -C /opt/sglang checkout --detach FETCH_HEAD && \
    git -C /opt/sglang submodule update --init --recursive --depth 1
RUN --mount=type=cache,id=sb-serving-sglang-478b0b-20261015,target=/opt/sglang/python/sglang/kernels/aot/build \
    cd /opt/sglang/python/sglang/kernels/aot && \
    cp pyproject_rocm.toml pyproject.toml && \
    AMDGPU_TARGET=gfx1250 MAX_JOBS="${BUILD_JOBS}" \
        python3 setup_rocm.py bdist_wheel -d /opt/sglang-kernel-wheels && \
    python3 -m pip install --no-deps /opt/sglang-kernel-wheels/*.whl
RUN --mount=type=cache,id=sb-serving-sglang-478b0b-cargo,target=/opt/sglang/rust/target \
    cd /opt/sglang && cp python/pyproject_other.toml python/pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.5.18.dev20260829+g478b0b0062 \
    MAX_JOBS="${BUILD_JOBS}" CARGO_BUILD_JOBS=16 \
        python3 -m pip wheel --no-build-isolation --no-deps -w /opt/sglang-wheels ./python && \
    python3 -m pip install -c /opt/base-provenance/constraints-app.txt /opt/sglang-wheels/sglang-*.whl && \
    python3 -m pip install -c /opt/base-provenance/constraints-app.txt \
        'sglang[runtime_common,tracing]==0.5.18.dev20260829+g478b0b0062' && \
    python3 -m pip check && sha256sum --check /opt/base-provenance/fixed-triton-installed.sha256 && \
    python3 -m pip freeze > /opt/base-provenance/sglang-requirements.txt
RUN python3 -c 'import sglang, sgl_kernel; print("SGLang imported")' && \
    python3 -m sglang.launch_server --help
LABEL com.superbench.sglang.commit="478b0b00627bfba44e273aaad6ebefd8decb72d8" \
    com.superbench.profile="LLM serving with AITER and Triton; no Petit/Wave/diffusion"
EXPOSE 30000
CMD ["/bin/bash"]
