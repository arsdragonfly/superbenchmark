ARG BASE_IMAGE=superbench-serving-base:10.2.0a20261015-py3.12-llvmfix
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG VLLM_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6
ARG BUILD_JOBS=24
ENV VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_PRECOMPILED=0 \
    VLLM_USE_PRECOMPILED_RUST=0
COPY dockerfile/rocm10.2-serving/constraints-vllm.txt /opt/base-provenance/constraints-app.txt
RUN git init /opt/vllm && \
    git -C /opt/vllm remote add origin https://github.com/vllm-project/vllm.git && \
    git -C /opt/vllm fetch --depth 1 origin "${VLLM_COMMIT}" && \
    git -C /opt/vllm checkout --detach FETCH_HEAD && \
    git -C /opt/vllm submodule update --init --recursive --depth 1
WORKDIR /opt/vllm
COPY dockerfile/rocm10.2-serving/vllm-no-tilelang.patch /opt/base-provenance/vllm-no-tilelang.patch
RUN git apply --check /opt/base-provenance/vllm-no-tilelang.patch && \
    git apply /opt/base-provenance/vllm-no-tilelang.patch && \
    python3 tools/use_existing_torch.py --prefix && \
    python3 -m pip install -c /opt/base-provenance/constraints-app.txt -r requirements/rocm.txt
RUN --mount=type=cache,id=sb-serving-vllm-db9527a-20261015,target=/opt/vllm/build \
    --mount=type=cache,id=sb-serving-vllm-db9527a-cargo,target=/opt/vllm/rust/target \
    CMAKE_PREFIX_PATH="${ROCM_PATH}" CXX="${ROCM_PATH}/lib/llvm/bin/clang++" \
    CC="${ROCM_PATH}/lib/llvm/bin/clang" MAX_JOBS="${BUILD_JOBS}" CARGO_BUILD_JOBS=16 \
    VLLM_VERSION_OVERRIDE=0.31.0+rocm10.2.0a20261015.llvmfix \
        python3 -m pip wheel --no-build-isolation --no-deps -w /opt/vllm-wheels . && \
    python3 -m pip install --no-deps /opt/vllm-wheels/*.whl && python3 -m pip check && \
    sha256sum --check /opt/base-provenance/fixed-triton-installed.sha256 && \
    python3 -m pip freeze > /opt/base-provenance/vllm-requirements.txt
WORKDIR /
RUN python3 -c 'import vllm, vllm._C; print(vllm.__version__)'
LABEL com.superbench.vllm.commit="db9527a46873454610df6dbedf79a36d6bf1a7f6" \
    com.superbench.profile="gfx1250 LLM serving; unqualified TileLang compiler excluded"
EXPOSE 8000
CMD ["/bin/bash"]
