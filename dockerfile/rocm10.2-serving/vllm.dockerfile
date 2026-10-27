ARG BASE_IMAGE=superbench-serving-base:10.2.0a20261015-py3.12-llvmfix
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG VLLM_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6
ARG BUILD_JOBS=24
ARG CONSTRAINTS_FILE=dockerfile/rocm10.2-serving/constraints-vllm.txt
ARG VLLM_VERSION=0.31.0+rocm10.2.0a20261015.llvmfix
ENV VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_PRECOMPILED=0 \
    VLLM_USE_PRECOMPILED_RUST=0
COPY ${CONSTRAINTS_FILE} /opt/base-provenance/constraints-app.txt
RUN git init /opt/vllm && \
    git -C /opt/vllm remote add origin https://github.com/vllm-project/vllm.git && \
    git -C /opt/vllm fetch --depth 1 origin "${VLLM_COMMIT}" && \
    git -C /opt/vllm checkout --detach FETCH_HEAD && \
    git -C /opt/vllm submodule update --init --recursive --depth 1
WORKDIR /opt/vllm
COPY dockerfile/rocm10.2-serving/vllm-no-tilelang.patch /opt/base-provenance/vllm-no-tilelang.patch
COPY dockerfile/rocm10.2-serving/vllm-1359.patch dockerfile/rocm10.2-serving/vllm-1354.patch /opt/base-provenance/amd-vllm/
RUN git apply --check /opt/base-provenance/vllm-no-tilelang.patch && \
    git apply /opt/base-provenance/vllm-no-tilelang.patch && \
    echo 'a38b087a2343f87761e50c8d3787d6cef0463fdb2a96079467684989868c22c8  /opt/base-provenance/amd-vllm/vllm-1359.patch' | sha256sum -c && \
    git apply --check /opt/base-provenance/amd-vllm/vllm-1359.patch && \
    git apply /opt/base-provenance/amd-vllm/vllm-1359.patch && \
    git apply --check /opt/base-provenance/amd-vllm/vllm-1354.patch && \
    git apply /opt/base-provenance/amd-vllm/vllm-1354.patch && \
    python3 tools/use_existing_torch.py --prefix && \
    python3 -m pip install -c /opt/base-provenance/constraints-app.txt -r requirements/rocm.txt
RUN --mount=type=cache,id=sb-serving-vllm-db9527a-20261015,target=/opt/vllm/build \
    --mount=type=cache,id=sb-serving-vllm-db9527a-cargo,target=/opt/vllm/rust/target \
    CMAKE_PREFIX_PATH="${ROCM_PATH}" CXX="${ROCM_PATH}/lib/llvm/bin/clang++" \
    CC="${ROCM_PATH}/lib/llvm/bin/clang" MAX_JOBS="${BUILD_JOBS}" CARGO_BUILD_JOBS=16 \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /opt/vllm-wheels . && \
    python3 -m pip install --no-deps /opt/vllm-wheels/*.whl && python3 -m pip check && \
    sha256sum --check /opt/base-provenance/fixed-triton-installed.sha256 && \
    python3 -m pip freeze > /opt/base-provenance/vllm-requirements.txt
WORKDIR /
COPY dockerfile/rocm10.2-serving/record_vllm_patches.py /opt/base-provenance/amd-vllm/
RUN python3 /opt/base-provenance/amd-vllm/record_vllm_patches.py
RUN python3 -c 'import vllm, vllm._C; print(vllm.__version__)'
LABEL com.superbench.vllm.commit="db9527a46873454610df6dbedf79a36d6bf1a7f6" \
    com.superbench.amd-vllm-prs="ROCm/vllm#1359,ROCm/vllm#1354 (existing K-alignment guard preserved)" \
    com.superbench.profile="gfx1250 LLM serving; unqualified TileLang compiler excluded"
EXPOSE 8000
CMD ["/bin/bash"]
