ARG BASE_IMAGE=superbench-serving-base:10.2.0a20261015-py3.12-llvmfix
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=32
ARG FLYDSL_COMMIT=1f3511677662afc58bcf141dec812ccf05334fd8
ARG LLVM_COMMIT=e2a39f504fee836e4def9581bed817ecc327b9dc
RUN apt-get update && apt-get install -y --no-install-recommends \
        lld patchelf libzstd-dev zlib1g-dev && apt-get clean && \
    python3 -m pip install -c /opt/base-provenance/constraints-base.txt \
        cmake==3.31.10 nanobind==2.12.0 pybind11 numpy && \
    git init /src/FlyDSL && \
    git -C /src/FlyDSL remote add origin https://github.com/ROCm/FlyDSL.git && \
    git -C /src/FlyDSL fetch --depth 1 origin "${FLYDSL_COMMIT}" && \
    git -C /src/FlyDSL checkout --detach FETCH_HEAD && \
    git -C /src/FlyDSL submodule update --init --recursive --depth 1 && \
    git init /src/flydsl-llvm && \
    git -C /src/flydsl-llvm remote add origin https://github.com/llvm/llvm-project.git && \
    git -C /src/flydsl-llvm fetch --depth 1 --filter=blob:none origin "${LLVM_COMMIT}" && \
    git -C /src/flydsl-llvm checkout --detach FETCH_HEAD
COPY dockerfile/rocm10.2-nightly/llvm-unsigned-offset.patch /src/llvm-fix.patch
RUN echo '5653cbd792aa071647e307b39e932d8fbd2e39bf9d7aa9e4115b3ebce508bf07  /src/llvm-fix.patch' | sha256sum -c && \
    git -C /src/flydsl-llvm apply --check /src/FlyDSL/thirdparty/llvm-extensions/rocdl-lld-argv0.patch && \
    git -C /src/flydsl-llvm apply /src/FlyDSL/thirdparty/llvm-extensions/rocdl-lld-argv0.patch && \
    git -C /src/flydsl-llvm apply --check /src/llvm-fix.patch && \
    git -C /src/flydsl-llvm apply /src/llvm-fix.patch && \
    cmake -S /src/flydsl-llvm/llvm -B /build/flydsl-llvm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
        -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
        -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_ENABLE_LLD=ON -DLLVM_PARALLEL_LINK_JOBS=2 \
        -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF \
        -DBUILD_SHARED_LIBS=OFF -DMLIR_INCLUDE_TESTS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
        -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
        -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)" \
        -DHIP_PLATFORM=amd -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_INSTALL_PREFIX=/opt/flydsl-llvm && \
    cmake --build /build/flydsl-llvm --parallel "${BUILD_JOBS}" && cmake --install /build/flydsl-llvm
COPY dockerfile/rocm10.2-nightly/unsigned-offset.ll /opt/provenance/unsigned-offset.ll
RUN for selector in 0 1; do \
        /opt/flydsl-llvm/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
            -O3 -global-isel="${selector}" -global-isel-abort=1 \
            /opt/provenance/unsigned-offset.ll -o "/opt/provenance/flydsl-llvm-${selector}.s"; \
        grep -q 'v_lshl_add_u64' "/opt/provenance/flydsl-llvm-${selector}.s"; \
    done && \
    cmake -S /src/FlyDSL -B /src/FlyDSL/build-fly -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=/opt/flydsl-llvm/lib/cmake/mlir \
        -DLLVM_DIR=/opt/flydsl-llvm/lib/cmake/llvm -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
        -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
        -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)" -DHIP_PLATFORM=amd && \
    cmake --build /src/FlyDSL/build-fly --parallel "${BUILD_JOBS}" && \
    cd /src/FlyDSL && \
    MLIR_PATH=/opt/flydsl-llvm FLY_REBUILD=0 FLYDSL_PACKAGE_VERSION_OVERRIDE=0.4.0 \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels .

FROM scratch
COPY --from=builder /wheels /wheels
COPY --from=builder /opt/provenance /opt/provenance
LABEL com.superbench.flydsl.commit="1f3511677662afc58bcf141dec812ccf05334fd8" \
    com.superbench.flydsl.llvm.commit="e2a39f504fee836e4def9581bed817ecc327b9dc" \
    com.superbench.flydsl.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485"
