ARG SDK_IMAGE=sb-rocm103-sdk:20261026-py314
ARG LLVM_IMAGE=superbench-serving-triton-llvm:5f07f818-3b2c3b02
FROM ${LLVM_IMAGE} AS llvm
FROM ${SDK_IMAGE} AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=24
COPY --from=llvm /opt/triton-llvm /opt/triton-llvm
RUN apt-get update && apt-get install -y --no-install-recommends libzstd-dev zlib1g-dev && apt-get clean && \
    python3 -m pip install cmake==4.0.3 ninja==1.13.2 pybind11==3.0.4 lit==18.1.8 \
        wheel==0.48.0 setuptools==79.0.1 setuptools-scm==10.3.4 packaging==26.3 && \
    git init /src/triton && git -C /src/triton remote add origin https://github.com/ROCm/triton.git && \
    git -C /src/triton fetch --depth 1 origin a3c5afb73ecd9e70046a983a8dd18fd676a30438 && \
    git -C /src/triton checkout --detach FETCH_HEAD && cd /src/triton && \
    LLVM_SYSPATH=/opt/triton-llvm TRITON_WHEEL_VERSION_SUFFIX=.rocm10.3.0a20261026 \
        TRITON_BUILD_PROTON=OFF TRITON_PARALLEL_LINK_JOBS=2 MAX_JOBS="${BUILD_JOBS}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    python3 -m pip wheel --no-build-isolation --no-deps -w /wheels ./python/triton_kernels

FROM ${SDK_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
COPY --from=builder /wheels /opt/fixed-triton-wheels
COPY dockerfile/rocm10.2-nightly/protect_core.py dockerfile/rocm10.2-nightly/pin_fixed_wheel.py \
     dockerfile/rocm10.2-nightly/triton_unsigned_offset.py \
     dockerfile/rocm10.2-serving/hiprtc_unsigned_offset.cpp \
     dockerfile/rocm10.2-serving/validate_translation_cache.py /opt/base-provenance/
COPY dockerfile/rocm10.3-nightly/validate.py /opt/base-provenance/check_base.py
COPY dockerfile/rocm10.3-nightly/constraints-base.txt /opt/base-provenance/constraints-base.txt
RUN python3 -m pip install --force-reinstall --no-deps /opt/fixed-triton-wheels/*.whl && \
    python3 /opt/base-provenance/protect_core.py && \
    python3 /opt/base-provenance/pin_fixed_wheel.py triton && \
    sha256sum "$(python3 -c 'from triton._C import libtriton;print(libtriton.__file__)')" \
        > /opt/base-provenance/fixed-triton-library.sha256 && \
    c++ -std=c++17 -D__HIP_PLATFORM_AMD__ -I"${ROCM_PATH}/include" \
        /opt/base-provenance/hiprtc_unsigned_offset.cpp -L"${ROCM_PATH}/lib" -lhiprtc \
        -o /opt/base-provenance/hiprtc_unsigned_offset && \
    python3 "${ROCM_PATH}/bin/rocjitsu-pretranslate.py" --target gfx1250 --fail-on-skipped \
        --bundler "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" \
        --report-dir /opt/base-provenance/aot-rccl "${ROCM_PATH}/.kpack/rccl_lib_gfx1250.kpack" && \
    mkdir -p "${ROCM_PATH}/../_rocm_sdk_core/share/rocjitsu/translations" && \
    cp -a "${ROCM_PATH}/share/rocjitsu/translations/." "${ROCM_PATH}/../_rocm_sdk_core/share/rocjitsu/translations/" && \
    python3 /opt/base-provenance/check_base.py --record-assets && python3 -m pip check
ENV PIP_CONSTRAINT=/opt/base-provenance/core-constraints.txt
ARG SB_SOURCE_REVISION=unknown
LABEL com.superbench.source.commit="${SB_SOURCE_REVISION}" \
    com.superbench.rocm.version="10.3.0a20261026" \
    com.superbench.therock.commit="b6b24b2b8b40aeb6ddd1262283cc1c87442e4f50" \
    com.superbench.rocm-systems.commit="39fd0ea559a5970e42afd87c30f275744e00b8d1" \
    com.superbench.rocm-libraries.commit="565c4c748eaf0b1198bf5912198cc1bc531ecb18" \
    com.superbench.llvm.commit="2792f8ceeb5aa6e188c1ce2b735923b2406a8e59" \
    com.superbench.hipblaslt.profile="vendor-gfx1250v0" \
    com.superbench.triton.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485" \
    com.superbench.gpu-validation="NOT_RUN: CPU build and compiler checks only"
