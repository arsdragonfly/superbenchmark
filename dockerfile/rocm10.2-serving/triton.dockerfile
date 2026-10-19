ARG BASE_IMAGE=superbench-serving-build:10.2.0a20261015-py3.12
ARG LLVM_IMAGE=superbench-serving-triton-llvm:5f07f818-3b2c3b02
FROM ${LLVM_IMAGE} AS llvm
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG TRITON_COMMIT=66b3f9ff575eb8db6adec658bbf0badd9a8d2e25
ARG BUILD_JOBS=24
COPY --from=llvm /opt/triton-llvm /opt/triton-llvm
RUN apt-get update && apt-get install -y --no-install-recommends libzstd-dev zlib1g-dev && \
    apt-get clean && \
    python3 -m pip install cmake==4.0.3 ninja==1.13.0 pybind11==3.0.4 lit==18.1.8 wheel==0.48.0 \
        setuptools==79.0.1 setuptools-scm==10.3.4 packaging==26.3 && \
    git init /src/triton && \
    git -C /src/triton remote add origin https://github.com/ROCm/triton.git && \
    git -C /src/triton fetch --depth 1 origin "${TRITON_COMMIT}" && \
    git -C /src/triton checkout --detach FETCH_HEAD && \
    cd /src/triton && \
    LLVM_SYSPATH=/opt/triton-llvm TRITON_WHEEL_VERSION_SUFFIX=.rocm10.2.0a20261015 \
    TRITON_BUILD_PROTON=OFF TRITON_PARALLEL_LINK_JOBS=2 MAX_JOBS="${BUILD_JOBS}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . && \
    python3 -m pip wheel --no-build-isolation --no-deps -w /wheels ./python/triton_kernels

FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
COPY --from=builder /wheels /opt/fixed-triton-wheels
COPY dockerfile/rocm10.2-nightly/pin_fixed_wheel.py \
     dockerfile/rocm10.2-nightly/triton_unsigned_offset.py /opt/base-provenance/
RUN python3 -m pip install --force-reinstall --no-deps /opt/fixed-triton-wheels/*.whl && \
    python3 -m pip check && \
    sha256sum "$(python3 -c 'from triton._C import libtriton; print(libtriton.__file__)')" \
        > /opt/base-provenance/fixed-triton-library.sha256 && \
    python3 /opt/base-provenance/pin_fixed_wheel.py triton
ENV PIP_CONSTRAINT=/opt/base-provenance/core-constraints.txt
RUN python3 /opt/base-provenance/validate.py base
LABEL com.superbench.triton.llvm.commit="5f07f818b51b786b0a87b4e514882600ecba112f" \
    com.superbench.triton.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485"
