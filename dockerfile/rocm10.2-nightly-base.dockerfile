ARG SDK_IMAGE=sdk-source
ARG TRITON_LLVM_IMAGE=llvm-source

FROM ubuntu:26.04@sha256:2260313b31c8c011cd2eebe728008efac1b3982be73eb71348ea2648d2c0e09b AS sdk-source
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl file git gpg kmod \
        libatomic1 libdw1t64 libelf1t64 libgomp1 libncurses6 libnuma1 \
        libssl3t64 libunwind8 perl python3 python3-dev python3-pip python3-venv \
    && apt-get clean \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 14); assert sys.version_info[:3] != (3, 14, 1)'
COPY dockerfile/rocm10.2-nightly/requirements-base.lock /opt/base-provenance/requirements-base.lock
RUN python3 -m pip install \
        --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
        --extra-index-url https://pypi.org/simple \
        --report /opt/base-provenance/pip-report.json \
        -r /opt/base-provenance/requirements-base.lock \
    && rocm-sdk init \
    && python3 -m pip check \
    && ROCM_PATH="$(rocm-sdk path --root)" \
    && printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}" > /etc/profile.d/rocm-sdk-env.sh \
    && printf '%s\n' \
        'export ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}"' \
        'export PATH="${ROCM_PATH}/bin:${PATH}"' \
        'export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"' \
        >> /etc/profile.d/rocm-sdk-env.sh \
    && python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "amdsmi.pth").write_text("_rocm_sdk_core/share/amd_smi\n")'
ENV BASH_ENV=/etc/profile.d/rocm-sdk-env.sh

# The optional image arguments reuse locally qualified build caches. Defaults
# build everything from the pinned sources below; no private cache is required.
FROM ${SDK_IMAGE} AS sdk

FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac AS llvm-source
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG LLVM_BUILD_JOBS=64
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git lld ninja-build python3 \
        python3-dev zlib1g-dev libzstd-dev && apt-get clean \
    && git init /src/llvm-project \
    && git -C /src/llvm-project remote add origin https://github.com/ROCm/llvm-project.git \
    && git -C /src/llvm-project fetch --depth 1 --filter=blob:none origin 5f07f818b51b786b0a87b4e514882600ecba112f \
    && git -C /src/llvm-project checkout --detach FETCH_HEAD
COPY dockerfile/rocm10.2-nightly/llvm-unsigned-offset.patch /src/llvm-fix.patch
RUN echo "4339f2c8cb6103d2e6d6471c22ad8c241a7284d47529c1fd3803505f2a4a48ea  /src/llvm-fix.patch" | sha256sum -c \
    && git -C /src/llvm-project apply --check --include='llvm/lib/Target/AMDGPU/*' /src/llvm-fix.patch \
    && git -C /src/llvm-project apply --include='llvm/lib/Target/AMDGPU/*' /src/llvm-fix.patch \
    && cmake -S /src/llvm-project/llvm -B /build/llvm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;AMDGPU' \
        -DLLVM_ENABLE_LLD=ON -DLLVM_PARALLEL_LINK_JOBS=2 \
        -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INSTALL_UTILS=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/triton-llvm \
    && cmake --build /build/llvm --parallel "${LLVM_BUILD_JOBS}" \
    && cmake --install /build/llvm
FROM ${TRITON_LLVM_IMAGE} AS llvm

FROM sdk AS triton-builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=32
COPY --from=llvm /opt/triton-llvm /opt/triton-llvm
RUN apt-get update && apt-get install -y --no-install-recommends libzstd-dev zlib1g-dev \
    && apt-get clean \
    && python3 -m pip install cmake==4.0.3 ninja==1.13.2 pybind11==3.0.4 lit wheel==0.48.0 \
        setuptools==79.0.1 setuptools-scm packaging \
    && git init /src/triton \
    && git -C /src/triton remote add origin https://github.com/ROCm/triton.git \
    && git -C /src/triton fetch --depth 1 origin 66b3f9ff575eb8db6adec658bbf0badd9a8d2e25 \
    && git -C /src/triton checkout --detach FETCH_HEAD \
    && cd /src/triton \
    && LLVM_SYSPATH=/opt/triton-llvm TRITON_WHEEL_VERSION_SUFFIX=.rocm10.2.0a20261018 \
        TRITON_BUILD_PROTON=OFF TRITON_PARALLEL_LINK_JOBS=2 MAX_JOBS="${BUILD_JOBS}" \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . \
    && python3 -m pip wheel --no-build-isolation --no-deps -w /wheels ./python/triton_kernels

FROM sdk AS a0-builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=16
ENV CMAKE_PREFIX_PATH=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel \
    CMAKE_BUILD_PARALLEL_LEVEL=16 \
    CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/opt/venv/bin/amdclang++" \
    TENSILE_DISABLE_HELPER_CACHE=1 SKBUILD_EDITABLE_REBUILD=false
RUN git init /src \
    && git -C /src remote add origin https://github.com/ROCm/rocm-libraries.git \
    && git -C /src sparse-checkout set projects/hipblaslt/tensilelite projects/hipblaslt/library/include \
        projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250-strict/gfx1250-strict \
        shared/origami shared/stinkytofu cmake \
    && git -C /src fetch --depth 1 --filter=blob:none origin 8a29d2e82b93f5065c7fe5630248d3689f0c3f06 \
    && git -C /src checkout --detach FETCH_HEAD \
    && python3 -m pip install cmake==4.0.3 ninja==1.13.2 msgpack joblib simplejson ujson orjson yappi \
        "nanobind>=2.0,!=3.0.0" scikit-build-core setuptools invoke filelock packaging pyyaml \
    && python3 -m pip install --no-build-isolation /src/projects/hipblaslt/tensilelite/rocisa \
    && python3 -m pip install --no-build-isolation --no-deps /src/projects/hipblaslt/tensilelite
COPY dockerfile/rocm10.2-nightly/select_a0_logic.py dockerfile/rocm10.2-nightly/manifest_a0_library.py /build/
RUN python3 /build/select_a0_logic.py \
        /src/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250-strict/gfx1250-strict \
        /build/logic \
    && python3 -m tensilelite.TensileCreateLibrary \
        --architecture gfx1250 --gfx1250v0 \
        --cxx-compiler /opt/venv/bin/amdclang++ --c-compiler /opt/venv/bin/amdclang \
        --assembler /opt/venv/bin/amdclang++ \
        --offload-bundler "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" \
        --code-object-version 6 --jobs "${BUILD_JOBS}" --no-enumerate \
        --library-format msgpack --no-compress --disable-asm-comments \
        /build/logic /build/generated HIP \
    && python3 /build/manifest_a0_library.py /build/generated/library/gfx1250v0 /build/a0-library-manifest.json \
        8a29d2e82b93f5065c7fe5630248d3689f0c3f06

FROM sdk
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
COPY --from=triton-builder /wheels /opt/fixed-triton-wheels
COPY --from=a0-builder /build/generated/library/gfx1250v0/ /opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1250v0/
COPY --from=a0-builder /build/generated/library/gfx1250v0/ /opt/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/hipblaslt/library/gfx1250v0/
COPY --from=a0-builder /build/a0-library-manifest.json /opt/base-provenance/a0-library-manifest.json
COPY dockerfile/rocm10.2-nightly/expected-manifest.json dockerfile/rocm10.2-nightly/requirements-base.lock \
     dockerfile/rocm10.2-nightly/unsigned-offset.ll dockerfile/rocm10.2-nightly/check_base.py \
     dockerfile/rocm10.2-nightly/protect_core.py dockerfile/rocm10.2-nightly/triton_unsigned_offset.py /opt/base-provenance/
RUN python3 -m pip install --force-reinstall --no-deps /opt/fixed-triton-wheels/*.whl \
    && python3 /opt/base-provenance/protect_core.py \
    && sha256sum "$(python3 -c 'from triton._C import libtriton; print(libtriton.__file__)')" \
        > /opt/base-provenance/fixed-triton-library.sha256 \
    && python3 /opt/base-provenance/check_base.py \
    && PROBE_OUTPUT=/opt/base-provenance/triton-offset TRITON_CACHE_DIR=/tmp/triton-qualification \
        python3 /opt/base-provenance/triton_unsigned_offset.py --expect-fixed \
    && python3 -m pip check
RUN python3 "${ROCM_PATH}/bin/rocjitsu-pretranslate.py" --target gfx1250 --fail-on-skipped \
        --bundler "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" \
        --report-dir /opt/base-provenance/aot-rccl "${ROCM_PATH}/.kpack/rccl_lib_gfx1250.kpack" \
    && python3 -c 'import hashlib, os, sysconfig; from pathlib import Path; p=Path(sysconfig.get_path("purelib")); a=p/"_rocm_sdk_core/lib/librocjitsu_gfx1250_b0_to_a0.so.0"; b=Path(os.environ["ROCM_PATH"])/"lib/librocjitsu_gfx1250_b0_to_a0.so.0"; assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()' \
    && mkdir -p /opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/share/rocjitsu/translations \
    && cp -a "${ROCM_PATH}/share/rocjitsu/translations/." \
        /opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/share/rocjitsu/translations/
ENV PIP_CONSTRAINT=/opt/base-provenance/core-constraints.txt \
    HSA_HOTSWAP_ENABLE=1
LABEL org.opencontainers.image.title="Experimental ROCm 10.2 nightly PyTorch base" \
    org.opencontainers.image.version="10.2.0a20261018-pytorch2.13.0" \
    com.superbench.therock.commit="ac008259bb0d7be08d4a2ce36dab2e46383c9e63" \
    com.superbench.rocm-systems.commit="2a8fffb7d64c59f83fe0d2efa22ed31d87d8adbf" \
    com.superbench.rocm-libraries.commit="8a29d2e82b93f5065c7fe5630248d3689f0c3f06" \
    com.superbench.llvm.commit="bdc35841c97ff51400273b91d7bc05c96e19c2d9" \
    com.superbench.triton.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485" \
    com.superbench.hipblaslt.profile="gfx1250v0-generic-grid"
ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \"$@\"", "--"]
CMD ["/bin/bash"]
