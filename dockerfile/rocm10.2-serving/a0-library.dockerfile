ARG BASE_IMAGE=superbench-serving-sdk:10.2.0a20261015-py3.12
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=16
ENV CMAKE_PREFIX_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel \
    CMAKE_BUILD_PARALLEL_LEVEL=16 \
    CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/opt/venv/bin/amdclang++" \
    TENSILE_DISABLE_HELPER_CACHE=1 SKBUILD_EDITABLE_REBUILD=false
RUN git init /src && \
    git -C /src remote add origin https://github.com/ROCm/rocm-libraries.git && \
    git -C /src sparse-checkout set projects/hipblaslt/tensilelite projects/hipblaslt/library/include \
        projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250-strict/gfx1250-strict \
        shared/origami shared/stinkytofu cmake && \
    git -C /src fetch --depth 1 --filter=blob:none origin 3dc098eb2db3c8329b7d447bab37021d529bc23d && \
    git -C /src checkout --detach FETCH_HEAD && \
    python3 -m pip install cmake==4.0.3 ninja==1.13.2 msgpack==1.2.3 joblib==1.6.0 \
        simplejson==4.2.0 ujson==6.0.0 orjson==3.13.0 yappi==1.7.6 nanobind==3.1.0 \
        scikit-build-core==1.1.1 setuptools==79.0.1 invoke==3.0.3 filelock==4.2.0 \
        packaging==26.3 pyyaml==6.0.3 && \
    python3 -m pip install --no-build-isolation --no-deps /src/projects/hipblaslt/tensilelite/rocisa && \
    python3 -m pip install --no-build-isolation --no-deps /src/projects/hipblaslt/tensilelite
COPY dockerfile/rocm10.2-nightly/select_a0_logic.py dockerfile/rocm10.2-nightly/manifest_a0_library.py /build/
RUN python3 /build/select_a0_logic.py \
        /src/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250-strict/gfx1250-strict \
        /build/logic && \
    python3 -m tensilelite.TensileCreateLibrary \
        --architecture gfx1250 --gfx1250v0 \
        --cxx-compiler /opt/venv/bin/amdclang++ --c-compiler /opt/venv/bin/amdclang \
        --assembler /opt/venv/bin/amdclang++ \
        --offload-bundler "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler" \
        --code-object-version 6 --jobs "${BUILD_JOBS}" --no-enumerate \
        --library-format msgpack --no-compress --disable-asm-comments \
        /build/logic /build/generated HIP && \
    python3 /build/manifest_a0_library.py /build/generated/library/gfx1250v0 /build/a0-library-manifest.json \
        3dc098eb2db3c8329b7d447bab37021d529bc23d && \
    python3 -c 'import json;from pathlib import Path;m=json.loads(Path("/build/a0-library-manifest.json").read_text());assert len(m["sha256"])==271 and m["code_objects"]==135'

FROM scratch
COPY --from=builder /build/generated/library/gfx1250v0 /gfx1250v0
COPY --from=builder /build/a0-library-manifest.json /build/logic-manifest.json /provenance/
