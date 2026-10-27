ARG BASE_IMAGE=mi455xacr.azurecr.io/rocm-pytorch-base@sha256:bb83ff8487887bdd91be9e041dfedff7a15ab3dd8238d6a199735867d8631201
ARG FLYDSL_WHEEL_IMAGE=flydsl-source

FROM ${BASE_IMAGE} AS flydsl-builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG BUILD_JOBS=64
RUN apt-get update && apt-get install -y --no-install-recommends lld patchelf libzstd-dev zlib1g-dev \
    && apt-get clean \
    && python3 -m pip install cmake==4.0.3 ninja==1.13.2 nanobind==2.12.0 pybind11==3.0.4 \
        wheel==0.48.0 setuptools==79.0.1 \
    && git init /src/FlyDSL \
    && git -C /src/FlyDSL remote add origin https://github.com/ROCm/FlyDSL.git \
    && git -C /src/FlyDSL fetch --depth 1 origin 9dda109684ce26b97d2504503378df5266fb41b0 \
    && git -C /src/FlyDSL checkout --detach FETCH_HEAD \
    && git -C /src/FlyDSL submodule update --init --recursive --depth 1 \
    && git init /src/flydsl-llvm \
    && git -C /src/flydsl-llvm remote add origin https://github.com/llvm/llvm-project.git \
    && git -C /src/flydsl-llvm fetch --depth 1 --filter=blob:none origin e2a39f504fee836e4def9581bed817ecc327b9dc \
    && git -C /src/flydsl-llvm checkout --detach FETCH_HEAD
COPY dockerfile/rocm10.2-nightly/llvm-unsigned-offset.patch dockerfile/rocm10.2-nightly/unsigned-offset.ll /opt/provenance/
RUN cd /src/flydsl-llvm \
    && git apply --check /src/FlyDSL/thirdparty/llvm-rocdl-lld-argv0.patch \
    && git apply /src/FlyDSL/thirdparty/llvm-rocdl-lld-argv0.patch \
    && git apply --check --include='llvm/lib/Target/AMDGPU/*' /opt/provenance/llvm-unsigned-offset.patch \
    && git apply --include='llvm/lib/Target/AMDGPU/*' /opt/provenance/llvm-unsigned-offset.patch \
    && cmake -S llvm -B /build/flydsl-llvm -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
        -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
        -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_ENABLE_LLD=ON -DLLVM_PARALLEL_LINK_JOBS=2 \
        -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF -DBUILD_SHARED_LIBS=OFF \
        -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DMLIR_INCLUDE_TESTS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
        -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
        -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)" \
        -DHIP_PLATFORM=amd -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_INSTALL_PREFIX=/opt/flydsl-llvm \
    && cmake --build /build/flydsl-llvm --parallel "${BUILD_JOBS}" \
    && cmake --install /build/flydsl-llvm \
    && for selector in 0 1; do \
        /opt/flydsl-llvm/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3 \
            -global-isel="${selector}" -global-isel-abort=1 /opt/provenance/unsigned-offset.ll \
            -o "/opt/provenance/flydsl-llvm-${selector}.s"; \
        grep -q 'v_lshl_add_u64' "/opt/provenance/flydsl-llvm-${selector}.s"; \
    done \
    && cmake -S /src/FlyDSL -B /src/FlyDSL/build-fly -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=/opt/flydsl-llvm/lib/cmake/mlir \
        -DLLVM_DIR=/opt/flydsl-llvm/lib/cmake/llvm -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
        -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
        -Dnanobind_DIR="$(python3 -m nanobind --cmake_dir)" -DHIP_PLATFORM=amd \
    && cmake --build /src/FlyDSL/build-fly --parallel "${BUILD_JOBS}" \
    && cd /src/FlyDSL \
    && MLIR_PATH=/opt/flydsl-llvm FLY_REBUILD=0 FLYDSL_PACKAGE_VERSION_OVERRIDE=0.3.4.1 \
        python3 -m pip wheel --no-build-isolation --no-deps -w /wheels .
FROM scratch AS flydsl-source
COPY --from=flydsl-builder /wheels /wheels
COPY --from=flydsl-builder /opt/provenance /opt/provenance
FROM ${FLYDSL_WHEEL_IMAGE} AS flydsl

FROM ${BASE_IMAGE}

LABEL maintainer="SuperBench"
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Preserve the matched October 18 Python 3.14 / PyTorch 2.13 nightly stack.
# SDK packages include all devices; this argument selects the targets of locally
# built kernels (SuperBench benchmarks, RCCL and TransformerEngine).
ARG AMDGPU_TARGETS="gfx942 gfx950 gfx1250"
ARG NUM_MAKE_JOBS=64
ARG ROCM_VERSION=10.2.0a20261018
ARG TORCH_VERSION=2.13.0+rocm10.2.0a20261018
ARG TRITON_VERSION=3.8.0+git66b3f9ff.rocm10.2.0a20261018
ARG FLYDSL_COMMIT=9dda109684ce26b97d2504503378df5266fb41b0
ARG ROCM_SYSTEMS_COMMIT=2a8fffb7d64c59f83fe0d2efa22ed31d87d8adbf
# Default to the matched, qualified SDK RCCL; an explicit override remains available.
ARG RCCL_SOURCE_COMMIT=

ENV DEBIAN_FRONTEND=noninteractive \
    AMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${NUM_MAKE_JOBS}" \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    MPI_HOME=/usr/local/mpi \
    RCCL_SOURCE_PREFIX=/opt/rccl \
    EXTRAS_PATH=/opt/rocm/extras-10 \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake bc build-essential ca-certificates cmake curl \
        dmidecode flex gfortran git gnupg iproute2 jq libaio-dev \
        libatomic1 libboost-program-options-dev libcap2 libcurl4-openssl-dev \
        libibumad-dev libibverbs-dev libnuma-dev libpci-dev libquadmath0 librdmacm-dev \
        libssl-dev libtinfo6 libtool lshw net-tools ninja-build numactl \
        openssh-client openssh-server pciutils pkg-config python3-dev rsync sudo \
        util-linux wget && \
    apt-get clean

# Install the Docker CLI client only.
ARG DOCKER_VERSION=29.6.2
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" -o /tmp/docker.tgz && \
    tar --extract --file /tmp/docker.tgz --strip-components 1 --directory /usr/local/bin/ docker/docker && \
    rm /tmp/docker.tgz

# Update system config.
RUN mkdir -p /root/.ssh /var/run/sshd && \
    touch /root/.ssh/authorized_keys && \
    sed -i 's/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/; s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    printf '%s\n' '* soft nofile 1048576' '* hard nofile 1048576' >> /etc/security/limits.conf

# Match the Ubuntu 26.04 Azure HPC userspace stack; never install host drivers here.
# TODO: Add a selectable alternative networking stack after its qualification.
ARG DOCA_VERSION=3.4.0
ARG DOCA_PACKAGE_VERSION=3.4.0-085000
ARG DOCA_OPENMPI_VERSION=5.0.10rc2.2605121430-1.b99be7132e
ARG DOCA_BUNDLE_SHA256=60789bbbfc9b283a87a64589a0d4bd5d08d1b746494c24ce05e10bbd694de6bb
# Use NVIDIA's signed local repository bundle: the online repository's TLS EOF
# handling fails with Ubuntu 26.04 APT. Do not disable certificate/signature checks.
RUN curl -fsSL "https://content.mellanox.com/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_PACKAGE_VERSION}-26.04-ubuntu2604_amd64.deb" -o /tmp/doca-host.deb && \
    printf '%s  %s\n' "${DOCA_BUNDLE_SHA256}" /tmp/doca-host.deb | sha256sum --check --strict && \
    dpkg -i /tmp/doca-host.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        "doca-ofed-userspace=${DOCA_PACKAGE_VERSION}" "openmpi=${DOCA_OPENMPI_VERSION}" && \
    mpi_prefix=$(dpkg-query -L openmpi | sed -n 's#/bin/mpirun$##p') && \
    test -n "${mpi_prefix}" && \
    test -x "${mpi_prefix}/bin/mpicc" && \
    ln -s "${mpi_prefix}" "${MPI_HOME}" && \
    printf '%s\n' "${MPI_HOME}/lib" > /etc/ld.so.conf.d/superbench-mpi.conf && \
    ldconfig && \
    "${MPI_HOME}/bin/mpirun" --version && \
    apt-get purge -y doca-host && \
    rm /tmp/doca-host.deb && \
    apt-get update && apt-get clean

# Discover the repaired base SDK without re-expanding or replacing its A0 assets.
RUN python3 -c 'import sys;from importlib.metadata import version;assert version("rocm-sdk-core")==sys.argv[1];assert version("torch")==sys.argv[2]' "${ROCM_VERSION}" "${TORCH_VERSION}" && \
    ROCM_PATH="$(rocm-sdk path --root)" && \
    test -r "${ROCM_PATH}/.info/version" && \
    test -r "${ROCM_PATH}/lib/librccl.so" && \
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}" > /etc/profile.d/rocm10-env.sh && \
    printf '%s\n' \
        'export ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}" RCCL_HOME="${ROCM_PATH}"' \
        'export PATH="${EXTRAS_PATH}/bin:${MPI_HOME}/bin:${SB_MICRO_PATH}/bin:${ROCM_PATH}/bin:${PATH}"' \
        'export LD_LIBRARY_PATH="${EXTRAS_PATH}/lib:${MPI_HOME}/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"' \
        >> /etc/profile.d/rocm10-env.sh

# Resolve the SDK once at build time; shell startup must not reinitialize it.
ENV BASH_ENV=/etc/profile.d/rocm10-env.sh
ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \"$@\"", "--"]
CMD ["/bin/bash"]

# Register the bundled module until https://github.com/ROCm/TheRock/issues/3054
# is fixed (upstream registration: https://github.com/ROCm/TheRock/pull/8424).
# A relative .pth entry preserves the wrapper's lookup of its matching library.
RUN python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "amdsmi.pth").write_text("_rocm_sdk_core/share/amd_smi\n")' && \
    python3 -c 'import amdsmi'

# Built-in MPI calls exchange host data; RCCL exchanges GPU buffers. The DOCA
# Open MPI package needs neither a ROCm-aware rebuild nor unused OpenSHMEM.
ENV SB_ROCM_MEM_BW_BACKEND=transferbench \
    USE_HIP_DATATYPE=1 \
    USE_HIPBLAS_COMPUTETYPE=1

# Keep the signed RVS payload, but use the qualified wheel SDK for its runtime.
# The versioned nightly native APT index is unsigned; do not enable trusted=yes
# or install a second, conflicting ROCm runtime just to satisfy package metadata.
ARG RVS_VERSION=1.6.131-844
ARG RVS_PACKAGE_SHA256=47b4bc75f4bc747edd33ba6861069e06210eed38b66ff7052b3175e2f8cabe6c
RUN install -d -m 0755 /etc/apt/keyrings && \
    curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg | \
        gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg && \
    printf '%s\n' 'Types: deb' \
        'URIs: https://stable.repo.amd.com/rocm/extras/rvs/packages/ubuntu2604/' \
        'Suites: stable' 'Components: main' 'Architectures: amd64' \
        'Signed-By: /etc/apt/keyrings/amdrocm.gpg' > /etc/apt/sources.list.d/amdrocm-rvs.sources && \
    apt-get update && \
    mkdir -p /tmp/rvs-package && cd /tmp/rvs-package && \
    apt-get download "amdrocm10-rvs=${RVS_VERSION}" && \
    echo "${RVS_PACKAGE_SHA256}  amdrocm10-rvs_${RVS_VERSION}_amd64.deb" | sha256sum -c && \
    dpkg-deb --extract "amdrocm10-rvs_${RVS_VERSION}_amd64.deb" extracted && \
    mkdir -p "${EXTRAS_PATH}" && \
    cp -a extracted/opt/rocm/extras-10/. "${EXTRAS_PATH}/" && \
    if [ -d extracted/usr/share/doc ]; then cp -a extracted/usr/share/doc/. /usr/share/doc/; fi && \
    cd / && rm -rf /tmp/rvs-package && \
    apt-get clean && \
    test -x "${EXTRAS_PATH}/bin/rvs" && \
    test -x "${EXTRAS_PATH}/bin/TransferBench" && \
    for binary in "${EXTRAS_PATH}/bin/rvs" "${EXTRAS_PATH}/bin/TransferBench" "${EXTRAS_PATH}/lib/rvs/"*.so; do \
        ldd "${binary}" > /tmp/rvs-linkage.log; \
        if grep -q 'not found' /tmp/rvs-linkage.log; then cat /tmp/rvs-linkage.log; exit 1; fi; \
    done

WORKDIR ${SB_HOME}
COPY third_party third_party

# Build RCCL-tests, and RCCL itself when RCCL_SOURCE_COMMIT is set, from one
# rocm-systems revision. The source-built library precedes the SDK library on
# LD_LIBRARY_PATH, which RUNPATH alone cannot override, so soname lookups
# (including SuperBench's mpirun -x LD_LIBRARY_PATH) resolve the fixed RCCL.
RUN rccl_architectures=$(printf '%s' "${AMDGPU_TARGETS}" | tr -s '[:space:]' ';' | sed 's/^;//; s/;$//') && \
    if [ -z "${rccl_architectures}" ]; then \
        echo 'AMDGPU_TARGETS must contain at least one GPU architecture.' >&2; exit 1; \
    fi && \
    if [ -n "${RCCL_SOURCE_COMMIT}" ]; then \
        rocm_systems_commit="${RCCL_SOURCE_COMMIT}"; rccl_prefix="${RCCL_SOURCE_PREFIX}"; \
        rccl_runtime_prefix="${RCCL_SOURCE_PREFIX}"; \
        rccl_projects='projects/rccl projects/rccl-tests'; \
    else \
        rocm_systems_commit="${ROCM_SYSTEMS_COMMIT}"; rccl_prefix="${ROCM_PATH}"; \
        rccl_runtime_prefix=/opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries; \
        rccl_projects='projects/rccl-tests'; \
    fi && \
    git init /tmp/rocm-systems && \
    git -C /tmp/rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git && \
    git -C /tmp/rocm-systems sparse-checkout set ${rccl_projects} && \
    git -C /tmp/rocm-systems fetch --depth 1 --filter=blob:none origin "${rocm_systems_commit}" && \
    git -C /tmp/rocm-systems checkout --detach FETCH_HEAD && \
    if [ -n "${RCCL_SOURCE_COMMIT}" ]; then \
        cmake -S /tmp/rocm-systems/projects/rccl -B /tmp/rccl-build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/hipcc" \
            -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
            -DROCM_PATH="${ROCM_PATH}" \
            -DGPU_TARGETS="${rccl_architectures}" \
            -DBUILD_TESTS=OFF \
            -DCMAKE_INSTALL_PREFIX="${rccl_prefix}" \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_INSTALL_RPATH="${ROCM_PATH}/lib" && \
        cmake --build /tmp/rccl-build -j"${NUM_MAKE_JOBS}" && \
        cmake --install /tmp/rccl-build && \
        test -r "${rccl_prefix}/lib/librccl.so" && \
        printf 'export RCCL_HOME=%q\n' "${rccl_prefix}" >> /etc/profile.d/rocm10-env.sh && \
        printf 'export LD_LIBRARY_PATH=%q:"${LD_LIBRARY_PATH}"\n' "${rccl_prefix}/lib" >> /etc/profile.d/rocm10-env.sh; \
    fi && \
    cmake -S /tmp/rocm-systems/projects/rccl-tests -B /tmp/rccl-tests-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${rccl_prefix};${ROCM_PATH};${MPI_HOME}" \
        -DROCM_PATH="${ROCM_PATH}" \
        -DGPU_TARGETS="${rccl_architectures}" \
        -DUSE_MPI=ON \
        -DCMAKE_INSTALL_PREFIX="${SB_HOME}" \
        -DCMAKE_INSTALL_RPATH="${rccl_runtime_prefix}/lib;${ROCM_PATH}/lib;${MPI_HOME}/lib" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON && \
    cmake --build /tmp/rccl-tests-build -j"${NUM_MAKE_JOBS}" && \
    cmake --install /tmp/rccl-tests-build && \
    rm -rf /tmp/rocm-systems /tmp/rccl-build /tmp/rccl-tests-build && \
    bash -c 'ldd "${SB_HOME}/bin/all_reduce_perf"' | grep -F "${rccl_runtime_prefix}/lib/librccl.so"

# Do not invoke the legacy ROCm aggregate target: it clones HIP 5.5 samples,
# rebuilds math libraries, and installs unqualified Megatron/DeepSpeed dependencies.
RUN test -f third_party/fio/Makefile && \
    test -f third_party/perftest/autogen.sh && \
    make -C third_party fio rocm_perftest

# Extract only matching prebuilt clients and their external GPU kernel packs.
# The ELFs alone cannot initialize matrices without the blas_test_*.kpack files.
ARG ROCM_TESTS_SHA256=b9ba9c69944100c78ad091db40c790b46cc8664beb689ccd6b3c11f357c06b15
RUN curl -fsSL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-tests-${ROCM_VERSION}.tar.gz" -o /tmp/rocm-tests.tar.gz && \
    printf '%s  %s\n' "${ROCM_TESTS_SHA256}" /tmp/rocm-tests.tar.gz | sha256sum --check --strict && \
    tar -xzf /tmp/rocm-tests.tar.gz -C "${SB_HOME}" --wildcards --no-wildcards-match-slash \
        './bin/rocblas-bench' './bin/hipblaslt-bench' './.kpack/blas_test_*.kpack' \
        './share/therock/therock_manifest.json' && \
    rm /tmp/rocm-tests.tar.gz && \
    test -x "${SB_HOME}/bin/rocblas-bench" && \
    test -x "${SB_HOME}/bin/hipblaslt-bench" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx942.kpack" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx950.kpack" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx1250.kpack" && \
    python3 -c 'import json, os; from pathlib import Path; a=json.loads((Path(os.environ["ROCM_PATH"])/"share/therock/therock_manifest.json").read_text()); b=json.loads((Path(os.environ["SB_HOME"])/"share/therock/therock_manifest.json").read_text()); pins=lambda m:{x["submodule_name"]:x["pin_sha"] for x in m["submodules"]}; assert a["rocm_package_version"] == b["rocm_package_version"]; assert a["the_rock_commit"] == b["the_rock_commit"]; assert pins(a) == pins(b)'

COPY . .
ARG SB_SOURCE_REVISION=unknown
# Keep AMD's framework and all-device SDK packages when resolving SuperBench deps.
# The same constraints protect them while building the frameworks below.
# The SDK supplies amdsmi via .pth, not a separate distribution; use the torch
# extra so amdworker does not install a PyPI wrapper that shadows the SDK module.
RUN test "${SB_SOURCE_REVISION}" != unknown && \
    cp "${PIP_CONSTRAINT}" /tmp/rocm10-constraints.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SUPERBENCH="$(python3 -c 'import superbench; print(superbench.__version__)')+g$(printf '%s' "${SB_SOURCE_REVISION}" | cut -c1-12)" \
        python3 -m pip install -c /tmp/rocm10-constraints.txt '.[torch]' && \
    python3 -m pip install pytest pytest-timeout vcrpy

# TE 2.15 still requires the legacy SMI headers/library, which the wheel SDK
# no longer ships. Build only that compatibility library from the matched source.
RUN apt-get install -y --no-install-recommends libdrm-dev pkg-config && \
    git init /tmp/rocm-smi-source && \
    git -C /tmp/rocm-smi-source remote add origin https://github.com/ROCm/rocm-systems.git && \
    git -C /tmp/rocm-smi-source sparse-checkout set projects/rocm-smi-lib && \
    git -C /tmp/rocm-smi-source fetch --depth 1 --filter=blob:none origin "${ROCM_SYSTEMS_COMMIT}" && \
    git -C /tmp/rocm-smi-source checkout --detach FETCH_HEAD && \
    cmake -S /tmp/rocm-smi-source/projects/rocm-smi-lib -B /tmp/rocm-smi-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/rocm-smi \
        -DBUILD_TESTS=OFF -DROCM_SMI_DISABLE=OFF && \
    cmake --build /tmp/rocm-smi-build --parallel 16 && \
    cmake --install /tmp/rocm-smi-build && \
    printf '%s\n' 'export LD_LIBRARY_PATH="/opt/rocm-smi/lib:${LD_LIBRARY_PATH}"' >> /etc/profile.d/rocm10-env.sh && \
    rm -rf /tmp/rocm-smi-source /tmp/rocm-smi-build
ENV CPATH=/opt/rocm-smi/include \
    LIBRARY_PATH=/opt/rocm-smi/lib
COPY --from=flydsl /wheels /opt/fixed-flydsl-wheels
COPY --from=flydsl /opt/provenance /opt/base-provenance/flydsl-compiler
RUN python3 -m pip install --no-deps /opt/fixed-flydsl-wheels/*.whl && \
    python3 dockerfile/rocm10.2-nightly/pin_fixed_wheel.py flydsl && \
    cp "${PIP_CONSTRAINT}" /tmp/rocm10-constraints.txt

# Build TransformerEngine, AITER and Megatron-LM from pinned sources against the
# base's PyTorch, Triton and SDK, without build isolation or dependency resolution.
# TE v2.15_rocm, AITER snapshot 2026-09-30, Megatron Core v0.19.2.
# Keep TE attention unfused and use its explicit FP32 bias-gradient fallback.
ARG TE_COMMIT=e7835ed1b134f56e58c575fba2455471064cab9b
ARG AITER_COMMIT=89a47b84ac4b576339a50047e290c383c2377389
ARG MEGATRON_COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
ARG AITER_PACKAGE_VERSION=
ENV SB_ROCM10_FRAMEWORK_ROOT=/opt/rocm10-frameworks \
    SB_MEGATRON_PATH=/opt/rocm10-frameworks/Megatron-LM \
    AITER_USE_SYSTEM_TRITON=1 \
    TRITON_F32_DEFAULT=ieee \
    MAX_JOBS=16 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=0 \
    NVTE_FUSED_ATTN_CK=0 \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_UNFUSED_FP32_BGRAD=1
RUN framework_jobs="${NUM_MAKE_JOBS}" && \
    if [ "${framework_jobs}" -gt 16 ]; then framework_jobs=16; fi && \
    te_architectures=$(printf '%s' "${AMDGPU_TARGETS}" | tr -s '[:space:]' ';' | sed 's/^;//; s/;$//') && \
    export MAX_JOBS="${framework_jobs}" CMAKE_BUILD_PARALLEL_LEVEL="${framework_jobs}" \
        NVTE_BUILD_MAX_JOBS="${framework_jobs}" PIP_CONSTRAINT=/tmp/rocm10-constraints.txt \
        TMPDIR=/tmp/rocm10-frameworks-scratch NVTE_USE_ROCM=1 NVTE_FRAMEWORK=pytorch \
        NVTE_ROCM_ARCH="${te_architectures}" NVTE_NO_LOCAL_VERSION=1 NVTE_CK_JIT=1 \
        PREBUILD_KERNELS=0 BUILD_TARGET=rocm RCCL_HOME="${ROCM_PATH}" && \
    if [ -n "${AITER_PACKAGE_VERSION}" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AITER="${AITER_PACKAGE_VERSION}"; \
    fi && \
    python3 -c 'import sys;from importlib.metadata import version;expected={"torch":sys.argv[1],"triton":sys.argv[2],"rocm-sdk-core":sys.argv[3],"rocm-sdk-devel":sys.argv[3]};assert all(version(n)==v for n,v in expected.items()),expected' "${TORCH_VERSION}" "${TRITON_VERSION}" "${ROCM_VERSION}" && \
    mkdir -p "${TMPDIR}" && \
    python3 -m pip install \
        pybind11==3.0.1 pydantic==2.13.5 einops==0.8.2 psutil==7.2.2 mpi4py==4.1.1 \
        pandas==3.0.5 nvdlfw-inspect==0.2.2 \
        onnxscript==0.7.2 onnx-ir==1.0.0 onnx==1.23.1 ml_dtypes==0.6.0 && \
    for source in "TransformerEngine https://github.com/ROCm/TransformerEngine.git ${TE_COMMIT}" \
                  "aiter https://github.com/ROCm/aiter.git ${AITER_COMMIT}" \
                  "Megatron-LM https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_COMMIT}"; do \
        set -- ${source} && \
        git init -q "${SB_ROCM10_FRAMEWORK_ROOT}/$1" && \
        git -C "${SB_ROCM10_FRAMEWORK_ROOT}/$1" remote add origin "$2" && \
        git -C "${SB_ROCM10_FRAMEWORK_ROOT}/$1" fetch -q --depth 1 origin "$3" && \
        git -C "${SB_ROCM10_FRAMEWORK_ROOT}/$1" checkout -q --detach FETCH_HEAD && \
        git -C "${SB_ROCM10_FRAMEWORK_ROOT}/$1" submodule update --init --recursive --depth 1 || exit 1; \
    done && \
    git -C "${SB_ROCM10_FRAMEWORK_ROOT}/TransformerEngine" apply --check "${SB_HOME}/third_party/TransformerEngine/rocm_fp32_bgrad.patch" && \
    git -C "${SB_ROCM10_FRAMEWORK_ROOT}/TransformerEngine" apply "${SB_HOME}/third_party/TransformerEngine/rocm_fp32_bgrad.patch" && \
    git -C "${SB_MEGATRON_PATH}" apply "${SB_HOME}/third_party/Megatron/megatron_core_0.19.2_rocm10.patch" && \
    if [ -n "${AITER_PACKAGE_VERSION}" ]; then \
        python3 -m pip install -r "${SB_ROCM10_FRAMEWORK_ROOT}/aiter/requirements.txt"; \
    fi && \
    python3 -m pip wheel --no-build-isolation --no-deps -w /tmp/rocm10-wheels \
        "${SB_ROCM10_FRAMEWORK_ROOT}/TransformerEngine" "${SB_ROCM10_FRAMEWORK_ROOT}/aiter" "${SB_MEGATRON_PATH}" && \
    python3 -m pip install --no-deps /tmp/rocm10-wheels/*.whl && \
    python3 -m pip check && \
    sha256sum --check /opt/base-provenance/fixed-triton-library.sha256 && \
    python3 dockerfile/rocm10.2-nightly/verify_core.py && \
    rm -rf /tmp/rocm10-wheels "${TMPDIR}"

# Build-only framework settings above stay out of the runtime environment.
RUN printf '%s\n' \
        "PATH=${PATH}" \
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        "SB_MICRO_PATH=${SB_MICRO_PATH}" \
        "SB_ROCM_MEM_BW_BACKEND=${SB_ROCM_MEM_BW_BACKEND}" \
        "ROCM_PATH=${ROCM_PATH}" \
        "ROCM_HOME=${ROCM_HOME}" \
        "HIP_PATH=${HIP_PATH}" \
        "RCCL_HOME=${RCCL_HOME}" \
        "SB_ROCM10_FRAMEWORK_ROOT=${SB_ROCM10_FRAMEWORK_ROOT}" \
        "SB_MEGATRON_PATH=${SB_MEGATRON_PATH}" \
        "AITER_USE_SYSTEM_TRITON=${AITER_USE_SYSTEM_TRITON}" \
        "TRITON_F32_DEFAULT=${TRITON_F32_DEFAULT}" \
        "MAX_JOBS=${MAX_JOBS}" \
        "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}" \
        "NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN}" \
        "NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN}" \
        "NVTE_FUSED_ATTN_CK=${NVTE_FUSED_ATTN_CK}" \
        "NVTE_FUSED_ATTN_AOTRITON=${NVTE_FUSED_ATTN_AOTRITON}" \
        "NVTE_ROCM_UNFUSED_FP32_BGRAD=${NVTE_ROCM_UNFUSED_FP32_BGRAD}" \
        > /etc/environment

# Build SuperBench and finish installation.
RUN CXX="${ROCM_PATH}/bin/hipcc" make cppbuild && \
    make postinstall

RUN python3 -m pip check && \
    python3 -c 'import transformer_engine.pytorch, flydsl; from importlib.metadata import version; print({name: version(name) for name in ("amd-aiter", "megatron-core")}); print("CPU-safe framework imports passed; AITER/Megatron GPU checks remain required")' && \
    test -x "${SB_HOME}/bin/dist_inference" && \
    sb --help && \
    rm /tmp/rocm10-constraints.txt && \
    rm -rf .git
LABEL com.superbench.source.commit="${SB_SOURCE_REVISION}" \
    com.superbench.rocm.version="${ROCM_VERSION}" \
    com.superbench.transformerengine.commit="${TE_COMMIT}" \
    com.superbench.transformerengine.profile="basic-no-fused-attention-fp32-bgrad-fallback" \
    com.superbench.aiter.commit="${AITER_COMMIT}" \
    com.superbench.megatron.commit="${MEGATRON_COMMIT}" \
    com.superbench.flydsl.commit="${FLYDSL_COMMIT}" \
    com.superbench.flydsl.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485"
