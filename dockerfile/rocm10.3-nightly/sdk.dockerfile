ARG UBUNTU_IMAGE=ubuntu:26.04@sha256:2260313b31c8c011cd2eebe728008efac1b3982be73eb71348ea2648d2c0e09b
FROM ${UBUNTU_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG PYTHON_VERSION=3.14
ARG DEVICE_EXTRAS=devel,libraries,device-all
ARG TORCH_DEVICE_EXTRAS=device-all
ENV DEBIAN_FRONTEND=noninteractive VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl file git gpg kmod \
        libatomic1 libdw1t64 libelf1t64 libgomp1 libncurses6 libnuma1 \
        libssl3t64 libunwind8 perl python3 python3-dev python3-pip python3-venv && \
    apt-get clean && python3 -m venv "${VIRTUAL_ENV}" && \
    python3 -c 'import sys; assert ".".join(map(str,sys.version_info[:2])) == sys.argv[1]' "${PYTHON_VERSION}"
COPY dockerfile/rocm10.3-nightly/requirements-base.lock dockerfile/rocm10.3-nightly/expected-manifest.json /opt/base-provenance/
RUN python3 -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
        --extra-index-url https://pypi.org/simple --report /opt/base-provenance/pip-report.json \
        -r /opt/base-provenance/requirements-base.lock \
        "rocm[${DEVICE_EXTRAS}]==10.3.0a20261026" \
        "torch[${TORCH_DEVICE_EXTRAS}]==2.13.0+rocm10.3.0a20261026" \
        "torchvision[${TORCH_DEVICE_EXTRAS}]==0.28.0+rocm10.3.0a20261026" && \
    rocm-sdk init && python3 -m pip check && \
    ROCM_PATH="$(rocm-sdk path --root)" && \
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}" > /etc/profile.d/rocm-sdk-env.sh && \
    printf '%s\n' \
        'export ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}"' \
        'export PATH="${ROCM_PATH}/bin:${PATH}"' \
        'export LD_LIBRARY_PATH="${ROCM_PATH}/../_rocm_sdk_core/lib:${ROCM_PATH}/../_rocm_sdk_libraries/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"' \
        >> /etc/profile.d/rocm-sdk-env.sh && \
    python3 -c 'import sysconfig;from pathlib import Path;Path(sysconfig.get_path("purelib"),"amdsmi.pth").write_text("_rocm_sdk_core/share/amd_smi\n")'
ENV BASH_ENV=/etc/profile.d/rocm-sdk-env.sh HSA_HOTSWAP_ENABLE=1 \
    TRITON_HIP_USE_ASYNC_COPY=0 TRITON_HIP_USE_EXPERT_SCHEDULING=0
COPY dockerfile/rocm10.2-nightly/triton_unsigned_offset.py dockerfile/rocm10.2-nightly/unsigned-offset.ll /opt/base-provenance/
ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \"$@\"", "--"]
CMD ["/bin/bash"]
