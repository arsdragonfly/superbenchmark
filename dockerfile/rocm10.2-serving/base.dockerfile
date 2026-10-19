FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl file git gpg kmod \
        libatomic1 libdw1t64 libelf1t64 libgomp1 libncurses6 libnuma1 \
        libssl3t64 libunwind8 perl python3 python3-dev python3-pip python3-venv && \
    apt-get clean && python3 -m venv "${VIRTUAL_ENV}" && \
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
COPY dockerfile/rocm10.2-serving/requirements-base.lock /opt/base-provenance/requirements-base.lock
COPY dockerfile/rocm10.2-serving/constraints-base.txt /opt/base-provenance/constraints-base.txt
RUN python3 -m pip install \
        --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
        --extra-index-url https://pypi.org/simple \
        --report /opt/base-provenance/pip-report.json \
        -c /opt/base-provenance/constraints-base.txt \
        -r /opt/base-provenance/requirements-base.lock && \
    rocm-sdk init && python3 -m pip check && \
    ROCM_PATH="$(rocm-sdk path --root)" && \
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}" > /etc/profile.d/rocm-sdk-env.sh && \
    printf '%s\n' \
        'export ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}"' \
        'export PATH="${ROCM_PATH}/bin:${PATH}"' \
        'export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"' \
        >> /etc/profile.d/rocm-sdk-env.sh
ENV BASH_ENV=/etc/profile.d/rocm-sdk-env.sh
COPY dockerfile/rocm10.2-serving/validate.py dockerfile/rocm10.2-serving/validate_translation_cache.py \
     dockerfile/rocm10.2-serving/expected-manifest.json dockerfile/rocm10.2-serving/hiprtc_unsigned_offset.cpp \
     /opt/base-provenance/
COPY dockerfile/rocm10.2-nightly/unsigned-offset.ll /opt/base-provenance/
RUN c++ -std=c++17 -D__HIP_PLATFORM_AMD__ -I"${ROCM_PATH}/include" \
        /opt/base-provenance/hiprtc_unsigned_offset.cpp \
        -L"${ROCM_PATH}/lib" -lhiprtc -o /opt/base-provenance/hiprtc_unsigned_offset && \
    python3 /opt/base-provenance/validate.py sdk
LABEL com.superbench.profile="gfx1250 inference Python 3.12 SDK intermediate" \
    com.superbench.therock.commit="98a3c880016a5d7fae72dda6c5a60356a4fcce2b" \
    com.superbench.rocm-systems.commit="b2d4b1ec26326e45153e36c7cb2eb94458b72188" \
    com.superbench.llvm.commit="bdc35841c97ff51400273b91d7bc05c96e19c2d9" \
    com.superbench.rocjitsu.a0-fix="upstream-73d0efc9286d"
ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \"$@\"", "--"]
CMD ["/bin/bash"]
