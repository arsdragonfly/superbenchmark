ARG APP_IMAGE=sb-rocm103-vllm-build:20261026
ARG FLYDSL_IMAGE=superbench-serving-flydsl:0.4.0-llvmfix
FROM ${FLYDSL_IMAGE} AS flydsl
FROM ${APP_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG ROLE=vllm
ARG SB_SOURCE_REVISION=unknown
COPY --from=flydsl /wheels /opt/fixed-flydsl-wheels
COPY --from=flydsl /opt/provenance /opt/base-provenance/flydsl-compiler
COPY --chmod=755 dockerfile/rocm10.2-serving/launch.py /usr/local/bin/sb-serve
COPY dockerfile/rocm10.2-serving/flydsl_unsigned_offset.py /opt/base-provenance/
COPY dockerfile/rocm10.3-nightly/validate.py /opt/base-provenance/check_base.py
RUN test "${SB_SOURCE_REVISION}" != unknown && \
    python3 -m pip install --force-reinstall --no-deps /opt/fixed-flydsl-wheels/*.whl && \
    python3 /opt/base-provenance/pin_fixed_wheel.py flydsl && \
    printf '%s\n' "${ROLE}" > /opt/base-provenance/serving-role && \
    python3 /opt/base-provenance/check_base.py --role "${ROLE}" && \
    python3 -m pip freeze > /opt/base-provenance/final-requirements.txt
ENV FLYDSL_RUNTIME_CACHE_DIR=/root/.cache/flydsl-rocm103-20261026
LABEL com.superbench.source.commit="${SB_SOURCE_REVISION}" \
    com.superbench.flydsl.commit="1f3511677662afc58bcf141dec812ccf05334fd8" \
    com.superbench.gpu-validation="NOT_RUN: CPU build and compiler checks only"
