ARG APP_IMAGE
ARG FLYDSL_IMAGE=superbench-serving-flydsl:0.4.0-llvmfix
FROM ${FLYDSL_IMAGE} AS flydsl
FROM ${APP_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
ARG ROLE
COPY --chmod=755 dockerfile/rocm10.2-serving/launch.py /usr/local/bin/sb-serve
COPY --from=flydsl /wheels /opt/fixed-flydsl-wheels
COPY --from=flydsl /opt/provenance /opt/base-provenance/flydsl-compiler
COPY dockerfile/rocm10.2-serving/flydsl_unsigned_offset.py dockerfile/rocm10.2-serving/validate.py \
     dockerfile/rocm10.2-serving/validate_translation_cache.py /opt/base-provenance/
RUN python3 -m pip install --force-reinstall --no-deps /opt/fixed-flydsl-wheels/*.whl && \
    python3 /opt/base-provenance/pin_fixed_wheel.py flydsl && \
    printf '%s\n' "${ROLE}" > /opt/base-provenance/serving-role && \
    python3 /opt/base-provenance/validate.py "${ROLE}" && \
    python3 -m pip freeze > /opt/base-provenance/final-requirements.txt
ENV FLYDSL_RUNTIME_CACHE_DIR=/root/.cache/flydsl-rocm102-20261015-a0-retest
LABEL com.superbench.flydsl.commit="1f3511677662afc58bcf141dec812ccf05334fd8" \
    com.superbench.flydsl.llvm.commit="e2a39f504fee836e4def9581bed817ecc327b9dc" \
    com.superbench.flydsl.llvm.patch="3b2c3b02c0374cba15e30739b71780ad820e6485" \
    com.superbench.gpu-validation="CPU compiler and import checks only; real GPU inference requires separate testing"
