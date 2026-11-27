# Inkling 2 Mini · CIRU Halo Agent (gfx1151) — self-contained ROCm vLLM runtime
# for AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S).
#
# Runtime: v1.6.0 (native W6 WMMA DFlash 2 block-diffusion speculation,
# --dflash2). The default launch uses DFlash 2; base weights and wheels are
# unchanged from v1.5.0.
#
# No compilation here: the release ships a prebuilt portable gfx1151 ABI v3
# all-HIP library, so the image is pinned wheels + a patched vLLM/Transformers
# source overlay installed by scripts/install.sh. The ~23 GB checkpoint is NOT
# baked in — bind-mount it read-only at /models.
#
# Do NOT add a system ROCm to this base: the wheels are self-contained and
# scripts/rocm/rocr-env.sh preloads its own patched libhsa-runtime64.so.1.
# Ubuntu 24.04 is load-bearing — rocr-env.sh rejects glibc/runtime combinations
# where the LD_PRELOAD override cannot be the only mapped HSA library.
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Runtime libraries only: libnuma1 (torch/vLLM), git+curl (installer),
# python3 (3.12: the ROCm wheels are cp312-only), build-essential as a safety
# net for any sdist fallback during wheel resolution.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates libnuma1 python3 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# Paths are baked into the prefix launcher written by install.sh, so they must
# stay identical at build and run time. UV_PYTHON_DOWNLOADS=never makes a
# missing system 3.12 fail loudly instead of pulling a uv-managed interpreter.
ENV INKLING_VENV=/opt/inkling/venv \
    INKLING_DEPS=/opt/inkling/transformers-deps \
    INKLING_PREFIX=/opt/inkling/halo-v1.6.0 \
    INKLING_RUNTIME=/opt/inkling/runtime \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv --python /usr/bin/python3.12 "$INKLING_VENV"

# Pinned ROCm 10 vLLM wheel plus the older hub/safetensors pair that the pinned
# Transformers source expects, isolated in --target "$INKLING_DEPS".
# torch/triton are pinned explicitly: install.sh verifies these exact versions
# and an unpinned resolve could take them from PyPI instead of the ROCm index.
RUN uv pip install \
      --python "$INKLING_VENV/bin/python" \
      --extra-index-url https://wheels.vllm.ai/rocm/0.34.0/rocm100/ \
      'vllm==0.34.0+rocm100' \
      'torch==2.13.0+rocm10.0.0' \
      'triton==3.8.0+git669b31ac' \
      'huggingface-hub==2.4.0' \
      'safetensors==0.8.0' \
    && uv pip install \
      --python "$INKLING_VENV/bin/python" \
      --target "$INKLING_DEPS" --no-deps \
      'huggingface-hub==1.31.0' 'safetensors==0.8.0'

# CIRU runtime bundle: overlay/, patches/, scripts/ (including the prebuilt
# HIP .so and the patched HSA runtime), MANIFEST.sha256, WEIGHTS.sha256.
# Kept after installation (~15 MB) so the upstream smoke test and the weight
# manifest stay reachable inside the container.
ARG CIRU_REPO=https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git
ARG CIRU_REVISION=64788d4695649aba68d91af384300b8ce370d0b5
RUN git clone --depth 1 --branch v1.6.0 "$CIRU_REPO" "$INKLING_RUNTIME" \
    && test "$(git -C "$INKLING_RUNTIME" rev-parse HEAD)" = "$CIRU_REVISION"

# install.sh verifies MANIFEST.sha256, checks the wheel versions and the
# _rocm_C.abi3.so native extension, fetches the pinned vLLM + Transformers
# sources, copies overlay/vllm/. over the vLLM source tree, symlinks the wheel's
# *.so/_version.py/vllm-rs into it, and writes "$INKLING_PREFIX/serve.sh" — a
# generated wrapper that execs scripts/serve.sh with --overlay/--transformers-src/
# --transformers-deps/--python already fixed. Prefix must not exist beforehand.
# Network required: github.com, wheels.vllm.ai, pypi.org.
RUN "$INKLING_RUNTIME/scripts/install.sh" \
      --venv "$INKLING_VENV" \
      --transformers-deps "$INKLING_DEPS" \
      --prefix "$INKLING_PREFIX"

# Triton's AMD backend JIT-compiles a small C shim (hip_utils.c) with
# gcc -I/usr/include/python3.12 at first kernel launch, so the CPython headers
# must be present at runtime even though nothing here is precompiled. Kept as a
# separate late layer so the multi-GB wheel + install.sh layers above stay
# cached across edits.
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Triton kernel cache, ROCm/HIP JIT cache and HF stub caches all resolve under
# $HOME/.cache; symlink it into the runtime bind-mount so a recreated container
# keeps the cold-start kernel build instead of repeating it. rm -rf first:
# ln -sfn against an existing *directory* nests the link inside it.
RUN rm -rf /root/.cache && mkdir -p /cache && ln -s /cache /root/.cache

COPY entrypoint.sh /opt/inkling/entrypoint.sh
RUN chmod +x /opt/inkling/entrypoint.sh

ENV MODEL_PATH=/models \
    INKLING_TRITON_CACHE=/cache/inkling-ciru-triton \
    INKLING_PREFIX=/opt/inkling/halo-v1.6.0
EXPOSE 18753
# /health is the vLLM route (vllm/entrypoints/serve/instrumentator/health.py):
# 200 while the engine is alive, 503 on EngineDeadError. start.sh polls the
# published port; podman only applies HEALTHCHECK when --health-check is on.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
  CMD curl -fsS http://127.0.0.1:18753/health || exit 1
CMD ["/opt/inkling/entrypoint.sh"]
