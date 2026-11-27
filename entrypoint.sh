#!/usr/bin/env bash
# Container entrypoint: turns the container's INKLING_* environment into the
# argument list of the CIRU launcher (serve.sh), then execs it so the engine is
# PID 1 and receives SIGTERM from `podman stop` directly.
#
# The container always binds 18753 (serve.sh's own default, also the upstream
# smoke test's default); start.sh remaps the host port with -p.
set -euo pipefail

PREFIX="${INKLING_PREFIX:-/opt/inkling/halo-v1.6.0}"
MODEL="${MODEL_PATH:-/models}"

args=(
  --model "$MODEL"
  --host 0.0.0.0
  --port 18753
  --mode "${INKLING_MODE:-vision}"
  --context "${INKLING_CONTEXT:-1048576}"
  --batch-tokens "${INKLING_BATCH_TOKENS:-4096}"
  --triton-cache "${INKLING_TRITON_CACHE:-/cache/inkling-ciru-triton}"
)

# Passed only when set, so serve.sh keeps its own mode-derived defaults (seqs 8
# in vision/concurrent, 1 in single/long-context, KV pool sized from context).
[[ -n "${INKLING_SEQS:-}" ]] && args+=(--seqs "$INKLING_SEQS")
[[ -n "${INKLING_KV_MIB:-}" ]] && args+=(--kv-mib "$INKLING_KV_MIB")
[[ "${INKLING_TOOLS:-1}" == "1" ]] && args+=(--tools)

# DFlash 2 drafter (serve.sh also reads INKLING_DFLASH2_MODEL from the env;
# passing the flag explicitly keeps --dry-run output unambiguous).
[[ -n "${INKLING_DFLASH2_MODEL:-}" ]] && args+=(--dflash2 "${INKLING_DFLASH2_MODEL}")
# Space-separated extra serve.sh flags, e.g. INKLING_EXTRA_ARGS="--no-graphs"
# or "--prefill-chunk-tokens 8192". No '=' inside values.
if [[ -n "${INKLING_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  args+=(${INKLING_EXTRA_ARGS})
fi

# Command-line arguments append to the composed list, so the wrapper forwards
# them unchanged: override the default CMD with the wrapper itself, e.g.
# `podman run IMAGE /opt/inkling/entrypoint.sh --dry-run` reaches serve.sh.
args+=("$@")

exec "$PREFIX/serve.sh" "${args[@]}"
