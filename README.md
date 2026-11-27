# Inkling 2 Mini · CIRU Halo Agent — one-command Podman deploy (AMD Strix Halo 128 GB)

Serves [jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151](https://huggingface.co/jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151)
(vLLM 0.34.0+rocm100, native RDNA 3.5 all-HIP kernels, native W6 WMMA DFlash 2
block-diffusion speculation by default — bundled adaptive DSpark as fallback —
native ROCm vision, native tool calling) in a rootless Podman container on an
AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151`).

Built on the [CIRU runtime](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151),
which ships a **prebuilt** portable `gfx1151` ABI v3 all-HIP library — so there is no
`hipcc`/compile step here, unlike the GLM example.

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every var has a default
./start.sh            # build (cached) -> download model -> run -> wait healthy
./stop.sh
```

## What's here

| File            | Role                                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Containerfile` | Pinned ROCm 10 wheels (vLLM/torch/triton/hub/safetensors) plus a patched vLLM + Transformers source overlay installed by `scripts/install.sh`. Ships the prebuilt `gfx1151` HIP library — **no compilation**. Checksum-pinned git revision. |
| `entrypoint.sh` | Container launcher: composes the `serve.sh` argument list from the container's `INKLING_*` env, then `exec`s it so the engine is PID 1.                                                            |
| `start.sh`      | One-command entrypoint: preflight → build (cached) → download checkpoint (resumable, checksum-verified) → run → health-poll.                                                                   |
| `stop.sh`       | Stops the container with a 60 s grace period; idempotent.                                                                                                                                       |
| `.env.sample`   | Optional overrides — copy to `.env`.                                                                                                                                                            |

The ~23 GB checkpoint (base + fitted weights) is **not** baked into the image; it is
bind-mounted read-only at `/models`. The image is built on first run and cached for
subsequent launches.

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- **AMD GPU driver** present: `/dev/kfd` and `/dev/dri/renderD128` exist
- Current user in the **`render`** (and ideally `video`) group — check with `id -nG`
- **~50 GB free disk**: ≈23 GB weights + ≈18 GB image + caches
- **≥128 GB unified RAM** — the default 1M-token profile (~52 GiB BF16 KV pool) uses most of a 128 GB box; see Troubleshooting if the engine can't allocate
- `curl`, `python3`, and `hf`/`huggingface-cli` (from `pip install huggingface_hub[cli]`) or the `huggingface_hub` Python package on the host for the first-model download
- Outbound network on the first build (github.com, wheels.vllm.ai, pypi.org) and the first download

No root/sudo is required for the normal path. Podman runs rootless; GPU access comes from `--group-add keep-groups` (carries your host `render` group into the container) plus SELinux `:z` bind relabeling.

## Quick start

```bash
cp .env.sample .env      # optional
./start.sh
```

First run:

1. Builds the image: pinned ROCm wheels + the CIRU overlay installed by `scripts/install.sh` (~a few minutes, no kernel compilation).
2. Downloads the ~23 GB checkpoint to `./models` (resumable), then verifies `WEIGHTS.sha256`.
3. Starts the container and streams the engine log to `.inkling.log`.
4. Exits "ready" once `/health` returns 200 (~1–3 min on a warm cache, longer on the first Triton kernel build).

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> main
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` after the first time skips the build and the download and reaches ready materially faster (the Triton kernel cache lives in `./.cache`).

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first): **exported shell env → `.env` → `start.sh` defaults.** Values must not contain `=` (the `.env` parser splits each line on the first `=`).

| Var              | Default                                       | Meaning                                                             |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| `IMAGE`          | `inkling-mini-ciru-halo:v1.6.0`           | Image tag                                                           |
| `CONTAINER_NAME` | `inkling-mini-ciru-halo`                      | Container name                                                      |
| `PORT`           | `8080`                                        | Host publish port (container always listens on 18753)               |
| `MODEL_DIR`      | `./models`                                    | Where the checkpoint lives / downloads to                           |
| `CACHE_DIR`      | `./.cache`                                    | Triton/ROCm/HF caches (persist the first kernel build across restarts) |
| `HF_MODEL`       | `jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151` | HuggingFace model repo                                              |
| `HF_REVISION`    | *(latest)*                                    | Pin a commit SHA for reproducibility                                |
| `LOG_FILE`       | `./.inkling.log`                              | Engine output log                                                   |
| `READY_TIMEOUT`  | `3600`                                        | Seconds to wait for the health endpoint                             |
| `MODE`           | `vision`                                      | `vision`/`concurrent`/`single`/`long-context`/`sampling` (serve.sh preset) |
| `CONTEXT`        | `1048576`                                     | Max prompt + completion tokens; KV pool is sized from it            |
| `SEQ_S`          | *(derive)*                                    | Max concurrent sequences; empty = serve.sh derives (8 vision/concurrent, 1 else) |
| `KV_MIB`         | *(derive)*                                    | BF16 KV budget in MiB; empty = serve.sh sizes it; e.g. `24576` on a tight box |
| `BATCH_TOKENS`   | `4096`                                        | Scheduler `max_num_batched_tokens`                                  |
| `TOOLS`          | `1`                                           | `1` = native Inkling tool calling; `0` = off                        |
| `EXTRA_ARGS`     | *(empty)*                                     | Extra `serve.sh` flags, space-separated (no `=` inside), e.g. `--no-graphs` |
| `DFLASH2`        | `1`                                           | `1` = DFlash 2 drafter (default, separate ~942 MiB checkpoint); `0` = bundled DSpark |
| `DFLASH2_DIR`    | `./models/dflash2`                            | Drafter checkpoint dir; must live under `MODEL_DIR` (rides the ro `/models` mount) |
| `HF_DFLASH2_MODEL` | `incoai/Inkling-2-Mini-DFlash2`             | Drafter HuggingFace repo                                            |
| `HF_DFLASH2_REVISION` | *(latest)*                               | Pin a drafter commit for reproducibility                            |
| `HF_TOKEN`       | *(unset)*                                     | Optional HuggingFace token — higher download rate limits            |

The container **always** listens on `18753` internally (serve.sh's own default, also what the upstream smoke test hard-codes); `start.sh` remaps it to `PORT` with `-p "${PORT}:18753"`.

## How it works

The `Containerfile` builds a self-contained image — no compilation:

1. Ubuntu 24.04 base with runtime libs only (`libnuma1`, `git`/`curl`, `python3` 3.12).
2. `uv` for Python management; `uv venv --python /usr/bin/python3.12` (the ROCm wheels are `cp312`-only; `UV_PYTHON_DOWNLOADS=never` makes a missing 3.12 fail loudly).
3. Pinned ROCm 10 wheels installed from `https://wheels.vllm.ai/rocm/0.34.0/rocm100/`: `vllm==0.34.0+rocm100`, `torch==2.13.0+rocm10.0.0`, `triton==3.8.0+git669b31ac`, plus `huggingface-hub==2.4.0`/`safetensors==0.8.0`. The older `hub==1.31.0`/`safetensors==0.8.0` pair the pinned Transformers source expects is isolated in a separate `--target` deps dir.
4. The CIRU runtime bundle is git-cloned at the pinned `v1.6.0` revision and `scripts/install.sh` runs: it verifies `MANIFEST.sha256`, checks the wheel versions and the `_rocm_C.abi3.so` native extension, fetches the two pinned source revisions (vLLM + Transformers), copies the overlay over the vLLM source tree, symlinks the wheel's native `.so`/`_version.py`/`vllm-rs` into it, and writes the prefix launcher `serve.sh`.
5. At runtime, `entrypoint.sh` turns `INKLING_*` env into `serve.sh` flags, then the generated wrapper sets `PYTHONPATH` (overlay first, then Transformers `src`, then its isolated deps), and the real `serve.sh` sources `rocr-env.sh` which `LD_PRELOAD`s the checksummed patched `libhsa-runtime64.so.1`. `$HOME/.cache` is a symlink to the `/cache` bind-mount so the Triton kernel build survives container recreation. `HF_HUB_OFFLINE=1` is set, so the weights must already be local at `/models`.

The image does **not** contain a system ROCm — the wheels are self-contained and the runtime ships its own patched HSA runtime. The container distro (Ubuntu 24.04) is independent of the host distro; only the kernel and `/dev/kfd` are shared.

## Recommended sampling

Inkling's default speculative path (DFlash 2 when the drafter is present, otherwise
DSpark) is lossless at any temperature, so there is no recommended sampling window
(unlike a GLM-style recipe). `temperature 0` is exact; `temperature > 0` works via
exact rejection sampling. The served model id is `main` in every payload.

Text request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "main",
    "messages": [{"role": "user", "content": "Compute 17*23 and reply with the number only."}],
    "temperature": 0,
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

Reasoning request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "main",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
    "temperature": 0,
    "max_tokens": 512
  }' | jq -r '.choices[0].message.content'
```

Native tool call (`get_weather` tool from the model card):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "main",
    "messages": [{"role": "user", "content": "What is the weather in Austin, TX?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "temperature": 0,
    "max_tokens": 128
  }' | jq -r '.choices[0].message.tool_calls[0].function.name'
# -> get_weather
```

Base64 image request against the default vision profile (native ROCm vision, zero text-turn penalty):

```bash
# one small PNG/JPEG, base64-encoded inline
IMG=$(base64 -w0 example.png)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"main\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG}\"}},
      {\"type\": \"text\", \"text\": \"Describe the main subject and its color in one sentence.\"}
    ]}],
    \"temperature\": 0,
    \"max_tokens\": 256
  }' | jq -r '.choices[0].message.content'
```

Full upstream suite inside the container (hits the engine on its internal port):

```bash
podman exec inkling-mini-ciru-halo /opt/inkling/venv/bin/python \
  /opt/inkling/runtime/scripts/smoke-test.py \
  --base-url http://127.0.0.1:18753/v1 --model main --check-tools
```

## Troubleshooting

**First launch is slow (a few minutes)** — the first request builds a local Triton kernel cache. This only happens on a cold cache; subsequent launches warm up fast (the cache persists in `./.cache` via the `/cache` bind).

**`Cannot allocate memory` / HSA pin errors at startup** — the user session's `memlock` limit is too low for ROCm memory pinning. `start.sh` only warns about this (it does not block). If the engine actually fails to pin, raise it once (needs sudo):

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/10-memlock.conf
sudo systemctl daemon-reload
# then log out and back in
```

**SELinux `Permission denied` on `/models`** — confirm the `:z` suffix on the bind (start.sh sets it). If you relabeled the directory manually, `restorecon -R` it or just re-run `./start.sh`.

**Host OOM at engine init** — the default 1M-token profile sizes a ~52 GiB BF16 KV pool. On a shared or smaller-budget box, set `KV_MIB=24576` and `CONTEXT=32768` in `.env` (the documented smaller-budget launch). `MODE=long-context` + `CONTEXT=66048` is the alternative for single-stream long prompts.

**`hf: command not found`** — install `huggingface_hub` on the host:

```bash
pip install huggingface_hub[cli]
```

Or ensure `python3`'s environment has `huggingface_hub` installed. `start.sh` tries `hf`, then `huggingface-cli`, then a Python one-liner using `huggingface_hub`.

**Container exits immediately** — check logs:

```bash
podman logs inkling-mini-ciru-halo
```

Common causes: missing GPU driver (`/dev/kfd` not present or not in `render`), `memlock` too low, or an incomplete `models/` directory (a missing `config.json`, `mtp.safetensors`, or `optimized-weights/` file).

**`Ciru ROCm: the corrected library could not be loaded`** — the base image drifted off Ubuntu 24.04 (the patched `libhsa-runtime64.so.1` LD_PRELOAD needs the matching glibc). Rebuild from the pinned base image; do not install a system ROCm into the image.

**`missing fitted expert weights for layer 6`** — truncated download. Delete `./models` and re-run `./start.sh` (the marker + `WEIGHTS.sha256` checks should have caught this before launch).

**Multi-GPU host** — set `ROCR_VISIBLE_DEVICES` to expose only the Strix Halo GPU (serve.sh/INSTALL.md guidance).

**Docker instead of Podman** — the same `Containerfile` works under a rootful Docker daemon with `docker run --device /dev/kfd --device /dev/dri --group-add video,render --ipc host --ulimit memlock=-1:-1`. Swap the CLI in `start.sh`/`stop.sh` if you prefer Docker.

## Notes

- **No system ROCm in the image.** The wheels are self-contained and the runtime preloads its own patched HSA runtime; mixing a system ROCm breaks the LD_PRELOAD override. Ubuntu 24.04 is load-bearing for the same reason.
- **`--ipc host` is load-bearing** — the GPU runtime dies at startup without it; no `shm_size` substitutes.
- **`HF_HUB_OFFLINE=1`** is set by `serve.sh`, so the weights must already be on disk at `/models` — the host-side download in `start.sh` is the only path that brings them in.
- **`python3-dev` is required at runtime.** Triton's AMD backend JIT-compiles a small C shim (`hip_utils.c`) with `gcc -I/usr/include/python3.12` on the first kernel launch, so the CPython headers must be installed even though nothing here is precompiled. Strip it and the engine dies at startup with `fatal error: Python.h: No such file or directory`.

## Model speed reference (AMD Strix Halo, Ryzen AI Max+ 395, 128 GB)

Upstream **v1.5.0** baselines (adaptive DSpark, upstream host) from the [model card](https://huggingface.co/jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151):

| Metric                                | Client tok/s | Server tok/s |
| ------------------------------------- | ------------ | ------------ |
| Prefill, 4K prompt                    | 2,299.4      | 2,421.9      |
| Prefill, 8K prompt                    | 2,309.1      | 2,368.0      |
| Prefill, 32K prompt                   | 2,100.8      | 2,117.8      |
| Prefill, 58K prompt                   | 1,904.3      | 1,915.5      |
| Decode, 128K context                  | 46.8         | 50.2         |
| Decode, `B=8` (concurrent)            | 194–208      | —            |
| Cold startup                          | ~65 s        | —            |

Measured **v1.6.0 with DFlash 2** on this host (2026-11-27, default vision profile:
`--dflash2 ./models/dflash2 --context 1048576` with the derived ~52 GiB KV pool,
`--batch-tokens 4096`, `--tools`, FULL target+draft graphs; `vllm bench serve`
against the chat endpoint, random-prompt workloads, seed 42):

| Workload                              | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Acc. len |
| ------------------------------------- | ------------ | ----------- | --------- | --------- | -------- |
| Decode `B=1` `T=1.0` (`3×512t`)       | 57.1         | 73.0        | 191 ms    | 17.2 ms   | 2.74     |
| Decode `B=1` greedy (`3×512t`)        | 61.4         | 78.6        | 135 ms    | 16.1 ms   | 2.87     |
| Decode `B=8` `T=1.0` (`8×512t`)       | 150.7        | 192.8       | 494 ms    | 44.9 ms   | 2.66     |
| Decode `B=8` greedy (`8×512t`)        | 167.6        | 214.4       | 279 ms    | 43.4 ms   | 2.73     |
| Prefill 4K (`2×4096t→1t`)             | 2,546        | 2,546       | 1,608 ms  | —         | —        |
| 128K sustained decode (`127k→512t`)   | 40.0         | 1,320       | 83.9 s    | 25.0 ms   | 2.99     |

- 128K prefill derived from the last row's TTFT: 127,015 t / 83.9 s = **1,514 tok/s**
  (upstream v1.6.0 reference 1,499–1,511 — parity). 4K prefill = input tokens / TTFT.
- Random-token prompts are adversarial for the drafter: acceptance length lands at
  2.7–2.9 here vs ~3.9–4.4 on the natural-text workloads behind the upstream v1.6.0
  targets (`81.1`/`79.7`/`220`/`222` tok/s). Throughput scales with acceptance — the
  128K row (acc 2.99) reproduces the upstream `40.96` tok/s sustained-decode target
  almost exactly, and the same engine decodes the smoke-test math prompt at ~92 tok/s.
- The 4K prefill row was measured against `/v1/completions`: with a 1-token budget the
  chat endpoint emits only a reasoning delta, which the bench client cannot time as TTFT.

## Licence

Apache-2.0. Base model: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — subject to its [acceptable-use policy](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/README.md). Runtime: [ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151). Depends on vLLM, Transformers, Triton, PyTorch and AMD ROCm, each under their own licences.

The CIRU fitted weights live under `optimized-weights/` in the checkpoint and are enabled automatically when present; pass `--original-weights` (via `EXTRA_ARGS`) to use only the original Thinking Machines Lab checkpoint files.

- This recipe: authored and published by [piotreknow02](https://github.com/piotreknow02) — provided as-is, without warranty
