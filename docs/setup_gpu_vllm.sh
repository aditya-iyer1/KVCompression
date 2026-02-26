#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/tmp/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"

EXTRA_ARGS=()
if [[ "${VLLM_EAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

uv venv .venv_vllm --python 3.12

uv pip -p .venv_vllm install -U pip >/dev/null
uv pip -p .venv_vllm install "vllm==0.15.1" >/dev/null

exec env VLLM_LOGGING_LEVEL=INFO .venv_vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name local-qwen \
  --dtype auto --max-model-len 10240 \
  "${EXTRA_ARGS[@]}"