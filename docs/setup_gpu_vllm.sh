#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- cache locations (safe for big downloads) ----
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/tmp/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"

export PYTHONUNBUFFERED=1

# ---- reuse existing environment ----
VENV="${VLLM_VENV_DIR:-$HOME/.venv_vllm}"

if [[ ! -d "$VENV" ]]; then
    echo "Creating venv at $VENV"
    uv venv "$VENV" --python 3.12
fi

source "$VENV/bin/activate"

# ---- install vllm only if missing ----
if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "Installing vLLM (first time only)"
    uv pip install -U pip
    uv pip install vllm==0.15.1
fi

LOG=/tmp/vllm_server.log

echo "Starting vLLM server..."
echo "Logs: $LOG"

python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port 8000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --served-model-name local-qwen \
    --dtype auto \
    --max-model-len 10240 \
    --enforce-eager \
    > "$LOG" 2>&1