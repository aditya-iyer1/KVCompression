#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep heavy IO off /home to avoid D-state stalls
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/tmp/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

VENV="${VLLM_VENV_DIR:-/tmp/.venv_vllm}"
PY="$VENV/bin/python"
LOG="${VLLM_LOG:-/tmp/vllm_server.log}"

mkdir -p "$(dirname "$LOG")"

if [[ ! -x "$PY" ]]; then
  uv venv "$VENV" --python 3.12
fi

# Install only if missing/wrong
if ! "$PY" -c "import vllm,sys; sys.exit(0 if vllm.__version__=='0.15.1' else 1)" >/dev/null 2>&1; then
  uv pip install --python "$PY" -U pip
  uv pip install --python "$PY" "vllm==0.15.1"
fi

EXTRA_ARGS=()
if [[ "${VLLM_EAGER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

echo "Starting vLLM (logs: $LOG)"
exec env VLLM_LOGGING_LEVEL=INFO "$PY" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name local-qwen \
  --dtype auto --max-model-len 10240 \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG"