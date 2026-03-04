#!/usr/bin/env bash
set -euo pipefail

# ===== KVCompression vLLM server launcher (robust) =====
# Goals:
# - avoid bricking the instance (no exec+pipes, conservative defaults)
# - minimize repeated installs
# - keep caches off $HOME
# - make logs + PID easy to inspect
# - bind to localhost by default (safer in managed notebook GPU envs)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ----- Configurable knobs -----
: "${VLLM_HOST:=127.0.0.1}"          # safer than 0.0.0.0 in managed envs
: "${VLLM_PORT:=8000}"
: "${MODEL_ID:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${SERVED_MODEL_NAME:=local-qwen}"
: "${VLLM_VERSION:=0.15.1}"
: "${MAX_MODEL_LEN:=10240}"
: "${DTYPE:=auto}"
: "${VLLM_EAGER:=1}"                 # 1 => --enforce-eager
: "${VLLM_VENV_DIR:=/tmp/.venv_vllm}"

# Caches (keep heavy IO off $HOME)
: "${HF_HOME:=/tmp/hf_cache}"
: "${VLLM_CACHE_DIR:=/tmp/vllm_cache}"
: "${TORCHINDUCTOR_CACHE_DIR:=/tmp/torchinductor_cache}"
: "${TRITON_CACHE_DIR:=/tmp/triton_cache}"
: "${XDG_CACHE_HOME:=/tmp/xdg_cache}"

# Logs / pid
: "${LOG_DIR:=/tmp/vllm}"
: "${LOG_FILE:=$LOG_DIR/vllm_server.log}"
: "${PID_FILE:=$LOG_DIR/vllm_server.pid}"

mkdir -p "$LOG_DIR" "$HF_HOME" "$VLLM_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HOME VLLM_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR XDG_CACHE_HOME
export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

VENV="$VLLM_VENV_DIR"
PY="$VENV/bin/python"

# ----- Helpers -----
die() { echo "ERROR: $*" >&2; exit 2; }

cleanup_on_error() {
  echo "Launcher failed. Last 120 log lines:" >&2
  tail -n 120 "$LOG_FILE" 2>/dev/null || true
}
trap cleanup_on_error ERR

stop_existing() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping existing vLLM server (pid=$pid)..."
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}

# ----- Quick environment sanity -----
echo "== Env =="
echo "ROOT=$ROOT"
echo "VENV=$VENV"
echo "HOST=$VLLM_HOST PORT=$VLLM_PORT"
echo "MODEL_ID=$MODEL_ID SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN DTYPE=$DTYPE EAGER=$VLLM_EAGER"
echo "LOG_FILE=$LOG_FILE"
echo

# Optional: show GPU visibility if available (won't fail script)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo

# ----- Create venv if missing -----
if [[ ! -x "$PY" ]]; then
  echo "Creating venv at $VENV ..."
  uv venv "$VENV" --python 3.12
fi

# ----- Install vLLM only if needed -----
if ! "$PY" -c "import vllm,sys; sys.exit(0 if getattr(vllm,'__version__','')=='$VLLM_VERSION' else 1)" >/dev/null 2>&1; then
  echo "Installing vLLM==$VLLM_VERSION into $VENV ..."
  uv pip install --python "$PY" -U pip
  uv pip install --python "$PY" "vllm==$VLLM_VERSION"
fi

# ----- Build args -----
ARGS=(
  -m vllm.entrypoints.openai.api_server
  --host "$VLLM_HOST" --port "$VLLM_PORT"
  --model "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$VLLM_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

# ----- Stop old server (if any) -----
stop_existing

# ----- Start server (no pipes; log redirected safely) -----
echo "Starting vLLM server..."
echo "Command: $PY ${ARGS[*]}"
echo "Logging to: $LOG_FILE"
echo

# Append header to log
{
  echo "================================================================================"
  date -Is
  echo "Command: $PY ${ARGS[*]}"
  echo "HF_HOME=$HF_HOME"
  echo "VLLM_CACHE_DIR=$VLLM_CACHE_DIR"
  echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "================================================================================"
} >> "$LOG_FILE"

# Run in background so your shell returns (and avoids session supervisors killing it)
# If your environment requires foreground, set VLLM_FOREGROUND=1
: "${VLLM_FOREGROUND:=0}"

if [[ "$VLLM_FOREGROUND" == "1" ]]; then
  VLLM_LOGGING_LEVEL=INFO "$PY" "${ARGS[@]}" >>"$LOG_FILE" 2>&1
else
  nohup env VLLM_LOGGING_LEVEL=INFO "$PY" "${ARGS[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
  echo $! > "$PID_FILE"
  echo "vLLM started (pid=$(cat "$PID_FILE"))."
fi

# ----- Quick health check -----
# We keep it lightweight and non-fatal (some envs lack curl).
echo
echo "Health check (best-effort):"
if command -v curl >/dev/null 2>&1; then
  for _ in {1..30}; do
    if curl -s "http://$VLLM_HOST:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
      echo "OK: /v1/models reachable"
      exit 0
    fi
    sleep 1
  done
  echo "WARN: /v1/models not reachable yet. Check logs:"
  echo "  tail -n 120 $LOG_FILE"
  exit 0
else
  echo "curl not available; skip. Check logs:"
  echo "  tail -n 120 $LOG_FILE"
  exit 0
fi