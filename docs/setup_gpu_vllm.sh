#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv_vllm/bin/python" ]]; then
  python3.12 -m venv .venv_vllm
fi

source .venv_vllm/bin/activate

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -U pip >/dev/null

python -c "import vllm; import sys; sys.exit(0 if vllm.__version__=='0.15.1' else 1)" >/dev/null 2>&1 || \
  python -m pip install "vllm==0.15.1"

exec env VLLM_LOGGING_LEVEL=INFO python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name local-qwen \
  --dtype auto --max-model-len 10240