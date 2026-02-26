python3.12 -m venv .venv_vllm
source .venv_vllm/bin/activate
python -m pip install -U pip
python -m pip install vllm==0.15.1

# run git pull, bash scripts/setup_gpu_vllm.sh