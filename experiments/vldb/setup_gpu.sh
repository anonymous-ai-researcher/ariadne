#!/usr/bin/env bash
# GPU instance for the open model: AWS g6e.2xlarge, 1x NVIDIA L40S 48 GB, Ubuntu 24.04 Deep Learning AMI (driver and CUDA preinstalled).
# Run as the login user; leaves the server in the foreground (use tmux).  Port 8000 must be open only to the machine that runs run_llm.sh (security group).
set -euo pipefail
python3 -m venv ~/vllm && . ~/vllm/bin/activate && pip install -U pip && pip install vllm
python -c "import vllm; print('vllm', vllm.__version__)" | tee ~/vllm_version.txt
nvidia-smi | tee ~/nvidia_smi.txt
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-model-len 8192 --max-num-seqs 128 \
     --gpu-memory-utilization 0.90 --seed 0 --host 0.0.0.0 --port 8000 2>&1 | tee ~/vllm_serve.log
# the model revision: ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/
