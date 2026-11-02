#!/usr/bin/env bash
# Language-model experiments (C3, C4, A9).  bash experiments/vldb/run_llm.sh gpt4omini | qwen7b
set -uo pipefail
TAG=${1:?gpt4omini or qwen7b}
case "$TAG" in
  gpt4omini) export LLM_MODEL=${LLM_MODEL:-gpt-4o-mini-2024-07-18}; unset OPENAI_BASE_URL ;;
  qwen7b)    export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}; export OPENAI_BASE_URL=${OPENAI_BASE_URL:?http://<gpu host>:8000/v1}
             export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY} ;;
  *) echo "unknown tag $TAG"; exit 2 ;;
esac
OUT=results_vldb_$TAG; export LLM_CACHE=$OUT/llm_cache.jsonl ARIADNE_REPLAY_SAMPLE=${ARIADNE_REPLAY_SAMPLE:-2000}; mkdir -p "$OUT" logs
bash experiments/vldb/record_env.sh "$OUT"
echo "$LLM_MODEL ${OPENAI_BASE_URL:-openai}" > "$OUT/ENV/model.txt"
if [ "$TAG" = qwen7b ]; then   # the GPU host: its nvidia-smi, vLLM version and the model revision (written by setup_gpu.sh)
  GPU_HOST=${GPU_HOST:?ssh name of the GPU instance, e.g. ubuntu@10.0.1.12}
  scp "$GPU_HOST":nvidia_smi.txt "$OUT/ENV/gpu_nvidia_smi.txt" && scp "$GPU_HOST":vllm_version.txt "$OUT/ENV/vllm_version.txt" \
    && ssh "$GPU_HOST" 'ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/' > "$OUT/ENV/qwen_revision.txt" \
    || { echo "cannot read the GPU host $GPU_HOST"; exit 2; }
fi
L=logs/vldb_llm_$TAG.log
python3 experiments/run.py experiments/vldb/C3_llm_decisions.yaml --out "$OUT" 2>&1 | tee -a $L
# C4 always answers and judges with GPT-4o mini on OpenAI, whatever model built the memory
( unset OPENAI_BASE_URL; export OPENAI_API_KEY=${OPENAI_API_KEY_OPENAI:-$OPENAI_API_KEY}
  python3 experiments/run.py experiments/vldb/C4_downstream_qa.yaml --out "$OUT" 2>&1 | tee -a $L )
# A9 (Mem0 through its public API, GPT-4o mini): only in the GPT-4o mini directory
if [ "$TAG" = gpt4omini ]; then
  python3 experiments/run.py experiments/A9_memory_frameworks.yaml --out "$OUT" 2>&1 | tee -a $L
fi
