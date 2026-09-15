#!/usr/bin/env bash
set -euo pipefail

# --- 运行环境 (venv 根目录) ---
# VENV 未指定时用 ~/sglang; 也可显式: VENV=~/sglang ./serve.sh
VENV="$(readlink -f "${VENV:-$HOME/sglang}")"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# venv 里 python / python3 只保证存在其一 (bootstrap-env.sh 也会再探一次)
if [[ ! -x "$VENV/bin/python" && ! -x "$VENV/bin/python3" ]]; then
  echo "!! 找不到 $VENV/bin/python{,3}" >&2
  echo "   请设置 VENV 指向 SGLang 虚拟环境, 例如: VENV=~/sglang $0" >&2
  exit 1
fi
export VENV

# --- 环境修复 (CUDA 软链 + nvcc-wrapper + deep_ep CUDA_HOME) ---
# 逻辑集中在 bootstrap.sh 里 (幂等), 这里只负责调用与 export
source "$D/bootstrap-env.sh"

# --- 离线: 只用本地缓存 ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 避免碎片化 OOM ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- extra_buffer 策略下省一个 mamba state slot ---
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1

# --- 编译并发保护 ---
export MAX_JOBS="${MAX_JOBS:-2}"

source "$VENV/bin/activate"

exec sglang serve \
  --trust-remote-code \
  --model-path QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --kv-cache-dtype nvfp4 \
  --mem-fraction-static 0.98 \
  --attention-backend flashinfer \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend trtllm_mha \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --speculative-algorithm EAGLE \
  --speculative-attention-mode decode \
  --speculative-num-steps 2 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disable-prefill-cuda-graph \
  --enable-cache-report \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults openai \
  --preferred-sampling-params '{"temperature": 0.6, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --mamba-full-memory-ratio 0.8 \
  --max-mamba-cache-size 6 \
  --chunked-prefill-size 2048 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype float32 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --decode-log-interval 20 \
  --tp 2 \
  "$@"
