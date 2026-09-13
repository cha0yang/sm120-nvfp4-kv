#!/usr/bin/env bash
set -euo pipefail

# --- 限制启动期编译并发, 保护主机内存 ---
# FlashInfer/Torch JIT 用 ninja, MAX_JOBS = 每个 worker 的 nvcc 并行数。
# TP=2 -> 2 个 worker, 故实际 nvcc 进程数 = MAX_JOBS * 2。
export MAX_JOBS="${MAX_JOBS:-2}"
# TorchInductor 每个编译任务只开 1 线程, 避免多任务叠加吃内存
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# 限制底层 OpenMP 线程数
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# 运行环境 (venv 根目录) — 必须显式指定, 不设默认值
if [[ -z "${VENV:-}" ]]; then
  echo "!! 未设置 VENV。请指向 vLLM 虚拟环境, 例如:" >&2
  echo "     VENV=~/vllm ./serve.sh" >&2
  exit 1
fi
VENV="$(readlink -f "$VENV")"
[[ -x "$VENV/bin/python3" ]] || { echo "!! 找不到 $VENV/bin/python3" >&2; exit 1; }
# site-packages 动态取, 不写死 python 版本
SITE="$("$VENV/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
# 找到 nvcc, 避免 deep_gemm 等模块找不到 CUDA 报 warning
export CUDA_HOME="${CUDA_HOME:-$SITE/nvidia/cu13}"
# nvcc 必须在 PATH 里, 否则 vllm 的 has_flashinfer() 返回 False, XQA decode 会炸
export PATH="$CUDA_HOME/bin:$PATH"
# 工具链是 nvcc 13.4 + cudart headers 13.2 (同主版本兼容), wrapper 关掉 CCCL 严格版本检查
export FLASHINFER_NVCC="$VENV/bin/nvcc-wrapper"
# HF 离线: 只用本地缓存, 不联网
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 避免碎片化 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SM120 NVFP4 KV: HND(head-major)布局让每页 K/V side 连续, 匹配 store/读取
export VLLM_KV_CACHE_LAYOUT=HND

# 激活 venv (需要 bash 可用 source)
source "$VENV/bin/activate"

vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --override-generation-config '{"temperature": 0.6, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}' \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 200000 \
  --max-num-seqs 1 \
  --performance-mode interactivity \
  --cudagraph-capture-sizes 3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --trust-remote-code \
  "$@"
