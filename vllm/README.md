# vLLM: NVFP4 KV Cache on SM120

Example deployment: 2× RTX 5060 Ti + Qwen3.8-27B-NVFP4

[中文版](README.zh.md) · [← project root](../README.md)

> Shared requirements and the vLLM-vs-SGLang comparison live in the
> [top-level README](../README.md). This file covers vLLM only.
> Patch internals: [`patches/README.md`](patches/README.md).

## Environment

| Item | Value |
|---|---|
| GPU | 2× RTX 5060 Ti 16GB |
| CPU | AMD Ryzen 7 9700X |
| Memory | 16GB DDR5 6000MHz × 2 (30Gi usable) |
| Motherboard | MAXSUN B850 AIGA |
| OS | Ubuntu 26.04.1 LTS (Resolute Raccoon) |
| Python | 3.14 venv (path given via `VENV`) |
| vllm | 0.29.0 |
| torch | 2.13.0+cu132 |
| flashinfer-python | 0.6.18 |
| CUDA toolkit | pip `nvidia/cu13` (nvcc 13.4 + headers 13.2 mixed; `nvcc-wrapper` relaxes strict checks) |

## Deployment

```bash
# 0) Point VENV at your target virtualenv
export VENV=~/vllm13

# 1) Create venv
python3 -m venv "$VENV" && source "$VENV/bin/activate"
pip install uv
uv pip install vllm==0.29.0 --torch-backend auto

# 2) Environment self-check + fixes (CUDA symlinks, nvcc-wrapper)
./bootstrap.sh

# 3) Apply the NVFP4-KV patch (verifies version + hashes)
./patches/patch.sh

# 4) Start the server (first run needs JIT compilation, ~10–20 min)
./run.sh
```

## Key Config (`serve.sh`)

| Flag | Value | Notes |
|---|---|---|
| `--tensor-parallel-size` | 2 | Dual GPU |
| `--kv-cache-dtype` | nvfp4 | The whole point |
| (no `--max-model-len`) | 262,144 | Left unset, so it takes the model's own limit. KV pool 292,103 tokens (1.11× concurrency, measured) |
| `--max-num-seqs` | 1 | Single concurrent request |
| `--gpu-memory-utilization` | 0.97 | Display on iGPU, so give all VRAM to vLLM |
| `--performance-mode` | interactivity | Single-user low latency |
| `--cudagraph-capture-sizes` | 3 | = seqs×(K+1) = 1×3 |
| `--speculative-config` | mtp K=2 | Speculative decoding |
| `--override-generation-config` | T=0.6 / top_p .8 / top_k 20 / presence_penalty 1.5 | Qwen3.8 recommended Instruct sampling params |
| `--default-chat-template-kwargs` | enable_thinking:false | Thinking off by default |
