# Troubleshooting

[中文](TROUBLESHOOTING.zh.md)

NVFP4-KV-specific issues. Toolchain and memory budgeting are not NVFP4 problems,
so they are kept short here — engine-specific detail lives in each engine's own
file:

- vLLM: [`vllm/patches/README.md`](vllm/patches/README.md)
- SGLang: [`sglang/TROUBLESHOOTING.md`](sglang/TROUBLESHOOTING.md)

---

## Shared: the two NVFP4-specific ones

| Symptom | Cause | Fix |
|---|---|---|
| Garbage output under nvfp4 (vLLM only) | Missing `VLLM_KV_CACHE_LAYOUT=HND`. Under the interleaved (NHD) layout FlashInfer derives the SF stride from `data_stride/8`, making page entries off by 8× | Set `VLLM_KV_CACHE_LAYOUT=HND`; SGLang handles the layout internally |
| Prefill slows down at long context | NVFP4 prefill needs a dequant pass; this is inherent, not a config problem | Rely on prefix caching (keep sessions alive) |

---

## vLLM

| Symptom | Cause | Fix |
|---|---|---|
| `FlashInfer XQA speculative decode is not wired` | Patch is missing the upstream #53543 part | Re-run `./patches/patch.sh` |
| `ValueError: No valid attention backend found ... kv_cache_dtype=nvfp4` | The SM90/SM12x decode decision is blocked by an online probe that has nothing to do with XQA | The patch decouples it; if it reappears on another version, temporarily set `HTTP(S)_PROXY` |
| `ValueError: Free memory on device ... is less than desired GPU memory utilization` | `--gpu-memory-utilization` is checked against free memory at startup, not total | Lower it (0.98 fails here, 0.97 works) |
| MTP acceptance rate 0% | Bad checkpoint (fused-layer scale misalignment) | Switch to an official/QAT-verified checkpoint |

Launch reference: [`vllm/README.md`](vllm/README.md) ·
patch internals: [`vllm/patches/README.md`](vllm/patches/README.md)

---

## SGLang

See [`sglang/TROUBLESHOOTING.md`](sglang/TROUBLESHOOTING.md). The headline one:
with NVFP4 as the draft head's KV dtype, output stays correct but acceptance
collapses to ~0.1 — fix with `--speculative-draft-kv-cache-dtype fp8_e4m3`.

---

## Appendix: environment (not NVFP4)

Kept here because both engines need it, but it is a pip-CUDA packaging issue, not
an NVFP4 one. Each engine's `bootstrap.sh` applies these idempotently.

| Symptom | Cause | Fix |
|---|---|---|
| `ld: cannot find -lcudart` / `-lcublas` / `-lcublasLt` | pip `nvidia/cu13` ships only versioned `.so.13`, and JIT links against `lib64` while libs live in `lib/` | `bootstrap.sh` adds the unversioned symlinks and `lib64 -> lib` |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.4 mixed with older cudart headers → CCCL strict check | `nvcc-wrapper` injects `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`; point `FLASHINFER_NVCC` at it |
| `RuntimeError: FlashInfer backend is not available` (vLLM) | nvcc not in PATH → `has_flashinfer()` False | `serve.sh` exports `PATH` with `$CUDA_HOME/bin` |
| `AssertionError` in `deep_ep/__init__.py` `find_cuda_home` (SGLang) | `deep_ep` needs a real CUDA toolchain dir | Export `CUDA_HOME` at `site-packages/nvidia/cu13` |
| `custom_all_reduce.cuh:508 invalid argument` | Consumer GPUs don't support the built-in all-reduce kernels | `--disable-custom-all-reduce` |
| OOM during CUDA graph capture | Capture needs `max_prefill_tokens` worth of activation at once | `--disable-prefill-cuda-graph` (vLLM: lower `--gpu-memory-utilization`) |
