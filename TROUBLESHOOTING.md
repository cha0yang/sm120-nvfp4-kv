# Troubleshooting

[中文](TROUBLESHOOTING.zh.md)

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: FlashInfer backend is not available` | nvcc not in PATH → `has_flashinfer()` returns False | Make sure `serve.sh` has `export PATH="$CUDA_HOME/bin:$PATH"` (and that CUDA_HOME and PATH are exported on separate lines) |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.4 vs cudart headers 13.2 | Use `FLASHINFER_NVCC=bin/nvcc-wrapper` |
| `ptxas: Unsupported .version 9.4` | Half-upgraded toolchain | Unify everything on 13.4 + wrapper to relax checks |
| `ld: cannot find -lcudart` | pip package has no dev symlink | `bootstrap.sh` adds the symlink |
| `ld: cannot find -lcublas` | pip package has no unversioned symlink | `bootstrap.sh` adds the symlink |
| `custom_all_reduce.cuh:164 'invalid argument'` | Consumer GPUs don't support vLLM's built-in AR kernels | `--disable-custom-all-reduce` |
| MTP acceptance rate 0% | Bad checkpoint (fused-layer scale misalignment) | Switch to an official/QAT-verified checkpoint |
| `FlashInfer XQA speculative decode is not wired` | Patch is missing the #53543 part | Re-run `./patches/patch.sh` |
| Garbage output (under nvfp4) | Missing `VLLM_KV_CACHE_LAYOUT=HND` | Add it |
| `ValueError: No valid attention backend found ... kv_cache_dtype=nvfp4` | Online probe timed out → gating returns False | This patch decouples it; if it reappears on another version, temporarily set HTTP(S)_PROXY |
| OOM at startup | Context/activation over budget | Lower `--max-model-len` or `--gpu-memory-utilization` |
| Leftover `VLLM::Worker` after kill | SIGTERM doesn't kill it | Use `kill.sh` (has a SIGKILL fallback) |
| Long prompts are slow | prefill is O(n^1.3); inherent to cold start | Rely on prefix caching (keep sessions alive) |
