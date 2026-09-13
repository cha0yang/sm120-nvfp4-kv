# 故障排查

[English](TROUBLESHOOTING.md)

| 症状 | 原因 | 处理 |
|---|---|---|
| `RuntimeError: FlashInfer backend is not available` | nvcc 不在 PATH → `has_flashinfer()` False | 确认 serve.sh 有 `export PATH="$CUDA_HOME/bin:$PATH"`（且 CUDA_HOME 与 PATH 分两行 export） |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.4 vs cudart headers 13.2 | 用 `FLASHINFER_NVCC=bin/nvcc-wrapper` |
| `ptxas: Unsupported .version 9.4` | 工具链半新半旧 | 全套统一 13.4 + wrapper 关检查 |
| `ld: 找不到 -lcudart` | pip 包无 dev symlink | `bootstrap.sh` 补软链 |
| `ld: 找不到 -lcublas` | pip 包无 unversioned symlink | `bootstrap.sh` 补软链 |
| `custom_all_reduce.cuh:164 'invalid argument'` | 消费卡不支持 vLLM 自带 AR 内核 | `--disable-custom-all-reduce` |
| MTP 接受率 0% | checkpoint 坏（fused 层 scale 错位） | 换官方/QAT 验证过的检查点 |
| `FlashInfer XQA speculative decode is not wired` | 补丁没含 #53543 部分 | 重新 `./patches/patch.sh` |
| 输出乱码（nvfp4 下） | 少了 `VLLM_KV_CACHE_LAYOUT=HND` | 补上 |
| `ValueError: No valid attention backend found ... kv_cache_dtype=nvfp4` | 联网探测超时 → 门控 False | 本补丁已解耦；若换版本复现，临时给 HTTP(S)_PROXY |
| 启动 OOM | 上下文/激活超预算 | 降 `--max-model-len` 或 `--gpu-memory-utilization` |
| kill 后残留 `VLLM::Worker` | SIGTERM 杀不死 | 用 `kill.sh`（内部 SIGKILL 兜底） |
| 长 prompt 很慢 | prefill 是 O(n^1.3)，冷启动固有 | 靠 prefix cache（保持会话） |
