# 故障排查

[English](TROUBLESHOOTING.md)

NVFP4 KV 特有的问题。工具链和显存预算不是 NVFP4 问题，所以压缩成附录 ——
各引擎的细节见各自的文件：

- vLLM：[`vllm/patches/README.zh.md`](vllm/patches/README.zh.md)
- SGLang：[`sglang/TROUBLESHOOTING.zh.md`](sglang/TROUBLESHOOTING.zh.md)

---

## 共用：两个 NVFP4 特有的问题

| 症状 | 原因 | 修复 |
|---|---|---|
| nvfp4 下输出乱码（仅 vLLM） | 缺 `VLLM_KV_CACHE_LAYOUT=HND`。交错布局（NHD）下 FlashInfer 用 `data_stride/8` 推 SF 步长，导致页条目差 8 倍 | 设 `VLLM_KV_CACHE_LAYOUT=HND`；SGLang 内部处理了布局 |
| 长上下文时 prefill 变慢 | NVFP4 prefill 需要一次反量化，这是固有的，不是配置问题 | 靠 prefix cache（保持会话存活） |

---

## vLLM

| 症状 | 原因 | 修复 |
|---|---|---|
| `FlashInfer XQA speculative decode is not wired` | 补丁缺了上游 #53543 那部分 | 重跑 `./patches/patch.sh` |
| `ValueError: No valid attention backend found ... kv_cache_dtype=nvfp4` | SM90/SM12x 的 decode 判定被一个与 XQA 无关的在线探测挡住 | 补丁解耦了它；若在别的版本复现，临时设 `HTTP(S)_PROXY` |
| `ValueError: Free memory on device ... is less than desired GPU memory utilization` | `--gpu-memory-utilization` 是拿**启动时空闲**显存比对的，不是总量 | 调低（本机 0.98 失败、0.97 可用） |
| MTP 接受率 0% | 检查点有问题（融合层 scale 未对齐） | 换官方 / QAT 验证过的检查点 |

启动参考：[`vllm/README.zh.md`](vllm/README.zh.md) ·
补丁原理：[`vllm/patches/README.zh.md`](vllm/patches/README.zh.md)

---

## SGLang

见 [`sglang/TROUBLESHOOTING.zh.md`](sglang/TROUBLESHOOTING.zh.md)。最关键的一条：
draft head 的 KV dtype 用 NVFP4 时，输出仍然正确，但接受率会塌到 ~0.1 —— 用
`--speculative-draft-kv-cache-dtype fp8_e4m3` 修复。

---

## 附录：环境问题（与 NVFP4 无关）

放在这里是因为两个引擎都需要，但它是 pip CUDA 打包问题，不是 NVFP4 问题。
各引擎的 `bootstrap.sh` 会幂等地应用这些修复。

| 症状 | 原因 | 修复 |
|---|---|---|
| `ld: cannot find -lcudart` / `-lcublas` / `-lcublasLt` | pip `nvidia/cu13` 只提供带版本号的 `.so.13`，而 JIT 链接 `lib64` 但库在 `lib/` | `bootstrap.sh` 补无版本软链和 `lib64 -> lib` |
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.4 混了旧 cudart 头文件 → CCCL 严格检查 | `nvcc-wrapper` 注入 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`，并让 `FLASHINFER_NVCC` 指向它 |
| `RuntimeError: FlashInfer backend is not available`（vLLM） | nvcc 不在 PATH → `has_flashinfer()` 返回 False | `serve.sh` 导出 `PATH` 含 `$CUDA_HOME/bin` |
| `AssertionError` in `deep_ep/__init__.py` `find_cuda_home`（SGLang） | `deep_ep` 需要一个真实的 CUDA 工具链目录 | 导出 `CUDA_HOME` = `site-packages/nvidia/cu13` |
| `custom_all_reduce.cuh:508 invalid argument` | 消费级 GPU 不支持内置 all-reduce kernel | `--disable-custom-all-reduce` |
| CUDA graph capture 时 OOM | capture 一次需要 `max_prefill_tokens` 量的激活 | `--disable-prefill-cuda-graph`（vLLM：调低 `--gpu-memory-utilization`） |
