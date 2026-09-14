# NVFP4 KV Cache for vLLM on SM120 GPUs

Enable **NVFP4 KV cache** on consumer Blackwell (SM120) GPUs in vLLM.

[中文](README.zh.md)

---

## Contents

- [1. What This Repo Does](#1-what-this-repo-does)
- [2. How to Use](#2-how-to-use)
- [3. Benchmarks](#3-benchmarks)
- [4. Other](#4-other)
  - [Troubleshooting](#troubleshooting)
  - [Related Work](#related-work)
  - [Acknowledgments](#acknowledgments)

---

## 1. What This Repo Does

- **Problem**: vLLM gates `--kv-cache-dtype nvfp4` on SM120, and the upstream PRs remain unmerged.
  Meanwhile FlashInfer's XQA nvfp4 kernels are **SM120-specific** (a dedicated hardware path) —
  leaving them unused wastes the silicon.
- **Solution**: 2 Python file patches + 1 environment variable (`VLLM_KV_CACHE_LAYOUT=HND`).
  No C++ recompilation, no FlashInfer kernel changes.

### Requirements

| Item | Value | Notes |
|---|---|---|
| GPU | SM120 | RTX 50 series / RTX PRO 6000 Blackwell; XQA nvfp4 kernels target SM120 |
| Python | 3.14 | venv |
| vllm | **0.29.0** | Patch line numbers are locked; `patch.sh` rejects other versions |
| flashinfer-python | **≥ 0.6.15** | Needs the strided-SF fix; older versions misread nvfp4 KV bytes |
| torch | 2.13.0+cu132 | Ships with vllm 0.29.0 |
| CUDA toolkit | pip `nvidia/cu13` | nvcc 13.4 + headers 13.2 mixed; `nvcc-wrapper` relaxes strict checks |

Critical version constraints (must be met, or patches/kernels will not work):

- `nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-crt` = **13.4.59** (all three must match; partial installs fail with ptxas errors)
- `nvidia-cuda-runtime` = **13.2.75** (mixed with nvcc 13.4; `nvcc-wrapper` injects `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`)
- To fix versions: `uv pip install --no-deps nvidia-cuda-nvcc==13.4.59 nvidia-nvvm==13.4.59 nvidia-cuda-crt==13.4.59`

---

## 2. How to Use

```bash
# 0) Point VENV at your target virtualenv
export VENV=~/vllm

# 1) Create venv and install
python3 -m venv "$VENV" && source "$VENV/bin/activate"
pip install uv
uv pip install vllm==0.29.0 --torch-backend auto

# 2) Environment self-check + fixes (CUDA symlinks, nvcc-wrapper)
./bootstrap.sh

# 3) Apply the NVFP4-KV patch (verifies version + file hashes)
./patches/patch.sh

# 4) Run
./examples/run.sh
```

> Patch details: see `patches/README.md`

---

## 3. Benchmarks

> Environment: 2× RTX 5060 Ti 16GB (SM120), `nvidia/Qwen3.8-27B-NVFP4`, TP=2.
> Full config: see [`examples/README.md`](examples/README.md).

| Context length | prefill speed | prefill time | decode speed |
|---|---|---|---|
| 8k | 4,102 tok/s | 2.0s | 72.3 tok/s |
| 16k | 3,619 tok/s | 4.5s | 69.9 tok/s |
| 32k | 3,106 tok/s | 10.4s | 69.4 tok/s |
| 64k | 2,375 tok/s | 27.1s | 67.6 tok/s |
| **128k** | **1,617 tok/s** | **79.6s** | **62.4 tok/s** |

## 4. Other

### Troubleshooting

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

### Related Work

There are currently **two independent routes** to working NVFP4 KV cache on SM120.
Both share the same root cause (under vLLM's interleaved layout, FlashInfer derives the
SF stride from `data_stride/8`, making page entries off by 8×), but they fix it at
different layers:

| | [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) (FA2 kernel patch) | **This project** (Python gating + HND layout) |
|---|---|---|
| Scope | FlashInfer **CUDA kernels** + jit utils + vLLM python = 4 files (incl. C++) | **2 Python files** + 1 env var |
| Core trick | Kernels accept an **explicit SF stride**; V-SF is **un-swizzled element-wise in registers** | **`VLLM_KV_CACHE_LAYOUT=HND`** keeps each page's K/V sides contiguous |
| Decode backend | FA2 (arch-generic) | XQA decode + fa2 prefill |
| Version anchor | vLLM `0.1.dev16944` + FlashInfer `0.6.11.post2` | vLLM **0.29.0** + FlashInfer **0.6.18** |
| Layout support | NHD / HND both | Requires HND (**garbage output without it**) |
| Capacity | 1.78× fp8 (theoretical ceiling) | ~1.7× measured |
| MTP | Compatible (verified at K=1) | Compatible (K=2, backports upstream #53543) |

**Takeaway**: for broader applicability (any layout / mixed precision), hikari's kernel
route is more thorough; for **minimal changes, easy rollback, and keeping up with new
vLLM releases**, this project is lighter.

### Acknowledgments

This project builds on a lot of prior work. **It introduces no new algorithms or kernels** —
NVFP4 KV working on SM120 comes entirely from FlashInfer and vLLM themselves, and the
direction comes from the upstream community ([issue #49011](https://github.com/vllm-project/vllm/issues/49011)).
All we did was combine existing findings and document the process:

1. Inspired by the upstream gating approach (#49818) and
   [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)'s
   layout analysis, we found that `VLLM_KV_CACHE_LAYOUT=HND` makes the store and read paths
   agree naturally, removing the need for un-swizzling and kernel changes. This is another
   take on the same root cause, not a new discovery.
2. Backported a small change from upstream [#53543](https://github.com/vllm-project/vllm/pull/53543)
   so MTP and NVFP4 KV can be used together.
3. Engineered the process and documented it (verifiable patch scripts, rollback, self-checks)
   so others can reproduce it.

If upstream merges an official fix, this repo can be retired.

**Upstream code and projects**

| Project | Purpose |
|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | The inference engine itself; our patches target its `vllm/v1/attention/backends/flashinfer.py` |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Provides XQA decode / fa2 prefill NVFP4-KV kernels |
| [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | The most important reference: the FA2 kernel-patch route |
| [MiaAI-Lab/exllamav3](https://github.com/MiaAI-Lab/exllamav3) | Reference for its Triton paged-attention on-the-fly dequant approach |

**Key issues / PRs**

Main thread: [vllm issue #49011 — nvfp4 KV cache on SM120](https://github.com/vllm-project/vllm/issues/49011)

| Contributor | Contribution |
|---|---|
| **@0xdespot** | Opened issue #49011; first working prototype on RTX 5090; author of PR #49818 |
| **@gtrak** | First to raise 2×5060 Ti support; independently reproduced on 2×RTX 5060 Ti |
| **@seanyourhighness** | Found the concurrency crash; proved #44455 is the causal misread; found the V block-scale write defect |
| **@heungwing** | Independent verification on RTX PRO 6000 + WSL2 |
| **@stevenmoto** | Reproduced on 4× RTX 6000 Pro Blackwell |
| **@gaby** | Pointed out the same failure class on B200/B300 |

Related PRs:

| PR | Content | Status |
|---|---|---|
| [#49718](https://github.com/vllm-project/vllm/pull/49718) | FlashInfer XQA decode on SM12x | Merged |
| [#49818](https://github.com/vllm-project/vllm/pull/49818) | Enable NVFP4 KV cache on SM120 | Closed; our gating approach is based on it |
| [#50085](https://github.com/vllm-project/vllm/pull/50085) | Write linear V block scales on SM120/SM121 | Closed; we bypass it with the HND layout |
| [#53543](https://github.com/vllm-project/vllm/pull/53543) | Enable masked NVFP4 XQA on SM120 | Open; we backport its core ~30 lines |
| [#53681](https://github.com/vllm-project/vllm/pull/53681) | NVFP4 KV cache: fix block_size/layout detection | Open; reference |

---

*Licensed under `LICENSE` (MIT); third-party notices in `NOTICE`.*
