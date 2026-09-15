# NVFP4 KV Cache on SM120 GPUs

Enable **NVFP4 KV cache** on consumer Blackwell (SM120) GPUs — in **two engines**:

| | [vLLM](vllm/) | [SGLang](sglang/) |
|---|---|---|
| KV cache dtype | `nvfp4` (XQA decode + fa2 prefill) | `nvfp4` (XQA decode + flashinfer prefill) |
| Patch scope | 2 Python files | 2 Python files |
| Extra requirement | `VLLM_KV_CACHE_LAYOUT=HND` | `--speculative-draft-kv-cache-dtype fp8_e4m3` |
| MTP supported | ✅ (K=2) | ✅ (K=2) |
| Vision | ✅ | ✅ |
| Measured KV pool | **292,103 tokens** | 193,472 tokens (both 2×16 GB) |
| Status | mature | works; upstream fix pending |

> Both rows are on `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, with vision on and
> SSM state at the model's own `float32`. vLLM reaches a larger pool because it
> reserves the mamba state once at profile time, while SGLang has to carve it out
> of the same budget through `--mamba-full-memory-ratio`.

[中文](README.zh.md)

---

## Contents

- [1. What This Repo Does](#1-what-this-repo-does)
- [2. Which Engine Should I Use](#2-which-engine-should-i-use)
- [3. Quick Start](#3-quick-start)
- [4. Benchmarks](#4-benchmarks)
- [5. Other](#5-other)
  - [Troubleshooting](#troubleshooting)
  - [Related Work](#related-work)
  - [Acknowledgments](#acknowledgments)

---

## 1. What This Repo Does

- **Problem**: both engines gate or mis-wire NVFP4 KV cache on SM120. vLLM's
  `--kv-cache-dtype nvfp4` is refused outright; SGLang's works alone but breaks
  as soon as speculative decoding is enabled. Meanwhile FlashInfer's XQA nvfp4
  kernels are **SM120-specific** (a dedicated hardware path) — leaving them
  unused wastes the silicon.
- **Solution**: small Python patches per engine plus a few environment/config
  flags. No C++ recompilation, no FlashInfer kernel changes.

### Requirements (shared)

| Item | Value | Notes |
|---|---|---|
| GPU | SM120 | RTX 50 series / RTX PRO 6000 Blackwell; XQA nvfp4 kernels target SM120 |
| flashinfer-python | **≥ 0.6.15** | Needs the strided-SF fix; older versions misread nvfp4 KV bytes |
| torch | 2.13.x | ships with the engines below |
| CUDA toolkit | pip `nvidia/cu13` | nvcc 13.4 + headers 13.2 mixed; an `nvcc-wrapper` relaxes the CCCL check |

Per-engine pins (line numbers are locked, `patch.sh` refuses mismatches):

| | vLLM | SGLang |
|---|---|---|
| engine version | **0.29.0** | **0.5.19** |
| Python | **3.14** (verified here) | **3.12** (verified here) |
| patch targets | `vllm/v1/attention/backends/flashinfer.py`, `vllm/utils/flashinfer.py` | `sglang/srt/layers/attention/trtllm_mha_backend.py`, `sglang/srt/speculative/draft_utils.py` |

> Each engine lives in its own venv, so their Python versions are independent.
> The scripts never hardcode a Python version — they resolve `site-packages`
> from `$VENV` at runtime.

---

## 2. Which Engine Should I Use

Both reach **NVFP4 KV cache with MTP speculative decoding**. They differ in how
much you have to change and what you get:

| | vLLM | SGLang |
|---|---|---|
| Patch burden | 2 files + **mandatory** `VLLM_KV_CACHE_LAYOUT=HND` env var | 2 files |
| Layout handling | Requires HND (**garbage output without it**) | Internal (NHD pool, HND view for decode) |
| Prefill path | fa2 (nvfp4-aware) | flashinfer + FP8 dequant workspace |
| Decode path | XQA | XQA (`trtllm_mha`) |
| Draft KV dtype | inherits nvfp4, works | **must be `fp8_e4m3`** — nvfp4 silently kills acceptance |
| Config complexity | lower | higher (mamba ratio, backend split, spec mode) |
| Extra capability | — | 147k KV pool, vision verified end-to-end |

**Rule of thumb**

- Want the smaller change set, or already run vLLM → **vLLM**
- Want a longer context / hybrid-GDN (Qwen3.5/3.8) tuning knobs → **SGLang**
- Either way, read the engine's `TROUBLESHOOTING` — the failure modes are
  different and not obvious.

---

## 3. Quick Start

### vLLM

```bash
export VENV=~/vllm13
cd /path/to/sm120-nvfp4-kv/vllm

./bootstrap.sh                    # environment self-check + fixes
./patches/patch.sh                # verify version + hashes, then apply
./run.sh
```

Details: [`vllm/README.md`](vllm/README.md) · patch internals: [`vllm/patches/README.md`](vllm/patches/README.md)

### SGLang

```bash
export VENV=~/sglang
cd /path/to/sm120-nvfp4-kv/sglang

./bootstrap.sh                    # environment self-check + fixes
./patches/patch.sh                # verify version + hashes, then apply
./run.sh
```

Details: [`sglang/README.md`](sglang/README.md) · patch internals: [`sglang/patches/README.md`](sglang/patches/README.md)

---

## 4. Benchmarks

> Environment: 2× RTX 5060 Ti 16GB (SM120), TP=2, single stream,
> `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, 30 s per decode cell.

### Prefill (single stream)

| Context | vLLM tok/s | vLLM TTFT (s) | SGLang tok/s | SGLang TTFT (s) |
|---|---|---|---|---|
| 8k | 4,501 | 1.82 | 4,183 | 1.96 |
| 16k | 3,941 | 4.12 | 3,925 | 4.13 |
| 32k | 3,356 | 9.62 | 3,233 | 9.99 |
| 64k | 2,510 | 25.66 | 2,383 | 27.03 |
| 128k | 1,679 | 76.64 | 1,541 | 83.54 |

### Decode (single stream)

| Context | vLLM tok/s | vLLM accept len | SGLang tok/s | SGLang accept len |
|---|---|---|---|---|
| 0 | 69.1 | 2.32 | 58.4 | 2.17 |
| 16k | 68.3 | 2.32 | 58.6 | 2.22 |
| 32k | 67.9 | 2.35 | 54.1 | 2.08 |
| 64k | 64.0 | 2.29 | 56.4 | 2.23 |
| 128k | 60.9 | 2.31 | 52.6 | 2.20 |

---

## 5. Other

### Troubleshooting

[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) starts with the issues **both engines**
share (toolchain / CUDA symlinks, SM120 + TP=2, memory budgeting), then has a
section per engine. SGLang's acceptance-collapse and mamba-sizing detail lives in
[`sglang/TROUBLESHOOTING.md`](sglang/TROUBLESHOOTING.md).

### Related Work

There are currently **three routes** to NVFP4 KV cache on SM120, sharing the same
root cause (under an interleaved layout, FlashInfer derives the SF stride from
`data_stride/8`, making page entries off by 8×) but fixing it at different
layers:

| | [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | This repo → vLLM | This repo → SGLang |
|---|---|---|---|
| Scope | FlashInfer **CUDA kernels** + jit utils + vLLM python = 4 files (incl. C++) | **2 Python files** + `VLLM_KV_CACHE_LAYOUT=HND` | **2 Python files** |
| Core trick | Kernels accept an **explicit SF stride**; V-SF un-swizzled in registers | **HND layout** keeps each page's K/V sides contiguous | Split backends + build the XQA draft mask + draft KV dtype |
| Decode backend | FA2 (arch-generic) | XQA + fa2 prefill | `trtllm_mha` (XQA) + `flashinfer` prefill |
| Version anchor | vLLM `0.1.dev16944` + FlashInfer `0.6.11.post2` | vLLM **0.29.0** + FlashInfer **0.6.18** | SGLang **0.5.19** + FlashInfer **0.6.18** |
| Layout support | NHD / HND both | Requires HND | Internal |
| MTP | Compatible | Compatible (K=2) | Compatible (K=2) |
| Measured KV pool (2×16 GB) | N/A (kernel-level) | 292,103 | 193,472 |

**Takeaway**: for the broadest applicability (any layout / mixed precision),
hikari's kernel route is the most thorough. For minimal changes and easy
rollback, the two routes here are lighter — and the SGLang one is the smallest,
because SGLang already ships the NVFP4 KV machinery; only the
speculative-decoding wiring was missing.

### Acknowledgments

This project builds on a lot of prior work. **It introduces no new algorithms or
kernels** — NVFP4 KV working on SM120 comes entirely from FlashInfer, vLLM and
SGLang themselves, and the direction comes from the upstream community
([vLLM issue #49011](https://github.com/vllm-project/vllm/issues/49011)). All we
did was combine existing findings and document the process:

1. Inspired by the upstream gating approach (#49818) and
   [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)'s
   layout analysis, we found that `VLLM_KV_CACHE_LAYOUT=HND` makes the store and
   read paths agree naturally, removing the need for un-swizzling and kernel
   changes. This is another take on the same root cause, not a new discovery.
2. Backported a small change from upstream
   [vLLM #53543](https://github.com/vllm-project/vllm/pull/53543) so MTP and
   NVFP4 KV can be used together in vLLM.
3. For SGLang, located and fixed four independent wiring gaps (native-FP4 guard,
   missing `kv_cache_sf`, missing XQA draft mask, draft-extend backend routing),
   plus one configuration trap (draft KV dtype collapsing acceptance). The XQA
   mask contract came from reading FlashInfer's `csrc/xqa/mha.cu`; #53543
   independently proved the same conclusion on the vLLM side.
4. Engineered the process and documented it (verifiable patch scripts, rollback,
   self-checks) so others can reproduce it.

If upstream merges official fixes, this repo can be retired.

**Upstream code and projects**

| Project | Purpose |
|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Inference engine; the vLLM patches target `vllm/v1/attention/backends/flashinfer.py` and `vllm/utils/flashinfer.py` |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Inference engine; the SGLang patches target `trtllm_mha_backend.py` and `draft_utils.py` |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | XQA decode / fa2 prefill NVFP4-KV kernels, NVFP4 scales, draft mask contract |
| [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | The most important reference: the FA2 kernel-patch route |
| [MiaAI-Lab/exllamav3](https://github.com/MiaAI-Lab/exllamav3) | Reference for its Triton paged-attention on-the-fly dequant approach |

**Key issues / PRs**

Main thread: [vLLM issue #49011 — nvfp4 KV cache on SM120](https://github.com/vllm-project/vllm/issues/49011)

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
| [vLLM #49718](https://github.com/vllm-project/vllm/pull/49718) | FlashInfer XQA decode on SM12x | Merged |
| [vLLM #49818](https://github.com/vllm-project/vllm/pull/49818) | Enable NVFP4 KV cache on SM120 | Closed; our vLLM gating approach is based on it |
| [vLLM #50085](https://github.com/vllm-project/vllm/pull/50085) | Write linear V block scales on SM120/SM121 | Closed; we bypass it with the HND layout |
| [vLLM #53543](https://github.com/vllm-project/vllm/pull/53543) | Enable masked NVFP4 XQA on SM120 | Open; we backport its core ~30 lines (vLLM) and it independently confirms the SGLang mask gap |
| [vLLM #53681](https://github.com/vllm-project/vllm/pull/53681) | NVFP4 KV cache: fix block_size/layout detection | Open; reference |

---

*Licensed under `LICENSE` (MIT); third-party notices in `NOTICE`.*
