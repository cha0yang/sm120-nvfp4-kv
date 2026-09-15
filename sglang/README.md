# SGLang: NVFP4 KV Cache + MTP Speculative Decoding on SM120

[中文](README.zh.md) · [← 项目总入口](../README.md)

Enable **NVFP4 KV cache together with MTP speculative decoding** in SGLang on
consumer Blackwell (SM120) GPUs.

- **Problem**: SGLang 0.5.19 already supports `--kv-cache-dtype nvfp4` on SM120
  (prefill = `flashinfer` with a dequant workspace, decode = `trtllm_mha` / XQA).
  Combining it with **speculative decoding** is broken in four separate places,
  so the server either aborts or emits garbled text.
- **Solution**: 2 Python file patches, plus one configuration flag. No C++
  recompilation, no FlashInfer kernel changes.

> Shared requirements (GPU, Python, flashinfer, torch, CUDA toolkit) and the
> cross-engine comparison live in the [top-level README](../README.md).
> This file covers SGLang only.

## The four defects

SGLang's FP4-KV path breaks in four places once speculative decoding is enabled,
all in two Python files:

| # | Where | Problem |
|---|---|---|
| 1 | `trtllm_mha_backend.py` | `forward_extend` rejects native FP4, but spec-decode target-verify runs through it |
| 2 | same | the verify path fetched packed FP4 without passing `kv_cache_sf` |
| 3 | same | XQA needs a draft-block mask for `q_seq_len > 1`; SGLang never built it |
| 4 | `draft_utils.py` | `speculative_attention_mode=decode` also routed draft-extend to the decode backend, which is prefill-shaped |

Result: either `AssertionError: Mask is required for speculative decoding`, or
text that degenerates into repeated/skipped tokens.

A fifth gap is configuration, not code. **Only the target's KV should be NVFP4** —
with NVFP4 as the draft head's dtype, acceptance collapses to ~0.1:

```bash
--speculative-draft-kv-cache-dtype fp8_e4m3
```

Patch internals (including the mask-stride bug behind #3):
[`patches/README.md`](patches/README.md).

---

## How to Use

```bash
./bootstrap.sh        # CUDA symlinks / nvcc-wrapper self-check
./patches/patch.sh    # apply (verifies sglang version + file hashes, idempotent)
./run.sh              # kill + serve, log to /tmp/serve.log (first run: JIT, ~10-20 min)
```

`VENV` defaults to `~/sglang`. Other scripts: `serve.sh` (launch only),
`kill.sh` (kill workers + print VRAM), `bootstrap-env.sh` (the fix library
sourced by both `bootstrap.sh` and `serve.sh`).

### Launch flags that matter

```bash
--kv-cache-dtype nvfp4 \
--prefill-attention-backend flashinfer \
--decode-attention-backend trtllm_mha \
--speculative-algorithm EAGLE \
--speculative-attention-mode decode \
--speculative-draft-kv-cache-dtype fp8_e4m3 \
--speculative-num-steps 2 \
--speculative-num-draft-tokens 3 \
--disable-custom-all-reduce \
--disable-prefill-cuda-graph \
--enable-cache-report
```

| Flag | Why |
|---|---|
| **`--speculative-draft-kv-cache-dtype fp8_e4m3`** | **With NVFP4 draft KV, acceptance collapses to ~0.1** |
| `--speculative-attention-mode decode` | With the default `prefill`, target-verify runs `trtllm_fmha_v2_prefill`, which has no NVFP4 KV support at all |
| `--prefill-attention-backend` / `--decode-attention-backend` | Decode goes to XQA (`trtllm_mha`); prefill stays on `flashinfer` |
| `--disable-custom-all-reduce` | SM120 + TP=2 does not support the custom all-reduce kernels |
| `--disable-prefill-cuda-graph` | Avoids OOM during prefill graph capture on 16 GB cards |
| `--enable-cache-report` | Without it `usage.prompt_tokens_details` is always `null`, so clients cannot show prefix-cache hit rate |

The rest of [`serve.sh`](serve.sh) is mamba sizing, sampling defaults, parsers
and metrics — nothing that affects the NVFP4 path.

---

## Notes

- The patch lives in `site-packages`, so a reinstall wipes it — re-run
  `./patches/patch.sh`.
- A wrong draft-KV dtype still produces correct text; only acceptance drops.
  Check it with `/metrics` (`spec_accept_rate`), not by reading the output. See
  [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md#speculative-acceptance-the-main-cause-of-slow-decode).

---

## Troubleshooting

[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) (SGLang-specific) ·
[shared issues](../TROUBLESHOOTING.md)

---

*Licensed under the repository's `LICENSE` (MIT); third-party notices in `NOTICE`.*
