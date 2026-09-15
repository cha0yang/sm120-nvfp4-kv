# NVFP4-KV + MTP Patches

[中文](README.zh.md)

Make `--kv-cache-dtype nvfp4` and MTP speculative decoding work together on
SM120 in SGLang 0.5.19.

## Files

| File | Purpose |
|---|---|
| `patch.sh` | Apply patches (verifies version + file hashes, idempotent) |
| `revert.sh` | Shortcut for `patch.sh --revert` |
| `VERSIONS.txt` | Baseline: sglang version, flashinfer minimum, target file hashes |
| `01-attention-trtllm-mha.patch` | Patch 1 → `sglang/srt/layers/attention/trtllm_mha_backend.py` |
| `02-speculative-draft-utils.patch` | Patch 2 → `sglang/srt/speculative/draft_utils.py` |

## Usage

Run from this directory (`sglang/patches/`). `VENV` defaults to `~/sglang`;
override it (`VENV=~/elsewhere ./patch.sh`) if your virtualenv lives elsewhere.
(The vLLM-side script instead requires `VENV` to be set explicitly.)

```bash
./patch.sh            # verify sglang version + flashinfer version + file hashes → apply
./patch.sh --check    # check only, no changes
./patch.sh --revert   # restore from the .pre-nvfp4kv.bak backups
./patch.sh --force    # apply even if version/hash mismatch (not recommended)
```

Verification performed:

- `sglang.__version__ == VERSIONS.txt:sglang_expected` (currently 0.5.19)
- `flashinfer.__version__ >= VERSIONS.txt:flashinfer_min` (currently 0.6.15)
- each target file's sha256: equals the patched hash → already applied; equals
  the orig hash → applicable; anything else → refused

After applying, both hashes are re-checked against the patched baseline and an
`ast.parse` syntax check runs on each file.

## What patch 1 changes (`trtllm_mha_backend.py`)

### 1a. Let target-verify through the native-FP4 guard

`forward_extend` used to refuse unconditionally:

```python
if self.decode_uses_native_fp4:
    raise RuntimeError("TRTLLM MHA with native FP4 KV cache supports decode only; ...")
```

But spec-decode target-verify and draft-extend run through `forward_extend` and
use the *decode* kernel, so they now pass:

```python
if self.decode_uses_native_fp4 and not (
    forward_batch.forward_mode.is_target_verify()
    or forward_batch.forward_mode.is_draft_extend_v2()
):
    raise RuntimeError(...)
```

Real prefill still raises — NVFP4 prefill must use `flashinfer` (dequant
workspace) or `triton`.

### 1b. Fetch packed FP4 + block scales on the verify path

Only `forward_decode` did this. The verify path got raw packed FP4 from
`get_kv_buffer()` and passed no `kv_cache_sf`, so the kernel read 4-bit storage
as if it were unquantized:

```python
nvfp4_verify = self.is_nvfp4_kvcache and is_decode_mode
if nvfp4_verify:
    kv_cache, kv_cache_block_scales = self._get_nvfp4_decode_kv_cache(layer)
    k_scale, v_scale = self._get_nvfp4_bmm_scales(layer)
    bmm1_scale = q_scale * k_scale * layer.scaling
    bmm2_scale = v_scale
else:
    ... # original path, kv_cache_block_scales = None
```

`kv_cache_sf=kv_cache_block_scales` is then threaded through all three decode
call sites (ENCODER_ONLY verify, ragged verify, fixed-q-len verify) and through
`_run_fixed_q_len_decode`.

### 1c. Build the XQA draft-block mask

XQA asserts `mask is not None` whenever `q_seq_len > 1`. The mask contract
(from `flashinfer/xqa.py` and `csrc/xqa/mha.cu`) is:

```
shape:  [batch_size, q_seq_len, ((q_seq_len + 31) // 32) * 2]
dtype:  torch.uint16   (bit-packed, aligned to 32 bits)
bit i of a row = draft position i is visible
```

That is **the same storage** as `[batch_size, q_seq_len, divUp(q_seq_len, 32)]`
**uint32** words. The helper allocates uint32 words and views as uint16:

```python
def _spec_verify_mask_row(self):
    spec_q = self.speculative_num_draft_tokens
    words = (spec_q + 31) // 32          # <-- uint32 words, NOT *2
    row = torch.zeros(spec_q, words, dtype=torch.int32, device=self.device)
    for r in range(spec_q):
        for bit in range(r + 1):
            row[r, bit // 32] |= 1 << (bit % 32)
    return row.view(torch.uint16).unsqueeze(0)   # uint16 view => words*2
```

> **This is the bug that produced the garbled output.** Allocating
> `words = (spec_q + 31) // 32 * 2` *int32* values doubles the element count;
> `.view(torch.uint16)` then yields 4 uint16 per row instead of 2, shifting every
> mask row and producing the `1, 2, 3, 4, 4, 5, 6, 7, 7, 9, ...` pattern.

The row is batch-independent for chain speculation (`topk == 1`), so it is
broadcast via `.expand(batch_size, -1, -1)` and built lazily on first use.

## What patch 2 changes (`draft_utils.py`)

`speculative_attention_mode == "decode"` used to send **draft-extend** to the
decode backend:

```python
backend_name = (
    "decode_attention_backend"
    if get_spec().speculative_attention_mode == "decode"
    else "prefill_attention_backend"
)
```

But draft-extend is a prefill-shaped pass (EXTEND mode, writes draft KV). With
split backends (`prefill=flashinfer`, `decode=trtllm_mha`) that lands on a
backend with no FP4 prefill kernel and fails during warmup. It now follows the
mode only when both phases share one backend:

```python
prefill_backend_name, decode_backend_name = attention_backends()
if (
    get_spec().speculative_attention_mode == "decode"
    and prefill_backend_name == decode_backend_name
):
    backend_name = "decode_attention_backend"
else:
    backend_name = "prefill_attention_backend"
```

## Required outside the patches (configuration)

With the patches applied the output is correct, but acceptance is still broken. Two flags:

### Mandatory: `--speculative-draft-kv-cache-dtype fp8_e4m3`

If the draft model's (MTP head) KV is stored as NVFP4 too, its read/write path is wrong and
**every draft token is predicted incorrectly**:

| draft KV dtype | accept len | accept rate |
|---|---|---|
| `nvfp4` (inherited default) | 1.10~1.25 | 0.10~0.25 |
| `fp8_e4m3` | 2.90~3.00 | 0.95~1.00 |

No patch fixes this — it is not in `trtllm_mha_backend` but in the draft KV pool's store/fetch
path.

### Recommended: `--speculative-num-steps 2` / `--speculative-num-draft-tokens 3`

K=2. The MTP head is a single layer, so the third draft token only accepts 0.31~0.36 of the
time, which does not pay for its verify cost:

| | K=2 | K=3 |
|---|---|---|
| accept len | 2.4 | 2.91 |
| tokens per round | 3.4 | 3.91 |
| verify rows | 3 | 4 |
| **efficiency tok/verify-row** | **1.13** | 0.98 |

### Use with care: `--enable-linear-replayssm-spec`

Pushes the KV pool from 147k to 166k (`mamba_ratio` 4→3, verify intermediates on a fixed
ring), but drops acceptance from 0.62 to 0.29. Pick per your workload.

## Not needed on this route

- No de-swizzling of V block scales (XQA handles the layout)
- No C++ recompilation
- No FlashInfer kernel changes
- **No code change for the draft KV** — one config flag works around it

## Maintenance

| Event | Action |
|---|---|
| Reinstall / upgrade sglang | Patches are lost → `./patch.sh` (version mismatch is refused; regenerate diffs against the new source if needed) |
| Full rollback | `./patch.sh --revert` |
| Regenerating the patched hash | apply, then `sha256sum <target>` and update `VERSIONS.txt` |
