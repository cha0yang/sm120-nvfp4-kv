# NVFP4-KV Patch

Enables **NVFP4 KV cache** on consumer Blackwell (SM120) in vLLM (XQA decode + fa2 prefill).
This addresses upstream issue [#49011](https://github.com/vllm-project/vllm/issues/49011);
upstream PRs #49818 / #50085 remain unmerged.

[中文版](README.zh.md)

## Files

| File | Purpose |
|---|---|
| `patch.sh` | Installs the patch (verifies version + file hashes; idempotent) |
| `revert.sh` | Shortcut for `patch.sh --revert` |
| `VERSIONS.txt` | Patch baseline: vLLM version, minimum FlashInfer version, target file hashes |
| `01-attention-flashinfer.patch` | Patch 1 → `vllm/v1/attention/backends/flashinfer.py` |
| `02-utils-flashinfer.patch` | Patch 2 → `vllm/utils/flashinfer.py` |

## Usage

Run from this directory (`vllm/patches/`):

```bash
./patch.sh            # verify vLLM version + FlashInfer version + file hashes, then patch
./patch.sh --check    # check only, change nothing
./patch.sh --revert   # revert (restores from pre-patch .bak files)
./patch.sh --force    # apply even if version/hashes mismatch (not recommended)
```

`VENV` must point at the vLLM virtualenv; the script exits with an error if it is unset
(it never guesses, so a wrong environment cannot be patched by accident).

What gets verified:
- vLLM version == `vllm_expected` in `VERSIONS.txt` (currently 0.29.0)
- flashinfer-python >= `flashinfer_min` (currently 0.6.15)
- sha256 of each target file: matches patched hash → already applied; matches orig hash → not yet applied; anything else → refuse

## Two Critical Points (both required)

1. **`VLLM_KV_CACHE_LAYOUT=HND`** (environment variable, set in `serve.sh`)
   Under the default NHD layout, each page is laid out as `[K0,K1,V0,V1]`, alternating per token.
   But the store kernel and `nvfp4_split_data_scale` both assume "all data first, then all scale,
   per side". HND (head-major) makes each side contiguous, so the two agree exactly.
   **Without this, the model emits garbage.**

2. **`kv_data_type` must be `torch.uint8`** (not the string `"nvfp4"`)
   FlashInfer uses `getattr(torch, "nvfp4")` internally, which does not exist in torch 2.13.

## Patch Scope

**2 files** in total:

1. `vllm/v1/attention/backends/flashinfer.py`
   SM120 gating (allow nvfp4 KV), dtype plumbing (`kv_data_type` as `torch.uint8`),
   prefill via fa2 / decode via auto (xqa), skipping the FP8 staging buffer,
   plus a backport of the core ~30 lines from upstream
   [#53543](https://github.com/vllm-project/vllm/pull/53543) (dedicated XQA path passing
   nvfp4 data + `kv_cache_sf`) → makes MTP usable.

2. `vllm/utils/flashinfer.py`
   Stops the SM90/SM12x decode decision from being **blocked by the "artifactory" online check**
   (that check only matters for the SM100 path that downloads trtllm-gen FHMA cubins;
   XQA uses local kernels/JIT).

## What You Do NOT Need With This Route

- No extra V-scale de-swizzling (the fa2/XQA kernels already support the SM100 swizzled layout)
- No C++ recompilation (the `.cu` changes in upstream PR #50085 are not required)

## Maintenance

| Event | Action |
|---|---|
| Upgrade / reinstall vLLM | Patch is lost → run `./patch.sh` (a version mismatch is rejected; regenerate the diff against the new source) |
| Full rollback | `./patch.sh --revert` |
