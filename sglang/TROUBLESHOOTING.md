# Troubleshooting

[中文](TROUBLESHOOTING.zh.md)

NVFP4-KV-specific failures, in the order you hit them. For toolchain, CUDA
symlinks and memory budgeting, see the [shared file](../TROUBLESHOOTING.md).

---

## 1. Bring-up: the four patch targets

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: Mask is required for speculative decoding` (`flashinfer/xqa.py:418`) | SGLang never passes the XQA draft mask for `q_seq_len > 1` | `patches/01-attention-trtllm-mha.patch` |
| `RuntimeError: TRTLLM MHA with native FP4 KV cache supports decode only` | Raised from `forward_extend`, which target-verify also goes through | same patch |
| Output degenerates: `1, 2, 3, 4, 4, 5, 6, 7, 7, 9, 10, 10, 12, 12, ...` | Mask allocated as `divUp(qSeqLen,32)*2` **int32** words → element count doubled → every row shifted | same patch |
| Garbled verify text (`We- : /: . . /. /.`), acceptance ~0 | Verify path read raw packed FP4 and passed no `kv_cache_sf` | same patch |
| `RuntimeError: ... draft_extend ... TRTLLM MHA ... decode only` during warmup | `speculative_attention_mode=decode` routed draft-extend to the decode backend | `patches/02-speculative-draft-utils.patch` |

---

## 2. The silent one: correct output, broken acceptance

**Symptom**: output is perfectly correct, but decode is slow.

This is the failure mode that does not announce itself. Every draft token is
wrong, but the verify cost is still paid, so you only see it in the throughput.

**Diagnosis** — use a highly predictable prompt and read the acceptance counter:

```bash
curl -s -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user",
       "content":"Repeat exactly 30 times, nothing else: A B C D E F G H."}],
       "max_tokens":300,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' >/dev/null

curl -s http://localhost:30000/metrics | grep spec_accept_rate | grep 'tp_rank="0"'
```

In a repeating pattern the next token is essentially certain, so acceptance
should be **near 1.0**. If it is 0.1~0.4, the draft path is broken — it is not
"the model being inaccurate".

**Root cause**: the draft head's own KV cache (1 layer) inherits
`--kv-cache-dtype nvfp4`, and that read/write path is incorrect. Measured:

| draft KV dtype | accept len | accept rate |
|---|---|---|
| `nvfp4` (inherited, default) | 1.10~1.25 | **0.10~0.25** |
| `fp8_e4m3` | 2.90~3.00 | **0.95~1.00** |

**Fix**:

```bash
--speculative-draft-kv-cache-dtype fp8_e4m3
```

Only the draft head's own KV is affected (1 layer, negligible memory). The target
model's KV stays NVFP4, so the capacity win is preserved.

---

## 3. Watching the NVFP4 path

With `--enable-metrics --enable-metrics-for-all-schedulers`:

```bash
curl -s http://localhost:30000/metrics | grep -E "spec_accept" | grep 'tp_rank="0"'
```

Or the decode log line (`--decode-log-interval`):

```
Decode batch, #running-req: 1, accept len: 1.57, accept rate: 0.29,
  cuda graph: True, gen throughput (token/s): 42.78, #queue-req: 0
```

| Metric | Healthy | Broken |
|---|---|---|
| accept rate (predictable text) | 0.9~1.0 | < 0.4 → the draft KV dtype bug above |
| accept rate (open-ended text) | 0.5~0.7 | |
| accept len | ≈ 1 + accept_rate × steps | |
