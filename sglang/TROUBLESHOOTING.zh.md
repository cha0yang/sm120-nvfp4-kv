# 故障排查

[English](TROUBLESHOOTING.md)

NVFP4 KV 相关的故障，按遇到顺序排列。工具链、CUDA 软链和显存预算见
[共用文件](../TROUBLESHOOTING.zh.md)。

---

## 1. 启动阶段：四个补丁目标

| 症状 | 原因 | 修复 |
|---|---|---|
| `AssertionError: Mask is required for speculative decoding`（`flashinfer/xqa.py:418`） | SGLang 从没为 `q_seq_len > 1` 传 XQA draft mask | `patches/01-attention-trtllm-mha.patch` |
| `RuntimeError: TRTLLM MHA with native FP4 KV cache supports decode only` | 抛自 `forward_extend`，而 target-verify 也走这里 | 同上 |
| 输出退化成 `1, 2, 3, 4, 4, 5, 6, 7, 7, 9, 10, 10, 12, 12, ...` | mask 按 `divUp(qSeqLen,32)*2` 个 **int32** 字分配 → 元素数翻倍 → 每行错位 | 同上 |
| verify 输出乱码（`We- : /: . . /. /.`），接受率 ~0 | verify 路径读到裸 packed FP4，且没传 `kv_cache_sf` | 同上 |
| warmup 时 `RuntimeError: ... draft_extend ... TRTLLM MHA ... decode only` | `speculative_attention_mode=decode` 把 draft-extend 路由到了 decode backend | `patches/02-speculative-draft-utils.patch` |

---

## 2. 最隐蔽的那个：输出正确，但接受率崩了

**症状**：输出完全正确，但 decode 很慢。

这个故障不会自己暴露 —— draft token 全错，但 verify 开销照付，只体现在吞吐上。

**诊断** —— 用高度可预测的 prompt，读接受率：

```bash
curl -s -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user",
       "content":"Repeat exactly 30 times, nothing else: A B C D E F G H."}],
       "max_tokens":300,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' >/dev/null

curl -s http://localhost:30000/metrics | grep spec_accept_rate | grep 'tp_rank="0"'
```

重复模式下下一个 token 几乎是确定的，所以接受率应当**接近 1.0**。
若只有 0.1~0.4，说明 draft 路径坏了 —— 不是"模型不准"。

**根因**：draft head（1 层）自己的 KV cache 继承了 `--kv-cache-dtype nvfp4`，
而那条读写路径是错的。实测：

| draft KV dtype | accept len | accept rate |
|---|---|---|
| `nvfp4`（默认继承） | 1.10~1.25 | **0.10~0.25** |
| `fp8_e4m3` | 2.90~3.00 | **0.95~1.00** |

**修复**：

```bash
--speculative-draft-kv-cache-dtype fp8_e4m3
```

只影响 draft head 自己的 KV（1 层，显存开销可忽略），target 的 KV 仍是 NVFP4，
容量优势保留。

---

## 3. 观察 NVFP4 路径

需要 `--enable-metrics --enable-metrics-for-all-schedulers`：

```bash
curl -s http://localhost:30000/metrics | grep -E "spec_accept" | grep 'tp_rank="0"'
```

或者看 decode 日志行（`--decode-log-interval`）：

```
Decode batch, #running-req: 1, accept len: 1.57, accept rate: 0.29,
  cuda graph: True, gen throughput (token/s): 42.78, #queue-req: 0
```

| 指标 | 健康 | 异常 |
|---|---|---|
| 接受率（可预测文本） | 0.9~1.0 | < 0.4 → 上面那个 draft KV dtype bug |
| 接受率（开放文本） | 0.5~0.7 | |
| accept len | ≈ 1 + accept_rate × steps | |
