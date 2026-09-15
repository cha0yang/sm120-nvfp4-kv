# SGLang：SM120 上的 NVFP4 KV Cache + MTP 投机解码

[English](README.md) · [← 项目总入口](../README.zh.md)

让 SM120 上的 SGLang 同时用上 **NVFP4 KV cache** 与 **MTP 投机解码**。

- **问题**：SGLang 0.5.19 本身已支持 SM120 上的 `--kv-cache-dtype nvfp4`
  （prefill 走 `flashinfer` 的 dequant workspace，decode 走 `trtllm_mha` / XQA），
  但与**投机解码同时开启**时链路断在四个地方，服务要么直接崩，要么输出乱码。
- **方案**：2 个 Python 文件补丁 + 一个配置开关。不需要重编译 C++，
  不需要改 FlashInfer 内核。

> 共用环境要求（GPU、Python、flashinfer、torch、CUDA 工具链）与跨引擎对比见
> [顶层 README](../README.zh.md)。本文件只讲 SGLang。

## 需要修什么

SGLang 的 FP4-KV 路径在打开投机解码后有四处断开，都在两个 Python 文件里：

| # | 位置 | 问题 |
|---|---|---|
| 1 | `trtllm_mha_backend.py` | `forward_extend` 拒绝 native FP4，但投机解码的 target-verify 正是走这里 |
| 2 | 同上 | verify 路径取到裸 packed FP4，且没传 `kv_cache_sf` |
| 3 | 同上 | XQA 在 `q_seq_len > 1` 时需要 draft-block mask；SGLang 从没构造过 |
| 4 | `draft_utils.py` | `speculative_attention_mode=decode` 把 draft-extend 也路由到了 decode backend，而它是 prefill 语义 |

表现是 `AssertionError: Mask is required for speculative decoding`，或者输出退化成
重复/跳号。

第五处是配置而非代码：**只有 target 的 KV 该是 NVFP4**。draft head 也用 NVFP4 时
接受率会崩到 ~0.1：

```bash
--speculative-draft-kv-cache-dtype fp8_e4m3
```

补丁细节（包括 #3 背后的 mask 步长 bug）：[`patches/README.zh.md`](patches/README.zh.md)。

---

## 操作方法

```bash
./bootstrap.sh        # CUDA 软链 / nvcc-wrapper 自检
./patches/patch.sh    # 打补丁（校验 sglang 版本 + 文件哈希，幂等）
./run.sh              # kill + serve，日志写 /tmp/serve.log（首次需 JIT，约 10-20 分钟）
```

`VENV` 默认 `~/sglang`。其余脚本：`serve.sh`（只启动）、`kill.sh`（杀 worker 并打印显存）、
`bootstrap-env.sh`（被 `bootstrap.sh` 和 `serve.sh` 共用的修复库）。

### 关键启动参数

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

| 参数 | 原因 |
|---|---|
| **`--speculative-draft-kv-cache-dtype fp8_e4m3`** | **draft KV 用 NVFP4 时接受率塌到 ~0.1** |
| `--speculative-attention-mode decode` | 默认 `prefill` 时 target-verify 走 `trtllm_fmha_v2_prefill`，那条路完全不支持 NVFP4 KV |
| `--prefill-attention-backend` / `--decode-attention-backend` | decode 走 XQA（`trtllm_mha`），prefill 留在 `flashinfer` |
| `--disable-custom-all-reduce` | SM120 + TP=2 不支持内置的 all-reduce kernel |
| `--disable-prefill-cuda-graph` | 16GB 卡上避免 prefill graph capture 时 OOM |
| `--enable-cache-report` | 不开的话 `usage.prompt_tokens_details` 永远是 `null`，客户端无法显示 prefix cache 命中率 |

[`serve.sh`](serve.sh) 其余部分是 mamba 显存配比、采样默认值、parser 和 metrics ——
不影响 NVFP4 路径。

---

## 注意事项

- 补丁打在 `site-packages` 上，重装即失效 —— 重跑 `./patches/patch.sh`。
- draft KV dtype 弄错时输出仍是正确的，只是接受率下降。要用 `/metrics` 的
  `spec_accept_rate` 判断，而不是读输出。见
  [`TROUBLESHOOTING.zh.md`](TROUBLESHOOTING.zh.md#投机解码接受率decode-慢的主因)。

---

## 故障排查

[`TROUBLESHOOTING.zh.md`](TROUBLESHOOTING.zh.md)（SGLang 特有）·
[共用问题](../TROUBLESHOOTING.zh.md)

---

*本仓库许可见 `LICENSE`（MIT）；第三方代码说明见 `NOTICE`。*
