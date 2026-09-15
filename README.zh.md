# SM120 GPU 上的 NVFP4 KV Cache

让消费级 Blackwell (SM120) GPU 用上 **NVFP4 KV cache** —— **两种引擎**各有方案：

| | [vLLM](vllm/) | [SGLang](sglang/) |
|---|---|---|
| KV dtype | `nvfp4`（XQA decode + fa2 prefill） | `nvfp4`（XQA decode + flashinfer prefill） |
| 改动面 | 2 个 Python 文件 | 2 个 Python 文件 |
| 额外要求 | `VLLM_KV_CACHE_LAYOUT=HND` | `--speculative-draft-kv-cache-dtype fp8_e4m3` |
| MTP | ✅（K=2） | ✅（K=2） |
| 视觉 | ✅ | ✅ |
| 实测 KV 池 | **292,103 tokens** | 193,472 tokens（同为 2×16GB） |
| 状态 | 成熟 | 可用，上游修复未合并 |

> 两行都是 `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`，开视觉，SSM 状态用模型自带的
> `float32`。vLLM 的池子更大，是因为它在 profiling 时一次性预留下 mamba state，
> 而 SGLang 得通过 `--mamba-full-memory-ratio` 从同一笔预算里划。

[English](README.md)

---

## 目录

- [1. 本仓库做什么](#1-本仓库做什么)
- [2. 该选哪个引擎](#2-该选哪个引擎)
- [3. 快速开始](#3-快速开始)
- [4. 实测对比](#4-实测对比)
- [5. 其余内容](#5-其余内容)
  - [故障排查](#故障排查)
  - [相关工作与实现对比](#相关工作与实现对比)
  - [致谢与溯源](#致谢与溯源)

---

## 1. 本仓库做什么

- **问题**：两个引擎在 SM120 上都做不好 NVFP4 KV cache。vLLM 直接门控拒绝
  `--kv-cache-dtype nvfp4`；SGLang 单独能跑，但一旦开启投机解码就断链。
  而 FlashInfer 的 XQA nvfp4 内核**是 SM120 专属**的（硬件路径），不用等于浪费。
- **方案**：每个引擎 2 个 Python 文件补丁 + 少量环境/配置开关。
  不需要重编译 C++，不需要改 FlashInfer 内核。

### 共用运行环境要求

| 项 | 值 | 说明 |
|---|---|---|
| GPU | SM120 | RTX 50 系 / RTX PRO 6000 Blackwell；XQA nvfp4 内核是 SM120 专属 |
| flashinfer-python | **≥ 0.6.15** | strided SF 修复，低于此版本 nvfp4 KV 读错字节 |
| torch | 2.13.x | 跟下面两个引擎配套 |
| CUDA 工具链 | pip `nvidia/cu13` | nvcc 13.4 + headers 13.2 混装，靠 `nvcc-wrapper` 关严格检查 |

各引擎的版本锚定（补丁行号锁定，版本不符 `patch.sh` 会拒绝）：

| | vLLM | SGLang |
|---|---|---|
| 引擎版本 | **0.29.0** | **0.5.19** |
| Python | **3.14**（本机验证） | **3.12**（本机验证） |
| 补丁目标 | `vllm/v1/attention/backends/flashinfer.py`、`vllm/utils/flashinfer.py` | `sglang/srt/layers/attention/trtllm_mha_backend.py`、`sglang/srt/speculative/draft_utils.py` |

> 两个引擎各自独立 venv，Python 版本互不相干。脚本不写死 Python 版本 ——
> 运行时从 `$VENV` 解析 `site-packages`。

---

## 2. 该选哪个引擎

两者都能达到 **NVFP4 KV cache + MTP 投机解码**，差别在改动量和额外收益：

| | vLLM | SGLang |
|---|---|---|
| 补丁负担 | 2 文件 + **必须**设 `VLLM_KV_CACHE_LAYOUT=HND` | 2 文件 |
| 布局处理 | 必须 HND（**缺了会乱码**） | 内部处理（pool 存 NHD，decode 时 HND 视图） |
| prefill 路径 | fa2（nvfp4-aware） | flashinfer + FP8 dequant workspace |
| decode 路径 | XQA | XQA（`trtllm_mha`） |
| draft KV dtype | 继承 nvfp4 可用 | **必须显式 `fp8_e4m3`** —— 继承 nvfp4 会静默毁掉接受率 |
| 配置复杂度 | 较低 | 较高（mamba ratio、后端拆分、投机模式） |
| 额外收益 | — | 147k KV 池，视觉能力完整验证 |

**选择建议**

- 想要改动最少，或本来就在用 vLLM → **vLLM**
- 想要更长上下文 / 需要 hybrid GDN（Qwen3.5/3.8）的调优旋钮 → **SGLang**
- 无论哪边，都要读对应引擎的 `TROUBLESHOOTING` —— 两边的坑完全不同且不直观

---

## 3. 快速开始

### vLLM

```bash
export VENV=~/vllm13
cd /path/to/sm120-nvfp4-kv/vllm

./bootstrap.sh                    # 环境自检 + 修复
./patches/patch.sh                # 校验版本 + 哈希后打补丁
./run.sh
```

细节见 [`vllm/README.zh.md`](vllm/README.zh.md) · 补丁原理见 [`vllm/patches/README.zh.md`](vllm/patches/README.zh.md)

### SGLang

```bash
export VENV=~/sglang
cd /path/to/sm120-nvfp4-kv/sglang

./bootstrap.sh                    # 环境自检 + 修复
./patches/patch.sh                # 校验版本 + 哈希后打补丁
./run.sh
```

细节见 [`sglang/README.zh.md`](sglang/README.zh.md) · 补丁原理见 [`sglang/patches/README.zh.md`](sglang/patches/README.zh.md)

---

## 4. 实测对比

> 环境：2× RTX 5060 Ti 16GB (SM120)，TP=2，单并发，
> `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`，每个 decode 单元 30 秒。

### Prefill（单并发）

| 上下文 | vLLM tok/s | vLLM TTFT (s) | SGLang tok/s | SGLang TTFT (s) |
|---|---|---|---|---|
| 8k | 4,501 | 1.82 | 4,183 | 1.96 |
| 16k | 3,941 | 4.12 | 3,925 | 4.13 |
| 32k | 3,356 | 9.62 | 3,233 | 9.99 |
| 64k | 2,510 | 25.66 | 2,383 | 27.03 |
| 128k | 1,679 | 76.64 | 1,541 | 83.54 |

### Decode（单并发）

| 上下文 | vLLM tok/s | vLLM accept len | SGLang tok/s | SGLang accept len |
|---|---|---|---|---|
| 0 | 69.1 | 2.32 | 58.4 | 2.17 |
| 16k | 68.3 | 2.32 | 58.6 | 2.22 |
| 32k | 67.9 | 2.35 | 54.1 | 2.08 |
| 64k | 64.0 | 2.29 | 56.4 | 2.23 |
| 128k | 60.9 | 2.31 | 52.6 | 2.20 |

---

## 5. 其余内容

### 故障排查

[`TROUBLESHOOTING.zh.md`](TROUBLESHOOTING.zh.md) 先列**两个引擎共用**的问题
（工具链 / CUDA 软链、SM120 + TP=2、显存预算），然后每个引擎一节。
SGLang 的接受率崩塌与 mamba 配比细节在
[`sglang/TROUBLESHOOTING.zh.md`](sglang/TROUBLESHOOTING.zh.md)。

### 相关工作与实现对比

SM120 上打通 NVFP4 KV cache 目前有**三条路线**，根因相同（交错布局下
FlashInfer 用 `data_stride/8` 推导 SF 步长，页项错 8 倍），解法层次不同：

| | [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | 本仓库 → vLLM | 本仓库 → SGLang |
|---|---|---|---|
| 改动面 | FlashInfer **CUDA 内核** + jit utils + vLLM python = 4 文件（含 C++） | **2 个 Python 文件** + `VLLM_KV_CACHE_LAYOUT=HND` | **2 个 Python 文件** |
| 核心手法 | 内核接受**显式 SF stride**；V-SF 在寄存器内反 swizzle | **HND 布局**让每页 K/V side 连续 | 拆分后端 + 构造 XQA draft mask + draft KV dtype |
| decode 后端 | FA2（arch-generic） | XQA + fa2 prefill | `trtllm_mha`(XQA) + `flashinfer` prefill |
| 版本锚定 | vLLM `0.1.dev16944` + FlashInfer `0.6.11.post2` | vLLM **0.29.0** + FlashInfer **0.6.18** | SGLang **0.5.19** + FlashInfer **0.6.18** |
| 适用布局 | NHD / HND 均可 | 需 HND | 内部处理 |
| MTP | 兼容 | 兼容（K=2） | 兼容（K=2） |
| 实测 KV 池（2×16GB） | 不适用（内核级） | 292,103 | 193,472 |

**要点**：若要更普适（任意布局 / 混合精度），hikari 的内核路线最彻底；
若要最小改动、可回滚，本仓库这两条更轻 —— 其中 **SGLang 这条改动最小**，
因为 SGLang 已经把 NVFP4 KV 的机制都做好了，缺的只是投机解码那一段接线。

### 致谢与溯源

本项目的实现建立在很多人的工作之上。**本项目没有提出任何新的算法或内核**，
NVFP4 KV 能在 SM120 上跑，能力来自 FlashInfer、vLLM 与 SGLang 本身，方向来自
上游社区（[vLLM issue #49011](https://github.com/vllm-project/vllm/issues/49011)）。
我们做的只是把已有结论组合起来、并把过程记录下来：

1. 受上游门控思路（#49818）与 [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)
   的布局分析启发，试出 `VLLM_KV_CACHE_LAYOUT=HND` 可以让 store 与读取的假设
   天然一致，省掉去 swizzle 与内核改动。这是同一根因的另一种解法，不是新发现。
2. 移植上游 [vLLM #53543](https://github.com/vllm-project/vllm/pull/53543) 的少量
   改动，让 vLLM 侧 MTP 与 NVFP4 KV 能同时用。
3. SGLang 侧定位并修好四个互相独立的接线断点（native-FP4 拦截、缺 `kv_cache_sf`、
   缺 XQA draft mask、draft-extend 后端路由），外加一个配置层的陷阱
   （draft KV dtype 会静默毁掉接受率）。XQA 的 mask 契约来自阅读 FlashInfer 的
   `csrc/xqa/mha.cu`；#53543 在 vLLM 侧独立证明了同一结论。
4. 把过程工程化并写清楚（可校验的补丁脚本、回滚、自检、文档），方便别人复现。

如果上游合并了正式修复，本仓库就可以退役了。

**上游代码与项目**

| 项目 | 用途 |
|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 推理引擎；vLLM 补丁针对其 `vllm/v1/attention/backends/flashinfer.py` 与 `vllm/utils/flashinfer.py` |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 推理引擎；SGLang 补丁针对其 `trtllm_mha_backend.py` 与 `draft_utils.py` |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | 提供 XQA decode / fa2 prefill 的 NVFP4-KV 内核、NVFP4 scales、draft mask 契约 |
| [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | 最重要的参考：FA2 内核 patch 路线 |
| [MiaAI-Lab/exllamav3](https://github.com/MiaAI-Lab/exllamav3) | 参考其 Triton paged-attention 内在线反量化思路 |

**关键 issue / PR**

主线程：[vLLM issue #49011 — nvfp4 KV cache on SM120](https://github.com/vllm-project/vllm/issues/49011)

| 贡献者 | 贡献 |
|---|---|
| **@0xdespot** | 发起 issue #49011；RTX 5090 首个可运行原型；PR #49818 作者 |
| **@gtrak** | 第一个提出 2×5060 Ti 支持；在 2×RTX 5060 Ti 上独立复现 |
| **@seanyourhighness** | 发现并发崩溃；证明 #44455 是读错字节的因果；发现 V block-scale 写入缺陷 |
| **@heungwing** | RTX PRO 6000 + WSL2 独立验证 |
| **@stevenmoto** | 4× RTX 6000 Pro Blackwell 问题复现 |
| **@gaby** | 指出 B200/B300 上同类失败 |

相关 PR：

| PR | 内容 | 状态 |
|---|---|---|
| [vLLM #49718](https://github.com/vllm-project/vllm/pull/49718) | FlashInfer XQA decode on SM12x | 已合并 |
| [vLLM #49818](https://github.com/vllm-project/vllm/pull/49818) | Enable NVFP4 KV cache on SM120 | closed，vLLM 侧门控思路据此移植 |
| [vLLM #50085](https://github.com/vllm-project/vllm/pull/50085) | Write linear V block scales on SM120/SM121 | closed，本仓库以 HND 布局绕过 |
| [vLLM #53543](https://github.com/vllm-project/vllm/pull/53543) | Enable masked NVFP4 XQA on SM120 | open，本仓库移植其核心 ~30 行（vLLM），它也独立印证了 SGLang 侧的 mask 断点 |
| [vLLM #53681](https://github.com/vllm-project/vllm/pull/53681) | NVFP4 KV cache: fix block_size/layout detection | open，参考 |

---

*本仓库许可见 `LICENSE`（MIT）；第三方代码说明见 `NOTICE`。*
