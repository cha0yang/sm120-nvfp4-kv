# SM120 GPU 上为 vLLM 启用 NVFP4 KV cache

[English](README.md)

---

## 目录

- [1. 本仓库做什么](#1-本仓库做什么)
- [2. 操作方法](#2-操作方法)
- [3. 实测](#3-实测)
- [4. 其余内容](#4-其余内容)
  - [故障排查](#故障排查)
  - [相关工作与实现对比](#相关工作与实现对比)
  - [致谢与溯源](#致谢与溯源)

---

## 1. 本仓库做什么

让消费级 Blackwell (SM120) GPU 在 vLLM 上用上 **NVFP4 KV cache**。

- **问题**：vLLM 的 `--kv-cache-dtype nvfp4` 在 SM120 上被门控拒绝，上游 PR 至今未合并。
  而 FlashInfer 的 XQA nvfp4 内核**是 SM120 专属**的（硬件路径），不用等于浪费。
- **方案**：2 个 Python 文件补丁 + 1 个环境变量（`VLLM_KV_CACHE_LAYOUT=HND`），
  不需要重编译 C++，不需要改 FlashInfer 内核。

### 运行环境要求

| 项 | 值 | 说明 |
|---|---|---|
| GPU | SM120 | RTX 50 系 / RTX PRO 6000 Blackwell；XQA nvfp4 内核SM120 |
| Python | 3.14 | venv |
| vllm | **0.29.0** | 补丁行号锁定，其他版本 `patch.sh` 会拒绝 |
| flashinfer-python | **≥ 0.6.15** | strided SF 修复，低于此版本 nvfp4 KV 读错字节 |
| torch | 2.13.0+cu132 | 与 vllm 0.29.0 配套 |
| CUDA 工具链 | pip `nvidia/cu13` | nvcc 13.4 + headers 13.2 混装，靠 `nvcc-wrapper` 关严格检查 |

关键版本约束（必须满足，否则补丁/内核不工作）：

- `nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-crt` = **13.4.59**（三者必须同版本，半套会报 ptxas 错误）
- `nvidia-cuda-runtime` = **13.2.75**（与 nvcc 13.4 混装，靠 `nvcc-wrapper` 注入 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`）
- 版本不符时：`uv pip install --no-deps nvidia-cuda-nvcc==13.4.59 nvidia-nvvm==13.4.59 nvidia-cuda-crt==13.4.59`

---

## 2. 操作方法

```bash
# 0) 指定虚拟环境路径
export VENV=~/vllm

# 1) 建 venv 并安装
python3 -m venv "$VENV" && source "$VENV/bin/activate"
pip install uv
uv pip install vllm==0.29.0 --torch-backend auto

# 2) 环境自检 + 修复（CUDA 软链、nvcc-wrapper）
./bootstrap.sh

# 3) 打 NVFP4-KV 补丁（自动校验版本 + 哈希）
./patches/patch.sh

# 4) 运行
./examples/run.sh
```

> 补丁细节见 `patches/README.zh.md`

---

## 3. 实测

> 环境：2× RTX 5060 Ti 16GB (SM120)，`nvidia/Qwen3.8-27B-NVFP4`，TP=2。完整配置见 [`examples/README.md`](examples/README.zh.md)。

| 上下文长度 | prefill 速度 | prefill 时间 | decode 速度 |
|---|---|---|---|
| 8k | 4,102 tok/s | 2.0s | 72.3 tok/s |
| 16k | 3,619 tok/s | 4.5s | 69.9 tok/s |
| 32k | 3,106 tok/s | 10.4s | 69.4 tok/s |
| 64k | 2,375 tok/s | 27.1s | 67.6 tok/s |
| **128k** | **1,617 tok/s** | **79.6s** | **62.4 tok/s** |

## 4. 其余内容

### 故障排查

见 [`TROUBLESHOOTING.zh.md`](TROUBLESHOOTING.zh.md)。

### 相关工作与实现对比

SM120 的 NVFP4 KV cache 目前有**两条独立打通的路线**，根因相同（vLLM 交错布局下
FlashInfer 用 `data_stride/8` 推导 SF 步长，页项错 8 倍），解法层次不同：

| | [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)（FA2 内核 patch） | **本项目**（Python 门控 + HND 布局） |
|---|---|---|
| 改动面 | FlashInfer **CUDA 内核** + jit utils + vLLM python = 4 文件（含 C++） | **2 个 Python 文件** + 1 个环境变量 |
| 核心手法 | 内核接受**显式 SF stride**；V-SF **在寄存器内逐元素反 swizzle** | **`VLLM_KV_CACHE_LAYOUT=HND`** 让每页 K/V side 连续 |
| decode 后端 | FA2（arch-generic） | XQA decode + fa2 prefill |
| 版本锚定 | vLLM `0.1.dev16944` + FlashInfer `0.6.11.post2` | vLLM **0.29.0** + FlashInfer **0.6.18** |
| 适用布局 | NHD / HND 均可 | 需 HND（**缺了会乱码**） |
| 容量 | 1.78× fp8（理论天花板） | 实测 ~1.7× |
| MTP | 兼容（K=1 验证） | 兼容（K=2，移植上游 #53543） |

**要点**：若要更普适（任意布局 / 混合精度），hikari 的内核路线更彻底；
若要**最小改动、可回滚、跟得上新版 vLLM**，本项目这条更轻。

### 致谢与溯源

本项目的实现建立在很多人的工作之上。**本项目没有提出任何新的算法或内核**，
NVFP4 KV 能在 SM120 上跑，能力来自 FlashInfer 与 vLLM 本身，方向来自上游社区
（[issue #49011](https://github.com/vllm-project/vllm/issues/49011) 里各位的探索）。
我们做的只是把已有结论组合起来、并把过程记录下来：

1. 受上游门控思路（#49818）与 [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)
   的布局分析启发，试出 `VLLM_KV_CACHE_LAYOUT=HND` 可以让 store 与读取的假设天然一致，
   省掉去 swizzle 与内核改动。这是同一根因的另一种解法，不是新发现。
2. 移植上游 [#53543](https://github.com/vllm-project/vllm/pull/53543) 的少量改动，让 MTP 与 NVFP4 KV 能同时用。
3. 把过程工程化并写清楚（可校验的补丁脚本、回滚、自检、文档），方便别人复现。

如果上游合并了正式修复，本仓库就可以退役了。

**上游代码与项目**

| 项目 | 用途 |
|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 推理引擎本体；本仓库的补丁就是针对其 `vllm/v1/attention/backends/flashinfer.py` |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | 提供 XQA decode / fa2 prefill 的 NVFP4-KV 内核 |
| [hikarioyama/vllm-nvfp4-kv-sm120](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120) | 最重要的参考：FA2 内核 patch 路线 |
| [MiaAI-Lab/exllamav3](https://github.com/MiaAI-Lab/exllamav3) | 参考其 Triton paged-attention 内在线反量化思路 |

**关键 issue / PR**

主线程：[vllm issue #49011 — nvfp4 KV cache on SM120](https://github.com/vllm-project/vllm/issues/49011)

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
| [#49718](https://github.com/vllm-project/vllm/pull/49718) | FlashInfer XQA decode on SM12x | 已合并 |
| [#49818](https://github.com/vllm-project/vllm/pull/49818) | Enable NVFP4 KV cache on SM120 | closed，本仓库据此移植门控思路 |
| [#50085](https://github.com/vllm-project/vllm/pull/50085) | Write linear V block scales on SM120/SM121 | closed，本仓库以 HND 布局绕过 |
| [#53543](https://github.com/vllm-project/vllm/pull/53543) | Enable masked NVFP4 XQA on SM120 | open，本仓库移植其核心 ~30 行 |
| [#53681](https://github.com/vllm-project/vllm/pull/53681) | NVFP4 KV cache: fix block_size/layout detection | open，参考 |

---

*本仓库许可见 `LICENSE`（MIT）；第三方代码说明见 `NOTICE`。*
