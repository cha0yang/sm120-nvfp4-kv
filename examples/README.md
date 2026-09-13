# 示例部署：2× RTX 5060 Ti + Qwen3.8-27B-NVFP4

## 环境

| 项 | 值 |
|---|---|
| GPU | 2× RTX 5060 Ti 16GB |
| CPU | AMD Ryzen 7 9700X |
| 内存 | 16GB DDR5 6000MHz × 2（30Gi 可用） |
| 主板 | 铭瑄 B850 AIGA |
| OS | Ubuntu 26.04.1 LTS（Resolute Raccoon） |
| Python | 3.14 venv（路径用 `VENV` 指定） |
| vllm | 0.29.0 |
| torch | 2.13.0+cu132 |
| flashinfer-python | 0.6.18 |
| CUDA 工具链 | pip `nvidia/cu13`（nvcc 13.4 + headers 13.2 混装，靠 `nvcc-wrapper` 关严格检查） |

## 部署步骤

```bash
# 0) 指定虚拟环境路径
export VENV=~/vllm

# 1) 建 venv
python3 -m venv "$VENV" && source "$VENV/bin/activate"
pip install uv
uv pip install vllm==0.29.0 --torch-backend auto

# 2) 环境自检 + 修复（CUDA 软链、nvcc-wrapper）
../bootstrap.sh

# 3) 打 NVFP4-KV 补丁（自动校验版本 + 哈希）
../patches/patch.sh

# 4) 起服务（首次启动需 JIT 编译，约 10–20 分钟）
./run.sh
```

## 关键配置（serve.sh）

| 参数 | 值 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | 2 | 双卡 |
| `--kv-cache-dtype` | nvfp4 | 我们的目的 |
| `--max-model-len` | 200000 | KV 池 214,457 tokens（1.07× 并发） |
| `--max-num-seqs` | 1 | 单并发 |
| `--gpu-memory-utilization` | 0.97 | 显示走核显，显存全给 |
| `--performance-mode` | interactivity | 单用户低延迟 |
| `--cudagraph-capture-sizes` | 3 | = seqs×(K+1) = 1×3 |
| `--speculative-config` | mtp K=2 | 投机解码 |
| `--override-generation-config` | T=0.6 / top_p .8 / top_k 20 / presence_penalty 1.5 | Qwen3.8 官方 Instruct 参数 |
| `--default-chat-template-kwargs` | enable_thinking:false | 默认不思考 |
