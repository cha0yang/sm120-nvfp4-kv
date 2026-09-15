# NVFP4-KV 补丁

[English](README.md)

让消费级 Blackwell (SM120) 在 vLLM 上启用 **NVFP4 KV cache**（XQA decode + fa2 prefill）。
上游 issue [#49011](https://github.com/vllm-project/vllm/issues/49011) 的诉求，上游 PR #49818/#50085 至今未合并。

## 文件

| 文件 | 作用 |
|---|---|
| `patch.sh` | 补丁安装（自动校验版本 + 文件哈希，幂等） |
| `revert.sh` | `patch.sh --revert` 的快捷方式 |
| `VERSIONS.txt` | 补丁基线：vllm 版本、flashinfer 最低版本、目标文件哈希 |
| `01-attention-flashinfer.patch` | 补丁文件 1 → `vllm/v1/attention/backends/flashinfer.py` |
| `02-utils-flashinfer.patch` | 补丁文件 2 → `vllm/utils/flashinfer.py` |

## 用法

在本目录（`vllm/patches/`）下运行：

```bash
./patch.sh            # 自动校验 vllm 版本 + flashinfer 版本 + 文件哈希 → 打补丁
./patch.sh --check    # 只检查，不改动
./patch.sh --revert   # 还原（用打补丁前的 .bak 备份）
./patch.sh --force    # 版本/哈希不符时强打（不推荐）
```

校验内容：
- vllm 版本 == `VERSIONS.txt` 里的 `vllm_expected`（当前 0.29.0）
- flashinfer-python >= `flashinfer_min`（当前 0.6.15）
- 每个目标文件 sha256：等于 patched 哈希 → 已打；等于 orig 哈希 → 可打；其他 → 拒绝

## 两个关键点（缺一不可）

1. **`VLLM_KV_CACHE_LAYOUT=HND`**（环境变量，写在 serve.sh 里）
   默认 NHD 下每页内存是 `[K0,K1,V0,V1]` 逐 token 交替；而 store kernel 与
   `nvfp4_split_data_scale` 都假设"每 side 先全部 data、再全部 scale"。
   HND（head-major）让每 side 连续 → 完全对齐。**缺了这个会输出乱码。**

2. **`kv_data_type` 必须传 `torch.uint8`**（不能传字符串 `"nvfp4"`）
   flashinfer 内部 `getattr(torch, "nvfp4")` 在 torch 2.13 不存在。

## 补丁覆盖范围

共 **2 个文件**：

1. `vllm/v1/attention/backends/flashinfer.py`
   SM120 门控（允许 nvfp4 KV）、dtype 传递（`kv_data_type` 传 `torch.uint8`）、
   prefill 走 fa2 / decode 走 auto(xqa)、跳过 FP8 staging、
   移植上游 [#53543](https://github.com/vllm-project/vllm/pull/53543) 核心 ~30 行（专用 XQA 路径传 nvfp4 data + `kv_cache_sf`）→ MTP 可用

2. `vllm/utils/flashinfer.py`
   让 SM90/SM12x 的 decode 判定**不再被"artifactory 联网检查"拦截**
   （该检查只对需要下载 trtllm-gen FHMA cubin 的 SM100 路径有意义；XQA 走本地内核/JIT）

## 使用这条路线时不需要做的事

- 不需要额外去 swizzle V scale（fa2/XQA 内核本身支持 SM100 swizzle 布局）
- 不需要重编译 C++（上游 PR #50085 的 .cu 修改不是必需）

## 维护

| 事件 | 动作 |
|---|---|
| 升级/重装 vllm | 补丁会丢 → `./patch.sh`（版本不符会拒绝，需按新源码重新生成 diff） |
| 彻底回滚 | `./patch.sh --revert` |
