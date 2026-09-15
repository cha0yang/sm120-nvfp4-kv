# NVFP4-KV + MTP 补丁

[English](README.md)

让 SM120 上的 SGLang 0.5.19 同时启用 `--kv-cache-dtype nvfp4` 与 MTP 投机解码。

## 文件

| 文件 | 作用 |
|---|---|
| `patch.sh` | 补丁安装（自动校验版本 + 文件哈希，幂等） |
| `revert.sh` | `patch.sh --revert` 的快捷方式 |
| `VERSIONS.txt` | 补丁基线：sglang 版本、flashinfer 最低版本、目标文件哈希 |
| `01-attention-trtllm-mha.patch` | 补丁 1 → `sglang/srt/layers/attention/trtllm_mha_backend.py` |
| `02-speculative-draft-utils.patch` | 补丁 2 → `sglang/srt/speculative/draft_utils.py` |

## 用法

在本目录（`sglang/patches/`）下运行。`VENV` 默认 `~/sglang`；虚拟环境在别处时
显式指定（`VENV=~/elsewhere ./patch.sh`）。（vLLM 侧的脚本则是强制要求显式设置。）

```bash
./patch.sh            # 校验 sglang 版本 + flashinfer 版本 + 文件哈希 → 打补丁
./patch.sh --check    # 只检查，不改动
./patch.sh --revert   # 用 .pre-nvfp4kv.bak 备份还原
./patch.sh --force    # 版本/哈希不符时强打（不推荐）
```

校验内容：

- `sglang.__version__ == VERSIONS.txt:sglang_expected`（当前 0.5.19）
- `flashinfer.__version__ >= VERSIONS.txt:flashinfer_min`（当前 0.6.15）
- 每个目标文件 sha256：等于 patched 哈希 → 已打；等于 orig 哈希 → 可打；其他 → 拒绝

打完补丁后会再次比对两个文件的哈希与 patched 基线，并对每个文件跑一次
`ast.parse` 语法检查。

## 补丁 1 改了什么（`trtllm_mha_backend.py`）

### 1a. 让 target-verify 通过 native-FP4 的拦截

`forward_extend` 原来是无条件拒绝：

```python
if self.decode_uses_native_fp4:
    raise RuntimeError("TRTLLM MHA with native FP4 KV cache supports decode only; ...")
```

但投机解码的 target-verify 与 draft-extend 正是走 `forward_extend`，用的却是
**decode kernel**，所以现在放行：

```python
if self.decode_uses_native_fp4 and not (
    forward_batch.forward_mode.is_target_verify()
    or forward_batch.forward_mode.is_draft_extend_v2()
):
    raise RuntimeError(...)
```

真正的 prefill 仍然报错 —— NVFP4 的 prefill 必须走 `flashinfer`
（dequant workspace）或 `triton`。

### 1b. verify 路径上取到 packed FP4 + block scales

原来只有 `forward_decode` 做了这件事。verify 路径从 `get_kv_buffer()` 拿到裸
packed FP4，且不传 `kv_cache_sf`，于是 kernel 把 4bit 数据当成未量化数据读：

```python
nvfp4_verify = self.is_nvfp4_kvcache and is_decode_mode
if nvfp4_verify:
    kv_cache, kv_cache_block_scales = self._get_nvfp4_decode_kv_cache(layer)
    k_scale, v_scale = self._get_nvfp4_bmm_scales(layer)
    bmm1_scale = q_scale * k_scale * layer.scaling
    bmm2_scale = v_scale
else:
    ... # 原路径, kv_cache_block_scales = None
```

随后 `kv_cache_sf=kv_cache_block_scales` 被一路传到三个 decode 调用点
（ENCODER_ONLY verify、ragged verify、fixed-q-len verify）以及
`_run_fixed_q_len_decode` 内部。

### 1c. 构造 XQA 的 draft-block mask

XQA 在 `q_seq_len > 1` 时断言 `mask is not None`。mask 契约（来自
`flashinfer/xqa.py` 与 `csrc/xqa/mha.cu`）是：

```
shape:  [batch_size, q_seq_len, ((q_seq_len + 31) // 32) * 2]
dtype:  torch.uint16   （bit-packed，按 32 bit 对齐）
行内 bit i = 第 i 个 draft 位置可见
```

它**等价于** `[batch_size, q_seq_len, divUp(q_seq_len, 32)]` 个 **uint32** 字。
helper 按 uint32 分配再 view 成 uint16：

```python
def _spec_verify_mask_row(self):
    spec_q = self.speculative_num_draft_tokens
    words = (spec_q + 31) // 32          # <-- uint32 字数, 不是 *2
    row = torch.zeros(spec_q, words, dtype=torch.int32, device=self.device)
    for r in range(spec_q):
        for bit in range(r + 1):
            row[r, bit // 32] |= 1 << (bit % 32)
    return row.view(torch.uint16).unsqueeze(0)   # uint16 view => words*2
```

> **这就是产生乱码的那个 bug。** 若按 `words = (spec_q + 31) // 32 * 2` 个
> *int32* 分配，元素数翻倍；`.view(torch.uint16)` 后每行得到 4 个 uint16 而不是
> 2 个，导致每一行 mask 都错位，输出变成
> `1, 2, 3, 4, 4, 5, 6, 7, 7, 9, ...` 这种重复+跳号。

链式投机（`topk == 1`）下 mask 行与 batch 无关，因此用
`.expand(batch_size, -1, -1)` 广播，并在首次使用时惰性构造。

## 补丁 2 改了什么（`draft_utils.py`）

`speculative_attention_mode == "decode"` 原来会把 **draft-extend** 也送到
decode backend：

```python
backend_name = (
    "decode_attention_backend"
    if get_spec().speculative_attention_mode == "decode"
    else "prefill_attention_backend"
)
```

但 draft-extend 是 prefill 语义（EXTEND 模式，写 draft KV）。在后端拆分
（`prefill=flashinfer`、`decode=trtllm_mha`）时，这会落到一个没有 FP4 prefill
kernel 的后端上，warmup 阶段就失败。现在只有两个 phase 共用同一后端时才跟随
该模式：

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

## 补丁之外还必须做的事（配置层）

打完补丁输出就正确了，但投机接受率仍然是坏的。两个配置项：

### 必须：“--speculative-draft-kv-cache-dtype fp8_e4m3”

draft 模型（MTP head）的 KV 如果也跟着存 NVFP4，其读写路径不正确，
每个 draft token 都预测错：

| draft KV dtype | accept len | accept rate |
|---|---|---|
| `nvfp4`（默认继承） | 1.10~1.25 | 0.10~0.25 |
| `fp8_e4m3` | 2.90~3.00 | 0.95~1.00 |

这是补丁修不了的（不在 `trtllm_mha_backend` 里，而在 draft KV pool 的存取路径）。

### 建议：`--speculative-num-steps 2` / `--speculative-num-draft-tokens 3`

K=2。MTP head 只有 1 层，第 3 个 draft token 接受率只 0.31~0.36，抵不过 verify 成本：

| | K=2 | K=3 |
|---|---|---|
| accept len | 2.4 | 2.91 |
| 每轮生成 | 3.4 tok | 3.91 tok |
| verify 行数 | 3 | 4 |
| **效率 tok/verify行** | **1.13** | 0.98 |

### 慎用：`--enable-linear-replayssm-spec`

能把 KV 池从 147k 推到 166k（`mamba_ratio` 4→3，verify 中间态改固定 ring），
但接受率从 0.62 掉到 0.29。按需取舍。

## 这条路线不需要做的事

- 不需要额外去 swizzle V block scales（XQA 自己处理布局）
- 不需要重编译 C++
- 不需要改 FlashInfer 内核
- **不需要改 draft KV 的代码** —— 一个配置项就绕过了

## 维护

| 事件 | 动作 |
|---|---|
| 升级/重装 sglang | 补丁会丢 → `./patch.sh`（版本不符会拒绝；必要时按新源码重新生成 diff） |
| 彻底回滚 | `./patch.sh --revert` |
| 重新生成 patched 哈希 | 打完补丁后 `sha256sum <目标文件>`，更新 `VERSIONS.txt` |
