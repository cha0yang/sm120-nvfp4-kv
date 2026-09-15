#!/usr/bin/env bash
# bootstrap.sh — SGLang + FlashInfer 环境自检 (幂等; 不碰 sglang 源码)
#
#   VENV=~/sglang ./sglang/bootstrap.sh          执行修复并自检
#   VENV=~/sglang ./sglang/bootstrap.sh --check  只检查, 不改动
#
# 实际修复逻辑在 bootstrap-env.sh (serve.sh 也 source 它), 这里只做:
#   1. 调用修复
#   2. 打印版本/CUDA/补丁/环境变量的检查报告
#
# 与顶层 bootstrap.sh (vLLM 用) 是两个独立脚本: 目标 venv、版本基线、
# 补丁路径都不同, 硬合并会把两边都搞错。
set -uo pipefail
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

VENV="$(readlink -f "${VENV:-$HOME/sglang}")"
if [[ ! -x "$VENV/bin/python" && ! -x "$VENV/bin/python3" ]]; then
  echo "!! 找不到 python: $VENV/bin/python" >&2
  echo "   请指向 SGLang 虚拟环境, 例如: VENV=~/sglang $0" >&2
  exit 1
fi
export VENV
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY="$VENV/bin/python3"

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1"; }

echo "── 1. venv / 包版本 ──────────────────"
VER="$("$PY" - <<'EOF'
try:
    import sglang, flashinfer, torch
    print(sglang.__version__, flashinfer.__version__, torch.__version__)
except Exception as e:
    print("ERR", e)
EOF
)"
echo "  venv: $VENV"
echo "  $VER   (基线: sglang 0.5.19 / flashinfer >=0.6.15 / torch 2.13)"

# 修复 (幂等)。--check 模式下仍然执行: 这些补链操作本身幂等且无害,
# 而且不修的话后面的可用性检查没有意义。
echo "── 2. 环境修复 (bootstrap-env.sh) ────"
# shellcheck source=bootstrap-env.sh
source "$D/bootstrap-env.sh"
ok "CUDA_HOME=$CUDA_HOME"
ok "PATH 前置: $(dirname "$(command -v nvcc)")"

echo "── 3. CUDA 工具链 ────────────────────"
CU13="$CUDA_HOME"
if [[ -x "$CU13/bin/nvcc" ]]; then ok "nvcc: $("$CU13/bin/nvcc" --version | tail -1 | tr -s ' ')"; else bad "缺少 $CU13/bin/nvcc"; fi
[[ -e "$CU13/lib64" ]] && ok "lib64 -> lib 就绪" || bad "缺 lib64 软链"
[[ -e "$CU13/lib/libcudart.so" ]] && ok "libcudart.so 软链就绪" || bad "缺 libcudart.so 软链"
for lib in cublas cublasLt; do
  [[ -e "$CU13/lib/lib${lib}.so" ]] && ok "lib${lib}.so 软链就绪" || bad "缺 lib${lib}.so 软链"
done
[[ -x "$VENV/bin/nvcc-wrapper" ]] && ok "nvcc-wrapper 就绪" || bad "缺 nvcc-wrapper"

echo "── 4. deep_ep / FlashInfer 可用性 ────"
RES="$("$PY" - <<'EOF' 2>&1 | tail -1
import shutil, importlib.util
if importlib.util.find_spec("flashinfer") is None:
    print("FLASHINFER_MISSING")
elif not shutil.which("nvcc"):
    print("NO_NVCC_IN_PATH")
else:
    try:
        import deep_ep  # noqa: F401  (needs a real CUDA_HOME)
        print("OK")
    except Exception as e:
        print(f"DEEP_EP_FAIL: {type(e).__name__}: {e}")
EOF
)"
case "$RES" in
  OK) ok "flashinfer + deep_ep 可用";;
  NO_NVCC_IN_PATH) bad "nvcc 不在 PATH";;
  DEEP_EP_FAIL*) bad "deep_ep 失败: $RES";;
  *) bad "flashinfer 未安装";;
esac

echo "── 5. NVFP4-KV + MTP 补丁状态 ─────────"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
if grep -q "nvfp4_verify" "$SITE/sglang/srt/layers/attention/trtllm_mha_backend.py" 2>/dev/null; then
  ok "补丁已应用"
else
  bad "未打补丁 — 运行: $D/patches/patch.sh"
fi

echo "── 6. 关键环境变量 (必须在 serve.sh 里) ──"
SERVE="$D/serve.sh"
if [[ ! -f "$SERVE" ]]; then
  bad "找不到 $SERVE"
else
  grep -qE 'source .*bootstrap-env\.sh' "$SERVE" && ok "serve.sh 会 source bootstrap-env.sh" || bad "serve.sh 未 source bootstrap-env.sh"
  for kv in "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK" "HF_HUB_OFFLINE" "PYTORCH_CUDA_ALLOC_CONF"; do
    grep -qE "$kv" "$SERVE" && ok "serve.sh: $kv" || bad "serve.sh 缺: $kv"
  done
fi
ENVLIB="$D/bootstrap-env.sh"
if [[ ! -f "$ENVLIB" ]]; then
  bad "找不到 $ENVLIB"
else
  for kv in "CUDA_HOME" "nvcc-bin" "FLASHINFER_NVCC"; do
    grep -qE "$kv" "$ENVLIB" && ok "bootstrap-env.sh: $kv" || bad "bootstrap-env.sh 缺: $kv"
  done
fi

echo
if [[ "$CHECK" == "1" ]]; then
  echo "(--check: 只读检查完成; 软链/wrapper 由 bootstrap-env.sh 幂等补齐)"
else
  echo "完成。启动: $D/run.sh"
fi
