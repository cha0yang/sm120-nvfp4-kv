#!/usr/bin/env bash
# bootstrap.sh — 修复 vLLM + FlashInfer 在本机跑起来所需的环境细节
# (幂等; 不碰 vllm 源码, 只补齐工具链软链/包装脚本)
#
#   ./bootstrap.sh          执行修复并自检
#   ./bootstrap.sh --check  只检查
set -uo pipefail
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# VENV 必须由用户指定, 不设默认值, 避免改错环境
if [[ -z "${VENV:-}" ]]; then
  echo "!! 未设置 VENV。请指向 vLLM 虚拟环境, 例如:" >&2
  echo "     VENV=~/vllm13 $0" >&2
  exit 1
fi
VENV="$(readlink -f "$VENV")"
# venv 里 python / python3 只保证存在其一, 两个都试
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY="$VENV/bin/python3"
if [[ ! -x "$PY" ]]; then
  echo "!! 找不到 $VENV/bin/python{,3}" >&2
  echo "   请确认 VENV 指向一个已安装 vllm 的虚拟环境" >&2
  exit 1
fi
# site-packages 动态取, 不写死 python 版本
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
CU13="$SITE/nvidia/cu13"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

ok()   { echo "  ✓ $1"; }
bad()  { echo "  ✗ $1"; }
act()  { [[ "$CHECK" == "1" ]] || eval "$1"; }

echo "── 1. venv / 包版本 ──────────────────"
VER="$("$PY" - <<'EOF'
try:
    import vllm, flashinfer, torch
    print(vllm.__version__, flashinfer.__version__, torch.__version__)
except Exception as e:
    print("ERR", e)
EOF
)"
echo "  venv: $VENV"
echo "  $VER   (基线: vllm 0.29.0 / flashinfer >=0.6.15 / torch 2.13)"

echo "── 2. CUDA 工具链 ────────────────────"
if [[ -x "$CU13/bin/nvcc" ]]; then ok "nvcc: $("$CU13/bin/nvcc" --version | tail -1 | tr -s ' ')"; else bad "缺少 $CU13/bin/nvcc"; fi

# 2a. libcudart.so 软链 (链接 -lcudart 需要)
if [[ -e "$CU13/lib/libcudart.so.13" ]]; then
  if [[ -e "$CU13/lib/libcudart.so" ]]; then ok "lib/libcudart.so 已存在"
  else act "ln -sf libcudart.so.13 '$CU13/lib/libcudart.so'"; ok "已补 lib/libcudart.so -> libcudart.so.13"; fi
else bad "找不到 $CU13/lib/libcudart.so.13"; fi

# 2b. lib64 -> lib (flashinfer JIT 链接写死 -L$cuda_home/lib64)
if [[ -d "$CU13/lib" ]]; then
  if [[ -e "$CU13/lib64" ]]; then ok "lib64 已存在"
  else act "ln -sfn lib '$CU13/lib64'"; ok "已补 lib64 -> lib"; fi
fi

# 2c. libcublas.so / libcublasLt.so 软链 (flashinfer JIT 链接 -lcublas/-lcublasLt 需要)
#     pip 包只带 libcublas.so.13, 无 unversioned symlink; Qwen3.5 等模型触发 gemm JIT 时会断
for lib in cublas cublasLt; do
  if [[ -e "$CU13/lib/lib${lib}.so.13" ]]; then
    if [[ -e "$CU13/lib/lib${lib}.so" ]]; then ok "lib/lib${lib}.so 已存在"
    else act "ln -sf lib${lib}.so.13 '$CU13/lib/lib${lib}.so'"; ok "已补 lib/lib${lib}.so -> lib${lib}.so.13"; fi
  else bad "找不到 $CU13/lib/lib${lib}.so.13"; fi
done

echo "── 3. nvcc wrapper (关 CCCL 严格版本检查) ──"
WRAP="$VENV/bin/nvcc-wrapper"
if [[ -x "$WRAP" ]] && grep -q CCCL_DISABLE "$WRAP"; then ok "$WRAP 就绪"
else
  act "cat > '$WRAP' <<'EOF'
#!/usr/bin/env bash
exec '$CU13/bin/nvcc' -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \"\$@\"
EOF"
  act "chmod +x '$WRAP'"; ok "已创建 $WRAP"
fi

echo "── 4. FlashInfer 可用性 (需要 nvcc 在 PATH) ──"
RES="$(PATH="$CU13/bin:$PATH" "$PY" - <<'EOF'
import shutil, importlib.util
if importlib.util.find_spec("flashinfer") is None: print("MISSING")
else: print("OK" if shutil.which("nvcc") else "NO_NVCC_IN_PATH")
EOF
)"
case "$RES" in
  OK) ok "flashinfer 可用 (nvcc 在 PATH)";;
  NO_NVCC_IN_PATH) bad "nvcc 不在 PATH — serve.sh 必须有 export PATH=\"\$CUDA_HOME/bin:\$PATH\"";;
  *) bad "flashinfer 未安装";;
esac

echo "── 5. NVFP4-KV 补丁状态 ───────────────"
if grep -q "SM120-NVFP4-KV PATCH" "$SITE/vllm/v1/attention/backends/flashinfer.py" 2>/dev/null; then
  ok "补丁已应用"
else
  bad "未打补丁 — 运行: $D/patches/patch.sh"
fi

echo "── 6. 关键环境变量 (必须在 serve.sh 里) ──"
SERVE=""
SERVE="$D/serve.sh"
if [[ ! -f "$SERVE" ]]; then
  bad "找不到 $SERVE"
else
  for kv in "CUDA_HOME" "PATH.*bin" "FLASHINFER_NVCC" "VLLM_KV_CACHE_LAYOUT=HND"; do
    grep -qE "export .*$kv" "$SERVE" && ok "$(basename "$(dirname "$SERVE")")/serve.sh: $kv" || bad "serve.sh 缺: $kv"
  done
fi

echo
[[ "$CHECK" == "1" ]] && echo "(--check 模式, 未改动任何文件)" || echo "完成。启动: $D/run.sh"
