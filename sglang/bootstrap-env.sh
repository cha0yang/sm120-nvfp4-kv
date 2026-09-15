#!/usr/bin/env bash
# bootstrap-env.sh — 可 source 的环境修复库 (SGLang)
#
#   source bootstrap-env.sh   # 只做修复 + export, 不打印检查报告
#
# 逻辑与 bootstrap.sh 共用同一份实现; 改动请改这里。
# 需要 VENV 已设置 (serve.sh 里先算好再 source)。
set -uo pipefail

: "${VENV:?bootstrap-env.sh 需要 VENV 已设置}"

SG_PY="$VENV/bin/python"; [[ -x "$SG_PY" ]] || SG_PY="$VENV/bin/python3"
SG_SITE="$("$SG_PY" -c 'import site; print(site.getsitepackages()[0])')"
SG_CU13="$SG_SITE/nvidia/cu13"

# 1. CUDA_HOME + PATH (deep_ep 的 find_cuda_home() 需要)
export CUDA_HOME="${CUDA_HOME:-$SG_CU13}"
export PATH="$CUDA_HOME/bin:$PATH"

# 2. JIT 链接器写死 -L$CUDA_HOME/lib64, 但 pip 包里只有 lib/;
#    且 pip 包只带带版本号的 .so.13, 无 unversioned symlink
if [[ -d "$CUDA_HOME/lib" ]]; then
  [[ -e "$CUDA_HOME/lib64" ]] || ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
  for lib in cudart cublas cublasLt; do
    if [[ -e "$CUDA_HOME/lib/lib${lib}.so.13" && ! -e "$CUDA_HOME/lib/lib${lib}.so" ]]; then
      ln -sfn "lib${lib}.so.13" "$CUDA_HOME/lib/lib${lib}.so"
    fi
  done
fi

# 3. nvcc 13.4 + cudart headers 混装: CCCL 的严格版本检查会误报不兼容。
#    用 wrapper 注入 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK 关掉它。
NVCC_WRAPPER="$VENV/bin/nvcc-wrapper"
if [[ ! -x "$NVCC_WRAPPER" ]] || ! grep -q CCCL_DISABLE "$NVCC_WRAPPER" 2>/dev/null; then
  cat > "$NVCC_WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$CUDA_HOME/bin/nvcc" -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK "\$@"
EOF
  chmod +x "$NVCC_WRAPPER"
fi
# PATH 里的 nvcc 必须是 wrapper (nvcc-bin 放在 $CUDA_HOME/bin 之前)
mkdir -p "$VENV/bin/nvcc-bin"
ln -sfn "$NVCC_WRAPPER" "$VENV/bin/nvcc-bin/nvcc"
export PATH="$VENV/bin/nvcc-bin:$PATH"
export FLASHINFER_NVCC="$NVCC_WRAPPER"

unset SG_PY SG_SITE SG_CU13
