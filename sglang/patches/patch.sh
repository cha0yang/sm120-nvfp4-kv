#!/usr/bin/env bash
# patch.sh — 给 SGLang 打/还原 NVFP4 KV cache + MTP 补丁 (两个文件)
#
#   VENV=~/sglang ./patch.sh           自动校验版本后打补丁 (幂等)
#   VENV=~/sglang ./patch.sh --check   只检查是否可打 (不改动任何文件)
#   VENV=~/sglang ./patch.sh --revert  还原
#   VENV=~/sglang ./patch.sh --force   版本/哈希对不上也强行尝试 (谨慎)
#
# 补丁内容: 让 SM120 上 `--kv-cache-dtype nvfp4` 与 MTP 投机解码共存
#   1. sglang/srt/layers/attention/trtllm_mha_backend.py
#        - spec-decode verify 走 forward_extend 时不再被 "decode only" 拦截
#        - verify 路径取 packed FP4 + block scales (原来只 forward_decode 有)
#        - 补上 XQA 需要的 draft-block mask (原来没传 -> 乱码)
#   2. sglang/srt/speculative/draft_utils.py
#        - draft-extend 是 prefill 语义, 不能因 speculative_attention_mode=decode
#          丢给没有 FP4 prefill kernel 的 trtllm_mha
#
# 校验三件事:
#   1. sglang 版本 == VERSIONS.txt 里的 sglang_expected
#   2. flashinfer-python >= flashinfer_min
#   3. 每个目标文件 sha256: patched → 已打; orig → 可打; 其他 → 拒绝
set -uo pipefail
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$D"

MODE=apply; FORCE=0
case "${1:-}" in
  --check)  MODE=check ;;
  --revert) MODE=revert ;;
  --force)  FORCE=1 ;;
  "")       ;;
  *) echo "用法: $0 [--check|--revert|--force]"; exit 2 ;;
esac

# shellcheck disable=SC1091
source "$D/VERSIONS.txt"

# VENV 未指定时默认 ~/sglang
VENV="$(readlink -f "${VENV:-$HOME/sglang}")"
# venv 里 python / python3 只保证存在其一, 两个都试
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY="$VENV/bin/python3"
[[ -x "$PY" ]] || {
  echo "!! 找不到 python: $PY" >&2
  echo "   请指向 SGLang 虚拟环境, 例如: VENV=~/sglang $0" >&2
  exit 1
}
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

sha()  { sha256sum "$1" | awk '{print $1}'; }
norm() { # 把文件状态归一成 orig / patched / unknown / missing
  local f="$1" o="$2" p="$3"
  [[ -f "$f" ]] || { echo missing; return; }
  local h; h="$(sha "$f")"
  if   [[ "$h" == "$p" ]]; then echo patched
  elif [[ "$h" == "$o" ]]; then echo orig
  else echo "unknown:${h:0:16}"; fi
}
ver() { "$PY" - "$1" <<'EOF' 2>/dev/null || echo MISSING
import importlib, sys
m = importlib.import_module(sys.argv[1])
print(getattr(m, "__version__", "?"))
EOF
}

V_SG="$(ver sglang)"; V_FI="$(ver flashinfer)"; V_TORCH="$(ver torch)"
T1="$SITE/$target1"; T2="$SITE/$target2"
S1="$(norm "$T1" "$sha256_1_orig" "$sha256_1_patched")"
S2="$(norm "$T2" "$sha256_2_orig" "$sha256_2_patched")"
P1="01-attention-trtllm-mha.patch"; P2="02-speculative-draft-utils.patch"

echo "── 环境 ───────────────────────────────"
echo "  python       : $PY"
echo "  site-packages: $SITE"
echo "  sglang       : $V_SG        (需要 $sglang_expected)"
echo "  flashinfer   : $V_FI   (需要 >= $flashinfer_min)"
echo "  torch        : $V_TORCH   (基线 $torch_major)"
echo "  文件1 $target1"
echo "        -> $S1"
echo "  文件2 $target2"
echo "        -> $S2"

if [[ "$MODE" == "revert" ]]; then
  for f in "$T1" "$T2"; do
    if [[ -f "$f.pre-nvfp4kv.bak" ]]; then cp "$f.pre-nvfp4kv.bak" "$f"; echo "已从 .bak 还原: $f"
    else echo "!! 无 .bak: $f"; fi
  done
  echo "还原后: 文件1=$(norm "$T1" "$sha256_1_orig" "$sha256_1_patched") 文件2=$(norm "$T2" "$sha256_2_orig" "$sha256_2_patched")"
  exit 0
fi

if [[ "$S1" == "patched" && "$S2" == "patched" ]]; then
  echo "状态: 两个文件都已打过补丁"; exit 0
fi

# ---- 版本与哈希校验 ----
FAIL=0
[[ "$V_SG" == "$sglang_expected" ]] || { echo "!! sglang 版本不符 (得到 $V_SG, 需要 $sglang_expected)"; FAIL=1; }
"$PY" - "$V_FI" "$flashinfer_min" <<'EOF' || { echo "!! flashinfer 版本过低 (需要 >= $flashinfer_min)"; FAIL=1; }
import sys
from packaging.version import Version
sys.exit(0 if Version(sys.argv[1]) >= Version(sys.argv[2]) else 1)
EOF
for pair in "文件1:$S1" "文件2:$S2"; do
  case "${pair##*:}" in
    orig|patched) ;;
    *) echo "!! ${pair%%:*} 状态异常 (${pair##*:}) — 既非原始也非已打补丁, 可能是别的 sglang 构建"; FAIL=1 ;;
  esac
done
if [[ "$FAIL" == "1" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    echo "⚠ --force: 忽略校验, 强行尝试 patch -p1 --forward --fuzz=3"
    for pf in "$P1" "$P2"; do ( cd "$SITE" && patch -p1 --forward --fuzz=3 < "$D/$pf" ); done
    exit $?
  fi
  echo "拒绝执行。修复版本后重试, 或: $0 --force"; exit 1
fi

[[ "$MODE" == "check" ]] && { echo "✓ 校验通过, 可以打补丁"; exit 0; }

# ---- 打补丁 ----
for f in "$T1" "$T2"; do cp -n "$f" "$f.pre-nvfp4kv.bak" 2>/dev/null || true; done
apply_one() { # <patch> <current_state>
  local pf="$1" st="$2"
  if [[ "$st" == "patched" ]]; then echo "  [跳过] $pf (已应用)"; return 0; fi
  ( cd "$SITE" && patch -p1 --forward --fuzz=1 < "$D/$pf" ) || {
    echo "!! patch 失败: $pf — sglang 源码结构可能已变, 需重新生成补丁"; return 1; }
}
apply_one "$P1" "$S1" || exit 1
apply_one "$P2" "$S2" || exit 1

R1="$(norm "$T1" "$sha256_1_orig" "$sha256_1_patched")"
R2="$(norm "$T2" "$sha256_2_orig" "$sha256_2_patched")"
echo "结果: 文件1=$R1  文件2=$R2"
if [[ "$R1" == "patched" && "$R2" == "patched" ]]; then
  echo "✓ 补丁已应用 (两文件 hash 均与基线一致)"
else
  echo "⚠ 补丁已应用, 但结果 hash 与基线不同 (可能用了 fuzz), 建议跑启动检查"
fi
for f in "$T1" "$T2"; do "$PY" -c "import ast;ast.parse(open('$f').read())" || { echo "!! 语法错误: $f"; exit 1; }; done
echo "✓ 语法检查通过"
echo
echo "记得 serve.sh 里要有: --kv-cache-dtype nvfp4 --decode-attention-backend trtllm_mha"
echo "                      --speculative-algorithm EAGLE --speculative-attention-mode decode"
