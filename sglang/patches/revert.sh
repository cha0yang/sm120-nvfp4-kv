#!/usr/bin/env bash
# revert.sh — 还原 SGLang 的 NVFP4-KV + MTP 补丁 (等价于 patch.sh --revert)
#
#   VENV=~/sglang ./revert.sh
set -uo pipefail
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$D/patch.sh" --revert
