#!/usr/bin/env bash
# revert.sh — 还原 vLLM 的 NVFP4-KV 补丁 (等价于 patch.sh --revert)
#
#   VENV=~/vllm13 ./revert.sh
set -uo pipefail
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$D/patch.sh" --revert
