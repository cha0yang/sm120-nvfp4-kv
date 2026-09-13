#!/usr/bin/env bash
# 杀 vLLM 主进程 + TP worker + 遗留的资源跟踪进程
# 注意: 用进程可执行路径/内部名字匹配, 不用裸 "vllm serve"
# (裸字符串会匹配到正在执行含该字符串命令的 shell 自身, 自杀了)
set -uo pipefail

pkill -f "VLLM::" 2>/dev/null
pkill -f "bin/vllm serve" 2>/dev/null
pkill -f "multiprocessing.resource_tracker" 2>/dev/null
sleep 1
pkill -9 -f "VLLM::" 2>/dev/null
pkill -9 -f "bin/vllm serve" 2>/dev/null

LEFT="$(pgrep -af 'VLLM::|bin/vllm serve' | grep -v pgrep || true)"
if [ -z "$LEFT" ]; then
  echo "已清干净"
else
  echo "仍有残留:"; echo "$LEFT"
fi
