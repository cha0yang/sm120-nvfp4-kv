#!/usr/bin/env bash
# 杀 SGLang 主进程 + TP worker + JIT 编译残留 + 资源跟踪进程
# 注意: 用进程内部名字/可执行路径匹配, 不用裸 "sglang serve"
# (裸字符串会匹配到正在执行含该字符串命令的 shell 自身, 自杀了)
set -uo pipefail

pkill -f "sglang::" 2>/dev/null
pkill -f "bin/sglang serve" 2>/dev/null
pkill -f "nvcc-wrapper" 2>/dev/null
pkill -f "flashinfer.*cached_ops" 2>/dev/null
pkill -f "multiprocessing.resource_tracker" 2>/dev/null
sleep 1
pkill -9 -f "sglang::" 2>/dev/null
pkill -9 -f "bin/sglang serve" 2>/dev/null
pkill -9 -f "nvcc-wrapper" 2>/dev/null

LEFT="$(pgrep -af 'sglang::|bin/sglang serve' | grep -v pgrep || true)"
if [ -z "$LEFT" ]; then
  echo "已清干净"
else
  echo "仍有残留:"; echo "$LEFT"
fi

# 显示显存占用, 确认 GPU 已释放
if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "── GPU 显存 ──"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
fi
