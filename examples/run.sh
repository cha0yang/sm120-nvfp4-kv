#!/usr/bin/env bash
# run.sh — 后台启动 vLLM 并等待 READY
set -uo pipefail
cd "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

PORT="${PORT:-8000}"

# 已在运行则退出
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "已在运行 (:${PORT}), 先 kill.sh 再说"
  exit 1
fi

LOG=/tmp/serve.log
: > "$LOG"
setsid ./serve.sh >"$LOG" 2>&1 < /dev/null &
PID=$!
echo "启动中 pid=$PID ... 等待 READY"

for i in $(seq 1 240); do  # 最多等 20 分钟(首次 JIT 编译较慢)
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "READY: http://127.0.0.1:$PORT/  (启动耗时 ~$((i*5))s)"
    exit 0
  fi
  kill -0 "$PID" 2>/dev/null || { echo "进程挂了, 错误信息:"; grep -E "ERROR|Error|Traceback" "$LOG" | tail -10; exit 1; }
  sleep 5
done
echo "超时未 READY, 看日志: tail -f $LOG"
exit 1
