#!/usr/bin/env bash
# run.sh — 启动 SGLang, 日志同时输出到终端和 /tmp/serve.log
#
#   ./run.sh               前台 (tee 双输出)
#   nohup ./run.sh &       后台 (日志仍写 /tmp/serve.log)
#
# 额外参数透传给 serve.sh
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-/tmp/serve.log}"

# 清旧进程 (kill.sh 用具体进程名匹配, 不会匹配到本脚本)
"$D/kill.sh" >/dev/null 2>&1 || true

# 注意: 这里用 '>' 而不是 ': >', 后者在被 nohup 的写入者持有时会撕裂日志
: > "$LOG"

echo "启动中, 日志: $LOG"
echo "另开终端跟踪: tail -f $LOG"
echo

# tee: 终端和日志都拿到
exec "$D/serve.sh" "$@" 2>&1 | tee "$LOG"
