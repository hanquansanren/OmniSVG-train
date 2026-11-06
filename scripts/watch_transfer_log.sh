#!/usr/bin/env bash
# 实时查看传输日志（速度/进度行来自 rsync --info=progress2）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="${ROOT}/transfer_audit"
LOG="${1:-}"

if [[ -z "$LOG" ]]; then
  LOG="$(ls -t "${AUDIT}"/transfer_*.log 2>/dev/null | head -1 || true)"
fi

if [[ -z "$LOG" || ! -f "$LOG" ]]; then
  echo "用法: $0 [log文件]"
  echo "或先在 ${AUDIT} 下产生 transfer_*.log"
  exit 1
fi

echo "跟踪: $LOG"
echo "--- 最近进度行 (grep) ---"
grep -E 'progress|START |DONE |ERROR|bytes/sec|to-check' "$LOG" 2>/dev/null | tail -20 || true
echo "--- tail -f ---"
tail -f "$LOG"
