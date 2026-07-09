#!/usr/bin/env bash
# 将 output 下符合 YYYYMM（202602–202604）的顶层实验目录同步到备份 weights，并行 rsync，成功后删除源目录。
set -euo pipefail

SRC="${SRC:-/data/phd23_weiguang_zhang/works/OmniSVG-train/output}"
DST="${DST:-/Data_PHD_Backup/phd23_weiguang_zhang2/weights}"
JOBS="${JOBS:-6}"
# 空格分隔的目录 basename；与其它迁移工具争用同一目标时可暂时跳过，例如：SKIP_NAMES="run_a run_b"
SKIP_NAMES="${SKIP_NAMES:-}"

LOCK="${LOCK:-/tmp/omnisvg_rsync_output_weights.lock}"
exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Is) 已有实例在运行（$LOCK），退出"; exit 1; }

LOG="${LOG:-${SRC%/}/../rsync_parallel_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$DST"
touch "$LOG"

sync_one() {
  local srcdir="$1"
  local name rc
  name="$(basename "$srcdir")"
  if [[ -n "$SKIP_NAMES" ]] && [[ " $SKIP_NAMES " == *" $name "* ]]; then
    echo "$(date -Is) SKIP $name (SKIP_NAMES)" | tee -a "$LOG"
    return 0
  fi
  echo "$(date -Is) START $name" | tee -a "$LOG"
  # -a 归档； -W/--whole-file 跨盘/NFS 时常比增量比对更快； --partial 便于断点续传
  rsync -aW --partial --numeric-ids "${srcdir}/" "${DST}/${name}/"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "$(date -Is) RSYNC_OK $name -> rm source" | tee -a "$LOG"
    rm -rf "$srcdir"
    echo "$(date -Is) DONE $name" | tee -a "$LOG"
  else
    echo "$(date -Is) FAIL $name (rsync exit $rc)" | tee -a "$LOG"
    return 1
  fi
}

export -f sync_one
export DST LOG SKIP_NAMES

mapfile -t DIRS < <(
  find "$SRC" -maxdepth 1 -mindepth 1 \( -name '*_202602*' -o -name '*_202603*' -o -name '*_202604*' \) -print | sort -u
)

if [[ ${#DIRS[@]} -eq 0 ]]; then
  echo "No matching directories under $SRC"
  exit 0
fi

set +e
printf '%s\n' "${DIRS[@]}" | xargs -n1 -P"$JOBS" bash -c 'sync_one "$1"' bash
xs=$?
set -e

echo "$(date -Is) 批量结束 xargs_exit=$xs（日志中检查 FAIL/SKIP）" | tee -a "$LOG"
echo "Log: $LOG"
exit "$xs"
