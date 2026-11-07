# shellcheck shell=bash
# 共享：stage2 checkpoint 发现、分区选择、eval 提交（供 watch 脚本 source）

stage2_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# 分区负载：RUNNING+PENDING  job 数（越少越空）
stage2_partition_job_count() {
  local p="$1"
  squeue -p "$p" -h -t RUNNING,PENDING 2>/dev/null | wc -l | tr -d ' '
}

# 空闲节点数（sinfo idle）
stage2_partition_idle_nodes() {
  local p="$1"
  sinfo -p "$p" -h -t idle -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}'
}

# 在 PARTITION_CANDIDATES 中选相对更空的分区（先比队列 job 数，再比 idle 节点）
stage2_pick_partition() {
  local candidates="${PARTITION_CANDIDATES:-gpu4090 gpu40901t}"
  local best="" best_jobs=999999 best_idle=-1
  local p jobs idle
  for p in $candidates; do
    jobs=$(stage2_partition_job_count "$p")
    idle=$(stage2_partition_idle_nodes "$p")
    if [ "$jobs" -lt "$best_jobs" ] || { [ "$jobs" -eq "$best_jobs" ] && [ "$idle" -gt "$best_idle" ]; }; then
      best_jobs=$jobs
      best_idle=$idle
      best=$p
    fi
  done
  if [ -z "$best" ]; then
    best="${FALLBACK_PARTITION:-gpu4090}"
  fi
  stage2_log "[partition] pick=${best} (jobs=${best_jobs} idle_nodes=${best_idle} candidates=${candidates})"
  echo "$best"
}

stage2_job_name() {
  local run="$1" step="$2"
  local tag="${run#omnisvg_stage2_4b_}"
  echo "ev_${tag}_${step}"
}

stage2_ckpt_dir() {
  local run="$1" step="$2"
  echo "${REPO_ROOT}/output_stage2/${run}/step_${step}"
}

stage2_weight_path() {
  local run="$1" step="$2"
  echo "$(stage2_ckpt_dir "$run" "$step")/model.safetensors"
}

stage2_ckpt_file_size() {
  local f="$1"
  stat -c%s "$f" 2>/dev/null || echo 0
}

# 仍在 save_state / FSDP merge / 写 safetensors 中
stage2_ckpt_is_writing() {
  local run="$1" step="$2"
  local d
  d="$(stage2_ckpt_dir "$run" "$step")"
  local w="${d}/model.safetensors"

  if [ -f "${d}/model.safetensors.tmp" ]; then
    return 0
  fi
  if compgen -G "${d}/pytorch_model_fsdp_*" >/dev/null 2>&1; then
    if [ ! -f "$w" ]; then
      return 0
    fi
    local min_partial="${CKPT_MIN_BYTES:-1000000000}"
    if [ "$(stage2_ckpt_file_size "$w")" -lt "$min_partial" ]; then
      return 0
    fi
  fi
  return 1
}

# 大文件需多轮确认 size 不变再提交；状态见 metrics/.stage2_ckpt_stable.tsv
stage2_ckpt_ready_for_eval() {
  local run="$1" step="$2"
  local w
  w="$(stage2_weight_path "$run" "$step")"

  if [ ! -f "$w" ]; then
    return 1
  fi
  if stage2_ckpt_is_writing "$run" "$step"; then
    stage2_log "[ckpt] 仍在写入或 FSDP 合并中: ${run} step_${step}"
    return 1
  fi

  local min_bytes="${CKPT_MIN_BYTES:-1000000000}"
  local sz
  sz=$(stage2_ckpt_file_size "$w")
  if [ "$sz" -lt "$min_bytes" ]; then
    stage2_log "[ckpt] model.safetensors 过小 (${sz} < ${min_bytes}): ${run} step_${step}"
    return 1
  fi

  local stable_polls="${CKPT_STABLE_POLLS:-2}"
  local cache="${REPO_ROOT}/metrics/.stage2_ckpt_stable.tsv"
  local key="${run}|${step}"
  local prev_size prev_count now new_count
  mkdir -p "$(dirname "$cache")"

  prev_size=""
  prev_count=0
  if [ -f "$cache" ]; then
    while IFS=$'\t' read -r k ps pc _rest; do
      [ "$k" = "$key" ] || continue
      prev_size=$ps
      prev_count=$pc
    done < "$cache"
  fi

  if [ -z "$prev_size" ]; then
    new_count=1
  elif [ "$prev_size" = "$sz" ]; then
    new_count=$((prev_count + 1))
  else
    new_count=1
    stage2_log "[ckpt] 大小仍在变化 ${prev_size} -> ${sz} bytes: ${run} step_${step}"
  fi

  now=$(date +%s)
  if [ -f "$cache" ]; then
    grep -v -F "${key}"$'\t' "$cache" > "${cache}.new" 2>/dev/null || true
  else
    : > "${cache}.new"
  fi
  printf '%s\t%s\t%s\t%s\n' "$key" "$sz" "$new_count" "$now" >> "${cache}.new"
  mv "${cache}.new" "$cache"

  if [ "$new_count" -lt "$stable_polls" ]; then
    stage2_log "[ckpt] 等待大小稳定 (${new_count}/${stable_polls}, poll=${POLL_SEC:-120}s): ${run} step_${step}"
    return 1
  fi

  local min_age="${CKPT_MIN_AGE_SEC:-60}"
  if [ "$min_age" -gt 0 ]; then
    local mtime age
    mtime=$(stat -c %Y "$w" 2>/dev/null || echo 0)
    age=$((now - mtime))
    if [ "$age" -lt "$min_age" ]; then
      stage2_log "[ckpt] 距最后修改过近 (${age}s < ${min_age}s): ${run} step_${step}"
      return 1
    fi
  fi

  return 0
}

stage2_eval_csv_path() {
  echo "${STAGE2_EVAL_CSV:-${REPO_ROOT}/metrics/stage2_checkpoint_eval_summary.csv}"
}

stage2_eval_done() {
  local run="$1" step="$2"
  local summ="${REPO_ROOT}/eval_results/${run}_step_${step}/logs/summary_cand${CANDIDATE:-1}.txt"
  [ -f "$summ" ]
}

# 指标已在汇总 CSV 中（与 sync_stage2_eval_logs_csv 输出一致）
stage2_in_eval_csv() {
  local run="$1" step="$2"
  local csv task
  csv="$(stage2_eval_csv_path)"
  [ -f "$csv" ] || return 1
  task="${STAGE2_EVAL_CSV_TASK:-complete}"
  awk -F, -v r="$run" -v s="$step" -v t="$task" \
    'NR > 1 && $1 == r && $2 == s && ($3 == t || t == "") { found = 1 }
     END { exit(found ? 0 : 1) }' "$csv"
}

stage2_eval_already_satisfied() {
  local run="$1" step="$2"
  stage2_eval_done "$run" "$step" && return 0
  stage2_in_eval_csv "$run" "$step" && return 0
  return 1
}

# 仅对 step >= MIN_AUTO_EVAL_STEP 做自动提交（0 表示不限制）
stage2_meets_auto_eval_step() {
  local step="$1"
  local min="${MIN_AUTO_EVAL_STEP:-0}"
  [ "$min" -eq 0 ] && return 0
  [ "$step" -ge "$min" ]
}

stage2_in_queue() {
  local jname="$1"
  squeue -u "$USER" -n "$jname" -h 2>/dev/null | grep -q .
}

stage2_should_skip_step() {
  local step="$1"
  local s
  for s in ${SKIP_STEPS:-12500}; do
    [ "$step" = "$s" ] && return 0
  done
  return 1
}

# 列出某 run 下磁盘上已有、且尚未评测的 step（数字排序）
stage2_list_pending_steps() {
  local run="$1"
  local run_dir="${REPO_ROOT}/output_stage2/${run}"
  local d step w
  [ -d "$run_dir" ] || return 0
  for d in "$run_dir"/step_*; do
    [ -d "$d" ] || continue
    step="${d##*/step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if ! stage2_meets_auto_eval_step "$step"; then
      continue
    fi
    if stage2_should_skip_step "$step"; then
      continue
    fi
    if stage2_eval_already_satisfied "$run" "$step"; then
      continue
    fi
    w="${d}/model.safetensors"
    [ -f "$w" ] || continue
    if stage2_in_queue "$(stage2_job_name "$run" "$step")"; then
      continue
    fi
    if ! stage2_ckpt_ready_for_eval "$run" "$step"; then
      continue
    fi
    echo "$step"
  done | sort -n
}

stage2_submit_eval() {
  local run="$1" step="$2" partition="$3"
  local w jname out
  w="$(stage2_weight_path "$run" "$step")"
  jname="$(stage2_job_name "$run" "$step")"

  if [ ! -f "$w" ]; then
    stage2_log "[skip] 权重不存在: $w"
    return 0
  fi
  if stage2_ckpt_is_writing "$run" "$step"; then
    stage2_log "[skip] 提交前仍在写入/合并: ${run} step_${step}"
    return 0
  fi
  if stage2_in_queue "$jname"; then
    stage2_log "[skip] 已在队列: $jname"
    return 0
  fi
  if stage2_eval_already_satisfied "$run" "$step"; then
    stage2_log "[skip] 已有指标(CSV/summary): ${run} step_${step}"
    return 0
  fi
  if ! stage2_meets_auto_eval_step "$step"; then
    return 0
  fi

  local poll="${POLL_SEC:-120}"
  while true; do
    if out=$(sbatch --partition="$partition" --job-name="$jname" \
      --export=ALL,WEIGHT_MODEL="$w",TASKS="${TASKS:-complete}",STAGE="${STAGE:-all}",CANDIDATE="${CANDIDATE:-1}" \
      "${REPO_ROOT}/scripts/eval_run.sh" 2>&1); then
      stage2_log "[submit] partition=${partition} ${run} step_${step} -> $out"
      return 0
    fi
    if echo "$out" | grep -q QOSMaxSubmitJobPerUserLimit; then
      stage2_log "[wait] QOS 提交上限，${poll}s 后重试: $jname ($partition)"
      sleep "$poll"
      partition=$(stage2_pick_partition)
    else
      stage2_log "[error] ${run} step_${step} ($partition): $out"
      return 1
    fi
  done
}

# 扫描 RUNS，提交待评测 checkpoint；每轮最多 MAX_SUBMIT_PER_POLL 个
stage2_submit_pending_checkpoints() {
  local run step n=0 max="${MAX_SUBMIT_PER_POLL:-4}"
  local partition
  partition=$(stage2_pick_partition)
  for run in ${RUNS:-}; do
    while IFS= read -r step; do
      [ -n "$step" ] || continue
      stage2_submit_eval "$run" "$step" "$partition" || true
      n=$((n + 1))
      if [ "$n" -ge "$max" ]; then
        stage2_log "[limit] 本轮已提交 ${max} 个，下轮继续"
        return 0
      fi
      partition=$(stage2_pick_partition)
    done < <(stage2_list_pending_steps "$run")
  done
}
