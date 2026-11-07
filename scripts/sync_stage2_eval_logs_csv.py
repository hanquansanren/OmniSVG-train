#!/usr/bin/env python3
"""Scan logs / logs_* eval_*.out and append/update stage2 checkpoint eval summary CSV."""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_RUNS = (
    "omnisvg_stage2_4b_20261020_230852",
    "omnisvg_stage2_4b_20261020_231348",
)

TASKS = ("complete", "text2svg", "image2svg", "sketch2svg")

CSV_FIELDS = [
    "run",
    "step",
    "task",
    "cand",
    "ok_total",
    "tokens",
    "sec_per_sample",
    "mse",
    "ssim",
    "lpips",
    "stage",
    "output_root",
    "log_dir",
    "log_file",
    "slurm_job",
    "log_mtime_iso",
    "synced_at_iso",
]

OUTPUT_ROOT_RE = re.compile(r"Output root:\s+\./eval_results/(.+)")
STAGE_RE = re.compile(r"Stage:\s+(\S+)")
JOB_RE = re.compile(r"Job:\s+(\d+)")
STEP_RE = re.compile(r"_step_(\d+)$")


def parse_summary_rows(text):
    rows = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("task") and "MSE" in stripped:
            in_table = True
            rows = []
            continue
        if not in_table:
            continue
        if stripped.startswith("---"):
            continue
        if stripped.startswith("==="):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0] in TASKS:
            rows.append(parts)
    return rows


def row_dict(
    run,
    step,
    task,
    cand,
    ok_total,
    tokens,
    sec_per_sample,
    mse,
    ssim,
    lpips,
    stage,
    output_root,
    log_dir,
    log_file,
    slurm_job,
    log_mtime_iso,
    synced_at_iso,
):
    return {
        "run": run,
        "step": step,
        "task": task,
        "cand": cand,
        "ok_total": ok_total,
        "tokens": tokens,
        "sec_per_sample": sec_per_sample,
        "mse": mse,
        "ssim": ssim,
        "lpips": lpips,
        "stage": stage,
        "output_root": output_root,
        "log_dir": log_dir,
        "log_file": log_file,
        "slurm_job": slurm_job,
        "log_mtime_iso": log_mtime_iso,
        "synced_at_iso": synced_at_iso,
    }


def dedupe_key(row):
    return (row["run"], row["step"], row["task"], row["cand"], row["output_root"])


def job_int(row):
    try:
        return int(row["slurm_job"])
    except (TypeError, ValueError):
        return -1


def parse_log(path, repo_root, allowed_runs):
    text = path.read_text(errors="replace")
    run = None
    for name in allowed_runs:
        if name in text:
            run = name
            break
    if run is None:
        return []

    summary = parse_summary_rows(text)
    if not summary:
        return []

    m_out = OUTPUT_ROOT_RE.search(text)
    output_root = m_out.group(1) if m_out else ""
    step = "?"
    m_step = STEP_RE.search(output_root)
    if m_step:
        step = m_step.group(1)
    elif output_root.endswith("best_model"):
        step = "best_model"
    elif output_root.endswith("_final"):
        step = "final"

    m_stage = STAGE_RE.search(text[:800])
    stage = m_stage.group(1) if m_stage else "?"

    m_job = JOB_RE.search(text[:800])
    slurm_job = m_job.group(1) if m_job else "?"

    try:
        log_dir = str(path.parent.relative_to(repo_root))
    except ValueError:
        log_dir = path.parent.name

    mtime = datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z"
    synced = datetime.utcnow().isoformat() + "Z"

    out = []
    for parts in summary:
        sec = parts[4]
        if sec.endswith("s"):
            sec = sec[:-1]
        out.append(
            row_dict(
                run=run,
                step=step,
                task=parts[0],
                cand=parts[1],
                ok_total=parts[2],
                tokens=parts[3],
                sec_per_sample=sec,
                mse=parts[5],
                ssim=parts[6],
                lpips=parts[7],
                stage=stage,
                output_root=output_root,
                log_dir=log_dir,
                log_file=path.name,
                slurm_job=slurm_job,
                log_mtime_iso=mtime,
                synced_at_iso=synced,
            )
        )
    return out


def discover_log_dirs(repo_root):
    dirs = [repo_root / "logs"]
    dirs.extend(sorted(repo_root.glob("logs_*")))
    return [d for d in dirs if d.is_dir()]


def collect_rows(repo_root, runs):
    by_key = {}
    for log_dir in discover_log_dirs(repo_root):
        for path in sorted(log_dir.glob("eval_*.out")):
            for row in parse_log(path, repo_root, runs):
                key = dedupe_key(row)
                prev = by_key.get(key)
                if prev is None or job_int(row) >= job_int(prev):
                    by_key[key] = row

    def sort_key(r):
        step = r["step"]
        step_n = int(step) if str(step).isdigit() else 10 ** 9
        return (r["run"], step_n, r["task"])

    return sorted(by_key.values(), key=sort_key)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in CSV_FIELDS})
    tmp.replace(path)


def sync_once(repo_root, csv_path, runs):
    rows = collect_rows(repo_root, runs)
    write_csv(csv_path, rows)
    return len(rows), len(discover_log_dirs(repo_root))


def watch(repo_root, csv_path, runs, poll_sec):
    last_count = -1
    while True:
        n, n_dirs = sync_once(repo_root, csv_path, runs)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if n != last_count:
            print(
                "[{0}] updated {1} rows={2} log_dirs={3}".format(
                    ts, csv_path, n, n_dirs
                ),
                flush=True,
            )
            last_count = n
        else:
            print("[{0}] no change rows={1}".format(ts, n), flush=True)
        time.sleep(poll_sec)


def main():
    parser = argparse.ArgumentParser(description="Sync stage2 eval logs to CSV.")
    parser.add_argument(
        "--repo-root",
        default=os.path.join(os.path.dirname(__file__), ".."),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV (default: <repo>/metrics/stage2_checkpoint_eval_summary.csv)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=float(os.environ.get("POLL_SEC", "120")),
    )
    args = parser.parse_args()

    repo_root = Path(os.path.abspath(args.repo_root))
    csv_path = Path(args.csv) if args.csv else (
        repo_root / "metrics" / "stage2_checkpoint_eval_summary.csv"
    )
    runs = set(args.runs)

    if args.watch:
        print(
            "Watching {0}/logs* -> {1} runs={2} poll={3}s".format(
                repo_root, csv_path, sorted(runs), args.poll_sec
            ),
            flush=True,
        )
        watch(repo_root, csv_path, runs, args.poll_sec)
        return 0

    n, _ = sync_once(repo_root, csv_path, runs)
    print("Wrote {0} rows -> {1}".format(n, csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
