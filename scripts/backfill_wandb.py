#!/usr/bin/env python3
"""Backfill a W&B run from grpo_metrics.jsonl when local wandb logging failed."""
from __future__ import annotations
###############
# This script is used to backfill a W&B run from grpo_metrics.jsonl when local wandb logging failed.
"""
module load anaconda3 && source activate svg2
cd /gpfs/work/int/weiguangzhang21/project/OmniSVG-train

python scripts/backfill_wandb.py \
  output_grpo/grpo_4b_20260930_224442 \
  --notes "Backfilled from grpo_metrics.jsonl (job 150930; final)"
"""




import argparse
import json
import os
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help="GRPO run directory containing grpo_metrics.jsonl (and optional args.json)",
    )
    parser.add_argument("--project", default="omnisvg-grpo")
    parser.add_argument(
        "--name",
        default=None,
        help="W&B run name (default: basename of run_dir)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional fixed W&B run id; default creates a new id suffixed with -backfill",
    )
    parser.add_argument(
        "--notes",
        default="Backfilled from grpo_metrics.jsonl",
        help="Run notes shown on W&B",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metrics that would be uploaded without calling wandb",
    )
    return parser.parse_args()


def load_config(run_dir: str) -> Dict[str, Any]:
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        with open(args_path, encoding="utf-8") as handle:
            return json.load(handle)
    return {"run_dir": run_dir}


def load_metrics(metrics_path: str) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with open(metrics_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no metrics found in {metrics_path}")
    return records


def main() -> int:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    metrics_path = os.path.join(run_dir, "grpo_metrics.jsonl")
    if not os.path.isfile(metrics_path):
        raise SystemExit(f"missing metrics file: {metrics_path}")

    records = load_metrics(metrics_path)
    run_name = args.name or os.path.basename(run_dir.rstrip("/"))
    run_id = args.run_id or f"{run_name.replace('_', '-')[:64]}-backfill"

    if args.dry_run:
        steps = [record["step"] for record in records]
        print(f"would upload {len(records)} points for run {run_name!r} (steps {steps})")
        return 0

    import wandb

    config = load_config(run_dir)
    run = wandb.init(
        project=args.project,
        name=run_name,
        id=run_id,
        config=config,
        notes=args.notes,
        tags=["backfill"],
        resume="allow",
    )
    try:
        for record in records:
            step = int(record["step"])
            payload = {f"grpo/{key}": value for key, value in record.items() if key != "step"}
            run.log(payload, step=step)
    finally:
        run.finish()

    print(f"uploaded {len(records)} metric rows to {args.project}/{run_name}")
    print(f"view run: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
