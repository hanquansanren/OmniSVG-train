#!/usr/bin/env python3
"""Build ./eval test sets for img2svg (old2old / old2new) and code-complement.

Output layout:
  eval/eval_task1_1/  png/  svg1/  gt200/  test_meta.csv   (gt200 rendered from svg1)
  eval/eval_task1_2/  png/  svg2/  gt200/  test_meta.csv   (gt200 rendered from svg2)
  eval/eval_task2/    png/  svg/   json/  gt200/  test_meta.csv

All three sets share the same sample ids (sampled once, then filtered by availability).

Examples:
  python scripts/prepare_eval_sets.py --num 60
  python scripts/prepare_eval_sets.py --num 600 --allow_dup_char
  python scripts/prepare_eval_sets.py --num 600 --overwrite --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "metrics"))
from eval_svg_tasks import (  # noqa: E402
    VIEWBOX_SIZE,
    load_gt_image,
    load_gt_replacement_paths,
    rebuild_img2svg_gt200,
    to_uint8,
)

DEFAULTS = {
    "eval_root": os.path.join(REPO_ROOT, "eval"),
    "src_img2svg_old": (
        "/data/phd23_weiguang_zhang/works/svg_generation/"
        "phd23_weiguang_zhang_3/noisy_stage3/FZCaoQBLSJW/stage1_old2old"
    ),
    "src_img2svg_new": (
        "/data/phd23_weiguang_zhang/works/svg_generation/"
        "phd23_weiguang_zhang_3/noisy_stage3/FZCaoQBLSJW/stage1_old2new"
    ),
    "src_complete": "/data/phd23_weiguang_zhang/works/svg/my_lis2_2",
    "val_meta_name": "val_meta.csv",
    "num": 60,
    "seed": 42,
}


def read_val_rows(meta_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "id" not in fieldnames:
        raise ValueError(f"{meta_path} must have an 'id' column")
    return fieldnames, rows


def char_code(uid: str) -> str:
    return uid.split("_", 1)[0]


def files_exist(root: str, uid: str, specs: Sequence[Tuple[str, str]]) -> bool:
    return all(os.path.isfile(os.path.join(root, sub, uid + ext)) for sub, ext in specs)


def valid_img2svg_old(uid: str, src_old: str, src_new: str) -> bool:
    specs = [("png", ".png"), ("svg", ".svg")]
    return files_exist(src_old, uid, specs) and files_exist(src_new, uid, specs)


def valid_complete(uid: str, src: str) -> bool:
    if not files_exist(src, uid, [("png", ".png"), ("svg", ".svg"), ("json", ".json")]):
        return False
    json_path = os.path.join(src, "json", uid + ".json")
    return bool(load_gt_replacement_paths(json_path))


def pick_sample_ids(
    rows: List[Dict[str, str]],
    num: int,
    seed: int,
    allow_dup_char: bool,
    valid_fn: Callable[[str], bool],
) -> Tuple[List[str], List[Dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    picked_rows: List[Dict[str, str]] = []
    seen_chars: set[str] = set()
    for row in shuffled:
        uid = row["id"]
        if not allow_dup_char and char_code(uid) in seen_chars:
            continue
        if not valid_fn(uid):
            continue
        picked_rows.append(row)
        seen_chars.add(char_code(uid))
        if len(picked_rows) >= num:
            break

    picked_rows.sort(key=lambda r: r["id"])
    return [r["id"] for r in picked_rows], picked_rows


def ensure_empty_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path) and os.listdir(path):
        if not overwrite:
            raise SystemExit(f"Error: {path} is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def rebuild_complete_gt200(test_dir: str, ids: Sequence[str], size: int = VIEWBOX_SIZE) -> None:
    out_dir = os.path.join(test_dir, "gt200")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for uid in ids:
        gt = load_gt_image("complete", test_dir, uid, size)
        Image.fromarray(to_uint8(gt)).save(os.path.join(out_dir, uid + ".png"))


def write_test_meta(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_task1_1(
    out_dir: str,
    ids: Sequence[str],
    rows_old: List[Dict[str, str]],
    src_old: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    ensure_empty_dir(out_dir, overwrite)
    if dry_run:
        return
    for sub in ("png", "svg1", "gt200"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    id_set = set(ids)
    meta_rows = [r for r in rows_old if r["id"] in id_set]
    for uid in ids:
        dst_png = os.path.join(out_dir, "png", uid + ".png")
        dst_svg = os.path.join(out_dir, "svg1", uid + ".svg")
        shutil.copy2(os.path.join(src_old, "png", uid + ".png"), dst_png)
        shutil.copy2(os.path.join(src_old, "svg", uid + ".svg"), dst_svg)
        # png in old2old and old2new are identical; only svg differs in task1_2

    write_test_meta(
        os.path.join(out_dir, "test_meta.csv"),
        list(rows_old[0].keys()) if rows_old else ["id"],
        meta_rows,
    )
    rebuild_img2svg_gt200(out_dir, VIEWBOX_SIZE, list(ids), svg_subdir="svg1")


def build_task1_2(
    out_dir: str,
    ids: Sequence[str],
    rows_new: List[Dict[str, str]],
    src_new: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    ensure_empty_dir(out_dir, overwrite)
    if dry_run:
        return
    for sub in ("png", "svg2", "gt200"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    id_set = set(ids)
    meta_rows = [r for r in rows_new if r["id"] in id_set]
    for uid in ids:
        shutil.copy2(os.path.join(src_new, "png", uid + ".png"), os.path.join(out_dir, "png", uid + ".png"))
        shutil.copy2(os.path.join(src_new, "svg", uid + ".svg"), os.path.join(out_dir, "svg2", uid + ".svg"))

    write_test_meta(
        os.path.join(out_dir, "test_meta.csv"),
        list(rows_new[0].keys()) if rows_new else ["id"],
        meta_rows,
    )
    rebuild_img2svg_gt200(out_dir, VIEWBOX_SIZE, list(ids), svg_subdir="svg2")


def build_task2(
    out_dir: str,
    ids: Sequence[str],
    rows: List[Dict[str, str]],
    src: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    ensure_empty_dir(out_dir, overwrite)
    if dry_run:
        return
    for sub in ("png", "svg", "json", "gt200"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    id_set = set(ids)
    meta_rows = [r for r in rows if r["id"] in id_set]
    for uid in ids:
        shutil.copy2(os.path.join(src, "png", uid + ".png"), os.path.join(out_dir, "png", uid + ".png"))
        shutil.copy2(os.path.join(src, "svg", uid + ".svg"), os.path.join(out_dir, "svg", uid + ".svg"))
        shutil.copy2(os.path.join(src, "json", uid + ".json"), os.path.join(out_dir, "json", uid + ".json"))

    write_test_meta(
        os.path.join(out_dir, "test_meta.csv"),
        list(rows[0].keys()) if rows else ["id"],
        meta_rows,
    )
    rebuild_complete_gt200(out_dir, ids)


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_root", default=DEFAULTS["eval_root"])
    p.add_argument("--src_img2svg_old", default=DEFAULTS["src_img2svg_old"])
    p.add_argument("--src_img2svg_new", default=DEFAULTS["src_img2svg_new"])
    p.add_argument("--src_complete", default=DEFAULTS["src_complete"])
    p.add_argument("--val_meta_name", default=DEFAULTS["val_meta_name"],
                   help="Validation list filename under each source (default: val_meta.csv)")
    p.add_argument("--num", type=int, default=DEFAULTS["num"],
                   help="Number of samples per eval set (e.g. 60 or 600)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--allow_dup_char", action="store_true",
                   help="Allow multiple variants (_v00, _v01, ...) of the same character")
    p.add_argument("--overwrite", action="store_true", help="Replace existing eval_task* directories")
    p.add_argument("--dry_run", action="store_true", help="Only report how many ids would be selected")
    p.add_argument("--tasks", default="task1_1,task1_2,task2",
                   help="Comma-separated: task1_1, task1_2, task2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    complete_meta = os.path.join(args.src_complete, args.val_meta_name)
    old_meta = os.path.join(args.src_img2svg_old, args.val_meta_name)
    if not os.path.isfile(complete_meta):
        raise SystemExit(f"Missing {complete_meta}")
    if not os.path.isfile(old_meta):
        raise SystemExit(f"Missing {old_meta}")

    _, complete_rows = read_val_rows(complete_meta)
    _, old_rows = read_val_rows(old_meta)
    _, new_rows = read_val_rows(os.path.join(args.src_img2svg_new, args.val_meta_name))

    old_by_id = {r["id"]: r for r in old_rows}
    new_by_id = {r["id"]: r for r in new_rows}

    def valid_all(uid: str) -> bool:
        return (
            valid_complete(uid, args.src_complete)
            and valid_img2svg_old(uid, args.src_img2svg_old, args.src_img2svg_new)
        )

    # Sample from complete val list order/shuffle, require files in all three sources.
    ids, picked_complete_rows = pick_sample_ids(
        complete_rows, args.num, args.seed, args.allow_dup_char, valid_all
    )

    if len(ids) < args.num:
        print(
            f"Warning: only {len(ids)} ids available (requested {args.num}). "
            f"Try --allow_dup_char or lower --num."
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_requested": args.num,
        "num_selected": len(ids),
        "seed": args.seed,
        "allow_dup_char": args.allow_dup_char,
        "sources": {
            "img2svg_old2old": args.src_img2svg_old,
            "img2svg_old2new": args.src_img2svg_new,
            "complete": args.src_complete,
        },
        "outputs": {
            "task1_1": os.path.join(args.eval_root, "eval_task1_1"),
            "task1_2": os.path.join(args.eval_root, "eval_task1_2"),
            "task2": os.path.join(args.eval_root, "eval_task2"),
        },
        "sample_ids": ids,
    }

    print(f"Selected {len(ids)} ids (seed={args.seed}, allow_dup_char={args.allow_dup_char})")
    if args.dry_run:
        print("Dry run — no files written.")
        print(f"First 10 ids: {ids[:10]}")
        write_manifest(os.path.join(args.eval_root, "manifest_dry_run.json"), manifest)
        return

    os.makedirs(args.eval_root, exist_ok=True)

    picked_old_rows = [old_by_id[i] for i in ids]
    picked_new_rows = [new_by_id[i] for i in ids]

    if "task1_1" in tasks:
        build_task1_1(
            manifest["outputs"]["task1_1"],
            ids,
            picked_old_rows,
            args.src_img2svg_old,
            args.overwrite,
            dry_run=False,
        )
        print(f"Built {manifest['outputs']['task1_1']} ({len(ids)} samples, gt200 from svg1)")

    if "task1_2" in tasks:
        build_task1_2(
            manifest["outputs"]["task1_2"],
            ids,
            picked_new_rows,
            args.src_img2svg_new,
            args.overwrite,
            dry_run=False,
        )
        print(f"Built {manifest['outputs']['task1_2']} ({len(ids)} samples, gt200 from svg2)")

    if "task2" in tasks:
        build_task2(
            manifest["outputs"]["task2"],
            ids,
            picked_complete_rows,
            args.src_complete,
            args.overwrite,
            dry_run=False,
        )
        print(f"Built {manifest['outputs']['task2']} ({len(ids)} samples, gt200 from png+json)")

    manifest_path = os.path.join(args.eval_root, "manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path}")
    print("\nInference examples:")
    print(f"  image-to-svg:    --input {manifest['outputs']['task1_1']}/png")
    print(f"  code-complement: --input {manifest['outputs']['task2']}")


if __name__ == "__main__":
    main()
