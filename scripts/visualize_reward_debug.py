#!/usr/bin/env python
"""Dump every mask the GRPO reward pipeline builds, plus the reward breakdown.

Examples
--------
Score the ground truth itself (the reward ceiling for a sample)::

    python scripts/visualize_reward_debug.py \
        --data-dir /data/phd23_weiguang_zhang/works/svg/my_lis2_2 \
        --uid 7F24_FZCaoQBLSJW_v01 --oracle --out-dir output_reward_debug/oracle

Score a real completion produced by the model::

    python scripts/visualize_reward_debug.py \
        --data-dir /data/phd23_weiguang_zhang/works/svg/my_lis2_2 \
        --uid 7F24_FZCaoQBLSJW_v01 \
        --completion-file completion.txt \
        --out-dir output_reward_debug/step_000100

``--completion-file`` accepts tagged text (``<skeleton>...</skeleton>``),
marker text (``[197000 SKEL_S]...``) or a JSON list of token ids.

Written files
-------------
``damaged_mask.png``, ``gt_full_mask.png``, ``gt_missing_contour_mask.png``,
``gt_skeleton_mask.png``, ``pred_skeleton_mask.png``, ``pred_contour_mask.png``,
``pred_contour_skeleton_mask.png``, ``merged_pred_mask.png``,
``reward_report.json`` and an ``overview.png`` contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards import (  # noqa: E402
    MaskCache,
    RewardConfig,
    build_sample,
    compute_area_penalty,
    compute_contour_iou,
    compute_merge_iou,
    compute_overlap_penalty,
    compute_skeleton_contour_consistency,
    compute_skeleton_iou,
    get_backend,
    mask_area,
    oracle_completion,
    parse_model_output,
    reward_one_completion,
    save_mask_png,
)
from rewards.raster import overlay_masks_png  # noqa: E402

MASK_FILES = (
    "damaged_mask.png",
    "gt_full_mask.png",
    "gt_missing_contour_mask.png",
    "gt_skeleton_mask.png",
    "pred_skeleton_mask.png",
    "pred_contour_mask.png",
    "pred_contour_skeleton_mask.png",
    "merged_pred_mask.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the GRPO reward masks for one sample/completion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("sample source")
    source.add_argument("--data-dir", help="Stage-2 data directory containing svg/ json/ png/")
    source.add_argument("--uid", help="Sample id, e.g. 7F24_FZCaoQBLSJW_v01")
    source.add_argument(
        "--sample-json",
        help="JSON file with damaged_svg / gt_full_svg / gt_missing_contour_svg / gt_skeleton_svg",
    )

    completion = parser.add_argument_group("completion source")
    completion.add_argument("--completion", help="Completion text passed inline")
    completion.add_argument("--completion-file", help="File holding the completion")
    completion.add_argument(
        "--oracle",
        action="store_true",
        help="Use the ground truth as the completion (measures the reward ceiling)",
    )

    parser.add_argument("--out-dir", default="output_reward_debug", help="Output directory")
    parser.add_argument("--reward-config", help="YAML file with a `reward:` section")
    parser.add_argument("--canvas-size", type=int, help="Override the mask canvas size")
    parser.add_argument(
        "--consistency-mode",
        choices=("iou", "dice", "coverage"),
        help="Override the skeleton/contour consistency metric",
    )
    return parser.parse_args()


def load_sample(args: argparse.Namespace) -> dict:
    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as handle:
            sample = json.load(handle)
        sample.setdefault("uid", os.path.splitext(os.path.basename(args.sample_json))[0])
        return sample

    if not args.data_dir or not args.uid:
        raise SystemExit("provide --sample-json, or both --data-dir and --uid")

    sample = build_sample(args.data_dir, args.uid, require_image=False)
    if sample is None:
        raise SystemExit(f"could not build a sample for uid={args.uid} in {args.data_dir}")
    return sample


def load_completion(args: argparse.Namespace, sample: dict) -> object:
    if args.oracle:
        return oracle_completion(sample)
    if args.completion_file:
        with open(args.completion_file, "r", encoding="utf-8") as handle:
            raw = handle.read()
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                payload = json.loads(stripped)
                if isinstance(payload, list) and all(isinstance(v, int) for v in payload):
                    return payload
            except json.JSONDecodeError:
                pass
        return raw
    if args.completion is not None:
        return args.completion
    raise SystemExit("provide --completion, --completion-file or --oracle")


def build_config(args: argparse.Namespace) -> RewardConfig:
    config = RewardConfig.from_yaml(args.reward_config) if args.reward_config else RewardConfig()
    if args.canvas_size:
        config.canvas_size = args.canvas_size
    if args.consistency_mode:
        config.consistency_mode = args.consistency_mode
    return config


def main() -> int:
    args = parse_args()
    config = build_config(args)
    sample = load_sample(args)
    completion = load_completion(args, sample)

    os.makedirs(args.out_dir, exist_ok=True)

    parsed = parse_model_output(
        completion,
        viewbox=config.viewbox,
        tokenization_config_path=config.tokenization_config_path,
        model_size=config.model_size,
    )
    reward, info = reward_one_completion(completion, sample, config, return_info=True)

    masks = MaskCache(config)
    damaged = masks.fill(sample.get("damaged_svg"))
    gt_full = masks.fill(sample.get("gt_full_svg"))
    gt_contour = masks.fill(sample.get("gt_missing_contour_svg"))
    gt_skeleton = masks.stroke(sample.get("gt_skeleton_svg"))
    pred_skeleton = masks.stroke(parsed.skeleton_svg)
    pred_contour = masks.fill(parsed.contour_svg)
    pred_contour_axis = masks.contour_skeleton(parsed.contour_svg)
    merged = damaged | pred_contour

    named_masks = dict(
        zip(
            MASK_FILES,
            (
                damaged,
                gt_full,
                gt_contour,
                gt_skeleton,
                pred_skeleton,
                pred_contour,
                pred_contour_axis,
                merged,
            ),
        )
    )
    for filename, mask in named_masks.items():
        save_mask_png(mask, os.path.join(args.out_dir, filename))

    overlay_masks_png(
        [gt_contour, pred_contour, gt_skeleton, pred_skeleton],
        [(120, 200, 255), (255, 140, 120), (40, 90, 200), (200, 40, 40)],
        os.path.join(args.out_dir, "overview.png"),
    )

    report = {
        "uid": sample.get("uid"),
        "reward": reward,
        "breakdown": info,
        "parsed": parsed.to_dict(),
        "terms_recomputed": {
            "R_skeleton_iou": compute_skeleton_iou(
                parsed.skeleton_svg, sample.get("gt_skeleton_svg"), config, masks
            ),
            "R_contour_iou": compute_contour_iou(
                parsed.contour_svg, sample.get("gt_missing_contour_svg"), config, masks
            ),
            "R_consistency_iou": compute_skeleton_contour_consistency(
                parsed.skeleton_svg, parsed.contour_svg, config, masks
            ),
            "R_merge_iou": compute_merge_iou(
                sample.get("damaged_svg"),
                parsed.contour_svg,
                sample.get("gt_full_svg"),
                config,
                masks,
            ),
            "P_overlap": compute_overlap_penalty(
                sample.get("damaged_svg"), parsed.contour_svg, config, masks
            ),
            "P_area": compute_area_penalty(
                parsed.contour_svg, sample.get("gt_missing_contour_svg"), config, masks
            ),
        },
        "mask_areas": {name: mask_area(mask) for name, mask in named_masks.items()},
        "config": config.to_dict(),
        "raster_backend": get_backend(),
    }
    report_path = os.path.join(args.out_dir, "reward_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"uid            : {sample.get('uid')}")
    print(f"raster backend : {get_backend()}  canvas={config.canvas_size}")
    print(f"parse valid    : {parsed.valid}  source={parsed.source}  error={parsed.error}")
    print(f"reward         : {reward:.4f}")
    for key in (
        "R_format",
        "R_skeleton_iou",
        "R_contour_iou",
        "R_consistency_iou",
        "R_merge_iou",
        "P_overlap",
        "P_area",
        "skeleton_gate",
        "skeleton_missing",
    ):
        print(f"  {key:20s} {info.get(key)}")
    print(f"masks + report written to {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
