"""Mask-IoU reward pipeline for GRPO fine-tuning of the SVG completion model.

Quick start::

    from rewards import load_samples, reward_one_completion

    sample = load_samples("/path/to/my_lis2_2", split="val", limit=1)[0]
    reward, info = reward_one_completion(completion, sample, return_info=True)

See ``scripts/visualize_reward_debug.py`` for a per-term visual breakdown and
``train_grpo.py`` for the training entry point.
"""

from .data import (
    append_paths_to_svg,
    build_sample,
    decode_uid,
    iter_samples,
    load_samples,
    oracle_completion,
    read_meta_ids,
    wrap_paths,
)
from .parse import (
    MARKER_TOKEN_IDS,
    REPLACEMENT_END_TOKEN_ID,
    REPLACEMENT_START_TOKEN_ID,
    SKELETON_END_TOKEN_ID,
    SKELETON_START_TOKEN_ID,
    ParsedCompletion,
    get_token_decoder,
    parse_model_output,
    split_token_sections,
    wrap_path_data_as_svg,
)
from .raster import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_VIEWBOX,
    SVGRasterizeError,
    dilate_mask,
    empty_mask,
    get_backend,
    mask_area,
    mask_iou,
    overlay_masks_png,
    rasterize_svg_to_mask,
    save_mask_png,
    set_backend,
    skeletonize_mask,
)
from .reward import (
    DEFAULT_CONFIG,
    GLOBAL_STATS,
    LAST_BREAKDOWNS,
    SAMPLE_KEYS,
    MaskCache,
    RewardConfig,
    RewardStats,
    compute_area_penalty,
    compute_contour_iou,
    compute_merge_iou,
    compute_merged_pred_mask,
    compute_overlap_penalty,
    compute_skeleton_contour_consistency,
    compute_skeleton_iou,
    grpo_reward_func,
    reward_batch,
    reward_one_completion,
)

__all__ = [
    # raster
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_VIEWBOX",
    "SVGRasterizeError",
    "dilate_mask",
    "empty_mask",
    "get_backend",
    "mask_area",
    "mask_iou",
    "overlay_masks_png",
    "rasterize_svg_to_mask",
    "save_mask_png",
    "set_backend",
    "skeletonize_mask",
    # parse
    "MARKER_TOKEN_IDS",
    "ParsedCompletion",
    "REPLACEMENT_END_TOKEN_ID",
    "REPLACEMENT_START_TOKEN_ID",
    "SKELETON_END_TOKEN_ID",
    "SKELETON_START_TOKEN_ID",
    "get_token_decoder",
    "parse_model_output",
    "split_token_sections",
    "wrap_path_data_as_svg",
    # reward
    "DEFAULT_CONFIG",
    "GLOBAL_STATS",
    "LAST_BREAKDOWNS",
    "SAMPLE_KEYS",
    "MaskCache",
    "RewardConfig",
    "RewardStats",
    "compute_area_penalty",
    "compute_contour_iou",
    "compute_merge_iou",
    "compute_merged_pred_mask",
    "compute_overlap_penalty",
    "compute_skeleton_contour_consistency",
    "compute_skeleton_iou",
    "grpo_reward_func",
    "reward_batch",
    "reward_one_completion",
    # data
    "append_paths_to_svg",
    "build_sample",
    "decode_uid",
    "iter_samples",
    "load_samples",
    "oracle_completion",
    "read_meta_ids",
    "wrap_paths",
]
