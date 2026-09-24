"""Mask-IoU reward pipeline for GRPO fine-tuning of the SVG completion model.

The model is asked to complete the missing strokes of a damaged Chinese glyph
and answers with two regions: a stroke *skeleton* and a filled *contour*.  Every
reward term here is an IoU between binary masks rendered on the same
``0 0 200 200`` canvas, which keeps the signal dense and scale-free:

===========================  ============================================
term                         what it measures
===========================  ============================================
``R_format``                 the completion parsed into both regions
``R_skeleton_iou``           predicted skeleton vs. ground-truth skeleton
``R_contour_iou``            predicted contour vs. ground-truth missing contour
``R_consistency_iou``        predicted skeleton vs. skeleton *of* the
                             predicted contour - punishes an empty skeleton
                             that is followed by a confident contour
``R_merge_iou``              damaged glyph OR predicted contour, vs. the
                             complete glyph
``P_overlap``                predicted contour redrawing the damaged glyph
``P_area``                   predicted contour area vs. ground-truth area
===========================  ============================================

Every public function is failure-tolerant: an unparsable SVG yields an empty
mask and a low reward, never an exception that would abort a training step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .parse import ParsedCompletion, parse_model_output
from .raster import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_VIEWBOX,
    ViewBox,
    dilate_mask,
    mask_area,
    mask_iou,
    rasterize_svg_to_mask,
    skeletonize_mask,
)

logger = logging.getLogger(__name__)

SAMPLE_KEYS = ("damaged_svg", "gt_full_svg", "gt_missing_contour_svg", "gt_skeleton_svg")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class RewardConfig:
    """Weights and rasterization settings for the reward pipeline."""

    # Rasterization
    canvas_size: int = DEFAULT_CANVAS_SIZE
    viewbox: ViewBox = DEFAULT_VIEWBOX
    skeleton_stroke_width: float = 3.0
    skeleton_dilate_radius: int = 2

    # Consistency compares a stroked polyline against a one-pixel medial axis,
    # so the skeleton is stroked thin here to avoid a pure thickness mismatch.
    consistency_stroke_width: float = 1.0
    consistency_dilate_radius: int = 3
    # "iou" (spec default), "dice" (same overlap, softer normalisation) or
    # "coverage" (fraction of the contour medial axis the skeleton explains).
    consistency_mode: str = "iou"

    # Reward weights
    w_format: float = 0.10
    w_skeleton_iou: float = 0.30
    w_contour_iou: float = 0.20
    w_consistency_iou: float = 0.30
    w_merge_iou: float = 0.10
    w_overlap_penalty: float = 0.10
    w_area_penalty: float = 0.05

    # Gates and clipping
    invalid_reward: float = -1.0
    skeleton_gate: float = 0.1
    reward_min: float = -1.0
    reward_max: float = 1.0
    eps: float = 1e-6

    # Token decoding (only used when completions are token ids)
    tokenization_config_path: str = "./configs/tokenization.yaml"
    model_size: str = "4B"

    @classmethod
    def from_yaml(cls, path: str) -> "RewardConfig":
        """Load overrides from a YAML file; unknown keys are ignored."""
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        section = payload.get("reward", payload)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in section.items() if k in known}
        if "viewbox" in kwargs and kwargs["viewbox"] is not None:
            kwargs["viewbox"] = tuple(float(v) for v in kwargs["viewbox"])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RewardConfig()


def _resolve(config: Optional[RewardConfig]) -> RewardConfig:
    return config if config is not None else DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
# Mask cache
# --------------------------------------------------------------------------- #


class MaskCache:
    """Memoises rasterization within a single reward evaluation.

    One completion needs the damaged mask twice, the predicted contour mask
    three times and the skeletonised contour twice; caching roughly halves the
    rasterization cost per reward.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = _resolve(config)
        self._store: Dict[tuple, np.ndarray] = {}

    def _key(self, svg: Optional[str], kind: str, extra: Any = None) -> tuple:
        return (kind, extra, svg if svg else "")

    def fill(self, svg: Optional[str]) -> np.ndarray:
        key = self._key(svg, "fill")
        if key not in self._store:
            self._store[key] = rasterize_svg_to_mask(
                svg,
                canvas_size=self.config.canvas_size,
                mode="fill",
                viewbox=self.config.viewbox,
            )
        return self._store[key]

    def stroke(self, svg: Optional[str], stroke_width: Optional[float] = None) -> np.ndarray:
        width = self.config.skeleton_stroke_width if stroke_width is None else stroke_width
        key = self._key(svg, "stroke", width)
        if key not in self._store:
            self._store[key] = rasterize_svg_to_mask(
                svg,
                canvas_size=self.config.canvas_size,
                mode="stroke",
                stroke_width=width,
                viewbox=self.config.viewbox,
                drop_close=True,
            )
        return self._store[key]

    def contour_skeleton(self, svg: Optional[str]) -> np.ndarray:
        """Medial axis of a filled contour SVG."""
        key = self._key(svg, "contour_skeleton")
        if key not in self._store:
            self._store[key] = skeletonize_mask(self.fill(svg))
        return self._store[key]


def _cache(config: RewardConfig, cache: Optional[MaskCache]) -> MaskCache:
    return cache if cache is not None else MaskCache(config)


# --------------------------------------------------------------------------- #
# Individual reward terms
# --------------------------------------------------------------------------- #


def compute_skeleton_iou(
    pred_skeleton_svg: Optional[str],
    gt_skeleton_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """IoU between the predicted and ground-truth stroke skeletons.

    Both skeletons are rendered as strokes and dilated, because one-pixel-wide
    centrelines almost never overlap exactly even when the prediction is right.
    """
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    pred = dilate_mask(masks.stroke(pred_skeleton_svg), cfg.skeleton_dilate_radius)
    gt = dilate_mask(masks.stroke(gt_skeleton_svg), cfg.skeleton_dilate_radius)
    return mask_iou(pred, gt)


def compute_contour_iou(
    pred_contour_svg: Optional[str],
    gt_missing_contour_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """IoU between the predicted contour and the ground-truth missing contour."""
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    return mask_iou(masks.fill(pred_contour_svg), masks.fill(gt_missing_contour_svg))


def compute_skeleton_contour_consistency(
    pred_skeleton_svg: Optional[str],
    pred_contour_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """Agreement between the predicted skeleton and its own predicted contour.

    The contour is filled, thinned to its medial axis, and compared against the
    predicted skeleton.  This is the term that makes an empty (or throw-away)
    skeleton expensive even when the contour happens to be right.

    Note that a *correct* prediction does not score 1.0 here: the offline
    ground-truth skeleton and the medial axis of the vectorised replacement
    blobs are genuinely different curves, so the oracle sits around 0.2-0.5 on
    this dataset.  What matters for GRPO is the gap to a missing or misplaced
    skeleton, which scores 0.0.
    """
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    radius = cfg.consistency_dilate_radius
    contour_axis = masks.contour_skeleton(pred_contour_svg)
    pred_skeleton = masks.stroke(pred_skeleton_svg, cfg.consistency_stroke_width)

    if cfg.consistency_mode == "coverage":
        axis_area = mask_area(contour_axis)
        if axis_area == 0:
            return 1.0 if mask_area(pred_skeleton) == 0 else 0.0
        covered = mask_area(contour_axis & dilate_mask(pred_skeleton, radius))
        return covered / axis_area

    axis = dilate_mask(contour_axis, radius)
    skeleton = dilate_mask(pred_skeleton, radius)
    if cfg.consistency_mode == "dice":
        total = mask_area(axis) + mask_area(skeleton)
        if total == 0:
            return 1.0
        return 2.0 * mask_area(axis & skeleton) / total
    return mask_iou(skeleton, axis)


def compute_merge_iou(
    damaged_svg: Optional[str],
    pred_contour_svg: Optional[str],
    gt_full_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """IoU of the post-processed glyph against the complete glyph.

    Mirrors ``inference.py::combine_partial_and_completion``: the predicted
    paths are appended to the damaged SVG, which at mask level is a union.
    """
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    merged = masks.fill(damaged_svg) | masks.fill(pred_contour_svg)
    return mask_iou(merged, masks.fill(gt_full_svg))


def compute_overlap_penalty(
    damaged_svg: Optional[str],
    pred_contour_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """How much of the prediction redraws strokes the damaged glyph already has."""
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    return mask_iou(masks.fill(pred_contour_svg), masks.fill(damaged_svg))


def compute_area_penalty(
    pred_contour_svg: Optional[str],
    gt_missing_contour_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> float:
    """Log-ratio of predicted to ground-truth contour area, clipped to [0, 1]."""
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    pred_area = mask_area(masks.fill(pred_contour_svg))
    gt_area = mask_area(masks.fill(gt_missing_contour_svg))
    ratio = (pred_area + cfg.eps) / (gt_area + cfg.eps)
    return float(min(1.0, max(0.0, abs(math.log(ratio)))))


def compute_merged_pred_mask(
    damaged_svg: Optional[str],
    pred_contour_svg: Optional[str],
    config: Optional[RewardConfig] = None,
    cache: Optional[MaskCache] = None,
) -> np.ndarray:
    """The mask that post-processing would produce, for debugging/visualisation."""
    cfg = _resolve(config)
    masks = _cache(cfg, cache)
    return masks.fill(damaged_svg) | masks.fill(pred_contour_svg)


# --------------------------------------------------------------------------- #
# Full reward
# --------------------------------------------------------------------------- #


def reward_one_completion(
    completion: Any,
    sample: Dict[str, Any],
    config: Optional[RewardConfig] = None,
    return_info: bool = False,
) -> Any:
    """Score a single completion against its ground truth.

    Args:
        completion: Token ids, marker text or tagged text (see
            :func:`rewards.parse.parse_model_output`).
        sample: Must provide ``damaged_svg``, ``gt_full_svg``,
            ``gt_missing_contour_svg`` and ``gt_skeleton_svg``.
        config: Reward weights; defaults to :data:`DEFAULT_CONFIG`.
        return_info: Also return the per-term breakdown.

    Returns:
        ``float`` reward in ``[-1, 1]``, or ``(reward, info)`` when
        ``return_info`` is set.
    """
    cfg = _resolve(config)
    info: Dict[str, Any] = {
        "reward": cfg.invalid_reward,
        "raw_reward": cfg.invalid_reward,
        "valid": False,
        "error": None,
        "R_format": 0.0,
        "R_skeleton_iou": 0.0,
        "R_contour_iou": 0.0,
        "R_consistency_iou": 0.0,
        "R_merge_iou": 0.0,
        "P_overlap": 0.0,
        "P_area": 0.0,
        "skeleton_gate": 1.0,
        "skeleton_missing": False,
        "gt_skeleton_empty": True,
        "n_skeleton_sections": 0,
        "n_contour_sections": 0,
        "pred_contour_area": 0,
        "gt_contour_area": 0,
    }

    try:
        parsed: ParsedCompletion = parse_model_output(
            completion,
            viewbox=cfg.viewbox,
            tokenization_config_path=cfg.tokenization_config_path,
            model_size=cfg.model_size,
        )
    except Exception as exc:  # noqa: BLE001 - parsing must never abort a step
        logger.debug("parse_model_output crashed: %s: %s", type(exc).__name__, exc)
        info["error"] = f"parse_crash: {type(exc).__name__}"
        return (cfg.invalid_reward, info) if return_info else cfg.invalid_reward

    info["error"] = parsed.error
    info["n_skeleton_sections"] = parsed.n_skeleton_sections
    info["n_contour_sections"] = parsed.n_contour_sections
    info["parse_source"] = parsed.source

    if not parsed.valid:
        return (cfg.invalid_reward, info) if return_info else cfg.invalid_reward

    damaged_svg = sample.get("damaged_svg") or ""
    gt_full_svg = sample.get("gt_full_svg") or ""
    gt_missing_contour_svg = sample.get("gt_missing_contour_svg") or ""
    gt_skeleton_svg = sample.get("gt_skeleton_svg") or ""

    masks = MaskCache(cfg)

    try:
        r_format = 1.0
        r_skeleton = compute_skeleton_iou(
            parsed.skeleton_svg, gt_skeleton_svg, cfg, masks
        )
        r_contour = compute_contour_iou(
            parsed.contour_svg, gt_missing_contour_svg, cfg, masks
        )
        r_consistency = compute_skeleton_contour_consistency(
            parsed.skeleton_svg, parsed.contour_svg, cfg, masks
        )
        r_merge = compute_merge_iou(damaged_svg, parsed.contour_svg, gt_full_svg, cfg, masks)
        p_overlap = compute_overlap_penalty(damaged_svg, parsed.contour_svg, cfg, masks)
        p_area = compute_area_penalty(
            parsed.contour_svg, gt_missing_contour_svg, cfg, masks
        )

        gt_skeleton_empty = not masks.stroke(gt_skeleton_svg).any()
        pred_skeleton_empty = parsed.skeleton_empty or not masks.stroke(
            parsed.skeleton_svg
        ).any()
        skeleton_missing = bool(pred_skeleton_empty and not gt_skeleton_empty)
        gate = cfg.skeleton_gate if skeleton_missing else 1.0

        raw_reward = (
            cfg.w_format * r_format
            + cfg.w_skeleton_iou * r_skeleton
            + cfg.w_contour_iou * r_contour
            + cfg.w_consistency_iou * r_consistency
            + cfg.w_merge_iou * r_merge
            - cfg.w_overlap_penalty * p_overlap
            - cfg.w_area_penalty * p_area
        )
        reward = float(np.clip(gate * raw_reward, cfg.reward_min, cfg.reward_max))

        info.update(
            {
                "reward": reward,
                "raw_reward": float(raw_reward),
                "valid": True,
                "R_format": float(r_format),
                "R_skeleton_iou": float(r_skeleton),
                "R_contour_iou": float(r_contour),
                "R_consistency_iou": float(r_consistency),
                "R_merge_iou": float(r_merge),
                "P_overlap": float(p_overlap),
                "P_area": float(p_area),
                "skeleton_gate": float(gate),
                "skeleton_missing": skeleton_missing,
                "gt_skeleton_empty": bool(gt_skeleton_empty),
                "pred_contour_area": mask_area(masks.fill(parsed.contour_svg)),
                "gt_contour_area": mask_area(masks.fill(gt_missing_contour_svg)),
            }
        )
        return (reward, info) if return_info else reward

    except Exception as exc:  # noqa: BLE001 - a broken sample must not stop GRPO
        logger.warning(
            "reward computation failed for uid=%s: %s: %s",
            sample.get("uid", "?"),
            type(exc).__name__,
            exc,
        )
        info["error"] = f"reward_crash: {type(exc).__name__}"
        return (cfg.invalid_reward, info) if return_info else cfg.invalid_reward


# --------------------------------------------------------------------------- #
# TRL GRPOTrainer adapter
# --------------------------------------------------------------------------- #


@dataclass
class RewardStats:
    """Running means of the reward terms, for logging during training."""

    count: int = 0
    sums: Dict[str, float] = field(default_factory=dict)

    _NUMERIC_KEYS = (
        "reward",
        "raw_reward",
        "R_format",
        "R_skeleton_iou",
        "R_contour_iou",
        "R_consistency_iou",
        "R_merge_iou",
        "P_overlap",
        "P_area",
        "skeleton_gate",
    )
    _FLAG_KEYS = ("valid", "skeleton_missing")

    def update(self, infos: Sequence[Dict[str, Any]]) -> None:
        for info in infos:
            self.count += 1
            for key in self._NUMERIC_KEYS:
                if key in info:
                    self.sums[key] = self.sums.get(key, 0.0) + float(info[key])
            for key in self._FLAG_KEYS:
                self.sums[key] = self.sums.get(key, 0.0) + float(bool(info.get(key)))

    def means(self, prefix: str = "") -> Dict[str, float]:
        if self.count == 0:
            return {}
        out = {f"{prefix}{k}": v / self.count for k, v in self.sums.items()}
        out[f"{prefix}skeleton_missing_rate"] = out.pop(f"{prefix}skeleton_missing", 0.0)
        out[f"{prefix}format_valid_rate"] = out.pop(f"{prefix}valid", 0.0)
        return out

    def reset(self) -> None:
        self.count = 0
        self.sums = {}


# Populated by every grpo_reward_func call so a trainer callback can log the
# per-term breakdown without recomputing anything.
LAST_BREAKDOWNS: List[Dict[str, Any]] = []
GLOBAL_STATS = RewardStats()


def _samples_from_kwargs(batch_size: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """Rebuild per-completion ground truth from TRL's column-wise kwargs."""
    explicit = kwargs.get("samples") or kwargs.get("sample")
    if explicit is not None:
        if isinstance(explicit, dict):
            explicit = [explicit] * batch_size
        samples = list(explicit)
    else:
        columns = {key: kwargs.get(key) for key in SAMPLE_KEYS}
        samples = []
        for index in range(batch_size):
            sample: Dict[str, Any] = {}
            for key, values in columns.items():
                if values is None:
                    sample[key] = ""
                elif isinstance(values, str):
                    sample[key] = values
                else:
                    sample[key] = values[index] if index < len(values) else ""
            sample["uid"] = _column_value(kwargs.get("uid"), index)
            samples.append(sample)

    # TRL repeats each prompt num_generations times; broadcast if needed.
    if len(samples) < batch_size and samples:
        repeat = batch_size // len(samples)
        if repeat * len(samples) == batch_size:
            samples = [s for s in samples for _ in range(repeat)]
    while len(samples) < batch_size:
        samples.append({key: "" for key in SAMPLE_KEYS})
    return samples[:batch_size]


def _column_value(values: Any, index: int) -> Any:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    try:
        return values[index]
    except Exception:  # noqa: BLE001
        return ""


def grpo_reward_func(
    prompts: Optional[Sequence[Any]] = None,
    completions: Optional[Sequence[Any]] = None,
    config: Optional[RewardConfig] = None,
    **kwargs: Any,
) -> List[float]:
    """Reward function with the signature expected by TRL's ``GRPOTrainer``.

    The ground truth for each completion is read from the dataset columns that
    TRL forwards as keyword arguments (``damaged_svg``, ``gt_full_svg``,
    ``gt_missing_contour_svg``, ``gt_skeleton_svg``).  A pre-built ``samples``
    kwarg is also accepted for direct use outside TRL.

    Returns:
        One float per completion.  Never raises.
    """
    cfg = _resolve(config)
    completions = list(completions or [])
    samples = _samples_from_kwargs(len(completions), **kwargs)

    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    for completion, sample in zip(completions, samples):
        reward, info = reward_one_completion(completion, sample, cfg, return_info=True)
        rewards.append(float(reward))
        infos.append(info)

    LAST_BREAKDOWNS.clear()
    LAST_BREAKDOWNS.extend(infos)
    GLOBAL_STATS.update(infos)
    return rewards


def reward_batch(
    completions: Sequence[Any],
    samples: Sequence[Dict[str, Any]],
    config: Optional[RewardConfig] = None,
) -> tuple:
    """Score a batch and return ``(rewards, infos)``."""
    cfg = _resolve(config)
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    for completion, sample in zip(completions, samples):
        reward, info = reward_one_completion(completion, sample, cfg, return_info=True)
        rewards.append(float(reward))
        infos.append(info)
    return rewards, infos
