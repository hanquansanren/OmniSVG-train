#!/usr/bin/env python
"""Unit tests for the GRPO mask-IoU reward pipeline.

Run with either runner::

    python -m unittest discover -s tests -v
    python tests/test_grpo_reward.py

The central test builds five completion classes for the same sample and checks
the expected reward ordering:

==  ===========================================  ==========================
id  completion                                   expectation
==  ===========================================  ==========================
A   correct skeleton + correct contour           highest
C   wrong skeleton + correct contour             mid
D   correct skeleton + wrong contour             mid
B   empty skeleton + correct contour             low (skeleton gate applies)
E   malformed SVG                                -1.0
==  ===========================================  ==========================
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards import (  # noqa: E402
    RewardConfig,
    compute_area_penalty,
    compute_contour_iou,
    compute_merge_iou,
    compute_overlap_penalty,
    compute_skeleton_contour_consistency,
    compute_skeleton_iou,
    grpo_reward_func,
    load_samples,
    mask_area,
    mask_iou,
    oracle_completion,
    parse_model_output,
    rasterize_svg_to_mask,
    reward_one_completion,
    wrap_paths,
)
from rewards.raster import empty_mask, get_backend, set_backend  # noqa: E402

DATA_DIR = os.environ.get(
    "GRPO_TEST_DATA_DIR", "/data/phd23_weiguang_zhang/works/svg/my_lis2_2"
)

# --------------------------------------------------------------------------- #
# Synthetic fixture: a horizontal bar missing from a vertical bar glyph.
# --------------------------------------------------------------------------- #

DAMAGED_PATHS = ["M95 20 L105 20 L105 180 L95 180 Z"]
MISSING_CONTOUR_PATHS = ["M40 92 L160 92 L160 108 L40 108 Z"]
SKELETON_PATHS = ["M40 100 L160 100"]

WRONG_SKELETON = "M20 20 L60 25 L80 40"
WRONG_CONTOUR = "M10 150 L50 150 L50 190 L10 190 Z"


def synthetic_sample() -> dict:
    return {
        "uid": "synthetic_bar",
        "damaged_svg": wrap_paths(DAMAGED_PATHS),
        "gt_missing_contour_svg": wrap_paths(MISSING_CONTOUR_PATHS),
        "gt_skeleton_svg": wrap_paths(SKELETON_PATHS),
        "gt_full_svg": wrap_paths(DAMAGED_PATHS + MISSING_CONTOUR_PATHS),
        "gt_skeleton_path_data": SKELETON_PATHS,
        "gt_contour_path_data": MISSING_CONTOUR_PATHS,
    }


def tagged(skeleton: str, contour: str) -> str:
    return f"<skeleton>{skeleton}</skeleton><contour>{contour}</contour>"


def completion_variants() -> dict:
    skeleton = " ".join(SKELETON_PATHS)
    contour = " ".join(MISSING_CONTOUR_PATHS)
    return {
        "A": tagged(skeleton, contour),
        "B": tagged("", contour),
        "C": tagged(WRONG_SKELETON, contour),
        "D": tagged(skeleton, WRONG_CONTOUR),
        "E": tagged("M10 10 L oops", "<path d='M this is not svg'/>"),
    }


class TestRasterization(unittest.TestCase):
    def test_backend_is_reported(self):
        self.assertIn(get_backend(), ("cairosvg", "pil"))

    def test_fill_mask_area_matches_geometry(self):
        # 120x16 user units on a 200-unit viewBox rendered at 256px.
        svg = wrap_paths(MISSING_CONTOUR_PATHS)
        mask = rasterize_svg_to_mask(svg, canvas_size=256, mode="fill")
        scale = 256 / 200
        expected = 120 * 16 * scale * scale
        self.assertAlmostEqual(mask_area(mask) / expected, 1.0, delta=0.05)

    def test_stroke_mask_is_thin(self):
        svg = wrap_paths(SKELETON_PATHS)
        fill_like = rasterize_svg_to_mask(
            wrap_paths(MISSING_CONTOUR_PATHS), mode="fill"
        )
        stroke = rasterize_svg_to_mask(svg, mode="stroke", stroke_width=3)
        self.assertGreater(mask_area(stroke), 0)
        self.assertLess(mask_area(stroke), mask_area(fill_like))

    def test_unified_canvas_regardless_of_input_form(self):
        bare_d = " ".join(MISSING_CONTOUR_PATHS)
        document = wrap_paths(MISSING_CONTOUR_PATHS)
        fragment = f'<path fill="#000" d="{bare_d}"/>'
        masks = [
            rasterize_svg_to_mask(candidate, mode="fill")
            for candidate in (bare_d, document, fragment)
        ]
        for mask in masks:
            self.assertEqual(mask.shape, masks[0].shape)
            self.assertGreaterEqual(mask_iou(mask, masks[0]), 0.99)

    def test_bad_svg_returns_empty_mask_instead_of_raising(self):
        for bad in ("M10 10 L nope", "<svg><path d='@@@'/></svg>", "", None):
            mask = rasterize_svg_to_mask(bad, mode="fill")
            self.assertEqual(mask.shape, (256, 256))
            self.assertEqual(mask_area(mask), 0)

    def test_both_backends_agree_on_fill_area(self):
        # Exact agreement is not expected: cairo antialiases and samples pixel
        # centres while the fallback thresholds a supersampled polygon fill.
        # Agreement only has to be close enough that a switch of backend does
        # not change which completion wins.
        original = get_backend()
        svg = wrap_paths(MISSING_CONTOUR_PATHS)
        try:
            set_backend("pil")
            pil_mask = rasterize_svg_to_mask(svg, mode="fill")
            set_backend("cairosvg")
            try:
                cairo_mask = rasterize_svg_to_mask(svg, mode="fill")
            except Exception:  # pragma: no cover - cairo not installed
                self.skipTest("cairosvg unavailable")
            if mask_area(cairo_mask) == 0:
                self.skipTest("cairosvg unavailable")
            self.assertGreaterEqual(mask_iou(pil_mask, cairo_mask), 0.92)
        finally:
            set_backend(original)

    def test_reward_ordering_holds_under_the_pil_fallback(self):
        original = get_backend()
        sample = synthetic_sample()
        try:
            set_backend("pil")
            variants = completion_variants()
            rewards = {
                key: reward_one_completion(text, sample) for key, text in variants.items()
            }
            self.assertGreater(rewards["A"], rewards["C"])
            self.assertGreater(rewards["C"], rewards["B"])
            self.assertGreater(rewards["B"], rewards["E"])
        finally:
            set_backend(original)


class TestMaskIoU(unittest.TestCase):
    def test_identical_masks(self):
        mask = rasterize_svg_to_mask(wrap_paths(MISSING_CONTOUR_PATHS), mode="fill")
        self.assertEqual(mask_iou(mask, mask), 1.0)

    def test_both_empty_is_perfect_agreement(self):
        self.assertEqual(mask_iou(empty_mask(), empty_mask()), 1.0)

    def test_one_empty_is_zero(self):
        mask = rasterize_svg_to_mask(wrap_paths(MISSING_CONTOUR_PATHS), mode="fill")
        self.assertEqual(mask_iou(mask, empty_mask()), 0.0)

    def test_disjoint_masks(self):
        a = rasterize_svg_to_mask("M10 10 L40 10 L40 40 L10 40 Z", mode="fill")
        b = rasterize_svg_to_mask("M120 120 L160 120 L160 160 L120 160 Z", mode="fill")
        self.assertEqual(mask_iou(a, b), 0.0)

    def test_shape_mismatch_is_zero(self):
        a = rasterize_svg_to_mask("M10 10 L40 10 L40 40 Z", canvas_size=128, mode="fill")
        b = rasterize_svg_to_mask("M10 10 L40 10 L40 40 Z", canvas_size=256, mode="fill")
        self.assertEqual(mask_iou(a, b), 0.0)


class TestParseModelOutput(unittest.TestCase):
    def test_tagged_text(self):
        parsed = parse_model_output(completion_variants()["A"])
        self.assertTrue(parsed.valid)
        self.assertIsNone(parsed.error)
        self.assertFalse(parsed.skeleton_empty)
        self.assertIn("<path", parsed.skeleton_svg)
        self.assertIn("<path", parsed.contour_svg)

    def test_marker_text(self):
        parsed = parse_model_output(oracle_completion(synthetic_sample(), "markers"))
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.source, "markers")

    def test_multiple_sections_are_merged(self):
        text = (
            "<skeleton>M10 10 L40 10</skeleton><contour>M10 10 L40 10 L40 20 Z</contour>"
            "<skeleton>M60 60 L90 60</skeleton><contour>M60 60 L90 60 L90 70 Z</contour>"
        )
        parsed = parse_model_output(text)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.n_skeleton_sections, 2)
        self.assertEqual(parsed.n_contour_sections, 2)
        self.assertEqual(len(parsed.contour_path_data), 2)

    def test_empty_skeleton_is_valid_but_flagged(self):
        parsed = parse_model_output(completion_variants()["B"])
        self.assertTrue(parsed.valid)
        self.assertTrue(parsed.skeleton_empty)

    def test_missing_tags_are_invalid(self):
        self.assertFalse(parse_model_output("<contour>M10 10 L20 20 Z</contour>").valid)
        self.assertFalse(parse_model_output("<skeleton>M10 10 L20 20</skeleton>").valid)
        self.assertFalse(parse_model_output("just some prose").valid)
        self.assertFalse(parse_model_output("").valid)

    def test_unparsable_path_is_invalid(self):
        parsed = parse_model_output(completion_variants()["E"])
        self.assertFalse(parsed.valid)
        self.assertIsNotNone(parsed.error)

    def test_never_raises_on_odd_input(self):
        for payload in (None, 12345, {"content": "x"}, [{"role": "a", "content": "b"}]):
            parse_model_output(payload)


class TestRewardTerms(unittest.TestCase):
    def setUp(self):
        self.sample = synthetic_sample()
        self.config = RewardConfig()

    def test_skeleton_iou_perfect_and_zero(self):
        gt = self.sample["gt_skeleton_svg"]
        self.assertAlmostEqual(compute_skeleton_iou(gt, gt, self.config), 1.0, places=6)
        self.assertLess(compute_skeleton_iou(WRONG_SKELETON, gt, self.config), 0.05)
        self.assertEqual(compute_skeleton_iou("", gt, self.config), 0.0)

    def test_contour_iou_perfect_and_zero(self):
        gt = self.sample["gt_missing_contour_svg"]
        self.assertAlmostEqual(compute_contour_iou(gt, gt, self.config), 1.0, places=6)
        self.assertEqual(compute_contour_iou(WRONG_CONTOUR, gt, self.config), 0.0)

    def test_consistency_rewards_a_matching_skeleton(self):
        skeleton = self.sample["gt_skeleton_svg"]
        contour = self.sample["gt_missing_contour_svg"]
        matched = compute_skeleton_contour_consistency(skeleton, contour, self.config)
        empty = compute_skeleton_contour_consistency("", contour, self.config)
        misplaced = compute_skeleton_contour_consistency(
            WRONG_SKELETON, contour, self.config
        )
        self.assertGreater(matched, 0.3)
        self.assertEqual(empty, 0.0)
        self.assertLess(misplaced, matched)

    def test_merge_iou_reaches_one_for_the_ground_truth(self):
        merged = compute_merge_iou(
            self.sample["damaged_svg"],
            self.sample["gt_missing_contour_svg"],
            self.sample["gt_full_svg"],
            self.config,
        )
        self.assertGreater(merged, 0.99)

    def test_overlap_penalty_punishes_redrawing_the_damaged_glyph(self):
        clean = compute_overlap_penalty(
            self.sample["damaged_svg"], self.sample["gt_missing_contour_svg"], self.config
        )
        redrawn = compute_overlap_penalty(
            self.sample["damaged_svg"], self.sample["damaged_svg"], self.config
        )
        self.assertLess(clean, 0.1)
        self.assertAlmostEqual(redrawn, 1.0, places=6)

    def test_area_penalty_is_zero_at_the_ground_truth_and_clipped_at_one(self):
        gt = self.sample["gt_missing_contour_svg"]
        self.assertAlmostEqual(compute_area_penalty(gt, gt, self.config), 0.0, places=6)
        self.assertEqual(compute_area_penalty("", gt, self.config), 1.0)
        huge = "M0 0 L200 0 L200 200 L0 200 Z"
        self.assertEqual(compute_area_penalty(huge, gt, self.config), 1.0)
        # A 2x area mismatch sits strictly between the extremes.
        doubled = "M40 84 L160 84 L160 116 L40 116 Z"
        penalty = compute_area_penalty(doubled, gt, self.config)
        self.assertGreater(penalty, 0.5)
        self.assertLess(penalty, 0.8)


class TestRewardOrdering(unittest.TestCase):
    """A > C/D > B > E, on both the synthetic fixture and real data."""

    def _check_ordering(self, sample: dict, label: str):
        rewards = {}
        infos = {}
        for key, completion in completion_variants_for(sample).items():
            reward, info = reward_one_completion(completion, sample, return_info=True)
            rewards[key] = reward
            infos[key] = info

        message = f"[{label}] " + "  ".join(f"{k}={v:.4f}" for k, v in rewards.items())

        self.assertTrue(infos["A"]["valid"], message)
        self.assertFalse(infos["E"]["valid"], message)
        self.assertEqual(rewards["E"], -1.0, message)

        self.assertGreater(rewards["A"], rewards["C"], message)
        self.assertGreater(rewards["A"], rewards["D"], message)
        self.assertGreater(rewards["C"], rewards["B"], message)
        self.assertGreater(rewards["D"], rewards["B"], message)
        self.assertGreater(rewards["B"], rewards["E"], message)

        # The gate is what pushes B below a merely wrong skeleton.
        self.assertTrue(infos["B"]["skeleton_missing"], message)
        self.assertAlmostEqual(infos["B"]["skeleton_gate"], 0.1, places=6)
        self.assertFalse(infos["A"]["skeleton_missing"], message)

        for reward in rewards.values():
            self.assertGreaterEqual(reward, -1.0)
            self.assertLessEqual(reward, 1.0)

    def test_synthetic_sample(self):
        self._check_ordering(synthetic_sample(), "synthetic")

    def test_real_sample(self):
        samples = _real_samples(1)
        if not samples:
            self.skipTest(f"no stage-2 data available at {DATA_DIR}")
        self._check_ordering(samples[0], samples[0]["uid"])


class TestGRPORewardFunc(unittest.TestCase):
    def test_trl_style_column_kwargs(self):
        sample = synthetic_sample()
        variants = completion_variants()
        completions = [variants[k] for k in ("A", "B", "C", "D", "E")]
        rewards = grpo_reward_func(
            prompts=["p"] * len(completions),
            completions=completions,
            damaged_svg=[sample["damaged_svg"]] * len(completions),
            gt_full_svg=[sample["gt_full_svg"]] * len(completions),
            gt_missing_contour_svg=[sample["gt_missing_contour_svg"]] * len(completions),
            gt_skeleton_svg=[sample["gt_skeleton_svg"]] * len(completions),
        )
        self.assertEqual(len(rewards), len(completions))
        self.assertTrue(all(isinstance(r, float) for r in rewards))
        self.assertGreater(rewards[0], rewards[1])
        self.assertEqual(rewards[-1], -1.0)

    def test_prompt_columns_are_broadcast_across_generations(self):
        sample = synthetic_sample()
        variants = completion_variants()
        completions = [variants["A"], variants["B"]] * 2  # 2 prompts x 2 generations
        rewards = grpo_reward_func(
            prompts=["p1", "p2"],
            completions=completions,
            damaged_svg=[sample["damaged_svg"], sample["damaged_svg"]],
            gt_full_svg=[sample["gt_full_svg"], sample["gt_full_svg"]],
            gt_missing_contour_svg=[sample["gt_missing_contour_svg"]] * 2,
            gt_skeleton_svg=[sample["gt_skeleton_svg"]] * 2,
        )
        self.assertEqual(len(rewards), 4)

    def test_samples_kwarg(self):
        sample = synthetic_sample()
        rewards = grpo_reward_func(
            completions=[completion_variants()["A"]], samples=[sample]
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.0)

    def test_missing_ground_truth_does_not_raise(self):
        rewards = grpo_reward_func(completions=[completion_variants()["A"]])
        self.assertEqual(len(rewards), 1)


class TestRealDataOracle(unittest.TestCase):
    def test_oracle_scores_well_above_degenerate_completions(self):
        samples = _real_samples(5)
        if not samples:
            self.skipTest(f"no stage-2 data available at {DATA_DIR}")
        for sample in samples:
            oracle = reward_one_completion(oracle_completion(sample), sample)
            empty_skeleton = reward_one_completion(
                tagged("", " ".join(sample["gt_contour_path_data"])), sample
            )
            self.assertGreater(oracle, 0.5, sample["uid"])
            self.assertGreater(oracle, empty_skeleton, sample["uid"])

    def test_ground_truth_merge_iou_is_near_one(self):
        samples = _real_samples(5)
        if not samples:
            self.skipTest(f"no stage-2 data available at {DATA_DIR}")
        for sample in samples:
            _, info = reward_one_completion(
                oracle_completion(sample), sample, return_info=True
            )
            self.assertGreater(info["R_merge_iou"], 0.95, sample["uid"])
            self.assertGreater(info["R_contour_iou"], 0.95, sample["uid"])
            self.assertGreater(info["R_skeleton_iou"], 0.95, sample["uid"])


def completion_variants_for(sample: dict) -> dict:
    """Build the five completion classes for an arbitrary sample."""
    skeleton = " ".join(sample.get("gt_skeleton_path_data") or [])
    contour = " ".join(sample.get("gt_contour_path_data") or [])
    return {
        "A": tagged(skeleton, contour),
        "B": tagged("", contour),
        "C": tagged(WRONG_SKELETON, contour),
        "D": tagged(skeleton, WRONG_CONTOUR),
        "E": tagged("M10 10 L oops", "<path d='M this is not svg'/>"),
    }


_REAL_SAMPLES_CACHE: dict = {}


def _real_samples(count: int) -> list:
    if count not in _REAL_SAMPLES_CACHE:
        try:
            _REAL_SAMPLES_CACHE[count] = load_samples(
                DATA_DIR, split="val", limit=count, require_image=False
            )
        except Exception:  # noqa: BLE001
            _REAL_SAMPLES_CACHE[count] = []
    return _REAL_SAMPLES_CACHE[count]


if __name__ == "__main__":
    unittest.main(verbosity=2)
