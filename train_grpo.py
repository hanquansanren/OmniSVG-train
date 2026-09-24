#!/usr/bin/env python
"""GRPO fine-tuning of the stage-2 SVG completion model with mask-IoU rewards.

Why not ``trl.GRPOTrainer``?
---------------------------
``GRPOTrainer`` assumes the policy is a text model: it decodes completions with
a ``processing_class`` and feeds the resulting *strings* to the reward
functions.  This project's policy is ``decoder.SketchDecoder`` - a Qwen2.5-VL
backbone with a 197004-token vocabulary whose SVG tokens have no text form, a
prompt that is ``chat template + image + partial-SVG token ids`` concatenated by
hand, and a custom ``forward`` that recomputes RoPE indices.  So this file
implements the GRPO update directly against ``SketchDecoder``.

``rewards.grpo_reward_func`` still carries TRL's exact
``(prompts, completions, **kwargs) -> list[float]`` signature, so the reward
pipeline can be dropped into ``GRPOTrainer`` unchanged if the policy is ever
replaced by a plain text model.

The GRPO update
---------------
For each prompt, ``num_generations`` completions are sampled, scored, and turned
into group-relative advantages ``(r - mean) / (std + eps)``.  The per-token loss
is the usual clipped-free single-inner-step objective plus a k3 KL estimator
against the SFT reference::

    coef  = exp(logp - logp.detach())              # 1.0, keeps the grad exact
    kl    = exp(ref - logp) - (ref - logp) - 1     # Schulman k3, always >= 0
    loss  = mean_t[ -coef * advantage + beta * kl ]

Memory notes for 24 GB cards
----------------------------
A full second copy of the model for the KL reference does not fit next to
AdamW's fp32 moments.  ``--ref-mode swap`` (the default) instead keeps a bf16
backup of only the *trainable* parameters and swaps it in for the reference
forward pass, which is exact because every frozen parameter is shared.
Combined with ``--train-last-layers`` this keeps a 3B policy inside ~20 GB.

Typical usage
-------------
1. Validate the reward pipeline on real data without touching a GPU::

       python train_grpo.py --reward-smoke 100

2. Smoke-test the full loop for a few hundred updates::

       CUDA_VISIBLE_DEVICES=2 python train_grpo.py --max-steps 300

3. Or drive it through the wrapper script::

       CUDA_VISIBLE_DEVICES=2 bash scripts/grpo_run.sh

4. Data-parallel over several GPUs (each rank samples its own prompts, so one
   optimizer step sees ``nproc * --prompts-per-step`` groups)::

       torchrun --nproc_per_node 8 train_grpo.py --max-steps 300
       NUM_GPUS=8 bash scripts/grpo_run.sh
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rewards import (  # noqa: E402
    RewardConfig,
    RewardStats,
    iter_samples,
    load_samples,
    oracle_completion,
    reward_batch,
    reward_one_completion,
)
from rewards.parse import (  # noqa: E402
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    REPLACEMENT_START_TOKEN_ID,
    SKELETON_START_TOKEN_ID,
)

logger = logging.getLogger("train_grpo")

DEFAULT_DATA_DIR = "/data/phd23_weiguang_zhang/works/svg/my_lis2_2"
DEFAULT_SFT_CHECKPOINT = "output_stage2/omnisvg_4b_20260708_091120/step_30000"
BASE_MODEL_CANDIDATES = (
    "/data/phd23_weiguang_zhang/works/svg/qwen25vl3b",
    "/home/bingxing2/home/scx7l3f/weiguang_zhang/project/weights/qwen25vl3b",
    "/gpfs/work/int/weiguangzhang21/weights/qwen25vl3b",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    data.add_argument("--split", default="train", choices=("train", "val"))
    data.add_argument("--eval-split", default="val", choices=("train", "val"))
    data.add_argument("--num-samples", type=int, default=None,
                      help="Cap on prompts loaded from the split (None = all)")
    data.add_argument("--seed", type=int, default=2023)

    model = parser.add_argument_group("model")
    model.add_argument("--model-size", default="4B", choices=("4B", "8B"))
    model.add_argument("--base-model", default=None,
                       help="Qwen2.5-VL base dir; auto-detected when omitted")
    model.add_argument("--sft-checkpoint", default=DEFAULT_SFT_CHECKPOINT,
                       help="Best SFT checkpoint used to initialise the policy")
    model.add_argument("--no-flash-attn", action="store_true")
    model.add_argument("--no-gradient-checkpointing", action="store_true")
    model.add_argument("--condition-max-length", type=int, default=1024,
                       help="Token budget for the partial-SVG condition")

    grpo = parser.add_argument_group("GRPO")
    grpo.add_argument("--num-generations", type=int, default=4,
                      help="Completions sampled per prompt (the GRPO group size)")
    grpo.add_argument("--prompts-per-step", type=int, default=1,
                      help="Prompts (groups) accumulated into one optimizer step")
    grpo.add_argument("--learning-rate", type=float, default=5e-7,
                      help="5e-7 to 1e-6 is the usable band for this policy")
    grpo.add_argument("--beta", type=float, default=0.02,
                      help="KL coefficient against the SFT reference (0.01-0.05)")
    grpo.add_argument("--max-completion-length", type=int, default=2048,
                      help="Covers the full GT target length distribution; "
                           "check with --measure-target-length")
    grpo.add_argument("--max-steps", type=int, default=500)
    grpo.add_argument("--warmup-steps", type=int, default=20)
    grpo.add_argument("--max-grad-norm", type=float, default=1.0)
    grpo.add_argument("--weight-decay", type=float, default=0.0)
    grpo.add_argument("--no-scale-rewards", action="store_true",
                      help="Centre advantages without dividing by the group std")
    grpo.add_argument("--ref-mode", default="swap", choices=("swap", "full", "none"),
                      help="How to obtain reference logprobs for the KL term")
    grpo.add_argument("--train-last-layers", type=int, default=8,
                      help="Unfreeze only the top N decoder layers (0 = all layers)")
    grpo.add_argument("--train-embeddings", action="store_true",
                      help="Also unfreeze the token embeddings / lm_head")
    grpo.add_argument("--optimizer", default="adamw", choices=("adamw", "adafactor", "sgd"))
    grpo.add_argument("--no-fp32-master-weights", action="store_true",
                      help="Step the optimizer on the bf16 weights directly.  At GRPO "
                           "learning rates almost every update then rounds to zero")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--temperature", type=float, default=0.30,
                          help="Higher than inference on purpose: GRPO needs spread")
    sampling.add_argument("--top-p", type=float, default=0.95)
    sampling.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    sampling.add_argument("--repetition-penalty", type=float, default=1.0)

    reward = parser.add_argument_group("reward")
    reward.add_argument("--reward-config", default=None,
                        help="YAML file with a `reward:` section overriding the weights")
    reward.add_argument("--canvas-size", type=int, default=None)
    reward.add_argument("--consistency-mode", default=None,
                        choices=("iou", "dice", "coverage"))

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", default="./output_grpo")
    run.add_argument("--project-name", default=None)
    run.add_argument("--log-every", type=int, default=5)
    run.add_argument("--save-every", type=int, default=100,
                     help="Checkpoint interval; 0 disables saving entirely")
    run.add_argument("--eval-every", type=int, default=0,
                     help="Greedy eval on held-out prompts every N steps (0 = off)")
    run.add_argument("--eval-samples", type=int, default=64,
                     help="One malformed completion moves the mean by ~1.4/N")
    run.add_argument("--eval-temperature", type=float, default=0.01)
    run.add_argument("--eval-top-p", type=float, default=0.3)
    run.add_argument("--eval-top-k", type=int, default=5)
    run.add_argument("--dist-timeout-minutes", type=int, default=60,
                     help="NCCL collective timeout under torchrun; ranks wait for "
                          "the slowest 2048-token generation and for rank-0 saves")
    run.add_argument("--dump-completions", type=int, default=2,
                     help="Completions per logged step written to completions.jsonl")
    run.add_argument("--use-wandb", action="store_true")
    run.add_argument("--wandb-project", default="omnisvg-grpo")
    run.add_argument("--reward-smoke", type=int, default=0,
                     help="Score N samples with oracle/degenerate completions and exit")
    run.add_argument("--measure-target-length", type=int, default=0,
                     help="Report the GT completion token length over N samples and exit")
    run.add_argument("--probe-sampling", type=int, default=0,
                     help="Sweep sampling temperatures over N prompts and exit")
    run.add_argument("--probe-temperatures", default="0.2,0.4,0.6,0.8,1.0",
                     help="Comma-separated temperatures for --probe-sampling")
    run.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Reward configuration
# --------------------------------------------------------------------------- #


def build_reward_config(args: argparse.Namespace) -> RewardConfig:
    config = RewardConfig.from_yaml(args.reward_config) if args.reward_config else RewardConfig()
    if args.canvas_size:
        config.canvas_size = args.canvas_size
    if args.consistency_mode:
        config.consistency_mode = args.consistency_mode
    config.model_size = args.model_size
    return config


# --------------------------------------------------------------------------- #
# Reward smoke test (no GPU required)
# --------------------------------------------------------------------------- #


def degenerate_completions(sample: Dict[str, Any]) -> Dict[str, str]:
    """Completion variants used to probe the reward's dynamic range."""
    skeleton = " ".join(sample.get("gt_skeleton_path_data") or [])
    contour = " ".join(sample.get("gt_contour_path_data") or [])
    return {
        "oracle": oracle_completion(sample),
        "empty_skeleton": f"<skeleton></skeleton><contour>{contour}</contour>",
        "skeleton_only": f"<skeleton>{skeleton}</skeleton><contour></contour>",
        "wrong_contour": (
            f"<skeleton>{skeleton}</skeleton>"
            "<contour>M10 150 L60 150 L60 190 L10 190 Z</contour>"
        ),
        "redraw_damaged": (
            f"<skeleton>{skeleton}</skeleton>"
            f"<contour>{' '.join(_damaged_path_data(sample))}</contour>"
        ),
        "malformed": "<skeleton>M10 10 L oops</skeleton><contour>@@@</contour>",
    }


def _damaged_path_data(sample: Dict[str, Any]) -> List[str]:
    from rewards.raster import extract_path_data

    return extract_path_data(sample.get("damaged_svg") or "")[:4]


def run_reward_smoke(args: argparse.Namespace, config: RewardConfig) -> int:
    """Score N real samples with synthetic completions and report the spread.

    This is the first thing to run: it confirms the ground truth itself scores
    near the reward ceiling, that degenerate answers score far below it, and
    that nothing in the pipeline raises.
    """
    from rewards.raster import get_backend

    print("=" * 74)
    print(f"Reward smoke test on {args.reward_smoke} samples from {args.data_dir}")
    print(f"raster backend={get_backend()}  canvas={config.canvas_size}  "
          f"consistency_mode={config.consistency_mode}")
    print("=" * 74)

    samples = load_samples(
        args.data_dir,
        split=args.eval_split,
        limit=args.reward_smoke,
        shuffle_seed=args.seed,
        require_image=False,
    )
    if not samples:
        print(f"ERROR: no samples found in {args.data_dir}")
        return 1

    variants = list(degenerate_completions(samples[0]).keys())
    per_variant: Dict[str, List[float]] = {name: [] for name in variants}
    per_variant_info: Dict[str, RewardStats] = {name: RewardStats() for name in variants}

    started = time.time()
    failures: List[str] = []
    for sample in samples:
        completions = degenerate_completions(sample)
        for name, completion in completions.items():
            score, info = reward_one_completion(completion, sample, config, return_info=True)
            per_variant[name].append(score)
            per_variant_info[name].update([info])
            if name == "oracle" and (not info["valid"] or score < 0.3):
                failures.append(f"{sample['uid']} oracle={score:.3f} error={info['error']}")

    elapsed = time.time() - started
    total = len(samples) * len(variants)
    print(f"\nscored {total} completions in {elapsed:.1f}s "
          f"({1000 * elapsed / max(total, 1):.1f} ms each)\n")

    header = (f"{'variant':16s} {'reward':>16s} {'skel':>6s} {'cont':>6s} "
              f"{'cons':>6s} {'merge':>6s} {'ovl':>6s} {'area':>6s} {'miss%':>6s}")
    print(header)
    print("-" * len(header))
    for name in variants:
        scores = per_variant[name]
        means = per_variant_info[name].means()
        spread = f"{statistics.mean(scores):+.3f} ± {statistics.pstdev(scores):.3f}"
        print(
            f"{name:16s} {spread:>16s} "
            f"{means.get('R_skeleton_iou', 0):6.3f} {means.get('R_contour_iou', 0):6.3f} "
            f"{means.get('R_consistency_iou', 0):6.3f} {means.get('R_merge_iou', 0):6.3f} "
            f"{means.get('P_overlap', 0):6.3f} {means.get('P_area', 0):6.3f} "
            f"{100 * means.get('skeleton_missing_rate', 0):6.1f}"
        )

    oracle = statistics.mean(per_variant["oracle"])
    empty = statistics.mean(per_variant["empty_skeleton"])
    print(f"\noracle mean reward          : {oracle:+.4f}   <- the practical ceiling")
    print(f"empty-skeleton mean reward  : {empty:+.4f}   <- what the gate costs")
    print(f"oracle - empty_skeleton gap : {oracle - empty:+.4f}")

    if failures:
        print(f"\n{len(failures)} sample(s) where the ground truth scored poorly:")
        for line in failures[:10]:
            print(f"  {line}")
    ok = oracle > 0.5 and oracle - empty > 0.3 and not failures
    print(f"\nsmoke test: {'PASS' if ok else 'CHECK THE ABOVE'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


class PromptBuilder:
    """Build model inputs for one code-complement sample.

    Reproduces the stage-2 prompt exactly: the instruction and image go through
    the chat template (``train.py::create_collate_fn``), then the partial SVG is
    appended as raw token ids (``inference.py::prepare_code_complement_inputs``).
    """

    SYSTEM_PROMPT = "You are an expert SVG code generator."

    def __init__(
        self,
        processor: Any,
        token_config: Any,
        target_image_size: int = 448,
        condition_max_length: Optional[int] = None,
    ) -> None:
        import torch
        from utils.dataset import SVGTokenizer as TrainingSVGEncoder

        self.torch = torch
        self.processor = processor
        self.encoder = TrainingSVGEncoder(token_config)
        self.target_image_size = target_image_size
        self.condition_max_length = condition_max_length

    def _load_image(self, sample: Dict[str, Any]):
        from PIL import Image

        path = sample.get("image_path")
        if path and os.path.isfile(path):
            image = Image.open(path)
        else:
            image = Image.new(
                "RGB", (self.target_image_size, self.target_image_size), "white"
            )
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image = image.resize(
            (self.target_image_size, self.target_image_size), Image.Resampling.LANCZOS
        )
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def _condition_tokens(self, sample: Dict[str, Any]) -> List[int]:
        from deepsvg.svglib.svg import SVG as DeepSVG

        svg = DeepSVG.load_svg(sample["svg_path"])
        svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
        tokens = self.encoder.tokenize_svg_tensors(svg_tensors, color_tensors)
        tokens = self.encoder.add_special_tokens(tokens).tolist()
        if self.condition_max_length and len(tokens) > self.condition_max_length:
            # Keep the tail, matching the truncation in train.py::_process_sample.
            tokens = tokens[-self.condition_max_length :]
        return tokens

    def build(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        torch = self.torch
        image = self._load_image(sample)
        instruction = (
            "Complete the missing SVG path code for this Chinese glyph.\n"
            f"Character: {sample.get('char_label', '')}\n"
            "The partial SVG S_d is provided after this instruction as SVG tokens. "
            "The attached PNG image I_d shows the corresponding incomplete glyph. "
            "Output only the SVG path code fragments that should be appended to S_d."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        condition = self._condition_tokens(sample)
        condition_ids = torch.tensor([condition], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], condition_ids], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(condition_ids)], dim=1
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "condition_length": len(condition),
        }


# --------------------------------------------------------------------------- #
# Model setup
# --------------------------------------------------------------------------- #


def resolve_base_model(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in BASE_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(
        "could not locate the Qwen2.5-VL base model; pass --base-model explicitly"
    )


def find_decoder_layers(transformer: Any) -> Tuple[str, Any]:
    """Locate the decoder ``ModuleList`` across transformers layouts."""
    import torch.nn as nn

    for path in (
        "model.language_model.layers",
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
    ):
        node = transformer
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList):
            return path, node
    raise RuntimeError("could not find the decoder layers on the policy model")


def configure_trainable(model: Any, args: argparse.Namespace) -> List[Tuple[str, Any]]:
    """Freeze everything, then unfreeze the top decoder layers (and optionally
    the embeddings).  Returns the trainable ``(name, parameter)`` pairs.

    Partial fine-tuning is not just a memory trick here: GRPO at 5e-7 only needs
    to nudge the policy, and keeping the vision tower and lower layers frozen
    removes most of the drift risk on a 60k-sample dataset.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    path, layers = find_decoder_layers(model.transformer)
    total_layers = len(layers)
    if args.train_last_layers and args.train_last_layers > 0:
        first_trainable = max(0, total_layers - args.train_last_layers)
        for layer in layers[first_trainable:]:
            for param in layer.parameters():
                param.requires_grad_(True)
    else:
        first_trainable = 0
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad_(True)

    # The final norm sits next to the trainable block and is cheap to include.
    for attr in ("model.language_model.norm", "model.norm"):
        node = model.transformer
        for part in attr.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            for param in node.parameters():
                param.requires_grad_(True)
            break

    if args.train_embeddings:
        for module in (
            getattr(model.transformer, "lm_head", None),
            model.transformer.get_input_embeddings(),
        ):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(True)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    tunable = sum(p.numel() for _, p in trainable)
    print(f"decoder layers      : {total_layers} at {path}")
    print(f"trainable layers    : {total_layers - first_trainable} "
          f"(from index {first_trainable})")
    print(f"trainable parameters: {tunable / 1e6:.1f}M / {total / 1e6:.1f}M "
          f"({100 * tunable / max(total, 1):.2f}%)")
    if not trainable:
        raise SystemExit("no trainable parameters; check --train-last-layers")
    return trainable


def enable_nonreentrant_checkpointing(model: Any) -> None:
    """Re-enable gradient checkpointing with ``use_reentrant=False``.

    ``SketchDecoder.__init__`` turns on checkpointing with PyTorch's default
    reentrant implementation.  That variant drops parameter gradients for any
    checkpointed block whose *inputs* do not require grad - exactly the case
    here, because the layers below ``--train-last-layers`` are frozen.  The
    non-reentrant implementation tracks parameters explicitly and keeps the
    gradients.
    """
    transformer = model.transformer
    transformer.gradient_checkpointing_disable()
    transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    transformer.config.use_cache = False
    print("gradient checkpoint : enabled (use_reentrant=False)")


def check_gradient_flow(trainable: Sequence[Tuple[str, Any]]) -> Dict[str, float]:
    """Summarise which trainable tensors actually received a gradient.

    Guards against silent no-op training: if the decoder layers report zero
    gradients while the loss is finite, the update is doing nothing.
    """
    import torch

    with_grad = 0
    without_grad: List[str] = []
    total_sq = 0.0
    for name, param in trainable:
        if param.grad is None:
            without_grad.append(name)
            continue
        norm = float(torch.linalg.vector_norm(param.grad.detach().float()))
        if norm == 0.0:
            without_grad.append(name)
            continue
        with_grad += 1
        total_sq += norm * norm
    return {
        "tensors_with_grad": with_grad,
        "tensors_without_grad": len(without_grad),
        "grad_norm": math.sqrt(total_sq),
        "first_missing": without_grad[0] if without_grad else "",
    }


class ReferencePolicy:
    """Reference logprobs for the KL term.

    ``swap``
        Keep a bf16 backup of the trainable parameters only and swap it in for
        the reference forward pass.  Exact, because frozen parameters are
        shared, and costs the size of the trainable slice rather than a whole
        second model.
    ``full``
        A deep copy of the policy.  Simplest, needs ~2x the weight memory.
    ``none``
        No KL term; ``beta`` is ignored.
    """

    def __init__(self, mode: str, model: Any, trainable: Sequence[Tuple[str, Any]]) -> None:
        import torch

        self.torch = torch
        self.mode = mode
        self.model = model
        self._params = [param for _, param in trainable]
        self._backup: List[Any] = []
        self._full: Optional[Any] = None

        if mode == "swap":
            self._backup = [param.detach().clone() for param in self._params]
            bytes_held = sum(t.numel() * t.element_size() for t in self._backup)
            print(f"reference policy    : swap ({bytes_held / 1e9:.2f} GB backup)")
        elif mode == "full":
            import copy

            self._full = copy.deepcopy(model).eval()
            for param in self._full.parameters():
                param.requires_grad_(False)
            print("reference policy    : full frozen copy")
        else:
            print("reference policy    : none (KL disabled)")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @contextlib.contextmanager
    def activated(self):
        """Context in which ``self.model`` holds the reference weights."""
        if self.mode != "swap":
            yield self.model if self.mode != "full" else self._full
            return
        for param, backup in zip(self._params, self._backup):
            param.data, backup.data = backup.data, param.data
        try:
            yield self.model
        finally:
            for param, backup in zip(self._params, self._backup):
                param.data, backup.data = backup.data, param.data


class MasterWeights:
    """fp32 master copies of the trainable parameters.

    The policy is loaded in bf16, whose ~3 significant digits cannot represent
    an AdamW step of ~lr (5e-7) on weights of magnitude ~1e-2: stepping the
    bf16 tensors directly left 99.86% of the top-layer weights bit-identical
    after 500 steps.  The optimizer therefore steps these fp32 copies, and the
    bf16 weights used for forward/backward are refreshed from them.
    """

    def __init__(self, params: Sequence[Any]) -> None:
        self.params = list(params)
        self.masters = [
            param.detach().float().clone().requires_grad_(True) for param in self.params
        ]
        bytes_held = sum(t.numel() * t.element_size() for t in self.masters)
        print(f"fp32 master weights : {bytes_held / 1e9:.2f} GB")

    def load_grads(self) -> None:
        for param, master in zip(self.params, self.masters):
            master.grad = None if param.grad is None else param.grad.float()
            param.grad = None

    def write_back(self) -> None:
        for param, master in zip(self.params, self.masters):
            param.data.copy_(master.data)


def build_optimizer(args: argparse.Namespace, params: Sequence[Any]):
    import torch

    params = list(params)
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            params, lr=args.learning_rate, scale_parameter=False, relative_step=False
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9)
    return torch.optim.AdamW(
        params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )


# --------------------------------------------------------------------------- #
# Generation and log-probabilities
# --------------------------------------------------------------------------- #


def trim_completion(token_ids: Sequence[int]) -> Tuple[List[int], bool]:
    """Drop padding and everything past the first EOS.

    Returns ``(tokens, finished)`` where ``finished`` records whether the model
    actually emitted EOS rather than being cut off by ``max_new_tokens``.  A low
    finish rate means the sampling temperature is too high for this policy and
    every completion is being truncated mid-path.
    """
    trimmed: List[int] = []
    for raw in token_ids:
        token = int(raw)
        if token == EOS_TOKEN_ID:
            return trimmed, True
        if token == PAD_TOKEN_ID:
            continue
        trimmed.append(token)
    return trimmed, False


def generate_group(
    model: Any,
    prompt: Dict[str, Any],
    args: argparse.Namespace,
    device: Any,
) -> Tuple[List[List[int]], List[bool]]:
    """Sample ``num_generations`` completions for one prompt.

    Returns the trimmed token lists and, per completion, whether generation
    stopped on EOS instead of hitting ``--max-completion-length``.
    """
    import torch

    model_inputs = {
        "input_ids": prompt["input_ids"].to(device),
        "attention_mask": prompt["attention_mask"].to(device),
    }
    if prompt.get("pixel_values") is not None:
        model_inputs["pixel_values"] = prompt["pixel_values"].to(
            device, dtype=model.transformer.dtype
        )
    if prompt.get("image_grid_thw") is not None:
        model_inputs["image_grid_thw"] = prompt["image_grid_thw"].to(device)

    generation_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "bos_token_id": model.bos_token_id,
    }
    if args.top_k and args.top_k > 0:
        generation_kwargs["top_k"] = int(args.top_k)

    previous_cache_flag = getattr(model.transformer.config, "use_cache", True)
    model.transformer.config.use_cache = True
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.transformer.generate(
                **model_inputs,
                max_new_tokens=args.max_completion_length,
                num_return_sequences=args.num_generations,
                use_cache=True,
                **generation_kwargs,
            )
    finally:
        model.transformer.config.use_cache = previous_cache_flag
        if was_training:
            model.train()

    prompt_length = model_inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_length:]
    trimmed = [trim_completion(row.tolist()) for row in generated]
    return [tokens for tokens, _ in trimmed], [finished for _, finished in trimmed]


def completion_logprobs(
    model: Any,
    prompt: Dict[str, Any],
    completion_ids: Sequence[int],
    device: Any,
    with_grad: bool,
):
    """Per-token log-probabilities of ``completion_ids`` under ``model``.

    Runs one sequence at a time: ``SketchDecoder.forward`` derives RoPE indices
    from the attention mask, and left-padding a group would shift the image
    token positions.
    """
    import torch

    prompt_ids = prompt["input_ids"].to(device)
    completion = torch.tensor([list(completion_ids)], dtype=prompt_ids.dtype, device=device)
    input_ids = torch.cat([prompt_ids, completion], dim=1)
    attention_mask = torch.cat(
        [prompt["attention_mask"].to(device), torch.ones_like(completion)], dim=1
    )

    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if prompt.get("pixel_values") is not None:
        forward_kwargs["pixel_values"] = prompt["pixel_values"].to(
            device, dtype=model.transformer.dtype
        )
    if prompt.get("image_grid_thw") is not None:
        forward_kwargs["image_grid_thw"] = prompt["image_grid_thw"].to(device)

    context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with context:
        outputs = model(**forward_kwargs)
        # Logit at position t-1 predicts the token at position t.
        prompt_length = prompt_ids.shape[1]
        logits = outputs.logits[:, prompt_length - 1 : -1, :].float()
        targets = completion
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.squeeze(0)


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #


def section_counts(completion_ids: Sequence[int]) -> Tuple[int, int]:
    skeletons = sum(1 for token in completion_ids if token == SKELETON_START_TOKEN_ID)
    contours = sum(1 for token in completion_ids if token == REPLACEMENT_START_TOKEN_ID)
    return skeletons, contours


def summarise(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


class NullWriter:
    def append(self, record: Dict[str, Any]) -> None:
        pass


class Dist:
    """Data parallelism over ``torchrun`` processes, without DDP.

    Every rank holds a full copy of the policy and samples its own prompts.
    After the local backward passes the gradients are averaged with an
    all-reduce, so every rank applies the identical optimizer step and the
    weights never diverge.  DDP is not used because the number of backward
    passes per step differs between ranks (degenerate or OOM-skipped groups),
    which DDP's per-backward bucket hooks cannot handle.
    """

    def __init__(self, from_env: bool = True) -> None:
        env = os.environ if from_env else {}
        self.world_size = int(env.get("WORLD_SIZE", "1"))
        self.rank = int(env.get("RANK", "0"))
        self.local_rank = int(env.get("LOCAL_RANK", "0"))

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def init(self, timeout_minutes: int) -> None:
        if not self.enabled:
            return
        import datetime

        import torch
        import torch.distributed as td

        torch.cuda.set_device(self.local_rank)
        td.init_process_group("nccl", timeout=datetime.timedelta(minutes=timeout_minutes))

    def close(self) -> None:
        if self.enabled:
            import torch.distributed as td

            if td.is_initialized():
                td.destroy_process_group()

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as td

            td.barrier()

    def broadcast_object(self, obj: Any) -> Any:
        if not self.enabled:
            return obj
        import torch.distributed as td

        box = [obj]
        td.broadcast_object_list(box, src=0)
        return box[0]

    def gather_objects(self, obj: Any) -> List[Any]:
        if not self.enabled:
            return [obj]
        import torch.distributed as td

        out: List[Any] = [None] * self.world_size
        td.all_gather_object(out, obj)
        return out

    def any_true(self, flag: bool) -> bool:
        if not self.enabled:
            return flag
        import torch
        import torch.distributed as td

        value = torch.tensor([1.0 if flag else 0.0], device="cuda")
        td.all_reduce(value, op=td.ReduceOp.MAX)
        return bool(value.item())

    def average_grads(self, tensors: Sequence[Any]) -> None:
        """Average ``.grad`` over ranks; a rank with no local grad contributes zeros."""
        if not self.enabled:
            return
        import torch
        import torch.distributed as td

        for tensor in tensors:
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)
            td.all_reduce(tensor.grad)
            tensor.grad.div_(self.world_size)


def merge_rank_stats(
    dist: Dist, window: Dict[str, List[float]], stats: RewardStats
) -> Tuple[Dict[str, List[float]], RewardStats]:
    """Pool every rank's logging window and reward sums into one view."""
    if not dist.enabled:
        return window, stats
    parts = dist.gather_objects((window, stats.count, stats.sums))
    merged_window = {key: [v for part, _, _ in parts for v in part[key]] for key in window}
    merged = RewardStats()
    for _, count, sums in parts:
        merged.count += count
        for key, value in sums.items():
            merged.sums[key] = merged.sums.get(key, 0.0) + value
    return merged_window, merged


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def run_training(args: argparse.Namespace, reward_config: RewardConfig) -> int:
    import torch
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    from train import load_model
    from utils.config import TokenizationConfig

    dist = Dist()
    dist.init(args.dist_timeout_minutes)
    if dist.enabled and args.probe_sampling:
        raise SystemExit("--probe-sampling is a single-process diagnostic; run it without torchrun")
    if not dist.is_main:
        sys.stdout = open(os.devnull, "w")

    # Different sampling noise per rank; the training set order comes from
    # load_samples(shuffle_seed=...) and stays identical on every rank.
    random.seed(args.seed + dist.rank)
    np.random.seed(args.seed + dist.rank)
    torch.manual_seed(args.seed + dist.rank)

    project_name = dist.broadcast_object(
        args.project_name or f"grpo_{args.model_size.lower()}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(args.output_dir, project_name)
    if dist.is_main:
        os.makedirs(run_dir, exist_ok=True)
        metrics_log: Any = JsonlWriter(os.path.join(run_dir, "grpo_metrics.jsonl"))
        completions_log: Any = JsonlWriter(os.path.join(run_dir, "completions.jsonl"))
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"args": vars(args), "world_size": dist.world_size,
                 "reward": reward_config.to_dict()},
                handle, indent=2,
            )
    else:
        metrics_log = NullWriter()
        completions_log = NullWriter()

    base_model = resolve_base_model(args.base_model)
    token_config = TokenizationConfig.from_yaml(
        reward_config.tokenization_config_path, model_size=args.model_size
    )
    vocab_size = token_config.replacement_end_token + 1

    print("=" * 74)
    print("GRPO fine-tuning - OmniSVG stage-2 code complement")
    print("=" * 74)
    print(f"run directory       : {run_dir}")
    print(f"base model          : {base_model}")
    print(f"SFT checkpoint      : {args.sft_checkpoint}")
    print(f"vocab size          : {vocab_size}")
    print(f"num_generations     : {args.num_generations}")
    print(f"learning rate       : {args.learning_rate:g}")
    print(f"beta (KL)           : {args.beta:g}")
    print(f"max completion len  : {args.max_completion_length}")
    print(f"data parallel       : {dist.world_size} rank(s), "
          f"{dist.world_size * args.prompts_per_step} prompt group(s) per optimizer step")

    processor = AutoProcessor.from_pretrained(base_model, padding_side="left", use_fast=True)
    processor.tokenizer.padding_side = "left"

    model = load_model(
        model_size=args.model_size,
        pix_len=args.max_completion_length,
        text_len=args.condition_max_length,
        use_flash_attn=not args.no_flash_attn,
        checkpoint_path=args.sft_checkpoint,
        device_map=None,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        vocab_size=vocab_size,
    )
    device = (
        torch.device("cuda", dist.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)
    model.train()

    trainable = configure_trainable(model, args)
    if not args.no_gradient_checkpointing:
        enable_nonreentrant_checkpointing(model)
    reference = ReferencePolicy(args.ref_mode, model, trainable)
    master: Optional[MasterWeights] = None
    if not args.no_fp32_master_weights and any(
        param.dtype != torch.float32 for _, param in trainable
    ):
        master = MasterWeights([param for _, param in trainable])
    optimizer = build_optimizer(
        args, master.masters if master is not None else [param for _, param in trainable]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    prompt_builder = PromptBuilder(
        processor,
        token_config,
        condition_max_length=args.condition_max_length,
    )

    if args.probe_sampling:
        return run_sampling_probe(model, prompt_builder, args, reward_config, device)

    samples = load_samples(
        args.data_dir,
        split=args.split,
        limit=args.num_samples,
        shuffle_seed=args.seed,
        require_image=True,
    )
    if not samples:
        raise SystemExit(f"no usable samples in {args.data_dir}")
    print(f"training prompts    : {len(samples)}")

    wandb_run = None
    if args.use_wandb and dist.is_main:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=project_name,
                config={"args": vars(args), "reward": reward_config.to_dict()},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("wandb disabled: %s: %s", type(exc).__name__, exc)

    print("=" * 74)
    print("starting GRPO updates")
    print("=" * 74)

    stats = RewardStats()
    window: Dict[str, List[float]] = {
        "reward": [],
        "reward_std": [],
        "loss": [],
        "kl": [],
        "grad_norm": [],
        "completion_length": [],
        "eos_finished": [],
        "degenerate_groups": [],
        "empty_completions": [],
    }
    started = time.time()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_losses: List[float] = []
        step_kls: List[float] = []
        step_rewards: List[float] = []
        step_group_stds: List[float] = []
        step_lengths: List[int] = []
        step_finished: List[float] = []
        degenerate = 0
        empty = 0
        step_infos: List[Dict[str, Any]] = []
        step_dumps: List[Tuple[str, List[int], Dict[str, Any]]] = []
        total_groups = 0

        for slot in range(args.prompts_per_step):
            index = ((step - 1) * dist.world_size + dist.rank) * args.prompts_per_step + slot
            sample = samples[index % len(samples)]

            try:
                prompt = prompt_builder.build(sample)
            except Exception as exc:  # noqa: BLE001
                logger.warning("prompt build failed for %s: %s: %s",
                               sample["uid"], type(exc).__name__, exc)
                continue

            try:
                completions, finished = generate_group(model, prompt, args, device)
            except torch.cuda.OutOfMemoryError as exc:
                logger.warning(
                    "OOM while generating for %s (prompt_len=%d): %s",
                    sample["uid"], prompt["input_ids"].shape[1], exc,
                )
                torch.cuda.empty_cache()
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("generation failed for %s: %s: %s",
                               sample["uid"], type(exc).__name__, exc)
                continue

            rewards, infos = reward_batch(completions, [sample] * len(completions), reward_config)
            step_infos.extend(infos)
            step_dumps.extend(
                (sample["uid"], completion, info) for completion, info in zip(completions, infos)
            )
            step_rewards.extend(rewards)
            step_lengths.extend(len(c) for c in completions)
            step_finished.extend(float(flag) for flag in finished)
            empty += sum(1 for c in completions if not c)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            group_std = float(reward_tensor.std(unbiased=False))
            step_group_stds.append(group_std)
            if group_std < 1e-6:
                # No spread inside the group: advantages are all zero, so the
                # update would be a no-op.  Skip it instead of burning a forward.
                degenerate += 1
                continue

            advantages = reward_tensor - reward_tensor.mean()
            if not args.no_scale_rewards:
                advantages = advantages / (reward_tensor.std(unbiased=False) + 1e-4)

            # Reference logprobs first: with --ref-mode swap the weights are
            # temporarily replaced, which must not happen while an autograd
            # graph over the policy weights is alive.
            reference_logprobs: List[Optional[Any]] = [None] * len(completions)
            if reference.enabled and args.beta > 0:
                with reference.activated() as ref_model:
                    for index, completion in enumerate(completions):
                        if not completion:
                            continue
                        reference_logprobs[index] = completion_logprobs(
                            ref_model, prompt, completion, device, with_grad=False
                        ).detach()

            total_groups += 1
            for index, completion in enumerate(completions):
                if not completion:
                    continue
                try:
                    logprobs = completion_logprobs(
                        model, prompt, completion, device, with_grad=True
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM on policy forward (len=%d); skipping completion",
                                   len(completion))
                    torch.cuda.empty_cache()
                    continue

                coefficient = torch.exp(logprobs - logprobs.detach())
                per_token_loss = -coefficient * advantages[index]

                kl_value = 0.0
                if reference_logprobs[index] is not None and args.beta > 0:
                    delta = reference_logprobs[index] - logprobs
                    kl = torch.exp(delta) - delta - 1.0
                    per_token_loss = per_token_loss + args.beta * kl
                    kl_value = float(kl.mean().detach())

                loss = per_token_loss.mean() / (args.prompts_per_step * args.num_generations)
                loss.backward()

                step_losses.append(float(loss.detach()) * args.prompts_per_step * args.num_generations)
                step_kls.append(kl_value)

        grad_norm = 0.0
        if dist.any_true(bool(step_losses)):
            if step == 1 and step_losses:
                flow = check_gradient_flow(trainable)
                print(
                    f"gradient flow check : {flow['tensors_with_grad']} tensors with "
                    f"gradients, {flow['tensors_without_grad']} without "
                    f"(norm={flow['grad_norm']:.4g})"
                )
                if flow["tensors_with_grad"] == 0:
                    raise SystemExit(
                        "no trainable tensor received a gradient; the update would be "
                        "a no-op (first missing: "
                        f"{flow['first_missing']})"
                    )
                if flow["tensors_without_grad"] > flow["tensors_with_grad"]:
                    logger.warning(
                        "most trainable tensors have no gradient (first: %s); check "
                        "gradient checkpointing and --train-last-layers",
                        flow["first_missing"],
                    )
            if master is not None:
                master.load_grads()
                stepped = master.masters
            else:
                stepped = [param for _, param in trainable]
            dist.average_grads(stepped)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(stepped, args.max_grad_norm))
            optimizer.step()
            if master is not None:
                master.write_back()
        scheduler.step()

        stats.update(step_infos)
        window["reward"].extend(step_rewards)
        window["reward_std"].extend(step_group_stds)
        window["loss"].extend(step_losses)
        window["kl"].extend(step_kls)
        window["grad_norm"].append(grad_norm)
        window["completion_length"].extend(step_lengths)
        window["eos_finished"].extend(step_finished)
        window["degenerate_groups"].append(degenerate)
        window["empty_completions"].append(empty)

        if args.dump_completions:
            for uid, completion, info in step_dumps[: args.dump_completions]:
                skeleton_sections, contour_sections = section_counts(completion)
                completions_log.append(
                    {
                        "step": step,
                        "uid": uid,
                        "reward": info["reward"],
                        "valid": info["valid"],
                        "error": info["error"],
                        "length": len(completion),
                        "skeleton_sections": skeleton_sections,
                        "contour_sections": contour_sections,
                        "token_ids": completion[:512],
                    }
                )

        if step % args.log_every == 0 or step == 1:
            pooled, pooled_stats = merge_rank_stats(dist, window, stats)
            reward_summary = summarise(pooled["reward"])
            means = pooled_stats.means()
            record = {
                "step": step,
                "elapsed_s": round(time.time() - started, 1),
                "lr": scheduler.get_last_lr()[0],
                "reward_mean": reward_summary["mean"],
                "reward_std": reward_summary["std"],
                "reward_min": reward_summary["min"],
                "reward_max": reward_summary["max"],
                "group_reward_std": summarise(pooled["reward_std"])["mean"],
                "loss": summarise(pooled["loss"])["mean"],
                "kl": summarise(pooled["kl"])["mean"],
                "grad_norm": summarise(pooled["grad_norm"])["mean"],
                "completion_length": summarise(pooled["completion_length"])["mean"],
                "completion_length_max": summarise(pooled["completion_length"])["max"],
                "eos_finish_rate": summarise(pooled["eos_finished"])["mean"],
                "degenerate_group_rate": summarise(pooled["degenerate_groups"])["mean"],
                "empty_completion_rate": summarise(pooled["empty_completions"])["mean"],
                **means,
            }
            metrics_log.append(record)
            if wandb_run is not None:
                wandb_run.log({f"grpo/{k}": v for k, v in record.items() if k != "step"},
                              step=step)
            print(
                f"step {step:5d} | reward {record['reward_mean']:+.4f} "
                f"(±{record['reward_std']:.3f}) | loss {record['loss']:+.4f} "
                f"| kl {record['kl']:.4f} | gnorm {record['grad_norm']:.2f} "
                f"| valid {100 * means.get('format_valid_rate', 0):5.1f}% "
                f"| skel_miss {100 * means.get('skeleton_missing_rate', 0):5.1f}% "
                f"| cons {means.get('R_consistency_iou', 0):.3f} "
                f"| merge {means.get('R_merge_iou', 0):.3f} "
                f"| len {record['completion_length']:.0f} "
                f"| eos {100 * record['eos_finish_rate']:3.0f}%"
            )
            stats.reset()
            for values in window.values():
                values.clear()

        if args.save_every and step % args.save_every == 0:
            if dist.is_main:
                save_checkpoint(model, run_dir, step)
            dist.barrier()

        if args.eval_every and step % args.eval_every == 0:
            evaluate(model, prompt_builder, args, reward_config, device, step, metrics_log, dist)

    if args.save_every:
        if dist.is_main:
            save_checkpoint(model, run_dir, args.max_steps, final=True)
        dist.barrier()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nGRPO finished in {(time.time() - started) / 60:.1f} min; artefacts in {run_dir}")
    dist.close()
    return 0


def save_checkpoint(model: Any, run_dir: str, step: int, final: bool = False) -> None:
    """Write ``step_N/pytorch_model.bin`` so inference_run.sh can load it."""
    import torch

    name = "final" if final else f"step_{step}"
    path = os.path.join(run_dir, name)
    os.makedirs(path, exist_ok=True)
    state_dict = {key: value.to("cpu") for key, value in model.state_dict().items()}
    torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
    with open(os.path.join(path, "training_info.json"), "w", encoding="utf-8") as handle:
        json.dump({"step": step, "trainer": "grpo"}, handle, indent=2)
    print(f"  checkpoint saved: {path}")
    del state_dict
    torch.cuda.empty_cache()


def evaluate(
    model: Any,
    prompt_builder: PromptBuilder,
    args: argparse.Namespace,
    reward_config: RewardConfig,
    device: Any,
    step: int,
    metrics_log: Any,
    dist: Optional[Dist] = None,
) -> None:
    """Score held-out prompts with a single low-temperature sample each.

    Under torchrun every rank scores a disjoint slice and rank 0 writes the
    pooled record, so the numbers match a single-process run on the same set.
    """
    dist = dist if dist is not None else Dist(from_env=False)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.num_generations = 1
    eval_args.temperature = getattr(args, "eval_temperature", 0.01)
    eval_args.top_p = getattr(args, "eval_top_p", 0.3)
    eval_args.top_k = getattr(args, "eval_top_k", 5)

    samples = load_samples(
        args.data_dir,
        split=args.eval_split,
        limit=args.eval_samples,
        shuffle_seed=args.seed,
        require_image=True,
    )
    stats = RewardStats()
    rewards: List[float] = []
    for sample in samples[dist.rank :: dist.world_size]:
        try:
            prompt = prompt_builder.build(sample)
            completions, _finished = generate_group(model, prompt, eval_args, device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval failed for %s: %s", sample["uid"], type(exc).__name__)
            continue
        scores, infos = reward_batch(completions, [sample] * len(completions), reward_config)
        rewards.extend(scores)
        stats.update(infos)

    if dist.enabled:
        pooled, stats = merge_rank_stats(dist, {"reward": rewards}, stats)
        rewards = pooled["reward"]
    means = stats.means(prefix="eval_")
    record = {"step": step, "eval_reward_mean": summarise(rewards)["mean"], **means}
    metrics_log.append(record)
    print(
        f"  [eval @ {step}] reward {record['eval_reward_mean']:+.4f} "
        f"| valid {100 * means.get('eval_format_valid_rate', 0):5.1f}% "
        f"| skel_miss {100 * means.get('eval_skeleton_missing_rate', 0):5.1f}% "
        f"| cons {means.get('eval_R_consistency_iou', 0):.3f} "
        f"| merge {means.get('eval_R_merge_iou', 0):.3f}"
    )


def run_target_length_measurement(args: argparse.Namespace, config: RewardConfig) -> int:
    """Report how many tokens the ground-truth skeleton-CoT targets need.

    This is what ``--max-completion-length`` has to cover.  It reproduces
    ``utils/dataset.py::_build_skeleton_cot_sequence``: every skeleton and
    replacement path is wrapped as a minimal SVG, tokenized, and bracketed by
    its marker pair.  Anything above the cap is a target whose EOS the SFT run
    never saw, which is why such completions never terminate.
    """
    import os as _os
    import tempfile
    from xml.sax.saxutils import quoteattr

    from deepsvg.svglib.svg import SVG as DeepSVG
    from utils.config import TokenizationConfig
    from utils.dataset import SVGTokenizer as TrainingSVGEncoder

    token_config = TokenizationConfig.from_yaml(
        config.tokenization_config_path, model_size=args.model_size
    )
    encoder = TrainingSVGEncoder(token_config)

    def tokenize(path_d: str) -> int:
        document = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">\n'
            f"  <path fill={quoteattr('#000')} d={quoteattr(path_d)}/>\n</svg>"
        )
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False)
        try:
            handle.write(document)
            handle.close()
            svg = DeepSVG.load_svg(handle.name)
            tensors, colors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
            return len(encoder.tokenize_svg_tensors(tensors, colors))
        except Exception:  # noqa: BLE001
            return 0
        finally:
            _os.unlink(handle.name)

    samples = load_samples(
        args.data_dir,
        split=args.eval_split,
        limit=args.measure_target_length,
        shuffle_seed=args.seed,
        require_image=False,
    )
    if not samples:
        print(f"ERROR: no samples found in {args.data_dir}")
        return 1

    lengths: List[int] = []
    for sample in samples:
        total = 1  # trailing EOS
        for path_d in sample.get("gt_skeleton_path_data") or []:
            if path_d.strip():
                total += tokenize(path_d) + 2  # SKEL_S / SKEL_E
        for path_d in sample.get("gt_contour_path_data") or []:
            total += tokenize(path_d) + 2  # REPL_S / REPL_E
        lengths.append(total)

    lengths.sort()

    def quantile(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, int(fraction * len(lengths)))]

    print("=" * 74)
    print(f"Ground-truth skeleton-CoT target length over {len(lengths)} "
          f"{args.eval_split} samples")
    print("=" * 74)
    print(f"mean {statistics.mean(lengths):.0f}  median {quantile(0.50)}  "
          f"p75 {quantile(0.75)}  p90 {quantile(0.90)}  p95 {quantile(0.95)}  "
          f"p99 {quantile(0.99)}  max {lengths[-1]}")
    print()
    for cap in (1024, 1536, 2048, 3072, 4096):
        covered = sum(1 for length in lengths if length <= cap) / len(lengths)
        marker = "  <- current --max-completion-length" if cap == args.max_completion_length else ""
        print(f"fits in {cap:5d} tokens: {100 * covered:5.1f}%{marker}")
    print("\nSet --max-completion-length at or above p99 so completions can emit EOS.")
    return 0


def run_sampling_probe(
    model: Any,
    prompt_builder: PromptBuilder,
    args: argparse.Namespace,
    reward_config: RewardConfig,
    device: Any,
) -> int:
    """Sweep the sampling temperature and report what each setting produces.

    GRPO needs within-group reward spread, but too high a temperature stops the
    policy from ever emitting EOS, so every completion gets truncated at
    ``--max-completion-length``.  This picks the temperature that buys spread
    while keeping the finish rate high, and shows whether the length budget is
    the binding constraint.
    """
    temperatures = [float(t) for t in args.probe_temperatures.split(",") if t.strip()]
    samples = load_samples(
        args.data_dir,
        split=args.eval_split,
        limit=args.probe_sampling,
        shuffle_seed=args.seed,
        require_image=True,
    )
    if not samples:
        print(f"ERROR: no samples found in {args.data_dir}")
        return 1

    prompts = []
    for sample in samples:
        try:
            prompts.append((sample, prompt_builder.build(sample)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prompt build failed for %s: %s", sample["uid"], type(exc).__name__)

    print("=" * 96)
    print(f"Sampling probe: {len(prompts)} prompts x {args.num_generations} generations "
          f"x {len(temperatures)} temperatures")
    print(f"max_completion_length={args.max_completion_length}  top_p={args.top_p}  "
          f"top_k={args.top_k}")
    print("=" * 96)
    header = (f"{'temp':>5s} {'reward':>16s} {'grp_std':>8s} {'len':>7s} {'len_max':>7s} "
              f"{'eos%':>5s} {'valid%':>7s} {'miss%':>6s} {'cons':>6s} {'merge':>6s}")
    print(header)
    print("-" * len(header))

    best: List[Tuple[float, float]] = []
    for temperature in temperatures:
        probe_args = argparse.Namespace(**vars(args))
        probe_args.temperature = temperature
        rewards: List[float] = []
        group_stds: List[float] = []
        lengths: List[int] = []
        finishes: List[float] = []
        stats = RewardStats()

        for sample, prompt in prompts:
            try:
                completions, finished = generate_group(model, prompt, probe_args, device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("probe generation failed for %s: %s",
                               sample["uid"], type(exc).__name__)
                continue
            scores, infos = reward_batch(
                completions, [sample] * len(completions), reward_config
            )
            rewards.extend(scores)
            group_stds.append(summarise(scores)["std"])
            lengths.extend(len(c) for c in completions)
            finishes.extend(float(flag) for flag in finished)
            stats.update(infos)

        means = stats.means()
        reward_summary = summarise(rewards)
        length_summary = summarise(lengths)
        group_std = summarise(group_stds)["mean"]
        best.append((group_std, temperature))
        print(
            f"{temperature:5.2f} "
            f"{reward_summary['mean']:+.3f} ± {reward_summary['std']:.3f} "
            f"{group_std:8.3f} {length_summary['mean']:7.0f} {length_summary['max']:7.0f} "
            f"{100 * summarise(finishes)['mean']:5.0f} "
            f"{100 * means.get('format_valid_rate', 0):7.1f} "
            f"{100 * means.get('skeleton_missing_rate', 0):6.1f} "
            f"{means.get('R_consistency_iou', 0):6.3f} {means.get('R_merge_iou', 0):6.3f}"
        )

    if best:
        widest = max(best)
        print(f"\nwidest within-group reward spread at temperature {widest[1]:.2f} "
              f"(std={widest[0]:.3f})")
    print("Pick the highest temperature whose eos% stays high: truncated completions "
          "waste tokens and bias the contour reward.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rank_tag = f"[rank {os.environ['RANK']}] " if int(os.environ.get("WORLD_SIZE", "1")) > 1 else ""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=f"%(asctime)s %(levelname)s {rank_tag}%(name)s: %(message)s",
    )
    reward_config = build_reward_config(args)

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and (args.reward_smoke or args.measure_target_length):
        raise SystemExit("--reward-smoke / --measure-target-length are single-process "
                         "diagnostics; run them without torchrun")
    if args.reward_smoke:
        return run_reward_smoke(args, reward_config)
    if args.measure_target_length:
        return run_target_length_measurement(args, reward_config)
    return run_training(args, reward_config)


if __name__ == "__main__":
    raise SystemExit(main())
