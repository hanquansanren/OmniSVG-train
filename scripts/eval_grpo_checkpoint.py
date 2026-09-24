#!/usr/bin/env python3
"""Run the same held-out eval as train_grpo.evaluate() on any checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor

from train import load_model
from train_grpo import JsonlWriter, PromptBuilder, build_reward_config, evaluate, resolve_base_model
from utils.config import TokenizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO-style eval on an SFT or GRPO checkpoint")
    parser.add_argument("--sft-checkpoint", required=True, help="Checkpoint directory or weight file")
    parser.add_argument("--data-dir", default="/data/phd23_weiguang_zhang/works/svg/my_lis2_2")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--model-size", default="4B")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--condition-max-length", type=int, default=1024)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--consistency-mode", default="iou")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL path for the eval record")
    parser.add_argument("--label", default="baseline", help="Label printed in the summary")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.eval_temperature = args.temperature
    args.eval_top_p = args.top_p
    args.eval_top_k = args.top_k
    reward_config = build_reward_config(args)

    base_model = resolve_base_model(None)
    token_config = TokenizationConfig.from_yaml(
        reward_config.tokenization_config_path, model_size=args.model_size
    )
    vocab_size = token_config.replacement_end_token + 1

    print("=" * 74)
    print(f"GRPO-style eval: {args.label}")
    print("=" * 74)
    print(f"checkpoint          : {args.sft_checkpoint}")
    print(f"eval split/samples  : {args.eval_split} / {args.eval_samples}")
    print(f"seed                : {args.seed}")
    print(f"default eval sampling       : T=0.01 top_p=0.3 top_k=5 (same as train_grpo.evaluate)")
    print(f"eval sampling       : T={args.temperature} top_p={args.top_p} top_k={args.top_k}")

    processor = AutoProcessor.from_pretrained(base_model, padding_side="left", use_fast=True)
    processor.tokenizer.padding_side = "left"

    model = load_model(
        model_size=args.model_size,
        pix_len=args.max_completion_length,
        text_len=args.condition_max_length,
        use_flash_attn=not args.no_flash_attn,
        checkpoint_path=args.sft_checkpoint,
        device_map=None,
        use_gradient_checkpointing=False,
        vocab_size=vocab_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_builder = PromptBuilder(
        processor,
        token_config,
        condition_max_length=args.condition_max_length,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        metrics_log = JsonlWriter(args.output)
    else:
        class _Sink:
            def append(self, record):  # noqa: ANN001
                print(json.dumps(record, ensure_ascii=False, indent=2))

        metrics_log = _Sink()

    evaluate(model, prompt_builder, args, reward_config, device, step=0, metrics_log=metrics_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
