#!/usr/bin/env python3
"""Merge FSDP .distcp shards into model.safetensors (same logic as train.py)."""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def merge_fsdp_checkpoint(ckpt_path: str | Path, tmp_dir: str | Path | None = None) -> Path:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_path}")

    output_path = ckpt_path / "model.safetensors"
    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        print(f"Already merged: {output_path} ({output_path.stat().st_size / 1024 / 1024:.0f} MB)")
        return output_path

    fsdp_dirs = sorted(glob.glob(str(ckpt_path / "pytorch_model_fsdp_*")))
    if not fsdp_dirs:
        raise FileNotFoundError(f"No pytorch_model_fsdp_* dir under {ckpt_path}")

    fsdp_dir = fsdp_dirs[0]
    distcp_files = glob.glob(os.path.join(fsdp_dir, "*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files in {fsdp_dir}")

    tmp_dir = Path(tmp_dir or os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"merge_fsdp_{ckpt_path.name}_{os.getpid()}.pt"
    broken = ckpt_path / "model.safetensors.tmp"
    if broken.exists():
        print(f"Removing broken tmp: {broken}")
        broken.unlink()

    print(f"Merging {len(distcp_files)} FSDP shards from {fsdp_dir}")
    print(f"Temp file: {tmp_path}")
    print(f"Output:    {output_path}")

    t0 = time.time()
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    dcp_to_torch_save(fsdp_dir, str(tmp_path))
    state_dict = torch.load(str(tmp_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    save_file(state_dict, str(output_path))
    tmp_path.unlink(missing_ok=True)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done: {output_path} ({size_mb:.0f} MB, {time.time() - t0:.1f}s)")
    return output_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CKPT_PATH", "output_stage2/omnisvg_stage2_4b_20261005_123246/step_7500"
    )
    merge_fsdp_checkpoint(path)
