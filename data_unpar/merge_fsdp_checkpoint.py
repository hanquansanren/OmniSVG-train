"""
Merge FSDP sharded checkpoint (.distcp files) into a single model file.

Usage:
    python merge_fsdp_checkpoint.py <checkpoint_dir> [--format safetensors|bin]

Examples:
    # Merge step_5000 checkpoint to model.safetensors (default)
    python merge_fsdp_checkpoint.py output/omnisvg_4b_xxx/step_5000

    # Merge to pytorch_model.bin format
    python merge_fsdp_checkpoint.py output/omnisvg_4b_xxx/step_5000 --format bin

    # Merge best_model checkpoint
    python merge_fsdp_checkpoint.py output/omnisvg_4b_xxx/best_model
"""

import argparse
import os
import sys
import glob


def find_fsdp_dir(checkpoint_dir: str) -> str:
    """Find the FSDP model shard directory inside a checkpoint."""
    candidates = glob.glob(os.path.join(checkpoint_dir, "pytorch_model_fsdp_*"))
    if candidates:
        return sorted(candidates)[0]

    distcp_files = glob.glob(os.path.join(checkpoint_dir, "*.distcp"))
    if distcp_files:
        return checkpoint_dir

    for sub in sorted(os.listdir(checkpoint_dir)):
        sub_path = os.path.join(checkpoint_dir, sub)
        if os.path.isdir(sub_path):
            if glob.glob(os.path.join(sub_path, "*.distcp")):
                return sub_path

    return ""


def merge(checkpoint_dir: str, output_format: str = "safetensors"):
    fsdp_dir = find_fsdp_dir(checkpoint_dir)
    if not fsdp_dir:
        print(f"Error: No FSDP shard directory found in {checkpoint_dir}")
        print("Expected to find a directory containing .distcp files")
        print(f"Contents of {checkpoint_dir}:")
        for f in sorted(os.listdir(checkpoint_dir)):
            print(f"  {f}")
        sys.exit(1)

    print(f"Found FSDP shards in: {fsdp_dir}")
    distcp_files = sorted(glob.glob(os.path.join(fsdp_dir, "*.distcp")))
    print(f"  Shard files: {len(distcp_files)}")
    for f in distcp_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"    {os.path.basename(f)} ({size_mb:.1f} MB)")

    if output_format == "safetensors":
        output_path = os.path.join(checkpoint_dir, "model.safetensors")
    else:
        output_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(output_path):
        print(f"\nOutput file already exists: {output_path}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            return

    print(f"\nMerging shards -> {output_path} ...")

    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    tmp_path = output_path + ".tmp"
    dcp_to_torch_save(fsdp_dir, tmp_path)

    state_dict = torch.load(tmp_path, map_location="cpu", weights_only=False)

    # Accelerate's save_state wraps model weights under a 'model' key
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        print(f"  Unwrapping nested 'model' key...")
        state_dict = state_dict['model']

    # Flatten any remaining nested dicts (e.g. {'state_dict': {...}})
    while len(state_dict) == 1:
        key = next(iter(state_dict))
        if isinstance(state_dict[key], dict):
            print(f"  Unwrapping nested '{key}' key...")
            state_dict = state_dict[key]
        else:
            break

    if output_format == "safetensors":
        from safetensors.torch import save_file
        save_file(state_dict, output_path)
    else:
        torch.save(state_dict, output_path)

    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nDone! Merged model saved to: {output_path} ({size_mb:.1f} MB)")
    print(f"Total keys: {len(state_dict)}")

    sample_keys = sorted(state_dict.keys())[:5]
    print(f"Sample keys: {sample_keys}")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoint into a single file")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory (e.g. output/xxx/step_5000)")
    parser.add_argument("--format", choices=["safetensors", "bin"], default="safetensors",
                        help="Output format (default: safetensors)")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        print(f"Error: {args.checkpoint_dir} is not a directory")
        sys.exit(1)

    merge(args.checkpoint_dir, args.format)


if __name__ == "__main__":
    main()
