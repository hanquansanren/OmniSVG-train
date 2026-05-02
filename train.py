"""
OmniSVG Training Script

Supports:
- 4B model (Qwen2.5-VL-3B based)
- 8B model (Qwen2.5-VL-7B based)
- Flash Attention 2 acceleration
- Training from local files or HuggingFace datasets
- Multi-task learning (Text-to-SVG and Image-to-SVG)
- Distributed training with Accelerate

Usage:
    # Train 4B model
    accelerate launch train.py --model_size 4B --data_dir ./data
    
    # Train 8B model with Flash Attention
    accelerate launch train.py --model_size 8B --use_flash_attn --data_dir ./data
    
    # Train with HuggingFace data (auto-download)
    accelerate launch train.py --model_size 4B --use_hf_data --datasets illustration icon
    
    # Resume training from OmniSVG checkpoint
    accelerate launch train.py --model_size 4B --resume_from_checkpoint auto
"""

import os
import sys

# ⭐⭐⭐ 关键：在导入torch之前设置NCCL超时环境变量 ⭐⭐⭐
# PyTorch 2.5.0 在导入时就初始化NCCL，必须提前设置这些变量

# 注意：这些必须在 import torch 之前设置！
# if "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ:
#     os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"  # 60分钟
# if "TORCH_NCCL_BLOCKING_WAIT" not in os.environ:
#     os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
#     os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 打印NCCL配置（仅主进程）
if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"🔧 NCCL Configuration (set before torch import):")
    print(f"   TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC = {os.environ.get('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', 'not set')}")
    print(f"   TORCH_NCCL_BLOCKING_WAIT = {os.environ.get('TORCH_NCCL_BLOCKING_WAIT', 'not set')}")
    print()

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from safetensors.torch import load_file

# Weights & Biases for cloud visualization
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Local imports
from utils import (
    OmniSVGConfig,
    TokenizationConfig,
    TrainConfig,
    OmniSVGDataset,
    create_dataloader,
    download_omnisvg_data,
    list_available_datasets,
)
from utils.config import MODEL_DEFAULTS
from decoder import SketchDecoder

# For Qwen2.5-VL
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Image processing may be limited.")
    process_vision_info = None

# ============================================================================
# Logging Configuration - 禁用FSDP详细日志
# ============================================================================

import logging
import warnings

# 禁用FSDP和分布式训练的详细日志
logging.getLogger("torch.distributed.fsdp").setLevel(logging.WARNING)
logging.getLogger("torch.distributed").setLevel(logging.WARNING)
logging.getLogger("torch.nn.parallel").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)

# 禁用transformers的详细日志
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ============================================================================
# Model Configuration
# ============================================================================
# MODEL_DEFAULTS is now imported from utils.config
# It reads from tokenization.yaml to support custom model paths


# ============================================================================
# Model Loading Utilities
# ============================================================================

def is_hf_repo_id(path: str) -> bool:
    """
    Check if a string looks like a HuggingFace repo ID.
    
    Args:
        path: String to check
    
    Returns:
        True if it looks like a HuggingFace repo ID
    """
    # HuggingFace repo IDs have format "owner/repo_name"
    # Local paths typically have more slashes or start with . / ~
    if path is None:
        return False
    
    # If path exists locally, it's not a repo ID
    if os.path.exists(path):
        return False
    
    # Check for HuggingFace repo ID pattern
    parts = path.split("/")
    if len(parts) == 2 and all(part and not part.startswith(('.', '~')) for part in parts):
        return True
    
    return False


def download_omnisvg_checkpoint(
    repo_id_or_model_size: str, 
    cache_dir: str = "./checkpoints"
) -> str:
    """
    Download OmniSVG checkpoint from HuggingFace Hub.
    
    Args:
        repo_id_or_model_size: HuggingFace repo ID (e.g., "OmniSVG/OmniSVG1.1_4B") 
                               or model size ("4B" or "8B")
        cache_dir: Directory to cache downloaded checkpoints
    
    Returns:
        Path to downloaded checkpoint
    """
    # If it's a model size, get the default checkpoint
    if repo_id_or_model_size in MODEL_DEFAULTS:
        repo_id = MODEL_DEFAULTS[repo_id_or_model_size]["checkpoint"]
    else:
        repo_id = repo_id_or_model_size
    
    print(f"Downloading OmniSVG checkpoint from {repo_id}...")
    
    # Create local directory name from repo ID
    local_name = repo_id.replace("/", "_")
    local_dir = os.path.join(cache_dir, local_name)
    
    # Download the repository
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        resume_download=True,
    )
    
    print(f"Checkpoint downloaded to: {local_path}")
    return local_path


def find_checkpoint_file(checkpoint_path: str) -> Optional[str]:
    """
    Find the model checkpoint file in a directory.
    
    Args:
        checkpoint_path: Directory containing checkpoint files
    
    Returns:
        Path to checkpoint file or None if not found
    """
    # Priority order for checkpoint files
    checkpoint_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    
    for filename in checkpoint_files:
        path = os.path.join(checkpoint_path, filename)
        if os.path.exists(path):
            return path
    
    # Also check for sharded checkpoints
    import glob
    sharded_patterns = [
        "pytorch_model-*.bin",
        "model-*.safetensors",
    ]
    
    for pattern in sharded_patterns:
        matches = glob.glob(os.path.join(checkpoint_path, pattern))
        if matches:
            # Return the directory for sharded checkpoints
            return checkpoint_path
    
    return None


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from checkpoint file or directory.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
    
    Returns:
        State dictionary
    """
    import glob
    
    if os.path.isfile(checkpoint_path):
        # Single file
        if checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path)
        else:
            return torch.load(checkpoint_path, map_location='cpu')
    
    elif os.path.isdir(checkpoint_path):
        # Check for sharded safetensors
        safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors")))
        if safetensor_files:
            print(f"Loading sharded safetensors from {len(safetensor_files)} files...")
            state_dict = {}
            for sf_file in safetensor_files:
                state_dict.update(load_file(sf_file))
            return state_dict
        
        # Check for sharded pytorch model
        bin_files = sorted(glob.glob(os.path.join(checkpoint_path, "pytorch_model-*.bin")))
        if bin_files:
            print(f"Loading sharded pytorch model from {len(bin_files)} files...")
            state_dict = {}
            for bin_file in bin_files:
                state_dict.update(torch.load(bin_file, map_location='cpu'))
            return state_dict
        
        # Single file in directory
        model_file = find_checkpoint_file(checkpoint_path)
        if model_file and os.path.isfile(model_file):
            if model_file.endswith(".safetensors"):
                return load_file(model_file)
            else:
                return torch.load(model_file, map_location='cpu')
    
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")


def load_model(
    model_size: str,
    pix_len: int,
    text_len: int,
    use_flash_attn: bool = True,
    checkpoint_path: Optional[str] = None,
    device_map: str = "auto",
    **kwargs,
) -> nn.Module:
    """
    Load OmniSVG model with appropriate settings.
    
    Args:
        model_size: "4B" or "8B"
        pix_len: Maximum SVG token length
        text_len: Maximum text length
        use_flash_attn: Whether to use Flash Attention 2
        checkpoint_path: Path to checkpoint, HuggingFace repo ID, or "auto"
        device_map: Device mapping strategy
    
    Returns:
        Loaded model
    """
    if model_size not in MODEL_DEFAULTS:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(MODEL_DEFAULTS.keys())}")
    
    # Load config to get model paths
    from utils import OmniSVGConfig as _TempConfig
    temp_config = _TempConfig(model_size=model_size)
    base_model = temp_config.base_model_path
    default_checkpoint = temp_config.checkpoint_path
    
    # Set attention implementation
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    
    print(f"=" * 60)
    print(f"Loading OmniSVG {model_size} Model")
    print(f"=" * 60)
    print(f"Base model: {base_model}")
    print(f"Attention implementation: {attn_implementation}")
    print(f"Max SVG tokens: {pix_len}")
    print(f"Max text length: {text_len}")
    
    # Get gradient checkpointing setting
    use_gradient_checkpointing = kwargs.get('use_gradient_checkpointing', False)
    
    # Initialize model from base
    # Note: device_map is passed from train() function, will be None for distributed training
    model = SketchDecoder(
        pix_len=pix_len,
        text_len=text_len,
        model_path=base_model,
        attn_implementation=attn_implementation,
        use_gradient_checkpointing=use_gradient_checkpointing,
        device_map=device_map,  # 分布式训练时为None，单GPU时可为"auto"
    )
    
    # Load checkpoint if specified
    if checkpoint_path:
        resolved_path = None
        
        # Handle "auto" keyword - use default checkpoint for model size
        if checkpoint_path.lower() == "auto":
            print(f"\nUsing 'auto' - downloading default checkpoint: {default_checkpoint}")
            resolved_path = download_omnisvg_checkpoint(default_checkpoint)
        
        # Handle HuggingFace repo ID (e.g., "OmniSVG/OmniSVG1.1_4B")
        elif is_hf_repo_id(checkpoint_path):
            print(f"\nDetected HuggingFace repo ID: {checkpoint_path}")
            resolved_path = download_omnisvg_checkpoint(checkpoint_path)
        
        # Handle local path
        elif os.path.exists(checkpoint_path):
            print(f"\nUsing local checkpoint: {checkpoint_path}")
            resolved_path = checkpoint_path
        
        else:
            # Try to download as HuggingFace repo
            print(f"\nPath not found locally, attempting to download from HuggingFace: {checkpoint_path}")
            try:
                resolved_path = download_omnisvg_checkpoint(checkpoint_path)
            except Exception as e:
                print(f"Warning: Failed to download checkpoint: {e}")
                resolved_path = None
        
        # Load the checkpoint
        if resolved_path:
            print(f"\nLoading checkpoint from: {resolved_path}")
            
            try:
                # Find and load checkpoint file
                if os.path.isfile(resolved_path):
                    model_file = resolved_path
                else:
                    model_file = find_checkpoint_file(resolved_path)
                
                if model_file:
                    print(f"Found checkpoint: {model_file}")
                    state_dict = load_checkpoint_state_dict(model_file if os.path.isfile(model_file) else resolved_path)
                    
                    # Load state dict with detailed logging
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    print(f"\nCheckpoint loading summary:")
                    print(f"  - Total keys in checkpoint: {len(state_dict)}")
                    print(f"  - Missing keys: {len(missing_keys)}")
                    print(f"  - Unexpected keys: {len(unexpected_keys)}")
                    
                    if missing_keys and len(missing_keys) < 20:
                        print(f"  - Missing key examples: {missing_keys[:5]}")
                    if unexpected_keys and len(unexpected_keys) < 20:
                        print(f"  - Unexpected key examples: {unexpected_keys[:5]}")
                    
                    print("\n✓ Checkpoint loaded successfully!")
                else:
                    print(f"Warning: No checkpoint file found in {resolved_path}")
                    print("Available files:", os.listdir(resolved_path) if os.path.isdir(resolved_path) else "N/A")
                    
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with base model weights...")
    else:
        print("\nNo checkpoint specified - using base model weights")
    
    print(f"=" * 60)
    
    return model


# ============================================================================
# Collate Functions
# ============================================================================

def create_collate_fn(
    processor: Any,
    text_len: int = 800,
    text_only_ratio: float = 0.5,
):
    """
    Create collate function for DataLoader.
    
    Args:
        processor: HuggingFace processor
        text_len: Maximum text length
        text_only_ratio: Ratio of text-only samples
    
    Returns:
        Collate function
    """
    system_prompt = "You are an expert SVG code generator."
    
    task_counter = {"total": 0, "text": 0}

    def collate_fn(batch):
        text_oris, pil_images, pix_seq_lists = zip(*batch)
        
        batch_messages = []
        batch_task_types = []
        
        task_assignments = []
        for _ in range(len(text_oris)):
            task_counter["total"] += 1
            # Use text_only_ratio instead of hardcoded 50%
            target_text_count = int(task_counter["total"] * text_only_ratio)
            
            if task_counter["text"] < target_text_count:
                task_assignments.append("text")
                task_counter["text"] += 1
            else:
                task_assignments.append("image")
        
        indices = list(range(len(task_assignments)))
        np.random.shuffle(indices)
        task_assignments = [task_assignments[i] for i in indices]
        
        for text_ori, pil_image, task_type in zip(text_oris, pil_images, task_assignments):
            if task_type == 'text':
                # Text-to-SVG task
                messages = [{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Generate SVG code for this text description: {text_ori}"}]
                }]
                batch_task_types.append("text")
            else:
                # Image-to-SVG task
                messages = [{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                        {"type": "image", "image": pil_image},
                    ]
                }]
                batch_task_types.append("image")
            
            batch_messages.append(messages)
        
        return batch_messages, list(pix_seq_lists), batch_task_types
    
    return collate_fn




def process_mixed_batch(
    batch_messages: List[Any],
    pix_seq_lists: List[List[int]],
    batch_task_types: List[str],
    processor: Any,
    config: OmniSVGConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], 
           Optional[torch.Tensor], torch.Tensor, Dict[str, List[int]]]:
    """
    Process a mixed batch of text and image tasks.
    
    Returns:
        input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    task_masks = {'text': [], 'image': []}
    
    pad_token_id = config.tokenization.pad_token_id
    max_len = config.training.max_seq_length + config.training.text_max_length
    
    # Separate tasks
    text_indices = []
    image_indices = []
    image_messages = []
    
    for i, (task_type, messages) in enumerate(zip(batch_task_types, batch_messages)):
        if task_type == "text":
            text_indices.append(i)
            task_masks['text'].append(i)
        else:
            image_indices.append(i)
            task_masks['image'].append(i)
            image_messages.append(messages)
    
    # Process text-only samples
    if text_indices:
        text_inputs = []
        for i in text_indices:
            text_input = processor.apply_chat_template(
                batch_messages[i], tokenize=False, add_generation_prompt=True
            )
            text_inputs.append(text_input)
        
        inputs = processor(
            text=text_inputs,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        
        for idx, i in enumerate(text_indices):
            input_ids, attention_mask, labels = _process_sample(
                inputs['input_ids'][idx],
                inputs['attention_mask'][idx],
                pix_seq_lists[i],
                max_len,
                pad_token_id,
            )
            batch_input_ids.append((i, input_ids))
            batch_attention_mask.append((i, attention_mask))
            batch_labels.append((i, labels))
    
    # Process image samples
    pixel_values = None
    image_grid_thw = None
    
    if image_indices and process_vision_info is not None:
        all_text_inputs = []
        all_image_inputs = []
        
        for messages in image_messages:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            all_text_inputs.append(text_input)
            all_image_inputs.extend(image_inputs)
        
        inputs = processor(
            text=all_text_inputs,
            images=all_image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )
        
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        
        for idx, i in enumerate(image_indices):
            input_ids, attention_mask, labels = _process_sample(
                inputs['input_ids'][idx].tolist(),
                inputs['attention_mask'][idx].tolist(),
                pix_seq_lists[i],
                max_len,
                pad_token_id,
            )
            batch_input_ids.append((i, input_ids))
            batch_attention_mask.append((i, attention_mask))
            batch_labels.append((i, labels))
    
    # Sort by original index and stack
    batch_input_ids.sort(key=lambda x: x[0])
    batch_attention_mask.sort(key=lambda x: x[0])
    batch_labels.sort(key=lambda x: x[0])
    
    input_ids = torch.stack([x[1] for x in batch_input_ids])
    attention_mask = torch.stack([x[1] for x in batch_attention_mask])
    labels = torch.stack([x[1] for x in batch_labels])
    
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks


def _process_sample(
    base_input_ids: List[int],
    base_attention_mask: List[int],
    pix_seq: List[int],
    max_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a single sample."""
    current_input_ids = base_input_ids + pix_seq
    current_attention_mask = base_attention_mask + [1] * len(pix_seq)
    
    instruction_len = len(base_input_ids)
    current_labels = [-100] * instruction_len + pix_seq
    
    pad_len = max_len - len(current_input_ids)
    
    if pad_len > 0:
        input_ids = [pad_token_id] * pad_len + current_input_ids
        attention_mask = [0] * pad_len + current_attention_mask
        labels = [-100] * pad_len + current_labels
    else:
        input_ids = current_input_ids[:max_len]
        attention_mask = current_attention_mask[:max_len]
        labels = current_labels[:max_len] if len(current_labels) <= max_len else current_labels[:max_len]
    
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ============================================================================
# Loss Computation
# ============================================================================

def compute_task_specific_losses(
    outputs: Any,
    labels: torch.Tensor,
    task_masks: Dict[str, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate losses for text and image tasks.
    
    Returns:
        text_loss, image_loss
    """
    batch_size = outputs.logits.size(0)
    device = outputs.logits.device
    
    logits = outputs.logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous().to(device)
    
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    per_token_loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).view(batch_size, -1)
    
    valid_tokens = (labels != -100).float()
    per_sample_loss = per_token_loss.sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
    
    # Separate by task
    text_losses = [per_sample_loss[i] for i in task_masks['text']]
    image_losses = [per_sample_loss[i] for i in task_masks['image']]
    
    text_loss = torch.stack(text_losses).mean() if text_losses else torch.tensor(0.0, device=device)
    image_loss = torch.stack(image_losses).mean() if image_losses else torch.tensor(0.0, device=device)
    
    return text_loss, image_loss


# ============================================================================
# Training Functions
# ============================================================================

def train(args, config: OmniSVGConfig):
    """Main training function."""
    
    # Initialize accelerator with extended timeout for FSDP
    # 设置分布式后端的超时时间（对FSDP很重要）
    from datetime import timedelta
    from accelerate.utils import InitProcessGroupKwargs
    
    # 创建超时配置：60分钟（3600秒）
    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ]
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        kwargs_handlers=kwargs_handlers,  # 传递timeout配置
        log_with=None,  # 禁用额外的日志系统
    )
    
    if accelerator.is_main_process:
        print(f"🔧 Initialized Accelerator with NCCL timeout = 3600 seconds (60 minutes)")
    
    # Set seed
    set_seed(config.training.seed)
    
    # Get base model path
    base_model_path = config.base_model_path
    accelerator.print(f"Using base model: {base_model_path}")
    accelerator.print(f"Model size: {config.model_size}")
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, padding_side="left"
    )
    processor = AutoProcessor.from_pretrained(
        base_model_path, padding_side="left", use_fast=True
    )
    processor.tokenizer.padding_side = "left"
    
    # Load or download data
    if args.use_hf_data:
        accelerator.print("Downloading datasets from HuggingFace...")
        train_csv, val_csv, svg_folder, png_folder = download_omnisvg_data(
            output_dir=args.data_dir,
            datasets=args.datasets,
            train_ratio=0.95,
            max_token_length=config.training.max_seq_length,
        )
    else:
        # Use local data
        train_csv = config.data.train_meta_file
        val_csv = config.data.val_meta_file
        svg_folder = config.data.svg_folder
        png_folder = config.data.png_folder
    
    accelerator.print(f"Train data: {train_csv}")
    accelerator.print(f"Val data: {val_csv}")
    accelerator.print(f"SVG folder: {svg_folder}")
    accelerator.print(f"PNG folder: {png_folder}")
    
    # Create datasets
    train_dataset = OmniSVGDataset(
        meta_file=train_csv,
        svg_folder=svg_folder,
        png_folder=png_folder,
        max_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        tokenizer=tokenizer,
        processor=processor,
        token_config=config.tokenization,
        train_config=config.training,
    )
    
    val_dataset = OmniSVGDataset(
        meta_file=val_csv,
        svg_folder=svg_folder,
        png_folder=png_folder,
        max_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        tokenizer=tokenizer,
        processor=processor,
        token_config=config.tokenization,
        train_config=config.training,
    )
    
    # Create collate functions
    train_collate = create_collate_fn(
        processor, 
        text_len=config.training.text_max_length,
        text_only_ratio=config.training.text_only_ratio,
    )
    val_collate = create_collate_fn(
        processor,
        text_len=config.training.text_max_length,
        text_only_ratio=config.training.text_only_ratio,
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=train_collate,
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=val_collate,
    )
    
    # Initialize model
    # Important: device_map=None for distributed training (let Accelerate manage devices)
    model = load_model(
        model_size=config.model_size,
        pix_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        use_flash_attn=config.training.use_flash_attn,
        checkpoint_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None,
        use_gradient_checkpointing=config.training.use_gradient_checkpointing,
        device_map=None,  # 关键：分布式训练时必须为None
    )
    
    # Optimizer
    lr = config.training.learning_rate * accelerator.num_processes
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # Scheduler
    # Use optimizer steps (after gradient accumulation), not raw batch steps
    num_update_steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * config.training.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    accelerator.print(f"Scheduler: {total_steps} total optimizer steps, {config.training.warmup_steps} warmup steps")
    
    # Prepare for distributed training
    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )
    
    # 如何禁用：设置环境变量 DISABLE_TORCH_COMPILE=1
    #   export DISABLE_TORCH_COMPILE=1
    #   bash debug_run_multi.sh
    # 
    # 注意：这是实验性功能，如果遇到Triton错误，建议禁用
    use_compile = os.environ.get("DISABLE_TORCH_COMPILE", "0") != "1"
    
    if use_compile:
        try:
            if accelerator.is_main_process:
                print("\n" + "="*60)
                print("🔥 Compiling model with torch.compile()...")
                print("   首次编译需要5-10分钟，请耐心等待...")
                print("   编译后每步训练速度将提升10-30%")
                print("   如遇到错误，设置 DISABLE_TORCH_COMPILE=1 禁用")
                print("="*60 + "\n")
            
            # 编译模型
            # mode选项:
            #   - "default": 平衡编译时间和性能
            #   - "reduce-overhead": 减少Python开销，适合小batch（推荐）
            #   - "max-autotune": 最大性能，但编译时间很长
            model = torch.compile(
                model, 
                mode="reduce-overhead",  # 推荐：减少Python开销
                fullgraph=False,  # FSDP/DDP不支持fullgraph
                dynamic=True,  # 支持动态shape
            )
            
            if accelerator.is_main_process:
                print("✓ Model compiled successfully!")
                print("  后续训练步骤将自动加速\n")
                
        except Exception as e:
            # 如果编译失败，继续使用未编译的模型
            if accelerator.is_main_process:
                print(f"\n⚠️  torch.compile() 失败，将使用未编译模型")
                print(f"   错误: {str(e)[:100]}...")
                print("   建议: 设置 DISABLE_TORCH_COMPILE=1 禁用编译优化")
                print("   训练将继续，但速度不会有torch.compile的加速\n")
    else:
        if accelerator.is_main_process:
            print("\n" + "="*60)
            print("ℹ️  torch.compile() 已禁用 (DISABLE_TORCH_COMPILE=1)")
            print("   训练将使用未编译模型")
            print("="*60 + "\n")
    
    # Setup logging
    output_dir = Path(args.output_dir) / args.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    
    # Initialize Weights & Biases (只在主进程)
    use_wandb = WANDB_AVAILABLE and args.use_wandb and accelerator.is_main_process
    if use_wandb:
        # 准备wandb配置
        wandb_config = {
            "model_size": config.model_size,
            "batch_size": args.batch_size,
            "num_gpus": accelerator.num_processes,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "max_seq_length": config.training.max_seq_length,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "use_flash_attn": config.training.use_flash_attn,
            "use_gradient_checkpointing": config.training.use_gradient_checkpointing,
            "text_loss_weight": config.training.text_loss_weight,
            "image_loss_weight": config.training.image_loss_weight,
        }
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project or "omnisvg-training",
            name=args.project_name,
            config=wandb_config,
            dir=str(output_dir),
            resume="allow" if args.resume_from_checkpoint else False,
        )
        
        print(f"\n✅ Weights & Biases initialized!")
        print(f"📊 View training at: {wandb.run.get_url()}\n")
    
    # Save config
    if accelerator.is_main_process:
        config.save(str(output_dir / "config.yaml"))
        with open(output_dir / "args.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Resume from checkpoint
    starting_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training")
    print(f"{'='*60}")
    print(f"Total Epochs:     {config.training.epochs}")
    print(f"Steps per Epoch:  {num_update_steps_per_epoch}")
    print(f"Total Steps:      {total_steps}")
    print(f"Batch Size:       {args.batch_size} per GPU × {accelerator.num_processes} GPUs")
    print(f"Grad Accum:       {config.training.gradient_accumulation_steps}")
    print(f"Learning Rate:    {lr:.2e}")
    print(f"Log Every:        {config.training.log_every} steps")
    print(f"Save Every:       {config.training.save_every} steps")
    print(f"Validate Every:   {config.training.val_every} steps")
    print(f"{'='*60}\n")
    
    # ⚡ 性能优化：这些列表现在存储detached tensor而不是float
    # 只在log时才转换为Python数值，减少CPU-GPU同步开销
    text_losses = []
    image_losses = []
    grad_norms = []
    
    for epoch in range(starting_epoch, config.training.epochs):
        model.train()
        print(f"\n{'='*60}")
        print(f"📚 Epoch {epoch + 1}/{config.training.epochs}")
        print(f"{'='*60}\n")
        
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}"
        )
        
        for batch_messages, pix_seq_lists, batch_task_types in train_dataloader:
            with accelerator.accumulate(model):
                # Process batch
                input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks = \
                    process_mixed_batch(
                        batch_messages, pix_seq_lists, batch_task_types, processor, config
                    )
                
                # Move to device
                input_ids = input_ids.to(accelerator.device)
                attention_mask = attention_mask.to(accelerator.device)
                labels = labels.to(accelerator.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(accelerator.device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(accelerator.device)
                
                # # Debug: check token bounds before forward
                # _vocab = 197000
                # _max_id = input_ids.max().item()
                # _min_id = input_ids.min().item()
                # _lbl_valid = labels[labels != -100]
                # if _max_id >= _vocab or _min_id < 0:
                #     print(f"[OOB] input_ids range: [{_min_id}, {_max_id}], vocab={_vocab}")
                #     _oob_mask = (input_ids >= _vocab) | (input_ids < 0)
                #     _oob_positions = _oob_mask.nonzero(as_tuple=True)
                #     for b, s in zip(_oob_positions[0][:5], _oob_positions[1][:5]):
                #         print(f"  input_ids[{b},{s}] = {input_ids[b,s].item()}")
                # if len(_lbl_valid) > 0:
                #     _lbl_max = _lbl_valid.max().item()
                #     _lbl_min = _lbl_valid.min().item()
                #     if _lbl_max >= _vocab or _lbl_min < 0:
                #         print(f"[OOB] labels range: [{_lbl_min}, {_lbl_max}], vocab={_vocab}")
                #         _oob_lbl = (_lbl_valid >= _vocab) | (_lbl_valid < 0)
                #         print(f"  OOB label values: {_lbl_valid[_oob_lbl][:10].tolist()}")
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,
                )
                
                # Compute losses
                text_loss, image_loss = compute_task_specific_losses(outputs, labels, task_masks)
                
                # Weighted loss
                loss = (config.training.text_loss_weight * text_loss + 
                       config.training.image_loss_weight * image_loss)
                
                # ⚡ 性能优化：保存tensor而不是立即.item()，避免CPU-GPU同步
                # 只在需要记录日志时才转换为Python数值
                current_loss = loss.detach()  # 保存detached tensor
                if text_loss > 0:
                    text_losses.append(text_loss.detach())
                if image_loss > 0:
                    image_losses.append(image_loss.detach())
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.training.max_grad_norm
                    )
                    # ⚡ 性能优化：保存tensor而不是立即转换
                    if grad_norm is not None:
                        if hasattr(grad_norm, 'detach'):
                            grad_norms.append(grad_norm.detach())
                        else:
                            grad_norms.append(grad_norm)  # 已经是标量
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # ⚡ 性能优化：只在需要显示时才.item()（每10步更新一次进度条）
                    if global_step % 10 == 0:
                        # 这里才转换为Python数值用于显示
                        display_loss = current_loss.item() if hasattr(current_loss, 'item') else current_loss
                        progress_bar.set_postfix({
                            'loss': f'{display_loss:.4f}',
                            'step': global_step
                        })
                        progress_bar.update(10)
                    
                    # Logging
                    if global_step % config.training.log_every == 0:
                        log_metrics(
                            writer, global_step, text_losses, image_losses, 
                            grad_norms, lr_scheduler, accelerator, use_wandb
                        )
                        text_losses = []
                        image_losses = []
                        grad_norms = []
                    
                    # Save checkpoint
                    if global_step % config.training.save_every == 0:
                        if accelerator.is_main_process:
                            print(f"\n⏱️  Step {global_step}: Checkpoint save triggered...")
                        
                        save_checkpoint(
                            output_dir, global_step, epoch,
                            model, optimizer, lr_scheduler,
                            accelerator,
                        )
                        
                        if accelerator.is_main_process:
                            print(f"✓ Checkpoint save completed, continuing training...\n")
                    
                    # Validation
                    if global_step % config.training.val_every == 0:
                        if accelerator.is_main_process:
                            print(f"\n⏱️  Step {global_step}: Validation triggered...")
                        
                        val_loss = validate(
                            model, val_dataloader, processor, config,
                            accelerator, writer, global_step
                        )
                        
                        if accelerator.is_main_process:
                            print(f"✓ Validation completed, continuing training...\n")
                        
                        # 检查是否是最佳模型（所有进程都需要参与判断）
                        # 使用broadcast确保所有进程知道是否需要保存
                        is_best_model = val_loss < best_val_loss
                        
                        # 广播决定到所有进程（避免进程分歧）
                        if torch.distributed.is_initialized():
                            is_best_tensor = torch.tensor(
                                [1.0 if is_best_model else 0.0], 
                                device=accelerator.device
                            )
                            torch.distributed.broadcast(is_best_tensor, src=0)
                            is_best_model = is_best_tensor.item() > 0.5
                        
                        # 所有进程都调用save_checkpoint（关键！）
                        if is_best_model:
                            if accelerator.is_main_process:
                                best_val_loss = val_loss
                                print(f"\n🌟 New best model! Val loss: {val_loss:.4f}")
                            
                            # 所有进程都必须调用，即使只有主进程实际保存
                            save_checkpoint(
                                output_dir, global_step, epoch,
                                model, optimizer, lr_scheduler,
                                accelerator, is_best=True,
                            )
                        
                        model.train()
        
        progress_bar.close()
        
        # End of epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            writer.flush()
        
        torch.cuda.empty_cache()
    
    # Cleanup
    if accelerator.is_main_process:
        writer.close()
        
        # Finish wandb run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Final Step:       {global_step}")
    print(f"Best Val Loss:    {best_val_loss:.4f}")
    print(f"Checkpoints:      {output_dir}")
    print(f"{'='*60}\n")


def validate(
    model: nn.Module,
    val_dataloader: Any,
    processor: Any,
    config: OmniSVGConfig,
    accelerator: Accelerator,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Run validation."""
    model.eval()
    accelerator.print(f"Validation at step {step}...")
    
    total_losses = []
    text_losses = []
    image_losses = []
    
    with torch.no_grad():
        for batch_messages, pix_seq_lists, batch_task_types in tqdm(
            val_dataloader, 
            disable=not accelerator.is_local_main_process
        ):
            input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks = \
                process_mixed_batch(
                    batch_messages, pix_seq_lists, batch_task_types, processor, config
                )
            
            # Move to device
            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)
            labels = labels.to(accelerator.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(accelerator.device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(accelerator.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
            )
            
            text_loss, image_loss = compute_task_specific_losses(outputs, labels, task_masks)
            total_loss = text_loss + image_loss
            
            # Gather across processes
            all_total = accelerator.gather_for_metrics(total_loss)
            all_text = accelerator.gather_for_metrics(text_loss)
            all_image = accelerator.gather_for_metrics(image_loss)
            
            total_losses.append(all_total.mean().item())
            if all_text.sum() > 0:
                text_losses.append(all_text.mean().item())
            if all_image.sum() > 0:
                image_losses.append(all_image.mean().item())
    
    # Compute averages
    avg_total = np.mean(total_losses) if total_losses else 0
    avg_text = np.mean(text_losses) if text_losses else 0
    avg_image = np.mean(image_losses) if image_losses else 0
    
    # 打印验证结果
    print(f"\n{'='*60}")
    print(f"[Validation @ Step {step}]")
    print(f"  Total Loss:  {avg_total:.4f}")
    print(f"  Image Loss:  {avg_image:.4f}")
    print(f"  Text Loss:   {avg_text:.4f}")
    print(f"{'='*60}\n")
    
    if accelerator.is_main_process:
        # TensorBoard logging
        writer.add_scalar("validation/total_loss", avg_total, step)
        # writer.add_scalar("validation/text_loss", avg_text, step)
        writer.add_scalar("validation/image_loss", avg_image, step)
        
        # Weights & Biases logging
        if WANDB_AVAILABLE and wandb.run is not None:
            val_metrics = {
                "val/loss_total": avg_total,
                "val/loss_image": avg_image,
            }
            if avg_text > 0:
                val_metrics["val/loss_text"] = avg_text
            wandb.log(val_metrics, step=step)
    
    return avg_total


def log_metrics(
    writer: SummaryWriter,
    step: int,
    text_losses: List,  # 现在接受tensor列表
    image_losses: List,  # 现在接受tensor列表
    grad_norms: List,  # 现在接受tensor列表
    lr_scheduler: Any,
    accelerator: Accelerator,
    use_wandb: bool = False,
):
    """Log training metrics to TensorBoard and Weights & Biases.
    
    ⚡ 性能优化：接受tensor列表，只在这里转换为Python数值
    这样避免了训练循环中的频繁CPU-GPU同步
    """
    if not accelerator.is_main_process:
        return
    
    # ⚡ 只在这里转换tensor为数值（一次性批量转换）
    def to_float_list(tensor_list):
        """将tensor列表转换为float列表"""
        if not tensor_list:
            return []
        # 批量转换：stack所有tensor，一次性.item()
        if hasattr(tensor_list[0], 'item'):
            return [t.item() for t in tensor_list]
        else:
            return list(tensor_list)  # 已经是数值
    
    text_losses_float = to_float_list(text_losses)
    image_losses_float = to_float_list(image_losses)
    grad_norms_float = to_float_list(grad_norms)
    
    avg_text = np.mean(text_losses_float) if text_losses_float else 0
    avg_image = np.mean(image_losses_float) if image_losses_float else 0
    avg_total = np.mean(text_losses_float + image_losses_float) if (text_losses_float + image_losses_float) else 0
    avg_grad = np.mean(grad_norms_float) if grad_norms_float else 0
    current_lr = lr_scheduler.get_last_lr()[0]
    
    # 打印训练指标到控制台
    print(f"\n[Step {step}] Loss: {avg_total:.4f} (Image: {avg_image:.4f}, Text: {avg_text:.4f}) | "
          f"Grad Norm: {avg_grad:.4f} | LR: {current_lr:.2e}")
    
    # TensorBoard logging
    writer.add_scalar("loss/total", avg_total, step)
    # writer.add_scalar("loss/text_task", avg_text, step)
    writer.add_scalar("loss/image_task", avg_image, step)
    writer.add_scalar("lr", current_lr, step)
    writer.add_scalar("grad_norm", avg_grad, step)
    
    # Weights & Biases logging
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "train/loss_total": avg_total,
            "train/loss_image": avg_image,
            "train/loss_text": avg_text,
            "train/learning_rate": current_lr,
            "train/grad_norm": avg_grad,
        }, step=step)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: Any,
    lr_scheduler: Any,
    accelerator: Accelerator,
    is_best: bool = False,
):
    """Save training checkpoint using Accelerate's save_state for FSDP compatibility."""
    
    # 先确保所有GPU完成当前工作，避免保存时的竞争
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"💾 Saving checkpoint at step {step}...")
        print(f"{'='*60}")
    
    # 清空CUDA缓存，释放不必要的显存
    torch.cuda.empty_cache()
    
    # 确保所有进程同步到这里
    accelerator.wait_for_everyone()
    
    if is_best:
        ckpt_name = "best_model"
    else:
        ckpt_name = f"step_{step}"
    
    ckpt_path = output_dir / ckpt_name
    
    # 使用Accelerate的save_state方法，自动处理FSDP/DDP的checkpoint保存
    # 这个方法会正确处理FSDP的状态字典收集，避免NCCL超时
    try:
        import time
        start_time = time.time()
        
        # save_state会自动在所有进程间同步
        # 对于FSDP，这会调用正确的FSDP state_dict方法
        accelerator.save_state(str(ckpt_path))
        
        elapsed = time.time() - start_time
        
        # 只在主进程保存额外信息和打印日志
        if accelerator.is_main_process:
            # Save training state info
            info_dict = {
                'step': step,
                'epoch': epoch,
                'is_best': is_best,
                'save_time_seconds': elapsed,
            }
            
            # save_state 已经创建了目录，直接保存额外信息
            with open(ckpt_path / "training_info.json", 'w') as f:
                json.dump(info_dict, f, indent=2)
            
            print(f"✓ Checkpoint saved to: {ckpt_path}")
            print(f"  Save time: {elapsed:.1f} seconds")
            
            # Auto-merge FSDP sharded checkpoint to a single file
            _merge_fsdp_if_needed(ckpt_path)
            
            print(f"{'='*60}\n")
    
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"⚠ WARNING: Failed to save checkpoint at step {step}")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
        
        # 即使保存失败也继续训练
        # 只记录警告，不中断训练
    
    finally:
        # 确保所有进程都同步，即使发生异常
        # 这是关键！避免某些进程提前退出导致进程不匹配
        accelerator.wait_for_everyone()

def _merge_fsdp_if_needed(ckpt_path: Path):
    """Merge FSDP sharded checkpoint (.distcp) into a single model.safetensors file."""
    import glob as _glob
    fsdp_dirs = sorted(_glob.glob(str(ckpt_path / "pytorch_model_fsdp_*")))
    if not fsdp_dirs:
        return
    
    fsdp_dir = fsdp_dirs[0]
    distcp_files = _glob.glob(os.path.join(fsdp_dir, "*.distcp"))
    if not distcp_files:
        return
    
    output_path = str(ckpt_path / "model.safetensors")
    if os.path.exists(output_path):
        return
    
    print(f"  Merging {len(distcp_files)} FSDP shards -> model.safetensors ...")
    try:
        import time
        t0 = time.time()
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        
        tmp_path = output_path + ".tmp"
        dcp_to_torch_save(fsdp_dir, tmp_path)
        
        state_dict = torch.load(tmp_path, map_location="cpu", weights_only=False)
        
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        
        from safetensors.torch import save_file
        save_file(state_dict, output_path)
        
        os.remove(tmp_path)
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ Merged model saved: {output_path} ({size_mb:.0f} MB, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠ Failed to merge FSDP shards: {e}")
        if os.path.exists(output_path + ".tmp"):
            os.remove(output_path + ".tmp")



def _merge_fsdp_if_needed(ckpt_path: Path):
    """Merge FSDP sharded checkpoint (.distcp) into a single model.safetensors file."""
    import glob as _glob
    fsdp_dirs = sorted(_glob.glob(str(ckpt_path / "pytorch_model_fsdp_*")))
    if not fsdp_dirs:
        return
    
    fsdp_dir = fsdp_dirs[0]
    distcp_files = _glob.glob(os.path.join(fsdp_dir, "*.distcp"))
    if not distcp_files:
        return
    
    output_path = str(ckpt_path / "model.safetensors")
    if os.path.exists(output_path):
        return
    
    print(f"  Merging {len(distcp_files)} FSDP shards -> model.safetensors ...")
    try:
        import time
        t0 = time.time()
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        
        tmp_path = output_path + ".tmp"
        dcp_to_torch_save(fsdp_dir, tmp_path)
        
        state_dict = torch.load(tmp_path, map_location="cpu", weights_only=False)
        
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        
        from safetensors.torch import save_file
        save_file(state_dict, output_path)
        
        os.remove(tmp_path)
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ Merged model saved: {output_path} ({size_mb:.0f} MB, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠ Failed to merge FSDP shards: {e}")
        if os.path.exists(output_path + ".tmp"):
            os.remove(output_path + ".tmp")


# ============================================================================
# Main Entry Point
# ============================================================================

def print_model_info():
    """Print available model configurations."""
    print("\n" + "=" * 60)
    print("Available OmniSVG Model Configurations")
    print("=" * 60)
    
    for size, info in MODEL_DEFAULTS.items():
        print(f"\n{size} Model:")
        print(f"  Base Model:  {info['base_model']}")
        print(f"  Checkpoint:  {info['checkpoint']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OmniSVG Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Train 4B model with local data
    accelerate launch train.py --model_size 4B --data_dir ./data
    
    # Train 8B model with Flash Attention
    accelerate launch train.py --model_size 8B --use_flash_attn --data_dir ./data
    
    # Train with HuggingFace data (auto-download)
    accelerate launch train.py --model_size 4B --use_hf_data --datasets illustration icon
    
    # Resume from OmniSVG checkpoint (auto-download based on model size)
    accelerate launch train.py --model_size 4B --resume_from_checkpoint auto
    
    # Resume from specific HuggingFace checkpoint
    accelerate launch train.py --model_size 4B --resume_from_checkpoint OmniSVG/OmniSVG1.1_4B
    """
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_size", type=str, default="4B",
                            choices=["4B", "8B"],
                            help="Model size: 4B (Qwen2.5-VL-3B based) or 8B (Qwen2.5-VL-7B based)")
    model_group.add_argument("--use_flash_attn", action="store_true",
                            help="Enable Flash Attention 2 for faster training")
    model_group.add_argument("--no_flash_attn", action="store_true",
                            help="Disable Flash Attention 2")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data_dir", type=str, default=None,
                           help="Data directory containing train_meta.csv, val_meta.csv, svg/, png/ (overrides config file)")
    data_group.add_argument("--use_hf_data", action="store_true",
                           help="Download and use HuggingFace datasets")
    data_group.add_argument("--datasets", type=str, nargs="+",
                           default=["illustration", "icon"],
                           choices=["illustration", "icon"],
                           help="HuggingFace datasets to use (when --use_hf_data)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--config_dir", type=str, default="./configs",
                            help="Directory containing config files")
    train_group.add_argument("--train_config_file", type=str, default="train_config.yaml",
                            help="Training config filename (e.g., train_config.yaml, train_config_low_memory.yaml)")
    train_group.add_argument("--output_dir", type=str, default="./output",
                            help="Output directory for checkpoints and logs")
    train_group.add_argument("--project_name", type=str, default=None,
                            help="Project name (default: omnisvg_{model_size})")
    train_group.add_argument("--batch_size", type=int, default=4,
                            help="Batch size per device")
    train_group.add_argument("--max_seq_length", type=int, default=2048,
                            help="Maximum SVG sequence length")
    train_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="Path to checkpoint, HuggingFace repo ID (e.g., OmniSVG/OmniSVG1.1_4B), or 'auto'")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument("--use_wandb", action="store_true", default=True,
                              help="Enable Weights & Biases for cloud visualization")
    logging_group.add_argument("--wandb_project", type=str, default=None,
                              help="Weights & Biases project name (default: omnisvg-training)")
    
    # Utility options
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available HuggingFace datasets and exit")
    parser.add_argument("--list_models", action="store_true",
                       help="List available model configurations and exit")
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list_datasets: # False
        list_available_datasets()
        return
    
    # List models and exit
    if args.list_models: # False
        print_model_info()
        return
    
    # Set default project name
    if args.project_name is None:
        args.project_name = f"omnisvg_{args.model_size.lower()}"
    
    # Load configuration 导入tokenization.yaml和train_config.yaml
    config = OmniSVGConfig(
        config_dir=args.config_dir,
        train_file=args.train_config_file,
        model_size=args.model_size,
    )
    
    # Override config with command line args
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length
    if args.data_dir:
        config.training.data_dir = args.data_dir
        config.data.data_dir = args.data_dir
    
    # Handle flash attention flag
    if args.no_flash_attn:
        config.training.use_flash_attn = False
    elif args.use_flash_attn:
        config.training.use_flash_attn = True
    
    print(f"\n{'='*60}")
    print(f"OmniSVG Training Configuration")
    print(f"{'='*60}")
    print(f"Model Size:        {config.model_size}")
    print(f"Base Model:        {config.base_model_path}")
    print(f"Default Checkpoint:{config.checkpoint_path}")
    print(f"Flash Attention:   {config.training.use_flash_attn}")
    print(f"Gradient Checkpoint:{config.training.use_gradient_checkpointing}")
    print(f"Data Directory:    {config.training.data_dir}")
    print(f"Target Image Size: {config.training.target_image_size}")
    print(f"Max Seq Length:    {config.training.max_seq_length}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Grad Accum Steps:  {config.training.gradient_accumulation_steps}")
    print(f"Output Directory:  {args.output_dir}/{args.project_name}")
    if args.resume_from_checkpoint:
        print(f"Resume From:       {args.resume_from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Run training
    train(args, config)


if __name__ == "__main__":
    main()
