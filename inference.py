import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from PIL import Image
import cairosvg
import io
import tempfile
import argparse
import gc
import yaml
import glob
import re
import numpy as np
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from decoder import SketchDecoder
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tokenizer import SVGTokenizer, TrainAlignedSVGTokenizer
from deepsvg.svglib.svg import SVG as DeepSVG
from utils.dataset import SVGTokenizer as TrainingSVGEncoder

# Load config
CONFIG_PATH = "./configs/config_code_complement.yaml"
# "./config_code_complement.yaml"
# "./config_zhuan.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Use a default device, but we'll get the actual device from the model later
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Global Models
tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None
svg_condition_encoder = None
current_model_size = None

# Constants from config
SYSTEM_PROMPT = """You are an expert SVG code generator. 
Generate precise, valid SVG path commands that accurately represent the described scene or object.
Focus on capturing key shapes, spatial relationships, and visual composition."""

SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
AVAILABLE_MODEL_SIZES = list(config.get('models', {}).keys())
DEFAULT_MODEL_SIZE = config.get('default_model_size', '8B')


def get_config_value(model_size, *keys):
    """Get config value with model-specific override support."""
    model_cfg = config.get('models', {}).get(model_size, {})
    value = model_cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    if value is None:
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    
    return value


# Image processing settings from config
image_config = config.get('image', {})
TARGET_IMAGE_SIZE = image_config.get('target_size', 448)
RENDER_SIZE = image_config.get('render_size', 512)
BACKGROUND_THRESHOLD = image_config.get('background_threshold', 240)
EMPTY_THRESHOLD_ILLUSTRATION = image_config.get('empty_threshold_illustration', 250)
EMPTY_THRESHOLD_ICON = image_config.get('empty_threshold_icon', 252)
EDGE_SAMPLE_RATIO = image_config.get('edge_sample_ratio', 0.1)
COLOR_SIMILARITY_THRESHOLD = image_config.get('color_similarity_threshold', 30)
MIN_EDGE_SAMPLES = image_config.get('min_edge_samples', 10)

# Color settings from config
colors_config = config.get('colors', {})
BLACK_COLOR_TOKEN = colors_config.get('black_color_token', 
                                       colors_config.get('color_token_start', 40010) + 2)

# Model settings from config
model_config = config.get('model', {})
BOS_TOKEN_ID = model_config.get('bos_token_id', 196998)
EOS_TOKEN_ID = model_config.get('eos_token_id', 196999)
PAD_TOKEN_ID = model_config.get('pad_token_id', 151643)
MAX_LENGTH = model_config.get('max_length', 1024)
MIN_MAX_LENGTH = 256
MAX_MAX_LENGTH = 2048

# Skeleton CoT marker tokens
SKELETON_START_TOKEN_ID = model_config.get('skeleton_start_token_id', 197000)
SKELETON_END_TOKEN_ID = model_config.get('skeleton_end_token_id', 197001)
REPLACEMENT_START_TOKEN_ID = model_config.get('replacement_start_token_id', 197002)
REPLACEMENT_END_TOKEN_ID = model_config.get('replacement_end_token_id', 197003)
SKELETON_COT_MARKERS = {SKELETON_START_TOKEN_ID, SKELETON_END_TOKEN_ID,
                        REPLACEMENT_START_TOKEN_ID, REPLACEMENT_END_TOKEN_ID}

# Task configurations with defaults from config
task_config = config.get('task_configs', {})

TASK_CONFIGS = {
    "text-to-svg-icon": task_config.get('text_to_svg_icon', {
        "default_temperature": 0.5,
        "default_top_p": 0.88,
        "default_top_k": 50,
        "default_repetition_penalty": 1.05,
    }),
    "text-to-svg-illustration": task_config.get('text_to_svg_illustration', {
        "default_temperature": 0.6,
        "default_top_p": 0.90,
        "default_top_k": 60,
        "default_repetition_penalty": 1.03,
    }),
    "image-to-svg": task_config.get('image_to_svg', {
        "default_temperature": 0.3,
        "default_top_p": 0.90,
        "default_top_k": 50,
        "default_repetition_penalty": 1.05,
    }),
    "code-complement": task_config.get('code_complement', {
        "default_temperature": 0.3,
        "default_top_p": 0.90,
        "default_top_k": 50,
        "default_repetition_penalty": 1.05,
    })
}

# Generation parameters from config
gen_config = config.get('generation', {})
DEFAULT_NUM_CANDIDATES = gen_config.get('default_num_candidates', 4)
MAX_NUM_CANDIDATES = gen_config.get('max_num_candidates', 8)
EXTRA_CANDIDATES_BUFFER = gen_config.get('extra_candidates_buffer', 4)

# Validation settings from config
validation_config = config.get('validation', {})
MIN_SVG_LENGTH = validation_config.get('min_svg_length', 20)


def get_model_input_device():
    """
    Get the device where model inputs should be placed.
    This handles multi-GPU scenarios where the model is distributed across devices.
    """
    global sketch_decoder
    
    if sketch_decoder is None:
        return default_device
    
    try:
        # Get the transformer model
        model = sketch_decoder.transformer
        
        # Try to get device from the embedding layer (this is where input_ids will be processed)
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_device = next(model.model.embed_tokens.parameters()).device
            return embed_device
        elif hasattr(model, 'embed_tokens'):
            embed_device = next(model.embed_tokens.parameters()).device
            return embed_device
        
        # Alternative: try to get from the first parameter
        first_param = next(model.parameters())
        return first_param.device
        
    except (StopIteration, AttributeError) as e:
        print(f"Warning: Could not determine model device, using default: {default_device}")
        return default_device


def get_model_devices_info():
    """Get information about which devices the model is using (for debugging)."""
    global sketch_decoder
    
    if sketch_decoder is None:
        return "Model not loaded"
    
    devices = set()
    try:
        model = sketch_decoder.transformer
        for name, param in model.named_parameters():
            devices.add(str(param.device))
    except Exception as e:
        return f"Error getting device info: {e}"
    
    return f"Model distributed across: {sorted(devices)}"


def parse_args():
    parser = argparse.ArgumentParser(description='OmniSVG Inference Script')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True, choices=['text-to-svg', 'image-to-svg', 'code-complement'],
                        help='Task type: text-to-svg, image-to-svg, or code-complement')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (.txt for text-to-svg) or directory (for image-to-svg)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for generated SVGs')
    
    # Model settings
    parser.add_argument('--model-size', type=str, default=DEFAULT_MODEL_SIZE,
                        choices=AVAILABLE_MODEL_SIZES,
                        help=f'Model size to use (default: {DEFAULT_MODEL_SIZE})')
    parser.add_argument('--model-path', type=str, default="/data/phd23_weiguang_zhang/works/svg/qwen25vl3b",
                        help='Local path or HuggingFace repo ID for Qwen model (overrides config)')
    parser.add_argument('--weight-path', type=str, default="output_stage2/omnisvg_4b_20260603_081248/step_20000/model.safetensors",
    # "output_stage2/omnisvg_4b_20260604_100343/step_180/model.safetensors",
    # "output_stage2/omnisvg_4b_20260603_081248/step_12000/model.safetensors",
    # "/home/bingxing2/home/scx7l3f/weiguang_zhang/project/OmniSVG-train/output/omnisvg_4b_20260410_215008/step_7500/pytorch_model_fsdp_0"
    # "output/omnisvg_4b_20260210_022748/step_33000/pytorch_model.bin",
    # "output/omnisvg_4b_20260210_022748/step_3000",
    # "output/omnisvg_4b_20260406_081050/step_5000/model.safetensors",
    # "output/omnisvg_4b_20260407_022123/step_5000",
    # "output/omnisvg_4b_20260407_175744/step_2000",
    # "output/omnisvg_4b_20260408_020736/step_8000",
    # "output/omnisvg_4b_20260409_045251/step_2000",
    # "/data/phd23_weiguang_zhang/works/svg/models--OmniSVG--OmniSVG1.1_4B/snapshots/e4d03a89aaa28468520b45dc2541098102264d4e",
                        help='Local path or HuggingFace repo ID for OmniSVG weights (overrides config)')
    
    # Generation parameters
    parser.add_argument('--num-candidates', type=int, default=DEFAULT_NUM_CANDIDATES,
                        help=f'Number of candidates to generate (default: {DEFAULT_NUM_CANDIDATES})')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: task-specific)')
    parser.add_argument('--top-p', type=float, default=None,
                        help='Top-p sampling (default: task-specific)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling (default: task-specific)')
    parser.add_argument('--repetition-penalty', type=float, default=None,
                        help='Repetition penalty (default: task-specific)')
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH,
                        help=f'Max token length (default: {MAX_LENGTH})')
    parser.add_argument('--condition-max-length', type=int, default=1524,
                        help='Max partial SVG condition tokens for code-complement')
    
    # Image-specific options
    parser.add_argument('--replace-background', action='store_true', default=True,
                        help='Replace non-white background in images (default: True)')
    parser.add_argument('--no-replace-background', action='store_false', dest='replace_background',
                        help='Do not replace background')
    
    # Output options
    parser.add_argument('--save-svg', action='store_true', default=False,
                        help='Save SVG code files (default: True)')
    # parser.add_argument('--no-save-svg', action='store_false', dest='save_svg',
    #                     help='Do not save SVG code files')
    parser.add_argument('--save-png', action='store_true', default=False,
                        help='Also save rendered PNG images')
    parser.add_argument('--save-all-candidates', action='store_true', default=False,
                        help='Save all candidates (default: save only the best one)')
    
    # Tokenizer selection
    parser.add_argument('--use-train-tokenizer', action='store_true', default=False,
                        help='Use TrainAlignedSVGTokenizer (for models trained with configs/tokenization.yaml)')
    parser.add_argument('--tokenization-config', type=str, default='./configs/tokenization.yaml',
                        help='Path to tokenization.yaml (used with --use-train-tokenizer)')
    
    # Skeleton CoT
    parser.add_argument('--skeleton-cot', action='store_true', default=False,
                        help='Enable skeleton CoT mode: strip skeleton tokens and keep only replacement sections')
    
    # Debug
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    
    return parser.parse_args()


def download_model_weights(repo_id: str, filename: str = "pytorch_model.bin") -> str:
    """Download model weights from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )
        print(f"Successfully downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading from {repo_id}: {e}")
        raise


def is_local_path(path: str) -> bool:
    """Check if a path is a local filesystem path or a HuggingFace repo ID."""
    if os.path.exists(path):
        return True
    if path.startswith('/') or path.startswith('./') or path.startswith('../'):
        return True
    if os.path.sep in path and os.path.exists(os.path.dirname(path)):
        return True
    if len(path) > 1 and path[1] == ':':
        return True
    return False


def load_models(model_size: str, weight_path: str = None, model_path: str = None,
                use_train_tokenizer: bool = False, tokenization_config_path: str = None):
    """Load all models for a specific model size."""
    global tokenizer, processor, sketch_decoder, svg_tokenizer, svg_condition_encoder, current_model_size
    
    if weight_path is None:
        weight_path = get_config_value(model_size, 'huggingface', 'omnisvg_model')
    if model_path is None:
        model_path = get_config_value(model_size, 'huggingface', 'qwen_model')
    
    print(f"\n{'='*60}")
    print(f"Loading {model_size} Model")
    print(f"{'='*60}")
    print(f"Qwen model: {model_path}")
    print(f"OmniSVG weights: {weight_path}")
    print(f"Precision: {DTYPE}")
    
    # Load Qwen tokenizer and processor
    print("\n[1/3] Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="left",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        padding_side="left",
        trust_remote_code=True,
        use_fast=True
    )
    processor.tokenizer.padding_side = "left"
    print("Tokenizer and processor loaded successfully!")

    # Initialize sketch decoder with model_size
    print("\n[2/3] Initializing SketchDecoder...")
    infer_vocab_size = (config.get('models', {})
                        .get(model_size, {})
                        .get('model', {})
                        .get('vocab_size', 197000))
    sketch_decoder = SketchDecoder(
        config_path=CONFIG_PATH,
        model_path=model_path,
        model_size=model_size,
        pix_len=MAX_MAX_LENGTH,
        text_len=config.get('text', {}).get('max_length', 200),
        torch_dtype=DTYPE,
        vocab_size=infer_vocab_size,
    )
    
    # Load OmniSVG weights
    print("\n[3/3] Loading OmniSVG weights...")
    
    if is_local_path(weight_path):
        # Try multiple checkpoint formats in priority order
        candidates = [
            (os.path.join(weight_path, "pytorch_model.bin"), "bin"),
            (os.path.join(weight_path, "model.safetensors"), "safetensors"),
        ]
        # Also accept direct file paths
        if os.path.isfile(weight_path):
            ext = "safetensors" if weight_path.endswith('.safetensors') else "bin"
            candidates = [(weight_path, ext)]
        
        resolved_path, fmt = None, None
        for path, f in candidates:
            if os.path.exists(path):
                resolved_path, fmt = path, f
                break
        
        if resolved_path is None:
            raise FileNotFoundError(
                f"No checkpoint found at {weight_path}. "
                f"Looked for pytorch_model.bin and model.safetensors."
            )
        print(f"Loading weights from local path: {resolved_path} (format: {fmt})")
    else:
        print(f"Downloading weights from HuggingFace: {weight_path}")
        resolved_path = download_model_weights(weight_path, "pytorch_model.bin")
        fmt = "bin"
    
    if fmt == "safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(resolved_path)
    else:
        state_dict = torch.load(resolved_path, map_location='cpu')
    missing, unexpected = sketch_decoder.load_state_dict_flexible(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    print("OmniSVG weights loaded successfully!")
    
    sketch_decoder = sketch_decoder.eval()
    # SketchDecoder 未传 device_map 时 from_pretrained 默认在 CPU；显式搬到 GPU
    if torch.cuda.is_available():
        sketch_decoder = sketch_decoder.to(default_device)
        print(f"✓ Model on device: {default_device}")
    else:
        print("⚠ CUDA 不可用，使用 CPU 推理（较慢）。若本机有 GPU，请检查驱动/CUDA，以及 CUDA_VISIBLE_DEVICES 是否指向存在的卡。")
    
    # Initialize SVG tokenizer
    if use_train_tokenizer:
        from utils.config import TokenizationConfig
        tok_cfg_path = tokenization_config_path or './configs/tokenization.yaml'
        token_cfg = TokenizationConfig.from_yaml(tok_cfg_path, model_size)
        svg_tokenizer = TrainAlignedSVGTokenizer(token_cfg)
        svg_condition_encoder = TrainingSVGEncoder(token_cfg)
        print(f"Using TrainAlignedSVGTokenizer (config: {tok_cfg_path})")
    else:
        svg_tokenizer = SVGTokenizer(CONFIG_PATH, model_size=model_size)
        svg_condition_encoder = None
    
    current_model_size = model_size
    
    # Print device distribution info
    print(f"\n{get_model_devices_info()}")
    print(f"Input device will be: {get_model_input_device()}")
    
    print("\n" + "="*60)
    print(f"All {model_size} models loaded successfully!")
    print("="*60 + "\n")


def detect_text_subtype(text_prompt):
    """Auto-detect text prompt subtype"""
    text_lower = text_prompt.lower()
    
    icon_keywords = ['icon', 'logo', 'symbol', 'badge', 'button', 'emoji', 'glyph', 'simple', 
                     'arrow', 'triangle', 'circle', 'square', 'heart', 'star', 'checkmark']
    if any(kw in text_lower for kw in icon_keywords):
        return "icon"
    
    illustration_keywords = [
        'illustration', 'scene', 'person', 'people', 'character', 'man', 'woman', 'boy', 'girl',
        'avatar', 'portrait', 'face', 'head', 'body',
        'cat', 'dog', 'bird', 'animal', 'pet', 'fox', 'rabbit',
        'sitting', 'standing', 'walking', 'running', 'sleeping', 'holding', 'playing',
        'house', 'building', 'tree', 'garden', 'landscape', 'mountain', 'forest', 'city',
        'ocean', 'beach', 'sunset', 'sunrise', 'sky'
    ]
    
    match_count = sum(1 for kw in illustration_keywords if kw in text_lower)
    if match_count >= 1 or len(text_prompt) > 50:
        return "illustration"
    
    return "icon"


def detect_and_replace_background(image, threshold=None, edge_sample_ratio=None):
    """Detect if image has non-white background and optionally replace it."""
    if threshold is None:
        threshold = BACKGROUND_THRESHOLD
    if edge_sample_ratio is None:
        edge_sample_ratio = EDGE_SAMPLE_RATIO
    
    img_array = np.array(image)
    
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, image)
        return composite.convert('RGB'), True
    
    h, w = img_array.shape[:2]
    edge_pixels = []
    
    sample_count = max(MIN_EDGE_SAMPLES, int(min(h, w) * edge_sample_ratio))
    
    for i in range(0, w, max(1, w // sample_count)):
        edge_pixels.append(img_array[0, i])
        edge_pixels.append(img_array[h-1, i])
    
    for i in range(0, h, max(1, h // sample_count)):
        edge_pixels.append(img_array[i, 0])
        edge_pixels.append(img_array[i, w-1])
    
    edge_pixels = np.array(edge_pixels)
    
    if len(edge_pixels) > 0:
        mean_edge = edge_pixels.mean(axis=0)
        if np.all(mean_edge > threshold):
            return image, False
    
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        if img_array.shape[2] == 4:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = np.mean(img_array, axis=2)
        
        edge_colors = []
        for i in range(w):
            edge_colors.append(tuple(img_array[0, i, :3]))
            edge_colors.append(tuple(img_array[h-1, i, :3]))
        for i in range(h):
            edge_colors.append(tuple(img_array[i, 0, :3]))
            edge_colors.append(tuple(img_array[i, w-1, :3]))
        
        from collections import Counter
        color_counts = Counter(edge_colors)
        bg_color = color_counts.most_common(1)[0][0]
        
        color_diff = np.sqrt(np.sum((img_array[:, :, :3].astype(float) - np.array(bg_color)) ** 2, axis=2))
        bg_mask = color_diff < COLOR_SIMILARITY_THRESHOLD
        
        result = img_array.copy()
        if result.shape[2] == 4:
            result[bg_mask] = [255, 255, 255, 255]
        else:
            result[bg_mask] = [255, 255, 255]
        
        return Image.fromarray(result).convert('RGB'), True
    
    return image, False


def preprocess_image_for_svg(image, replace_background=True, target_size=None):
    """Preprocess image for SVG generation."""
    if target_size is None:
        target_size = TARGET_IMAGE_SIZE
    
    if isinstance(image, str):
        raw_img = Image.open(image)
    else:
        raw_img = image
    
    was_modified = False
    
    if raw_img.mode == 'RGBA':
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode == 'LA' or raw_img.mode == 'PA':
        raw_img = raw_img.convert('RGBA')
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode != 'RGB':
        img_with_bg = raw_img.convert('RGB')
    else:
        img_with_bg = raw_img
    
    if replace_background:
        img_with_bg, bg_replaced = detect_and_replace_background(img_with_bg)
        was_modified = was_modified or bg_replaced
    
    img_resized = img_with_bg.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img_resized, was_modified


def prepare_inputs(task_type, content):
    """Prepare model inputs"""
    if task_type == "text-to-svg":
        prompt_text = str(content).strip()
        
        instruction = f"""Generate an SVG illustration for: {prompt_text}
        
Requirements:
- Create complete SVG path commands
- Include proper coordinates and colors
- Maintain visual clarity and composition"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], padding=True, truncation=True, return_tensors="pt")
        
    else:  # image-to-svg
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                {"type": "image", "image": content},
            ]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, padding=True, truncation=True, return_tensors="pt")

    return inputs


def clean_generated_svg_token_ids(token_ids, skeleton_cot=False):
    """Remove sequence-control tokens before SVG token decoding.

    When *skeleton_cot* is True the generated sequence contains interleaved
    skeleton and replacement sections delimited by marker tokens.  Only the
    replacement sections are kept for SVG decoding; skeleton sections are
    discarded (but logged when verbose).
    """
    ids = list(token_ids)

    if ids and ids[0] == BOS_TOKEN_ID:
        ids = ids[1:]

    if EOS_TOKEN_ID in ids:
        ids = ids[:ids.index(EOS_TOKEN_ID)]

    ids = [tok for tok in ids if tok != PAD_TOKEN_ID]

    if skeleton_cot:
        ids = extract_replacement_tokens(ids)

    return ids


def extract_replacement_tokens(ids):
    """Extract only replacement SVG tokens from a skeleton-CoT token sequence.

    The expected layout is:
        [SKEL_S] ... [SKEL_E] [REPL_S] ... [REPL_E] (repeated)

    Returns the concatenation of all REPL sections (marker tokens removed).
    If no markers are found the original ids are returned unchanged so that
    the function degrades gracefully on non-CoT outputs.
    """
    has_markers = any(tok in SKELETON_COT_MARKERS for tok in ids)
    if not has_markers:
        return ids

    replacement_ids = []
    in_replacement = False
    for tok in ids:
        if tok == REPLACEMENT_START_TOKEN_ID:
            in_replacement = True
            continue
        if tok == REPLACEMENT_END_TOKEN_ID:
            in_replacement = False
            continue
        if tok in SKELETON_COT_MARKERS:
            in_replacement = False
            continue
        if in_replacement:
            replacement_ids.append(tok)

    return replacement_ids


def extract_skeleton_and_replacement_sections(ids):
    """Split a skeleton-CoT token sequence into (skeleton_sections, replacement_sections).

    Each section is a list of SVG token ids (markers removed).  Useful for
    visualization / debugging of the CoT reasoning.
    """
    skeleton_sections = []
    replacement_sections = []

    current_section = []
    current_type = None  # 'skel' or 'repl'

    for tok in ids:
        if tok == SKELETON_START_TOKEN_ID:
            current_section = []
            current_type = 'skel'
            continue
        if tok == SKELETON_END_TOKEN_ID:
            if current_type == 'skel' and current_section:
                skeleton_sections.append(current_section)
            current_section = []
            current_type = None
            continue
        if tok == REPLACEMENT_START_TOKEN_ID:
            current_section = []
            current_type = 'repl'
            continue
        if tok == REPLACEMENT_END_TOKEN_ID:
            if current_type == 'repl' and current_section:
                replacement_sections.append(current_section)
            current_section = []
            current_type = None
            continue
        if current_type is not None:
            current_section.append(tok)

    return skeleton_sections, replacement_sections


# ---------------------------------------------------------------------------
# Skeleton-CoT visualization helpers
# ---------------------------------------------------------------------------

def decode_token_ids_to_svg(token_ids, override_fill=None):
    """Decode a flat list of SVG token IDs into an SVG string and rendered PIL Image.

    Args:
        token_ids: list[int] of SVG token IDs (without BOS/EOS wrapper).
        override_fill: if set, replace every path ``fill`` with this CSS colour.

    Returns:
        (svg_str, pil_image) or (None, None) on failure.
    """
    if not token_ids:
        return None, None
    try:
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device='cpu')
        wrapped = torch.cat([
            torch.full((1, 1), BOS_TOKEN_ID, device='cpu'),
            ids_tensor,
            torch.full((1, 1), EOS_TOKEN_ID, device='cpu'),
        ], dim=1)

        generated_xy = svg_tokenizer.process_generated_tokens(wrapped)
        if len(generated_xy) == 0:
            return None, None

        svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
        if not svg_tensors or not svg_tensors[0]:
            return None, None

        num_paths = len(svg_tensors[0])
        while len(color_tensors) < num_paths:
            color_tensors.append(BLACK_COLOR_TOKEN)

        svg_obj = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
        svg_str = svg_obj.to_str()

        if 'width=' not in svg_str:
            svg_str = svg_str.replace(
                '<svg',
                f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
                1,
            )

        if override_fill:
            svg_str = re.sub(r'\s*fill-opacity="[^"]*"', '', svg_str)
            svg_str = re.sub(r'fill="[^"]*"', f'fill="{override_fill}"', svg_str)

        rendered = render_svg_to_image(svg_str, size=RENDER_SIZE)
        return svg_str, rendered
    except Exception:
        return None, None


def _make_overlay_svg(skeleton_svg, replacement_svg,
                      skel_color="#E74C3C", skel_opacity="0.7"):
    """Combine skeleton and replacement paths into one overlay SVG.

    Replacement paths are drawn first (bottom layer), then skeleton paths
    are drawn on top in *skel_color* so they remain clearly visible.
    """
    skel_paths = extract_svg_paths(skeleton_svg) if skeleton_svg else []
    repl_paths = extract_svg_paths(replacement_svg) if replacement_svg else []

    colored_skel = []
    for p in skel_paths:
        p = re.sub(r'\s*fill-opacity="[^"]*"', '', p)
        p = re.sub(r'fill="[^"]*"',
                   f'fill="{skel_color}" fill-opacity="{skel_opacity}"', p)
        colored_skel.append(p)

    all_paths = repl_paths + colored_skel
    if not all_paths:
        return None

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 200 200" width="{RENDER_SIZE}" height="{RENDER_SIZE}">\n'
        '  ' + '\n  '.join(all_paths) + '\n'
        '</svg>'
    )


def create_cot_visualization(cot_sections, partial_svg_path=None,
                             combined_svg=None, cell_size=384):
    """Build a grid PNG comparing skeleton vs replacement for each CoT section.

    Layout (per row = one section pair):
        | Skeleton (red) | Replacement | Overlay |

    An extra final row shows partial SVG, all-sections overlay, and the
    combined result if *partial_svg_path* / *combined_svg* are provided.

    Returns a PIL ``Image`` or ``None`` on failure.
    """
    from PIL import ImageDraw

    skel_secs = cot_sections.get('skeleton', [])
    repl_secs = cot_sections.get('replacement', [])
    num_pairs = max(len(skel_secs), len(repl_secs))
    if num_pairs == 0:
        return None

    skel_images, skel_svgs = [], []
    for tokens in skel_secs:
        s, img = decode_token_ids_to_svg(tokens, override_fill="#E74C3C")
        skel_svgs.append(s)
        skel_images.append(img)

    repl_images, repl_svgs = [], []
    for tokens in repl_secs:
        s, img = decode_token_ids_to_svg(tokens)
        repl_svgs.append(s)
        repl_images.append(img)

    overlay_images = []
    for si in range(num_pairs):
        s_svg = skel_svgs[si] if si < len(skel_svgs) else None
        r_svg = repl_svgs[si] if si < len(repl_svgs) else None
        ov_svg = _make_overlay_svg(s_svg, r_svg)
        overlay_images.append(
            render_svg_to_image(ov_svg, size=cell_size) if ov_svg else None
        )

    # Optional summary row images
    partial_img = None
    if partial_svg_path and os.path.exists(partial_svg_path):
        with open(partial_svg_path, 'r', encoding='utf-8') as f:
            partial_img = render_svg_to_image(f.read(), size=cell_size)

    combined_img = None
    if combined_svg:
        combined_img = render_svg_to_image(combined_svg, size=cell_size)

    has_summary = partial_img or combined_img
    total_rows = num_pairs + (1 if has_summary else 0)

    ncols = 3
    pad = 6
    label_h = 28
    row_label_w = 36
    total_w = row_label_w + ncols * cell_size + (ncols + 1) * pad
    total_h = total_rows * cell_size + (total_rows + 1) * pad + label_h

    viz = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(viz)

    headers = ["Skeleton", "Replacement", "Overlay"]
    for ci, hdr in enumerate(headers):
        x = row_label_w + pad + ci * (cell_size + pad) + cell_size // 2
        draw.text((x, 6), hdr, fill='black', anchor='mt')

    def _paste(img, col, row):
        x = row_label_w + pad + col * (cell_size + pad)
        y = label_h + pad + row * (cell_size + pad)
        if img is not None:
            viz.paste(img.resize((cell_size, cell_size), Image.Resampling.LANCZOS),
                      (x, y))
        else:
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1],
                           outline='#CCCCCC')

    for row in range(num_pairs):
        ry = label_h + pad + row * (cell_size + pad) + cell_size // 2
        draw.text((4, ry), f"S{row+1}", fill='#555', anchor='lm')

        _paste(skel_images[row] if row < len(skel_images) else None, 0, row)
        _paste(repl_images[row] if row < len(repl_images) else None, 1, row)
        _paste(overlay_images[row] if row < len(overlay_images) else None, 2, row)

    if has_summary:
        srow = num_pairs
        ry = label_h + pad + srow * (cell_size + pad) + cell_size // 2
        draw.text((4, ry), "All", fill='#555', anchor='lm')

        _paste(partial_img, 0, srow)
        _paste(combined_img, 1, srow)

        all_skel_tokens = [t for sec in skel_secs for t in sec]
        all_repl_tokens = [t for sec in repl_secs for t in sec]
        all_s_svg, _ = decode_token_ids_to_svg(all_skel_tokens, override_fill="#E74C3C")
        all_r_svg, _ = decode_token_ids_to_svg(all_repl_tokens)
        full_ov = _make_overlay_svg(all_s_svg, all_r_svg)
        full_ov_img = render_svg_to_image(full_ov, size=cell_size) if full_ov else None
        _paste(full_ov_img, 2, srow)

    return viz


def tokenize_partial_svg_file(svg_path, condition_max_length=None):
    """Tokenize a partial SVG file with the same SVG tokenizer used in training."""
    if svg_condition_encoder is None:
        raise RuntimeError(
            "code-complement requires --use-train-tokenizer so partial SVGs "
            "can be encoded with configs/tokenization.yaml"
        )

    svg = DeepSVG.load_svg(svg_path)
    svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
    tokens = svg_condition_encoder.tokenize_svg_tensors(svg_tensors, color_tensors)
    tokens = svg_condition_encoder.add_special_tokens(tokens).tolist()

    if condition_max_length and len(tokens) > condition_max_length:
        # Match training: keep the tail so the latest partial paths remain visible.
        tokens = tokens[-condition_max_length:]

    return tokens


def prepare_code_complement_inputs(svg_path, image, condition_max_length=None):
    """Prepare Qwen image/text inputs plus partial SVG condition tokens."""
    partial_tokens = tokenize_partial_svg_file(svg_path, condition_max_length)

    base_name = os.path.splitext(os.path.basename(svg_path))[0]
    codepoint, _, font_name = base_name.partition("_")
    try:
        char = chr(int(codepoint, 16))
    except ValueError:
        char = ""
    char_label = f"{char} (U+{codepoint})" if char else f"U+{codepoint}"

    instruction = (
        "Complete the missing SVG path code for this Chinese glyph.\n"
        f"Character: {char_label}\n"
        "The partial SVG S_d is provided after this instruction as SVG tokens. "
        "The attached PNG image I_d shows the corresponding incomplete glyph. "
        "Output only the SVG path code fragments that should be appended to S_d."
    )

    messages = [
        {"role": "system", "content": "You are an expert SVG code generator."},
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image},
        ]}
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )

    condition_ids = torch.tensor([partial_tokens], dtype=inputs["input_ids"].dtype)
    condition_mask = torch.ones_like(condition_ids)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], condition_ids], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], condition_mask], dim=1)

    return inputs, {
        "partial_token_count": len(partial_tokens),
        "char_label": char_label,
        "uid": base_name,
    }


def find_code_complement_samples(input_path):
    """Find partial SVG/PNG pairs for code-complement inference."""
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        svg_files = [input_path] if input_path.lower().endswith(".svg") else []
        root = os.path.dirname(input_path)
    else:
        root = input_path
        svg_dir = os.path.join(root, "svg")
        search_dir = svg_dir if os.path.isdir(svg_dir) else root
        svg_files = sorted(glob.glob(os.path.join(search_dir, "*.svg")))

    samples = []
    for svg_path in svg_files:
        stem = os.path.splitext(os.path.basename(svg_path))[0]
        candidate_dirs = [
            os.path.join(root, "png"),
            os.path.join(root, "images"),
            os.path.dirname(svg_path),
        ]
        image_path = None
        for folder in candidate_dirs:
            for ext in SUPPORTED_FORMATS:
                candidate = os.path.join(folder, f"{stem}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
                candidate = os.path.join(folder, f"{stem}{ext.upper()}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path:
                break

        samples.append({
            "svg_path": svg_path,
            "image_path": image_path,
            "stem": stem,
        })

    return samples


def extract_svg_paths(svg_str):
    """Extract path elements from an SVG string for appending."""
    path_tags = re.findall(
        r"<path\b[^>]*(?:/>\s*|>\s*</path\s*>)",
        svg_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = []
    for tag in path_tags:
        tag = tag.strip()
        if tag.lower().endswith("</path>"):
            tag = re.sub(r">\s*</path\s*>$", "/>", tag, flags=re.IGNORECASE)
        normalized.append(tag)
    return normalized


def combine_partial_and_completion(partial_svg, completion_svg):
    """Append generated path elements to the partial SVG document."""
    generated_paths = extract_svg_paths(completion_svg)
    if not generated_paths:
        return partial_svg

    insertion = "\n  " + "\n  ".join(generated_paths) + "\n"
    if re.search(r"</svg\s*>", partial_svg, flags=re.IGNORECASE):
        return re.sub(r"</svg\s*>", insertion + "</svg>", partial_svg, count=1, flags=re.IGNORECASE)

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">\n'
        f"{partial_svg}\n"
        f"{insertion}"
        "</svg>"
    )


def load_or_render_partial_image(svg_path, image_path, replace_background=True):
    """Load the incomplete PNG, or render the partial SVG as a fallback."""
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        processed, _ = preprocess_image_for_svg(
            image,
            replace_background=replace_background,
            target_size=TARGET_IMAGE_SIZE,
        )
        return processed, False

    with open(svg_path, "r", encoding="utf-8") as f:
        partial_svg = f.read()
    rendered = render_svg_to_image(partial_svg, size=TARGET_IMAGE_SIZE)
    if rendered is None:
        rendered = Image.new("RGB", (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), "white")
    return rendered.convert("RGB"), True


def save_code_complement_results(candidates, partial_svg_path, output_dir, base_name,
                                 save_svg=True, save_png=False, save_all=False):
    """Save generated fragments and merged partial+completion SVGs."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    if not candidates:
        return saved_files

    with open(partial_svg_path, "r", encoding="utf-8") as f:
        partial_svg = f.read()

    items = candidates if save_all else candidates[:1]
    for i, cand in enumerate(items):
        suffix = f"_candidate_{i+1}" if save_all else ""
        fragment_svg = cand["svg"]
        combined_svg = combine_partial_and_completion(partial_svg, fragment_svg)

        if save_svg:
            fragment_path = os.path.join(output_dir, f"{base_name}{suffix}_completion.svg")
            with open(fragment_path, "w", encoding="utf-8") as f:
                f.write(fragment_svg)
            saved_files.append(fragment_path)

            combined_path = os.path.join(output_dir, f"{base_name}{suffix}_combined.svg")
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write(combined_svg)
            saved_files.append(combined_path)

        if save_png:
            combined_img = render_svg_to_image(combined_svg, size=RENDER_SIZE)
            if combined_img is not None:
                png_path = os.path.join(output_dir, f"{base_name}{suffix}_combined.png")
                combined_img.save(png_path)
                saved_files.append(png_path)

        if cand.get('cot_sections'):
            viz_img = create_cot_visualization(
                cand['cot_sections'],
                partial_svg_path=partial_svg_path,
                combined_svg=combined_svg,
            )
            if viz_img is not None:
                viz_path = os.path.join(output_dir,
                                        f"{base_name}{suffix}_cot_viz.png")
                viz_img.save(viz_path)
                saved_files.append(viz_path)

    return saved_files


def render_svg_to_image(svg_str, size=None):
    """Render SVG to high-quality PIL Image"""
    if size is None:
        size = RENDER_SIZE
    
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])
        return bg
    except Exception as e:
        print(f"Render error: {e}")
        return None


def is_valid_candidate(svg_str, img, subtype="illustration"):
    """Check candidate validity"""
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False, "too_short"
    
    if '<svg' not in svg_str:
        return False, "no_svg_tag"
    
    if img is None:
        return False, "render_failed"
    
    img_array = np.array(img)
    mean_val = img_array.mean()
    
    threshold = EMPTY_THRESHOLD_ILLUSTRATION if subtype == "illustration" else EMPTY_THRESHOLD_ICON
    
    if mean_val > threshold:
        return False, "empty_image"
    
    return True, "ok"


def generate_candidates(inputs, task_type, subtype, temperature, top_p, top_k, repetition_penalty, 
                       max_length, num_samples, verbose=False, skeleton_cot=False):
    """Generate candidate SVGs with full parameter control"""
    
    # Get the correct device from the model's embedding layer
    input_device = get_model_input_device()
    
    if verbose:
        print(f"  Using input device: {input_device}")
    
    input_ids = inputs['input_ids'].to(input_device)
    attention_mask = inputs['attention_mask'].to(input_device)
    
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if 'pixel_values' in inputs:
        model_inputs["pixel_values"] = inputs['pixel_values'].to(input_device, dtype=DTYPE)
    
    if 'image_grid_thw' in inputs:
        model_inputs["image_grid_thw"] = inputs['image_grid_thw'].to(input_device)
    
    all_candidates = []
    
    gen_cfg = {
        'do_sample': True,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': int(top_k),
        'repetition_penalty': repetition_penalty,
        'early_stopping': True,
        'no_repeat_ngram_size': 0,
        'eos_token_id': EOS_TOKEN_ID,
        'pad_token_id': PAD_TOKEN_ID,
        'bos_token_id': BOS_TOKEN_ID,
    }
    
    actual_samples = num_samples + EXTRA_CANDIDATES_BUFFER
    
    try:
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                **model_inputs,
                max_new_tokens=max_length,
                num_return_sequences=actual_samples,
                use_cache=True,
                **gen_cfg
            )
            
            input_len = input_ids.shape[1]
            generated_ids_batch = results[:, input_len:]
        
        for i in range(min(actual_samples, generated_ids_batch.shape[0])):
            try:
                current_ids = generated_ids_batch[i:i+1]
                raw_ids = current_ids[0].detach().cpu().tolist()
                cleaned_ids = clean_generated_svg_token_ids(raw_ids, skeleton_cot=skeleton_cot)

                cot_skel_secs, cot_repl_secs = None, None
                if skeleton_cot:
                    _raw_clean = [t for t in raw_ids if t != PAD_TOKEN_ID]
                    if _raw_clean and _raw_clean[0] == BOS_TOKEN_ID:
                        _raw_clean = _raw_clean[1:]
                    if EOS_TOKEN_ID in _raw_clean:
                        _raw_clean = _raw_clean[:_raw_clean.index(EOS_TOKEN_ID)]
                    cot_skel_secs, cot_repl_secs = extract_skeleton_and_replacement_sections(_raw_clean)

                if verbose:
                    print(f"  Candidate {i} generated token count: {len(raw_ids)}")
                    print(f"  Candidate {i} first 30 ids: {raw_ids[:30]}")
                    print(f"  Candidate {i} last 30 ids: {raw_ids[-30:]}")
                    special_tokens = {
                        "BOS": BOS_TOKEN_ID,
                        "EOS": EOS_TOKEN_ID,
                        "PAD": PAD_TOKEN_ID,
                    }
                    for name, token_id in special_tokens.items():
                        positions = [idx for idx, tok in enumerate(raw_ids) if tok == token_id]
                        if positions:
                            print(
                                f"  Candidate {i} {name} token {token_id} "
                                f"count={len(positions)} positions={positions[:20]}"
                            )
                    print(f"  Candidate {i} cleaned token count: {len(cleaned_ids)}")
                    print(f"  Candidate {i} cleaned first 30 ids: {cleaned_ids[:30]}")
                    print(f"  Candidate {i} cleaned last 30 ids: {cleaned_ids[-30:]}")
                    if cot_skel_secs is not None:
                        print(f"  Candidate {i} skeleton sections: {len(cot_skel_secs)}, "
                              f"replacement sections: {len(cot_repl_secs)}")
                        for si, ss in enumerate(cot_skel_secs):
                            print(f"    skel[{si}] len={len(ss)} first10={ss[:10]}")
                        for ri, rs in enumerate(cot_repl_secs):
                            print(f"    repl[{ri}] len={len(rs)} first10={rs[:10]}")
                if not cleaned_ids:
                    if verbose:
                        print(f"  Candidate {i} skipped: no SVG tokens after cleanup")
                    continue
                
                # Move to CPU for post-processing to avoid device issues
                current_ids_cpu = torch.tensor([cleaned_ids], dtype=torch.long, device='cpu')
                
                fake_wrapper = torch.cat([
                    torch.full((1, 1), BOS_TOKEN_ID, device='cpu'),
                    current_ids_cpu,
                    torch.full((1, 1), EOS_TOKEN_ID, device='cpu')
                ], dim=1)

                generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
                if len(generated_xy) == 0: #(2039, 2)
                    continue

                svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
                if not svg_tensors or not svg_tensors[0]:
                    continue

                num_paths = len(svg_tensors[0])
                while len(color_tensors) < num_paths:
                    color_tensors.append(BLACK_COLOR_TOKEN)
                
                svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
                svg_str = svg.to_str()
                
                if 'width=' not in svg_str:
                    svg_str = svg_str.replace('<svg', f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"', 1)
                
                png_image = render_svg_to_image(svg_str, size=RENDER_SIZE)
                
                is_valid, reason = is_valid_candidate(svg_str, png_image, subtype)
                if is_valid:
                    cand_dict = {
                        'svg': svg_str,
                        'img': png_image,
                        'path_count': num_paths,
                        'index': len(all_candidates) + 1,
                    }
                    if cot_skel_secs is not None:
                        cand_dict['cot_sections'] = {
                            'skeleton': cot_skel_secs,
                            'replacement': cot_repl_secs,
                        }
                    all_candidates.append(cand_dict)
                    
                    if verbose:
                        print(f"  Found valid candidate {len(all_candidates)} with {num_paths} paths")
                    
                    if len(all_candidates) >= num_samples:
                        break
                elif verbose:
                    print(f"  Candidate {i} invalid: {reason}")
                        
            except Exception as e:
                if verbose:
                    print(f"  Candidate {i} error: {e}")
                continue

    except Exception as e:
        print(f"Generation Error: {e}")
        import traceback
        traceback.print_exc()
    
    return all_candidates


def save_results(candidates, output_dir, base_name, save_svg=True, save_png=False, save_all=False):
    """Save generated SVG(s) and optionally PNG(s)"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    if not candidates:
        return saved_files
    
    if save_all:
        for i, cand in enumerate(candidates):
            if save_svg:
                svg_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.svg")
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(cand['svg'])
                saved_files.append(svg_path)
            
            if save_png and cand['img'] is not None:
                png_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.png")
                cand['img'].save(png_path)
                saved_files.append(png_path)
    else:
        # Save only the best (first valid) candidate
        best = candidates[0]
        if save_svg:
            svg_path = os.path.join(output_dir, f"{base_name}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(best['svg'])
            saved_files.append(svg_path)
        
        if save_png and best['img'] is not None:
            png_path = os.path.join(output_dir, f"{base_name}.png")
            best['img'].save(png_path)
            saved_files.append(png_path)
    
    return saved_files


def process_text_to_svg(args):
    """Process text-to-svg task"""
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Read prompts from text file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prompts = [line.strip() for line in lines if line.strip()]
    
    if not prompts:
        print("Error: No prompts found in input file")
        return
    
    print(f"\nFound {len(prompts)} prompts to process")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each prompt
    total_success = 0
    total_failed = 0
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Processing: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        start_time = time.time()
        
        # Detect subtype
        subtype = detect_text_subtype(prompt)
        task_key = f"text-to-svg-{subtype}"
        
        # Get default parameters based on task
        temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.5)
        top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
        top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
        rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
        
        if args.verbose:
            print(f"  Subtype: {subtype}")
            print(f"  Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
        
        # Prepare inputs
        inputs = prepare_inputs("text-to-svg", prompt)
        
        # Generate candidates
        candidates = generate_candidates(
            inputs, "text-to-svg", subtype,
            temperature, top_p, top_k, rep_penalty,
            args.max_length, args.num_candidates,
            verbose=args.verbose
        )
        
        elapsed = time.time() - start_time
        
        if candidates:
            # Create safe filename from prompt
            safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt[:50]).strip()
            safe_name = f"{idx+1:04d}_{safe_name}"
            
            saved = save_results(candidates, args.output, safe_name, 
                               save_svg=args.save_svg, save_png=args.save_png, save_all=args.save_all_candidates)
            
            print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
            print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
            total_success += 1
        else:
            print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
            total_failed += 1
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Text-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(prompts)}")
    print(f"  Failed: {total_failed}/{len(prompts)}")
    print(f"  Output: {args.output}")
    print("="*60)


def process_image_to_svg(args):
    """Process image-to-svg task"""
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        return
    
    # Find all image files
    if os.path.isfile(input_path):
        image_files = [input_path]
    else:
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"Error: No image files found in {input_path}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get default parameters
    task_key = "image-to-svg"
    temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.3)
    top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
    top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
    
    if args.verbose:
        print(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
    
    # Process each image
    total_success = 0
    total_failed = 0
    
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_name}")
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(img_path)
            img_processed, was_modified = preprocess_image_for_svg(
                image, 
                replace_background=args.replace_background,
                target_size=TARGET_IMAGE_SIZE
            )
            
            if args.verbose and was_modified:
                print("  Background processed/replaced")
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img_processed.save(tmp_file.name, format='PNG', quality=100)
                tmp_path = tmp_file.name
            
            try:
                # Prepare inputs
                inputs = prepare_inputs("image-to-svg", tmp_path)
                
                # Generate candidates
                candidates = generate_candidates(
                    inputs, "image-to-svg", "image",
                    temperature, top_p, top_k, rep_penalty,
                    args.max_length, args.num_candidates,
                    verbose=args.verbose
                )
                
                elapsed = time.time() - start_time
                
                if candidates:
                    # Use original filename (without extension) as base name
                    base_name = os.path.splitext(img_name)[0]
                    
                    saved = save_results(candidates, args.output, base_name, 
                                       save_png=args.save_png, save_all=args.save_all_candidates)
                    
                    print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
                    print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
                    total_success += 1
                else:
                    print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
                    total_failed += 1
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            total_failed += 1
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Image-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(image_files)}")
    print(f"  Failed: {total_failed}/{len(image_files)}")
    print(f"  Output: {args.output}")
    print("="*60)


def process_code_complement(args):
    """Process SVG code-complement task."""
    input_path = args.input

    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        return

    if svg_condition_encoder is None:
        print("Error: code-complement requires --use-train-tokenizer")
        print("       This uses configs/tokenization.yaml to encode partial SVG inputs.")
        return

    samples = find_code_complement_samples(input_path)
    if not samples:
        print(f"Error: No partial SVG files found in {input_path}")
        return

    print(f"\nFound {len(samples)} code-complement samples to process")
    print("="*60)

    os.makedirs(args.output, exist_ok=True)

    task_key = "code-complement"
    temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.3)
    top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
    top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)

    use_skeleton_cot = getattr(args, 'skeleton_cot', False)

    if args.verbose:
        print(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
        print(f"Condition max length: {args.condition_max_length}")
        print(f"Skeleton CoT: {use_skeleton_cot}")

    total_success = 0
    total_failed = 0

    for idx, sample in enumerate(samples):
        svg_path = sample["svg_path"]
        image_path = sample["image_path"]
        base_name = sample["stem"]
        print(f"\n[{idx+1}/{len(samples)}] Processing: {base_name}")

        start_time = time.time()

        try:
            image, rendered_fallback = load_or_render_partial_image(
                svg_path,
                image_path,
                replace_background=args.replace_background,
            )
            if args.verbose:
                if image_path:
                    print(f"  Image: {image_path}")
                if rendered_fallback:
                    print("  PNG not found; rendered partial SVG as image fallback")

            inputs, meta = prepare_code_complement_inputs(
                svg_path,
                image,
                condition_max_length=args.condition_max_length,
            )

            if args.verbose:
                print(f"  Character: {meta['char_label']}")
                print(f"  Partial SVG condition tokens: {meta['partial_token_count']}")

            candidates = generate_candidates(
                inputs, "code-complement", "image",
                temperature, top_p, top_k, rep_penalty,
                args.max_length, args.num_candidates,
                verbose=args.verbose,
                skeleton_cot=use_skeleton_cot,
            )

            elapsed = time.time() - start_time

            if candidates:
                saved = save_code_complement_results(
                    candidates,
                    svg_path,
                    args.output,
                    base_name,
                    save_svg=True,
                    save_png=args.save_png,
                    save_all=args.save_all_candidates,
                )
                print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
                print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
                total_success += 1
            else:
                print(f"  ✗ Failed to generate valid SVG completion ({elapsed:.2f}s)")
                total_failed += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            total_failed += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("Code-Complement Complete!")
    print(f"  Success: {total_success}/{len(samples)}")
    print(f"  Failed: {total_failed}/{len(samples)}")
    print(f"  Output: {args.output}")
    print("="*60)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    print("="*60)
    print("OmniSVG Inference Script")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Model Size: {args.model_size}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Default Device: {default_device}")
    print(f"Precision: {DTYPE}")
    print(f"Num Candidates: {args.num_candidates}")
    print(f"Max Length: {args.max_length}")
    print("="*60)
    
    # Load models
    load_models(
        args.model_size, args.weight_path, args.model_path,
        use_train_tokenizer=args.use_train_tokenizer,
        tokenization_config_path=args.tokenization_config,
    )
    
    # Process based on task type
    if args.task == "text-to-svg":
        process_text_to_svg(args)
    elif args.task == "image-to-svg":
        process_image_to_svg(args)
    else:  # code-complement
        process_code_complement(args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
