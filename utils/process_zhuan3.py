#!/usr/bin/env python3
"""
Process my_zhuan3 dataset:
1. Preprocess SVG files: V/H→L, Q→C, scale to 200x200, add viewBox
2. Rename files to remove Chinese characters
3. Generate train/val CSV (9:1 split) with pix_len computation
"""

import os
import re
import csv
import sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "/data/phd23_weiguang_zhang/works/svg/my_zhuan4"
SVG_DIR = os.path.join(DATA_DIR, "svg")
PNG_DIR = os.path.join(DATA_DIR, "png")
MODEL_SIZE = "4B"
CONFIG_PATH = "./configs/tokenization.yaml"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

TOKEN_RE = re.compile(r'[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def remove_chinese_from_name(name):
    """Remove Chinese characters (and the preceding underscore) from filename stem.
    Example: 4E00_一_FZLiYBZSFU_min -> 4E00_FZLiYBZSFU_min
    """
    return re.sub(r'_[^\x00-\x7F]+', '', name, count=1)


def has_chinese(name):
    return bool(re.search(r'[^\x00-\x7F]', name))


def tokenize_path_data(d):
    return TOKEN_RE.findall(d)


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_path_data(d, scale_x, scale_y):
    """Parse SVG path, convert V/H→L and Q→C, scale coordinates to 200x200."""
    tokens = tokenize_path_data(d)
    result = []
    cur_x, cur_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    i = 0
    cmd = None

    while i < len(tokens):
        t = tokens[i]

        if not is_num(t):
            cmd = t
            i += 1
            if cmd in ('Z', 'z'):
                result.append('Z')
                cur_x, cur_y = start_x, start_y
                cmd = None
            continue

        if cmd is None:
            i += 1
            continue

        if cmd == 'M':
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            result.append(f'M{round(x * scale_x)} {round(y * scale_y)}')
            cur_x, cur_y = x, y
            start_x, start_y = x, y
            cmd = 'L'

        elif cmd == 'L':
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            result.append(f'L{round(x * scale_x)} {round(y * scale_y)}')
            cur_x, cur_y = x, y

        elif cmd == 'H':
            x = float(tokens[i])
            i += 1
            result.append(f'L{round(x * scale_x)} {round(cur_y * scale_y)}')
            cur_x = x

        elif cmd == 'V':
            y = float(tokens[i])
            i += 1
            result.append(f'L{round(cur_x * scale_x)} {round(y * scale_y)}')
            cur_y = y

        elif cmd == 'C':
            x1 = float(tokens[i]);     y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2]); y2 = float(tokens[i + 3])
            x  = float(tokens[i + 4]); y  = float(tokens[i + 5])
            i += 6
            result.append(
                f'C{round(x1 * scale_x)} {round(y1 * scale_y)} '
                f'{round(x2 * scale_x)} {round(y2 * scale_y)} '
                f'{round(x * scale_x)} {round(y * scale_y)}'
            )
            cur_x, cur_y = x, y

        elif cmd == 'Q':
            cx = float(tokens[i]);     cy = float(tokens[i + 1])
            x  = float(tokens[i + 2]); y  = float(tokens[i + 3])
            i += 4
            cp1x = cur_x + 2 / 3 * (cx - cur_x)
            cp1y = cur_y + 2 / 3 * (cy - cur_y)
            cp2x = x + 2 / 3 * (cx - x)
            cp2y = y + 2 / 3 * (cy - y)
            result.append(
                f'C{round(cp1x * scale_x)} {round(cp1y * scale_y)} '
                f'{round(cp2x * scale_x)} {round(cp2y * scale_y)} '
                f'{round(x * scale_x)} {round(y * scale_y)}'
            )
            cur_x, cur_y = x, y

        else:
            print(f"  Warning: unhandled command '{cmd}', skipping token '{t}'")
            i += 1

    return ' '.join(result)


def process_svg_content(content):
    """Process a single SVG string: scale, convert commands, add viewBox."""
    width_m = re.search(r'width="(\d+)"', content)
    height_m = re.search(r'height="(\d+)"', content)
    if not width_m or not height_m:
        return None

    orig_w = int(width_m.group(1))
    orig_h = int(height_m.group(1))
    if orig_w == 0 or orig_h == 0:
        return None

    sx = 200.0 / orig_w
    sy = 200.0 / orig_h

    def _replace_d(m):
        return f'd="{convert_path_data(m.group(1), sx, sy)}"'

    content = re.sub(r'd="([^"]*)"', _replace_d, content)

    content = re.sub(r'width="\d+"', 'width="200"', content)
    content = re.sub(r'height="\d+"', 'height="200"', content)

    if 'viewBox' not in content:
        content = content.replace('<svg ', '<svg viewBox="0.0 0.0 200.0 200.0" ', 1)

    return content


# ── Step 1: Preprocess SVG files and rename ──────────────────────────────────

def step1_preprocess_and_rename():
    print("=" * 60)
    print("Step 1: Preprocess SVGs + rename (remove Chinese)")
    print("=" * 60)

    svg_files = sorted(Path(SVG_DIR).glob("*.svg"))
    print(f"Found {len(svg_files)} SVG files")

    success, fail, renamed = 0, 0, 0

    for svg_path in tqdm(svg_files, desc="Processing SVGs"):
        try:
            content = svg_path.read_text(encoding='utf-8')
            new_content = process_svg_content(content)
            if new_content is None:
                print(f"  Skip (no width/height): {svg_path.name}")
                fail += 1
                continue

            stem = svg_path.stem
            new_stem = remove_chinese_from_name(stem)
            need_rename = (new_stem != stem)

            new_svg_path = svg_path.parent / f"{new_stem}.svg"
            new_svg_path.write_text(new_content, encoding='utf-8')

            if need_rename and svg_path.exists() and svg_path != new_svg_path:
                svg_path.unlink()

            png_path = Path(PNG_DIR) / f"{stem}.png"
            if need_rename and png_path.exists():
                new_png_path = png_path.parent / f"{new_stem}.png"
                shutil.move(str(png_path), str(new_png_path))
                renamed += 1

            success += 1
        except Exception as e:
            print(f"  Error processing {svg_path.name}: {e}")
            fail += 1

    print(f"\nStep 1 done: {success} OK, {fail} failed, {renamed} renamed")
    return success > 0


# ── Step 2: Generate train/val CSV with pix_len ─────────────────────────────

def step2_generate_csv():
    print("\n" + "=" * 60)
    print("Step 2: Generate train/val CSV (9:1) with pix_len")
    print("=" * 60)

    sys.path.insert(0, '.')
    from utils.config import TokenizationConfig
    from utils.dataset import SVGTokenizer
    from deepsvg.svglib.svg import SVG

    cfg = TokenizationConfig.from_yaml(CONFIG_PATH, MODEL_SIZE)
    tokenizer = SVGTokenizer(cfg)

    svg_files = sorted(Path(SVG_DIR).glob("*.svg"))
    file_ids = [f.stem for f in svg_files]
    total = len(file_ids)
    print(f"Total files: {total}")

    random.seed(RANDOM_SEED)
    random.shuffle(file_ids)

    train_count = int(total * TRAIN_RATIO)
    train_ids = file_ids[:train_count]
    val_ids = file_ids[train_count:]
    print(f"Train: {train_count}, Val: {total - train_count}")

    def compute_pix_len(uid):
        svg_path = os.path.join(SVG_DIR, f"{uid}.svg")
        try:
            svg = SVG.load_svg(svg_path)
            svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
            tokens = tokenizer.tokenize_svg_tensors(svg_tensors, color_tensors)
            return len(tokens)
        except Exception as e:
            print(f"  Error computing pix_len for {uid}: {e}")
            return 0

    for csv_name, ids in [("train_meta.csv", train_ids), ("val_meta.csv", val_ids)]:
        csv_path = os.path.join(DATA_DIR, csv_name)
        print(f"\nGenerating {csv_name} ({len(ids)} entries)...")

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'len_pix'])
            writer.writeheader()
            for uid in tqdm(ids, desc=csv_name):
                pix_len = compute_pix_len(uid)
                writer.writerow({'id': uid, 'len_pix': pix_len})

        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df)} rows, "
              f"len_pix range [{df['len_pix'].min()}, {df['len_pix'].max()}]")
        for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 99999)]:
            cnt = ((df['len_pix'] > lo) & (df['len_pix'] <= hi)).sum()
            print(f"    ({lo}, {hi}]: {cnt}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    step2_generate_csv()
    # ok = step1_preprocess_and_rename()
    # if ok:
    #     step2_generate_csv()
    # else:
    #     print("Step 1 produced no output, aborting.")
