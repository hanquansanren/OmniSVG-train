#!/usr/bin/env python3
"""
Simplify SVG paths in my_zhuan3 to reduce token count to ~2048.
Uses Ramer-Douglas-Peucker (RDP) algorithm + curve flattening.
"""

import os
import re
import csv
import sys
import math
import random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "/data/phd23_weiguang_zhang/works/svg/my_zhuan3"
SVG_DIR = os.path.join(DATA_DIR, "svg")
MODEL_SIZE = "4B"
CONFIG_PATH = "./configs/tokenization.yaml"
TARGET_TOKENS = 1700
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

TOKEN_RE = re.compile(r'[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)')

EPSILON_SCHEDULE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


# ── Geometry helpers ─────────────────────────────────────────────────────────

def point_line_dist(p, a, b):
    """Perpendicular distance from point p to line segment a-b."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def rdp(points, epsilon):
    """Iterative Ramer-Douglas-Peucker simplification."""
    if len(points) <= 2:
        return list(points)

    n = len(points)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        si, ei = stack.pop()
        dmax, idx = 0.0, si
        for i in range(si + 1, ei):
            d = point_line_dist(points[i], points[si], points[ei])
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon:
            keep[idx] = True
            if idx - si > 1:
                stack.append((si, idx))
            if ei - idx > 1:
                stack.append((idx, ei))

    return [points[i] for i in range(n) if keep[i]]


# ── Path parsing / reconstruction ────────────────────────────────────────────

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_path(d):
    """Parse processed SVG path into list of sub-paths.
    Each sub-path = list of segments: ('M',(x,y)), ('L',(x,y)),
    ('C',(x1,y1,x2,y2,x,y)), ('Z',None)
    """
    tokens = TOKEN_RE.findall(d)
    sub_paths = []
    cur = []
    cmd = None
    i = 0

    while i < len(tokens):
        t = tokens[i]
        if not is_num(t):
            cmd = t
            i += 1
            if cmd in ('Z', 'z'):
                cur.append(('Z', None))
                sub_paths.append(cur)
                cur = []
                cmd = None
            continue

        if cmd == 'M':
            x, y = int(float(tokens[i])), int(float(tokens[i + 1]))
            i += 2
            if cur:
                sub_paths.append(cur)
            cur = [('M', (x, y))]
            cmd = 'L'
        elif cmd == 'L':
            x, y = int(float(tokens[i])), int(float(tokens[i + 1]))
            i += 2
            cur.append(('L', (x, y)))
        elif cmd == 'C':
            x1, y1 = int(float(tokens[i])), int(float(tokens[i + 1]))
            x2, y2 = int(float(tokens[i + 2])), int(float(tokens[i + 3]))
            x, y = int(float(tokens[i + 4])), int(float(tokens[i + 5]))
            i += 6
            cur.append(('C', (x1, y1, x2, y2, x, y)))
        else:
            i += 1

    if cur:
        sub_paths.append(cur)
    return sub_paths


def rebuild_path(sub_paths):
    """Reconstruct path data string from sub-paths."""
    parts = []
    for sp in sub_paths:
        for seg_type, data in sp:
            if seg_type == 'M':
                parts.append(f'M{data[0]} {data[1]}')
            elif seg_type == 'L':
                parts.append(f'L{data[0]} {data[1]}')
            elif seg_type == 'C':
                parts.append(
                    f'C{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]}'
                )
            elif seg_type == 'Z':
                parts.append('Z')
    return ' '.join(parts)


# ── Endpoint helper ──────────────────────────────────────────────────────────

def endpoint(seg):
    st, d = seg
    if st == 'M':
        return d
    if st == 'L':
        return d
    if st == 'C':
        return (d[4], d[5])
    return None


# ── Simplification ───────────────────────────────────────────────────────────

def simplify_subpath(sp, epsilon, curve_thresh):
    """Simplify one sub-path: flatten near-straight C→L, then RDP on L runs."""
    if len(sp) <= 2:
        return sp

    # Step 1: flatten C→L if nearly straight
    flat = [sp[0]]
    for i in range(1, len(sp)):
        st, d = sp[i]
        if st == 'C':
            prev_pt = endpoint(flat[-1])
            end_pt = (d[4], d[5])
            cp1, cp2 = (d[0], d[1]), (d[2], d[3])
            if prev_pt and max(
                point_line_dist(cp1, prev_pt, end_pt),
                point_line_dist(cp2, prev_pt, end_pt),
            ) <= curve_thresh:
                flat.append(('L', end_pt))
            else:
                flat.append(sp[i])
        else:
            flat.append(sp[i])

    # Step 2: RDP on consecutive L runs
    result = []
    i = 0
    while i < len(flat):
        st, _ = flat[i]
        if st == 'L':
            prev_pt = endpoint(result[-1]) if result else endpoint(flat[0])
            polyline = [prev_pt]
            while i < len(flat) and flat[i][0] == 'L':
                polyline.append(flat[i][1])
                i += 1
            simplified = rdp(polyline, epsilon)
            for pt in simplified[1:]:
                result.append(('L', pt))
        else:
            result.append(flat[i])
            i += 1

    # Step 3: remove zero-length L segments
    cleaned = [result[0]]
    for seg in result[1:]:
        if seg[0] == 'L':
            prev_pt = endpoint(cleaned[-1])
            if seg[1] != prev_pt:
                cleaned.append(seg)
        else:
            cleaned.append(seg)

    return cleaned


def estimate_tokens(sub_paths):
    """Fast estimate of token count from command structure."""
    n = 0
    for sp in sub_paths:
        for st, d in sp:
            if st == 'M':
                n += 3
            elif st == 'L':
                n += 3
            elif st == 'C':
                n += 7
            elif st == 'Z':
                n += 1
    return n


def simplify_path_data(d, target=TARGET_TOKENS):
    """Simplify path data with adaptive epsilon until under target tokens."""
    sub_paths = parse_path(d)
    est = estimate_tokens(sub_paths)
    if est <= target:
        return d, est, 0.0

    for eps in EPSILON_SCHEDULE:
        simplified = [simplify_subpath(sp, eps, eps) for sp in sub_paths]
        est = estimate_tokens(simplified)
        if est <= target:
            return rebuild_path(simplified), est, eps

    return rebuild_path(simplified), est, EPSILON_SCHEDULE[-1]


# ── SVG-level processing ────────────────────────────────────────────────────

def simplify_svg_content(content, target=TARGET_TOKENS):
    """Simplify all paths in an SVG string, controlling TOTAL token budget."""
    d_attrs = re.findall(r'd="([^"]*)"', content)
    all_parsed = [parse_path(d) for d in d_attrs]
    total_before = sum(estimate_tokens(sp) for sp in all_parsed)

    if total_before <= target:
        return content, total_before, total_before, 0.0

    best_simplified = all_parsed
    best_eps = 0.0

    for eps in EPSILON_SCHEDULE:
        simplified = [
            [simplify_subpath(sp, eps, eps) for sp in sp_list]
            for sp_list in all_parsed
        ]
        total_after = sum(estimate_tokens(sp) for sp in simplified)
        best_simplified = simplified
        best_eps = eps
        if total_after <= target:
            break

    total_after = sum(estimate_tokens(sp) for sp in best_simplified)

    idx = 0
    def _replace(m):
        nonlocal idx
        new_d = rebuild_path(best_simplified[idx])
        idx += 1
        return f'd="{new_d}"'

    new_content = re.sub(r'd="([^"]*)"', _replace, content)
    return new_content, total_before, total_after, best_eps


# ── Main ─────────────────────────────────────────────────────────────────────

def step1_simplify():
    print("=" * 60)
    print(f"Step 1: Simplify SVGs (target ≤ {TARGET_TOKENS} est-tokens)")
    print("=" * 60)

    svg_files = sorted(Path(SVG_DIR).glob("*.svg"))
    total = len(svg_files)
    print(f"Total SVG files: {total}")

    stats = {"ok": 0, "still_over": 0, "already_ok": 0, "fail": 0}
    eps_used = {}

    for svg_path in tqdm(svg_files, desc="Simplifying"):
        try:
            content = svg_path.read_text(encoding="utf-8")
            new_content, est_before, est_after, eps = simplify_svg_content(content)
            svg_path.write_text(new_content, encoding="utf-8")

            if est_before <= TARGET_TOKENS:
                stats["already_ok"] += 1
            elif est_after <= TARGET_TOKENS:
                stats["ok"] += 1
            else:
                stats["still_over"] += 1

            eps_key = f"{eps:.1f}"
            eps_used[eps_key] = eps_used.get(eps_key, 0) + 1

        except Exception as e:
            print(f"  Error: {svg_path.name}: {e}")
            stats["fail"] += 1

    print(f"\nResults:")
    print(f"  Already ≤ {TARGET_TOKENS}: {stats['already_ok']}")
    print(f"  Simplified to ≤ {TARGET_TOKENS}: {stats['ok']}")
    print(f"  Still > {TARGET_TOKENS}: {stats['still_over']}")
    print(f"  Failed: {stats['fail']}")
    print(f"\nEpsilon distribution: {dict(sorted(eps_used.items()))}")


def step2_rebuild_csv():
    print("\n" + "=" * 60)
    print("Step 2: Rebuild CSV with updated pix_len")
    print("=" * 60)

    sys.path.insert(0, ".")
    from utils.config import TokenizationConfig
    from utils.dataset import SVGTokenizer
    from deepsvg.svglib.svg import SVG

    cfg = TokenizationConfig.from_yaml(CONFIG_PATH, MODEL_SIZE)
    tokenizer = SVGTokenizer(cfg)

    svg_files = sorted(Path(SVG_DIR).glob("*.svg"))
    file_ids = [f.stem for f in svg_files]
    total = len(file_ids)

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
            svg_t, color_t = svg.to_tensor(concat_groups=False, PAD_VAL=0)
            tokens = tokenizer.tokenize_svg_tensors(svg_t, color_t)
            return len(tokens)
        except Exception as e:
            print(f"  Error pix_len {uid}: {e}")
            return 0

    import pandas as pd

    for csv_name, ids in [("train_meta.csv", train_ids), ("val_meta.csv", val_ids)]:
        csv_path = os.path.join(DATA_DIR, csv_name)
        print(f"\nGenerating {csv_name} ({len(ids)} entries)...")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "len_pix"])
            writer.writeheader()
            for uid in tqdm(ids, desc=csv_name):
                writer.writerow({"id": uid, "len_pix": compute_pix_len(uid)})

        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df)} rows, "
              f"len_pix [{df['len_pix'].min()}, {df['len_pix'].max()}]")
        for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 99999)]:
            cnt = ((df["len_pix"] > lo) & (df["len_pix"] <= hi)).sum()
            print(f"    ({lo}, {hi}]: {cnt}")


if __name__ == "__main__":
    step1_simplify()
    step2_rebuild_csv()
