"""SSIM / LPIPS / MSE evaluation for the img2svg and SVG code-complement tasks.

Subcommands:
  split     Sample a fixed test set from a dataset's val split (val_meta.csv).
  img2svg   Compare rendered predicted SVGs against the input (GT) PNGs.
  complete  Compare (input PNG + predicted completion) against
            (input PNG + GT replacement paths from json/*.json).

All images are brought onto the 200x200 viewBox canvas before comparison:
  * my_zhuan4 PNGs are 512x512 renderings of the 200x200 canvas.
  * my_lis2_2 PNGs are the 200x200 canvas stretched non-uniformly to WxH,
    so resizing back to a square restores the viewBox alignment.

Examples:
  # 1) Build test sets (60 samples each)
  python metrics/eval_svg_tasks.py split --task img2svg \\
      --data_root /data/phd23_weiguang_zhang/works/svg/my_zhuan4 \\
      --out_dir /data/phd23_weiguang_zhang/works/svg/test60_img2svg_zhuan4
  python metrics/eval_svg_tasks.py split --task complete \\
      --data_root /data/phd23_weiguang_zhang/works/svg/my_lis2_2 \\
      --out_dir /data/phd23_weiguang_zhang/works/svg/test60_complete_lis2_2

  # 2) Inference (existing script)
  python inference.py --task image-to-svg --input <img2svg_test>/png --output <pred1> --save-svg
  python inference.py --task code-complement --input <complete_test> --output <pred2> --save-svg

  # 3) Evaluate
  python metrics/eval_svg_tasks.py img2svg --test_dir <img2svg_test> --pred_dir <pred1>
  python metrics/eval_svg_tasks.py complete --test_dir <complete_test> --pred_dir <pred2>
"""

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import sys
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_ssim_lpips_mse import (  # noqa: E402
    LPIPS_AVAILABLE,
    calculate_lpips,
    calculate_mse,
    skimage,
    ssim,
)

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg is required. pip install cairosvg")
    sys.exit(1)

if LPIPS_AVAILABLE:
    import lpips
    import torch

VIEWBOX_SIZE = 200
METRICS = ("mse", "ssim", "lpips")
HIGHER_IS_BETTER = {"mse": False, "ssim": True, "lpips": False}


# ---------------------------------------------------------------------------
# Image helpers (all return float64 arrays in [0, 1])
# ---------------------------------------------------------------------------

def load_png_on_canvas(png_path, size, background=(255, 255, 255)):
    """Load a PNG, composite onto background and resize to the square canvas."""
    img = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, background + (255,))
    img = Image.alpha_composite(bg, img).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def render_svg_rgba(svg_str, size):
    """Render an SVG string to an RGBA float array (HxWx4), or None on failure."""
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
    except Exception as e:
        print(f"  Render error: {e}")
        return None
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def over(base_rgb, top_rgba):
    """Alpha-composite an RGBA layer over an RGB base."""
    alpha = top_rgba[:, :, 3:4]
    return top_rgba[:, :, :3] * alpha + base_rgb * (1.0 - alpha)


def white_canvas(size):
    return np.ones((size, size, 3), dtype=np.float64)


def paths_to_svg(paths):
    """Wrap (d, fill) pairs as a 200x200 viewBox SVG document."""
    body = "\n  ".join(
        f"<path fill={quoteattr(fill)} d={quoteattr(d)}/>" for d, fill in paths
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}">\n  {body}\n</svg>'
    )


def load_gt_replacement_paths(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        patch = json.load(f)
    paths = []
    for op in patch.get("operations", []):
        d = str(op.get("replacement", "")).strip()
        if d:
            paths.append((d, str(op.get("fill", "#000") or "#000")))
    return paths


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def render_svg_file_on_white(svg_path, size):
    """Render an SVG file onto a white square canvas."""
    with open(svg_path, "r", encoding="utf-8") as f:
        layer = render_svg_rgba(f.read(), size)
    if layer is None:
        return None
    return over(white_canvas(size), layer)


def default_img2svg_svg_subdir(test_dir):
    """Which GT SVG folder to use: eval_task1_1 -> svg1, eval_task1_2 -> svg2."""
    base = os.path.basename(os.path.normpath(test_dir))
    if base == "eval_task1_2":
        return "svg2"
    if os.path.isdir(os.path.join(test_dir, "svg2")) and not os.path.isdir(
        os.path.join(test_dir, "svg1")
    ):
        return "svg2"
    if os.path.isdir(os.path.join(test_dir, "svg1")):
        return "svg1"
    return None


def load_gt200_png(test_dir, uid, size):
    png_path = os.path.join(test_dir, "gt200", uid + ".png")
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"GT png not found: {png_path}")
    img = Image.open(png_path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def rebuild_img2svg_gt200(test_dir, size=VIEWBOX_SIZE, ids=None, svg_subdir=None):
    """Write gt200/ by rendering svg1/ (task1_1) or svg2/ (task1_2), not from png/."""
    if ids is None:
        ids = read_ids(test_dir)
    if svg_subdir is None:
        svg_subdir = default_img2svg_svg_subdir(test_dir)
    if not svg_subdir:
        raise ValueError(f"No svg1/ or svg2/ under {test_dir}")

    for legacy in ("gt200_svg1", "gt200_svg2"):
        path = os.path.join(test_dir, legacy)
        if os.path.isdir(path):
            shutil.rmtree(path)

    svg_dir = os.path.join(test_dir, svg_subdir)
    if not os.path.isdir(svg_dir):
        raise FileNotFoundError(f"Missing {svg_dir}")

    out_dir = os.path.join(test_dir, "gt200")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for uid in ids:
        svg_path = os.path.join(svg_dir, uid + ".svg")
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"Missing {svg_path}")
        gt = render_svg_file_on_white(svg_path, size)
        if gt is None:
            raise RuntimeError(f"Failed to render {svg_path}")
        Image.fromarray(to_uint8(gt)).save(os.path.join(out_dir, uid + ".png"))
    return svg_subdir


def load_gt_image(task, test_dir, uid, size, gt_source="auto"):
    """GT image on the square canvas.

    img2svg:  gt200/ (from svg1 or svg2), or render svg1/svg2 / input png on demand.
    complete: the input (partial) PNG with the GT replacement paths rendered on top.
    """
    if task == "img2svg":
        if gt_source == "png":
            return load_png_on_canvas(os.path.join(test_dir, "png", uid + ".png"), size)
        if gt_source == "gt200":
            return load_gt200_png(test_dir, uid, size)
        if gt_source == "auto":
            gt_png = os.path.join(test_dir, "gt200", uid + ".png")
            if os.path.exists(gt_png):
                return load_gt200_png(test_dir, uid, size)
            gt_source = default_img2svg_svg_subdir(test_dir) or "png"
        if gt_source in ("svg1", "svg2"):
            svg_path = os.path.join(test_dir, gt_source, uid + ".svg")
            if not os.path.exists(svg_path):
                raise FileNotFoundError(f"GT SVG not found: {svg_path}")
            gt = render_svg_file_on_white(svg_path, size)
            if gt is None:
                raise RuntimeError(f"Failed to render GT SVG {svg_path}")
            return gt
        return load_png_on_canvas(os.path.join(test_dir, "png", uid + ".png"), size)

    base = load_png_on_canvas(os.path.join(test_dir, "png", uid + ".png"), size)
    paths = load_gt_replacement_paths(os.path.join(test_dir, "json", uid + ".json"))
    layer = render_svg_rgba(paths_to_svg(paths), size)
    if layer is None:
        raise RuntimeError(f"Failed to render GT replacement for {uid}")
    return over(base, layer)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_lpips_model(enabled, net="alex"):
    if not enabled:
        return None
    if not LPIPS_AVAILABLE:
        print("Warning: lpips not installed, LPIPS will be skipped.")
        return None
    model = lpips.LPIPS(net=net, verbose=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def calculate_ssim(img1, img2):
    major, minor = [int(x) for x in skimage.__version__.split(".")[:2]]
    if major == 0 and minor < 19:
        return ssim(img1, img2, multichannel=True, data_range=1.0)
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


def compute_metrics(gt, pred, lpips_model):
    return {
        "mse": float(calculate_mse(gt, pred)),
        "ssim": float(calculate_ssim(gt, pred)),
        "lpips": (
            float(calculate_lpips(gt, pred, lpips_model))
            if lpips_model is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Test-set helpers
# ---------------------------------------------------------------------------

def read_ids(test_dir, meta_name="test_meta.csv"):
    """Read sample ids from test_meta.csv, falling back to png/*.png."""
    meta_path = os.path.join(test_dir, meta_name)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return [row["id"] for row in csv.DictReader(f)]
    png_dir = os.path.join(test_dir, "png")
    return sorted(os.path.splitext(n)[0] for n in os.listdir(png_dir) if n.lower().endswith(".png"))


def find_candidates(pred_dir, uid, kind):
    """Return {candidate_index: path} for one sample.

    kind: "svg" -> {uid}.svg / {uid}_candidate_{k}.svg          (image-to-svg)
          "completion" / "combined"
              -> {uid}_{kind}.svg / {uid}_candidate_{k}_{kind}.svg (code-complement)
    Files without a candidate suffix are treated as candidate 1.
    """
    tail = ".svg" if kind == "svg" else f"_{kind}.svg"
    found = {}
    plain = os.path.join(pred_dir, f"{uid}{tail}")
    if os.path.exists(plain):
        found[1] = plain
    pattern = re.compile(rf"^{re.escape(uid)}_candidate_(\d+){re.escape(tail)}$")
    for name in os.listdir(pred_dir):
        m = pattern.match(name)
        if m:
            found.setdefault(int(m.group(1)), os.path.join(pred_dir, name))
    return found


def select_candidate(uid, cands, gt, make_pred, lpips_model, candidate, best_by):
    """Pick which candidate to score.

    candidate: an int k (use candidate k) or "best" (oracle best-of-N w.r.t. GT).
    Returns (cand_idx, pred_image, metrics) or (None, None, None) if unavailable.
    """
    if candidate != "best":
        k = int(candidate)
        if k not in cands:
            return None, None, None
        pred = make_pred(uid, cands[k])
        if pred is None:
            return k, None, None
        return k, pred, compute_metrics(gt, pred, lpips_model)

    if best_by == "lpips" and lpips_model is None:
        best_by = "mse"
    best = (None, None, None)
    for k in sorted(cands):
        pred = make_pred(uid, cands[k])
        if pred is None:
            continue
        m = compute_metrics(gt, pred, lpips_model)
        if best[2] is None:
            best = (k, pred, m)
            continue
        better = m[best_by] > best[2][best_by] if HIGHER_IS_BETTER[best_by] else m[best_by] < best[2][best_by]
        if better:
            best = (k, pred, m)
    return best


def evaluate_samples(ids, load_gt, load_fallback, make_pred, find, args, lpips_model):
    """Shared evaluation loop for both tasks."""
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    rows = []
    for i, uid in enumerate(ids):
        gt = load_gt(uid)
        cands = find(uid)
        k, pred, metrics = select_candidate(
            uid, cands, gt, make_pred, lpips_model, args.candidate, args.best_by
        )

        status = "ok"
        if pred is None:
            tried = k is not None or (cands and args.candidate == "best")
            status = "render_failed" if tried else "missing"
            if args.missing == "skip":
                print(f"[{i + 1}/{len(ids)}] {uid}: {status}, skipped")
                rows.append({"id": uid, "candidate": "", "status": status,
                             "mse": None, "ssim": None, "lpips": None})
                continue
            pred = load_fallback(uid)
            metrics = compute_metrics(gt, pred, lpips_model)

        rows.append({"id": uid, "candidate": k if k is not None else "",
                     "status": status, **metrics})
        lp = f"{metrics['lpips']:.4f}" if metrics["lpips"] is not None else "N/A"
        print(f"[{i + 1}/{len(ids)}] {uid} cand={k} {status}: "
              f"MSE={metrics['mse']:.5f} SSIM={metrics['ssim']:.4f} LPIPS={lp}")

        if args.save_vis:
            vis = np.concatenate([gt, np.zeros((gt.shape[0], 2, 3)), pred], axis=1)
            Image.fromarray(to_uint8(vis)).save(os.path.join(vis_dir, f"{uid}_gt_vs_pred.png"))

    return rows


def merge_inference_stats(rows, pred_dir):
    """Attach per-sample infer timing/tokens from inference_stats.json if present."""
    path = os.path.join(pred_dir, "inference_stats.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {s["id"]: s for s in data.get("samples", [])}
    for r in rows:
        st = by_id.get(r["id"])
        if not st:
            continue
        r["output_tokens"] = st.get("output_tokens")
        r["elapsed_sec"] = st.get("elapsed_sec")
        r["gen_seconds"] = st.get("gen_seconds")
    return data.get("summary") or {}


def summarize(rows, args, task):
    infer_summary = merge_inference_stats(rows, args.pred_dir)
    scored = [r for r in rows if r["mse"] is not None]
    summary = {
        "task": task,
        "test_dir": os.path.abspath(args.test_dir),
        "pred_dir": os.path.abspath(args.pred_dir),
        "candidate": args.candidate,
        "missing_policy": args.missing,
        "canvas_size": args.size,
        "num_samples": len(rows),
        "num_scored": len(scored),
        "num_ok": sum(r["status"] == "ok" for r in rows),
        "num_missing": sum(r["status"] == "missing" for r in rows),
        "num_render_failed": sum(r["status"] == "render_failed" for r in rows),
    }
    for m in METRICS:
        vals = [r[m] for r in scored if r[m] is not None]
        summary[m] = float(np.mean(vals)) if vals else None
        summary[f"{m}_std"] = float(np.std(vals)) if vals else None
    if task == "complete":
        summary["pred_source"] = args.pred_source

    tok_vals = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    sec_vals = [r["elapsed_sec"] for r in rows if r.get("elapsed_sec") is not None]
    if tok_vals:
        summary["avg_output_tokens"] = float(np.mean(tok_vals))
        summary["output_tokens_std"] = float(np.std(tok_vals))
    elif infer_summary.get("avg_output_tokens") is not None:
        summary["avg_output_tokens"] = infer_summary["avg_output_tokens"]
        if infer_summary.get("output_tokens_std") is not None:
            summary["output_tokens_std"] = infer_summary["output_tokens_std"]
    else:
        summary["avg_output_tokens"] = None

    if sec_vals:
        summary["avg_elapsed_sec"] = float(np.mean(sec_vals))
        summary["elapsed_sec_std"] = float(np.std(sec_vals))
    elif infer_summary.get("avg_elapsed_sec") is not None:
        summary["avg_elapsed_sec"] = infer_summary["avg_elapsed_sec"]
        if infer_summary.get("elapsed_sec_std") is not None:
            summary["elapsed_sec_std"] = infer_summary["elapsed_sec_std"]
    else:
        summary["avg_elapsed_sec"] = None

    csv_path = os.path.join(args.out_dir, f"{task}_per_sample.csv")
    extra_cols = []
    if any(r.get("output_tokens") is not None for r in rows):
        extra_cols.append("output_tokens")
    if any(r.get("elapsed_sec") is not None for r in rows):
        extra_cols.append("elapsed_sec")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "candidate", "status", *extra_cols, *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.out_dir, f"{task}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Task: {task} | candidate={args.candidate} | missing={args.missing}")
    print(f"Samples: {summary['num_samples']} (ok={summary['num_ok']}, "
          f"missing={summary['num_missing']}, render_failed={summary['num_render_failed']})")
    for m, arrow in (("mse", "↓"), ("ssim", "↑"), ("lpips", "↓")):
        if summary[m] is None:
            print(f"  {m.upper():<6}{arrow}: N/A")
        else:
            print(f"  {m.upper():<6}{arrow}: {summary[m]:.5f} ± {summary[f'{m}_std']:.5f}")
    if summary.get("avg_output_tokens") is not None:
        std = summary.get("output_tokens_std")
        std_s = f" ± {std:.1f}" if std is not None else ""
        print(f"  TOKENS  : {summary['avg_output_tokens']:.1f}{std_s} (1st valid candidate)")
    else:
        print("  TOKENS  : N/A (re-run inference to produce inference_stats.json)")
    if summary.get("avg_elapsed_sec") is not None:
        std = summary.get("elapsed_sec_std")
        std_s = f" ± {std:.3f}" if std is not None else ""
        print(f"  SEC/SAMPLE: {summary['avg_elapsed_sec']:.3f}s{std_s}")
    else:
        print("  SEC/SAMPLE: N/A")
    print(f"Saved: {csv_path}\n       {json_path}")
    print("=" * 60)
    return summary


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_split(args):
    root = args.data_root
    meta_path = os.path.join(root, "val_meta.csv")
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    subdirs = ["png", "svg"] + (["json"] if args.task == "complete" else [])
    exts = {"png": ".png", "svg": ".svg", "json": ".json"}

    def valid(uid):
        if not all(os.path.exists(os.path.join(root, d, uid + exts[d])) for d in subdirs):
            return False
        if args.task == "complete":
            return bool(load_gt_replacement_paths(os.path.join(root, "json", uid + ".json")))
        return True

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    picked, seen_chars = [], set()
    for row in rows:
        uid = row["id"]
        char = uid.split("_", 1)[0]
        if not args.allow_dup_char and char in seen_chars:
            continue
        if not valid(uid):
            continue
        picked.append(row)
        seen_chars.add(char)
        if len(picked) == args.num:
            break
    if len(picked) < args.num:
        print(f"Warning: only {len(picked)} valid samples found (requested {args.num}).")
    picked.sort(key=lambda r: r["id"])

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        print(f"Error: {args.out_dir} is not empty. Use --overwrite to replace it.")
        sys.exit(1)
    for d in subdirs:
        path = os.path.join(args.out_dir, d)
        if args.overwrite and os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    for row in picked:
        uid = row["id"]
        for d in subdirs:
            shutil.copy2(os.path.join(root, d, uid + exts[d]),
                         os.path.join(args.out_dir, d, uid + exts[d]))

    if args.task == "img2svg" and default_img2svg_svg_subdir(args.out_dir):
        sub = rebuild_img2svg_gt200(args.out_dir, VIEWBOX_SIZE, [r["id"] for r in picked])
        print(f"Built gt200/ from {sub}/")

    with open(os.path.join(args.out_dir, "test_meta.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(picked)

    print(f"Picked {len(picked)} samples from {meta_path} (seed={args.seed}) -> {args.out_dir}")
    if args.task == "img2svg":
        print(f"Inference input: --task image-to-svg --input {os.path.join(args.out_dir, 'png')}")
    else:
        print(f"Inference input: --task code-complement --input {args.out_dir}")


def cmd_rebuild_gt200(args):
    ids = read_ids(args.test_dir)
    svg_subdir = args.svg_subdir
    sub = rebuild_img2svg_gt200(args.test_dir, args.size, ids, svg_subdir=svg_subdir)
    print(f"Rebuilt gt200/ from {sub}/ under {args.test_dir} ({len(ids)} samples)")


def cmd_img2svg(args):
    ids = read_ids(args.test_dir)
    lpips_model = build_lpips_model(not args.no_lpips, args.lpips_net)
    size = args.size

    def make_pred(uid, svg_path):
        with open(svg_path, "r", encoding="utf-8") as f:
            layer = render_svg_rgba(f.read(), size)
        return None if layer is None else over(white_canvas(size), layer)

    rows = evaluate_samples(
        ids,
        load_gt=lambda uid: load_gt_image(
            "img2svg", args.test_dir, uid, size, gt_source=args.gt_source
        ),
        load_fallback=lambda uid: white_canvas(size),
        make_pred=make_pred,
        find=lambda uid: find_candidates(args.pred_dir, uid, "svg"),
        args=args,
        lpips_model=lpips_model,
    )
    summarize(rows, args, "img2svg")


def cmd_complete(args):
    ids = read_ids(args.test_dir)
    lpips_model = build_lpips_model(not args.no_lpips, args.lpips_net)
    size = args.size
    base_cache = {}

    def base_image(uid):
        if uid not in base_cache:
            base_cache[uid] = load_png_on_canvas(os.path.join(args.test_dir, "png", uid + ".png"), size)
        return base_cache[uid]

    kind = "completion" if args.pred_source == "overlay" else "combined"

    def make_pred(uid, svg_path):
        with open(svg_path, "r", encoding="utf-8") as f:
            layer = render_svg_rgba(f.read(), size)
        if layer is None:
            return None
        if args.pred_source == "overlay":
            return over(base_image(uid), layer)
        return over(white_canvas(size), layer)

    rows = evaluate_samples(
        ids,
        load_gt=lambda uid: load_gt_image("complete", args.test_dir, uid, size),
        load_fallback=base_image,
        make_pred=make_pred,
        find=lambda uid: find_candidates(args.pred_dir, uid, kind),
        args=args,
        lpips_model=lpips_model,
    )
    summarize(rows, args, "complete")


def parse_candidate(value):
    if value == "best":
        return value
    try:
        k = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--candidate must be a positive int or 'best'")
    if k < 1:
        raise argparse.ArgumentTypeError("--candidate must be >= 1")
    return k


def add_eval_args(p):
    p.add_argument("--test_dir", required=True,
                   help="Test set dir created by `split` (contains png/, test_meta.csv, ...)")
    p.add_argument("--pred_dir", required=True, help="Output dir of inference.py")
    p.add_argument("--out_dir", default=None,
                   help="Where to write per-sample CSV / summary JSON (default: <pred_dir>/metrics)")
    p.add_argument("--candidate", type=parse_candidate, default=1,
                   help="Candidate index to score (default 1), or 'best' for oracle best-of-N")
    p.add_argument("--best_by", choices=METRICS, default="lpips",
                   help="Metric used to pick the candidate when --candidate best")
    p.add_argument("--missing", choices=["fallback", "skip"], default="fallback",
                   help="Missing/unrenderable prediction: 'fallback' scores a blank result "
                        "(white canvas for img2svg, the unmodified input for complete); "
                        "'skip' excludes the sample")
    p.add_argument("--size", type=int, default=VIEWBOX_SIZE,
                   help="Square resolution of the comparison canvas (default 200 = viewBox size)")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    p.add_argument("--save_vis", action="store_true", help="Save GT|Pred side-by-side images")


def add_img2svg_gt_args(p):
    p.add_argument(
        "--gt_source",
        default="auto",
        choices=["auto", "png", "svg1", "svg2", "gt200"],
        help="img2svg GT: auto uses gt200/ if present (else svg1/svg2 by test set), or explicit",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="Sample a test set from val_meta.csv")
    p.add_argument("--task", choices=["img2svg", "complete"], required=True)
    p.add_argument("--data_root", required=True, help="e.g. .../svg/my_zhuan4 or .../svg/my_lis2_2")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_dup_char", action="store_true",
                   help="Allow multiple samples of the same character (default: one per codepoint)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("img2svg", help="Evaluate the image-to-svg task")
    add_eval_args(p)
    add_img2svg_gt_args(p)
    p.set_defaults(func=cmd_img2svg)

    p = sub.add_parser(
        "rebuild-gt200",
        help="Regenerate gt200/ from svg1/ (eval_task1_1) or svg2/ (eval_task1_2)",
    )
    p.add_argument("--test_dir", required=True)
    p.add_argument("--size", type=int, default=VIEWBOX_SIZE)
    p.add_argument("--svg_subdir", choices=["svg1", "svg2"], default=None,
                   help="Force GT SVG folder (default: infer from test_dir name)")
    p.set_defaults(func=cmd_rebuild_gt200)

    p = sub.add_parser("complete", help="Evaluate the SVG code-complement task")
    add_eval_args(p)
    p.add_argument("--pred_source", choices=["overlay", "combined"], default="overlay",
                   help="overlay: input PNG + *_completion.svg (same construction as GT, default); "
                        "combined: render *_combined.svg (partial SVG + completion) on white")
    p.set_defaults(func=cmd_complete)

    args = parser.parse_args()
    if getattr(args, "pred_dir", None) and args.out_dir is None:
        args.out_dir = os.path.join(args.pred_dir, "metrics")
    args.func(args)


if __name__ == "__main__":
    main()
