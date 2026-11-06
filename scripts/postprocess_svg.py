#!/usr/bin/env python
"""对 img2svg / code-complement 推理结果做纯几何后处理（不重新推理）。

处理内容（逐条 path）:
  1. 去掉开头的 "M0 0" 占位点、连续重复点
  2. 去离群小段：跳远后几个点内又回到原处的一段（坐标 token 预测错）
  3. 重描截断：当一段连续点序列与本 path 前面已出现过的序列重复时截断并闭合
  4. 曲线控制点离群时把 C 退化为 L

用法:
  python scripts/postprocess_svg.py --pred_dir <in>/task1_img2svg --out_dir <out>/task1_img2svg
  python scripts/postprocess_svg.py --task complete --pred_dir <in>/task2_complete --out_dir <out>/task2_complete
然后用 metrics/eval_svg_tasks.py img2svg|complete --pred_dir <out>/... 评测。
complete 只清理模型生成的 *_completion.svg，*_combined.svg 中的残缺部分保持原样。
"""
import argparse
import glob
import json
import math
import os
import re
import shutil

CMD_RE = re.compile(r"([MLCAZmlcaz])([^MLCAZmlcaz]*)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PATH_RE = re.compile(r'(<path\b[^>]*?\sd=")([^"]*)(")', re.S)


def parse_d(d):
    segs = []
    for cmd, args in CMD_RE.findall(d):
        nums = [float(x) for x in NUM_RE.findall(args)]
        cmd = cmd.upper()
        if cmd == "Z":
            segs.append(("Z", []))
        elif cmd in ("M", "L"):
            for i in range(0, len(nums) - 1, 2):
                segs.append((cmd if i == 0 else "L", nums[i:i + 2]))
        elif cmd == "C":
            for i in range(0, len(nums) - 5, 6):
                segs.append(("C", nums[i:i + 6]))
        elif cmd == "A":
            for i in range(0, len(nums) - 6, 7):
                segs.append(("A", nums[i:i + 7]))
    return segs


def end_pt(seg):
    return tuple(seg[1][-2:]) if seg[1] else None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def split_subpaths(segs):
    # SVG 语义：Z 之后若直接跟绘制命令，新子路径从上一子路径起点开始
    subs, cur, last_start = [], [], None
    for s in segs:
        if s[0] == "M":
            if cur:
                subs.append(cur)
            cur = [s]
            last_start = s[1]
        elif s[0] == "Z":
            if cur:
                cur.append(s)
                subs.append(cur)
            cur = []
        else:
            if not cur:
                if last_start is None:
                    continue
                cur = [("M", list(last_start))]
            cur.append(s)
    if cur:
        subs.append(cur)
    return subs


def clean_subpath(sub, jump, close, ngram, lookahead):
    closed = sub[-1][0] == "Z"
    body = [s for s in sub if s[0] != "Z"]
    if not body:
        return []
    start = body[0]
    draws = body[1:]

    # 去连续重复点
    out, prev = [], end_pt(start)
    for s in draws:
        p = end_pt(s)
        if s[0] == "L" and p == prev:
            continue
        out.append(s)
        prev = p
    draws = out

    # 曲线控制点离群 -> 直线
    prev = end_pt(start)
    for i, s in enumerate(draws):
        p = end_pt(s)
        if s[0] == "C":
            c1, c2 = tuple(s[1][0:2]), tuple(s[1][2:4])
            span = dist(prev, p)
            if min(dist(c1, prev), dist(c1, p)) > max(jump, 2 * span) or \
               min(dist(c2, prev), dist(c2, p)) > max(jump, 2 * span):
                draws[i] = ("L", list(p))
        prev = p

    # 去离群小段：从 a 跳远后，lookahead 个点内又回到 a 附近，则删掉中间这段
    out, a = [], end_pt(start)
    i = 0
    while i < len(draws):
        b = end_pt(draws[i])
        if dist(a, b) > jump:
            back = None
            for j in range(i + 1, min(i + 1 + lookahead, len(draws))):
                if dist(a, end_pt(draws[j])) < close:
                    back = j
                    break
            if back is None and i + 1 >= len(draws) and dist(a, end_pt(start)) < close:
                back = len(draws)
            if back is not None:
                i = back
                continue
        out.append(draws[i])
        a = b
        i += 1
    draws = out

    # 重描截断：出现与前文重复的 ngram 连续点序列
    if ngram > 0:
        pts = [end_pt(start)] + [end_pt(s) for s in draws]
        seen = {}
        cut = None
        for i in range(len(pts) - ngram + 1):
            key = tuple(pts[i:i + ngram])
            if len(set(key)) < ngram:
                continue
            if key in seen and i - seen[key] >= ngram:
                cut = i
                break
            seen.setdefault(key, i)
        if cut is not None:
            draws = draws[:max(cut - 1, 0)]
            closed = True

    if not draws:
        return []
    res = [start] + draws
    if closed:
        res.append(("Z", []))
    return res


def fmt(v):
    return f"{v:.1f}"


def fmt_args(cmd, args):
    if cmd == "A":
        # 第 4、5 个参数是 large-arc / sweep 标志位，必须输出为整数
        return " ".join(fmt(x) if i not in (3, 4) else str(int(x)) for i, x in enumerate(args))
    return " ".join(fmt(x) for x in args)


def to_d(subs):
    parts = []
    for sub in subs:
        for cmd, args in sub:
            parts.append(cmd if cmd == "Z" else cmd + fmt_args(cmd, args))
    return " ".join(parts)


def clean_d(d, jump, close, ngram, lookahead):
    segs = parse_d(d)
    # 开头的 M0 0 占位
    segs = [s for i, s in enumerate(segs)
            if not (s[0] == "M" and s[1] == [0.0, 0.0]
                    and i + 1 < len(segs) and segs[i + 1][0] == "M")]
    subs = [clean_subpath(s, jump, close, ngram, lookahead) for s in split_subpaths(segs)]
    subs = [s for s in subs if len(s) >= 3]
    return to_d(subs)


PATH_TAG_RE = re.compile(r"<path\b[^>]*(?:/>|>\s*</path\s*>)", re.I | re.S)


def clean_svg_text(text, args):
    def repl(m):
        return m.group(1) + clean_d(m.group(2), args.jump, args.close, args.ngram, args.lookahead) + m.group(3)

    new = PATH_RE.sub(repl, text)
    new = re.sub(r'<path\b[^>]*?\sd=""[^>]*?>\s*</path>', "", new)
    new = re.sub(r'<path\b[^>]*?\sd=""[^>]*?/>', "", new)
    return new


def process_file(src, dst, args):
    with open(src, encoding="utf-8") as f:
        new = clean_svg_text(f.read(), args)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(new)


def rebuild_combined(src_combined, old_completion, new_completion, dst):
    """combined = 残缺 SVG + 补全 path（追加在末尾）；只替换末尾那批补全 path。"""
    with open(src_combined, encoding="utf-8") as f:
        combined = f.read()
    n_old = len(PATH_TAG_RE.findall(old_completion))
    tags = list(PATH_TAG_RE.finditer(combined))
    if n_old > len(tags):
        raise ValueError(f"{src_combined}: completion path 数多于 combined path 数")
    if n_old:
        head = combined[:tags[-n_old].start()]
        tail = combined[tags[-1].end():]
    else:
        head, tail = re.split(r"(?=</svg\s*>)", combined, maxsplit=1, flags=re.I)
    new_paths = [t.group(0) for t in PATH_TAG_RE.finditer(new_completion)]
    with open(dst, "w", encoding="utf-8") as f:
        f.write(head.rstrip() + ("\n  " + "\n  ".join(new_paths) if new_paths else "") + "\n" + tail.lstrip())


def run_img2svg(args):
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.svg")))
    for src in files:
        process_file(src, os.path.join(args.out_dir, os.path.basename(src)), args)
    return len(files)


def run_complete(args):
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*_completion.svg")))
    for src in files:
        name = os.path.basename(src)
        with open(src, encoding="utf-8") as f:
            old = f.read()
        new = clean_svg_text(old, args)
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            f.write(new)
        combined = src[:-len("_completion.svg")] + "_combined.svg"
        if os.path.exists(combined):
            rebuild_combined(combined, old, new,
                             os.path.join(args.out_dir, os.path.basename(combined)))
    return len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--jump", type=float, default=25.0, help="离群跳变阈值（200 视口坐标）")
    p.add_argument("--close", type=float, default=12.0, help="尖刺两端点的最大距离")
    p.add_argument("--ngram", type=int, default=6, help="重描检测的连续点数，0 关闭重描截断")
    p.add_argument("--lookahead", type=int, default=6, help="离群小段最多包含的点数")
    p.add_argument("--task", choices=["img2svg", "complete"], default="img2svg",
                   help="complete: 只清理 *_completion.svg，并据此重建 *_combined.svg（残缺部分不动）")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = run_complete(args) if args.task == "complete" else run_img2svg(args)
    stats_json = os.path.join(args.pred_dir, "inference_stats.json")
    if os.path.exists(stats_json):
        shutil.copy(stats_json, args.out_dir)
    with open(os.path.join(args.out_dir, "postprocess_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[{args.task}] processed {n} svg -> {args.out_dir}")


if __name__ == "__main__":
    main()
