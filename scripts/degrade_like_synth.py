#!/usr/bin/env python
"""把干净黑白字图近似退化成训练合成数据（noisy_stage3）的风格，用于对照实验。

模拟: 灰色纹理背景(~192) + 竖条纹理、灰绿墨色(~73,78,74)、锯齿/毛刺边缘、可选断笔。
只是近似，不等同于真实的合成退化流水线。

用法:
  python scripts/degrade_like_synth.py --src eval/real31/gt200 --dst eval/real31_synthlike_light/png --level light
  python scripts/degrade_like_synth.py --src eval/real31/gt200 --dst eval/real31_synthlike_heavy/png --level heavy
"""
import argparse
import glob
import os
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SIZE = 448
BG_GRAY = 192.0
INK_RGB = np.array([73.0, 78.0, 74.0])


def norm_noise(rng, shape, sigma):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (n.std() + 1e-8)


def degrade(img, rng, level):
    gray = np.asarray(img.convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=float)
    ink = gray < 128
    shape = ink.shape

    # 锯齿边缘 + 竖条毛刺：对软化的 mask 加细噪声和竖向拉长的噪声后重新阈值
    soft = ndi.gaussian_filter(ink.astype(float), 2.0)
    edge_noise = 0.22 * norm_noise(rng, shape, 1.2) + 0.25 * norm_noise(rng, shape, (7.0, 0.8))
    near = ndi.binary_dilation(ink, iterations=3)
    ink = ((soft + edge_noise) > 0.5) & near

    if level == "heavy":
        # 断笔：低频噪声最低的一部分区域擦掉墨迹
        low = norm_noise(rng, shape, 18.0)
        ink &= low > np.quantile(low[ink], 0.25) if ink.any() else True

    bg = (BG_GRAY
          + 14 * norm_noise(rng, shape, 30.0)
          + 6 * norm_noise(rng, shape, 8.0)
          + 3 * norm_noise(rng, shape, (40.0, 3.0))
          + 2 * norm_noise(rng, shape, 1.0))
    bg = np.repeat(bg[..., None], 3, axis=2)

    ink_tex = 10 * norm_noise(rng, shape, (6.0, 0.7)) + 4 * norm_noise(rng, shape, 1.0)
    ink_rgb = INK_RGB[None, None, :] + ink_tex[..., None]

    alpha = ndi.gaussian_filter(ink.astype(float), 0.6)[..., None]
    out = bg * (1 - alpha) + ink_rgb * alpha
    out = ndi.gaussian_filter(out, (0.7, 0.7, 0))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="干净黑白图目录（白底黑字）")
    p.add_argument("--dst", required=True)
    p.add_argument("--level", choices=["light", "heavy"], default="light",
                   help="light: 背景/墨色/毛刺边缘；heavy: 再加断笔")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.png")))
    for f in files:
        name = os.path.basename(f)
        rng = np.random.default_rng(zlib.crc32(name.encode()) + args.seed)
        degrade(Image.open(f), rng, args.level).save(os.path.join(args.dst, name))
    print(f"[{args.level}] {len(files)} images -> {args.dst}")


if __name__ == "__main__":
    main()
