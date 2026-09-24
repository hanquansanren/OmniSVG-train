"""Build GRPO samples from the stage-2 code-complement data directory.

Layout expected (same as ``utils/dataset.py``)::

    data_dir/
      train_meta.csv        # column `id`
      val_meta.csv
      svg/{uid}.svg         # damaged glyph S_d
      png/{uid}.png         # rendered damaged glyph I_d
      json/{uid}.json       # patch metadata

Each sample carries the four ground-truth SVGs the reward pipeline needs:

``damaged_svg``
    ``svg/{uid}.svg`` verbatim.
``gt_missing_contour_svg``
    Union of ``operations[].replacement`` - the contour of the missing strokes.
``gt_skeleton_svg``
    Top-level ``skeleton_svg`` (the patch-region skeleton), falling back to the
    concatenation of ``operations[].skeleton_svg``.
``gt_full_svg``
    ``damaged_svg`` with the replacement paths appended, i.e. exactly what
    ``inference.py::combine_partial_and_completion`` builds at inference time.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .raster import DEFAULT_VIEWBOX, extract_path_data

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
_SVG_CLOSE_RE = re.compile(r"</svg\s*>", re.IGNORECASE)


def wrap_paths(path_data: Iterable[str], fill: str = "#000") -> str:
    """Wrap ``d`` attributes into an SVG document on the project canvas."""
    paths = [d for d in path_data if d and d.strip()]
    if not paths:
        return ""
    min_x, min_y, width, height = DEFAULT_VIEWBOX
    body = "".join(f'<path fill="{fill}" d="{d}"/>' for d in paths)
    return (
        f'<svg xmlns="{SVG_NS}" viewBox="{min_x} {min_y} {width} {height}" '
        f'width="{int(width)}" height="{int(height)}">{body}</svg>'
    )


def append_paths_to_svg(base_svg: str, path_data: Iterable[str], fill: str = "#000") -> str:
    """Append paths to an SVG document, mirroring inference-time merging."""
    paths = [d for d in path_data if d and d.strip()]
    if not paths:
        return base_svg
    body = "".join(f'<path fill="{fill}" d="{d}"/>' for d in paths)
    if _SVG_CLOSE_RE.search(base_svg):
        return _SVG_CLOSE_RE.sub(body + "</svg>", base_svg, count=1)
    return wrap_paths(extract_path_data(base_svg) + paths, fill=fill)


def decode_uid(uid: str) -> Dict[str, str]:
    """Split ``{codepoint_hex}_{font}_v{NN}`` into its parts."""
    codepoint, _, rest = uid.partition("_")
    font_name, _, version = rest.rpartition("_")
    try:
        char = chr(int(codepoint, 16))
    except ValueError:
        char = ""
    return {
        "codepoint": codepoint,
        "font_name": font_name or rest,
        "version": version,
        "char": char,
        "char_label": f"{char} (U+{codepoint})" if char else f"U+{codepoint}",
    }


def build_sample(data_dir: str, uid: str, require_image: bool = True) -> Optional[Dict[str, Any]]:
    """Assemble one GRPO sample, or ``None`` when required files are missing."""
    svg_path = os.path.join(data_dir, "svg", f"{uid}.svg")
    json_path = os.path.join(data_dir, "json", f"{uid}.json")
    image_path = os.path.join(data_dir, "png", f"{uid}.png")

    if not os.path.isfile(svg_path):
        logger.debug("skip %s: missing %s", uid, svg_path)
        return None
    if not os.path.isfile(json_path):
        logger.debug("skip %s: missing %s", uid, json_path)
        return None
    if require_image and not os.path.isfile(image_path):
        logger.debug("skip %s: missing %s", uid, image_path)
        return None

    try:
        with open(svg_path, "r", encoding="utf-8") as handle:
            damaged_svg = handle.read()
        with open(json_path, "r", encoding="utf-8") as handle:
            patch = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("skip %s: %s: %s", uid, type(exc).__name__, exc)
        return None

    operations = patch.get("operations") or []
    replacements = [str(op.get("replacement", "")).strip() for op in operations]
    replacements = [d for d in replacements if d]
    if not replacements:
        logger.debug("skip %s: no replacement operations", uid)
        return None

    skeleton = str(patch.get("skeleton_svg", "") or "").strip()
    if not skeleton:
        skeleton = " ".join(
            str(op.get("skeleton_svg", "")).strip() for op in operations
        ).strip()

    sample: Dict[str, Any] = {
        "uid": uid,
        "svg_path": svg_path,
        "json_path": json_path,
        "image_path": image_path if os.path.isfile(image_path) else None,
        "damaged_svg": damaged_svg,
        "gt_missing_contour_svg": wrap_paths(replacements),
        "gt_skeleton_svg": wrap_paths([skeleton]) if skeleton else "",
        "gt_full_svg": append_paths_to_svg(damaged_svg, replacements),
        "gt_skeleton_path_data": [skeleton] if skeleton else [],
        "gt_contour_path_data": replacements,
        "n_operations": len(replacements),
    }
    sample.update(decode_uid(uid))
    return sample


def read_meta_ids(data_dir: str, split: str = "train") -> List[str]:
    """Read the ``id`` column of ``{split}_meta.csv``."""
    meta_path = os.path.join(data_dir, f"{split}_meta.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta file not found: {meta_path}")

    ids: List[str] = []
    with open(meta_path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        try:
            id_index = header.index("id")
        except ValueError:
            id_index = 0
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) > id_index and fields[id_index]:
                ids.append(fields[id_index])
    return ids


def iter_samples(
    data_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    require_image: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield GRPO samples, skipping any uid whose files are incomplete."""
    ids = read_meta_ids(data_dir, split)
    if shuffle_seed is not None:
        import random

        random.Random(shuffle_seed).shuffle(ids)

    yielded = 0
    for uid in ids:
        if limit is not None and yielded >= limit:
            return
        sample = build_sample(data_dir, uid, require_image=require_image)
        if sample is None:
            continue
        yielded += 1
        yield sample


def load_samples(
    data_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    require_image: bool = True,
) -> List[Dict[str, Any]]:
    """Eagerly collect samples into a list."""
    return list(iter_samples(data_dir, split, limit, shuffle_seed, require_image))


def oracle_completion(sample: Dict[str, Any], tag_style: str = "tags") -> str:
    """Build the completion a perfect model would emit for ``sample``.

    Used by the reward smoke test and the debug visualiser to confirm that the
    ground truth itself scores near the maximum reward.
    """
    skeleton = " ".join(sample.get("gt_skeleton_path_data") or [])
    contour = " ".join(sample.get("gt_contour_path_data") or [])
    if tag_style == "markers":
        return (
            f"[197000 SKEL_S]{skeleton}[197001 SKEL_E]"
            f"[197002 REPL_S]{contour}[197003 REPL_E]"
        )
    return f"<skeleton>{skeleton}</skeleton><contour>{contour}</contour>"
