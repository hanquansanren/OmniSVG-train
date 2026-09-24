"""SVG rasterization and binary-mask utilities for the GRPO reward pipeline.

Every SVG produced by this project lives in a ``viewBox="0 0 200 200"`` user
space (see ``utils/dataset.py::_wrap_single_path_as_svg`` and
``configs/tokenization.yaml::coordinates.bbox_size``).  All masks are rendered
into the same square canvas so that IoU between any two SVGs is meaningful.

Two rasterization backends are available:

``cairosvg``
    Preferred.  Handles fill rules, arcs and beziers exactly as the renderer
    used elsewhere in the repo (``inference.py::render_svg_to_image``).

``pil``
    Pure-python fallback that flattens path commands to polylines and draws
    them with ``PIL.ImageDraw``.  Keeps the reward pipeline importable in
    environments without cairo system libraries.

The backend is selected once per process and reused, so a predicted mask and a
ground-truth mask are always produced by the same renderer.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_CANVAS_SIZE = 256
DEFAULT_VIEWBOX = (0.0, 0.0, 200.0, 200.0)
DEFAULT_STROKE_WIDTH = 3.0
DEFAULT_ALPHA_THRESHOLD = 64
BEZIER_SAMPLES = 12
ARC_SAMPLES = 16

SVG_NS = "http://www.w3.org/2000/svg"

ViewBox = Tuple[float, float, float, float]

# A path element is a list of sub-paths; a sub-path is (points, closed).
SubPath = Tuple[List[Tuple[float, float]], bool]
PathElement = List[SubPath]


class SVGRasterizeError(RuntimeError):
    """Raised when an SVG cannot be turned into a mask."""


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #

_BACKEND: Optional[str] = None


def get_backend() -> str:
    """Return the active rasterization backend (``"cairosvg"`` or ``"pil"``)."""
    global _BACKEND
    if _BACKEND is None:
        try:
            import cairosvg  # noqa: F401

            _BACKEND = "cairosvg"
        except Exception:
            _BACKEND = "pil"
            logger.info("cairosvg unavailable, falling back to the PIL rasterizer")
    return _BACKEND


def set_backend(backend: str) -> None:
    """Force a backend.  Useful for tests and for cross-checking IoU values."""
    if backend not in ("cairosvg", "pil"):
        raise ValueError(f"unknown backend: {backend}")
    global _BACKEND
    _BACKEND = backend


# --------------------------------------------------------------------------- #
# SVG text parsing
# --------------------------------------------------------------------------- #

_PATH_D_RE = re.compile(r"<path\b[^>]*?\bd\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE | re.DOTALL)
_VIEWBOX_RE = re.compile(r"viewBox\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+(?:[eE][-+]?\d+)?|\d+\.?\d*(?:[eE][-+]?\d+)?)")
_COMMAND_CHARS = "MmZzLlHhVvCcSsQqTtAa"
_DRAW_COMMAND_RE = re.compile(r"[MmLlHhVvCcSsQqTtAa]")


def looks_like_path_data(text: str) -> bool:
    """True when ``text`` is a bare ``d`` attribute rather than an SVG document."""
    stripped = text.strip()
    if not stripped or "<" in stripped:
        return False
    return bool(re.match(r"^[Mm][\s,]*[-+.\d]", stripped))


def extract_viewbox(svg: str) -> Optional[ViewBox]:
    """Read the ``viewBox`` of an SVG document, or ``None`` when absent."""
    match = _VIEWBOX_RE.search(svg)
    if not match:
        return None
    parts = _NUMBER_RE.findall(match.group(1))
    if len(parts) != 4:
        return None
    min_x, min_y, width, height = (float(p) for p in parts)
    if width <= 0 or height <= 0:
        return None
    return (min_x, min_y, width, height)


def extract_path_data(svg: str) -> List[str]:
    """Return the ``d`` attribute of every ``<path>`` element in document order.

    Accepts a full ``<svg>`` document, a bare fragment of ``<path>`` elements or
    a single ``d`` string.
    """
    if not svg or not svg.strip():
        return []
    if looks_like_path_data(svg):
        return [svg.strip()]
    found = [match.group(1)[1:-1].strip() for match in _PATH_D_RE.finditer(svg)]
    return [d for d in found if d]


def has_drawable_commands(path_d: str) -> bool:
    """True when ``path_d`` contains at least one command that paints something."""
    return bool(_DRAW_COMMAND_RE.search(path_d or ""))


def strip_close_commands(path_d: str) -> str:
    """Remove ``Z``/``z`` commands so a path is treated as an open polyline."""
    return re.sub(r"[Zz]", " ", path_d or "").strip()


# --------------------------------------------------------------------------- #
# Path flattening (pure-python, used by the PIL backend)
# --------------------------------------------------------------------------- #


def _tokenize_path_d(path_d: str) -> List[object]:
    tokens: List[object] = []
    index = 0
    length = len(path_d)
    while index < length:
        char = path_d[index]
        if char in _COMMAND_CHARS:
            tokens.append(char)
            index += 1
        elif char in " ,\t\r\n":
            index += 1
        else:
            match = _NUMBER_RE.match(path_d, index)
            if match is None:
                raise SVGRasterizeError(
                    f"unparsable character {char!r} at offset {index} of path data"
                )
            tokens.append(float(match.group()))
            index = match.end()
    return tokens


def _sample_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    samples: int,
) -> List[Tuple[float, float]]:
    points = []
    for step in range(1, samples + 1):
        t = step / samples
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        points.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            )
        )
    return points


def _sample_quadratic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    samples: int,
) -> List[Tuple[float, float]]:
    points = []
    for step in range(1, samples + 1):
        t = step / samples
        u = 1.0 - t
        a, b, c = u * u, 2 * u * t, t * t
        points.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0],
                a * p0[1] + b * p1[1] + c * p2[1],
            )
        )
    return points


def _sample_arc(
    start: Tuple[float, float],
    rx: float,
    ry: float,
    rotation_deg: float,
    large_arc: bool,
    sweep: bool,
    end: Tuple[float, float],
    samples: int,
) -> List[Tuple[float, float]]:
    """Endpoint -> center parameterization of an SVG elliptical arc (spec F.6.5)."""
    x1, y1 = start
    x2, y2 = end
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [end]

    phi = math.radians(rotation_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    if denominator <= 0:
        return [end]
    numerator = max(rx * rx * ry * ry - denominator, 0.0)
    coefficient = math.sqrt(numerator / denominator)
    if large_arc == sweep:
        coefficient = -coefficient
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    def angle_between(ux: float, uy: float, vx: float, vy: float) -> float:
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if norm == 0:
            return 0.0
        cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / norm))
        angle = math.acos(cosine)
        return -angle if (ux * vy - uy * vx) < 0 else angle

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = angle_between(1.0, 0.0, ux, uy)
    delta = angle_between(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    points = []
    for step in range(1, samples + 1):
        theta = theta1 + delta * step / samples
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        points.append(
            (
                cos_phi * rx * cos_t - sin_phi * ry * sin_t + cx,
                sin_phi * rx * cos_t + cos_phi * ry * sin_t + cy,
            )
        )
    return points


def flatten_path_d(
    path_d: str,
    bezier_samples: int = BEZIER_SAMPLES,
    arc_samples: int = ARC_SAMPLES,
) -> PathElement:
    """Flatten a ``d`` attribute into polyline sub-paths in user coordinates."""
    tokens = _tokenize_path_d(path_d)
    subpaths: PathElement = []
    points: List[Tuple[float, float]] = []
    closed = False

    current = (0.0, 0.0)
    subpath_start = (0.0, 0.0)
    previous_cubic_control: Optional[Tuple[float, float]] = None
    previous_quad_control: Optional[Tuple[float, float]] = None
    command: Optional[str] = None

    index = 0
    total = len(tokens)

    def flush() -> None:
        nonlocal points, closed
        if len(points) >= 1:
            subpaths.append((points, closed))
        points = []
        closed = False

    def take(count: int) -> List[float]:
        nonlocal index
        if index + count > total:
            raise SVGRasterizeError("truncated path data")
        values = [float(v) for v in tokens[index : index + count]]  # type: ignore[arg-type]
        index += count
        return values

    while index < total:
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command in "Zz":
                if points:
                    closed = True
                    flush()
                current = subpath_start
                previous_cubic_control = None
                previous_quad_control = None
                continue
            if index >= total or isinstance(tokens[index], str):
                continue
        if command is None:
            raise SVGRasterizeError("path data does not start with a command")

        relative = command.islower()
        upper = command.upper()

        if upper == "M":
            x, y = take(2)
            target = (current[0] + x, current[1] + y) if relative else (x, y)
            flush()
            current = subpath_start = target
            points = [current]
            previous_cubic_control = previous_quad_control = None
            # Subsequent coordinate pairs of an M command are implicit line-tos.
            command = "l" if relative else "L"
            continue

        if not points:
            # A draw command without a preceding move-to starts at the origin.
            points = [current]

        if upper == "L":
            x, y = take(2)
            current = (current[0] + x, current[1] + y) if relative else (x, y)
            points.append(current)
            previous_cubic_control = previous_quad_control = None
        elif upper == "H":
            (x,) = take(1)
            current = (current[0] + x, current[1]) if relative else (x, current[1])
            points.append(current)
            previous_cubic_control = previous_quad_control = None
        elif upper == "V":
            (y,) = take(1)
            current = (current[0], current[1] + y) if relative else (current[0], y)
            points.append(current)
            previous_cubic_control = previous_quad_control = None
        elif upper == "C":
            x1, y1, x2, y2, x, y = take(6)
            if relative:
                c1 = (current[0] + x1, current[1] + y1)
                c2 = (current[0] + x2, current[1] + y2)
                end = (current[0] + x, current[1] + y)
            else:
                c1, c2, end = (x1, y1), (x2, y2), (x, y)
            points.extend(_sample_cubic(current, c1, c2, end, bezier_samples))
            current, previous_cubic_control = end, c2
            previous_quad_control = None
        elif upper == "S":
            x2, y2, x, y = take(4)
            if relative:
                c2 = (current[0] + x2, current[1] + y2)
                end = (current[0] + x, current[1] + y)
            else:
                c2, end = (x2, y2), (x, y)
            if previous_cubic_control is None:
                c1 = current
            else:
                c1 = (
                    2 * current[0] - previous_cubic_control[0],
                    2 * current[1] - previous_cubic_control[1],
                )
            points.extend(_sample_cubic(current, c1, c2, end, bezier_samples))
            current, previous_cubic_control = end, c2
            previous_quad_control = None
        elif upper == "Q":
            x1, y1, x, y = take(4)
            if relative:
                c1 = (current[0] + x1, current[1] + y1)
                end = (current[0] + x, current[1] + y)
            else:
                c1, end = (x1, y1), (x, y)
            points.extend(_sample_quadratic(current, c1, end, bezier_samples))
            current, previous_quad_control = end, c1
            previous_cubic_control = None
        elif upper == "T":
            x, y = take(2)
            end = (current[0] + x, current[1] + y) if relative else (x, y)
            if previous_quad_control is None:
                c1 = current
            else:
                c1 = (
                    2 * current[0] - previous_quad_control[0],
                    2 * current[1] - previous_quad_control[1],
                )
            points.extend(_sample_quadratic(current, c1, end, bezier_samples))
            current, previous_quad_control = end, c1
            previous_cubic_control = None
        elif upper == "A":
            rx, ry, rotation, large_arc, sweep, x, y = take(7)
            end = (current[0] + x, current[1] + y) if relative else (x, y)
            points.extend(
                _sample_arc(
                    current, rx, ry, rotation, bool(large_arc), bool(sweep), end, arc_samples
                )
            )
            current = end
            previous_cubic_control = previous_quad_control = None
        else:
            raise SVGRasterizeError(f"unsupported path command: {command}")

    flush()
    return subpaths


# --------------------------------------------------------------------------- #
# Rasterization
# --------------------------------------------------------------------------- #


def build_canonical_svg(
    path_data: Sequence[str],
    viewbox: ViewBox,
    canvas_size: int,
    mode: str,
    stroke_width_user: float,
) -> str:
    """Wrap ``path_data`` into a single-colour SVG document on a fixed canvas."""
    min_x, min_y, width, height = viewbox
    if mode == "fill":
        style = 'fill="#000000" fill-rule="evenodd" stroke="none"'
    else:
        style = (
            'fill="none" stroke="#000000" '
            f'stroke-width="{stroke_width_user:.6f}" '
            'stroke-linecap="round" stroke-linejoin="round"'
        )
    paths = "".join(f'<path d="{d}"/>' for d in path_data)
    return (
        f'<svg xmlns="{SVG_NS}" '
        f'viewBox="{min_x} {min_y} {width} {height}" '
        f'width="{canvas_size}" height="{canvas_size}">'
        f"<g {style}>{paths}</g>"
        "</svg>"
    )


def _rasterize_with_cairosvg(
    svg_document: str,
    canvas_size: int,
    alpha_threshold: int,
) -> np.ndarray:
    import cairosvg
    from PIL import Image
    import io

    png_bytes = cairosvg.svg2png(
        bytestring=svg_document.encode("utf-8"),
        output_width=canvas_size,
        output_height=canvas_size,
        background_color="transparent",
    )
    with Image.open(io.BytesIO(png_bytes)) as image:
        alpha = np.array(image.convert("RGBA"))[:, :, 3]
    return alpha >= alpha_threshold


PIL_SUPERSAMPLE = 2


def _rasterize_with_pil(
    path_elements: Sequence[PathElement],
    viewbox: ViewBox,
    canvas_size: int,
    mode: str,
    stroke_width_px: float,
    supersample: int = PIL_SUPERSAMPLE,
) -> np.ndarray:
    """Draw flattened paths with ``PIL.ImageDraw``.

    ``ImageDraw`` fills boundary pixels inclusively while cairo samples pixel
    centres, so drawing is done on a ``supersample``-times finer grid with a
    half-pixel shift and then thresholded at 50% coverage.  That keeps the two
    backends within a few percent IoU of each other.
    """
    from PIL import Image, ImageDraw

    min_x, min_y, width, height = viewbox
    fine_size = canvas_size * supersample
    scale_x = fine_size / width
    scale_y = fine_size / height

    def to_canvas(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [
            ((x - min_x) * scale_x - 0.5, (y - min_y) * scale_y - 0.5) for x, y in points
        ]

    def downsample(fine: np.ndarray) -> np.ndarray:
        if supersample == 1:
            return fine
        blocks = fine.reshape(canvas_size, supersample, canvas_size, supersample)
        return blocks.mean(axis=(1, 3)) >= 0.5

    if mode == "fill":
        # Even-odd within a path element, union across path elements.
        accumulator = np.zeros((fine_size, fine_size), dtype=bool)
        for element in path_elements:
            element_mask = np.zeros((fine_size, fine_size), dtype=bool)
            for points, _closed in element:
                if len(points) < 3:
                    continue
                layer = Image.new("1", (fine_size, fine_size), 0)
                ImageDraw.Draw(layer).polygon(to_canvas(points), fill=1)
                element_mask ^= np.array(layer, dtype=bool)
            accumulator |= element_mask
        return downsample(accumulator)

    line_width = max(1, int(round(stroke_width_px * supersample)))
    canvas = Image.new("1", (fine_size, fine_size), 0)
    draw = ImageDraw.Draw(canvas)
    radius = line_width / 2.0
    for element in path_elements:
        for points, closed in element:
            canvas_points = to_canvas(points)
            if closed and len(canvas_points) >= 2:
                canvas_points = canvas_points + [canvas_points[0]]
            if len(canvas_points) >= 2:
                draw.line(canvas_points, fill=1, width=line_width, joint="curve")
            elif canvas_points:
                x, y = canvas_points[0]
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1)
    return downsample(np.array(canvas, dtype=bool))


def empty_mask(canvas_size: int = DEFAULT_CANVAS_SIZE) -> np.ndarray:
    """An all-false mask of the standard shape."""
    return np.zeros((canvas_size, canvas_size), dtype=bool)


def rasterize_svg_to_mask(
    svg: Optional[str],
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    mode: str = "fill",
    stroke_width: float = DEFAULT_STROKE_WIDTH,
    viewbox: Optional[ViewBox] = None,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    drop_close: bool = False,
    raise_on_error: bool = False,
) -> np.ndarray:
    """Render an SVG into a boolean mask of shape ``(canvas_size, canvas_size)``.

    Args:
        svg: Full ``<svg>`` document, a fragment of ``<path>`` elements, or a
            bare ``d`` attribute.  ``None`` and empty strings give an empty mask.
        canvas_size: Side length of the square output mask, in pixels.
        mode: ``"fill"`` paints the path interiors (used for contours),
            ``"stroke"`` paints the outlines only (used for skeletons).
        stroke_width: Stroke width in *canvas pixels*; converted to user units
            internally so that the rendered thickness is canvas-independent.
        viewbox: User-space box mapped onto the canvas.  Defaults to the
            document's own ``viewBox``, then to ``0 0 200 200``.
        alpha_threshold: Alpha value above which a pixel counts as painted
            (cairosvg backend only).
        drop_close: Strip ``Z``/``z`` commands before rendering.  Skeletons are
            open polylines, but the SVG tokenizer round-trip can append a close
            command; dropping it keeps predicted and ground-truth strokes
            comparable.
        raise_on_error: Re-raise :class:`SVGRasterizeError` instead of returning
            an empty mask.  Training code should leave this at ``False``.

    Returns:
        ``np.ndarray`` of dtype ``bool``.
    """
    if mode not in ("fill", "stroke"):
        raise ValueError(f"mode must be 'fill' or 'stroke', got {mode!r}")

    if not svg or not svg.strip():
        return empty_mask(canvas_size)

    try:
        resolved_viewbox = viewbox or extract_viewbox(svg) or DEFAULT_VIEWBOX
        path_data = extract_path_data(svg)
        if drop_close:
            path_data = [strip_close_commands(d) for d in path_data]
        path_data = [d for d in path_data if has_drawable_commands(d)]
        if not path_data:
            return empty_mask(canvas_size)

        # stroke_width is expressed in canvas pixels; convert to user units.
        scale = canvas_size / max(resolved_viewbox[2], resolved_viewbox[3])
        stroke_width_user = float(stroke_width) / max(scale, 1e-9)

        if get_backend() == "cairosvg":
            document = build_canonical_svg(
                path_data, resolved_viewbox, canvas_size, mode, stroke_width_user
            )
            mask = _rasterize_with_cairosvg(document, canvas_size, alpha_threshold)
        else:
            elements = [flatten_path_d(d) for d in path_data]
            mask = _rasterize_with_pil(
                elements, resolved_viewbox, canvas_size, mode, float(stroke_width)
            )

        if mask.shape != (canvas_size, canvas_size):
            raise SVGRasterizeError(f"unexpected mask shape {mask.shape}")
        return mask.astype(bool, copy=False)

    except Exception as exc:  # noqa: BLE001 - a bad SVG must never stop training
        if raise_on_error:
            if isinstance(exc, SVGRasterizeError):
                raise
            raise SVGRasterizeError(str(exc)) from exc
        logger.debug("rasterize_svg_to_mask failed (%s): %s", type(exc).__name__, exc)
        return empty_mask(canvas_size)


# --------------------------------------------------------------------------- #
# Mask operations
# --------------------------------------------------------------------------- #


def mask_area(mask: np.ndarray) -> int:
    """Number of painted pixels."""
    return int(np.count_nonzero(mask))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over union of two binary masks.

    When the union is empty the masks are either both empty (perfect agreement,
    ``1.0``) or of mismatched shape (``0.0``).
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        logger.debug("mask_iou shape mismatch: %s vs %s", a.shape, b.shape)
        return 0.0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(a & b))
    return intersection / union


def _disk(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius + 1e-9


def dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Grow a mask by a disk of ``radius`` pixels.

    Skeletons are one pixel wide, so a small tolerance is required before their
    IoU carries any signal.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=_disk(radius))
    except Exception:
        pass
    try:
        from skimage.morphology import dilation

        return dilation(mask, _disk(radius)).astype(bool)
    except Exception:
        logger.debug("no dilation backend available, returning mask unchanged")
        return mask


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a filled mask down to its one-pixel-wide medial axis."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask).astype(bool)
    except Exception as exc:  # noqa: BLE001
        logger.debug("skeletonize unavailable (%s), returning mask unchanged", exc)
        return mask


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """Write a boolean mask as a black-on-white PNG (painted pixels are black)."""
    from PIL import Image

    array = np.asarray(mask, dtype=bool)
    image = Image.fromarray(np.where(array, 0, 255).astype(np.uint8), mode="L")
    image.save(path)


def overlay_masks_png(
    masks: Sequence[np.ndarray],
    colors: Sequence[Tuple[int, int, int]],
    path: str,
) -> None:
    """Write several masks into one RGB PNG for visual comparison."""
    from PIL import Image

    if not masks:
        raise ValueError("no masks to overlay")
    height, width = np.asarray(masks[0]).shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        array = np.asarray(mask, dtype=bool)
        for channel in range(3):
            canvas[:, :, channel] = np.where(
                array,
                (canvas[:, :, channel].astype(int) * color[channel] // 255).astype(np.uint8),
                canvas[:, :, channel],
            )
    Image.fromarray(canvas).save(path)
