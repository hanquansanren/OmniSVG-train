"""Parse a model completion into its skeleton and contour SVG regions.

The stage-2 skeleton-CoT target produced by ``utils/dataset.py`` interleaves,
once per missing stroke group::

    [197000 SKEL_S] <skeleton path tokens> [197001 SKEL_E]
    [197002 REPL_S] <replacement path tokens> [197003 REPL_E]

Depending on where the reward is evaluated, a completion arrives in one of
three shapes, and all three are accepted here:

1. A list/tensor of token ids straight out of ``model.generate``.
2. Text carrying the literal marker names, e.g.
   ``"[197000 SKEL_S] M10 10 L20 20 [197001 SKEL_E] ..."``.
3. Text carrying XML-ish tags, e.g. ``"<skeleton>...</skeleton><contour>...</contour>"``.
   This is the form used by the unit tests and the debug visualiser.

The result always exposes ``skeleton_svg`` / ``contour_svg`` as complete SVG
documents on the project's ``0 0 200 200`` canvas, so downstream rasterization
never has to care which shape the completion came in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .raster import (
    DEFAULT_VIEWBOX,
    SVGRasterizeError,
    ViewBox,
    extract_path_data,
    flatten_path_d,
    has_drawable_commands,
)

logger = logging.getLogger(__name__)

# Marker token ids, mirroring configs/tokenization.yaml and utils/config.py.
SKELETON_START_TOKEN_ID = 197000
SKELETON_END_TOKEN_ID = 197001
REPLACEMENT_START_TOKEN_ID = 197002
REPLACEMENT_END_TOKEN_ID = 197003
BOS_TOKEN_ID = 196998
EOS_TOKEN_ID = 196999
PAD_TOKEN_ID = 151643

MARKER_TOKEN_IDS = frozenset(
    {
        SKELETON_START_TOKEN_ID,
        SKELETON_END_TOKEN_ID,
        REPLACEMENT_START_TOKEN_ID,
        REPLACEMENT_END_TOKEN_ID,
    }
)

_SKELETON_TAG_RE = re.compile(
    r"<\s*(?:skeleton|skel)\s*>(.*?)<\s*/\s*(?:skeleton|skel)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_CONTOUR_TAG_RE = re.compile(
    r"<\s*(?:contour|replacement|repl)\s*>(.*?)<\s*/\s*(?:contour|replacement|repl)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SKELETON_MARKER_RE = re.compile(
    r"\[\s*(?:197000\s+)?SKEL_S\s*\](.*?)\[\s*(?:197001\s+)?SKEL_E\s*\]",
    re.IGNORECASE | re.DOTALL,
)
_CONTOUR_MARKER_RE = re.compile(
    r"\[\s*(?:197002\s+)?REPL_S\s*\](.*?)\[\s*(?:197003\s+)?REPL_E\s*\]",
    re.IGNORECASE | re.DOTALL,
)
_INT_LIST_RE = re.compile(r"^[\s\[\(]*\d[\d\s,\]\)]*$")
_INT_RE = re.compile(r"\d+")

SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class ParsedCompletion:
    """Structured view of one model completion."""

    skeleton_svg: str = ""
    contour_svg: str = ""
    valid: bool = False
    error: Optional[str] = None
    skeleton_empty: bool = True
    contour_empty: bool = True
    n_skeleton_sections: int = 0
    n_contour_sections: int = 0
    source: str = "unknown"
    skeleton_path_data: List[str] = field(default_factory=list)
    contour_path_data: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skeleton_svg": self.skeleton_svg,
            "contour_svg": self.contour_svg,
            "valid": self.valid,
            "error": self.error,
            "skeleton_empty": self.skeleton_empty,
            "contour_empty": self.contour_empty,
            "n_skeleton_sections": self.n_skeleton_sections,
            "n_contour_sections": self.n_contour_sections,
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# SVG assembly helpers
# --------------------------------------------------------------------------- #


def wrap_path_data_as_svg(
    path_data: Sequence[str],
    viewbox: ViewBox = DEFAULT_VIEWBOX,
    fill: str = "#000",
) -> str:
    """Wrap ``d`` attributes into one SVG document on the project canvas."""
    if not path_data:
        return ""
    min_x, min_y, width, height = viewbox
    paths = "".join(f'<path fill="{fill}" d="{d}"/>' for d in path_data)
    return (
        f'<svg xmlns="{SVG_NS}" viewBox="{min_x} {min_y} {width} {height}" '
        f'width="{int(width)}" height="{int(height)}">{paths}</svg>'
    )


def collect_path_data(sections: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    """Flatten section bodies into a list of parsable ``d`` attributes.

    Returns ``(path_data, error)``.  ``error`` is non-``None`` when a section
    carried content that could not be parsed as SVG path data.
    """
    path_data: List[str] = []
    for section in sections:
        body = (section or "").strip()
        if not body:
            continue
        candidates = extract_path_data(body)
        if not candidates:
            return [], "no_path_element"
        for candidate in candidates:
            if not has_drawable_commands(candidate):
                continue
            try:
                subpaths = flatten_path_d(candidate)
            except SVGRasterizeError as exc:
                return [], f"unparsable_path: {exc}"
            except Exception as exc:  # noqa: BLE001
                return [], f"unparsable_path: {type(exc).__name__}"
            if not any(points for points, _closed in subpaths):
                continue
            path_data.append(candidate)
    return path_data, None


# --------------------------------------------------------------------------- #
# Token-id decoding
# --------------------------------------------------------------------------- #

_TOKEN_DECODER: Optional["SVGTokenDecoder"] = None


class SVGTokenDecoder:
    """Turn SVG token ids back into SVG path data.

    Thin wrapper over ``tokenizer.TrainAlignedSVGTokenizer`` (the decoder used
    by ``inference.py`` with ``--use-train-tokenizer``) that returns ``d``
    attributes instead of a coloured SVG object.
    """

    def __init__(self, tokenization_config_path: str = "./configs/tokenization.yaml",
                 model_size: str = "4B") -> None:
        import torch  # noqa: F401  (imported here to keep the module import light)

        from utils.config import TokenizationConfig
        from tokenizer import TrainAlignedSVGTokenizer

        self.token_config = TokenizationConfig.from_yaml(
            tokenization_config_path, model_size=model_size
        )
        self.tokenizer = TrainAlignedSVGTokenizer(self.token_config)
        self.bos_token_id = self.token_config.bos_token_id
        self.eos_token_id = self.token_config.eos_token_id

    def decode_to_path_data(self, token_ids: Sequence[int]) -> List[str]:
        """Decode one token section into a list of ``d`` attributes."""
        import torch

        ids = [int(t) for t in token_ids if int(t) not in MARKER_TOKEN_IDS]
        ids = [t for t in ids if t not in (self.bos_token_id, self.eos_token_id, PAD_TOKEN_ID)]
        if not ids:
            return []

        wrapped = torch.tensor([[self.bos_token_id] + ids + [self.eos_token_id]], dtype=torch.long)
        xy_pairs = self.tokenizer.process_generated_tokens(wrapped)
        if len(xy_pairs) == 0:
            return []
        svg_tensors, color_tensors = self.tokenizer.raster_svg(xy_pairs)
        if not svg_tensors or not svg_tensors[0]:
            return []
        colors = list(color_tensors)
        while len(colors) < len(svg_tensors[0]):
            colors.append(self.token_config.color_token_offset + 2)
        svg_obj = self.tokenizer.apply_colors_to_svg(svg_tensors[0], colors)
        return extract_path_data(svg_obj.to_str())


def get_token_decoder(
    tokenization_config_path: str = "./configs/tokenization.yaml",
    model_size: str = "4B",
) -> Optional[SVGTokenDecoder]:
    """Lazily build the shared token decoder, or ``None`` if unavailable."""
    global _TOKEN_DECODER
    if _TOKEN_DECODER is None:
        try:
            _TOKEN_DECODER = SVGTokenDecoder(tokenization_config_path, model_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SVG token decoder unavailable: %s: %s", type(exc).__name__, exc)
            return None
    return _TOKEN_DECODER


def split_token_sections(
    token_ids: Sequence[int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Split a token sequence into skeleton and replacement sections.

    Mirrors ``inference.py::extract_skeleton_and_replacement_sections`` but
    tolerates truncated completions: a section left open by a missing end
    marker is still returned.
    """
    skeleton_sections: List[List[int]] = []
    contour_sections: List[List[int]] = []
    current: List[int] = []
    current_kind: Optional[str] = None

    def close_section() -> None:
        nonlocal current, current_kind
        if current_kind == "skeleton":
            skeleton_sections.append(current)
        elif current_kind == "contour":
            contour_sections.append(current)
        current = []
        current_kind = None

    for raw in token_ids:
        token = int(raw)
        if token in (PAD_TOKEN_ID, BOS_TOKEN_ID):
            continue
        if token == EOS_TOKEN_ID:
            break
        if token == SKELETON_START_TOKEN_ID:
            close_section()
            current_kind = "skeleton"
            continue
        if token == REPLACEMENT_START_TOKEN_ID:
            close_section()
            current_kind = "contour"
            continue
        if token in (SKELETON_END_TOKEN_ID, REPLACEMENT_END_TOKEN_ID):
            close_section()
            continue
        if current_kind is not None:
            current.append(token)
    close_section()

    return skeleton_sections, contour_sections


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _coerce_completion(completion: Any) -> Any:
    """Normalise TRL-style completions into a token-id list or a plain string."""
    if completion is None:
        return ""

    # Conversational format: [{"role": ..., "content": ...}, ...]
    if isinstance(completion, (list, tuple)) and completion and isinstance(completion[0], dict):
        return "".join(str(turn.get("content", "")) for turn in completion)

    if isinstance(completion, dict):
        return str(completion.get("content", ""))

    if isinstance(completion, (list, tuple)):
        return [int(t) for t in completion]

    if isinstance(completion, str):
        return completion

    # torch.Tensor / np.ndarray
    tolist = getattr(completion, "tolist", None)
    if callable(tolist):
        flat = tolist()
        while isinstance(flat, list) and len(flat) == 1 and isinstance(flat[0], list):
            flat = flat[0]
        if isinstance(flat, list):
            return [int(t) for t in flat]

    return str(completion)


def _parse_token_ids(
    token_ids: Sequence[int],
    viewbox: ViewBox,
    tokenization_config_path: str,
    model_size: str,
) -> ParsedCompletion:
    result = ParsedCompletion(source="token_ids")

    has_marker = any(int(t) in MARKER_TOKEN_IDS for t in token_ids)
    if not has_marker:
        result.error = "missing_cot_markers"
        return result

    skeleton_sections, contour_sections = split_token_sections(token_ids)
    result.n_skeleton_sections = len(skeleton_sections)
    result.n_contour_sections = len(contour_sections)

    if not contour_sections:
        result.error = "missing_contour_section"
        return result

    decoder = get_token_decoder(tokenization_config_path, model_size)
    if decoder is None:
        result.error = "token_decoder_unavailable"
        return result

    def decode_all(sections: Sequence[Sequence[int]]) -> List[str]:
        path_data: List[str] = []
        for section in sections:
            try:
                path_data.extend(decoder.decode_to_path_data(section))
            except Exception as exc:  # noqa: BLE001 - never break the reward loop
                logger.debug("token section decode failed: %s: %s", type(exc).__name__, exc)
        return path_data

    result.skeleton_path_data = decode_all(skeleton_sections)
    result.contour_path_data = decode_all(contour_sections)

    if not result.contour_path_data:
        result.error = "undecodable_contour"
        return result

    result.skeleton_svg = wrap_path_data_as_svg(result.skeleton_path_data, viewbox)
    result.contour_svg = wrap_path_data_as_svg(result.contour_path_data, viewbox)
    result.skeleton_empty = not result.skeleton_path_data
    result.contour_empty = False
    result.valid = True
    return result


def _parse_tagged_text(
    text: str,
    viewbox: ViewBox,
) -> ParsedCompletion:
    skeleton_sections = _SKELETON_TAG_RE.findall(text)
    contour_sections = _CONTOUR_TAG_RE.findall(text)
    source = "tags"

    if not skeleton_sections and not contour_sections:
        skeleton_sections = _SKELETON_MARKER_RE.findall(text)
        contour_sections = _CONTOUR_MARKER_RE.findall(text)
        source = "markers"

    result = ParsedCompletion(source=source)
    result.n_skeleton_sections = len(skeleton_sections)
    result.n_contour_sections = len(contour_sections)

    if not skeleton_sections:
        result.error = "missing_skeleton_tag"
        return result
    if not contour_sections:
        result.error = "missing_contour_tag"
        return result

    skeleton_data, skeleton_error = collect_path_data(skeleton_sections)
    if skeleton_error:
        result.error = f"skeleton_{skeleton_error}"
        return result

    contour_data, contour_error = collect_path_data(contour_sections)
    if contour_error:
        result.error = f"contour_{contour_error}"
        return result
    if not contour_data:
        result.error = "empty_contour"
        return result

    result.skeleton_path_data = skeleton_data
    result.contour_path_data = contour_data
    result.skeleton_svg = wrap_path_data_as_svg(skeleton_data, viewbox)
    result.contour_svg = wrap_path_data_as_svg(contour_data, viewbox)
    result.skeleton_empty = not skeleton_data
    result.contour_empty = False
    result.valid = True
    return result


def parse_model_output(
    text: Any,
    viewbox: ViewBox = DEFAULT_VIEWBOX,
    tokenization_config_path: str = "./configs/tokenization.yaml",
    model_size: str = "4B",
) -> ParsedCompletion:
    """Extract the skeleton and contour SVGs from one model completion.

    Args:
        text: A completion as token ids, marker text, tagged text, or a
            conversational message list.
        viewbox: User-space box written into the returned SVG documents.
        tokenization_config_path: Config used to decode SVG token ids.
        model_size: Model size key for the tokenization config.

    Returns:
        :class:`ParsedCompletion`.  ``valid`` is ``False`` when a region tag is
        missing, when no ``<path>`` could be recovered, or when path data does
        not parse.  An explicitly empty skeleton region is *valid* but sets
        ``skeleton_empty=True`` so the caller can apply the skeleton gate.
    """
    payload = _coerce_completion(text)

    if isinstance(payload, list):
        return _parse_token_ids(payload, viewbox, tokenization_config_path, model_size)

    if not payload or not payload.strip():
        return ParsedCompletion(error="empty_completion", source="text")

    # Some launchers stringify token-id lists; recover them rather than fail.
    if _INT_LIST_RE.match(payload.strip()):
        token_ids = [int(m.group()) for m in _INT_RE.finditer(payload)]
        if any(t in MARKER_TOKEN_IDS for t in token_ids):
            return _parse_token_ids(
                token_ids, viewbox, tokenization_config_path, model_size
            )

    return _parse_tagged_text(payload, viewbox)
