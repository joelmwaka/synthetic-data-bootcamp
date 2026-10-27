"""Annotation overlays and false-colour maps for display (pure PIL/numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np
from aieng.syn_data.image.generate import night
from matplotlib import colormaps
from PIL import Image, ImageDraw


BOX_COLOUR = (255, 221, 0)
DARK_BOX_COLOUR = (255, 64, 64)
EGO_COLOUR = (0, 229, 255)
TRACK_COLOUR = (255, 255, 255)
MASK_COLOUR = np.array([0, 200, 120], dtype=np.float32)
HORIZON_COLOUR = (255, 0, 255)


def draw_overlay(
    image: Image.Image,
    *,
    boxes: list[dict[str, Any]] | None = None,
    box_scale: tuple[float, float] = (1.0, 1.0),
    visible: list[bool] | None = None,
    annotation: dict[str, Any] | None = None,
    mask: Image.Image | None = None,
    horizon_row: float | None = None,
    show_labels: bool = True,
) -> Image.Image:
    """Draw boxes, track polylines, a mask tint, and the horizon on a copy.

    ``boxes`` and ``annotation`` are in source-image pixels; ``box_scale`` maps
    them onto ``image``. With ``visible``, dark boxes are drawn red.
    """
    out = image.convert("RGB")
    sx, sy = box_scale
    if mask is not None:
        alpha = np.asarray(mask.convert("L").resize(out.size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        arr = np.asarray(out, dtype=np.float32)
        arr = arr * (1 - 0.45 * alpha[..., None]) + MASK_COLOUR * (0.45 * alpha[..., None])
        out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    width = max(2, round(out.size[0] / 400))
    if annotation:
        for track in night.track_polylines(annotation):
            colour = EGO_COLOUR if track["position"] == "ego" else TRACK_COLOUR
            for side in ("left", "right"):
                pts = [(float(x) * sx, float(y) * sy) for x, y in track[side]]
                if len(pts) >= 2:
                    draw.line(pts, fill=colour, width=width)
    if horizon_row is not None:
        y = horizon_row * sy
        draw.line([(0, y), (out.size[0], y)], fill=HORIZON_COLOUR, width=max(1, width // 2))
    for i, box in enumerate(boxes or []):
        x1, y1, x2, y2 = box["bbox_xyxy"]
        is_dark = visible is not None and i < len(visible) and not visible[i]
        colour = DARK_BOX_COLOUR if is_dark else BOX_COLOUR
        rect = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        draw.rectangle(rect, outline=colour, width=width + 1)
        if show_labels:
            text = f"{i}: {box['label']}"
            if box.get("kind"):
                text += f" ({box['kind']})"
            draw.text((rect[0] + 3, max(0, rect[1] - 14)), text, fill=colour)
    return out


def colourise(
    values: np.ndarray[Any, Any],
    *,
    cmap: str = "inferno",
    log: bool = False,
    eps: float = 1e-3,
) -> Image.Image:
    """Map a 2D array to an RGB image (NaN → black), robust to outliers."""
    arr = np.asarray(values, dtype=np.float64)
    if log:
        arr = np.log10(np.clip(arr, 0, None) + eps)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float64)
    if finite.any():
        lo, hi = np.quantile(arr[finite], [0.01, 0.99])
        scale = hi - lo if hi > lo else 1.0
        out[finite] = np.clip((arr[finite] - lo) / scale, 0, 1)
    rgba = colormaps[cmap](out)
    rgba[~finite] = (0, 0, 0, 1)
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))


def class_map(geometry: night.SceneGeometry) -> Image.Image:
    """Sky (dark blue) / ground (green) / vertical objects (amber)."""
    h, w = geometry.sky.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[geometry.sky] = (30, 40, 110)
    rgb[geometry.ground] = (60, 170, 90)
    rgb[~geometry.sky & ~geometry.ground] = (240, 170, 40)
    return Image.fromarray(rgb)


def thumbnail(image: Image.Image, max_side: int = 360) -> Image.Image:
    """Return a downscaled RGB copy."""
    out = image.convert("RGB")
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out
