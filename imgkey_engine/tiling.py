from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from .types import CancelCallback, KeySettings, ProgressCallback


def _expanded_mask_bounds(mask: np.ndarray, margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    h, w = shape
    pad = max(0, int(margin))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def _tile_intersects_crop(core_y: slice, core_x: slice, crop: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = crop
    return core_y.start < y1 and core_y.stop > y0 and core_x.start < x1 and core_x.stop > x0


def _iter_tiles(
    h: int,
    w: int,
    settings: KeySettings,
    edge_radius: int,
    extra_overlap: int = 0,
) -> Iterator[tuple[slice, slice, slice, slice]]:
    tile_size = max(1, int(settings.tile_size))
    if not settings.use_tiling or max(h, w) <= tile_size:
        yield slice(0, h), slice(0, w), slice(0, h), slice(0, w)
        return
    overlap = max(int(settings.tile_overlap), int(edge_radius) * 4, int(extra_overlap), 0)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            read_y = slice(max(0, y0 - overlap), min(h, y1 + overlap))
            read_x = slice(max(0, x0 - overlap), min(w, x1 + overlap))
            yield read_y, read_x, slice(y0, y1), slice(x0, x1)


def _odd_kernel_from_radius(radius: float) -> int:
    if radius <= 0:
        return 0
    k = int(round(radius * 2.0 + 1.0))
    return k + 1 if k % 2 == 0 else k


def _ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _effective_edge_radius(settings: KeySettings) -> int:
    if settings.edge_refine_radius > 0:
        return max(1, int(settings.edge_refine_radius))
    return max(2, int(round(max(0.0, float(settings.edge_blur)) * 4.0 + 1.0)))


def _screen_model_radius_for_shape(shape: tuple[int, int]) -> int:
    h, w = shape
    return int(np.clip(max(int(h), int(w)) // 18, 24, 181))


def _normalized_crop(crop: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in crop)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("full_res_crop must be (x0, y0, x1, y1) with positive area")
    return x0, y0, x1, y1


def _report(callback: ProgressCallback | None, value: float, stage: str) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), stage)


def _raise_if_cancelled(cancel_callback: CancelCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError("Processing cancelled")
