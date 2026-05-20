from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .color_math import _smoothstep
from .tiling import _ellipse_kernel, _raise_if_cancelled, _report, _screen_model_radius_for_shape
from .types import CancelCallback, KeySettings, ProgressCallback


DEFAULT_MAX_FULL_RES_SCREEN_PLATE_PIXELS = 4_000_000
DEFAULT_LOW_RES_MAX_SIDE = 512


@dataclass(slots=True)
class ScreenPlateRGB:
    """Low-frequency screen plate with capped full-resolution storage.

    ``full_res_rgb`` is populated only when the caller's cap allows it. Large
    images retain a low-resolution uint8 plate and resolve requested tiles on
    demand without storing a full HxWx3 plate.
    """

    source_shape: tuple[int, int]
    fallback_rgb: tuple[int, int, int]
    low_res_rgb: np.ndarray | None = None
    full_res_rgb: np.ndarray | None = None
    low_res_valid: np.ndarray | None = None

    @property
    def is_full_res_retained(self) -> bool:
        return self.full_res_rgb is not None

    def debug_image(self) -> np.ndarray:
        if self.full_res_rgb is not None:
            return self.full_res_rgb.copy()
        if self.low_res_rgb is not None:
            return self.low_res_rgb.copy()
        return np.broadcast_to(np.asarray(self.fallback_rgb, dtype=np.uint8).reshape(1, 1, 3), (32, 32, 3)).copy()

    def resolve(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Resolve a full plate or ``(x0, y0, x1, y1)`` tile as uint8 RGB."""

        h, w = self.source_shape
        if region is None:
            return self.resolve_tile(slice(0, h), slice(0, w))
        x0, y0, x1, y1 = region
        return self.resolve_tile(slice(y0, y1), slice(x0, x1))

    def resolve_tile(self, y_slice: slice, x_slice: slice) -> np.ndarray:
        h, w = self.source_shape
        y0, y1 = _slice_bounds(y_slice, h)
        x0, x1 = _slice_bounds(x_slice, w)
        out_h = max(0, y1 - y0)
        out_w = max(0, x1 - x0)
        if out_h == 0 or out_w == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if self.full_res_rgb is not None:
            return self.full_res_rgb[y0:y1, x0:x1].copy()
        fallback = np.asarray(self.fallback_rgb, dtype=np.uint8).reshape(1, 1, 3)
        if self.low_res_rgb is None:
            return np.broadcast_to(fallback, (out_h, out_w, 3)).copy()

        low = np.asarray(self.low_res_rgb, dtype=np.uint8)
        low_h, low_w = low.shape[:2]
        if low_h <= 1 and low_w <= 1:
            return np.broadcast_to(low.reshape(1, 1, 3), (out_h, out_w, 3)).copy()

        xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) * (low_w / max(float(w), 1.0)) - 0.5
        ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * (low_h / max(float(h), 1.0)) - 0.5
        map_x = np.broadcast_to(xs.reshape(1, out_w), (out_h, out_w)).astype(np.float32, copy=False)
        map_y = np.broadcast_to(ys.reshape(out_h, 1), (out_h, out_w)).astype(np.float32, copy=False)
        return cv2.remap(low, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build_screen_plate_rgb(
    rgb_u8: np.ndarray,
    background_candidates: np.ndarray | None,
    fallback_color: tuple[int, int, int] | np.ndarray,
    *,
    max_full_res_pixels: int = DEFAULT_MAX_FULL_RES_SCREEN_PLATE_PIXELS,
    low_res_max_side: int = DEFAULT_LOW_RES_MAX_SIDE,
) -> ScreenPlateRGB:
    rgb = _ensure_rgb_u8(rgb_u8)
    h, w = rgb.shape[:2]
    fallback = _sanitize_rgb_tuple(fallback_color)
    candidates = _mask_to_bool(background_candidates, (h, w), "background_candidates")
    if candidates is None:
        candidates = np.ones((h, w), dtype=bool)

    low_h, low_w = _low_res_shape(h, w, low_res_max_side)
    fallback_arr = np.asarray(fallback, dtype=np.float32).reshape(1, 1, 3)
    if h == 0 or w == 0 or low_h == 0 or low_w == 0:
        return ScreenPlateRGB(source_shape=(h, w), fallback_rgb=fallback)

    sums, counts = _accumulate_low_res_plate(rgb, candidates, (low_h, low_w))
    expected_per_cell = max(1.0, (h * w) / float(max(1, low_h * low_w)))
    valid = counts >= max(1.0, expected_per_cell * 0.025)
    low = np.broadcast_to(fallback_arr, (low_h, low_w, 3)).copy()
    if np.any(valid):
        for channel in range(3):
            value = np.divide(
                sums[:, :, channel],
                np.maximum(counts, 1.0),
                out=np.full((low_h, low_w), fallback[channel], dtype=np.float32),
                where=valid,
            )
            low[:, :, channel] = np.where(valid, value, low[:, :, channel])
        low = _fill_low_res_plate(low, valid, fallback)

    low = cv2.GaussianBlur(low, (0, 0), sigmaX=max(0.65, min(low_h, low_w) / 90.0), sigmaY=0.0, borderType=cv2.BORDER_REPLICATE)
    low_u8 = np.clip(np.rint(low), 0, 255).astype(np.uint8)
    full_res = None
    if h * w <= max(0, int(max_full_res_pixels)):
        full_res = cv2.resize(low_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    return ScreenPlateRGB(
        source_shape=(h, w),
        fallback_rgb=fallback,
        low_res_rgb=low_u8,
        full_res_rgb=full_res,
        low_res_valid=valid.astype(np.uint8) * 255,
    )


def _sample_screen_color(rgb: np.ndarray, settings: KeySettings) -> tuple[int, int, int]:
    key = np.asarray(settings.key_color, dtype=np.float32)
    key = np.clip(key, 0, 255)
    if not settings.auto_border_sample:
        return tuple(key.astype(np.uint8).tolist())

    border = _border_pixels(rgb, max(1, int(settings.border_sample_width), int(settings.sample_size)))
    if border.size == 0:
        return tuple(key.astype(np.uint8).tolist())
    if len(border) > 160_000:
        step = max(1, len(border) // 160_000)
        border = border[::step]

    if settings.auto_detect_key_color:
        auto_key = _auto_detect_border_screen_color(border, key)
        if auto_key is not None:
            return auto_key

    candidates = _initial_border_candidates(border, key, settings)
    if np.count_nonzero(candidates) < max(16, len(border) // 80):
        return tuple(key.astype(np.uint8).tolist())

    sampled = np.median(border[candidates].astype(np.float32), axis=0)
    if np.linalg.norm(sampled - key) > 110.0:
        return tuple(key.astype(np.uint8).tolist())
    blended = sampled * 0.70 + key * 0.30
    return tuple(np.clip(np.rint(blended), 0, 255).astype(np.uint8).tolist())


def _auto_detect_border_screen_color(border_rgb: np.ndarray, fallback_key: np.ndarray) -> tuple[int, int, int] | None:
    """Detect the dominant saturated screen color from image borders.

    UI Auto mode must be unseeded: a blue plate should not be rejected just
    because the compatibility default key is green. We therefore find the most
    common saturated border hue first, then use the median RGB in that hue band
    as the screen color. Low-saturation borders fall back to the seeded key.
    """

    if border_rgb.size == 0:
        return None
    hsv = cv2.cvtColor(border_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float32)
    val = hsv[:, 2].astype(np.float32)
    usable = (sat >= 48.0) & (val >= 42.0)
    if np.count_nonzero(usable) < max(24, len(border_rgb) // 160):
        return None

    hue = hsv[:, 0]
    weights = (sat * np.maximum(val, 1.0))[usable]
    hist = np.bincount(hue[usable].astype(np.int32), weights=weights, minlength=180)
    if float(hist.max()) <= 0.0:
        return None
    peak = int(np.argmax(hist))
    hue_delta = np.abs(hue.astype(np.int16) - peak)
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    cluster = usable & (hue_delta <= 7)
    if np.count_nonzero(cluster) < max(16, len(border_rgb) // 220):
        return None

    sampled = np.median(border_rgb[cluster].astype(np.float32), axis=0)
    fallback = np.asarray(fallback_key, dtype=np.float32)
    # If the detected cluster is almost grayscale after all, ignore it.
    if float(np.max(sampled) - np.min(sampled)) < 28.0:
        return None
    # Blend very slightly with the fallback only when both hues are already close;
    # otherwise keep the detected color unseeded for true Auto behavior.
    if np.linalg.norm(sampled - fallback) < 55.0:
        sampled = sampled * 0.85 + fallback * 0.15
    return tuple(np.clip(np.rint(sampled), 0, 255).astype(np.uint8).tolist())


def _border_pixels(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = min(width, max(1, h // 2), max(1, w // 2))
    parts = [rgb[:bw, :, :], rgb[h - bw :, :, :]]
    if h > bw * 2:
        parts.extend([rgb[bw : h - bw, :bw, :], rgb[bw : h - bw, w - bw :, :]])
    return np.concatenate([p.reshape(-1, 3) for p in parts], axis=0)


def _initial_border_candidates(border_rgb: np.ndarray, key: np.ndarray, settings: KeySettings) -> np.ndarray:
    pix = border_rgb.astype(np.float32) / 255.0
    key_n = np.clip(key / 255.0, 1e-4, 1.0)
    pix_sum = np.maximum(np.sum(pix, axis=1, keepdims=True), 1e-4)
    key_chroma = key_n / max(float(np.sum(key_n)), 1e-4)
    chroma_dist = np.linalg.norm(pix / pix_sum - key_chroma.reshape(1, 3), axis=1)
    key_channel = int(np.argmax(key_n))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key_n[key_channel] - max(key_n[other[0]], key_n[other[1]]))
    if key_dom > 0.12:
        dominance = pix[:, key_channel] - np.maximum(pix[:, other[0]], pix[:, other[1]])
        dom_ok = dominance > max(0.015, key_dom * 0.20)
    else:
        dom_ok = np.ones(len(pix), dtype=bool)
    return (chroma_dist <= max(0.05, settings.tolerance + settings.softness * 2.0)) & dom_ok


def _compute_screen_probability(
    rgb: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: KeySettings,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.empty((h, w), dtype=np.uint8)
    # Stripe pass keeps float32 RGB/HSV intermediates bounded even for large
    # stills; only the uint8 probability map is retained globally.
    stripe_rows = max(96, min(h, 512))
    stripes = list(range(0, h, stripe_rows))
    total = max(1, len(stripes))
    for index, y0 in enumerate(stripes, start=1):
        _raise_if_cancelled(cancel_callback)
        y1 = min(h, y0 + stripe_rows)
        out[y0:y1] = _compute_screen_probability_block(rgb[y0:y1], screen_color, settings)
        _report(progress_callback, 0.02 + 0.08 * (index / total), "screen probability")
    return out


def _compute_screen_probability_block(rgb: np.ndarray, screen_color: tuple[int, int, int], settings: KeySettings) -> np.ndarray:
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key = np.clip(key, 1e-4, 1.0)
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0
    total = np.maximum(r + g + b, 1e-4)
    key_chroma = key / max(float(np.sum(key)), 1e-4)
    chroma_dist = np.sqrt(
        (r / total - key_chroma[0]) ** 2
        + (g / total - key_chroma[1]) ** 2
        + (b / total - key_chroma[2]) ** 2
    )
    tol = max(0.015, float(settings.tolerance))
    soft = max(0.015, float(settings.softness))
    chroma_prob = 1.0 - _smoothstep(tol, tol + soft * 2.15, chroma_dist)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    key_hsv = cv2.cvtColor(np.asarray([[screen_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    hue = hsv[:, :, 0].astype(np.float32)
    hue_diff = np.abs(hue - float(key_hsv[0]))
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0
    sat_diff = np.abs(hsv[:, :, 1].astype(np.float32) - float(key_hsv[1])) / 255.0
    hue_score = hue_diff + sat_diff * 0.18
    hue_prob = 1.0 - _smoothstep(tol * 1.35, tol * 1.35 + soft * 2.8, hue_score)

    luma = r * 0.2126 + g * 0.7152 + b * 0.0722
    key_luma = float(key @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    brightness = 1.0 - _smoothstep(
        max(0.04, float(settings.brightness_tolerance)),
        max(0.06, float(settings.brightness_tolerance) + soft * 2.0),
        np.abs(luma - key_luma),
    )

    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        channels = [r, g, b]
        dominance = channels[key_channel] - np.maximum(channels[other[0]], channels[other[1]])
        dom_prob = _smoothstep(0.015, max(0.05, key_dom * 0.90), dominance)
        vector_prob = np.clip(chroma_prob * 0.48 + hue_prob * 0.32 + brightness * 0.20, 0.0, 1.0)
        probability = np.maximum(vector_prob, dom_prob * (0.82 + 0.18 * brightness))
    else:
        probability = np.clip(chroma_prob * 0.68 + hue_prob * 0.22 + brightness * 0.10, 0.0, 1.0)

    probability = np.clip(probability, 0.0, 1.0)
    return np.rint(probability * 255.0).astype(np.uint8)


def _estimate_screen_map(
    rgb: np.ndarray,
    known_background: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: KeySettings,
) -> np.ndarray | None:
    if not settings.local_screen_model:
        return None
    h, w = rgb.shape[:2]
    if h * w > int(settings.max_local_screen_model_pixels):
        return None
    if float(np.mean(known_background.astype(np.float32))) < 0.01:
        return None
    radius = _screen_model_radius_for_shape((h, w))
    return _estimate_screen_tile(rgb, known_background, screen_color, radius)


def _estimate_screen_tile(
    rgb_tile: np.ndarray,
    known_bg_tile: np.ndarray,
    fallback_color: tuple[int, int, int] | np.ndarray,
    radius: int,
) -> np.ndarray:
    rgb_arr = np.asarray(rgb_tile, dtype=np.uint8)
    h, w = rgb_arr.shape[:2]
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        raise ValueError("rgb_tile must have shape HxWx3")
    known = np.asarray(known_bg_tile).astype(bool, copy=False)
    if known.shape != (h, w):
        raise ValueError("known_bg_tile must match rgb_tile height/width")

    fallback = np.clip(np.rint(np.asarray(fallback_color, dtype=np.float32).reshape(3)), 0, 255)
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:, :, :] = fallback.astype(np.uint8).reshape(1, 1, 3)
    if h == 0 or w == 0 or not np.any(known):
        return out

    radius_i = max(0, int(radius))
    ksize = (radius_i * 2 + 1, radius_i * 2 + 1)
    known_f = known.astype(np.float32)
    denom = cv2.boxFilter(known_f, cv2.CV_32F, ksize, normalize=False, borderType=cv2.BORDER_REPLICATE)
    valid = denom >= 1.0
    for channel in range(3):
        src = rgb_arr[:, :, channel].astype(np.float32) * known_f
        num = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=False, borderType=cv2.BORDER_REPLICATE)
        value = np.divide(
            num,
            np.maximum(denom, 1.0),
            out=np.full((h, w), float(fallback[channel]), dtype=np.float32),
            where=valid,
        )
        out[:, :, channel] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return out


def _fill_low_res_plate(low: np.ndarray, valid: np.ndarray, fallback: tuple[int, int, int]) -> np.ndarray:
    out = low.astype(np.float32, copy=True)
    if np.all(valid):
        return out
    valid_u8 = valid.astype(np.uint8)
    if np.any(valid):
        for _ in range(5):
            blurred = cv2.blur(out, (5, 5), borderType=cv2.BORDER_REPLICATE)
            grow = cv2.dilate(valid_u8, _ellipse_kernel(2)) > 0
            fill = grow & ~valid
            if not np.any(fill):
                break
            out[fill] = blurred[fill]
            valid = valid | fill
            valid_u8 = valid.astype(np.uint8)
    if not np.all(valid):
        out[~valid] = np.asarray(fallback, dtype=np.float32).reshape(1, 3)
    return out


def _accumulate_low_res_plate(rgb: np.ndarray, candidates: np.ndarray, low_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate candidate RGB into low-res bins with only stripe-bounded floats."""

    h, w = rgb.shape[:2]
    low_h, low_w = low_shape
    flat_size = max(1, low_h * low_w)
    sums_flat = np.zeros((3, flat_size), dtype=np.float64)
    counts_flat = np.zeros(flat_size, dtype=np.float64)
    if h == 0 or w == 0 or low_h == 0 or low_w == 0 or not np.any(candidates):
        return np.zeros((low_h, low_w, 3), dtype=np.float32), np.zeros((low_h, low_w), dtype=np.float32)

    x_bins = np.floor((np.arange(w, dtype=np.float64) + 0.5) * (low_w / float(w))).astype(np.int32)
    x_bins = np.clip(x_bins, 0, low_w - 1)
    max_stripe_pixels = 750_000
    stripe_rows = max(1, min(h, 512, max_stripe_pixels // max(1, w)))
    for y0 in range(0, h, stripe_rows):
        y1 = min(h, y0 + stripe_rows)
        known = candidates[y0:y1]
        if not np.any(known):
            continue
        yy, xx = np.nonzero(known)
        if yy.size == 0:
            continue
        source_y = yy.astype(np.float64) + float(y0) + 0.5
        y_bins = np.floor(source_y * (low_h / float(h))).astype(np.int32)
        y_bins = np.clip(y_bins, 0, low_h - 1)
        flat_bins = y_bins * low_w + x_bins[xx]
        counts_flat += np.bincount(flat_bins, minlength=flat_size)
        stripe = rgb[y0:y1]
        for channel in range(3):
            sums_flat[channel] += np.bincount(flat_bins, weights=stripe[:, :, channel][yy, xx], minlength=flat_size)

    counts = counts_flat.reshape(low_h, low_w).astype(np.float32)
    sums = np.stack([sums_flat[channel].reshape(low_h, low_w) for channel in range(3)], axis=2).astype(np.float32)
    return sums, counts


def _low_res_shape(h: int, w: int, max_side: int) -> tuple[int, int]:
    if h <= 0 or w <= 0:
        return 0, 0
    side = max(1, int(max_side))
    scale = min(1.0, side / max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _ensure_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb_u8 must have shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if (arr.size and float(np.nanmax(arr)) <= 1.0) else 1.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0) * scale
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _mask_to_u8(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if (arr.size and float(np.nanmax(arr)) <= 1.0) else 1.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0) * scale
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr.shape != shape:
        raise ValueError(f"{name} must match image shape")
    return np.ascontiguousarray(arr)


def _mask_to_bool(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    arr = _mask_to_u8(mask, shape, name)
    if arr is None:
        return None
    return arr > 127


def _sanitize_rgb_tuple(value: tuple[int, int, int] | np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError("screen color must have at least 3 channels")
    return tuple(np.clip(np.rint(arr[:3]), 0, 255).astype(np.uint8).tolist())


def _slice_bounds(value: slice, limit: int) -> tuple[int, int]:
    start = 0 if value.start is None else int(value.start)
    stop = limit if value.stop is None else int(value.stop)
    if start < 0:
        start += limit
    if stop < 0:
        stop += limit
    start = max(0, min(limit, start))
    stop = max(start, min(limit, stop))
    return start, stop
