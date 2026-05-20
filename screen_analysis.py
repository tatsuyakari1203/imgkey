from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from imgkey_engine.screen_model import ScreenPlateRGB, build_screen_plate_rgb


DEFAULT_SCREEN_COLOR_RGB = (0, 220, 50)
DEFAULT_MAX_FULL_RES_SCREEN_PLATE_PIXELS = 4_000_000
DEFAULT_LOW_RES_MAX_SIDE = 512
_MAX_SCREEN_PICK_PIXELS = 200_000

@dataclass(slots=True)
class ScreenAnalysisResult:
    screen_color_rgb: tuple[int, int, int]
    screen_plate_rgb: ScreenPlateRGB
    screen_probability: np.ndarray
    screen_distance: np.ndarray
    spill_probability: np.ndarray
    classical_confidence: np.ndarray
    edge_mask: np.ndarray
    fringe_mask: np.ndarray

    def debug_images(self) -> dict[str, np.ndarray]:
        swatch = np.broadcast_to(np.asarray(self.screen_color_rgb, dtype=np.uint8).reshape(1, 1, 3), (32, 32, 3)).copy()
        return {
            "screen_color": swatch,
            "screen_probability": self.screen_probability.copy(),
            "screen_distance": self.screen_distance.copy(),
            "screen_plate": self.screen_plate_rgb.debug_image(),
            "spill_probability": self.spill_probability.copy(),
            "classical_confidence": self.classical_confidence.copy(),
            "edge_mask": self.edge_mask.copy(),
            "fringe_mask": self.fringe_mask.copy(),
        }


def analyze_screen(
    rgb_u8: np.ndarray,
    classical_alpha: np.ndarray | None = None,
    *,
    background_mask: np.ndarray | None = None,
    settings: Any | None = None,
    picked_screen_color: tuple[int, int, int] | np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    max_full_res_screen_plate_pixels: int | None = None,
    low_res_max_side: int = DEFAULT_LOW_RES_MAX_SIDE,
) -> ScreenAnalysisResult:
    """Build deterministic screen-analysis maps for later classical cleanup.

    The function intentionally keeps retained maps compact: scalar maps are
    uint8, masks are uint8, and full-resolution RGB screen plates are retained
    only below ``max_full_res_screen_plate_pixels``.
    """

    rgb = _ensure_rgb_u8(rgb_u8)
    h, w = rgb.shape[:2]
    shape = (h, w)
    alpha = _mask_to_u8(classical_alpha, shape, "classical_alpha")
    keep = _mask_to_bool(keep_mask, shape, "keep_mask")
    remove = _mask_to_bool(remove_mask, shape, "remove_mask")
    bg_mask = _mask_to_bool(background_mask, shape, "background_mask")
    if bg_mask is None and alpha is not None:
        bg_mask = alpha <= 8

    if picked_screen_color is not None:
        screen_color = _sanitize_rgb_tuple(picked_screen_color)
    else:
        screen_color = _estimate_screen_color(rgb, settings, alpha, bg_mask, keep)

    screen_distance, screen_probability, spill_probability = _compute_screen_maps(rgb, screen_color, settings)
    classical_confidence = _compute_classical_confidence(alpha, bg_mask, screen_probability)
    edge_mask = _build_edge_mask(alpha, bg_mask, screen_distance, screen_probability, settings)
    fringe_mask = _build_fringe_mask(alpha, edge_mask, screen_probability, spill_probability, settings)

    plate_candidates = _screen_plate_candidates(screen_probability, alpha, bg_mask, keep, remove, settings)
    cap = _screen_plate_cap(settings, max_full_res_screen_plate_pixels)
    screen_plate = build_screen_plate_rgb(
        rgb,
        plate_candidates,
        screen_color,
        max_full_res_pixels=cap,
        low_res_max_side=low_res_max_side,
    )

    return ScreenAnalysisResult(
        screen_color_rgb=screen_color,
        screen_plate_rgb=screen_plate,
        screen_probability=screen_probability,
        screen_distance=screen_distance,
        spill_probability=spill_probability,
        classical_confidence=classical_confidence,
        edge_mask=edge_mask,
        fringe_mask=fringe_mask,
    )




def _estimate_screen_color(
    rgb: np.ndarray,
    settings: Any | None,
    alpha: np.ndarray | None,
    background_mask: np.ndarray | None,
    keep_mask: np.ndarray | None,
) -> tuple[int, int, int]:
    fallback = _settings_rgb(settings, "key_color", DEFAULT_SCREEN_COLOR_RGB)
    width = max(4, int(_settings_value(settings, "border_sample_width", 24)), int(_settings_value(settings, "sample_size", 5)))
    border = _border_mask(rgb.shape[:2], width)
    eligible = border
    if keep_mask is not None:
        eligible &= ~keep_mask
    if background_mask is not None:
        bg_border = eligible & background_mask
        if np.count_nonzero(bg_border) >= 32:
            eligible = bg_border
    elif alpha is not None:
        bg_border = eligible & (alpha <= 128)
        if np.count_nonzero(bg_border) >= 32:
            eligible = bg_border

    samples = rgb[eligible]
    if samples.size == 0:
        return fallback
    if len(samples) > _MAX_SCREEN_PICK_PIXELS:
        step = max(1, len(samples) // _MAX_SCREEN_PICK_PIXELS)
        samples = samples[::step]

    hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float32)
    val = hsv[:, 2].astype(np.float32)
    hue = hsv[:, 0].astype(np.int16)
    saturated = (sat >= 36.0) & (val >= 36.0)
    screenish = saturated & (hue >= 35) & (hue <= 135)
    if np.count_nonzero(screenish) < max(24, len(samples) // 160):
        screenish = saturated
    if np.count_nonzero(screenish) < 16:
        return fallback

    weights = (sat * np.maximum(val, 1.0))[screenish]
    hist = np.bincount(hue[screenish].astype(np.int32), weights=weights, minlength=180)
    if float(hist.max()) <= 0.0:
        return fallback
    peak = int(np.argmax(hist))
    hue_delta = np.abs(hue - peak)
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    hue_band = screenish & (hue_delta <= 9)
    if np.count_nonzero(hue_band) < 16:
        hue_band = screenish

    candidate_rgb = samples[hue_band]
    filtered = _reject_chroma_outliers(candidate_rgb)
    if len(filtered) < max(12, len(candidate_rgb) // 10):
        filtered = candidate_rgb
    sampled = _trimmed_median_rgb(filtered)
    if float(np.max(sampled) - np.min(sampled)) < 24.0:
        return fallback
    return tuple(np.clip(np.rint(sampled), 0, 255).astype(np.uint8).tolist())


def _reject_chroma_outliers(samples: np.ndarray) -> np.ndarray:
    pix = samples.astype(np.float32) / 255.0
    chroma = pix / np.maximum(np.sum(pix, axis=1, keepdims=True), 1e-4)
    center = np.median(chroma, axis=0)
    dist = np.linalg.norm(chroma - center.reshape(1, 3), axis=1)
    median = float(np.median(dist))
    mad = float(np.median(np.abs(dist - median)))
    limit = max(0.035, median + 3.0 * max(mad, 1e-4))
    keep = dist <= limit
    if np.count_nonzero(keep) < 8:
        cutoff = float(np.percentile(dist, 75.0))
        keep = dist <= max(cutoff, 0.035)
    return samples[keep]


def _trimmed_median_rgb(samples: np.ndarray) -> np.ndarray:
    values = samples.astype(np.float32)
    if len(values) <= 12:
        return np.median(values, axis=0)
    center = np.median(values, axis=0)
    dist = np.linalg.norm(values - center.reshape(1, 3), axis=1)
    cutoff = float(np.percentile(dist, 85.0))
    trimmed = values[dist <= cutoff]
    if len(trimmed) < 8:
        trimmed = values
    return np.median(trimmed, axis=0)


def _compute_screen_maps(
    rgb: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    distance = np.empty((h, w), dtype=np.uint8)
    probability = np.empty((h, w), dtype=np.uint8)
    spill = np.empty((h, w), dtype=np.uint8)
    stripe_rows = max(96, min(h, 512))
    for y0 in range(0, h, stripe_rows):
        y1 = min(h, y0 + stripe_rows)
        d, p, s = _compute_screen_maps_block(rgb[y0:y1], screen_color, settings)
        distance[y0:y1] = d
        probability[y0:y1] = p
        spill[y0:y1] = s
    return distance, probability, spill


def _compute_screen_maps_block(
    rgb: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key = np.clip(key, 1e-4, 1.0)
    pix = rgb.astype(np.float32) / 255.0
    pix_sum = np.maximum(np.sum(pix, axis=2), 1e-4)
    key_chroma = key / max(float(np.sum(key)), 1e-4)
    chroma = pix / pix_sum[:, :, None]
    chroma_dist = np.linalg.norm(chroma - key_chroma.reshape(1, 1, 3), axis=2)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    key_hsv = cv2.cvtColor(np.asarray([[screen_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    hue_diff = np.abs(hsv[:, :, 0].astype(np.float32) - float(key_hsv[0]))
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0
    sat_diff = np.abs(hsv[:, :, 1].astype(np.float32) - float(key_hsv[1])) / 255.0
    hue_score = hue_diff + sat_diff * 0.16

    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    key_ycc = cv2.cvtColor(np.asarray([[screen_color]], dtype=np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)[0, 0]
    ycc_chroma = np.sqrt((ycc[:, :, 1] - key_ycc[1]) ** 2 + (ycc[:, :, 2] - key_ycc[2]) ** 2) / (255.0 * np.sqrt(2.0))

    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = np.sum(pix * luma_weights.reshape(1, 1, 3), axis=2)
    key_luma = float(key @ luma_weights)
    brightness_delta = np.abs(luma - key_luma)

    distance_f = chroma_dist * 1.18 + hue_score * 0.26 + ycc_chroma * 0.34 + brightness_delta * 0.10
    distance_f = np.clip(distance_f, 0.0, 1.0)

    tol = max(0.045, float(_settings_value(settings, "tolerance", 0.18)) * 0.85)
    soft = max(0.045, float(_settings_value(settings, "softness", 0.075)) * 2.35)
    prob_f = 1.0 - _smoothstep(tol, tol + soft, distance_f)

    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        key_values = pix[:, :, key_channel]
        other_max = np.maximum(pix[:, :, other[0]], pix[:, :, other[1]])
        spill_strength = np.maximum(key_values - other_max, 0.0) / np.maximum(key_values, 1.0 / 255.0)
    else:
        key_vec = key - key_luma
        norm = float(np.linalg.norm(key_vec))
        if norm < 1e-4:
            spill_strength = np.zeros(rgb.shape[:2], dtype=np.float32)
        else:
            key_vec /= norm
            residual = pix - luma[:, :, None]
            spill_strength = np.maximum(np.sum(residual * key_vec.reshape(1, 1, 3), axis=2), 0.0)
    spill_f = np.maximum(_smoothstep(0.025, 0.42, spill_strength), prob_f * 0.35)

    return (
        np.rint(distance_f * 255.0).astype(np.uint8),
        np.rint(np.clip(prob_f, 0.0, 1.0) * 255.0).astype(np.uint8),
        np.rint(np.clip(spill_f, 0.0, 1.0) * 255.0).astype(np.uint8),
    )


def _compute_classical_confidence(
    alpha: np.ndarray | None,
    background_mask: np.ndarray | None,
    screen_probability: np.ndarray,
) -> np.ndarray:
    prob_sep = np.abs(screen_probability.astype(np.float32) - 127.5) / 127.5
    if alpha is None:
        conf = prob_sep
    else:
        alpha_sep = np.abs(alpha.astype(np.float32) - 127.5) / 127.5
        conf = alpha_sep * 0.68 + prob_sep * 0.32
        if background_mask is not None:
            bg_conf = (screen_probability.astype(np.float32) / 255.0) * background_mask.astype(np.float32)
            conf = np.maximum(conf, bg_conf)
    return np.rint(np.clip(conf, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_edge_mask(
    alpha: np.ndarray | None,
    background_mask: np.ndarray | None,
    screen_distance: np.ndarray,
    screen_probability: np.ndarray,
    settings: Any | None,
) -> np.ndarray:
    shape = screen_distance.shape
    edge = np.zeros(shape, dtype=bool)
    radius = max(1, min(12, int(_settings_value(settings, "edge_refine_radius", 2))))
    if alpha is not None:
        semi = (alpha > 8) & (alpha < 247)
        alpha_fg = alpha >= 128
        grad = cv2.morphologyEx(alpha_fg.astype(np.uint8), cv2.MORPH_GRADIENT, _ellipse_kernel(max(1, radius))) > 0
        edge |= semi | grad
    if background_mask is not None:
        grad = cv2.morphologyEx(background_mask.astype(np.uint8), cv2.MORPH_GRADIENT, _ellipse_kernel(max(1, radius))) > 0
        edge |= grad

    canny = cv2.Canny(screen_distance, 18, 58) > 0
    transition = (screen_probability > 24) & (screen_probability < 232)
    edge |= canny & transition
    if np.any(edge):
        edge = cv2.dilate(edge.astype(np.uint8), _ellipse_kernel(1)) > 0
    return edge.astype(np.uint8) * 255


def _build_fringe_mask(
    alpha: np.ndarray | None,
    edge_mask: np.ndarray,
    screen_probability: np.ndarray,
    spill_probability: np.ndarray,
    settings: Any | None,
) -> np.ndarray:
    edge = edge_mask > 0
    radius = max(1, min(8, int(_settings_value(settings, "fringe_band_radius", 3))))
    edge_band = cv2.dilate(edge.astype(np.uint8), _ellipse_kernel(radius)) > 0 if np.any(edge) else edge
    if alpha is None:
        live = np.ones(edge.shape, dtype=bool)
        semi = edge_band
    else:
        live = alpha > 1
        semi = (alpha > 2) & (alpha < 253)
    spillish = (spill_probability >= 48) | ((screen_probability >= 96) & live)
    fringe = (semi | edge_band) & spillish & live
    return fringe.astype(np.uint8) * 255


def _screen_plate_candidates(
    screen_probability: np.ndarray,
    alpha: np.ndarray | None,
    background_mask: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    settings: Any | None,
) -> np.ndarray:
    threshold = int(round(float(_settings_value(settings, "clip_background", 0.78)) * 255.0))
    threshold = int(np.clip(max(threshold, 180), 0, 255))
    candidates = screen_probability >= threshold
    if background_mask is not None:
        candidates |= background_mask
    if alpha is not None:
        candidates |= alpha <= 8
    if remove_mask is not None:
        candidates |= remove_mask
    if keep_mask is not None:
        candidates &= ~keep_mask
    return candidates.astype(bool)


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


def _screen_plate_cap(settings: Any | None, explicit_cap: int | None) -> int:
    if explicit_cap is not None:
        return int(explicit_cap)
    return int(_settings_value(settings, "max_local_screen_model_pixels", DEFAULT_MAX_FULL_RES_SCREEN_PLATE_PIXELS))


def _border_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if h == 0 or w == 0:
        return mask
    bw = min(max(1, int(width)), max(1, h // 2), max(1, w // 2))
    mask[:bw, :] = True
    mask[h - bw :, :] = True
    mask[:, :bw] = True
    mask[:, w - bw :] = True
    return mask


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


def _settings_value(settings: Any | None, name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


def _settings_rgb(settings: Any | None, name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    return _sanitize_rgb_tuple(_settings_value(settings, name, default))


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


def _ellipse_kernel(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = r * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    denom = max(float(edge1) - float(edge0), 1e-6)
    t = np.clip((value - float(edge0)) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)
