from __future__ import annotations

import cv2
import numpy as np

from .color_math import _clip01, _compute_key_spill_strength, _linear_luma_from_rgb_u8, _smoothstep
from .tiling import (
    _effective_edge_radius,
    _ellipse_kernel,
    _expanded_mask_bounds,
    _odd_kernel_from_radius,
    _raise_if_cancelled,
    _report,
)
from .types import CancelCallback, KeySettings, ProgressCallback


def _ensure_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb_u8 must have shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _mask_to_bool(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, -1] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.dtype == bool:
        if arr.shape != shape:
            arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if arr.shape != shape:
            raise ValueError(f"{name} must match image shape")
        return np.ascontiguousarray(arr.astype(bool, copy=False))
    if arr.shape != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr.shape != shape:
        raise ValueError(f"{name} must match image shape")
    return arr > 127


def _mask_to_u8(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0) * scale
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr.shape != shape:
        raise ValueError(f"{name} must match image shape")
    return np.ascontiguousarray(arr)


def _alpha_hint_foreground_mask(alpha_hint: np.ndarray | None, settings: KeySettings) -> np.ndarray | None:
    if alpha_hint is None:
        return None
    threshold = int(np.clip(int(settings.alpha_hint_foreground_threshold), 1, 255))
    mask = alpha_hint >= threshold
    return mask if np.any(mask) else None


def _apply_alpha_hint(
    alpha: np.ndarray,
    edge_mask: np.ndarray,
    background: np.ndarray,
    alpha_hint: np.ndarray,
    settings: KeySettings,
) -> None:
    """Merge a coarse external alpha hint without letting it create background.

    High-confidence hint pixels protect foreground/core from chroma removal.
    Mid-confidence hint pixels can raise alpha only where the connected-screen
    model has not already classified the pixel as background.
    """

    strength = _clip01(settings.alpha_hint_strength)
    if strength <= 0:
        return
    minimum = int(np.clip(int(settings.alpha_hint_minimum_alpha), 0, 255))
    guidance = alpha_hint >= minimum
    if not np.any(guidance):
        return
    allowed = guidance & ~background
    if not np.any(allowed):
        return
    hinted = alpha_hint.astype(np.float32)
    current = alpha.astype(np.float32)
    boosted = np.maximum(current, current * (1.0 - strength) + hinted * strength)
    alpha[allowed] = np.rint(np.clip(boosted[allowed], 0, 255)).astype(np.uint8)
    core = allowed & (alpha_hint >= int(np.clip(int(settings.alpha_hint_foreground_threshold), 1, 255)))
    alpha[core] = np.maximum(alpha[core], alpha_hint[core])
    edge_mask[core & (alpha >= 248)] = False


def _border_connected(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return np.zeros(mask.shape, dtype=bool)
    labels_count, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if labels_count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(labels, border_labels)


def _remove_small_components(mask: np.ndarray, min_area: int, *, protect_border: bool) -> np.ndarray:
    if min_area <= 1 or not np.any(mask):
        return mask.astype(bool, copy=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool, copy=True)
    keep = np.zeros(count, dtype=bool)
    keep[0] = False
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area)
    if protect_border:
        border_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        keep[border_labels] = True
        keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=True)
    inv = ~mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool, copy=True)
    border_labels = np.unique(np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])))
    fill = np.zeros(count, dtype=bool)
    for label in range(1, count):
        if label in border_labels:
            continue
        fill[label] = stats[label, cv2.CC_STAT_AREA] < int(min_area)
    out = mask.astype(bool, copy=True)
    out[fill[labels]] = True
    return out


def _build_alpha_from_trimap(
    background: np.ndarray,
    probability: np.ndarray,
    fg_threshold: int,
    bg_threshold: int,
    settings: KeySettings,
) -> tuple[np.ndarray, np.ndarray]:
    radius = _effective_edge_radius(settings)
    bg_u8 = background.astype(np.uint8)
    kernel = _ellipse_kernel(radius)
    dilated = cv2.dilate(bg_u8, kernel) > 0
    eroded = cv2.erode(bg_u8, kernel) > 0
    edge_mask = dilated & ~eroded

    transition = (probability > fg_threshold) & (probability < bg_threshold)
    if np.any(transition):
        near_bg = cv2.dilate(bg_u8, _ellipse_kernel(max(1, radius * 2))) > 0
        edge_mask |= transition & near_bg

    alpha_f = np.ones(probability.shape, dtype=np.float32)
    alpha_f[background & ~edge_mask] = 0.0
    prob_f = probability.astype(np.float32) / 255.0
    fg = fg_threshold / 255.0
    bg = bg_threshold / 255.0
    core_bias = (_clip01(settings.core_strength) - 0.5) * 0.08
    fg = np.clip(fg + core_bias, 0.0, 0.92)
    soft = max(0.0, float(settings.edge_softness))
    fg = max(0.0, fg - soft * 0.02)
    bg = min(1.0, bg + soft * 0.02)
    edge_alpha = 1.0 - _smoothstep(fg, bg, prob_f)
    gamma = max(0.05, float(settings.matte_gamma))
    if abs(gamma - 1.0) > 1e-3:
        edge_alpha = np.power(np.clip(edge_alpha, 0.0, 1.0), 1.0 / gamma)
    alpha_f[edge_mask] = edge_alpha[edge_mask]

    if soft > 0 and radius > 1 and np.any(edge_mask):
        k = _odd_kernel_from_radius(max(0.35, min(radius / 3.0, radius * soft * 0.55)))
        if k > 1:
            blurred = cv2.GaussianBlur(alpha_f, (k, k), sigmaX=max(0.2, k / 5.0))
            blend = min(0.55, soft * 0.45)
            alpha_f[edge_mask] = alpha_f[edge_mask] * (1.0 - blend) + blurred[edge_mask] * blend

    # Keep core regions exact after edge-only smoothing.
    alpha_f[background & ~edge_mask] = 0.0
    alpha_f[(~background) & (~edge_mask)] = 1.0
    alpha_f[(probability >= bg_threshold) & background] = 0.0
    alpha_f[(probability <= fg_threshold) & (~background)] = 1.0
    alpha_f[alpha_f < 0.004] = 0.0
    alpha_f[alpha_f > 0.996] = 1.0
    return edge_mask, np.rint(np.clip(alpha_f, 0.0, 1.0) * 255.0).astype(np.uint8)


def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    guide_f = np.asarray(guide, dtype=np.float32)
    src_f = np.asarray(src, dtype=np.float32)
    if guide_f.ndim != 2 or src_f.ndim != 2 or guide_f.shape != src_f.shape:
        raise ValueError("guided filter expects matching 2D guide/src arrays")
    radius = max(0, int(radius))
    if radius <= 0 or guide_f.size == 0:
        return np.clip(src_f, 0.0, 1.0).astype(np.float32, copy=False)

    eps = max(1e-8, float(eps))
    ksize = (radius * 2 + 1, radius * 2 + 1)
    guide_f = np.nan_to_num(guide_f, nan=0.0, posinf=1.0, neginf=0.0)
    src_f = np.nan_to_num(src_f, nan=0.0, posinf=1.0, neginf=0.0)

    mean_i = cv2.boxFilter(guide_f, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_p = cv2.boxFilter(src_f, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    corr_i = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    corr_ip = cv2.boxFilter(guide_f * src_f, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    refined = mean_a * guide_f + mean_b
    return np.clip(refined, 0.0, 1.0).astype(np.float32, copy=False)


def _refine_alpha_guided(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    edge_mask: np.ndarray,
    background: np.ndarray,
    probability: np.ndarray,
    fg_threshold: int,
    bg_threshold: int,
    settings: KeySettings,
) -> np.ndarray:
    strength = _clip01(settings.guided_alpha_refine)
    if strength <= 0.0:
        return alpha_u8

    radius = max(1, int(settings.guided_radius))
    max_pixels = max(0, int(settings.guided_max_pixels))
    refine_mask = edge_mask.astype(bool, copy=False) & (alpha_u8 > 0) & (alpha_u8 < 255)
    if max_pixels <= 0 or not np.any(refine_mask):
        return alpha_u8

    y0, y1, x0, x1 = _expanded_mask_bounds(refine_mask, margin=radius * 2 + 2, shape=alpha_u8.shape)
    if (y1 - y0) * (x1 - x0) > max_pixels:
        return alpha_u8

    roi_y = slice(y0, y1)
    roi_x = slice(x0, x1)
    guide = _linear_luma_from_rgb_u8(rgb[roi_y, roi_x])
    src = alpha_u8[roi_y, roi_x].astype(np.float32) / 255.0
    refined = _guided_filter_gray(guide, src, radius, settings.guided_eps)
    blended = src * (1.0 - strength) + refined * strength

    out = alpha_u8.copy()
    target = refine_mask[roi_y, roi_x]
    out_roi = out[roi_y, roi_x]
    out_roi[target] = np.rint(np.clip(blended[target], 0.0, 1.0) * 255.0).astype(np.uint8)

    # Reassert exact trimap decisions after edge-only smoothing. Pixels that
    # were exact 0/255 before guided filtering stay exact, and non-edge known
    # connected-background/foreground core regions remain authoritative.
    out[alpha_u8 <= 0] = 0
    out[alpha_u8 >= 255] = 255
    out[background & ~edge_mask] = 0
    out[(~background) & ~edge_mask] = 255
    out[(probability >= bg_threshold) & background & ~edge_mask] = 0
    out[(probability <= fg_threshold) & (~background) & ~edge_mask] = 255
    return out


def _apply_screen_residue_alpha_cleanup(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    probability: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_map: np.ndarray | None,
    settings: KeySettings,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    alpha_hint: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Lower alpha for live pixels that are still indistinguishable from screen.

    This is a hard-screen/detail two-pass cleanup: the normal trimap/recovery pass
    remains free to preserve soft and thin foreground detail, then this pass only
    removes pixels that are both high screen-probability and RGB-close to the
    sampled/local screen plate. Manual keep and strong imported matte foreground
    remain authoritative.
    """

    strength = _clip01(settings.screen_cleanup_strength)
    if strength <= 0.0:
        return alpha_u8, None
    if rgb.shape[:2] != alpha_u8.shape or probability.shape != alpha_u8.shape:
        raise ValueError("screen cleanup inputs must share image height/width")
    if screen_map is not None and screen_map.shape[:2] != alpha_u8.shape:
        raise ValueError("screen_map must match alpha shape")

    bg_threshold = int(round(_clip01(settings.clip_background) * 255.0))
    similarity = max(0, int(settings.screen_cleanup_similarity))
    h, w = alpha_u8.shape
    out = alpha_u8.copy()
    cleanup = np.zeros((h, w), dtype=bool)

    protected = np.zeros((h, w), dtype=bool)
    hint_foreground = _alpha_hint_foreground_mask(alpha_hint, settings)
    if hint_foreground is not None:
        protected |= hint_foreground
    if keep_mask is not None:
        protected |= keep_mask
    if remove_mask is not None:
        remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
        out[remove_effective] = 0

    fallback_screen = np.asarray(screen_color, dtype=np.int16).reshape(1, 1, 3)
    stripe_rows = max(96, min(h, 512))
    stripes = list(range(0, h, stripe_rows))
    total = max(1, len(stripes))
    for index, y0 in enumerate(stripes, start=1):
        _raise_if_cancelled(cancel_callback)
        y1 = min(h, y0 + stripe_rows)
        live = (out[y0:y1] > 0) & (probability[y0:y1] >= bg_threshold) & (~protected[y0:y1])
        if not np.any(live):
            continue
        if screen_map is None:
            diff = np.max(np.abs(rgb[y0:y1].astype(np.int16) - fallback_screen), axis=2)
        else:
            diff = np.max(np.abs(rgb[y0:y1].astype(np.int16) - screen_map[y0:y1].astype(np.int16)), axis=2)
        target = live & (diff <= similarity)
        if not np.any(target):
            continue
        if strength >= 0.999:
            out_block = out[y0:y1]
            out_block[target] = 0
        else:
            out_block = out[y0:y1]
            lowered = np.rint(out_block[target].astype(np.float32) * (1.0 - strength)).astype(np.uint8)
            out_block[target] = np.minimum(out_block[target], lowered)
        cleanup_block = cleanup[y0:y1]
        cleanup_block[target & (out_block == 0)] = True
        _report(progress_callback, 0.171 + 0.003 * (index / total), "screen cleanup")

    if not np.any(cleanup):
        return alpha_u8, None
    return out, cleanup


def _apply_original_alpha(alpha_u8: np.ndarray, original_alpha: np.ndarray | None) -> np.ndarray:
    if original_alpha is None:
        return alpha_u8
    original = np.asarray(original_alpha, dtype=np.float32)
    if original.shape != alpha_u8.shape:
        original = cv2.resize(original, (alpha_u8.shape[1], alpha_u8.shape[0]), interpolation=cv2.INTER_AREA)
    out = alpha_u8.astype(np.float32) * np.clip(original, 0.0, 1.0)
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def _build_fringe_mask(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: KeySettings,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> np.ndarray:
    """Return a uint8 map of edge pixels whose RGB is screen-contaminated.

    The mask is global and alpha-stable: it only describes where color repair is
    allowed to operate. It intentionally avoids foreground-core pixels so v4
    color reconstruction cannot become a broad grading pass.
    """

    h, w = alpha_u8.shape
    fringe = np.zeros((h, w), dtype=np.uint8)
    semi = (alpha_u8 > 2) & (alpha_u8 < 253)
    band = semi.copy()
    radius = max(0, int(settings.fringe_band_radius))
    if radius > 0 and np.any(band):
        band = cv2.dilate(band.astype(np.uint8), _ellipse_kernel(radius)) > 0
    # Preserve fully transparent background RGB as zero. Include matte edge
    # pixels with non-zero alpha so hard-but-contaminated poster edges still get
    # channel clamping when they carry obvious key-color excess.
    band |= edge_mask & (alpha_u8 > 1)
    band &= alpha_u8 > 1
    if not np.any(band):
        return fringe

    stripe_rows = max(96, min(h, 512))
    stripes = list(range(0, h, stripe_rows))
    total = max(1, len(stripes))
    for index, y0 in enumerate(stripes, start=1):
        _raise_if_cancelled(cancel_callback)
        y1 = min(h, y0 + stripe_rows)
        band_block = band[y0:y1]
        if not np.any(band_block):
            continue
        alpha = alpha_u8[y0:y1].astype(np.float32) / 255.0
        edge_strength = np.clip(alpha * (1.0 - alpha) * 4.0, 0.0, 1.0)
        edge_strength = np.maximum(edge_strength, edge_mask[y0:y1].astype(np.float32) * 0.55)
        edge_strength = np.maximum(edge_strength, band_block.astype(np.float32) * 0.35)
        spill = _compute_key_spill_strength(rgb[y0:y1], screen_color)
        near_screen = probability[y0:y1].astype(np.float32) / 255.0
        spill_weight = np.maximum(_smoothstep(0.02, 0.50, spill), near_screen * np.clip(1.0 - alpha, 0.0, 1.0) * 0.75)
        mask = band_block.astype(np.float32) * edge_strength * spill_weight
        fringe[y0:y1] = np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        _report(progress_callback, 0.17 + 0.004 * (index / total), "fringe map")
    return fringe
