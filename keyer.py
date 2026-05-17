from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

_MAX_INNER_LABEL_PIXELS = 16_000_000


@dataclass(slots=True)
class KeySettings:
    """Settings for the non-AI ImgKey v2 keying pipeline.

    The original app still constructs this class with only
    ``key_color/tolerance/softness/edge_blur/cleanup/despill``. Those fields
    remain first-class compatibility controls and feed the v2 pipeline.
    """

    # Original v1 positional/keyword fields, kept in the same order.
    key_color: tuple[int, int, int] = (0, 220, 50)
    tolerance: float = 0.18
    softness: float = 0.075
    edge_blur: float = 1.2
    cleanup: int = 1
    despill: float = 0.70

    # v2 sampling/model controls.
    mode: str = "GraphicExact"
    sample_size: int = 5
    auto_border_sample: bool = True
    auto_detect_key_color: bool = False
    border_sample_width: int = 24
    local_screen_model: bool = True
    max_local_screen_model_pixels: int = 12_000_000

    # Matte controls.
    brightness_tolerance: float = 0.34
    clip_background: float = 0.78
    clip_foreground: float = 0.14
    matte_gamma: float = 1.0
    core_strength: float = 0.55
    edge_refine_radius: int = 0
    edge_softness: float = 0.55
    erode_expand: int = 0
    despeckle_min_area: int = 48

    # Connected-background policy. Default preserves disconnected key-colored
    # foreground islands; aggressive mode removes interior high-confidence key.
    aggressive_interior_removal: bool = False
    aggressive_threshold: float = 0.84
    aggressive_min_area: int = 0

    # Optional AI/manual alpha hint seam. A grayscale hint is not an AI runtime;
    # it is a coarse protection matte imported from any external tool and merged
    # conservatively into the classical connected-background pipeline.
    alpha_hint_foreground_threshold: int = 192
    alpha_hint_minimum_alpha: int = 48
    alpha_hint_strength: float = 1.0

    # Color decontamination.
    decontaminate: float = 0.50
    luminance_restore: float = 0.35
    unmix_amount: float = 0.75

    # Export/preview hooks.
    preview_scale: float = 1.0
    full_res_crop: tuple[int, int, int, int] | None = None
    use_tiling: bool = True
    tile_size: int = 2048
    tile_overlap: int = 128

    # v4 edge color reconstruction. App/UI code can continue to drive
    # luminance_restore; luminance_protect is an optional API alias/override.
    fringe_remove: float = 0.75
    edge_color_repair: float = 0.65
    inner_color_pull: float = 0.45
    fringe_band_radius: int = 3
    luminance_protect: float | None = None

    # Optional v5 guided alpha refinement, appended to preserve existing
    # positional compatibility for earlier settings fields.
    guided_alpha_refine: float = 0.0
    guided_radius: int = 8
    guided_eps: float = 1e-3
    guided_max_pixels: int = 2_000_000


@dataclass(slots=True)
class KeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground: np.ndarray | None
    background_mask: np.ndarray | None
    edge_mask: np.ndarray | None
    despill_mask: np.ndarray | None
    preview_scale: float = 1.0
    screen_probability: np.ndarray | None = None
    screen_color: tuple[int, int, int] | None = None
    alpha_hint: np.ndarray | None = None
    fringe_mask: np.ndarray | None = None
    repaired_edge: np.ndarray | None = None
    foreground_rgb: np.ndarray | None = None


@dataclass(slots=True)
class _GlobalMatte:
    screen_color: tuple[int, int, int]
    screen_probability: np.ndarray
    screen_map: np.ndarray | None
    background_mask: np.ndarray
    edge_mask: np.ndarray
    alpha: np.ndarray
    alpha_hint: np.ndarray | None
    fringe_mask: np.ndarray
    inner_labels: np.ndarray | None
    inner_label_to_flat: np.ndarray | None


def read_image_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        image = ImageOps.exif_transpose(Image.open(path))
    except Exception as exc:
        raise ValueError(f"Cannot read image: {path}") from exc

    has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()
    original_alpha = rgba[:, :, 3].astype(np.float32) / 255.0 if has_alpha else None
    return rgb, original_alpha


def write_png_rgba(path: str | Path, rgba: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError(f"Cannot encode PNG: {path}")
    encoded.tofile(str(path))


def read_grayscale_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a manual keep/remove matte or AI alpha hint as uint8 grayscale.

    If ``shape`` is supplied, the mask is resized with nearest-neighbor
    interpolation so brush/import tools can pass it directly to the engine.
    """

    try:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Cannot read mask: {path}") from exc
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def read_alpha_hint_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read an externally generated coarse alpha hint as uint8 grayscale.

    ImgKey does not generate this with bundled AI. The mask can come from a
    user, future BiRefNet assist, or another external tool and is interpreted as
    foreground protection/alpha guidance for the classical v2 pipeline.
    """

    return read_grayscale_mask(path, shape)


def write_grayscale_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), mode="L").save(path)


def resize_for_preview(rgb: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale


def process_chroma_key(
    rgb_u8: np.ndarray,
    settings: KeySettings,
    original_alpha: np.ndarray | None = None,
    *,
    keep_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    alpha_hint: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> np.ndarray:
    """Compatibility wrapper returning straight-alpha RGBA uint8."""

    return process_key_image(
        rgb_u8,
        settings,
        original_alpha,
        keep_mask=keep_mask,
        remove_mask=remove_mask,
        alpha_hint=alpha_hint,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    ).rgba


def process_key_image(
    rgb_u8: np.ndarray,
    settings: KeySettings | None = None,
    original_alpha: np.ndarray | None = None,
    *,
    keep_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    alpha_hint: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> KeyResult:
    """Run the v2 classical keying engine and return debug outputs.

    Global sampling, connected-background decisions, manual mask merging, and
    trimap/alpha generation happen once for the whole image. Full-resolution
    color unmix/despill then runs in overlapped tiles and writes only tile cores,
    avoiding a full-image float32 RGB working copy in the export path.
    """

    settings = settings or KeySettings()
    rgb = _ensure_rgb_u8(rgb_u8)
    h, w = rgb.shape[:2]
    keep = _mask_to_bool(keep_mask, (h, w), "keep_mask")
    remove = _mask_to_bool(remove_mask, (h, w), "remove_mask")
    hint = _mask_to_u8(alpha_hint, (h, w), "alpha_hint")

    if settings.mode not in {"GraphicExact", "ProChroma", "AIHint"}:
        raise ValueError(f"Unsupported keying mode: {settings.mode}")

    _raise_if_cancelled(cancel_callback)
    global_matte = _build_global_matte(rgb, settings, original_alpha, keep, remove, hint, progress_callback, cancel_callback)
    _report(progress_callback, 0.18, "global matte")
    _raise_if_cancelled(cancel_callback)

    rgba, despill_mask = _render_tiled_rgba(rgb, settings, global_matte, progress_callback, cancel_callback)
    foreground = rgba[:, :, :3].copy()
    crop = _normalized_crop(settings.full_res_crop, w, h)
    if crop is not None:
        x0, y0, x1, y1 = crop
        rgba = rgba[y0:y1, x0:x1].copy()
        foreground = foreground[y0:y1, x0:x1].copy()
        alpha = global_matte.alpha[y0:y1, x0:x1].copy()
        background_mask = (global_matte.background_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
        edge_mask = (global_matte.edge_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
        despill_mask = despill_mask[y0:y1, x0:x1].copy()
        fringe_mask = global_matte.fringe_mask[y0:y1, x0:x1].copy()
        probability = global_matte.screen_probability[y0:y1, x0:x1].copy()
        hint_out = None if global_matte.alpha_hint is None else global_matte.alpha_hint[y0:y1, x0:x1].copy()
    else:
        alpha = global_matte.alpha
        background_mask = (global_matte.background_mask.astype(np.uint8) * 255)
        edge_mask = (global_matte.edge_mask.astype(np.uint8) * 255)
        fringe_mask = global_matte.fringe_mask
        probability = global_matte.screen_probability
        hint_out = None if global_matte.alpha_hint is None else global_matte.alpha_hint.copy()
    return KeyResult(
        rgba=rgba,
        alpha=alpha,
        foreground=foreground,
        background_mask=background_mask,
        edge_mask=edge_mask,
        despill_mask=despill_mask,
        preview_scale=float(settings.preview_scale),
        screen_probability=probability,
        screen_color=global_matte.screen_color,
        alpha_hint=hint_out,
        fringe_mask=fringe_mask,
    )


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


def _build_global_matte(
    rgb: np.ndarray,
    settings: KeySettings,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    alpha_hint: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> _GlobalMatte:
    h, w = rgb.shape[:2]
    screen_color = _sample_screen_color(rgb, settings)
    _report(progress_callback, 0.02, "sample screen")
    _raise_if_cancelled(cancel_callback)
    probability = _compute_screen_probability(rgb, screen_color, settings, progress_callback, cancel_callback)
    _report(progress_callback, 0.10, "screen probability")
    _raise_if_cancelled(cancel_callback)

    bg_threshold = int(round(_clip01(settings.clip_background) * 255.0))
    fg_threshold = int(round(_clip01(settings.clip_foreground) * 255.0))
    candidates = probability >= bg_threshold

    if settings.erode_expand != 0:
        k = _ellipse_kernel(abs(int(settings.erode_expand)))
        if settings.erode_expand > 0:
            candidates = cv2.dilate(candidates.astype(np.uint8), k) > 0
        else:
            candidates = cv2.erode(candidates.astype(np.uint8), k) > 0

    background = _border_connected(candidates)
    _report(progress_callback, 0.12, "connected background")
    _raise_if_cancelled(cancel_callback)
    if settings.aggressive_interior_removal:
        aggressive = probability >= int(round(_clip01(settings.aggressive_threshold) * 255.0))
        if settings.aggressive_min_area > 1:
            aggressive = _remove_small_components(aggressive, int(settings.aggressive_min_area), protect_border=False)
        background |= aggressive

    hint_foreground = _alpha_hint_foreground_mask(alpha_hint, settings)
    if hint_foreground is not None:
        background &= ~hint_foreground
    if keep_mask is not None:
        background &= ~keep_mask
    if remove_mask is not None:
        remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
        background |= remove_effective

    min_area = max(int(settings.despeckle_min_area), int(settings.cleanup) * 12)
    if min_area > 0:
        background = _remove_small_components(background, min_area, protect_border=True)
        background = _fill_small_holes(background, min_area)
        if hint_foreground is not None:
            background &= ~hint_foreground
        if keep_mask is not None:
            background &= ~keep_mask
        if remove_mask is not None:
            remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
            background |= remove_effective

    edge_mask, alpha = _build_alpha_from_trimap(background, probability, fg_threshold, bg_threshold, settings)
    _report(progress_callback, 0.15, "trimap")
    _raise_if_cancelled(cancel_callback)
    if keep_mask is not None:
        alpha[keep_mask] = 255
        edge_mask[keep_mask] = False
    if alpha_hint is not None:
        _apply_alpha_hint(alpha, edge_mask, background, alpha_hint, settings)
    if remove_mask is not None:
        remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
        alpha[remove_effective] = 0
        background[remove_effective] = True

    alpha = _refine_alpha_guided(rgb, alpha, edge_mask, background, probability, fg_threshold, bg_threshold, settings)

    alpha = _apply_original_alpha(alpha, original_alpha)
    screen_map = _estimate_screen_map(rgb, probability >= bg_threshold, screen_color, settings)
    _report(progress_callback, 0.17, "screen model")
    _raise_if_cancelled(cancel_callback)
    fringe_mask = _build_fringe_mask(rgb, alpha, edge_mask, probability, screen_color, settings, progress_callback, cancel_callback)
    _report(progress_callback, 0.175, "fringe map")
    _raise_if_cancelled(cancel_callback)
    inner_labels, inner_label_to_flat = _build_nearest_inner_label_map(alpha, background, probability, fringe_mask, settings)
    _report(progress_callback, 0.18, "inner color map")
    return _GlobalMatte(
        screen_color=screen_color,
        screen_probability=probability,
        screen_map=screen_map,
        background_mask=background,
        edge_mask=edge_mask,
        alpha=alpha,
        alpha_hint=alpha_hint,
        fringe_mask=fringe_mask,
        inner_labels=inner_labels,
        inner_label_to_flat=inner_label_to_flat,
    )


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


def _linear_luma_from_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    luma = _srgb_to_linear_f32(arr[:, :, 0].astype(np.float32) / 255.0) * 0.2126
    luma += _srgb_to_linear_f32(arr[:, :, 1].astype(np.float32) / 255.0) * 0.7152
    luma += _srgb_to_linear_f32(arr[:, :, 2].astype(np.float32) / 255.0) * 0.0722
    return np.clip(luma, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_original_alpha(alpha_u8: np.ndarray, original_alpha: np.ndarray | None) -> np.ndarray:
    if original_alpha is None:
        return alpha_u8
    original = np.asarray(original_alpha, dtype=np.float32)
    if original.shape != alpha_u8.shape:
        original = cv2.resize(original, (alpha_u8.shape[1], alpha_u8.shape[0]), interpolation=cv2.INTER_AREA)
    out = alpha_u8.astype(np.float32) * np.clip(original, 0.0, 1.0)
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


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
    known = known_background.astype(np.float32)
    if float(np.mean(known)) < 0.01:
        return None
    radius = int(np.clip(max(h, w) // 18, 24, 181))
    ksize = (radius * 2 + 1, radius * 2 + 1)
    denom = cv2.boxFilter(known, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    out = np.empty_like(rgb)
    fallback = np.asarray(screen_color, dtype=np.float32)
    for channel in range(3):
        src = rgb[:, :, channel].astype(np.float32) * known
        num = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
        value = np.where(denom > 1e-4, num / np.maximum(denom, 1e-4), fallback[channel])
        out[:, :, channel] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return out


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


def _compute_key_spill_strength(rgb: np.ndarray, screen_color: tuple[int, int, int]) -> np.ndarray:
    pix = rgb.astype(np.float32) / 255.0
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key = np.clip(key, 1e-4, 1.0)
    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        key_values = pix[:, :, key_channel]
        other_max = np.maximum(pix[:, :, other[0]], pix[:, :, other[1]])
        return np.clip(np.maximum(key_values - other_max, 0.0) / np.maximum(key_values, 1.0 / 255.0), 0.0, 1.0)

    # Custom-key fallback: subtract perceived luminance, then project the color
    # residual onto the screen-color residual vector. This detects magenta/cyan
    # halos without treating neutral bright edges as spill.
    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    key_luma = float(key @ luma_weights)
    key_vec = key - key_luma
    norm = float(np.linalg.norm(key_vec))
    if norm < 1e-4:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    key_vec /= norm
    pix_luma = np.sum(pix * luma_weights.reshape(1, 1, 3), axis=2)
    residual = pix - pix_luma[:, :, None]
    projection = np.sum(residual * key_vec.reshape(1, 1, 3), axis=2)
    return np.clip(np.maximum(projection, 0.0), 0.0, 1.0).astype(np.float32)


def _build_nearest_inner_label_map(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map each fringe pixel to a globally nearest clean foreground pixel.

    OpenCV's label image is retained globally; RGB is gathered lazily from the
    original uint8 source per tile so export never materializes a full repaired
    float/RGB debug image.
    """

    if _clip01(settings.inner_color_pull) <= 0 or _clip01(settings.edge_color_repair) <= 0:
        return None, None
    if not np.any(fringe_mask > 0):
        return None, None
    # Beyond this size, retaining global labels plus a label->source index table
    # can dominate memory; use the deterministic unmix+clamp fallback instead.
    if alpha_u8.size > _MAX_INNER_LABEL_PIXELS:
        return None, None
    prob_limit = max(48, int(round(_clip01(settings.clip_foreground) * 255.0)) + 32)
    inner = (alpha_u8 >= 250) & (~background_mask) & (fringe_mask <= 24) & (probability <= prob_limit)
    if np.count_nonzero(inner) == 0:
        return None, None
    try:
        src = np.where(inner, 0, 255).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    except (cv2.error, MemoryError):
        return None, None
    labels = np.ascontiguousarray(labels.astype(np.int32, copy=False))
    inner_flat = np.flatnonzero(inner.reshape(-1))
    inner_labels = labels.reshape(-1)[inner_flat]
    valid = inner_labels > 0
    if not np.any(valid):
        return None, None
    inner_flat = inner_flat[valid]
    inner_labels = inner_labels[valid]
    max_label = int(inner_labels.max())
    label_to_flat = np.full(max_label + 1, -1, dtype=np.int64)
    label_to_flat[inner_labels] = inner_flat.astype(np.int64, copy=False)
    return labels, label_to_flat


def _nearest_inner_rgb_for_slice(
    rgb: np.ndarray,
    labels: np.ndarray | None,
    label_to_flat: np.ndarray | None,
    read_y: slice,
    read_x: slice,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if labels is None or label_to_flat is None:
        return None, None
    label_tile = labels[read_y, read_x]
    valid = (label_tile > 0) & (label_tile < len(label_to_flat))
    if not np.any(valid):
        return None, None
    flat_tile = np.full(label_tile.shape, -1, dtype=np.int64)
    flat_tile[valid] = label_to_flat[label_tile[valid]]
    valid &= flat_tile >= 0
    if not np.any(valid):
        return None, None
    nearest = np.zeros((*label_tile.shape, 3), dtype=np.uint8)
    nearest[valid] = rgb.reshape(-1, 3)[flat_tile[valid]]
    return nearest, valid


def _render_tiled_rgba(
    rgb: np.ndarray,
    settings: KeySettings,
    matte: _GlobalMatte,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = matte.alpha
    despill_mask = np.zeros((h, w), dtype=np.uint8)
    tiles = list(_iter_tiles(h, w, settings, _effective_edge_radius(settings)))
    total = max(1, len(tiles))
    for index, tile in enumerate(tiles, start=1):
        _raise_if_cancelled(cancel_callback)
        read_y, read_x, core_y, core_x = tile
        rel_y = slice(core_y.start - read_y.start, core_y.stop - read_y.start)
        rel_x = slice(core_x.start - read_x.start, core_x.stop - read_x.start)
        screen_tile = None if matte.screen_map is None else matte.screen_map[read_y, read_x]
        nearest_inner_rgb, nearest_inner_valid = _nearest_inner_rgb_for_slice(
            rgb,
            matte.inner_labels,
            matte.inner_label_to_flat,
            read_y,
            read_x,
        )
        rgb_tile, spill_tile = _process_color_tile(
            rgb[read_y, read_x],
            matte.alpha[read_y, read_x],
            matte.edge_mask[read_y, read_x],
            matte.screen_probability[read_y, read_x],
            matte.fringe_mask[read_y, read_x],
            screen_tile,
            nearest_inner_rgb,
            nearest_inner_valid,
            matte.screen_color,
            settings,
        )
        rgba[core_y, core_x, :3] = rgb_tile[rel_y, rel_x]
        despill_mask[core_y, core_x] = spill_tile[rel_y, rel_x]
        _report(progress_callback, 0.18 + 0.82 * (index / total), f"tile {index}/{total}")
    rgba[matte.alpha <= 0, :3] = 0
    return rgba, despill_mask


def _process_color_tile(
    rgb_tile: np.ndarray,
    alpha_u8: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_tile: np.ndarray | None,
    nearest_inner_rgb: np.ndarray | None,
    nearest_inner_valid: np.ndarray | None,
    screen_color: tuple[int, int, int],
    settings: KeySettings,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_linear = _srgb_u8_to_linear_f32(rgb_tile)
    alpha = alpha_u8.astype(np.float32) / 255.0
    if screen_tile is None:
        screen = _srgb_u8_to_linear_f32(np.asarray(screen_color, dtype=np.uint8).reshape(1, 1, 3))
    else:
        screen = _srgb_u8_to_linear_f32(screen_tile)

    out = rgb_linear.copy()
    edge_strength = np.clip(alpha * (1.0 - alpha) * 4.0, 0.0, 1.0)
    edge_strength = np.maximum(edge_strength, edge_mask.astype(np.float32) * 0.35)
    live = alpha > 0.001

    legacy_spill = _compute_despill_mask(alpha, edge_strength, probability, settings)
    fringe_signal = fringe_mask.astype(np.float32) / 255.0
    fringe_signal[alpha <= 0.001] = 0.0
    repair_signal = fringe_signal
    edge_repair = _clip01(settings.edge_color_repair)
    fringe_remove = _clip01(settings.fringe_remove)
    decontaminate = 0.25 + 0.75 * _clip01(settings.decontaminate)
    despill_amount = _clip01(settings.despill)

    unmix_amount = _clip01(settings.unmix_amount) * edge_repair * decontaminate
    if unmix_amount > 0 and np.any(repair_signal > 0):
        safe_alpha = np.maximum(alpha[:, :, None], 0.06)
        unmixed = (rgb_linear - (1.0 - alpha[:, :, None]) * screen) / safe_alpha
        unmixed = np.clip(unmixed, 0.0, 1.0)
        blend = (repair_signal * unmix_amount)[:, :, None]
        out = out * (1.0 - blend) + unmixed * blend

    clamp_signal = np.maximum(repair_signal * despill_amount, legacy_spill * 0.40) * fringe_remove
    out = _apply_vlahos_clamp(out, screen, clamp_signal)

    pull_amount = _clip01(settings.inner_color_pull) * edge_repair * decontaminate
    if pull_amount > 0 and nearest_inner_rgb is not None and nearest_inner_valid is not None:
        pull = repair_signal * pull_amount
        pull = np.where(nearest_inner_valid, pull, 0.0)
        if np.any(pull > 0):
            nearest = _srgb_u8_to_linear_f32(nearest_inner_rgb)
            out = out * (1.0 - pull[:, :, None]) + nearest * pull[:, :, None]

    spill_mask = np.maximum(legacy_spill, repair_signal * max(despill_amount, edge_repair * decontaminate))
    out = _protect_luminance(out, rgb_linear, spill_mask, settings)
    out[~live] = 0.0
    rgb_out = rgb_tile.copy()
    changed = live & (spill_mask > 0.0)
    if np.any(changed):
        repaired = _linear_f32_to_srgb_u8(out)
        rgb_out[changed] = repaired[changed]
    rgb_out[~live] = 0
    return rgb_out, np.rint(np.clip(spill_mask, 0.0, 1.0) * 255.0).astype(np.uint8)


def _compute_despill_mask(
    alpha: np.ndarray,
    edge_strength: np.ndarray,
    probability: np.ndarray,
    settings: KeySettings,
) -> np.ndarray:
    amount = _clip01(settings.despill)
    if amount <= 0:
        return np.zeros(alpha.shape, dtype=np.float32)
    near_screen = probability.astype(np.float32) / 255.0
    mask = np.maximum(edge_strength, near_screen * np.clip(1.0 - alpha, 0.0, 1.0))
    mask *= amount
    mask[alpha <= 0.001] = 0.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _srgb_to_linear_f32(srgb: np.ndarray) -> np.ndarray:
    srgb_f = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(srgb_f <= 0.04045, srgb_f / 12.92, np.power((srgb_f + 0.055) / 1.055, 2.4)).astype(np.float32)


def _linear_to_srgb_f32(linear: np.ndarray) -> np.ndarray:
    linear_f = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(
        linear_f <= 0.0031308,
        linear_f * 12.92,
        1.055 * np.power(linear_f, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _srgb_u8_to_linear_f32(srgb: np.ndarray) -> np.ndarray:
    return _srgb_to_linear_f32(np.asarray(srgb, dtype=np.float32) / 255.0)


def _linear_f32_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(_linear_to_srgb_f32(linear) * 255.0), 0, 255).astype(np.uint8)


def _apply_vlahos_clamp(rgb: np.ndarray, screen_linear: np.ndarray, clamp_mask: np.ndarray) -> np.ndarray:
    if not np.any(clamp_mask > 0):
        return rgb
    out = rgb.copy()
    weight = np.clip(clamp_mask.astype(np.float32) * 1.50, 0.0, 1.0)
    key_map = np.clip(np.asarray(screen_linear, dtype=np.float32), 0.0, 1.0)
    key = np.mean(key_map.reshape(-1, 3), axis=0) if key_map.size else np.zeros(3, dtype=np.float32)
    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        target = np.maximum(out[:, :, other[0]], out[:, :, other[1]])
        excess = np.maximum(out[:, :, key_channel] - target, 0.0)
        out[:, :, key_channel] -= excess * weight
    else:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        key_luma = np.sum(key_map * luma_weights.reshape(1, 1, 3), axis=2)
        key_vec = key_map - key_luma[:, :, None]
        norm = np.linalg.norm(key_vec, axis=2)
        valid = norm >= 1e-4
        if np.any(valid):
            key_vec = np.divide(key_vec, np.maximum(norm[:, :, None], 1e-4), out=np.zeros_like(key_vec), where=valid[:, :, None])
            out_luma = np.sum(out * luma_weights.reshape(1, 1, 3), axis=2)
            residual = out - out_luma[:, :, None]
            excess = np.maximum(np.sum(residual * key_vec, axis=2), 0.0)
            out -= key_vec * (excess * weight)[:, :, None] * 0.70
    return np.clip(out, 0.0, 1.0)


def _protect_luminance(rgb: np.ndarray, original_rgb: np.ndarray, repair_mask: np.ndarray, settings: KeySettings) -> np.ndarray:
    protect = _effective_luminance_protect(settings)
    if protect <= 0 or not np.any(repair_mask > 0):
        return np.clip(rgb, 0.0, 1.0)
    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    src_luma = np.sum(original_rgb * luma_weights.reshape(1, 1, 3), axis=2)
    out_luma = np.sum(np.clip(rgb, 0.0, 1.0) * luma_weights.reshape(1, 1, 3), axis=2)
    scale = np.divide(src_luma, np.maximum(out_luma, 1e-4), out=np.ones_like(src_luma), where=out_luma > 1e-4)
    scale = np.clip(scale, 0.70, 1.45)
    amount = (np.clip(repair_mask, 0.0, 1.0) * protect)[:, :, None]
    protected = np.clip(rgb * scale[:, :, None], 0.0, 1.0)
    return np.clip(rgb * (1.0 - amount) + protected * amount, 0.0, 1.0)


def _effective_luminance_protect(settings: KeySettings) -> float:
    value = settings.luminance_restore if settings.luminance_protect is None else settings.luminance_protect
    return _clip01(float(value))


def _despill_tile(
    rgb: np.ndarray,
    original_rgb: np.ndarray,
    screen_color: tuple[int, int, int],
    spill_mask: np.ndarray,
    settings: KeySettings,
) -> np.ndarray:
    if not np.any(spill_mask > 0):
        return rgb
    out = rgb.copy()
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key_channel = int(np.argmax(key))
    other = [c for c in range(3) if c != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))

    if key_dom > 0.12:
        target = np.maximum(out[:, :, other[0]], out[:, :, other[1]])
        spill = np.maximum(out[:, :, key_channel] - target, 0.0)
        out[:, :, key_channel] -= spill * spill_mask
    else:
        # Custom-key decontamination: pull edge colors away from the screen
        # chroma vector while preserving most luminance.
        key_vec = key / max(float(np.linalg.norm(key)), 1e-4)
        projection = np.sum(out * key_vec.reshape(1, 1, 3), axis=2)
        neutral = np.mean(out, axis=2)
        excess = np.maximum(projection - neutral, 0.0)
        out -= key_vec.reshape(1, 1, 3) * (excess * spill_mask)[:, :, None] * 0.45

    restore = _clip01(settings.luminance_restore)
    if restore > 0:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        src_luma = np.sum(original_rgb * luma_weights.reshape(1, 1, 3), axis=2)
        out_luma = np.sum(np.clip(out, 0.0, 1.0) * luma_weights.reshape(1, 1, 3), axis=2)
        scale = np.divide(src_luma, np.maximum(out_luma, 1e-4), out=np.ones_like(src_luma), where=out_luma > 1e-4)
        scale = np.clip(scale, 0.60, 1.65)
        amount = (spill_mask * restore)[:, :, None]
        out = out * (1.0 - amount) + np.clip(out * scale[:, :, None], 0.0, 1.0) * amount
    return np.clip(out, 0.0, 1.0)


def _iter_tiles(
    h: int,
    w: int,
    settings: KeySettings,
    edge_radius: int,
) -> Iterator[tuple[slice, slice, slice, slice]]:
    tile_size = max(1, int(settings.tile_size))
    if not settings.use_tiling or max(h, w) <= tile_size:
        yield slice(0, h), slice(0, w), slice(0, h), slice(0, w)
        return
    overlap = max(int(settings.tile_overlap), int(edge_radius) * 4, 0)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            read_y = slice(max(0, y0 - overlap), min(h, y1 + overlap))
            read_x = slice(max(0, x0 - overlap), min(w, x1 + overlap))
            yield read_y, read_x, slice(y0, y1), slice(x0, x1)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


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


def checkerboard_composite(rgba: np.ndarray, cell: int = 18) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    board = ((xx // cell + yy // cell) % 2).astype(np.float32)
    bg = (0.78 + board[:, :, None] * 0.14).astype(np.float32)
    out = rgb * a + bg * (1.0 - a)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
