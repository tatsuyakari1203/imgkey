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
_MIN_TILE_LOCAL_INNER_PIXELS = 8
_MAX_TILE_LOCAL_INNER_LABEL_PIXELS = 8_000_000
_MAX_TILE_LOCAL_NEAREST_INNER_RADIUS = 256
_MAX_ALPHA_RECOVERY_BLOCK_PIXELS = 2_000_000
_LINEAR_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(slots=True)
class KeySettings:
    """Settings for the ImgKey classical keying pipeline.

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

    # v2 sampling controls.
    mode: str = "GraphicExact"
    sample_size: int = 5
    auto_border_sample: bool = True
    auto_detect_key_color: bool = False
    border_sample_width: int = 24
    # Local screen model builds a full-image uint8 screen map when the image is
    # small enough, and falls back to tile-local read-region estimates for large
    # tiled renders. It does not change matte probability decisions.
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

    # Optional imported matte. A grayscale matte is merged conservatively into
    # the classical connected-background pipeline as foreground protection and
    # alpha guidance.
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

    # v7 transition-unmix controls, appended for positional compatibility.
    transition_unmix: bool = True
    alpha_recover_strength: float = 0.85
    foreground_reference_pull: float = 0.65
    key_vector_despill: float = 0.75
    transition_spill_threshold: float = 0.08
    transition_reconstruction_error: float = 0.08
    foreground_reference_radius: int = 96
    foreground_candidate_count: int = 4
    transition_alpha_min: int = 2
    transition_alpha_max: int = 253
    preserve_foreground_luma: float = 0.85

    # v9 screen-residue cleanup. Off by default so connected-background mode can
    # still preserve disconnected key-colored foreground unless the app/profile
    # opts into aggressive cleanup explicitly.
    screen_cleanup_strength: float = 0.0
    screen_cleanup_similarity: int = 8

    # Optional no-AI compact CUDA DLL acceleration. Default is CPU/off so library
    # callers and tests never load the DLL unless they opt in explicitly.
    gpu_acceleration: str = "Off"


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
    gpu_acceleration: dict | None = None


@dataclass(slots=True)
class _GlobalMatte:
    screen_color: tuple[int, int, int]
    screen_probability: np.ndarray
    screen_map: np.ndarray | None
    background_mask: np.ndarray
    edge_mask: np.ndarray
    alpha: np.ndarray
    color_alpha: np.ndarray | None
    alpha_hint: np.ndarray | None
    fringe_mask: np.ndarray
    inner_labels: np.ndarray | None
    inner_label_to_flat: np.ndarray | None
    inner_distance: np.ndarray | None


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
    """Read a manual keep/remove/imported matte as uint8 grayscale.

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


def read_imported_matte_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read an imported foreground-protection matte as uint8 grayscale."""

    return read_grayscale_mask(path, shape)


def read_alpha_hint_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Backward-compatible alias for imported matte loading."""

    return read_imported_matte_mask(path, shape)


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
        include_debug=False,
    ).rgba


def _gpu_acceleration_mode(settings: KeySettings) -> str:
    raw = str(getattr(settings, "gpu_acceleration", "Off") or "Off").strip().lower().replace("_", " ")
    if raw in {"auto", "automatic"}:
        return "Auto"
    if raw in {"force", "force gpu", "forced", "on"}:
        return "Force GPU"
    return "Off"


def _new_gpu_stats(settings: KeySettings) -> dict:
    mode = _gpu_acceleration_mode(settings)
    return {
        "mode": mode,
        "status": "off" if mode == "Off" else "not_used",
        "backend": None,
        "attempted_tiles": 0,
        "used_tiles": 0,
        "fallback_tiles": 0,
        "error_tiles": 0,
        "elapsed_ms": 0.0,
        "last_reason": None,
        "last_message": "GPU acceleration is off; CPU color path used." if mode == "Off" else None,
    }


def _record_gpu_tile_result(gpu_stats: dict | None, result: dict) -> None:
    if gpu_stats is None:
        return
    gpu_stats["attempted_tiles"] = int(gpu_stats.get("attempted_tiles", 0)) + 1
    gpu_stats["backend"] = result.get("backend") or gpu_stats.get("backend")
    elapsed_ms = result.get("elapsed_ms")
    if elapsed_ms is not None:
        gpu_stats["elapsed_ms"] = float(gpu_stats.get("elapsed_ms", 0.0)) + float(elapsed_ms)
    if result.get("used"):
        gpu_stats["used_tiles"] = int(gpu_stats.get("used_tiles", 0)) + 1
        gpu_stats["status"] = "used"
    else:
        gpu_stats["fallback_tiles"] = int(gpu_stats.get("fallback_tiles", 0)) + 1
        reason = result.get("reason")
        if reason in {"cuda_dll_unavailable", "cuda_dll_probe_failed", "cuda_no_device", "cuda_unavailable", "cuda_execution_failed", "gpu_exception"}:
            gpu_stats["error_tiles"] = int(gpu_stats.get("error_tiles", 0)) + 1
        if gpu_stats.get("status") != "used":
            gpu_stats["status"] = "fallback"
    if result.get("reason"):
        gpu_stats["last_reason"] = result.get("reason")
    if result.get("message"):
        gpu_stats["last_message"] = result.get("message")


def _finalize_gpu_stats(settings: KeySettings, gpu_stats: dict | None) -> dict:
    stats = dict(gpu_stats or _new_gpu_stats(settings))
    mode = _gpu_acceleration_mode(settings)
    stats["mode"] = mode
    if mode == "Off":
        stats["status"] = "off"
        stats["message"] = "GPU acceleration is off; CPU color path used."
        return stats
    used = int(stats.get("used_tiles", 0))
    attempted = int(stats.get("attempted_tiles", 0))
    fallback = int(stats.get("fallback_tiles", 0))
    if used > 0:
        backend = stats.get("backend") or "GPU"
        stats["status"] = "used"
        stats["message"] = f"{backend} transition repair used on {used} tile(s); CPU remained the reference/fallback for {fallback} tile(s)."
    elif attempted > 0:
        stats["status"] = "fallback"
        stats["message"] = stats.get("last_message") or "GPU acceleration fell back to CPU for all attempted tiles."
    else:
        stats["status"] = "fallback"
        stats["message"] = "No GPU-eligible transition tile was encountered; CPU color path used."
    return stats


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
    include_debug: bool = True,
) -> KeyResult:
    """Run the v2 classical keying engine and return debug outputs.

    Global sampling, connected-background decisions, manual mask merging, and
    trimap/alpha generation happen once for the whole image. Full-resolution
    color unmix/despill then runs in overlapped tiles and writes only tile cores,
    avoiding a full-image float32 RGB working copy in the export path.
    Set ``include_debug=False`` for export/write-only callers to return only the
    RGBA output plus an alpha view and scalar metadata, without retaining debug
    masks or a foreground RGB copy.
    """

    settings = settings or KeySettings()
    rgb = _ensure_rgb_u8(rgb_u8)
    h, w = rgb.shape[:2]
    keep = _mask_to_bool(keep_mask, (h, w), "keep_mask")
    remove = _mask_to_bool(remove_mask, (h, w), "remove_mask")
    hint = _mask_to_u8(alpha_hint, (h, w), "alpha_hint")

    if settings.mode not in {"GraphicExact", "ProChroma", "ImportedMatte"}:
        raise ValueError(f"Unsupported keying mode: {settings.mode}")

    _raise_if_cancelled(cancel_callback)
    global_matte = _build_global_matte(rgb, settings, original_alpha, keep, remove, hint, progress_callback, cancel_callback)
    _report(progress_callback, 0.18, "global matte")
    _raise_if_cancelled(cancel_callback)

    crop = _normalized_crop(settings.full_res_crop, w, h)
    gpu_stats = _new_gpu_stats(settings)
    rgba, despill_mask = _render_tiled_rgba(
        rgb,
        settings,
        global_matte,
        progress_callback,
        cancel_callback,
        render_crop=crop,
        include_debug=include_debug,
        gpu_stats=gpu_stats,
    )
    if include_debug:
        foreground = rgba[:, :, :3].copy()
        if crop is not None:
            x0, y0, x1, y1 = crop
            alpha = global_matte.alpha[y0:y1, x0:x1].copy()
            background_mask = (global_matte.background_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
            edge_mask = (global_matte.edge_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
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
    else:
        foreground = None
        alpha = rgba[:, :, 3]
        background_mask = None
        edge_mask = None
        fringe_mask = None
        probability = None
        hint_out = None
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
        gpu_acceleration=_finalize_gpu_stats(settings, gpu_stats),
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

    screen_map = _estimate_screen_map(rgb, probability >= bg_threshold, screen_color, settings)
    _report(progress_callback, 0.17, "screen model")
    _raise_if_cancelled(cancel_callback)
    alpha = _apply_original_alpha(alpha, original_alpha)
    alpha, screen_cleanup = _apply_screen_residue_alpha_cleanup(
        rgb,
        alpha,
        probability,
        screen_color,
        screen_map,
        settings,
        keep_mask,
        remove_mask,
        alpha_hint,
        progress_callback,
        cancel_callback,
    )
    if screen_cleanup is not None and np.any(screen_cleanup):
        background |= screen_cleanup
        edge_mask[screen_cleanup] = False
        if keep_mask is not None:
            background &= ~keep_mask
            edge_mask[keep_mask] = False
        if remove_mask is not None:
            remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
            background |= remove_effective
            edge_mask[remove_effective] = False
    fringe_mask = _build_fringe_mask(rgb, alpha, edge_mask, probability, screen_color, settings, progress_callback, cancel_callback)
    _report(progress_callback, 0.175, "fringe map")
    _raise_if_cancelled(cancel_callback)
    inner_labels, inner_label_to_flat, inner_distance = _build_nearest_inner_reference_map(
        alpha,
        background,
        probability,
        fringe_mask,
        settings,
    )
    _report(progress_callback, 0.18, "inner color map")
    _raise_if_cancelled(cancel_callback)
    color_alpha = alpha.copy() if bool(settings.transition_unmix) and _clip01(settings.alpha_recover_strength) > 0 else None
    alpha = _recover_transition_alpha_global(
        rgb,
        alpha,
        background,
        edge_mask,
        probability,
        fringe_mask,
        screen_color,
        screen_map,
        inner_labels,
        inner_label_to_flat,
        inner_distance,
        settings,
        original_alpha,
        keep_mask,
        remove_mask,
        progress_callback,
        cancel_callback,
    )
    _report(progress_callback, 0.18, "transition alpha")
    return _GlobalMatte(
        screen_color=screen_color,
        screen_probability=probability,
        screen_map=screen_map,
        background_mask=background,
        edge_mask=edge_mask,
        alpha=alpha,
        color_alpha=color_alpha,
        alpha_hint=alpha_hint,
        fringe_mask=fringe_mask,
        inner_labels=inner_labels,
        inner_label_to_flat=inner_label_to_flat,
        inner_distance=inner_distance,
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


def _bool_mask_or_empty(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, -1] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(f"{name} must match alpha shape")
    if arr.dtype == bool:
        return arr.astype(bool, copy=False)
    return arr > 0


def _u8_mask_or_empty(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, -1] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(f"{name} must match alpha shape")
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8, copy=False)


def _build_foreground_core_mask(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    settings: KeySettings,
) -> np.ndarray:
    """Return protected opaque foreground-core pixels for transition repair."""

    alpha = np.asarray(alpha_u8, dtype=np.uint8)
    shape = alpha.shape
    background = _bool_mask_or_empty(background_mask, shape, "background_mask")
    probability_u8 = _u8_mask_or_empty(probability, shape, "probability")
    fringe_u8 = _u8_mask_or_empty(fringe_mask, shape, "fringe_mask")
    keep = _bool_mask_or_empty(keep_mask, shape, "keep_mask")
    remove = _bool_mask_or_empty(remove_mask, shape, "remove_mask")
    remove_effective = remove & ~keep

    prob_limit = max(64, int(round(_clip01(settings.clip_foreground) * 255.0)) + 32)
    core = (alpha >= 250) & (~background) & (probability_u8 <= prob_limit) & (fringe_u8 <= 24)
    if np.any(keep):
        core |= keep & (alpha >= 250)
    core &= ~remove_effective
    return core.astype(bool, copy=False)


def _build_transition_repair_mask(
    alpha_u8: np.ndarray,
    edge_mask: np.ndarray,
    fringe_mask: np.ndarray,
    spill_strength: np.ndarray,
    background_mask: np.ndarray,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    foreground_core_mask: np.ndarray,
    settings: KeySettings,
) -> np.ndarray:
    """Return pixels eligible for future v7 transition/fringe repair.

    The mask only describes where repair may run; it does not modify alpha.
    """

    alpha = np.asarray(alpha_u8, dtype=np.uint8)
    shape = alpha.shape
    edge = _bool_mask_or_empty(edge_mask, shape, "edge_mask")
    fringe_u8 = _u8_mask_or_empty(fringe_mask, shape, "fringe_mask")
    spill = np.asarray(spill_strength, dtype=np.float32)
    if spill.shape != shape:
        raise ValueError("spill_strength must match alpha shape")
    background = _bool_mask_or_empty(background_mask, shape, "background_mask")
    keep = _bool_mask_or_empty(keep_mask, shape, "keep_mask")
    remove = _bool_mask_or_empty(remove_mask, shape, "remove_mask")
    foreground_core = _bool_mask_or_empty(foreground_core_mask, shape, "foreground_core_mask")
    remove_effective = remove & ~keep

    alpha_min = int(np.clip(int(settings.transition_alpha_min), 0, 255))
    alpha_max = int(np.clip(int(settings.transition_alpha_max), 0, 255))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    semi = (alpha >= alpha_min) & (alpha <= alpha_max)
    protected_semi = semi & (alpha < 240)
    live = (alpha > 0) & (~background) & (~remove_effective)
    live_edge = edge & live
    live_fringe = (fringe_u8 > 0) & live
    protected_core_fringe = (fringe_u8 > 24) & live
    live_spill = (spill > float(settings.transition_spill_threshold)) & live
    eligible = semi | live_edge | live_fringe | live_spill

    near_opaque_core = (alpha >= 240) & (~background) & (fringe_u8 <= 24)
    protected_core = (foreground_core | near_opaque_core) & (alpha >= 240)
    core_allowed = (~protected_core) | protected_semi | protected_core_fringe
    return (live & eligible & core_allowed).astype(bool, copy=False)


def _build_nearest_inner_label_map(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Backward-compatible global nearest-inner labels for legacy callers."""

    labels, label_to_flat, _ = _build_nearest_inner_reference_map(
        alpha_u8,
        background_mask,
        probability,
        fringe_mask,
        settings,
    )
    return labels, label_to_flat


def _build_nearest_inner_reference_map(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Map repair pixels to a clean foreground seed plus optional radius map.

    OpenCV's label image is retained globally; RGB is gathered lazily from the
    original uint8 source per tile/stripe so export never materializes a full
    repaired float/RGB debug image. v7 transition recovery builds the same
    reference even when legacy color-pull sliders are disabled, and stores a
    compact clipped distance map for foreground-reference radius checks.
    """

    legacy_enabled = _legacy_inner_repair_enabled(settings)
    transition_enabled = _transition_reference_enabled(settings)
    if not legacy_enabled and not transition_enabled:
        return None, None, None
    if not transition_enabled and not np.any(fringe_mask > 0):
        return None, None, None
    # Beyond this size, retaining global labels plus a label->source index table
    # can dominate memory; use deterministic tile-local references instead.
    if alpha_u8.size > _MAX_INNER_LABEL_PIXELS:
        return None, None, None
    inner = _nearest_inner_seed_mask(alpha_u8, background_mask, probability, fringe_mask, settings)
    if np.count_nonzero(inner) == 0:
        return None, None, None
    try:
        src = np.where(inner, 0, 255).astype(np.uint8)
        distances, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    except (cv2.error, MemoryError):
        return None, None, None
    labels = np.ascontiguousarray(labels.astype(np.int32, copy=False))
    label_to_flat = _nearest_inner_label_to_flat(labels, inner)
    if label_to_flat is None:
        return None, None, None

    distance_u16: np.ndarray | None = None
    if transition_enabled:
        radius = _foreground_reference_radius(settings)
        if radius > 0:
            clip_to = min(max(radius + 1, 0), np.iinfo(np.uint16).max)
            distance_u16 = np.ceil(np.clip(distances, 0.0, float(clip_to))).astype(np.uint16)
            distance_u16 = np.ascontiguousarray(distance_u16)
    return labels, label_to_flat, distance_u16


def _nearest_inner_seed_mask(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
) -> np.ndarray:
    prob_limit = max(64, int(round(_clip01(settings.clip_foreground) * 255.0)) + 32)
    return (alpha_u8 >= 250) & (~background_mask) & (fringe_mask <= 24) & (probability <= prob_limit)


def _legacy_inner_repair_enabled(settings: KeySettings) -> bool:
    return _clip01(settings.inner_color_pull) > 0 and _clip01(settings.edge_color_repair) > 0


def _transition_reference_enabled(settings: KeySettings) -> bool:
    return bool(settings.transition_unmix) and _foreground_reference_radius(settings) > 0


def _foreground_reference_radius(settings: KeySettings) -> int:
    # Reserve uint16 max as the clipped "beyond radius" sentinel.
    return int(np.clip(int(settings.foreground_reference_radius), 0, np.iinfo(np.uint16).max - 1))


def _nearest_inner_label_to_flat(labels: np.ndarray, inner_mask: np.ndarray) -> np.ndarray | None:
    inner_flat = np.flatnonzero(inner_mask.reshape(-1))
    if inner_flat.size == 0:
        return None
    inner_labels = labels.reshape(-1)[inner_flat]
    valid = inner_labels > 0
    if not np.any(valid):
        return None
    inner_flat = inner_flat[valid]
    inner_labels = inner_labels[valid]
    max_label = int(inner_labels.max())
    label_to_flat = np.full(max_label + 1, -1, dtype=np.int64)
    label_to_flat[inner_labels] = inner_flat.astype(np.int64, copy=False)
    return label_to_flat


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


def _foreground_reference_for_slice(
    rgb: np.ndarray,
    labels: np.ndarray | None,
    label_to_flat: np.ndarray | None,
    distance_u16: np.ndarray | None,
    read_y: slice,
    read_x: slice,
    max_radius: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    foreground_ref_rgb, foreground_ref_valid = _nearest_inner_rgb_for_slice(rgb, labels, label_to_flat, read_y, read_x)
    if foreground_ref_rgb is None or foreground_ref_valid is None:
        return None, None, None
    foreground_ref_distance = None if distance_u16 is None else distance_u16[read_y, read_x]
    radius = int(max_radius)
    if foreground_ref_distance is not None and radius > 0:
        foreground_ref_valid = foreground_ref_valid & (foreground_ref_distance <= radius)
        foreground_ref_rgb = foreground_ref_rgb.copy()
        foreground_ref_rgb[~foreground_ref_valid] = 0
    if not np.any(foreground_ref_valid):
        return foreground_ref_rgb, foreground_ref_valid, foreground_ref_distance
    return foreground_ref_rgb, foreground_ref_valid.astype(bool, copy=False), foreground_ref_distance


def _build_tile_local_nearest_inner_rgb(
    rgb_tile: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
    max_radius: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    foreground_ref_rgb, foreground_ref_valid, _ = _build_tile_local_nearest_inner_reference(
        rgb_tile,
        alpha_u8,
        background_mask,
        probability,
        fringe_mask,
        settings,
        max_radius,
    )
    return foreground_ref_rgb, foreground_ref_valid


def _build_tile_local_nearest_inner_reference(
    rgb_tile: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: KeySettings,
    max_radius: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    legacy_enabled = _legacy_inner_repair_enabled(settings)
    transition_enabled = _transition_reference_enabled(settings)
    if not legacy_enabled and not transition_enabled:
        return None, None, None
    if not transition_enabled and not np.any(fringe_mask > 0):
        return None, None, None
    radius = int(max_radius)
    if radius <= 0:
        return None, None, None
    if alpha_u8.size > _MAX_TILE_LOCAL_INNER_LABEL_PIXELS:
        return None, None, None
    inner = _nearest_inner_seed_mask(alpha_u8, background_mask, probability, fringe_mask, settings)
    if np.count_nonzero(inner) < _MIN_TILE_LOCAL_INNER_PIXELS:
        return None, None, None
    try:
        src = np.where(inner, 0, 255).astype(np.uint8)
        distances, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    except (cv2.error, MemoryError):
        return None, None, None

    labels = np.ascontiguousarray(labels.astype(np.int32, copy=False))
    label_to_flat = _nearest_inner_label_to_flat(labels, inner)
    if label_to_flat is None:
        return None, None, None

    valid = (labels > 0) & (labels < len(label_to_flat)) & (distances <= float(radius))
    if not np.any(valid):
        distance_u16 = np.ceil(np.clip(distances, 0.0, float(min(radius + 1, np.iinfo(np.uint16).max)))).astype(np.uint16)
        return np.zeros((*labels.shape, 3), dtype=np.uint8), valid.astype(bool, copy=False), distance_u16
    flat_tile = np.full(labels.shape, -1, dtype=np.int64)
    flat_tile[valid] = label_to_flat[labels[valid]]
    valid &= flat_tile >= 0
    if not np.any(valid):
        distance_u16 = np.ceil(np.clip(distances, 0.0, float(min(radius + 1, np.iinfo(np.uint16).max)))).astype(np.uint16)
        return np.zeros((*labels.shape, 3), dtype=np.uint8), valid.astype(bool, copy=False), distance_u16
    nearest = np.zeros((*labels.shape, 3), dtype=np.uint8)
    nearest[valid] = rgb_tile.reshape(-1, 3)[flat_tile[valid]]
    distance_u16 = np.ceil(np.clip(distances, 0.0, float(min(radius + 1, np.iinfo(np.uint16).max)))).astype(np.uint16)
    return nearest, valid.astype(bool, copy=False), distance_u16


def _recover_transition_alpha_global(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_map: np.ndarray | None,
    inner_labels: np.ndarray | None,
    inner_label_to_flat: np.ndarray | None,
    inner_distance: np.ndarray | None,
    settings: KeySettings,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> np.ndarray:
    strength = _clip01(settings.alpha_recover_strength)
    radius = _foreground_reference_radius(settings)
    if not bool(settings.transition_unmix) or strength <= 0.0 or radius <= 0:
        return alpha_u8

    alpha = alpha_u8.copy()
    foreground_core = _build_foreground_core_mask(
        alpha_u8,
        background_mask,
        probability,
        fringe_mask,
        keep_mask,
        remove_mask,
        settings,
    )

    if inner_labels is not None and inner_label_to_flat is not None:
        h, w = alpha.shape
        stripe_rows = _alpha_recovery_stripe_rows(h, w)
        stripes = list(range(0, h, stripe_rows))
        total = max(1, len(stripes))
        for index, y0 in enumerate(stripes, start=1):
            _raise_if_cancelled(cancel_callback)
            y1 = min(h, y0 + stripe_rows)
            read_y = slice(y0, y1)
            read_x = slice(0, w)
            foreground_ref_rgb, foreground_ref_valid, foreground_ref_distance = _foreground_reference_for_slice(
                rgb,
                inner_labels,
                inner_label_to_flat,
                inner_distance,
                read_y,
                read_x,
                radius,
            )
            _recover_transition_alpha_block(
                rgb[read_y, read_x],
                alpha[read_y, read_x],
                background_mask[read_y, read_x],
                edge_mask[read_y, read_x],
                probability[read_y, read_x],
                fringe_mask[read_y, read_x],
                screen_color,
                None if screen_map is None else screen_map[read_y, read_x],
                foreground_ref_rgb,
                foreground_ref_valid,
                foreground_ref_distance,
                foreground_core[read_y, read_x],
                None if keep_mask is None else keep_mask[read_y, read_x],
                None if remove_mask is None else remove_mask[read_y, read_x],
                settings,
            )
            _report(progress_callback, 0.175 + 0.005 * (index / total), "transition alpha")
    else:
        _recover_transition_alpha_tile_local(
            rgb,
            alpha,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe_mask,
            screen_color,
            screen_map,
            foreground_core,
            settings,
            keep_mask,
            remove_mask,
            progress_callback,
            cancel_callback,
            radius,
        )

    return _finalize_recovered_transition_alpha(
        alpha,
        alpha_u8,
        background_mask,
        edge_mask,
        probability,
        original_alpha,
        keep_mask,
        remove_mask,
        settings,
    )


def _alpha_recovery_stripe_rows(h: int, w: int) -> int:
    if h <= 0 or w <= 0:
        return 1
    rows_by_pixels = max(1, _MAX_ALPHA_RECOVERY_BLOCK_PIXELS // max(1, int(w)))
    return int(max(1, min(int(h), 512, rows_by_pixels)))


def _recover_transition_alpha_tile_local(
    rgb: np.ndarray,
    alpha: np.ndarray,
    baseline_alpha: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_map: np.ndarray | None,
    foreground_core: np.ndarray,
    settings: KeySettings,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    radius: int,
) -> None:
    h, w = alpha.shape
    if h == 0 or w == 0:
        return
    tile_size = max(1, min(int(settings.tile_size), 512))
    overlap = max(int(radius), int(_screen_model_radius_for_shape((h, w)) if settings.local_screen_model and screen_map is None else 0))
    tiles: list[tuple[slice, slice, slice, slice]] = []
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            tiles.append(
                (
                    slice(max(0, y0 - overlap), min(h, y1 + overlap)),
                    slice(max(0, x0 - overlap), min(w, x1 + overlap)),
                    slice(y0, y1),
                    slice(x0, x1),
                )
            )
    total = max(1, len(tiles))
    for index, (read_y, read_x, core_y, core_x) in enumerate(tiles, start=1):
        _raise_if_cancelled(cancel_callback)
        bounded_radius = _bounded_tile_local_nearest_inner_radius(radius, read_y, read_x, core_y, core_x, (h, w))
        foreground_ref_rgb, foreground_ref_valid, foreground_ref_distance = _build_tile_local_nearest_inner_reference(
            rgb[read_y, read_x],
            baseline_alpha[read_y, read_x],
            background_mask[read_y, read_x],
            probability[read_y, read_x],
            fringe_mask[read_y, read_x],
            settings,
            bounded_radius,
        )
        if foreground_ref_rgb is None or foreground_ref_valid is None:
            _report(progress_callback, 0.175 + 0.005 * (index / total), "transition alpha")
            continue
        rel_y = slice(core_y.start - read_y.start, core_y.stop - read_y.start)
        rel_x = slice(core_x.start - read_x.start, core_x.stop - read_x.start)
        screen_tile = None if screen_map is None else screen_map[core_y, core_x]
        if screen_tile is None and settings.local_screen_model:
            screen_read = _estimate_screen_tile(
                rgb[read_y, read_x],
                background_mask[read_y, read_x],
                screen_color,
                _screen_model_radius_for_shape((h, w)),
            )
            screen_tile = screen_read[rel_y, rel_x]
        _recover_transition_alpha_block(
            rgb[core_y, core_x],
            alpha[core_y, core_x],
            background_mask[core_y, core_x],
            edge_mask[core_y, core_x],
            probability[core_y, core_x],
            fringe_mask[core_y, core_x],
            screen_color,
            screen_tile,
            foreground_ref_rgb[rel_y, rel_x],
            foreground_ref_valid[rel_y, rel_x],
            None if foreground_ref_distance is None else foreground_ref_distance[rel_y, rel_x],
            foreground_core[core_y, core_x],
            None if keep_mask is None else keep_mask[core_y, core_x],
            None if remove_mask is None else remove_mask[core_y, core_x],
            settings,
        )
        _report(progress_callback, 0.175 + 0.005 * (index / total), "transition alpha")


def _recover_transition_alpha_block(
    rgb_block: np.ndarray,
    alpha_block: np.ndarray,
    background_block: np.ndarray,
    edge_block: np.ndarray,
    probability_block: np.ndarray,
    fringe_block: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_block: np.ndarray | None,
    foreground_ref_rgb: np.ndarray | None,
    foreground_ref_valid: np.ndarray | None,
    foreground_ref_distance: np.ndarray | None,
    foreground_core_block: np.ndarray,
    keep_block: np.ndarray | None,
    remove_block: np.ndarray | None,
    settings: KeySettings,
) -> None:
    if foreground_ref_rgb is None or foreground_ref_valid is None or not np.any(foreground_ref_valid):
        return
    spill = _compute_key_spill_strength(rgb_block, screen_color)
    transition = _build_transition_repair_mask(
        alpha_block,
        edge_block,
        fringe_block,
        spill,
        background_block,
        keep_block,
        remove_block,
        foreground_core_block,
        settings,
    )
    eligible = transition & foreground_ref_valid & (alpha_block > 0)
    if foreground_ref_distance is not None:
        eligible &= foreground_ref_distance <= _foreground_reference_radius(settings)
    if not np.any(eligible):
        return

    source_linear = _srgb_u8_to_linear_f32(rgb_block)
    foreground_linear = _srgb_u8_to_linear_f32(foreground_ref_rgb)
    if screen_block is None:
        screen_u8 = np.empty_like(rgb_block)
        screen_u8[:, :, :] = np.asarray(screen_color, dtype=np.uint8).reshape(1, 1, 3)
        screen_linear = _srgb_u8_to_linear_f32(screen_u8)
    else:
        screen_linear = _srgb_u8_to_linear_f32(screen_block)

    i = source_linear[eligible]
    b = screen_linear[eligible]
    f = foreground_linear[eligible]
    v = f - b
    denom = np.sum(v * v, axis=1)
    stable = denom > 1e-6
    if not np.any(stable):
        return
    solved = np.zeros(denom.shape, dtype=np.float32)
    solved[stable] = np.sum((i[stable] - b[stable]) * v[stable], axis=1) / denom[stable]
    solved = np.clip(solved, 0.0, 1.0)
    recon = solved[:, None] * f + (1.0 - solved[:, None]) * b
    err = np.linalg.norm(i - recon, axis=1)
    plausible = stable & (err < float(settings.transition_reconstruction_error))
    if not np.any(plausible):
        return

    current_u8 = alpha_block[eligible]
    current = current_u8.astype(np.float32) / 255.0
    gain = np.maximum(solved - current, 0.0)
    recover = current + _clip01(settings.alpha_recover_strength) * gain
    recovered_u8 = np.rint(np.clip(recover, 0.0, 1.0) * 255.0).astype(np.uint8)
    updated = current_u8.copy()
    updated[plausible] = np.maximum(updated[plausible], recovered_u8[plausible])
    alpha_block[eligible] = updated


def _finalize_recovered_transition_alpha(
    alpha: np.ndarray,
    baseline_alpha: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    settings: KeySettings,
) -> np.ndarray:
    out = np.maximum(alpha.astype(np.uint8, copy=False), baseline_alpha).astype(np.uint8, copy=True)
    bg_threshold = int(round(_clip01(settings.clip_background) * 255.0))
    known_background = (background_mask & ~edge_mask) | ((probability >= bg_threshold) & background_mask)
    out[baseline_alpha <= 0] = 0
    out[known_background] = 0
    if remove_mask is not None:
        remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
        out[remove_effective] = 0
    if keep_mask is not None:
        out[keep_mask] = 255
    out = _cap_alpha_to_original(out, original_alpha)
    return out


def _cap_alpha_to_original(alpha_u8: np.ndarray, original_alpha: np.ndarray | None) -> np.ndarray:
    if original_alpha is None:
        return alpha_u8
    original = np.asarray(original_alpha, dtype=np.float32)
    if original.shape != alpha_u8.shape:
        original = cv2.resize(original, (alpha_u8.shape[1], alpha_u8.shape[0]), interpolation=cv2.INTER_AREA)
    cap = np.rint(np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.minimum(alpha_u8, cap).astype(np.uint8, copy=False)


def _render_tiled_rgba(
    rgb: np.ndarray,
    settings: KeySettings,
    matte: _GlobalMatte,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    *,
    render_crop: tuple[int, int, int, int] | None = None,
    include_debug: bool = True,
    gpu_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = rgb.shape[:2]
    crop = _normalized_crop(render_crop, w, h)
    if crop is None:
        out_x0 = out_y0 = 0
        out_h, out_w = h, w
        alpha_out = matte.alpha
    else:
        out_x0, out_y0, out_x1, out_y1 = crop
        out_h, out_w = out_y1 - out_y0, out_x1 - out_x0
        alpha_out = matte.alpha[out_y0:out_y1, out_x0:out_x1]

    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha_out
    despill_mask = np.zeros((out_h, out_w), dtype=np.uint8) if include_debug else None
    reference_alpha = matte.alpha if matte.color_alpha is None else matte.color_alpha
    screen_radius = _screen_model_radius_for_shape((h, w)) if settings.local_screen_model and matte.screen_map is None else 0
    local_nearest_radius = _tile_local_nearest_inner_radius(settings) if matte.inner_labels is None else 0
    extra_overlap = _tile_extra_overlap(settings, (h, w), screen_radius, local_nearest_radius)
    tiles = list(_iter_tiles(h, w, settings, _effective_edge_radius(settings), extra_overlap=extra_overlap))
    if crop is not None:
        tiles = [tile for tile in tiles if _tile_intersects_crop(tile[2], tile[3], crop)]
    total = max(1, len(tiles))
    for index, tile in enumerate(tiles, start=1):
        _raise_if_cancelled(cancel_callback)
        read_y, read_x, core_y, core_x = tile
        if crop is None:
            write_y0, write_y1 = core_y.start, core_y.stop
            write_x0, write_x1 = core_x.start, core_x.stop
        else:
            crop_x0, crop_y0, crop_x1, crop_y1 = crop
            write_y0 = max(core_y.start, crop_y0)
            write_y1 = min(core_y.stop, crop_y1)
            write_x0 = max(core_x.start, crop_x0)
            write_x1 = min(core_x.stop, crop_x1)
            if write_y1 <= write_y0 or write_x1 <= write_x0:
                continue
        rel_y = slice(write_y0 - read_y.start, write_y1 - read_y.start)
        rel_x = slice(write_x0 - read_x.start, write_x1 - read_x.start)
        out_y = slice(write_y0 - out_y0, write_y1 - out_y0)
        out_x = slice(write_x0 - out_x0, write_x1 - out_x0)
        rgb_read = rgb[read_y, read_x]
        if matte.screen_map is not None:
            screen_tile = matte.screen_map[read_y, read_x]
        elif settings.local_screen_model:
            screen_tile = _estimate_screen_tile(
                rgb_read,
                matte.background_mask[read_y, read_x],
                matte.screen_color,
                screen_radius,
            )
        else:
            screen_tile = None
        if matte.inner_labels is not None and matte.inner_label_to_flat is not None:
            nearest_inner_rgb, nearest_inner_valid = _nearest_inner_rgb_for_slice(
                rgb,
                matte.inner_labels,
                matte.inner_label_to_flat,
                read_y,
                read_x,
            )
            if _transition_reference_enabled(settings):
                transition_inner_rgb, transition_inner_valid, _ = _foreground_reference_for_slice(
                    rgb,
                    matte.inner_labels,
                    matte.inner_label_to_flat,
                    matte.inner_distance,
                    read_y,
                    read_x,
                    _foreground_reference_radius(settings),
                )
            else:
                transition_inner_rgb, transition_inner_valid = nearest_inner_rgb, nearest_inner_valid
        else:
            bounded_local_radius = _bounded_tile_local_nearest_inner_radius(
                local_nearest_radius,
                read_y,
                read_x,
                core_y,
                core_x,
                (h, w),
            )
            if _can_build_tile_local_nearest_inner(read_y, read_x, core_y, core_x, (h, w)):
                nearest_inner_rgb, nearest_inner_valid = _build_tile_local_nearest_inner_rgb(
                    rgb_read,
                    reference_alpha[read_y, read_x],
                    matte.background_mask[read_y, read_x],
                    matte.screen_probability[read_y, read_x],
                    matte.fringe_mask[read_y, read_x],
                    settings,
                    bounded_local_radius,
                )
                if _transition_reference_enabled(settings):
                    bounded_transition_radius = _bounded_tile_local_nearest_inner_radius(
                        _foreground_reference_radius(settings),
                        read_y,
                        read_x,
                        core_y,
                        core_x,
                        (h, w),
                    )
                    if bounded_transition_radius == bounded_local_radius:
                        transition_inner_rgb, transition_inner_valid = nearest_inner_rgb, nearest_inner_valid
                    elif bounded_transition_radius > 0:
                        transition_inner_rgb, transition_inner_valid = _build_tile_local_nearest_inner_rgb(
                            rgb_read,
                            reference_alpha[read_y, read_x],
                            matte.background_mask[read_y, read_x],
                            matte.screen_probability[read_y, read_x],
                            matte.fringe_mask[read_y, read_x],
                            settings,
                            bounded_transition_radius,
                        )
                    else:
                        transition_inner_rgb, transition_inner_valid = None, None
                else:
                    transition_inner_rgb, transition_inner_valid = nearest_inner_rgb, nearest_inner_valid
            else:
                nearest_inner_rgb, nearest_inner_valid = None, None
                transition_inner_rgb, transition_inner_valid = None, None
        rgb_tile, spill_tile = _process_color_tile(
            rgb_read,
            matte.alpha[read_y, read_x],
            matte.background_mask[read_y, read_x],
            matte.edge_mask[read_y, read_x],
            matte.screen_probability[read_y, read_x],
            matte.fringe_mask[read_y, read_x],
            screen_tile,
            nearest_inner_rgb,
            nearest_inner_valid,
            matte.screen_color,
            settings,
            transition_nearest_rgb=transition_inner_rgb,
            transition_nearest_valid=transition_inner_valid,
            gpu_stats=gpu_stats,
        )
        rgba[out_y, out_x, :3] = rgb_tile[rel_y, rel_x]
        if despill_mask is not None:
            despill_mask[out_y, out_x] = spill_tile[rel_y, rel_x]
        _report(progress_callback, 0.18 + 0.82 * (index / total), f"tile {index}/{total}")
    rgba[alpha_out <= 0, :3] = 0
    return rgba, despill_mask


def _tile_intersects_crop(core_y: slice, core_x: slice, crop: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = crop
    return core_y.start < y1 and core_y.stop > y0 and core_x.start < x1 and core_x.stop > x0


def _tile_extra_overlap(
    settings: KeySettings,
    shape: tuple[int, int],
    screen_radius: int | None = None,
    local_nearest_radius: int | None = None,
) -> int:
    if screen_radius is None:
        screen_radius = _screen_model_radius_for_shape(shape) if settings.local_screen_model else 0
    if local_nearest_radius is None:
        local_nearest_radius = _tile_local_nearest_inner_radius(settings)
    guided_radius = max(0, int(settings.guided_radius)) * 2 + 2 if _clip01(settings.guided_alpha_refine) > 0 else 0
    return max(
        int(screen_radius),
        max(0, int(settings.fringe_band_radius)),
        guided_radius,
        int(local_nearest_radius),
    )


def _tile_local_nearest_inner_radius(settings: KeySettings) -> int:
    legacy_enabled = _legacy_inner_repair_enabled(settings)
    transition_radius = _foreground_reference_radius(settings) if _transition_reference_enabled(settings) else 0
    if not legacy_enabled and transition_radius <= 0:
        return 0
    edge_radius = _effective_edge_radius(settings)
    fringe_radius = max(0, int(settings.fringe_band_radius))
    legacy_radius = max(_MIN_TILE_LOCAL_INNER_PIXELS, edge_radius * 4, edge_radius + fringe_radius) if legacy_enabled else 0
    radius = max(legacy_radius, transition_radius)
    return int(min(radius, _MAX_TILE_LOCAL_NEAREST_INNER_RADIUS))


def _bounded_tile_local_nearest_inner_radius(
    radius: int,
    read_y: slice,
    read_x: slice,
    core_y: slice,
    core_x: slice,
    shape: tuple[int, int],
) -> int:
    base = int(radius)
    if base <= 0:
        return 0
    h, w = shape
    margins: list[int] = []
    if core_y.start > 0:
        margins.append(core_y.start - read_y.start)
    if core_y.stop < h:
        margins.append(read_y.stop - core_y.stop)
    if core_x.start > 0:
        margins.append(core_x.start - read_x.start)
    if core_x.stop < w:
        margins.append(read_x.stop - core_x.stop)
    positive = [int(margin) for margin in margins if margin > 0]
    if not positive:
        return base
    return max(0, min(base, min(positive)))


def _can_build_tile_local_nearest_inner(
    read_y: slice,
    read_x: slice,
    core_y: slice,
    core_x: slice,
    shape: tuple[int, int],
) -> bool:
    read_h = int(read_y.stop - read_y.start)
    read_w = int(read_x.stop - read_x.start)
    if read_h <= 0 or read_w <= 0:
        return False
    if read_h * read_w > _MAX_TILE_LOCAL_INNER_LABEL_PIXELS:
        return False
    h, w = shape
    whole_read = read_y.start <= 0 and read_x.start <= 0 and read_y.stop >= h and read_x.stop >= w
    whole_core = core_y.start <= 0 and core_x.start <= 0 and core_y.stop >= h and core_x.stop >= w
    return not (whole_read and whole_core)


def _screen_linear_for_tile(
    shape: tuple[int, int],
    screen_color: tuple[int, int, int],
    screen_tile: np.ndarray | None,
) -> np.ndarray:
    h, w = shape
    if screen_tile is None:
        screen_u8 = np.empty((h, w, 3), dtype=np.uint8)
        screen_u8[:, :, :] = np.asarray(screen_color, dtype=np.uint8).reshape(1, 1, 3)
        return _srgb_u8_to_linear_f32(screen_u8)
    return _srgb_u8_to_linear_f32(screen_tile)


def _linear_luma(rgb_linear: np.ndarray) -> np.ndarray:
    return np.sum(np.clip(rgb_linear, 0.0, 1.0) * _LINEAR_LUMA_WEIGHTS.reshape(1, 1, 3), axis=2).astype(np.float32)


def _match_luma_linear(rgb_linear: np.ndarray, target_luma: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float32), 0.0, 1.0)
    src_luma = _linear_luma(rgb)
    target = np.clip(np.asarray(target_luma, dtype=np.float32), 0.0, 1.0)
    scale = np.divide(target, np.maximum(src_luma, 1e-5), out=np.ones_like(target), where=src_luma > 1e-5)
    scale = np.clip(scale, 0.0, 4.0)
    return np.clip(rgb * scale[:, :, None], 0.0, 1.0)


def _screen_chroma_unit_vectors(screen_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key_luma = _linear_luma(screen_linear)
    key_vec = np.clip(screen_linear, 0.0, 1.0) - key_luma[:, :, None]
    norm = np.linalg.norm(key_vec, axis=2).astype(np.float32)
    valid = norm >= 1e-5
    unit = np.divide(key_vec, np.maximum(norm[:, :, None], 1e-5), out=np.zeros_like(key_vec), where=valid[:, :, None])
    return unit.astype(np.float32, copy=False), valid


def _repair_transition_unmix(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_color: tuple[int, int, int],
    screen_tile: np.ndarray | None,
    nearest_fg_rgb: np.ndarray | None,
    nearest_fg_valid: np.ndarray | None,
    settings: KeySettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return repaired RGB plus the transition repair mask; alpha is read-only."""

    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_u8 must have shape HxWx3")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb[:, :, :3]
    shape = rgb.shape[:2]
    repair_mask = np.zeros(shape, dtype=np.uint8)
    original_rgb = rgb.copy()

    if not bool(settings.transition_unmix):
        return original_rgb, repair_mask
    if _foreground_reference_radius(settings) <= 0:
        return original_rgb, repair_mask
    if nearest_fg_rgb is None or nearest_fg_valid is None:
        return original_rgb, repair_mask

    foreground_valid = _bool_mask_or_empty(nearest_fg_valid, shape, "nearest_fg_valid")
    if not np.any(foreground_valid):
        return original_rgb, repair_mask

    foreground_rgb = np.asarray(nearest_fg_rgb)
    if foreground_rgb.ndim != 3 or foreground_rgb.shape[:2] != shape or foreground_rgb.shape[2] < 3:
        raise ValueError("nearest_fg_rgb must match rgb_u8 shape")
    foreground_rgb = foreground_rgb[:, :, :3]
    if foreground_rgb.dtype != np.uint8:
        foreground_rgb = np.clip(foreground_rgb, 0, 255).astype(np.uint8)

    alpha = _u8_mask_or_empty(alpha_u8, shape, "alpha_u8")
    background = _bool_mask_or_empty(background_mask, shape, "background_mask")
    edge = _bool_mask_or_empty(edge_mask, shape, "edge_mask")
    probability_u8 = _u8_mask_or_empty(probability, shape, "probability")
    fringe_u8 = _u8_mask_or_empty(fringe_mask, shape, "fringe_mask")

    spill_strength = _compute_key_spill_strength(rgb, screen_color)
    foreground_core = _build_foreground_core_mask(alpha, background, probability_u8, fringe_u8, None, None, settings)
    transition = _build_transition_repair_mask(
        alpha,
        edge,
        fringe_u8,
        spill_strength,
        background,
        None,
        None,
        foreground_core,
        settings,
    )
    eligible = transition & foreground_valid & (alpha > 0)
    if not np.any(eligible):
        return original_rgb, repair_mask

    source_linear = _srgb_u8_to_linear_f32(rgb)
    foreground_linear = _srgb_u8_to_linear_f32(foreground_rgb)
    screen_linear = _screen_linear_for_tile(shape, screen_color, screen_tile)
    alpha_f = alpha.astype(np.float32) / 255.0
    safe_alpha = np.maximum(alpha_f, 1.0 / 255.0)
    foreground_est = (source_linear - (1.0 - alpha_f[:, :, None]) * screen_linear) / safe_alpha[:, :, None]
    foreground_est = np.nan_to_num(foreground_est, nan=0.0, posinf=1.0, neginf=0.0)
    foreground_est = np.clip(foreground_est, 0.0, 1.0).astype(np.float32, copy=False)

    recon = alpha_f[:, :, None] * foreground_est + (1.0 - alpha_f[:, :, None]) * screen_linear
    recon_error = np.linalg.norm(source_linear - recon, axis=2)
    reconstruction_limit = max(float(settings.transition_reconstruction_error) * 1.25, 1e-4)
    eligible &= recon_error <= reconstruction_limit
    if not np.any(eligible):
        return original_rgb, repair_mask

    key_vec, key_vec_valid = _screen_chroma_unit_vectors(screen_linear)
    foreground_luma = _linear_luma(foreground_est)
    reference_luma = _linear_luma(foreground_linear)
    foreground_chroma = foreground_est - foreground_luma[:, :, None]
    vector_spill = np.maximum(np.sum(foreground_chroma * key_vec, axis=2), 0.0)
    vector_spill = np.where(key_vec_valid, vector_spill, 0.0).astype(np.float32)

    edge_strength = np.clip(alpha_f * (1.0 - alpha_f) * 4.0, 0.0, 1.0)
    edge_strength = np.maximum(edge_strength, edge.astype(np.float32) * 0.45)
    fringe_signal = fringe_u8.astype(np.float32) / 255.0
    near_screen = (probability_u8.astype(np.float32) / 255.0) * np.clip(1.0 - alpha_f, 0.0, 1.0)
    spill_gate = np.maximum.reduce(
        (
            np.clip(spill_strength, 0.0, 1.0),
            _smoothstep(0.005, 0.18, vector_spill),
            near_screen,
            fringe_signal * 0.75,
        )
    )
    transition_strength = np.maximum.reduce((edge_strength, fringe_signal, near_screen))
    repair_strength = np.clip(transition_strength * np.maximum(spill_gate, 0.35), 0.0, 1.0)
    repair_strength = np.where(eligible, repair_strength, 0.0).astype(np.float32)
    if not np.any(repair_strength > 0):
        return original_rgb, repair_mask

    cleaned = foreground_est.copy()
    despill_amount = _clip01(settings.key_vector_despill)
    if despill_amount > 0:
        cleaned -= key_vec * (vector_spill * despill_amount * repair_strength)[:, :, None]
        cleaned = np.clip(cleaned, 0.0, 1.0)

    pull_amount = _clip01(settings.foreground_reference_pull)
    if pull_amount > 0:
        # ``repair_strength`` already bakes in the spill/edge gate used by the
        # compact CUDA ABI. Applying ``spill_gate`` a second time makes the CPU
        # full-keyer path diverge from the parity-tested GPU tile kernel.
        pull = np.clip(repair_strength * pull_amount, 0.0, 1.0)
        if np.any(pull > 0):
            reference_luma_matched = _match_luma_linear(foreground_linear, _linear_luma(cleaned))
            cleaned = cleaned * (1.0 - pull[:, :, None]) + reference_luma_matched * pull[:, :, None]

    luma_preserve = _clip01(settings.preserve_foreground_luma)
    if luma_preserve > 0:
        preserve = np.clip(repair_strength * luma_preserve, 0.0, 1.0)
        if np.any(preserve > 0):
            luma_matched = _match_luma_linear(cleaned, reference_luma)
            cleaned = cleaned * (1.0 - preserve[:, :, None]) + luma_matched * preserve[:, :, None]

    cleaned = np.clip(np.nan_to_num(cleaned, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    repaired = _linear_f32_to_srgb_u8(cleaned)
    changed = repair_strength > (1.0 / 255.0)
    out = original_rgb.copy()
    out[changed] = repaired[changed]
    out[alpha <= 0] = 0

    delta = np.max(np.abs(out.astype(np.int16) - original_rgb.astype(np.int16)), axis=2).astype(np.float32) / 255.0
    repair_mask_f = np.maximum(repair_strength, delta)
    repair_mask[repair_mask_f > 0] = np.rint(np.clip(repair_mask_f[repair_mask_f > 0], 0.0, 1.0) * 255.0).astype(np.uint8)
    repair_mask[alpha <= 0] = 0
    return out, repair_mask


def _process_color_tile(
    rgb_tile: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_tile: np.ndarray | None,
    nearest_inner_rgb: np.ndarray | None,
    nearest_inner_valid: np.ndarray | None,
    screen_color: tuple[int, int, int],
    settings: KeySettings,
    transition_nearest_rgb: np.ndarray | None = None,
    transition_nearest_valid: np.ndarray | None = None,
    gpu_stats: dict | None = None,
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
    protected_core = None
    if bool(settings.transition_unmix):
        protected_core = _build_foreground_core_mask(alpha_u8, background_mask, probability, fringe_mask, None, None, settings)

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
    if protected_core is not None:
        changed &= ~protected_core
    if np.any(changed):
        repaired = _linear_f32_to_srgb_u8(out)
        rgb_out[changed] = repaired[changed]

    if bool(settings.transition_unmix):
        repair_nearest_rgb = nearest_inner_rgb if transition_nearest_rgb is None else transition_nearest_rgb
        repair_nearest_valid = nearest_inner_valid if transition_nearest_valid is None else transition_nearest_valid
        transition_rgb = transition_mask = None
        gpu_mode = _gpu_acceleration_mode(settings)
        if gpu_mode != "Off" and _transition_reference_enabled(settings):
            gpu_result: dict
            try:
                import gpu_accel

                gpu_result = gpu_accel.process_color_tile_gpu(
                    rgb_tile,
                    alpha_u8,
                    background_mask,
                    edge_mask,
                    probability,
                    fringe_mask,
                    screen_tile,
                    repair_nearest_rgb,
                    repair_nearest_valid,
                    screen_color,
                    settings,
                    force_gpu=gpu_mode == "Force GPU",
                )
            except Exception as exc:  # pragma: no cover - defensive backend boundary
                gpu_result = {
                    "ok": False,
                    "used": False,
                    "backend": "compact_cuda_dll",
                    "backend_name": "compact CUDA DLL",
                    "reason": "gpu_exception",
                    "message": f"GPU transition repair failed before launch; CPU fallback is required: {type(exc).__name__}: {exc}",
                    "elapsed_ms": None,
                }
            _record_gpu_tile_result(gpu_stats, gpu_result)
            if gpu_result.get("used") and isinstance(gpu_result.get("rgb"), np.ndarray) and isinstance(gpu_result.get("repair_mask"), np.ndarray):
                transition_rgb = gpu_result["rgb"]
                transition_mask = gpu_result["repair_mask"]
            elif gpu_mode == "Force GPU" and gpu_result.get("reason") in {"cuda_dll_unavailable", "cuda_dll_probe_failed", "cuda_no_device", "cuda_unavailable", "cuda_execution_failed", "gpu_exception"}:
                raise RuntimeError(str(gpu_result.get("message") or "Force GPU requested, but the CUDA backend is unavailable."))

        if transition_rgb is None or transition_mask is None:
            transition_rgb, transition_mask = _repair_transition_unmix(
                rgb_tile,
                alpha_u8,
                background_mask,
                edge_mask,
                probability,
                fringe_mask,
                screen_color,
                screen_tile,
                repair_nearest_rgb,
                repair_nearest_valid,
                settings,
            )
        transition_changed = live & (transition_mask > 0)
        if np.any(transition_changed):
            rgb_out[transition_changed] = transition_rgb[transition_changed]
            spill_mask = np.maximum(spill_mask, transition_mask.astype(np.float32) / 255.0)
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


def _screen_model_radius_for_shape(shape: tuple[int, int]) -> int:
    h, w = shape
    return int(np.clip(max(int(h), int(w)) // 18, 24, 181))


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
