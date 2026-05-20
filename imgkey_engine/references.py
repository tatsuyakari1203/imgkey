from __future__ import annotations

import cv2
import numpy as np
import sys

from .color_math import _clip01
from .tiling import _effective_edge_radius
from .types import (
    KeySettings,
    _MAX_INNER_LABEL_PIXELS,
    _MAX_TILE_LOCAL_INNER_LABEL_PIXELS,
    _MAX_TILE_LOCAL_NEAREST_INNER_RADIUS,
    _MIN_TILE_LOCAL_INNER_PIXELS,
)


def _compat_limit(name: str, default: int) -> int:
    facade = sys.modules.get("keyer")
    return int(getattr(facade, name, default)) if facade is not None else int(default)


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
    if alpha_u8.size > _compat_limit("_MAX_INNER_LABEL_PIXELS", _MAX_INNER_LABEL_PIXELS):
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
    if alpha_u8.size > _compat_limit("_MAX_TILE_LOCAL_INNER_LABEL_PIXELS", _MAX_TILE_LOCAL_INNER_LABEL_PIXELS):
        return None, None, None
    inner = _nearest_inner_seed_mask(alpha_u8, background_mask, probability, fringe_mask, settings)
    if np.count_nonzero(inner) < _compat_limit("_MIN_TILE_LOCAL_INNER_PIXELS", _MIN_TILE_LOCAL_INNER_PIXELS):
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


def _tile_local_nearest_inner_radius(settings: KeySettings) -> int:
    legacy_enabled = _legacy_inner_repair_enabled(settings)
    transition_radius = _foreground_reference_radius(settings) if _transition_reference_enabled(settings) else 0
    if not legacy_enabled and transition_radius <= 0:
        return 0
    edge_radius = _effective_edge_radius(settings)
    fringe_radius = max(0, int(settings.fringe_band_radius))
    legacy_radius = max(_compat_limit("_MIN_TILE_LOCAL_INNER_PIXELS", _MIN_TILE_LOCAL_INNER_PIXELS), edge_radius * 4, edge_radius + fringe_radius) if legacy_enabled else 0
    radius = max(legacy_radius, transition_radius)
    return int(min(radius, _compat_limit("_MAX_TILE_LOCAL_NEAREST_INNER_RADIUS", _MAX_TILE_LOCAL_NEAREST_INNER_RADIUS)))


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
    if read_h * read_w > _compat_limit("_MAX_TILE_LOCAL_INNER_LABEL_PIXELS", _MAX_TILE_LOCAL_INNER_LABEL_PIXELS):
        return False
    h, w = shape
    whole_read = read_y.start <= 0 and read_x.start <= 0 and read_y.stop >= h and read_x.stop >= w
    whole_core = core_y.start <= 0 and core_x.start <= 0 and core_y.stop >= h and core_x.stop >= w
    return not (whole_read and whole_core)
