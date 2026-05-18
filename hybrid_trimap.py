from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


DEFAULT_SPILL_THRESHOLD = 96
DEFAULT_KNOWN_BG_SCREEN_THRESHOLD = 245
DEFAULT_KNOWN_BG_ALPHA_THRESHOLD = 8
DEFAULT_KNOWN_BG_BIREF_THRESHOLD = 8
DEFAULT_KNOWN_FG_ALPHA_THRESHOLD = 245
DEFAULT_KNOWN_FG_BIREF_THRESHOLD = 200
DEFAULT_CONFLICT_SCREEN_THRESHOLD = 245
DEFAULT_CONFLICT_BIREF_THRESHOLD = 64


@dataclass(slots=True)
class HybridTrimapResult:
    known_bg: np.ndarray
    known_fg: np.ndarray
    unknown: np.ndarray
    conflict: np.ndarray
    soft_unknown: np.ndarray
    hard_unknown: np.ndarray
    spill_region: np.ndarray
    unmix_region: np.ndarray
    despill_region: np.ndarray
    protected_fg: np.ndarray
    safe_bg: np.ndarray
    manual_keep_core: np.ndarray
    manual_remove_effective: np.ndarray
    strong_edge_band: np.ndarray
    candidate_alpha: np.ndarray
    spill_threshold: int = DEFAULT_SPILL_THRESHOLD
    debug_masks: dict[str, np.ndarray] = field(default_factory=dict)


def build_hybrid_trimap(
    classical_alpha: np.ndarray,
    screen_probability: np.ndarray,
    screen_distance: np.ndarray | None = None,
    spill_probability: np.ndarray | None = None,
    classical_confidence: np.ndarray | None = None,
    background_mask: np.ndarray | None = None,
    edge_mask: np.ndarray | None = None,
    fringe_mask: np.ndarray | None = None,
    screen_plate_rgb: Any | None = None,
    biref_alpha: np.ndarray | None = None,
    keep_mask: np.ndarray | None = None,
    remove_mask: np.ndarray | None = None,
    *,
    spill_threshold: int = DEFAULT_SPILL_THRESHOLD,
    manual_keep_core: np.ndarray | None = None,
    strong_edge_band: np.ndarray | None = None,
    candidate_alpha: np.ndarray | None = None,
    known_bg_screen_threshold: int = DEFAULT_KNOWN_BG_SCREEN_THRESHOLD,
    known_bg_alpha_threshold: int = DEFAULT_KNOWN_BG_ALPHA_THRESHOLD,
    known_bg_biref_threshold: int = DEFAULT_KNOWN_BG_BIREF_THRESHOLD,
    known_fg_alpha_threshold: int = DEFAULT_KNOWN_FG_ALPHA_THRESHOLD,
    known_fg_biref_threshold: int = DEFAULT_KNOWN_FG_BIREF_THRESHOLD,
    conflict_screen_threshold: int = DEFAULT_CONFLICT_SCREEN_THRESHOLD,
    conflict_biref_threshold: int = DEFAULT_CONFLICT_BIREF_THRESHOLD,
) -> HybridTrimapResult:
    """Merge classical maps and a BiRefNet alpha hint into trimap regions.

    This helper does not compute final hybrid alpha. Its region outputs are
    candidate masks for the later alpha merge and RGB cleanup phases.
    """

    alpha = _ensure_u8(classical_alpha, None, "classical_alpha")
    shape = alpha.shape
    screen_prob = _ensure_u8(screen_probability, shape, "screen_probability")
    screen_dist = _ensure_u8(screen_distance, shape, "screen_distance") if screen_distance is not None else np.zeros(shape, dtype=np.uint8)
    spill_prob = _ensure_u8(spill_probability, shape, "spill_probability") if spill_probability is not None else np.zeros(shape, dtype=np.uint8)
    class_conf = _ensure_u8(classical_confidence, shape, "classical_confidence") if classical_confidence is not None else np.zeros(shape, dtype=np.uint8)
    bg = _mask_to_bool(background_mask, shape, "background_mask")
    edge = _mask_to_bool(edge_mask, shape, "edge_mask")
    fringe = _mask_to_bool(fringe_mask, shape, "fringe_mask")
    biref = _ensure_u8(biref_alpha, shape, "biref_alpha") if biref_alpha is not None else alpha.copy()
    keep = _mask_to_bool(keep_mask, shape, "keep_mask")
    remove = _mask_to_bool(remove_mask, shape, "remove_mask")
    if keep is None:
        keep = np.zeros(shape, dtype=bool)
    if remove is None:
        remove = np.zeros(shape, dtype=bool)

    if manual_keep_core is None:
        manual_keep = keep.copy()
    else:
        manual_keep = keep | _mask_to_bool(manual_keep_core, shape, "manual_keep_core")
    remove_effective = remove & ~manual_keep
    if edge is None:
        edge = np.zeros(shape, dtype=bool)
    if fringe is None:
        fringe = np.zeros(shape, dtype=bool)
    if bg is None:
        bg = np.zeros(shape, dtype=bool)

    edge_dilated = _dilate(edge, 1)
    conflict = (screen_prob >= int(conflict_screen_threshold)) & (biref >= int(conflict_biref_threshold))
    conflict_dilated = _dilate(conflict, 1)

    if strong_edge_band is None:
        strong_edge = _dilate(edge, 2) | ((screen_dist > 64) & edge_dilated)
    else:
        strong_edge = _mask_to_bool(strong_edge_band, shape, "strong_edge_band")

    known_bg = (
        (screen_prob >= int(known_bg_screen_threshold))
        & (alpha <= int(known_bg_alpha_threshold))
        & (biref <= int(known_bg_biref_threshold))
    ) | (bg & (alpha <= int(known_bg_alpha_threshold)) & (biref <= int(known_bg_biref_threshold)))

    known_fg = (biref >= int(known_fg_biref_threshold)) | (alpha >= int(known_fg_alpha_threshold))
    known_fg &= ~(conflict | strong_edge)

    unknown = ~(known_bg | known_fg)
    unknown |= edge_dilated
    unknown |= conflict_dilated
    unknown |= strong_edge
    unknown &= ~(known_bg | known_fg)

    soft_unknown = unknown | _dilate(fringe, 2)
    hard_unknown = _dilate(conflict, 4) | strong_edge

    # Conflict/hard-unknown regions can demote automatic foreground, but manual
    # keep is applied later as the only foreground override that wins over them.
    auto_demote = conflict | hard_unknown
    known_fg &= ~auto_demote
    unknown |= auto_demote
    unknown &= ~known_bg

    # Manual masks are last among trimap decisions: keep wins over remove;
    # remove only applies where keep is not set.
    if np.any(remove_effective):
        known_bg[remove_effective] = True
        known_fg[remove_effective] = False
    if np.any(manual_keep):
        known_fg[manual_keep] = True
        known_bg[manual_keep] = False

    reserved = known_bg | known_fg
    unknown &= ~reserved
    soft_unknown |= unknown
    soft_unknown &= ~reserved
    hard_unknown &= ~reserved
    conflict &= ~reserved

    cand_alpha = _ensure_u8(candidate_alpha, shape, "candidate_alpha") if candidate_alpha is not None else alpha.copy()
    spill_threshold_i = int(np.clip(int(spill_threshold), 0, 255))
    spill_region = (cand_alpha > 0) & (cand_alpha < 250) & (spill_prob > spill_threshold_i)
    unmix_region = hard_unknown | soft_unknown | ((cand_alpha > 8) & (cand_alpha < 245))
    despill_region = spill_region & ~known_bg & ~manual_keep
    protected_fg = known_fg & (screen_prob < 128)
    safe_bg = known_bg & (screen_prob >= int(known_bg_screen_threshold))

    # Final exclusivity clamp for the three durable trimap classes.
    known_fg &= ~known_bg
    unknown &= ~(known_bg | known_fg)
    soft_unknown &= ~(known_bg | known_fg)
    hard_unknown &= ~(known_bg | known_fg)
    conflict &= ~(known_bg | known_fg)
    despill_region &= ~known_bg & ~manual_keep
    unmix_region &= ~known_bg

    debug = {
        "screen_distance": screen_dist.copy(),
        "classical_confidence": class_conf.copy(),
        "edge_dilated": edge_dilated.astype(np.uint8) * 255,
        "conflict_dilated": conflict_dilated.astype(np.uint8) * 255,
    }
    # Keep a lightweight reference hook for Phase 8 without requiring a full RGB
    # plate here; debug masks remain bool/uint8 only.
    if screen_plate_rgb is not None:
        debug["has_screen_plate"] = np.full(shape, 255, dtype=np.uint8)

    return HybridTrimapResult(
        known_bg=known_bg.astype(bool, copy=False),
        known_fg=known_fg.astype(bool, copy=False),
        unknown=unknown.astype(bool, copy=False),
        conflict=conflict.astype(bool, copy=False),
        soft_unknown=soft_unknown.astype(bool, copy=False),
        hard_unknown=hard_unknown.astype(bool, copy=False),
        spill_region=spill_region.astype(bool, copy=False),
        unmix_region=unmix_region.astype(bool, copy=False),
        despill_region=despill_region.astype(bool, copy=False),
        protected_fg=protected_fg.astype(bool, copy=False),
        safe_bg=safe_bg.astype(bool, copy=False),
        manual_keep_core=manual_keep.astype(bool, copy=False),
        manual_remove_effective=remove_effective.astype(bool, copy=False),
        strong_edge_band=strong_edge.astype(bool, copy=False),
        candidate_alpha=cand_alpha,
        spill_threshold=spill_threshold_i,
        debug_masks=debug,
    )


def _ensure_u8(mask: np.ndarray | None, shape: tuple[int, int] | None, name: str) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError(f"{name} is required")
        return np.zeros(shape, dtype=np.uint8)
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
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask")
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must match image shape")
    return np.ascontiguousarray(arr)


def _mask_to_bool(mask: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if mask is None:
        return None
    return _ensure_u8(mask, shape, name) > 127


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool, copy=True)
    size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0
