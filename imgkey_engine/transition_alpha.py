from __future__ import annotations

import cv2
import numpy as np
import sys

from .color_math import _clip01, _compute_key_spill_strength, _srgb_u8_to_linear_f32
from .references import (
    _bounded_tile_local_nearest_inner_radius,
    _build_foreground_core_mask,
    _build_tile_local_nearest_inner_reference,
    _build_transition_repair_mask,
    _foreground_reference_for_slice,
    _foreground_reference_radius,
    _transition_reference_enabled,
)
from .screen_model import _estimate_screen_tile
from .tiling import _raise_if_cancelled, _report, _screen_model_radius_for_shape
from .profiling import time_block
from .types import (
    CancelCallback,
    KeySettings,
    ProgressCallback,
    _MAX_ALPHA_RECOVERY_BLOCK_PIXELS,
)


def _facade_callable(name: str, default):
    facade = sys.modules.get("keyer")
    return getattr(facade, name, default) if facade is not None else default


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
            with time_block("screen_reference.transition_nearest_inner_slice"):
                foreground_ref_rgb, foreground_ref_valid, foreground_ref_distance = _foreground_reference_for_slice(
                    rgb,
                    inner_labels,
                    inner_label_to_flat,
                    inner_distance,
                    read_y,
                    read_x,
                    radius,
                )
            with time_block("transition_alpha.block"):
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
        with time_block("screen_reference.transition_nearest_inner_tile_local"):
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
            with time_block("screen_reference.transition_screen_tile_local"):
                screen_read = _facade_callable("_estimate_screen_tile", _estimate_screen_tile)(
                    rgb[read_y, read_x],
                    background_mask[read_y, read_x],
                    screen_color,
                    _screen_model_radius_for_shape((h, w)),
                )
            screen_tile = screen_read[rel_y, rel_x]
        with time_block("transition_alpha.block"):
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
