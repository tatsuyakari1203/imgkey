from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from imgkey_engine.cache import (
    BaseMatteRecord,
    ProcessCache,
    ProcessCacheContext,
    ProcessCacheTransaction,
    ReferencePrepRecord,
    TilePrepEntry,
    TilePrepRecord,
    TransitionAlphaRecord,
    mutable_result_copy,
    readonly_array,
)
from imgkey_engine.cache_keys import (
    reference_prep_cache_fingerprint,
    runtime_base_matte_cache_fingerprint,
    tile_prep_cache_fingerprint,
    transition_alpha_cache_fingerprint,
)
from imgkey_engine.color_math import (
    _LINEAR_LUMA_WEIGHTS,
    _clip01,
    _compute_key_spill_strength,
    _linear_f32_to_srgb_u8,
    _linear_luma,
    _linear_luma_from_rgb_u8,
    _linear_to_srgb_f32,
    _match_luma_linear,
    _screen_chroma_unit_vectors,
    _smoothstep,
    _srgb_to_linear_f32,
    _srgb_u8_to_linear_f32,
)
from imgkey_engine.color_repair import (
    _apply_vlahos_clamp,
    _compute_despill_mask,
    _despill_tile,
    _effective_luminance_protect,
    _finalize_gpu_stats,
    _gpu_acceleration_mode,
    _new_gpu_stats,
    _process_color_tile,
    _protect_luminance,
    _record_gpu_tile_result,
    _repair_transition_unmix,
    _screen_linear_for_tile,
)
from imgkey_engine.image_io import (
    PNG_DEFAULT_COMPRESSION_LEVEL,
    PNG_FAST_COMPRESSION_LEVEL,
    checkerboard_composite,
    read_alpha_hint_mask,
    read_grayscale_mask,
    read_image_rgb,
    read_imported_matte_mask,
    resize_for_preview,
    write_grayscale_mask,
    write_png_rgba,
)
from imgkey_engine.matte import (
    _alpha_hint_foreground_mask,
    _apply_alpha_hint,
    _apply_original_alpha,
    _apply_screen_residue_alpha_cleanup,
    _border_connected,
    _build_alpha_from_trimap,
    _build_fringe_mask,
    _ensure_rgb_u8,
    _fill_small_holes,
    _guided_filter_gray,
    _mask_to_bool,
    _mask_to_u8,
    _refine_alpha_guided,
    _remove_small_components,
)
from imgkey_engine.profiling import record_count, record_metadata, time_block
from imgkey_engine.references import (
    _bool_mask_or_empty,
    _bounded_tile_local_nearest_inner_radius,
    _build_foreground_core_mask,
    _build_nearest_inner_label_map,
    _build_nearest_inner_reference_map,
    _build_tile_local_nearest_inner_reference,
    _build_tile_local_nearest_inner_rgb,
    _build_transition_repair_mask,
    _can_build_tile_local_nearest_inner,
    _foreground_reference_for_slice,
    _foreground_reference_radius,
    _legacy_inner_repair_enabled,
    _nearest_inner_label_to_flat,
    _nearest_inner_rgb_for_slice,
    _nearest_inner_seed_mask,
    _tile_local_nearest_inner_radius,
    _transition_reference_enabled,
    _u8_mask_or_empty,
)
from imgkey_engine.screen_model import (
    _auto_detect_border_screen_color,
    _border_pixels,
    _compute_screen_probability,
    _compute_screen_probability_block,
    _estimate_screen_map,
    _estimate_screen_tile,
    _initial_border_candidates,
    _sample_screen_color,
)
from imgkey_engine.tiling import (
    _effective_edge_radius,
    _ellipse_kernel,
    _expanded_mask_bounds,
    _iter_tiles,
    _normalized_crop,
    _odd_kernel_from_radius,
    _raise_if_cancelled,
    _report,
    _screen_model_radius_for_shape,
    _tile_intersects_crop,
)
from imgkey_engine.transition_alpha import (
    _alpha_recovery_stripe_rows,
    _cap_alpha_to_original,
    _finalize_recovered_transition_alpha,
    _recover_transition_alpha_block,
    _recover_transition_alpha_global,
    _recover_transition_alpha_tile_local,
)
from imgkey_engine.types import (
    CancelCallback,
    KeyResult,
    KeySettings,
    ProgressCallback,
    _GlobalMatte,
    _MAX_ALPHA_RECOVERY_BLOCK_PIXELS,
    _MAX_INNER_LABEL_PIXELS,
    _MAX_TILE_LOCAL_INNER_LABEL_PIXELS,
    _MAX_TILE_LOCAL_NEAREST_INNER_RADIUS,
    _MIN_TILE_LOCAL_INNER_PIXELS,
)


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
    process_cache: ProcessCache | None = None,
    cache_context: ProcessCacheContext | None = None,
    cache_transaction: ProcessCacheTransaction | None = None,
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

    transaction, auto_commit = _prepare_cache_transaction(process_cache, cache_context, cache_transaction)
    try:
        _raise_if_cancelled(cancel_callback)
        global_matte = _build_global_matte_with_cache(
            rgb,
            settings,
            original_alpha,
            keep,
            remove,
            hint,
            progress_callback,
            cancel_callback,
            transaction,
        )
        _raise_if_cancelled(cancel_callback)

        crop = _normalized_crop(settings.full_res_crop, w, h)
        gpu_stats = _new_gpu_stats(settings)
        with time_block("render.tiled_rgba_total"):
            rgba, despill_mask = _render_tiled_rgba(
                rgb,
                settings,
                global_matte,
                progress_callback,
                cancel_callback,
                render_crop=crop,
                include_debug=include_debug,
                gpu_stats=gpu_stats,
                cache_transaction=transaction,
            )
        cache_info = _cache_info_dict(transaction)
        if cache_info is not None:
            record_metadata("cache", cache_info)
        with time_block("result.debug_output_conversion"):
            if include_debug:
                foreground = rgba[:, :, :3].copy()
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    alpha = mutable_result_copy(global_matte.alpha[y0:y1, x0:x1])
                    background_mask = (global_matte.background_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
                    edge_mask = (global_matte.edge_mask[y0:y1, x0:x1].astype(np.uint8) * 255)
                    fringe_mask = mutable_result_copy(global_matte.fringe_mask[y0:y1, x0:x1])
                    probability = mutable_result_copy(global_matte.screen_probability[y0:y1, x0:x1])
                    hint_out = None if global_matte.alpha_hint is None else mutable_result_copy(global_matte.alpha_hint[y0:y1, x0:x1])
                else:
                    alpha = mutable_result_copy(global_matte.alpha)
                    background_mask = (global_matte.background_mask.astype(np.uint8) * 255)
                    edge_mask = (global_matte.edge_mask.astype(np.uint8) * 255)
                    fringe_mask = mutable_result_copy(global_matte.fringe_mask)
                    probability = mutable_result_copy(global_matte.screen_probability)
                    hint_out = None if global_matte.alpha_hint is None else mutable_result_copy(global_matte.alpha_hint)
            else:
                foreground = None
                alpha = rgba[:, :, 3]
                background_mask = None
                edge_mask = None
                fringe_mask = None
                probability = None
                hint_out = None
        result = KeyResult(
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
            cache_info=cache_info,
        )
        if auto_commit and transaction is not None:
            transaction.commit()
            result.cache_info = transaction.info.as_dict()
            record_metadata("cache", result.cache_info)
        return result
    except Exception:
        if auto_commit and transaction is not None:
            transaction.discard()
        raise


def _prepare_cache_transaction(
    process_cache: ProcessCache | None,
    cache_context: ProcessCacheContext | None,
    cache_transaction: ProcessCacheTransaction | None,
) -> tuple[ProcessCacheTransaction | None, bool]:
    if cache_transaction is not None:
        return cache_transaction, False
    if process_cache is None or cache_context is None:
        return None, False
    return process_cache.begin(cache_context), True


def _cache_info_dict(transaction: ProcessCacheTransaction | None) -> dict | None:
    if transaction is None:
        return None
    return transaction.info.as_dict()


def _build_global_matte_with_cache(
    rgb: np.ndarray,
    settings: KeySettings,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    alpha_hint: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    transaction: ProcessCacheTransaction | None,
) -> _GlobalMatte:
    h, w = rgb.shape[:2]
    if transaction is None:
        with time_block("global_matte.total"):
            matte = _build_global_matte(rgb, settings, original_alpha, keep_mask, remove_mask, alpha_hint, progress_callback, cancel_callback)
        _report(progress_callback, 0.18, "global matte")
        return matte

    context_shape = transaction.context.shape
    if context_shape is not None and context_shape != (h, w):
        transaction.info.base_matte = "disabled"
        transaction.info.reference_prep = "disabled"
        transaction.info.transition_alpha = "disabled"
        transaction.info.cache_hit = False
        transaction.info.cache_miss_reason = "context_shape_mismatch"
        with time_block("global_matte.total"):
            matte = _build_global_matte(rgb, settings, original_alpha, keep_mask, remove_mask, alpha_hint, progress_callback, cancel_callback)
        _report(progress_callback, 0.18, "global matte (cache context mismatch)")
        return matte

    transaction.remember_source(original_alpha is not None)
    source_key = transaction.context.source_key
    mask_key = transaction.context.mask_key
    base_key = runtime_base_matte_cache_fingerprint(settings, source_key, mask_key)
    reference_key = reference_prep_cache_fingerprint(settings, base_key)
    transition_key = transition_alpha_cache_fingerprint(settings, base_key, reference_key)
    transaction.info.details.update(
        {
            "base_key": base_key[:16],
            "reference_key": reference_key[:16],
            "transition_key": transition_key[:16],
            "source": transaction.context.source_fingerprint[:16],
            "mask": transaction.context.mask_fingerprint[:16],
        }
    )

    base = transaction.get_base(base_key)
    screen_map_for_reference: np.ndarray | None = None
    if base is None:
        transaction.info.base_matte = "miss"
        transaction.info.reference_prep = "miss"
        transaction.info.transition_alpha = "miss"
        transaction.info.cache_hit = False
        transaction.info.cache_miss_reason = "base_matte_not_found"
        record_count("cache.base_matte.miss")
        with time_block("global_matte.total"):
            base, screen_map_for_reference = _build_base_matte_record(
                rgb,
                settings,
                original_alpha,
                keep_mask,
                remove_mask,
                alpha_hint,
                progress_callback,
                cancel_callback,
                cache_key=base_key,
                source_fingerprint=transaction.context.source_fingerprint,
                mask_fingerprint=transaction.context.mask_fingerprint,
                resolution=transaction.context.resolution,
            )
            transaction.stage_base(base)
            reference = _build_reference_prep_record(
                rgb,
                settings,
                base,
                progress_callback,
                cancel_callback,
                cache_key=reference_key,
                screen_map=screen_map_for_reference,
            )
            transaction.stage_reference(reference)
            transition = _build_transition_alpha_record(
                rgb,
                settings,
                original_alpha,
                keep_mask,
                remove_mask,
                base,
                reference,
                progress_callback,
                cancel_callback,
                cache_key=transition_key,
            )
            transaction.stage_transition(transition)
        _report(progress_callback, 0.18, "global matte (cache miss: base matte)")
        return _compose_global_matte(base, reference, transition)

    transaction.info.base_matte = "hit"
    record_count("cache.base_matte.hit")

    reference = transaction.get_reference(reference_key)
    if reference is None:
        transaction.info.reference_prep = "miss"
        transaction.info.cache_hit = "base_matte"
        transaction.info.cache_miss_reason = "reference_prep_not_found"
        record_count("cache.reference_prep.miss")
        reference = _build_reference_prep_record(
            rgb,
            settings,
            base,
            progress_callback,
            cancel_callback,
            cache_key=reference_key,
            screen_map=None,
        )
        transaction.stage_reference(reference)
    else:
        transaction.info.reference_prep = "hit"
        record_count("cache.reference_prep.hit")

    transition = transaction.get_transition(transition_key)
    if transition is None:
        transaction.info.transition_alpha = "miss"
        transaction.info.cache_hit = "base_matte" if transaction.info.reference_prep == "hit" else "base_matte_partial"
        transaction.info.cache_miss_reason = "transition_alpha_not_found"
        record_count("cache.transition_alpha.miss")
        transition = _build_transition_alpha_record(
            rgb,
            settings,
            original_alpha,
            keep_mask,
            remove_mask,
            base,
            reference,
            progress_callback,
            cancel_callback,
            cache_key=transition_key,
        )
        transaction.stage_transition(transition)
        _report(progress_callback, 0.18, "transition alpha (cached base matte)")
    else:
        transaction.info.transition_alpha = "hit"
        transaction.info.cache_hit = "matte"
        transaction.info.cache_miss_reason = None
        record_count("cache.transition_alpha.hit")
        _report(progress_callback, 0.18, "cached matte")

    return _compose_global_matte(base, reference, transition)


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
    base, screen_map = _build_base_matte_record(
        rgb,
        settings,
        original_alpha,
        keep_mask,
        remove_mask,
        alpha_hint,
        progress_callback,
        cancel_callback,
        cache_key="uncached-base",
        source_fingerprint="uncached-source",
        mask_fingerprint="uncached-mask",
        resolution="direct",
    )
    reference = _build_reference_prep_record(
        rgb,
        settings,
        base,
        progress_callback,
        cancel_callback,
        cache_key="uncached-reference",
        screen_map=screen_map,
    )
    transition = _build_transition_alpha_record(
        rgb,
        settings,
        original_alpha,
        keep_mask,
        remove_mask,
        base,
        reference,
        progress_callback,
        cancel_callback,
        cache_key="uncached-transition",
    )
    return _compose_global_matte(base, reference, transition)


def _build_base_matte_record(
    rgb: np.ndarray,
    settings: KeySettings,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    alpha_hint: np.ndarray | None,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    *,
    cache_key: str,
    source_fingerprint: str,
    mask_fingerprint: str,
    resolution: str,
) -> tuple[BaseMatteRecord, np.ndarray | None]:
    h, w = rgb.shape[:2]
    with time_block("global_matte.sample_screen"):
        screen_color = _sample_screen_color(rgb, settings)
    _report(progress_callback, 0.02, "sample screen")
    _raise_if_cancelled(cancel_callback)
    with time_block("global_matte.screen_probability"):
        probability = _compute_screen_probability(rgb, screen_color, settings, progress_callback, cancel_callback)
    _report(progress_callback, 0.10, "screen probability")
    _raise_if_cancelled(cancel_callback)

    with time_block("global_matte.connected_background"):
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
    with time_block("global_matte.aggressive_background"):
        if settings.aggressive_interior_removal:
            aggressive = probability >= int(round(_clip01(settings.aggressive_threshold) * 255.0))
            if settings.aggressive_min_area > 1:
                aggressive = _remove_small_components(aggressive, int(settings.aggressive_min_area), protect_border=False)
            background |= aggressive
    _raise_if_cancelled(cancel_callback)

    with time_block("global_matte.mask_merge"):
        hint_foreground = _alpha_hint_foreground_mask(alpha_hint, settings)
        if hint_foreground is not None:
            background &= ~hint_foreground
        if keep_mask is not None:
            background &= ~keep_mask
        if remove_mask is not None:
            remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
            background |= remove_effective
    _raise_if_cancelled(cancel_callback)

    with time_block("global_matte.cleanup"):
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
    _raise_if_cancelled(cancel_callback)

    with time_block("global_matte.trimap_alpha"):
        edge_mask, alpha = _build_alpha_from_trimap(background, probability, fg_threshold, bg_threshold, settings)
    _report(progress_callback, 0.15, "trimap")
    _raise_if_cancelled(cancel_callback)
    with time_block("global_matte.mask_alpha_apply"):
        if keep_mask is not None:
            alpha[keep_mask] = 255
            edge_mask[keep_mask] = False
        if alpha_hint is not None:
            _apply_alpha_hint(alpha, edge_mask, background, alpha_hint, settings)
        if remove_mask is not None:
            remove_effective = remove_mask if keep_mask is None else (remove_mask & ~keep_mask)
            alpha[remove_effective] = 0
            background[remove_effective] = True

    with time_block("global_matte.guided_alpha_refine"):
        alpha = _refine_alpha_guided(rgb, alpha, edge_mask, background, probability, fg_threshold, bg_threshold, settings)
    _raise_if_cancelled(cancel_callback)

    with time_block("screen_reference.screen_map_global"):
        screen_map = _estimate_screen_map(rgb, probability >= bg_threshold, screen_color, settings)
    _report(progress_callback, 0.17, "screen model")
    _raise_if_cancelled(cancel_callback)
    with time_block("global_matte.original_alpha_apply"):
        alpha = _apply_original_alpha(alpha, original_alpha)
    _raise_if_cancelled(cancel_callback)
    with time_block("global_matte.screen_cleanup"):
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

    record = BaseMatteRecord(
        key=str(cache_key),
        source_fingerprint=str(source_fingerprint),
        mask_fingerprint=str(mask_fingerprint),
        resolution=str(resolution),
        shape=(int(h), int(w)),
        screen_color=tuple(int(c) for c in screen_color),
        screen_probability=readonly_array(probability),
        background_mask=readonly_array(background.astype(bool, copy=False)),
        edge_mask=readonly_array(edge_mask.astype(bool, copy=False)),
        alpha=readonly_array(alpha),
        alpha_hint=readonly_array(alpha_hint, copy=True),
    )
    return record, screen_map


def _build_reference_prep_record(
    rgb: np.ndarray,
    settings: KeySettings,
    base: BaseMatteRecord,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    *,
    cache_key: str,
    screen_map: np.ndarray | None,
) -> ReferencePrepRecord:
    bg_threshold = int(round(_clip01(settings.clip_background) * 255.0))
    if screen_map is None:
        with time_block("screen_reference.screen_map_global"):
            screen_map = _estimate_screen_map(rgb, base.screen_probability >= bg_threshold, base.screen_color, settings)
        _report(progress_callback, 0.17, "screen model")
        _raise_if_cancelled(cancel_callback)
    with time_block("global_matte.fringe_map"):
        fringe_mask = _build_fringe_mask(rgb, base.alpha, base.edge_mask, base.screen_probability, base.screen_color, settings, progress_callback, cancel_callback)
    _report(progress_callback, 0.175, "fringe map")
    _raise_if_cancelled(cancel_callback)
    with time_block("screen_reference.nearest_inner_global"):
        reference_settings = replace(
            settings,
            transition_unmix=True,
            inner_color_pull=max(float(settings.inner_color_pull), 1.0),
            edge_color_repair=max(float(settings.edge_color_repair), 1.0),
        )
        inner_labels, inner_label_to_flat, inner_distance = _build_nearest_inner_reference_map(
            base.alpha,
            base.background_mask,
            base.screen_probability,
            fringe_mask,
            reference_settings,
        )
    _report(progress_callback, 0.18, "inner color map")
    _raise_if_cancelled(cancel_callback)
    return ReferencePrepRecord(
        key=str(cache_key),
        base_key=str(base.key),
        screen_map=readonly_array(screen_map),
        fringe_mask=readonly_array(fringe_mask),
        inner_labels=readonly_array(inner_labels),
        inner_label_to_flat=readonly_array(inner_label_to_flat),
        inner_distance=readonly_array(inner_distance),
    )


def _build_transition_alpha_record(
    rgb: np.ndarray,
    settings: KeySettings,
    original_alpha: np.ndarray | None,
    keep_mask: np.ndarray | None,
    remove_mask: np.ndarray | None,
    base: BaseMatteRecord,
    reference: ReferencePrepRecord,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    *,
    cache_key: str,
) -> TransitionAlphaRecord:
    color_alpha = base.alpha.copy() if bool(settings.transition_unmix) and _clip01(settings.alpha_recover_strength) > 0 else None
    with time_block("transition_alpha.global_recovery"):
        alpha = _recover_transition_alpha_global(
            rgb,
            base.alpha,
            base.background_mask,
            base.edge_mask,
            base.screen_probability,
            reference.fringe_mask,
            base.screen_color,
            reference.screen_map,
            reference.inner_labels,
            reference.inner_label_to_flat,
            reference.inner_distance,
            settings,
            original_alpha,
            keep_mask,
            remove_mask,
            progress_callback,
            cancel_callback,
        )
    _report(progress_callback, 0.18, "transition alpha")
    return TransitionAlphaRecord(
        key=str(cache_key),
        base_key=str(base.key),
        reference_key=str(reference.key),
        alpha=readonly_array(alpha),
        color_alpha=readonly_array(color_alpha),
    )


def _compose_global_matte(
    base: BaseMatteRecord,
    reference: ReferencePrepRecord,
    transition: TransitionAlphaRecord,
) -> _GlobalMatte:
    return _GlobalMatte(
        screen_color=base.screen_color,
        screen_probability=base.screen_probability,
        screen_map=reference.screen_map,
        background_mask=base.background_mask,
        edge_mask=base.edge_mask,
        alpha=transition.alpha,
        color_alpha=transition.color_alpha,
        alpha_hint=base.alpha_hint,
        fringe_mask=reference.fringe_mask,
        inner_labels=reference.inner_labels,
        inner_label_to_flat=reference.inner_label_to_flat,
        inner_distance=reference.inner_distance,
        base_key=base.key,
        reference_key=reference.key,
        transition_key=transition.key,
    )


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
    cache_transaction: ProcessCacheTransaction | None = None,
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

    with time_block("render.result_allocation"):
        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        rgba[:, :, 3] = alpha_out
        despill_mask = np.zeros((out_h, out_w), dtype=np.uint8) if include_debug else None
    reference_alpha = matte.alpha if matte.color_alpha is None else matte.color_alpha
    with time_block("render.tile_enumeration"):
        screen_radius = _screen_model_radius_for_shape((h, w)) if settings.local_screen_model and matte.screen_map is None else 0
        local_nearest_radius = _tile_local_nearest_inner_radius(settings) if matte.inner_labels is None else 0
        extra_overlap = _tile_extra_overlap(settings, (h, w), screen_radius, local_nearest_radius)
        tiles = list(_iter_tiles(h, w, settings, _effective_edge_radius(settings), extra_overlap=extra_overlap))
        if crop is not None:
            tiles = [tile for tile in tiles if _tile_intersects_crop(tile[2], tile[3], crop)]
    record_count("render.tiles", len(tiles))
    total = max(1, len(tiles))
    transition_radius = _foreground_reference_radius(settings) if _transition_reference_enabled(settings) else 0
    tile_prep_key: str | None = None
    cached_tile_prep_entries: dict[str, TilePrepEntry] = {}
    staged_tile_prep_entries: dict[str, TilePrepEntry] = {}
    tile_prep_hits = 0
    tile_prep_misses = 0
    tile_prep_enabled = (
        cache_transaction is not None
        and matte.transition_key is not None
        and cache_transaction.info.transition_alpha != "disabled"
        and not str(matte.transition_key).startswith("uncached-")
        and (screen_radius > 0 or (matte.inner_labels is None and local_nearest_radius > 0))
    )
    if cache_transaction is not None:
        cache_transaction.info.tile_prep = "disabled"
    if tile_prep_enabled and cache_transaction is not None and matte.transition_key is not None:
        tile_prep_key = tile_prep_cache_fingerprint(
            settings,
            matte.transition_key,
            source_shape=(h, w),
            render_crop=crop,
            screen_radius=screen_radius,
            local_nearest_radius=local_nearest_radius,
            transition_nearest_radius=transition_radius,
            extra_overlap=extra_overlap,
            legacy_reference_enabled=_legacy_inner_repair_enabled(settings),
            transition_reference_enabled=_transition_reference_enabled(settings),
        )
        tile_prep_record = cache_transaction.get_tile_prep(tile_prep_key)
        if tile_prep_record is not None:
            cached_tile_prep_entries = dict(tile_prep_record.entries)
            cache_transaction.info.tile_prep = "hit"
            record_count("cache.tile_prep.hit")
        else:
            cache_transaction.info.tile_prep = "miss"
            record_count("cache.tile_prep.miss")
        cache_transaction.info.details["tile_prep_key"] = tile_prep_key[:16]
    gpu_session = None
    if _gpu_acceleration_mode(settings) != "Off":
        _raise_if_cancelled(cancel_callback)
        try:
            import gpu_backend

            required = {"rgb_only"}
            if matte.screen_map is not None or settings.local_screen_model:
                required.add("screen_tile")
            with time_block("gpu.begin_render"):
                gpu_session = gpu_backend.begin_render(settings, (h, w), required_capabilities=required)
        except Exception:
            gpu_session = None
    try:
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
            tile_entry_key = _tile_prep_entry_key(read_y, read_x, core_y, core_x)
            cached_tile_entry = cached_tile_prep_entries.get(tile_entry_key) if tile_prep_enabled else None
            cacheable_screen_tile: np.ndarray | None = None
            cacheable_nearest_rgb: np.ndarray | None = None
            cacheable_nearest_valid: np.ndarray | None = None
            cacheable_transition_rgb: np.ndarray | None = None
            cacheable_transition_valid: np.ndarray | None = None
            used_tile_entry = False
            with time_block("screen_reference.screen_tile_resolve"):
                if matte.screen_map is not None:
                    screen_tile = matte.screen_map[read_y, read_x]
                elif settings.local_screen_model:
                    if cached_tile_entry is not None and cached_tile_entry.screen_tile is not None:
                        screen_tile = cached_tile_entry.screen_tile
                        used_tile_entry = True
                    else:
                        screen_tile = _estimate_screen_tile(
                            rgb_read,
                            matte.background_mask[read_y, read_x],
                            matte.screen_color,
                            screen_radius,
                        )
                        cacheable_screen_tile = screen_tile
                else:
                    screen_tile = None
            with time_block("screen_reference.nearest_inner_tile"):
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
                elif cached_tile_entry is not None:
                    nearest_inner_rgb = cached_tile_entry.nearest_inner_rgb
                    nearest_inner_valid = cached_tile_entry.nearest_inner_valid
                    transition_inner_rgb = cached_tile_entry.transition_inner_rgb
                    transition_inner_valid = cached_tile_entry.transition_inner_valid
                    used_tile_entry = True
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
                        cacheable_nearest_rgb = nearest_inner_rgb
                        cacheable_nearest_valid = nearest_inner_valid
                        cacheable_transition_rgb = transition_inner_rgb
                        cacheable_transition_valid = transition_inner_valid
                    else:
                        nearest_inner_rgb, nearest_inner_valid = None, None
                        transition_inner_rgb, transition_inner_valid = None, None
                        cacheable_nearest_rgb = None
                        cacheable_nearest_valid = None
                        cacheable_transition_rgb = None
                        cacheable_transition_valid = None
            if tile_prep_enabled:
                if used_tile_entry:
                    tile_prep_hits += 1
                else:
                    tile_prep_misses += 1
                    staged_tile_prep_entries[tile_entry_key] = _make_tile_prep_entry(
                        screen_tile=cacheable_screen_tile,
                        nearest_inner_rgb=cacheable_nearest_rgb,
                        nearest_inner_valid=cacheable_nearest_valid,
                        transition_inner_rgb=cacheable_transition_rgb,
                        transition_inner_valid=cacheable_transition_valid,
                    )
            _raise_if_cancelled(cancel_callback)
            with time_block("render.per_tile_color_render"):
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
                    gpu_session=gpu_session,
                    cancel_callback=cancel_callback,
                )
            with time_block("render.tile_composite_write"):
                rgba[out_y, out_x, :3] = rgb_tile[rel_y, rel_x]
                if despill_mask is not None:
                    despill_mask[out_y, out_x] = spill_tile[rel_y, rel_x]
            _report(
                progress_callback,
                0.18 + 0.82 * (index / total),
                _color_render_progress_stage(settings, gpu_stats, index, total),
            )
        if tile_prep_enabled and cache_transaction is not None and tile_prep_key is not None:
            cache_transaction.info.details["tile_prep"] = {
                "hits": int(tile_prep_hits),
                "misses": int(tile_prep_misses),
                "entries": int(len(cached_tile_prep_entries) + len(staged_tile_prep_entries)),
            }
            record_count("cache.tile_prep.entry_hit", tile_prep_hits)
            record_count("cache.tile_prep.entry_miss", tile_prep_misses)
            if staged_tile_prep_entries:
                merged_entries = dict(cached_tile_prep_entries)
                merged_entries.update(staged_tile_prep_entries)
                cache_transaction.stage_tile_prep(
                    TilePrepRecord(
                        key=tile_prep_key,
                        transition_key=str(matte.transition_key),
                        entries=merged_entries,
                    )
                )
                if cached_tile_prep_entries:
                    cache_transaction.info.tile_prep = "partial"
    finally:
        if gpu_session is not None:
            try:
                with time_block("gpu.end_render"):
                    gpu_session.end_render()
            except Exception:
                pass
    with time_block("render.transparent_rgb_zero"):
        rgba[alpha_out <= 0, :3] = 0
    return rgba, despill_mask


def _color_render_progress_stage(settings: KeySettings, gpu_stats: dict | None, index: int, total: int) -> str:
    prefix = f"tile {index}/{total}"
    mode = _gpu_acceleration_mode(settings)
    if mode == "Off":
        return f"{prefix} · CPU color render"
    stats = gpu_stats or {}
    used = int(stats.get("used_tiles", 0) or 0)
    fallback = int(stats.get("fallback_tiles", 0) or 0)
    backend = str(stats.get("backend") or stats.get("backend_name") or "").strip()
    if used > 0:
        label = "D3D12 color render" if "d3d12" in backend.lower() else "GPU color render"
        return f"{prefix} · {label}"
    if fallback > 0 or str(stats.get("status") or "").lower() == "fallback":
        return f"{prefix} · CPU color render (GPU fallback)"
    return f"{prefix} · color render (GPU {mode} requested)"


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


def _tile_prep_entry_key(read_y: slice, read_x: slice, core_y: slice, core_x: slice) -> str:
    return (
        f"ry{int(read_y.start)}:{int(read_y.stop)}|rx{int(read_x.start)}:{int(read_x.stop)}|"
        f"cy{int(core_y.start)}:{int(core_y.stop)}|cx{int(core_x.start)}:{int(core_x.stop)}"
    )


def _make_tile_prep_entry(
    *,
    screen_tile: np.ndarray | None,
    nearest_inner_rgb: np.ndarray | None,
    nearest_inner_valid: np.ndarray | None,
    transition_inner_rgb: np.ndarray | None,
    transition_inner_valid: np.ndarray | None,
) -> TilePrepEntry:
    return TilePrepEntry(
        screen_tile=readonly_array(screen_tile),
        nearest_inner_rgb=readonly_array(nearest_inner_rgb),
        nearest_inner_valid=readonly_array(nearest_inner_valid),
        transition_inner_rgb=readonly_array(transition_inner_rgb),
        transition_inner_valid=readonly_array(transition_inner_valid),
    )
