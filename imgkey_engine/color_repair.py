from __future__ import annotations

import numpy as np

import gpu_backend

from .color_math import (
    _clip01,
    _compute_key_spill_strength,
    _linear_f32_to_srgb_u8,
    _linear_luma,
    _match_luma_linear,
    _screen_chroma_unit_vectors,
    _smoothstep,
    _srgb_u8_to_linear_f32,
)
from .references import (
    _bool_mask_or_empty,
    _build_foreground_core_mask,
    _build_transition_repair_mask,
    _foreground_reference_radius,
    _transition_reference_enabled,
    _u8_mask_or_empty,
)
from .types import KeySettings


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
        stats["message"] = f"{backend} color pipeline used on {used} tile(s); CPU remained the reference/fallback for {fallback} tile(s)."
    elif attempted > 0:
        stats["status"] = "fallback"
        stats["message"] = stats.get("last_message") or "GPU acceleration fell back to CPU for all attempted tiles."
    else:
        stats["status"] = "fallback"
        stats["message"] = "No GPU-eligible transition tile was encountered; CPU color path used."
    return stats


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
    gpu_session=None,
) -> tuple[np.ndarray, np.ndarray]:
    gpu_mode = _gpu_acceleration_mode(settings)
    if gpu_mode != "Off":
        full_result: dict | None = None
        try:
            required = {"rgb_only", "full_color_tile", "screen_tile"} if screen_tile is not None else {"rgb_only", "full_color_tile", "constant_screen"}
            full_result = gpu_backend.process_full_color_tile(
                rgb_tile,
                alpha_u8,
                background_mask,
                edge_mask,
                probability,
                fringe_mask,
                screen_tile,
                nearest_inner_rgb,
                nearest_inner_valid,
                screen_color,
                settings,
                transition_nearest_rgb=transition_nearest_rgb,
                transition_nearest_valid=transition_nearest_valid,
                session=gpu_session,
                required_capabilities=required,
            )
        except Exception as exc:  # pragma: no cover - defensive backend boundary
            full_result = {
                "ok": False,
                "used": False,
                "backend": "gpu_backend",
                "backend_name": "GPU backend registry",
                "reason": "gpu_exception",
                "message": f"GPU full color tile failed before launch; CPU fallback is required: {type(exc).__name__}: {exc}",
                "elapsed_ms": None,
            }
        if full_result.get("used") and isinstance(full_result.get("rgb"), np.ndarray) and isinstance(full_result.get("repair_mask"), np.ndarray):
            _record_gpu_tile_result(gpu_stats, full_result)
            return full_result["rgb"], full_result["repair_mask"]
        if gpu_mode == "Force GPU" and full_result.get("reason") in gpu_backend.GPU_BACKEND_ERROR_REASONS:
            _record_gpu_tile_result(gpu_stats, full_result)
            raise RuntimeError(str(full_result.get("message") or "Force GPU requested, but full color GPU processing failed."))

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
        if gpu_mode != "Off" and _transition_reference_enabled(settings):
            gpu_result: dict
            try:
                required = {"rgb_only", "screen_tile"} if screen_tile is not None else {"rgb_only", "constant_screen"}
                gpu_result = gpu_backend.process_color_tile(
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
                    session=gpu_session,
                    required_capabilities=required,
                )
            except Exception as exc:  # pragma: no cover - defensive backend boundary
                gpu_result = {
                    "ok": False,
                    "used": False,
                    "backend": "gpu_backend",
                    "backend_name": "GPU backend registry",
                    "reason": "gpu_exception",
                    "message": f"GPU transition repair failed before launch; CPU fallback is required: {type(exc).__name__}: {exc}",
                    "elapsed_ms": None,
                }
            _record_gpu_tile_result(gpu_stats, gpu_result)
            if gpu_result.get("used") and isinstance(gpu_result.get("rgb"), np.ndarray) and isinstance(gpu_result.get("repair_mask"), np.ndarray):
                transition_rgb = gpu_result["rgb"]
                transition_mask = gpu_result["repair_mask"]
            elif gpu_mode == "Force GPU" and gpu_result.get("reason") in gpu_backend.GPU_BACKEND_ERROR_REASONS:
                raise RuntimeError(str(gpu_result.get("message") or "Force GPU requested, but no compatible GPU backend is available."))

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
