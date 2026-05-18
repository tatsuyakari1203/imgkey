from __future__ import annotations

import copy
import importlib
import time
from typing import Any

import numpy as np


BACKEND_NAME = "torch_cuda"
_AVAILABILITY_CACHE: dict[str, Any] | None = None


def _import_torch() -> Any:
    """Import torch lazily; never call this at module import time."""

    return importlib.import_module("torch")


def _clip01(value: Any) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _setting(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _status_unavailable(reason: str, message: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": BACKEND_NAME,
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "message": message,
        "device": None,
        "device_index": None,
        "torch_version": None,
        "cuda_version": None,
    }
    if extra:
        result.update(extra)
    return result


def _status_available(torch_module: Any, device_index: int) -> dict[str, Any]:
    cuda = torch_module.cuda
    try:
        device_name = str(cuda.get_device_name(device_index))
    except Exception:
        device_name = f"cuda:{device_index}"
    version_obj = getattr(torch_module, "version", None)
    return {
        "backend": BACKEND_NAME,
        "status": "available",
        "available": True,
        "reason": None,
        "message": f"CUDA tensor backend available: {device_name}",
        "device": device_name,
        "device_index": int(device_index),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_version": getattr(version_obj, "cuda", None),
    }


def is_available(*, torch_loader: Any | None = None, refresh: bool = False) -> dict[str, Any]:
    """Return torch/CUDA availability without importing torch until called."""

    global _AVAILABILITY_CACHE
    use_cache = torch_loader is None
    if use_cache and _AVAILABILITY_CACHE is not None and not refresh:
        return copy.deepcopy(_AVAILABILITY_CACHE)

    try:
        torch_module = (torch_loader or _import_torch)()
    except Exception as exc:
        result = _status_unavailable(
            "torch_import_failed",
            f"Torch CUDA backend is unavailable because torch could not be imported: {type(exc).__name__}: {exc}",
            extra={"import_error": f"{type(exc).__name__}: {exc}"},
        )
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        result = _status_unavailable("cuda_unavailable", "Torch imported, but torch.cuda is not present.")
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    try:
        cuda_available = bool(cuda.is_available())
    except Exception as exc:
        result = _status_unavailable("cuda_unavailable", f"torch.cuda.is_available() failed: {type(exc).__name__}: {exc}")
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result
    if not cuda_available:
        result = _status_unavailable("cuda_unavailable", "torch.cuda.is_available() is false; CPU path will be used.")
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    try:
        device_count = int(cuda.device_count())
    except Exception as exc:
        result = _status_unavailable("cuda_unavailable", f"torch.cuda.device_count() failed: {type(exc).__name__}: {exc}")
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result
    if device_count <= 0:
        result = _status_unavailable("cuda_unavailable", "torch.cuda reported no CUDA devices; CPU path will be used.")
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    try:
        device_index = int(cuda.current_device())
    except Exception:
        device_index = 0
    result = _status_available(torch_module, device_index)
    if use_cache:
        _AVAILABILITY_CACHE = copy.deepcopy(result)
    return result


def _fallback(reason: str, message: str, *, elapsed_ms: float | None = None, availability: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "used": False,
        "backend": BACKEND_NAME,
        "reason": reason,
        "message": message,
        "rgb": None,
        "repair_mask": None,
        "elapsed_ms": elapsed_ms,
    }
    if availability is not None:
        result["availability"] = availability
    return result


def _ok(rgb: np.ndarray, repair_mask: np.ndarray, message: str, elapsed_ms: float, availability: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "used": True,
        "backend": BACKEND_NAME,
        "reason": None,
        "message": message,
        "rgb": rgb,
        "repair_mask": repair_mask,
        "elapsed_ms": float(elapsed_ms),
        "availability": availability,
    }


def _as_rgb_u8(tile: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(tile)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{name} must have shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_u8_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, -1] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(f"{name} must match tile shape")
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_bool_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, -1] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(f"{name} must match tile shape")
    if arr.dtype == bool:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr > 0)


def process_color_tile_gpu(
    rgb_tile: np.ndarray,
    alpha_tile: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_tile: np.ndarray | None,
    nearest_fg_rgb: np.ndarray | None,
    nearest_fg_valid: np.ndarray | None,
    screen_color: tuple[int, int, int],
    settings: Any,
    *,
    force_gpu: bool = False,
    torch_loader: Any | None = None,
) -> dict[str, Any]:
    """Run the classical transition RGB repair tile on torch/CUDA when useful.

    The CPU path remains the reference. This function returns a structured
    fallback instead of raising for normal backend/precondition misses.
    """

    start = time.perf_counter()
    if not bool(_setting(settings, "transition_unmix", True)):
        return _fallback("transition_disabled", "Transition unmix is disabled; CPU color path remains active.")
    if int(np.clip(int(_setting(settings, "foreground_reference_radius", 96)), 0, np.iinfo(np.uint16).max - 1)) <= 0:
        return _fallback("reference_radius_disabled", "Foreground reference radius is disabled; CPU transition repair leaves the tile unchanged.")
    if nearest_fg_rgb is None or nearest_fg_valid is None:
        return _fallback("no_foreground_reference", "No foreground reference tile is available for GPU transition repair.")

    try:
        rgb = _as_rgb_u8(rgb_tile, "rgb_tile")
        shape = rgb.shape[:2]
        alpha = _as_u8_mask(alpha_tile, shape, "alpha_tile")
        background = _as_bool_mask(background_mask, shape, "background_mask")
        edge = _as_bool_mask(edge_mask, shape, "edge_mask")
        probability_u8 = _as_u8_mask(probability, shape, "probability")
        fringe_u8 = _as_u8_mask(fringe_mask, shape, "fringe_mask")
        foreground_rgb = _as_rgb_u8(nearest_fg_rgb, "nearest_fg_rgb")
        foreground_valid = _as_bool_mask(nearest_fg_valid, shape, "nearest_fg_valid")
        if screen_tile is not None:
            screen_rgb = _as_rgb_u8(screen_tile, "screen_tile")
        else:
            screen_rgb = None
    except Exception as exc:
        return _fallback("invalid_inputs", f"GPU transition repair input validation failed: {type(exc).__name__}: {exc}")

    if not np.any(foreground_valid):
        return _fallback("no_foreground_reference", "Foreground reference mask is empty for this tile.")
    if int(np.max(alpha)) <= 0:
        return _fallback("transparent_tile", "Tile alpha is fully transparent; CPU zero-RGB invariant remains active.")

    availability = is_available(torch_loader=torch_loader, refresh=torch_loader is not None)
    if not availability.get("available"):
        message = str(availability.get("message") or "CUDA unavailable")
        if "cpu" not in message.lower():
            message += " CPU color path will be used."
        return _fallback(str(availability.get("reason") or "cuda_unavailable"), message, availability=availability)

    try:
        torch = (torch_loader or _import_torch)()
        device_index = int(availability.get("device_index") or 0)
        device = torch.device(f"cuda:{device_index}") if hasattr(torch, "device") else f"cuda:{device_index}"
        with torch.no_grad():
            luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=torch.float32).view(1, 1, 3)

            def srgb_to_linear(srgb: Any) -> Any:
                srgb_f = torch.clamp(srgb, 0.0, 1.0)
                return torch.where(srgb_f <= 0.04045, srgb_f / 12.92, torch.pow((srgb_f + 0.055) / 1.055, 2.4)).to(torch.float32)

            def linear_to_srgb(linear: Any) -> Any:
                linear_f = torch.clamp(linear, 0.0, 1.0)
                return torch.where(linear_f <= 0.0031308, linear_f * 12.92, 1.055 * torch.pow(linear_f, 1.0 / 2.4) - 0.055).to(torch.float32)

            def linear_luma(linear: Any) -> Any:
                return torch.sum(torch.clamp(linear, 0.0, 1.0) * luma_weights, dim=2)

            def smoothstep(edge0: float, edge1: float, x: Any) -> Any:
                if edge1 <= edge0:
                    return (x >= edge1).to(torch.float32)
                t = torch.clamp((x - float(edge0)) / float(edge1 - edge0), 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)

            def match_luma(linear: Any, target_luma: Any) -> Any:
                rgb_l = torch.clamp(linear, 0.0, 1.0)
                src_luma = linear_luma(rgb_l)
                target = torch.clamp(target_luma, 0.0, 1.0)
                scale = torch.where(src_luma > 1e-5, target / torch.clamp(src_luma, min=1e-5), torch.ones_like(target))
                scale = torch.clamp(scale, 0.0, 4.0)
                return torch.clamp(rgb_l * scale.unsqueeze(2), 0.0, 1.0)

            rgb_norm = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).div_(255.0)
            alpha_byte = torch.from_numpy(alpha).to(device=device, dtype=torch.uint8)
            alpha_f = alpha_byte.to(torch.float32).div(255.0)
            background_t = torch.from_numpy(background).to(device=device, dtype=torch.bool)
            edge_t = torch.from_numpy(edge).to(device=device, dtype=torch.bool)
            prob_t = torch.from_numpy(probability_u8).to(device=device, dtype=torch.uint8)
            fringe_t = torch.from_numpy(fringe_u8).to(device=device, dtype=torch.uint8)
            fg_norm = torch.from_numpy(foreground_rgb).to(device=device, dtype=torch.float32).div_(255.0)
            fg_valid_t = torch.from_numpy(foreground_valid).to(device=device, dtype=torch.bool)

            if screen_rgb is None:
                screen_norm = torch.tensor(np.asarray(screen_color, dtype=np.float32) / 255.0, device=device, dtype=torch.float32).view(1, 1, 3)
            else:
                screen_norm = torch.from_numpy(screen_rgb).to(device=device, dtype=torch.float32).div_(255.0)

            key = np.clip(np.asarray(screen_color, dtype=np.float32) / 255.0, 1e-4, 1.0)
            key_channel = int(np.argmax(key))
            other = [idx for idx in range(3) if idx != key_channel]
            key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
            if key_dom > 0.12:
                key_values = rgb_norm[:, :, key_channel]
                other_max = torch.maximum(rgb_norm[:, :, other[0]], rgb_norm[:, :, other[1]])
                spill_strength = torch.clamp(torch.clamp(key_values - other_max, min=0.0) / torch.clamp(key_values, min=1.0 / 255.0), 0.0, 1.0)
            else:
                key_t = torch.tensor(key, device=device, dtype=torch.float32)
                key_luma_scalar = torch.sum(key_t * luma_weights.view(3))
                key_vec_scalar = key_t - key_luma_scalar
                key_norm = torch.linalg.vector_norm(key_vec_scalar)
                if float(key_norm.detach().cpu().item()) < 1e-4:
                    spill_strength = torch.zeros(shape, device=device, dtype=torch.float32)
                else:
                    key_vec_scalar = key_vec_scalar / key_norm
                    pix_luma = torch.sum(rgb_norm * luma_weights, dim=2)
                    residual = rgb_norm - pix_luma.unsqueeze(2)
                    projection = torch.sum(residual * key_vec_scalar.view(1, 1, 3), dim=2)
                    spill_strength = torch.clamp(torch.clamp(projection, min=0.0), 0.0, 1.0)

            prob_limit = max(64, int(round(_clip01(_setting(settings, "clip_foreground", 0.14)) * 255.0)) + 32)
            foreground_core = (alpha_byte >= 250) & (~background_t) & (prob_t <= prob_limit) & (fringe_t <= 24)
            alpha_min = int(np.clip(int(_setting(settings, "transition_alpha_min", 2)), 0, 255))
            alpha_max = int(np.clip(int(_setting(settings, "transition_alpha_max", 253)), 0, 255))
            if alpha_max < alpha_min:
                alpha_min, alpha_max = alpha_max, alpha_min
            semi = (alpha_byte >= alpha_min) & (alpha_byte <= alpha_max)
            protected_semi = semi & (alpha_byte < 240)
            live = (alpha_byte > 0) & (~background_t)
            live_edge = edge_t & live
            live_fringe = (fringe_t > 0) & live
            protected_core_fringe = (fringe_t > 24) & live
            live_spill = (spill_strength > float(_setting(settings, "transition_spill_threshold", 0.08))) & live
            eligible = semi | live_edge | live_fringe | live_spill
            near_opaque_core = (alpha_byte >= 240) & (~background_t) & (fringe_t <= 24)
            protected_core = (foreground_core | near_opaque_core) & (alpha_byte >= 240)
            core_allowed = (~protected_core) | protected_semi | protected_core_fringe
            transition = live & eligible & core_allowed
            eligible_mask = transition & fg_valid_t & (alpha_byte > 0)
            if not bool(torch.any(eligible_mask).detach().cpu().item()):
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000.0
                return _fallback("no_eligible_pixels", "No transition pixels are eligible for GPU repair in this tile.", elapsed_ms=elapsed, availability=availability)

            source_linear = srgb_to_linear(rgb_norm)
            foreground_linear = srgb_to_linear(fg_norm)
            screen_linear = srgb_to_linear(screen_norm)
            safe_alpha = torch.clamp(alpha_f, min=1.0 / 255.0)
            foreground_est = (source_linear - (1.0 - alpha_f).unsqueeze(2) * screen_linear) / safe_alpha.unsqueeze(2)
            foreground_est = torch.nan_to_num(foreground_est, nan=0.0, posinf=1.0, neginf=0.0)
            foreground_est = torch.clamp(foreground_est, 0.0, 1.0)

            recon = alpha_f.unsqueeze(2) * foreground_est + (1.0 - alpha_f).unsqueeze(2) * screen_linear
            recon_error = torch.linalg.vector_norm(source_linear - recon, dim=2)
            reconstruction_limit = max(float(_setting(settings, "transition_reconstruction_error", 0.08)) * 1.25, 1e-4)
            eligible_mask = eligible_mask & (recon_error <= reconstruction_limit)

            key_luma = linear_luma(screen_linear)
            key_vec = torch.clamp(screen_linear, 0.0, 1.0) - key_luma.unsqueeze(2)
            norm = torch.linalg.vector_norm(key_vec, dim=2)
            key_vec_valid = norm >= 1e-5
            key_unit = torch.where(key_vec_valid.unsqueeze(2), key_vec / torch.clamp(norm, min=1e-5).unsqueeze(2), torch.zeros_like(key_vec))

            foreground_luma = linear_luma(foreground_est)
            reference_luma = linear_luma(foreground_linear)
            foreground_chroma = foreground_est - foreground_luma.unsqueeze(2)
            vector_spill = torch.clamp(torch.sum(foreground_chroma * key_unit, dim=2), min=0.0)
            vector_spill = torch.where(key_vec_valid, vector_spill, torch.zeros_like(vector_spill))

            edge_strength = torch.clamp(alpha_f * (1.0 - alpha_f) * 4.0, 0.0, 1.0)
            edge_strength = torch.maximum(edge_strength, edge_t.to(torch.float32) * 0.45)
            fringe_signal = fringe_t.to(torch.float32).div(255.0)
            near_screen = prob_t.to(torch.float32).div(255.0) * torch.clamp(1.0 - alpha_f, 0.0, 1.0)
            spill_gate = torch.maximum(
                torch.maximum(torch.clamp(spill_strength, 0.0, 1.0), smoothstep(0.005, 0.18, vector_spill)),
                torch.maximum(near_screen, fringe_signal * 0.75),
            )
            transition_strength = torch.maximum(torch.maximum(edge_strength, fringe_signal), near_screen)
            repair_strength = torch.clamp(transition_strength * torch.maximum(spill_gate, torch.tensor(0.35, device=device)), 0.0, 1.0)
            repair_strength = torch.where(eligible_mask, repair_strength, torch.zeros_like(repair_strength))
            if not bool(torch.any(repair_strength > (1.0 / 255.0)).detach().cpu().item()):
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000.0
                return _fallback("no_repair_pixels", "GPU transition repair found no pixels above the repair threshold in this tile.", elapsed_ms=elapsed, availability=availability)

            cleaned = foreground_est.clone()
            despill_amount = _clip01(_setting(settings, "key_vector_despill", 0.75))
            if despill_amount > 0:
                cleaned = cleaned - key_unit * (vector_spill * despill_amount * repair_strength).unsqueeze(2)
                cleaned = torch.clamp(cleaned, 0.0, 1.0)

            pull_amount = _clip01(_setting(settings, "foreground_reference_pull", 0.65))
            if pull_amount > 0:
                pull = torch.clamp(repair_strength * spill_gate * pull_amount, 0.0, 1.0)
                reference_luma_matched = match_luma(foreground_linear, linear_luma(cleaned))
                cleaned = cleaned * (1.0 - pull.unsqueeze(2)) + reference_luma_matched * pull.unsqueeze(2)

            luma_preserve = _clip01(_setting(settings, "preserve_foreground_luma", 0.85))
            if luma_preserve > 0:
                preserve = torch.clamp(repair_strength * luma_preserve, 0.0, 1.0)
                luma_matched = match_luma(cleaned, reference_luma)
                cleaned = cleaned * (1.0 - preserve.unsqueeze(2)) + luma_matched * preserve.unsqueeze(2)

            cleaned = torch.clamp(torch.nan_to_num(cleaned, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            repaired = torch.clamp(torch.round(linear_to_srgb(cleaned) * 255.0), 0, 255).to(torch.uint8)
            changed = repair_strength > (1.0 / 255.0)
            out = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.uint8)
            out[changed] = repaired[changed]
            out[alpha_byte <= 0] = 0

            original = torch.from_numpy(rgb).to(device=device, dtype=torch.int16)
            delta = torch.amax(torch.abs(out.to(torch.int16) - original), dim=2).to(torch.float32).div(255.0)
            repair_mask_f = torch.maximum(repair_strength, delta)
            repair_mask = torch.clamp(torch.round(torch.clamp(repair_mask_f, 0.0, 1.0) * 255.0), 0, 255).to(torch.uint8)
            repair_mask[alpha_byte <= 0] = 0

            torch.cuda.synchronize(device)
            out_np = out.detach().cpu().numpy().astype(np.uint8, copy=False)
            mask_np = repair_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return _fallback("cuda_execution_failed", f"GPU transition repair failed; CPU fallback is required: {type(exc).__name__}: {exc}", elapsed_ms=elapsed, availability=availability)

    elapsed = (time.perf_counter() - start) * 1000.0
    mode = "forced" if force_gpu else "auto"
    return _ok(out_np, mask_np, f"GPU transition repair completed on {BACKEND_NAME} ({mode}).", elapsed, availability)


def process_preview_gpu(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reserved preview-wide backend hook; no safe preview kernel is shipped yet."""

    return _fallback("not_implemented", "No preview-wide GPU kernel is shipped; CPU preview remains the reference.")
