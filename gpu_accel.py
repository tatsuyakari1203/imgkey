from __future__ import annotations

import copy
import ctypes
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


BACKEND_ID = "compact_cuda_dll"
BACKEND_NAME = "compact CUDA DLL"
CUDA_DLL_NAME = "imgkey_cuda.dll"
IMGKEY_CUDA_OK = 0
IMGKEY_CUDA_INVALID_ARGUMENT = 1
IMGKEY_CUDA_NO_DEVICE = 2
IMGKEY_CUDA_LAUNCH_FAILED = 3
IMGKEY_CUDA_UNSUPPORTED_VERSION = 4
_CUDA_ABI_VERSION = 1
_MAX_TILE_PIXELS = 16 * 1024 * 1024
_U8_PTR = ctypes.POINTER(ctypes.c_ubyte)

_AVAILABILITY_CACHE: dict[str, Any] | None = None
_DLL_CACHE: "_CudaDll" | None = None
_DLL_CACHE_KEY: str | None = None


class CudaDllUnavailable(RuntimeError):
    """Raised when the compact CUDA DLL cannot be found or loaded."""


class CudaDllError(RuntimeError):
    """Raised when the CUDA DLL returns a non-success ABI status."""

    def __init__(self, status: int, message: str):
        super().__init__(f"imgkey_cuda status {status}: {message}")
        self.status = int(status)
        self.message = message


class ImgKeyCudaTransitionParamsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_int),
        ("version", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("rgb_stride_bytes", ctypes.c_int),
        ("alpha_stride_bytes", ctypes.c_int),
        ("mask_stride_bytes", ctypes.c_int),
        ("out_stride_bytes", ctypes.c_int),
        ("foreground_reference_pull", ctypes.c_float),
        ("key_vector_despill", ctypes.c_float),
        ("preserve_foreground_luma", ctypes.c_float),
        ("transition_spill_threshold", ctypes.c_float),
        ("screen_r", ctypes.c_ubyte),
        ("screen_g", ctypes.c_ubyte),
        ("screen_b", ctypes.c_ubyte),
    ]


class _CudaDll:
    def __init__(self, path: Path, library: ctypes.CDLL):
        self.path = path
        self.library = library
        self.library.imgkey_cuda_version.argtypes = []
        self.library.imgkey_cuda_version.restype = ctypes.c_int
        self.library.imgkey_cuda_device_count.argtypes = []
        self.library.imgkey_cuda_device_count.restype = ctypes.c_int
        self.library.imgkey_cuda_last_error.argtypes = []
        self.library.imgkey_cuda_last_error.restype = ctypes.c_char_p
        self.library.imgkey_cuda_transition_repair_v1.argtypes = [
            ctypes.POINTER(ImgKeyCudaTransitionParamsV1),
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
        ]
        self.library.imgkey_cuda_transition_repair_v1.restype = ctypes.c_int

    def last_error(self) -> str:
        raw = self.library.imgkey_cuda_last_error()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    def version(self) -> int:
        return int(self.library.imgkey_cuda_version())

    def device_count(self) -> int:
        return int(self.library.imgkey_cuda_device_count())

    def transition_repair(
        self,
        params: ImgKeyCudaTransitionParamsV1,
        rgb: np.ndarray,
        alpha: np.ndarray,
        transition_mask: np.ndarray,
        foreground_ref_rgb: np.ndarray,
        foreground_ref_valid: np.ndarray,
        out_rgb: np.ndarray,
        out_repair_mask: np.ndarray,
    ) -> None:
        status = int(
            self.library.imgkey_cuda_transition_repair_v1(
                ctypes.byref(params),
                rgb.ctypes.data_as(_U8_PTR),
                alpha.ctypes.data_as(_U8_PTR),
                transition_mask.ctypes.data_as(_U8_PTR),
                foreground_ref_rgb.ctypes.data_as(_U8_PTR),
                foreground_ref_valid.ctypes.data_as(_U8_PTR),
                out_rgb.ctypes.data_as(_U8_PTR),
                out_repair_mask.ctypes.data_as(_U8_PTR),
            )
        )
        if status != IMGKEY_CUDA_OK:
            raise CudaDllError(status, self.last_error())


def _clip01(value: Any) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _setting(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _status_unavailable(reason: str, message: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": BACKEND_ID,
        "backend_name": BACKEND_NAME,
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "message": message,
        "device": None,
        "device_index": None,
        "device_count": 0,
        "version": None,
        "dll_path": None,
        "cuda_version": None,
    }
    if extra:
        result.update(extra)
    return result


def _status_available(backend: _CudaDll, device_count: int) -> dict[str, Any]:
    return {
        "backend": BACKEND_ID,
        "backend_name": BACKEND_NAME,
        "status": "available",
        "available": True,
        "reason": None,
        "message": f"Compact CUDA DLL backend available ({device_count} CUDA device(s)).",
        "device": f"CUDA device 0 ({device_count} device(s) visible)",
        "device_index": 0,
        "device_count": int(device_count),
        "version": backend.version(),
        "dll_path": str(backend.path),
        "cuda_version": None,
    }


def _candidate_dll_paths(dll_path: str | os.PathLike[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    if dll_path is not None:
        return [Path(dll_path).expanduser()]
    env_path = os.environ.get("IMGKEY_CUDA_DLL")
    if env_path:
        paths.append(Path(env_path))
    module_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            module_dir / CUDA_DLL_NAME,
            module_dir / "native" / "imgkey_cuda" / "build" / CUDA_DLL_NAME,
            Path.cwd() / "native" / "imgkey_cuda" / "build" / CUDA_DLL_NAME,
        ]
    )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        paths.append(Path(str(meipass)) / CUDA_DLL_NAME)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(path.expanduser())
    return unique


def _load_cuda_dll(dll_path: str | os.PathLike[str] | None = None, *, refresh: bool = False) -> _CudaDll:
    global _DLL_CACHE, _DLL_CACHE_KEY
    candidates = _candidate_dll_paths(dll_path)
    cache_key = str(candidates[0]) if dll_path is not None else "<default>"
    if not refresh and _DLL_CACHE is not None and _DLL_CACHE_KEY == cache_key:
        return _DLL_CACHE

    checked: list[str] = []
    load_errors: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            checked.append(str(candidate))
            continue
        try:
            library = ctypes.CDLL(str(resolved))
        except OSError as exc:
            load_errors.append(f"{resolved}: {exc}")
            continue
        backend = _CudaDll(resolved, library)
        if dll_path is None:
            _DLL_CACHE = backend
            _DLL_CACHE_KEY = cache_key
        return backend
    detail = "; ".join(load_errors) if load_errors else "checked " + ", ".join(checked)
    raise CudaDllUnavailable(f"{CUDA_DLL_NAME} was not found or could not be loaded ({detail})")


def is_available(
    *,
    refresh: bool = False,
    dll_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return compact CUDA DLL availability without loading anything at import time."""

    global _AVAILABILITY_CACHE
    use_cache = dll_path is None
    if use_cache and _AVAILABILITY_CACHE is not None and not refresh:
        return copy.deepcopy(_AVAILABILITY_CACHE)

    try:
        backend = _load_cuda_dll(dll_path, refresh=refresh)
    except Exception as exc:
        result = _status_unavailable(
            "cuda_dll_unavailable",
            f"Compact CUDA DLL backend is unavailable: {type(exc).__name__}: {exc}. CPU color path will be used.",
            extra={"load_error": f"{type(exc).__name__}: {exc}"},
        )
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    try:
        device_count = backend.device_count()
    except Exception as exc:
        result = _status_unavailable(
            "cuda_dll_probe_failed",
            f"Compact CUDA DLL device probe failed: {type(exc).__name__}: {exc}. CPU color path will be used.",
            extra={"dll_path": str(backend.path), "probe_error": f"{type(exc).__name__}: {exc}"},
        )
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result
    if device_count <= 0:
        result = _status_unavailable(
            "cuda_no_device",
            f"Compact CUDA DLL reported no CUDA devices: {backend.last_error() or 'device_count <= 0'}. CPU color path will be used.",
            extra={"dll_path": str(backend.path), "version": backend.version()},
        )
        if use_cache:
            _AVAILABILITY_CACHE = copy.deepcopy(result)
        return result

    result = _status_available(backend, device_count)
    if use_cache:
        _AVAILABILITY_CACHE = copy.deepcopy(result)
    return result


def _fallback(reason: str, message: str, *, elapsed_ms: float | None = None, availability: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "used": False,
        "backend": BACKEND_ID,
        "backend_name": BACKEND_NAME,
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
        "backend": BACKEND_ID,
        "backend_name": BACKEND_NAME,
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


def _require_u8_c_array(name: str, arr: np.ndarray, ndim: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if arr.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous before calling the CUDA DLL")
    return arr


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / float(edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


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


def _linear_luma(rgb_linear: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(np.clip(rgb_linear, 0.0, 1.0) * weights.reshape(1, 1, 3), axis=2).astype(np.float32)


def _match_luma_linear(rgb_linear: np.ndarray, target_luma: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float32), 0.0, 1.0)
    src_luma = _linear_luma(rgb)
    target = np.clip(np.asarray(target_luma, dtype=np.float32), 0.0, 1.0)
    scale = np.divide(target, np.maximum(src_luma, 1e-5), out=np.ones_like(target), where=src_luma > 1e-5)
    scale = np.clip(scale, 0.0, 4.0)
    return np.clip(rgb * scale[:, :, None], 0.0, 1.0)


def _compute_key_spill_strength(rgb: np.ndarray, screen_color: tuple[int, int, int]) -> np.ndarray:
    pix = rgb.astype(np.float32) / 255.0
    key = np.asarray(screen_color, dtype=np.float32) / 255.0
    key = np.clip(key, 1e-4, 1.0)
    key_channel = int(np.argmax(key))
    other = [idx for idx in range(3) if idx != key_channel]
    key_dom = float(key[key_channel] - max(key[other[0]], key[other[1]]))
    if key_dom > 0.12:
        key_values = pix[:, :, key_channel]
        other_max = np.maximum(pix[:, :, other[0]], pix[:, :, other[1]])
        return np.clip(np.maximum(key_values - other_max, 0.0) / np.maximum(key_values, 1.0 / 255.0), 0.0, 1.0).astype(np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    key_luma = float(key @ weights)
    key_vec = key - key_luma
    norm = float(np.linalg.norm(key_vec))
    if norm < 1e-4:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    key_vec /= norm
    pix_luma = np.sum(pix * weights.reshape(1, 1, 3), axis=2)
    residual = pix - pix_luma[:, :, None]
    projection = np.sum(residual * key_vec.reshape(1, 1, 3), axis=2)
    return np.clip(np.maximum(projection, 0.0), 0.0, 1.0).astype(np.float32)


def _build_foreground_core_mask(
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    settings: Any,
) -> np.ndarray:
    prob_limit = max(64, int(round(_clip01(_setting(settings, "clip_foreground", 0.14)) * 255.0)) + 32)
    return ((alpha_u8 >= 250) & (~background_mask) & (probability <= prob_limit) & (fringe_mask <= 24)).astype(bool, copy=False)


def _build_transition_repair_mask(
    alpha_u8: np.ndarray,
    edge_mask: np.ndarray,
    fringe_mask: np.ndarray,
    spill_strength: np.ndarray,
    background_mask: np.ndarray,
    foreground_core_mask: np.ndarray,
    settings: Any,
) -> np.ndarray:
    alpha_min = int(np.clip(int(_setting(settings, "transition_alpha_min", 2)), 0, 255))
    alpha_max = int(np.clip(int(_setting(settings, "transition_alpha_max", 253)), 0, 255))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    semi = (alpha_u8 >= alpha_min) & (alpha_u8 <= alpha_max)
    protected_semi = semi & (alpha_u8 < 240)
    live = (alpha_u8 > 0) & (~background_mask)
    live_edge = edge_mask & live
    live_fringe = (fringe_mask > 0) & live
    protected_core_fringe = (fringe_mask > 24) & live
    live_spill = (spill_strength > float(_setting(settings, "transition_spill_threshold", 0.08))) & live
    eligible = semi | live_edge | live_fringe | live_spill
    near_opaque_core = (alpha_u8 >= 240) & (~background_mask) & (fringe_mask <= 24)
    protected_core = (foreground_core_mask | near_opaque_core) & (alpha_u8 >= 240)
    core_allowed = (~protected_core) | protected_semi | protected_core_fringe
    return (live & eligible & core_allowed).astype(bool, copy=False)


def _screen_linear_for_shape(shape: tuple[int, int], screen_color: tuple[int, int, int]) -> np.ndarray:
    screen = np.empty((*shape, 3), dtype=np.uint8)
    screen[:, :, :] = np.asarray(screen_color, dtype=np.uint8).reshape(1, 1, 3)
    return _srgb_u8_to_linear_f32(screen)


def _screen_chroma_unit_vectors(screen_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key_luma = _linear_luma(screen_linear)
    key_vec = np.clip(screen_linear, 0.0, 1.0) - key_luma[:, :, None]
    norm = np.linalg.norm(key_vec, axis=2).astype(np.float32)
    valid = norm >= 1e-5
    unit = np.divide(key_vec, np.maximum(norm[:, :, None], 1e-5), out=np.zeros_like(key_vec), where=valid[:, :, None])
    return unit.astype(np.float32, copy=False), valid


def transition_repair_strength_mask_v1(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    foreground_ref_rgb: np.ndarray,
    foreground_ref_valid: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any,
) -> np.ndarray:
    """Precompute the compact DLL's final transition repair strength mask."""

    shape = alpha_u8.shape
    spill_strength = _compute_key_spill_strength(rgb, screen_color)
    foreground_core = _build_foreground_core_mask(alpha_u8, background_mask, probability, fringe_mask, settings)
    transition = _build_transition_repair_mask(alpha_u8, edge_mask, fringe_mask, spill_strength, background_mask, foreground_core, settings)
    eligible = transition & foreground_ref_valid & (alpha_u8 > 0)
    if not np.any(eligible):
        return np.zeros(shape, dtype=np.uint8)

    source_linear = _srgb_u8_to_linear_f32(rgb)
    foreground_linear = _srgb_u8_to_linear_f32(foreground_ref_rgb)
    screen_linear = _screen_linear_for_shape(shape, screen_color)
    alpha_f = alpha_u8.astype(np.float32) / 255.0
    safe_alpha = np.maximum(alpha_f, 1.0 / 255.0)
    foreground_est = (source_linear - (1.0 - alpha_f[:, :, None]) * screen_linear) / safe_alpha[:, :, None]
    foreground_est = np.nan_to_num(foreground_est, nan=0.0, posinf=1.0, neginf=0.0)
    foreground_est = np.clip(foreground_est, 0.0, 1.0).astype(np.float32, copy=False)

    recon = alpha_f[:, :, None] * foreground_est + (1.0 - alpha_f[:, :, None]) * screen_linear
    recon_error = np.linalg.norm(source_linear - recon, axis=2)
    reconstruction_limit = max(float(_setting(settings, "transition_reconstruction_error", 0.08)) * 1.25, 1e-4)
    eligible &= recon_error <= reconstruction_limit
    if not np.any(eligible):
        return np.zeros(shape, dtype=np.uint8)

    key_vec, key_vec_valid = _screen_chroma_unit_vectors(screen_linear)
    foreground_luma = _linear_luma(foreground_est)
    foreground_chroma = foreground_est - foreground_luma[:, :, None]
    vector_spill = np.maximum(np.sum(foreground_chroma * key_vec, axis=2), 0.0)
    vector_spill = np.where(key_vec_valid, vector_spill, 0.0).astype(np.float32)

    edge_strength = np.clip(alpha_f * (1.0 - alpha_f) * 4.0, 0.0, 1.0)
    edge_strength = np.maximum(edge_strength, edge_mask.astype(np.float32) * 0.45)
    fringe_signal = fringe_mask.astype(np.float32) / 255.0
    near_screen = (probability.astype(np.float32) / 255.0) * np.clip(1.0 - alpha_f, 0.0, 1.0)
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
    return np.rint(np.clip(repair_strength, 0.0, 1.0) * 255.0).astype(np.uint8)


def transition_repair_cpu_v1(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    transition_mask: np.ndarray,
    foreground_ref_rgb: np.ndarray,
    foreground_ref_valid: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference for the compact DLL v1 ABI inputs."""

    rgb_arr = _as_rgb_u8(rgb, "rgb")
    shape = rgb_arr.shape[:2]
    alpha = _as_u8_mask(alpha_u8, shape, "alpha_u8")
    repair_strength_u8 = _as_u8_mask(transition_mask, shape, "transition_mask")
    foreground_rgb = _as_rgb_u8(foreground_ref_rgb, "foreground_ref_rgb")
    foreground_valid = _as_bool_mask(foreground_ref_valid, shape, "foreground_ref_valid")
    if foreground_rgb.shape[:2] != shape:
        raise ValueError("foreground_ref_rgb must match rgb shape")

    out = rgb_arr.copy()
    repair_mask = np.zeros(shape, dtype=np.uint8)
    repair_strength = repair_strength_u8.astype(np.float32) / 255.0
    active = (repair_strength > (1.0 / 255.0)) & foreground_valid & (alpha > 0)
    out[alpha <= 0] = 0
    if not np.any(active):
        return out, repair_mask

    source_linear = _srgb_u8_to_linear_f32(rgb_arr)
    foreground_linear = _srgb_u8_to_linear_f32(foreground_rgb)
    screen_linear = _screen_linear_for_shape(shape, screen_color)
    alpha_f = alpha.astype(np.float32) / 255.0
    safe_alpha = np.maximum(alpha_f, 1.0 / 255.0)
    foreground_est = (source_linear - (1.0 - alpha_f[:, :, None]) * screen_linear) / safe_alpha[:, :, None]
    foreground_est = np.nan_to_num(foreground_est, nan=0.0, posinf=1.0, neginf=0.0)
    foreground_est = np.clip(foreground_est, 0.0, 1.0).astype(np.float32, copy=False)

    key_vec, key_vec_valid = _screen_chroma_unit_vectors(screen_linear)
    foreground_luma = _linear_luma(foreground_est)
    reference_luma = _linear_luma(foreground_linear)
    foreground_chroma = foreground_est - foreground_luma[:, :, None]
    vector_spill = np.maximum(np.sum(foreground_chroma * key_vec, axis=2), 0.0)
    vector_spill = np.where(key_vec_valid, vector_spill, 0.0).astype(np.float32)

    cleaned = foreground_est.copy()
    despill_amount = _clip01(_setting(settings, "key_vector_despill", 0.75))
    if despill_amount > 0:
        cleaned -= key_vec * (vector_spill * despill_amount * repair_strength)[:, :, None]
        cleaned = np.clip(cleaned, 0.0, 1.0)

    pull_amount = _clip01(_setting(settings, "foreground_reference_pull", 0.65))
    if pull_amount > 0:
        pull = np.clip(repair_strength * pull_amount, 0.0, 1.0)
        if np.any(pull > 0):
            reference_luma_matched = _match_luma_linear(foreground_linear, _linear_luma(cleaned))
            cleaned = cleaned * (1.0 - pull[:, :, None]) + reference_luma_matched * pull[:, :, None]

    luma_preserve = _clip01(_setting(settings, "preserve_foreground_luma", 0.85))
    if luma_preserve > 0:
        preserve = np.clip(repair_strength * luma_preserve, 0.0, 1.0)
        if np.any(preserve > 0):
            luma_matched = _match_luma_linear(cleaned, reference_luma)
            cleaned = cleaned * (1.0 - preserve[:, :, None]) + luma_matched * preserve[:, :, None]

    cleaned = np.clip(np.nan_to_num(cleaned, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    repaired = _linear_f32_to_srgb_u8(cleaned)
    out[active] = repaired[active]
    out[alpha <= 0] = 0

    delta = np.max(np.abs(out.astype(np.int16) - rgb_arr.astype(np.int16)), axis=2).astype(np.float32) / 255.0
    repair_mask_f = np.maximum(repair_strength, delta)
    repair_mask[repair_mask_f > 0] = np.rint(np.clip(repair_mask_f[repair_mask_f > 0], 0.0, 1.0) * 255.0).astype(np.uint8)
    repair_mask[alpha <= 0] = 0
    return out, repair_mask


def _params_for_call(
    rgb: np.ndarray,
    alpha: np.ndarray,
    transition_mask: np.ndarray,
    out_rgb: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any,
) -> ImgKeyCudaTransitionParamsV1:
    h, w = alpha.shape
    return ImgKeyCudaTransitionParamsV1(
        struct_size=ctypes.sizeof(ImgKeyCudaTransitionParamsV1),
        version=_CUDA_ABI_VERSION,
        width=int(w),
        height=int(h),
        rgb_stride_bytes=int(rgb.strides[0]),
        alpha_stride_bytes=int(alpha.strides[0]),
        mask_stride_bytes=int(transition_mask.strides[0]),
        out_stride_bytes=int(out_rgb.strides[0]),
        foreground_reference_pull=float(_clip01(_setting(settings, "foreground_reference_pull", 0.65))),
        key_vector_despill=float(_clip01(_setting(settings, "key_vector_despill", 0.75))),
        preserve_foreground_luma=float(_clip01(_setting(settings, "preserve_foreground_luma", 0.85))),
        transition_spill_threshold=float(_setting(settings, "transition_spill_threshold", 0.08)),
        screen_r=int(screen_color[0]),
        screen_g=int(screen_color[1]),
        screen_b=int(screen_color[2]),
    )


def transition_repair_dll_v1(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    transition_mask: np.ndarray,
    foreground_ref_rgb: np.ndarray,
    foreground_ref_valid: np.ndarray,
    screen_color: tuple[int, int, int],
    settings: Any,
    *,
    dll_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Call imgkey_cuda_transition_repair_v1 after strict Python-side validation."""

    rgb_arr = _require_u8_c_array("rgb", rgb, 3)
    if rgb_arr.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    h, w = rgb_arr.shape[:2]
    if h <= 0 or w <= 0 or h * w > _MAX_TILE_PIXELS:
        raise ValueError("tile dimensions must be positive and within the CUDA DLL max tile size")
    alpha = _require_u8_c_array("alpha_u8", alpha_u8, 2, (h, w))
    transition = _require_u8_c_array("transition_mask", transition_mask, 2, (h, w))
    foreground_rgb = _require_u8_c_array("foreground_ref_rgb", foreground_ref_rgb, 3, (h, w, 3))
    foreground_valid = _require_u8_c_array("foreground_ref_valid", foreground_ref_valid, 2, (h, w))
    screen = tuple(int(v) for v in screen_color)
    if len(screen) != 3 or any(v < 0 or v > 255 for v in screen):
        raise ValueError("screen_color must contain three 0..255 channel values")

    out_rgb = np.empty_like(rgb_arr)
    out_repair = np.empty_like(alpha)
    params = _params_for_call(rgb_arr, alpha, transition, out_rgb, screen, settings)
    backend = _load_cuda_dll(dll_path)
    backend.transition_repair(params, rgb_arr, alpha, transition, foreground_rgb, foreground_valid, out_rgb, out_repair)
    return out_rgb, out_repair


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
) -> dict[str, Any]:
    """Run compact CUDA DLL transition RGB repair when available and useful."""

    start = time.perf_counter()
    if not bool(_setting(settings, "transition_unmix", True)):
        return _fallback("transition_disabled", "Transition unmix is disabled; CPU color path remains active.")
    if int(np.clip(int(_setting(settings, "foreground_reference_radius", 96)), 0, np.iinfo(np.uint16).max - 1)) <= 0:
        return _fallback("reference_radius_disabled", "Foreground reference radius is disabled; CPU transition repair leaves the tile unchanged.")
    if nearest_fg_rgb is None or nearest_fg_valid is None:
        return _fallback("no_foreground_reference", "No foreground reference tile is available for CUDA DLL transition repair.")
    if screen_tile is not None:
        return _fallback("unsupported_screen_tile", "CUDA DLL v1 requires a constant screen color; CPU path is used for local screen tiles.")

    try:
        rgb = _as_rgb_u8(rgb_tile, "rgb_tile")
        shape = rgb.shape[:2]
        alpha = _as_u8_mask(alpha_tile, shape, "alpha_tile")
        background = _as_bool_mask(background_mask, shape, "background_mask")
        edge = _as_bool_mask(edge_mask, shape, "edge_mask")
        probability_u8 = _as_u8_mask(probability, shape, "probability")
        fringe_u8 = _as_u8_mask(fringe_mask, shape, "fringe_mask")
        foreground_rgb = _as_rgb_u8(nearest_fg_rgb, "nearest_fg_rgb")
        foreground_valid_bool = _as_bool_mask(nearest_fg_valid, shape, "nearest_fg_valid")
        if foreground_rgb.shape[:2] != shape:
            raise ValueError("nearest_fg_rgb must match tile shape")
    except Exception as exc:
        return _fallback("invalid_inputs", f"CUDA DLL transition repair input validation failed: {type(exc).__name__}: {exc}")

    if not np.any(foreground_valid_bool):
        return _fallback("no_foreground_reference", "Foreground reference mask is empty for this tile.")
    if int(np.max(alpha)) <= 0:
        return _fallback("transparent_tile", "Tile alpha is fully transparent; CPU zero-RGB invariant remains active.")

    availability = is_available(refresh=False)
    if not availability.get("available"):
        return _fallback(str(availability.get("reason") or "cuda_unavailable"), str(availability.get("message") or "CUDA DLL unavailable"), availability=availability)

    try:
        transition_strength = transition_repair_strength_mask_v1(
            rgb,
            alpha,
            background,
            edge,
            probability_u8,
            fringe_u8,
            foreground_rgb,
            foreground_valid_bool,
            tuple(int(np.clip(c, 0, 255)) for c in screen_color),
            settings,
        )
        if not np.any(transition_strength > 0):
            elapsed = (time.perf_counter() - start) * 1000.0
            return _fallback("no_eligible_pixels", "No transition pixels are eligible for CUDA DLL repair in this tile.", elapsed_ms=elapsed, availability=availability)
        foreground_valid = np.ascontiguousarray(foreground_valid_bool.astype(np.uint8) * 255)
        out_np, mask_np = transition_repair_dll_v1(
            rgb,
            alpha,
            transition_strength,
            foreground_rgb,
            foreground_valid,
            tuple(int(np.clip(c, 0, 255)) for c in screen_color),
            settings,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return _fallback("cuda_execution_failed", f"CUDA DLL transition repair failed; CPU fallback is required: {type(exc).__name__}: {exc}", elapsed_ms=elapsed, availability=availability)

    elapsed = (time.perf_counter() - start) * 1000.0
    mode = "forced" if force_gpu else "auto"
    return _ok(out_np, mask_np, f"CUDA DLL transition repair completed ({mode}).", elapsed, availability)


def process_preview_gpu(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reserved preview-wide backend hook; no safe preview kernel is shipped yet."""

    return _fallback("not_implemented", "No preview-wide GPU kernel is shipped; CPU preview remains the reference.")
