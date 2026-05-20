from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum, IntFlag
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np


IMGKEY_GPU_ABI_VERSION = 1
IMGKEY_GPU_BACKEND_API_VERSION = 1


class BackendCapability(IntFlag):
    CONSTANT_SCREEN = 1 << 0
    SCREEN_TILE = 1 << 1
    PERSISTENT_SESSION = 1 << 2
    TILE_BATCH = 1 << 3
    ALPHA_WRITE = 1 << 4
    RGB_ONLY = 1 << 5
    FULL_COLOR_TILE = 1 << 6


CAPABILITY_NAMES: dict[BackendCapability, str] = {
    BackendCapability.CONSTANT_SCREEN: "constant_screen",
    BackendCapability.SCREEN_TILE: "screen_tile",
    BackendCapability.PERSISTENT_SESSION: "persistent_session",
    BackendCapability.TILE_BATCH: "tile_batch",
    BackendCapability.ALPHA_WRITE: "alpha_write",
    BackendCapability.RGB_ONLY: "rgb_only",
    BackendCapability.FULL_COLOR_TILE: "full_color_tile",
}
CAPABILITY_BY_NAME = {name: capability for capability, name in CAPABILITY_NAMES.items()}


class NativeStatus(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    UNSUPPORTED_VERSION = 2
    UNSUPPORTED_CAPABILITY = 3
    BACKEND_UNAVAILABLE = 4
    EXECUTION_FAILED = 5
    FALLBACK = 6


class NativeFallbackReason(IntEnum):
    NONE = 0
    BAD_DTYPE = 1
    BAD_SHAPE = 2
    BAD_STRIDE = 3
    BAD_VERSION = 4
    NULL_POINTER = 5
    UNSUPPORTED_CAPABILITY = 6
    BACKEND_UNAVAILABLE = 7
    EXECUTION_FAILED = 8


class NativeDType(IntEnum):
    UINT8 = 1
    BOOL8 = 2


IMGKEY_GPU_OK = int(NativeStatus.OK)
IMGKEY_GPU_INVALID_ARGUMENT = int(NativeStatus.INVALID_ARGUMENT)
IMGKEY_GPU_UNSUPPORTED_VERSION = int(NativeStatus.UNSUPPORTED_VERSION)
IMGKEY_GPU_UNSUPPORTED_CAPABILITY = int(NativeStatus.UNSUPPORTED_CAPABILITY)
IMGKEY_GPU_BACKEND_UNAVAILABLE = int(NativeStatus.BACKEND_UNAVAILABLE)
IMGKEY_GPU_EXECUTION_FAILED = int(NativeStatus.EXECUTION_FAILED)
IMGKEY_GPU_FALLBACK = int(NativeStatus.FALLBACK)

_ERROR_REASONS = {
    "cuda_dll_unavailable",
    "cuda_dll_probe_failed",
    "cuda_no_device",
    "cuda_unavailable",
    "cuda_execution_failed",
    "backend_unavailable",
    "backend_probe_failed",
    "backend_execution_failed",
    "d3d12_dll_unavailable",
    "d3d12_unavailable",
    "d3d12_context_failed",
    "d3d12_execution_failed",
    "tile_too_large",
    "gpu_exception",
}
GPU_BACKEND_ERROR_REASONS = frozenset(_ERROR_REASONS)

_THREAD_STATE = threading.local()


class NativeAbiError(ValueError):
    def __init__(self, reason: NativeFallbackReason, message: str):
        super().__init__(message)
        self.reason = reason


class ImgKeyNativeTileBufferV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("row_stride_bytes", ctypes.c_int64),
        ("pixel_stride_bytes", ctypes.c_int64),
        ("byte_size", ctypes.c_uint64),
    ]


class ImgKeyNativeColorTileParamsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("required_capabilities", ctypes.c_uint64),
        ("status", ctypes.c_int32),
        ("fallback_reason", ctypes.c_int32),
        ("screen_r", ctypes.c_uint8),
        ("screen_g", ctypes.c_uint8),
        ("screen_b", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("foreground_reference_pull", ctypes.c_float),
        ("key_vector_despill", ctypes.c_float),
        ("preserve_foreground_luma", ctypes.c_float),
        ("transition_spill_threshold", ctypes.c_float),
        ("transition_reconstruction_error", ctypes.c_float),
        ("clip_foreground", ctypes.c_float),
        ("transition_alpha_min", ctypes.c_uint32),
        ("transition_alpha_max", ctypes.c_uint32),
    ]


class ImgKeyNativeColorTileParamsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("required_capabilities", ctypes.c_uint64),
        ("status", ctypes.c_int32),
        ("fallback_reason", ctypes.c_int32),
        ("screen_r", ctypes.c_uint8),
        ("screen_g", ctypes.c_uint8),
        ("screen_b", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("foreground_reference_pull", ctypes.c_float),
        ("key_vector_despill", ctypes.c_float),
        ("preserve_foreground_luma", ctypes.c_float),
        ("transition_spill_threshold", ctypes.c_float),
        ("transition_reconstruction_error", ctypes.c_float),
        ("clip_foreground", ctypes.c_float),
        ("transition_alpha_min", ctypes.c_uint32),
        ("transition_alpha_max", ctypes.c_uint32),
        ("despill", ctypes.c_float),
        ("decontaminate", ctypes.c_float),
        ("unmix_amount", ctypes.c_float),
        ("edge_color_repair", ctypes.c_float),
        ("inner_color_pull", ctypes.c_float),
        ("fringe_remove", ctypes.c_float),
        ("luminance_protect", ctypes.c_float),
        ("clamp_key_r", ctypes.c_float),
        ("clamp_key_g", ctypes.c_float),
        ("clamp_key_b", ctypes.c_float),
        ("transition_enabled", ctypes.c_uint32),
        ("transition_reference_enabled", ctypes.c_uint32),
    ]


@dataclass(slots=True)
class NativeColorTileCallV1:
    params: ImgKeyNativeColorTileParamsV1
    buffers: dict[str, ImgKeyNativeTileBufferV1]


@dataclass(slots=True)
class NativeColorTileCallV2:
    params: ImgKeyNativeColorTileParamsV2
    buffers: dict[str, ImgKeyNativeTileBufferV1]


def _set_last_error(message: str) -> None:
    _THREAD_STATE.last_error = str(message)


def native_last_error() -> str:
    return str(getattr(_THREAD_STATE, "last_error", ""))


def capabilities_to_mask(capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None) -> BackendCapability:
    if capabilities is None:
        return BackendCapability(0)
    if isinstance(capabilities, BackendCapability):
        return capabilities
    if isinstance(capabilities, int):
        return BackendCapability(capabilities)
    mask = BackendCapability(0)
    for raw in capabilities:
        name = str(raw).strip().lower()
        if name:
            mask |= CAPABILITY_BY_NAME[name]
    return mask


def capability_names(capabilities: BackendCapability | int) -> list[str]:
    mask = BackendCapability(int(capabilities))
    return [name for capability, name in CAPABILITY_NAMES.items() if capability & mask]


def _normalize_mode(mode: Any) -> str:
    raw = str(mode or "Off").strip().lower().replace("_", " ")
    if raw in {"auto", "automatic"}:
        return "Auto"
    if raw in {"force", "force gpu", "forced", "on"}:
        return "Force GPU"
    return "Off"


def _setting(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _clip01(value: Any) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _fallback_result(
    reason: str,
    message: str,
    *,
    backend: str | None = None,
    backend_name: str | None = None,
    elapsed_ms: float | None = None,
    availability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "used": False,
        "backend": backend,
        "backend_name": backend_name,
        "reason": reason,
        "message": message,
        "rgb": None,
        "repair_mask": None,
        "elapsed_ms": elapsed_ms,
    }
    if availability is not None:
        result["availability"] = availability
    return result


class GpuBackendSession(Protocol):
    backend_id: str | None
    backend_name: str | None

    def process_color_tile(
        self,
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
    ) -> dict[str, Any]: ...

    def process_full_color_tile(
        self,
        rgb_tile: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
    ) -> dict[str, Any]: ...

    def end_render(self) -> None: ...


class GpuBackend(Protocol):
    backend_id: str
    backend_name: str
    capabilities: BackendCapability

    def probe(self, *, refresh: bool = False) -> dict[str, Any]: ...

    def begin_render(self, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False) -> GpuBackendSession: ...


@dataclass(slots=True)
class BackendSelection:
    mode: str
    status: str
    backend: GpuBackend | None
    backend_info: dict[str, Any] | None
    required_capabilities: BackendCapability
    reason: str | None
    message: str

    @property
    def available(self) -> bool:
        return self.backend is not None and self.status == "selected"

    def as_dict(self) -> dict[str, Any]:
        backend = self.backend_info or {}
        return {
            "mode": self.mode,
            "status": self.status,
            "available": self.available,
            "backend": backend.get("id") if backend else None,
            "backend_name": backend.get("name") if backend else None,
            "reason": self.reason,
            "message": self.message,
            "required_capabilities": capability_names(self.required_capabilities),
            "capabilities": backend.get("capabilities", []) if backend else [],
        }


class NoOpGpuSession:
    def __init__(self, selection: BackendSelection):
        self.selection = selection
        info = selection.backend_info or {}
        self.backend_id = info.get("id")
        self.backend_name = info.get("name")

    def process_color_tile(
        self,
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
    ) -> dict[str, Any]:
        del rgb_tile, alpha_tile, background_mask, edge_mask, probability, fringe_mask, screen_tile, nearest_fg_rgb, nearest_fg_valid, screen_color, settings
        return _fallback_result(
            self.selection.reason or "backend_unavailable",
            self.selection.message,
            backend=self.backend_id,
            backend_name=self.backend_name,
        )

    def process_full_color_tile(
        self,
        rgb_tile: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del rgb_tile, alpha_tile, background_mask, edge_mask, probability, fringe_mask, screen_tile, nearest_inner_rgb, nearest_inner_valid, screen_color, settings, transition_nearest_rgb, transition_nearest_valid
        return _fallback_result(
            self.selection.reason or "backend_unavailable",
            self.selection.message,
            backend=self.backend_id,
            backend_name=self.backend_name,
        )

    def end_render(self) -> None:
        return None


class CudaCompatBackend:
    backend_id = "cuda_compat"
    backend_name = "CUDA compatibility backend"
    capabilities = BackendCapability.CONSTANT_SCREEN | BackendCapability.RGB_ONLY

    def __init__(self, cuda_probe: Callable[..., dict[str, Any]] | None = None):
        self._cuda_probe = cuda_probe
        self._last_availability: dict[str, Any] | None = None

    def probe(self, *, refresh: bool = False) -> dict[str, Any]:
        try:
            if self._cuda_probe is not None:
                try:
                    availability = self._cuda_probe(dll_path=None)
                except TypeError:
                    availability = self._cuda_probe()
            else:
                import gpu_accel

                availability = gpu_accel.is_available(refresh=refresh)
        except Exception as exc:
            availability = {
                "available": False,
                "status": "unavailable",
                "reason": "backend_probe_failed",
                "message": f"CUDA compatibility backend probe failed: {type(exc).__name__}: {exc}. CPU color path will be used.",
                "probe_error": f"{type(exc).__name__}: {exc}",
            }
        self._last_availability = dict(availability)
        return {
            "id": self.backend_id,
            "name": self.backend_name,
            "api_version": IMGKEY_GPU_BACKEND_API_VERSION,
            "status": availability.get("status") or ("available" if availability.get("available") else "unavailable"),
            "available": bool(availability.get("available")),
            "reason": availability.get("reason"),
            "message": availability.get("message"),
            "capability_mask": int(self.capabilities),
            "capabilities": capability_names(self.capabilities),
            "device": availability.get("device"),
            "device_index": availability.get("device_index"),
            "device_count": int(availability.get("device_count") or 0),
            "version": availability.get("version"),
            "dll_path": availability.get("dll_path"),
            "legacy_backend": {
                "id": availability.get("backend") or "compact_cuda_dll",
                "name": availability.get("backend_name") or "compact CUDA DLL",
            },
            "availability": availability,
        }

    def begin_render(self, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False) -> "CudaCompatSession":
        return CudaCompatSession(self, settings, image_shape, force_gpu=force_gpu)


class CudaCompatSession:
    def __init__(self, backend: CudaCompatBackend, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False):
        self.backend = backend
        self.backend_id = backend.backend_id
        self.backend_name = backend.backend_name
        self.settings = settings
        self.image_shape = tuple(int(v) for v in image_shape[:2])
        self.force_gpu = bool(force_gpu)
        self.started_at = time.perf_counter()
        self.ended = False

    def process_color_tile(
        self,
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
    ) -> dict[str, Any]:
        try:
            import gpu_accel

            result = gpu_accel.process_color_tile_gpu(
                rgb_tile,
                alpha_tile,
                background_mask,
                edge_mask,
                probability,
                fringe_mask,
                screen_tile,
                nearest_fg_rgb,
                nearest_fg_valid,
                screen_color,
                settings,
                force_gpu=self.force_gpu,
            )
        except Exception as exc:  # pragma: no cover - defensive backend boundary
            return _fallback_result(
                "gpu_exception",
                f"GPU transition repair failed before launch; CPU fallback is required: {type(exc).__name__}: {exc}",
                backend=self.backend_id,
                backend_name=self.backend_name,
            )
        result.setdefault("selected_backend", self.backend_id)
        result.setdefault("selected_backend_name", self.backend_name)
        result.setdefault("capabilities", capability_names(self.backend.capabilities))
        return result

    def process_full_color_tile(
        self,
        rgb_tile: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del rgb_tile, alpha_tile, background_mask, edge_mask, probability, fringe_mask, screen_tile, nearest_inner_rgb, nearest_inner_valid, screen_color, settings, transition_nearest_rgb, transition_nearest_valid
        return _fallback_result(
            "unsupported_capability",
            "CUDA compatibility backend exposes transition repair only; full color tile graph falls back to CPU or D3D12.",
            backend=self.backend_id,
            backend_name=self.backend_name,
        )

    def end_render(self) -> None:
        self.ended = True


NATIVE_GPU_DLL_NAME = "imgkey_gpu.dll"
D3D12_MVP_MAX_TILE_PIXELS = 3072 * 3072
D3D12_MAX_DISPATCH_PIXELS = 512 * 512
D3D12_NATIVE_CALL_MAX_TILE_PIXELS = 512 * 512
_NATIVE_GPU_DLL_CACHE: "_NativeGpuDll" | None = None
_NATIVE_GPU_DLL_CACHE_KEY: str | None = None


class NativeGpuDllUnavailable(RuntimeError):
    pass


class NativeGpuDllError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"imgkey_gpu status {status}: {message}")
        self.status = int(status)
        self.message = message


class _NativeGpuDll:
    def __init__(self, path: Path, library: ctypes.CDLL):
        self.path = path
        self.library = library
        self.library.imgkey_gpu_version.argtypes = []
        self.library.imgkey_gpu_version.restype = ctypes.c_uint32
        self.library.imgkey_gpu_last_error.argtypes = []
        self.library.imgkey_gpu_last_error.restype = ctypes.c_char_p
        self.library.imgkey_gpu_probe_v1.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.library.imgkey_gpu_probe_v1.restype = ctypes.c_int
        self.library.imgkey_gpu_create_context_v1.argtypes = [ctypes.POINTER(ImgKeyNativeColorTileParamsV1), ctypes.POINTER(ctypes.c_void_p)]
        self.library.imgkey_gpu_create_context_v1.restype = ctypes.c_int
        self.library.imgkey_gpu_destroy_context_v1.argtypes = [ctypes.c_void_p]
        self.library.imgkey_gpu_destroy_context_v1.restype = None
        self.library.imgkey_gpu_process_color_tile_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ImgKeyNativeColorTileParamsV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ctypes.POINTER(ImgKeyNativeTileBufferV1),
        ]
        self.library.imgkey_gpu_process_color_tile_v1.restype = ctypes.c_int
        if hasattr(self.library, "imgkey_gpu_process_color_tile_v2"):
            self.library.imgkey_gpu_process_color_tile_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ImgKeyNativeColorTileParamsV2),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ]
            self.library.imgkey_gpu_process_color_tile_v2.restype = ctypes.c_int
        if hasattr(self.library, "imgkey_gpu_identity_rgba_v1"):
            self.library.imgkey_gpu_identity_rgba_v1.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
                ctypes.POINTER(ImgKeyNativeTileBufferV1),
            ]
            self.library.imgkey_gpu_identity_rgba_v1.restype = ctypes.c_int

    def last_error(self) -> str:
        raw = self.library.imgkey_gpu_last_error()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    def version(self) -> int:
        return int(self.library.imgkey_gpu_version())

    def probe(self) -> dict[str, Any]:
        buffer = ctypes.create_string_buffer(16384)
        status = int(self.library.imgkey_gpu_probe_v1(buffer, ctypes.sizeof(buffer)))
        if status != IMGKEY_GPU_OK:
            raise NativeGpuDllError(status, self.last_error())
        raw = buffer.value.decode("utf-8", errors="replace")
        return json.loads(raw)

    def create_context(self) -> ctypes.c_void_p:
        context = ctypes.c_void_p()
        status = int(self.library.imgkey_gpu_create_context_v1(None, ctypes.byref(context)))
        if status != IMGKEY_GPU_OK or not context.value:
            raise NativeGpuDllError(status, self.last_error() or "D3D12 context creation returned null")
        return context

    def destroy_context(self, context: ctypes.c_void_p | None) -> None:
        if context is not None and context.value:
            self.library.imgkey_gpu_destroy_context_v1(context)

    def process_color_tile(self, context: ctypes.c_void_p, call: NativeColorTileCallV1, out_rgb: ImgKeyNativeTileBufferV1, out_repair: ImgKeyNativeTileBufferV1) -> int:
        def _ptr(buffer: ImgKeyNativeTileBufferV1 | None):
            return ctypes.byref(buffer) if buffer is not None else None

        return int(
            self.library.imgkey_gpu_process_color_tile_v1(
                context,
                ctypes.byref(call.params),
                ctypes.byref(call.buffers["rgb"]),
                ctypes.byref(call.buffers["alpha"]),
                ctypes.byref(call.buffers["background_mask"]),
                ctypes.byref(call.buffers["edge_mask"]),
                ctypes.byref(call.buffers["probability"]),
                ctypes.byref(call.buffers["fringe_mask"]),
                _ptr(call.buffers.get("screen_tile")),
                ctypes.byref(call.buffers["foreground_ref_rgb"]),
                ctypes.byref(call.buffers["foreground_ref_valid"]),
                ctypes.byref(out_rgb),
                ctypes.byref(out_repair),
            )
        )

    def process_color_tile_v2(self, context: ctypes.c_void_p, call: NativeColorTileCallV2, out_rgb: ImgKeyNativeTileBufferV1, out_repair: ImgKeyNativeTileBufferV1) -> int:
        if not hasattr(self.library, "imgkey_gpu_process_color_tile_v2"):
            raise NativeGpuDllError(IMGKEY_GPU_UNSUPPORTED_CAPABILITY, "imgkey_gpu_process_color_tile_v2 export is missing")

        def _ptr(buffer: ImgKeyNativeTileBufferV1 | None):
            return ctypes.byref(buffer) if buffer is not None else None

        return int(
            self.library.imgkey_gpu_process_color_tile_v2(
                context,
                ctypes.byref(call.params),
                ctypes.byref(call.buffers["rgb"]),
                ctypes.byref(call.buffers["alpha"]),
                ctypes.byref(call.buffers["background_mask"]),
                ctypes.byref(call.buffers["edge_mask"]),
                ctypes.byref(call.buffers["probability"]),
                ctypes.byref(call.buffers["fringe_mask"]),
                _ptr(call.buffers.get("screen_tile")),
                ctypes.byref(call.buffers["nearest_inner_rgb"]),
                ctypes.byref(call.buffers["nearest_inner_valid"]),
                ctypes.byref(call.buffers["transition_ref_rgb"]),
                ctypes.byref(call.buffers["transition_ref_valid"]),
                ctypes.byref(out_rgb),
                ctypes.byref(out_repair),
            )
        )

    def identity_rgba(self, context: ctypes.c_void_p, rgba: ImgKeyNativeTileBufferV1, out_rgba: ImgKeyNativeTileBufferV1) -> int:
        if not hasattr(self.library, "imgkey_gpu_identity_rgba_v1"):
            raise NativeGpuDllError(IMGKEY_GPU_UNSUPPORTED_CAPABILITY, "imgkey_gpu_identity_rgba_v1 export is missing")
        return int(self.library.imgkey_gpu_identity_rgba_v1(context, ctypes.byref(rgba), ctypes.byref(out_rgba)))


def _candidate_native_gpu_dll_paths(dll_path: str | os.PathLike[str] | None = None) -> list[Path]:
    if dll_path is not None:
        return [Path(dll_path).expanduser()]
    paths: list[Path] = []
    env_path = os.environ.get("IMGKEY_GPU_DLL")
    if env_path:
        paths.append(Path(env_path))
    module_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            module_dir / NATIVE_GPU_DLL_NAME,
            module_dir / "native" / "imgkey_gpu" / "build" / NATIVE_GPU_DLL_NAME,
            Path.cwd() / "native" / "imgkey_gpu" / "build" / NATIVE_GPU_DLL_NAME,
        ]
    )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        paths.append(Path(str(meipass)) / NATIVE_GPU_DLL_NAME)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path.expanduser())
    return unique


def _load_native_gpu_dll(dll_path: str | os.PathLike[str] | None = None, *, refresh: bool = False) -> _NativeGpuDll:
    global _NATIVE_GPU_DLL_CACHE, _NATIVE_GPU_DLL_CACHE_KEY
    candidates = _candidate_native_gpu_dll_paths(dll_path)
    cache_key = str(candidates[0]) if dll_path is not None else "<default>"
    if not refresh and _NATIVE_GPU_DLL_CACHE is not None and _NATIVE_GPU_DLL_CACHE_KEY == cache_key:
        return _NATIVE_GPU_DLL_CACHE
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
        dll = _NativeGpuDll(resolved, library)
        if dll_path is None:
            _NATIVE_GPU_DLL_CACHE = dll
            _NATIVE_GPU_DLL_CACHE_KEY = cache_key
        return dll
    detail = "; ".join(load_errors) if load_errors else "checked " + ", ".join(checked)
    raise NativeGpuDllUnavailable(f"{NATIVE_GPU_DLL_NAME} was not found or could not be loaded ({detail})")


def _status_native_unavailable(reason: str, message: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "d3d12_compute",
        "name": "D3D12 compute backend",
        "api_version": IMGKEY_GPU_BACKEND_API_VERSION,
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "message": message,
        "capability_mask": int(D3D12ComputeBackend.capabilities),
        "capabilities": capability_names(D3D12ComputeBackend.capabilities),
        "device": None,
        "device_index": None,
        "device_count": 0,
        "version": None,
        "dll_path": None,
        "max_tile_pixels": D3D12_MVP_MAX_TILE_PIXELS,
        "max_dispatch_pixels": D3D12_MAX_DISPATCH_PIXELS,
        "max_native_call_pixels": D3D12_NATIVE_CALL_MAX_TILE_PIXELS,
    }
    if extra:
        result.update(extra)
    return result


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
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{name} must match tile shape")
    if arr.dtype == np.bool_:
        return np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


_SRGB_U8_TO_LINEAR_LUT = np.where(
    (np.arange(256, dtype=np.float32) / 255.0) <= 0.04045,
    (np.arange(256, dtype=np.float32) / 255.0) / 12.92,
    np.power(((np.arange(256, dtype=np.float32) / 255.0) + 0.055) / 1.055, 2.4),
).astype(np.float32)


def _screen_clamp_key_linear(screen_tile: np.ndarray | None, screen_color: tuple[int, int, int]) -> tuple[float, float, float]:
    if screen_tile is None:
        values = _SRGB_U8_TO_LINEAR_LUT[np.asarray(screen_color, dtype=np.uint8)]
        return (float(values[0]), float(values[1]), float(values[2]))
    screen = np.asarray(screen_tile, dtype=np.uint8)
    if screen.ndim != 3 or screen.shape[2] < 3 or screen.size == 0:
        values = _SRGB_U8_TO_LINEAR_LUT[np.asarray(screen_color, dtype=np.uint8)]
        return (float(values[0]), float(values[1]), float(values[2]))
    return (
        float(np.mean(_SRGB_U8_TO_LINEAR_LUT[screen[:, :, 0]])),
        float(np.mean(_SRGB_U8_TO_LINEAR_LUT[screen[:, :, 1]])),
        float(np.mean(_SRGB_U8_TO_LINEAR_LUT[screen[:, :, 2]])),
    )


def _foreground_reference_radius(settings: Any) -> int:
    return int(np.clip(int(_setting(settings, "foreground_reference_radius", 96)), 0, np.iinfo(np.uint16).max - 1))


def _effective_luminance_protect(settings: Any) -> float:
    value = _setting(settings, "luminance_protect", None)
    if value is None:
        value = _setting(settings, "luminance_restore", 0.35)
    return _clip01(value)


class D3D12ComputeBackend:
    backend_id = "d3d12_compute"
    backend_name = "D3D12 compute backend"
    capabilities = BackendCapability.CONSTANT_SCREEN | BackendCapability.SCREEN_TILE | BackendCapability.PERSISTENT_SESSION | BackendCapability.RGB_ONLY | BackendCapability.FULL_COLOR_TILE

    def __init__(self, dll_path: str | os.PathLike[str] | None = None):
        self.dll_path = dll_path
        self._last_availability: dict[str, Any] | None = None

    def probe(self, *, refresh: bool = False) -> dict[str, Any]:
        try:
            dll = _load_native_gpu_dll(self.dll_path, refresh=refresh)
        except Exception as exc:
            result = _status_native_unavailable(
                "d3d12_dll_unavailable",
                f"D3D12 native backend is unavailable: {type(exc).__name__}: {exc}. CPU fallback will be used.",
                extra={"load_error": f"{type(exc).__name__}: {exc}"},
            )
            self._last_availability = dict(result)
            return result
        try:
            info = dll.probe()
        except Exception as exc:
            result = _status_native_unavailable(
                "d3d12_unavailable",
                f"D3D12 native backend probe failed: {type(exc).__name__}: {exc}. CPU fallback will be used.",
                extra={"dll_path": str(dll.path), "probe_error": f"{type(exc).__name__}: {exc}", "last_error": dll.last_error()},
            )
            self._last_availability = dict(result)
            return result
        info.setdefault("id", self.backend_id)
        info.setdefault("name", self.backend_name)
        info.setdefault("api_version", IMGKEY_GPU_BACKEND_API_VERSION)
        info.setdefault("capability_mask", int(self.capabilities))
        info.setdefault("capabilities", capability_names(self.capabilities))
        info.setdefault("available", bool(info.get("status") == "available"))
        info.setdefault("status", "available" if info.get("available") else "unavailable")
        info.setdefault("reason", None if info.get("available") else "d3d12_unavailable")
        info.setdefault("message", "D3D12 compute backend available." if info.get("available") else "D3D12 compute backend unavailable.")
        info["dll_path"] = str(dll.path)
        info["version"] = info.get("version") or dll.version()
        info["max_tile_pixels"] = int(info.get("max_tile_pixels") or D3D12_MVP_MAX_TILE_PIXELS)
        info["max_dispatch_pixels"] = int(info.get("max_dispatch_pixels") or D3D12_MAX_DISPATCH_PIXELS)
        info["max_native_call_pixels"] = int(info.get("max_native_call_pixels") or D3D12_NATIVE_CALL_MAX_TILE_PIXELS)
        self._last_availability = dict(info)
        return info

    def begin_render(self, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False) -> "D3D12ComputeSession":
        return D3D12ComputeSession(self, settings, image_shape, force_gpu=force_gpu)

    def identity_rgba(self, rgba: np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            dll = _load_native_gpu_dll(self.dll_path)
            context = dll.create_context()
        except Exception as exc:
            return _fallback_result("d3d12_context_failed", f"D3D12 identity context creation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)
        try:
            arr = np.asarray(rgba)
            if arr.ndim != 3 or arr.shape[2] != 4 or arr.dtype != np.uint8:
                raise ValueError("rgba must be a uint8 HxWx4 array")
            arr = np.ascontiguousarray(arr)
            out = np.empty_like(arr)
            in_buffer = native_buffer_from_array("rgba", arr, expected_channels=4)
            out_buffer = native_buffer_from_array("out_rgba", out, expected_channels=4)
            status = dll.identity_rgba(context, in_buffer, out_buffer)
            if status != IMGKEY_GPU_OK:
                raise NativeGpuDllError(status, dll.last_error())
            return {
                "ok": True,
                "used": True,
                "backend": self.backend_id,
                "backend_name": self.backend_name,
                "reason": None,
                "message": "D3D12 identity RGBA kernel completed.",
                "rgba": out,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "capabilities": capability_names(self.capabilities),
            }
        except Exception as exc:
            return _fallback_result("d3d12_execution_failed", f"D3D12 identity kernel failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0)
        finally:
            dll.destroy_context(context)


class D3D12ComputeSession:
    def __init__(self, backend: D3D12ComputeBackend, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False):
        self.backend = backend
        self.backend_id = backend.backend_id
        self.backend_name = backend.backend_name
        self.settings = settings
        self.image_shape = tuple(int(v) for v in image_shape[:2])
        self.force_gpu = bool(force_gpu)
        self.started_at = time.perf_counter()
        self.ended = False
        self.dll: _NativeGpuDll | None = None
        self.context: ctypes.c_void_p | None = None
        self.context_error: str | None = None
        try:
            self.dll = _load_native_gpu_dll(backend.dll_path)
            self.context = self.dll.create_context()
        except Exception as exc:
            self.context_error = f"{type(exc).__name__}: {exc}"

    def process_color_tile(
        self,
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
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if self.dll is None or self.context is None or not self.context.value:
            return _fallback_result("d3d12_context_failed", f"D3D12 context is unavailable: {self.context_error or 'context not created'}", backend=self.backend_id, backend_name=self.backend_name)
        if not bool(_setting(settings, "transition_unmix", True)):
            return _fallback_result("transition_disabled", "Transition unmix is disabled; CPU color path remains active.", backend=self.backend_id, backend_name=self.backend_name)
        if int(np.clip(int(_setting(settings, "foreground_reference_radius", 96)), 0, np.iinfo(np.uint16).max - 1)) <= 0:
            return _fallback_result("reference_radius_disabled", "Foreground reference radius is disabled; CPU transition repair leaves the tile unchanged.", backend=self.backend_id, backend_name=self.backend_name)
        if nearest_fg_rgb is None or nearest_fg_valid is None:
            return _fallback_result("no_foreground_reference", "No foreground reference tile is available for D3D12 transition repair.", backend=self.backend_id, backend_name=self.backend_name)
        try:
            rgb = _as_rgb_u8(rgb_tile, "rgb_tile")
            shape = rgb.shape[:2]
            if int(shape[0]) * int(shape[1]) > D3D12_NATIVE_CALL_MAX_TILE_PIXELS:
                return _fallback_result(
                    "tile_too_large",
                    f"D3D12 transition-only native call {shape[1]}x{shape[0]} exceeds safe native_call_pixels={D3D12_NATIVE_CALL_MAX_TILE_PIXELS}; full color tile processing uses TDR-bounded subtiles and CPU remains fallback for this legacy entry point.",
                    backend=self.backend_id,
                    backend_name=self.backend_name,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    availability=self.backend._last_availability,
                )
            max_tile_pixels = int((self.backend._last_availability or {}).get("max_tile_pixels") or D3D12_MVP_MAX_TILE_PIXELS)
            if int(shape[0]) * int(shape[1]) > max_tile_pixels:
                return _fallback_result(
                    "tile_too_large",
                    f"D3D12 MVP tile {shape[1]}x{shape[0]} exceeds max_tile_pixels={max_tile_pixels}; CPU color path is used to avoid TDR.",
                    backend=self.backend_id,
                    backend_name=self.backend_name,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    availability=self.backend._last_availability,
                )
            alpha = _as_u8_mask(alpha_tile, shape, "alpha_tile")
            background = _as_u8_mask(background_mask, shape, "background_mask")
            edge = _as_u8_mask(edge_mask, shape, "edge_mask")
            probability_u8 = _as_u8_mask(probability, shape, "probability")
            fringe_u8 = _as_u8_mask(fringe_mask, shape, "fringe_mask")
            foreground_rgb = _as_rgb_u8(nearest_fg_rgb, "nearest_fg_rgb")
            foreground_valid = _as_u8_mask(nearest_fg_valid, shape, "nearest_fg_valid")
            screen_u8 = _as_rgb_u8(screen_tile, "screen_tile") if screen_tile is not None else None
            out_rgb = np.empty_like(rgb)
            out_repair = np.empty(shape, dtype=np.uint8)
            required = {"rgb_only", "screen_tile"} if screen_u8 is not None else {"rgb_only", "constant_screen"}
            call = validate_native_color_tile_inputs(
                rgb,
                alpha,
                background,
                edge,
                probability_u8,
                fringe_u8,
                screen_u8,
                foreground_rgb,
                foreground_valid,
                tuple(int(np.clip(c, 0, 255)) for c in screen_color),
                settings,
                required_capabilities=required,
            )
            out_rgb_buffer = native_buffer_from_array("out_rgb", out_rgb, expected_channels=3)
            out_repair_buffer = native_buffer_from_array("out_repair_mask", out_repair, expected_channels=1)
        except Exception as exc:
            return _fallback_result("invalid_inputs", f"D3D12 transition repair input validation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)
        if int(np.max(alpha)) <= 0:
            return _fallback_result("transparent_tile", "Tile alpha is fully transparent; CPU zero-RGB invariant remains active.", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0)
        if not np.any(foreground_valid):
            return _fallback_result("no_foreground_reference", "Foreground reference mask is empty for this tile.", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0)
        try:
            status = self.dll.process_color_tile(self.context, call, out_rgb_buffer, out_repair_buffer)
            if status != IMGKEY_GPU_OK:
                reason = "d3d12_execution_failed"
                if int(call.params.fallback_reason) == int(NativeFallbackReason.UNSUPPORTED_CAPABILITY):
                    reason = "unsupported_capability"
                return _fallback_result(reason, f"D3D12 transition repair failed: status={status} {self.dll.last_error()}", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0, availability=self.backend._last_availability)
        except Exception as exc:
            return _fallback_result("d3d12_execution_failed", f"D3D12 transition repair failed; CPU fallback is required: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0, availability=self.backend._last_availability)
        mode = "forced" if self.force_gpu else "auto"
        return {
            "ok": True,
            "used": True,
            "backend": self.backend_id,
            "backend_name": self.backend_name,
            "reason": None,
            "message": f"D3D12 transition repair completed ({mode}).",
            "rgb": out_rgb,
            "repair_mask": out_repair,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "availability": self.backend._last_availability,
            "capabilities": capability_names(self.backend.capabilities),
        }

    def process_full_color_tile(
        self,
        rgb_tile: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if self.dll is None or self.context is None or not self.context.value:
            return _fallback_result("d3d12_context_failed", f"D3D12 context is unavailable: {self.context_error or 'context not created'}", backend=self.backend_id, backend_name=self.backend_name)
        try:
            rgb = _as_rgb_u8(rgb_tile, "rgb_tile")
            shape = rgb.shape[:2]
            if int(shape[0]) * int(shape[1]) > D3D12_NATIVE_CALL_MAX_TILE_PIXELS:
                return self._process_full_color_tile_split(
                    rgb,
                    alpha_tile,
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
                    start_time=start,
                )
            max_tile_pixels = int((self.backend._last_availability or {}).get("max_tile_pixels") or D3D12_MVP_MAX_TILE_PIXELS)
            if int(shape[0]) * int(shape[1]) > max_tile_pixels:
                return _fallback_result(
                    "tile_too_large",
                    f"D3D12 full color tile {shape[1]}x{shape[0]} exceeds max_tile_pixels={max_tile_pixels}; CPU color path is used to avoid TDR.",
                    backend=self.backend_id,
                    backend_name=self.backend_name,
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    availability=self.backend._last_availability,
                )
            alpha = _as_u8_mask(alpha_tile, shape, "alpha_tile")
            background = _as_u8_mask(background_mask, shape, "background_mask")
            edge = _as_u8_mask(edge_mask, shape, "edge_mask")
            probability_u8 = _as_u8_mask(probability, shape, "probability")
            fringe_u8 = _as_u8_mask(fringe_mask, shape, "fringe_mask")
            screen_u8 = _as_rgb_u8(screen_tile, "screen_tile") if screen_tile is not None else None
            zero_rgb = np.zeros_like(rgb)
            zero_valid = np.zeros(shape, dtype=np.uint8)
            nearest_rgb = _as_rgb_u8(nearest_inner_rgb, "nearest_inner_rgb") if nearest_inner_rgb is not None else zero_rgb
            nearest_valid = _as_u8_mask(nearest_inner_valid, shape, "nearest_inner_valid") if nearest_inner_valid is not None else zero_valid
            transition_rgb = _as_rgb_u8(transition_nearest_rgb, "transition_nearest_rgb") if transition_nearest_rgb is not None else nearest_rgb
            transition_valid = _as_u8_mask(transition_nearest_valid, shape, "transition_nearest_valid") if transition_nearest_valid is not None else nearest_valid
            out_rgb = np.empty_like(rgb)
            out_repair = np.empty(shape, dtype=np.uint8)
            required = {"rgb_only", "full_color_tile", "screen_tile"} if screen_u8 is not None else {"rgb_only", "full_color_tile", "constant_screen"}
            call = validate_native_full_color_tile_inputs(
                rgb,
                alpha,
                background,
                edge,
                probability_u8,
                fringe_u8,
                screen_u8,
                nearest_rgb,
                nearest_valid,
                transition_rgb,
                transition_valid,
                tuple(int(np.clip(c, 0, 255)) for c in screen_color),
                settings,
                clamp_key_linear=_screen_clamp_key_linear(screen_u8, tuple(int(np.clip(c, 0, 255)) for c in screen_color)),
                required_capabilities=required,
            )
            out_rgb_buffer = native_buffer_from_array("out_rgb", out_rgb, expected_channels=3)
            out_repair_buffer = native_buffer_from_array("out_repair_mask", out_repair, expected_channels=1)
        except Exception as exc:
            return _fallback_result("invalid_inputs", f"D3D12 full color tile input validation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)
        if int(np.max(alpha)) <= 0:
            out_rgb = np.zeros_like(rgb)
            out_repair = np.zeros(shape, dtype=np.uint8)
            return {
                "ok": True,
                "used": True,
                "backend": self.backend_id,
                "backend_name": self.backend_name,
                "reason": None,
                "message": "D3D12 full color tile skipped transparent tile on CPU-side invariant.",
                "rgb": out_rgb,
                "repair_mask": out_repair,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "availability": self.backend._last_availability,
                "capabilities": capability_names(self.backend.capabilities),
                "full_color_tile": True,
            }
        try:
            status = self.dll.process_color_tile_v2(self.context, call, out_rgb_buffer, out_repair_buffer)
            if status != IMGKEY_GPU_OK:
                reason = "d3d12_execution_failed"
                if int(call.params.fallback_reason) == int(NativeFallbackReason.UNSUPPORTED_CAPABILITY):
                    reason = "unsupported_capability"
                elif int(call.params.fallback_reason) == int(NativeFallbackReason.BAD_SHAPE):
                    reason = "tile_too_large" if "exceed" in self.dll.last_error().lower() else "invalid_inputs"
                return _fallback_result(reason, f"D3D12 full color tile failed: status={status} {self.dll.last_error()}", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0, availability=self.backend._last_availability)
        except Exception as exc:
            return _fallback_result("d3d12_execution_failed", f"D3D12 full color tile failed; CPU fallback is required: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name, elapsed_ms=(time.perf_counter() - start) * 1000.0, availability=self.backend._last_availability)
        mode = "forced" if self.force_gpu else "auto"
        return {
            "ok": True,
            "used": True,
            "backend": self.backend_id,
            "backend_name": self.backend_name,
            "reason": None,
            "message": f"D3D12 full color tile pipeline completed ({mode}).",
            "rgb": out_rgb,
            "repair_mask": out_repair,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "availability": self.backend._last_availability,
            "capabilities": capability_names(self.backend.capabilities),
            "full_color_tile": True,
        }

    def _process_full_color_tile_split(
        self,
        rgb: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
        start_time: float | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter() if start_time is None else start_time
        shape = rgb.shape[:2]
        try:
            alpha = _as_u8_mask(alpha_tile, shape, "alpha_tile")
            background = _as_u8_mask(background_mask, shape, "background_mask")
            edge = _as_u8_mask(edge_mask, shape, "edge_mask")
            probability_u8 = _as_u8_mask(probability, shape, "probability")
            fringe_u8 = _as_u8_mask(fringe_mask, shape, "fringe_mask")
            screen_u8 = _as_rgb_u8(screen_tile, "screen_tile") if screen_tile is not None else None
            zero_rgb = np.zeros_like(rgb)
            zero_valid = np.zeros(shape, dtype=np.uint8)
            nearest_rgb = _as_rgb_u8(nearest_inner_rgb, "nearest_inner_rgb") if nearest_inner_rgb is not None else zero_rgb
            nearest_valid = _as_u8_mask(nearest_inner_valid, shape, "nearest_inner_valid") if nearest_inner_valid is not None else zero_valid
            transition_rgb = _as_rgb_u8(transition_nearest_rgb, "transition_nearest_rgb") if transition_nearest_rgb is not None else nearest_rgb
            transition_valid = _as_u8_mask(transition_nearest_valid, shape, "transition_nearest_valid") if transition_nearest_valid is not None else nearest_valid
        except Exception as exc:
            return _fallback_result("invalid_inputs", f"D3D12 split full color tile input validation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)

        out_rgb = np.empty_like(rgb)
        out_repair = np.empty(shape, dtype=np.uint8)
        chunk = int(np.sqrt(D3D12_NATIVE_CALL_MAX_TILE_PIXELS))
        chunk = max(1, min(512, chunk))
        used_chunks = 0
        elapsed_reported = 0.0
        for y0 in range(0, shape[0], chunk):
            y1 = min(shape[0], y0 + chunk)
            ys = slice(y0, y1)
            for x0 in range(0, shape[1], chunk):
                x1 = min(shape[1], x0 + chunk)
                xs = slice(x0, x1)
                result = self.process_full_color_tile(
                    rgb[ys, xs],
                    alpha[ys, xs],
                    background[ys, xs],
                    edge[ys, xs],
                    probability_u8[ys, xs],
                    fringe_u8[ys, xs],
                    None if screen_u8 is None else screen_u8[ys, xs],
                    nearest_rgb[ys, xs],
                    nearest_valid[ys, xs],
                    screen_color,
                    settings,
                    transition_nearest_rgb=transition_rgb[ys, xs],
                    transition_nearest_valid=transition_valid[ys, xs],
                )
                if not result.get("used") or not isinstance(result.get("rgb"), np.ndarray) or not isinstance(result.get("repair_mask"), np.ndarray):
                    result.setdefault("elapsed_ms", (time.perf_counter() - start) * 1000.0)
                    return result
                out_rgb[ys, xs] = result["rgb"]
                out_repair[ys, xs] = result["repair_mask"]
                used_chunks += 1
                elapsed_reported += float(result.get("elapsed_ms") or 0.0)
        return {
            "ok": True,
            "used": True,
            "backend": self.backend_id,
            "backend_name": self.backend_name,
            "reason": None,
            "message": f"D3D12 full color tile pipeline completed via {used_chunks} TDR-bounded native subtile dispatch(es).",
            "rgb": out_rgb,
            "repair_mask": out_repair,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "native_elapsed_ms_total": elapsed_reported,
            "availability": self.backend._last_availability,
            "capabilities": capability_names(self.backend.capabilities),
            "full_color_tile": True,
            "subtile_dispatches": used_chunks,
            "subtile_max_pixels": D3D12_NATIVE_CALL_MAX_TILE_PIXELS,
        }

    def end_render(self) -> None:
        if not self.ended and self.dll is not None and self.context is not None:
            self.dll.destroy_context(self.context)
        self.ended = True
        self.context = None


class FakeNativeBackend:
    backend_id = "fake_native"
    backend_name = "fake native backend"

    def __init__(
        self,
        *,
        available: bool = True,
        capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
        fallback_reason: str = "fake_unavailable",
    ):
        self.available = bool(available)
        self.capabilities = capabilities_to_mask(capabilities) or (BackendCapability.CONSTANT_SCREEN | BackendCapability.SCREEN_TILE | BackendCapability.RGB_ONLY)
        self.fallback_reason = fallback_reason

    def probe(self, *, refresh: bool = False) -> dict[str, Any]:
        del refresh
        return {
            "id": self.backend_id,
            "name": self.backend_name,
            "api_version": IMGKEY_GPU_BACKEND_API_VERSION,
            "status": "available" if self.available else "unavailable",
            "available": self.available,
            "reason": None if self.available else self.fallback_reason,
            "message": "Fake native backend available for ABI validation." if self.available else "Fake native backend unavailable.",
            "capability_mask": int(self.capabilities),
            "capabilities": capability_names(self.capabilities),
            "device": "CPU fake device",
            "device_index": 0 if self.available else None,
            "device_count": 1 if self.available else 0,
            "version": IMGKEY_GPU_ABI_VERSION,
        }

    def begin_render(self, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False) -> "FakeNativeSession":
        return FakeNativeSession(self, settings, image_shape, force_gpu=force_gpu)


class FakeNativeSession:
    def __init__(self, backend: FakeNativeBackend, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], *, force_gpu: bool = False):
        self.backend = backend
        self.backend_id = backend.backend_id
        self.backend_name = backend.backend_name
        self.settings = settings
        self.image_shape = tuple(int(v) for v in image_shape[:2])
        self.force_gpu = bool(force_gpu)
        self.ended = False

    def process_color_tile(
        self,
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
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.backend.available:
            return _fallback_result(self.backend.fallback_reason, "Fake native backend unavailable.", backend=self.backend_id, backend_name=self.backend_name)
        if screen_tile is not None and not (self.backend.capabilities & BackendCapability.SCREEN_TILE):
            return _fallback_result("unsupported_screen_tile", "Fake native backend does not support screen_tile input.", backend=self.backend_id, backend_name=self.backend_name)
        try:
            validate_native_color_tile_inputs(
                rgb_tile,
                alpha_tile,
                background_mask,
                edge_mask,
                probability,
                fringe_mask,
                screen_tile,
                nearest_fg_rgb,
                nearest_fg_valid,
                screen_color,
                settings,
                required_capabilities=self.backend.capabilities,
            )
        except Exception as exc:
            return _fallback_result("invalid_inputs", f"Fake native ABI validation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)
        rgb = np.asarray(rgb_tile)[:, :, :3].copy()
        alpha = np.asarray(alpha_tile)
        repair_mask = np.zeros(alpha.shape[:2], dtype=np.uint8)
        rgb[alpha <= 0] = 0
        return {
            "ok": True,
            "used": True,
            "backend": self.backend_id,
            "backend_name": self.backend_name,
            "reason": None,
            "message": "Fake native backend validated ABI inputs and returned RGB unchanged.",
            "rgb": rgb,
            "repair_mask": repair_mask,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "capabilities": capability_names(self.backend.capabilities),
        }

    def process_full_color_tile(
        self,
        rgb_tile: np.ndarray,
        alpha_tile: np.ndarray,
        background_mask: np.ndarray,
        edge_mask: np.ndarray,
        probability: np.ndarray,
        fringe_mask: np.ndarray,
        screen_tile: np.ndarray | None,
        nearest_inner_rgb: np.ndarray | None,
        nearest_inner_valid: np.ndarray | None,
        screen_color: tuple[int, int, int],
        settings: Any,
        *,
        transition_nearest_rgb: np.ndarray | None = None,
        transition_nearest_valid: np.ndarray | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if not self.backend.available:
            return _fallback_result(self.backend.fallback_reason, "Fake native backend unavailable.", backend=self.backend_id, backend_name=self.backend_name)
        if not (self.backend.capabilities & BackendCapability.FULL_COLOR_TILE):
            return _fallback_result("unsupported_capability", "Fake native backend does not support full_color_tile.", backend=self.backend_id, backend_name=self.backend_name)
        try:
            validate_native_full_color_tile_inputs(
                rgb_tile,
                alpha_tile,
                background_mask,
                edge_mask,
                probability,
                fringe_mask,
                screen_tile,
                nearest_inner_rgb,
                nearest_inner_valid,
                transition_nearest_rgb,
                transition_nearest_valid,
                screen_color,
                settings,
                clamp_key_linear=(0.0, 0.0, 0.0),
                required_capabilities=self.backend.capabilities,
            )
        except Exception as exc:
            return _fallback_result("invalid_inputs", f"Fake native full ABI validation failed: {type(exc).__name__}: {exc}", backend=self.backend_id, backend_name=self.backend_name)
        rgb = np.asarray(rgb_tile)[:, :, :3].copy()
        alpha = np.asarray(alpha_tile)
        repair_mask = np.zeros(alpha.shape[:2], dtype=np.uint8)
        rgb[alpha <= 0] = 0
        return {
            "ok": True,
            "used": True,
            "backend": self.backend_id,
            "backend_name": self.backend_name,
            "reason": None,
            "message": "Fake native backend validated full color ABI inputs and returned RGB unchanged.",
            "rgb": rgb,
            "repair_mask": repair_mask,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "capabilities": capability_names(self.backend.capabilities),
            "full_color_tile": True,
        }

    def end_render(self) -> None:
        self.ended = True


class FakeNativeCAbi:
    def process_color_tile_v1(self, params: ImgKeyNativeColorTileParamsV1 | None, **buffers: ImgKeyNativeTileBufferV1) -> int:
        try:
            _validate_params_struct(params)
            for name, buffer in buffers.items():
                validate_native_buffer_v1(name, buffer)
        except NativeAbiError as exc:
            _set_last_error(str(exc))
            if params is not None:
                params.status = IMGKEY_GPU_UNSUPPORTED_VERSION if exc.reason == NativeFallbackReason.BAD_VERSION else IMGKEY_GPU_INVALID_ARGUMENT
                params.fallback_reason = int(exc.reason)
            return IMGKEY_GPU_UNSUPPORTED_VERSION if exc.reason == NativeFallbackReason.BAD_VERSION else IMGKEY_GPU_INVALID_ARGUMENT
        except Exception as exc:  # pragma: no cover - defensive fake ABI boundary
            _set_last_error(f"{type(exc).__name__}: {exc}")
            if params is not None:
                params.status = IMGKEY_GPU_EXECUTION_FAILED
                params.fallback_reason = int(NativeFallbackReason.EXECUTION_FAILED)
            return IMGKEY_GPU_EXECUTION_FAILED
        _set_last_error("")
        params.status = IMGKEY_GPU_OK
        params.fallback_reason = int(NativeFallbackReason.NONE)
        return IMGKEY_GPU_OK


def registered_backends() -> list[GpuBackend]:
    return [D3D12ComputeBackend(), CudaCompatBackend()]


def probe_backends(
    *,
    backends: list[GpuBackend] | None = None,
    include_cpu: bool = True,
    refresh: bool = False,
    cuda_probe: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if backends is None:
        backends = [D3D12ComputeBackend(), CudaCompatBackend(cuda_probe=cuda_probe)]
    results: list[dict[str, Any]] = []
    for backend in backends:
        try:
            results.append(backend.probe(refresh=refresh))
        except Exception as exc:  # pragma: no cover - backend probe isolation
            results.append(
                {
                    "id": getattr(backend, "backend_id", "unknown"),
                    "name": getattr(backend, "backend_name", "unknown backend"),
                    "status": "unavailable",
                    "available": False,
                    "reason": "backend_probe_failed",
                    "message": f"Backend probe failed: {type(exc).__name__}: {exc}",
                    "capability_mask": int(getattr(backend, "capabilities", 0)),
                    "capabilities": capability_names(getattr(backend, "capabilities", 0)),
                }
            )
    if include_cpu:
        cpu_caps = BackendCapability.CONSTANT_SCREEN | BackendCapability.SCREEN_TILE | BackendCapability.ALPHA_WRITE | BackendCapability.RGB_ONLY | BackendCapability.FULL_COLOR_TILE
        results.append(
            {
                "id": "cpu_fallback",
                "name": "CPU reference fallback",
                "api_version": IMGKEY_GPU_BACKEND_API_VERSION,
                "status": "available",
                "available": True,
                "reason": None,
                "message": "CPU reference path is always available and remains the correctness fallback.",
                "capability_mask": int(cpu_caps),
                "capabilities": capability_names(cpu_caps),
                "device": "CPU",
                "device_index": None,
                "device_count": 1,
                "version": None,
            }
        )
    return results


def select_backend(
    mode: Any = "Auto",
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
    *,
    backends: list[GpuBackend] | None = None,
    probed_backends: list[dict[str, Any]] | None = None,
    refresh: bool = False,
) -> BackendSelection:
    normalized = _normalize_mode(mode)
    required = capabilities_to_mask(required_capabilities)
    if normalized == "Off":
        return BackendSelection(
            mode=normalized,
            status="off",
            backend=None,
            backend_info=None,
            required_capabilities=required,
            reason="gpu_off",
            message="GPU acceleration is off; CPU color path used.",
        )

    backend_objects = backends if backends is not None else registered_backends()
    probed = probed_backends if probed_backends is not None else probe_backends(backends=backend_objects, include_cpu=False, refresh=refresh)
    objects_by_id = {backend.backend_id: backend for backend in backend_objects}
    rejected: list[dict[str, Any]] = []
    for info in probed:
        if info.get("id") == "cpu_fallback":
            continue
        caps = capabilities_to_mask(int(info.get("capability_mask") or 0))
        if required and (caps & required) != required:
            rejected.append(info)
            continue
        if bool(info.get("available")):
            return BackendSelection(
                mode=normalized,
                status="selected",
                backend=objects_by_id.get(str(info.get("id"))),
                backend_info=info,
                required_capabilities=required,
                reason=None,
                message=str(info.get("message") or f"Selected {info.get('name') or info.get('id')} backend."),
            )
        rejected.append(info)

    first = rejected[0] if rejected else None
    reason = str((first or {}).get("reason") or "backend_unavailable")
    message = str((first or {}).get("message") or "No GPU backend satisfying the required capabilities is available; CPU fallback will be used.")
    if normalized == "Force GPU" and reason not in _ERROR_REASONS:
        reason = "backend_unavailable"
    return BackendSelection(
        mode=normalized,
        status="unavailable",
        backend=None,
        backend_info=first,
        required_capabilities=required,
        reason=reason,
        message=message,
    )


def _estimated_render_tile_pixels(settings: Any, image_shape: tuple[int, int] | tuple[int, int, int]) -> int:
    h = int(image_shape[0]) if len(image_shape) >= 1 else 0
    w = int(image_shape[1]) if len(image_shape) >= 2 else 0
    if h <= 0 or w <= 0:
        return 0
    use_tiling = bool(_setting(settings, "use_tiling", True))
    if not use_tiling:
        return h * w
    tile_size = max(1, int(_setting(settings, "tile_size", 2048)))
    return min(h, tile_size) * min(w, tile_size)


def _should_try_next_backend_for_tile_limit(selection: BackendSelection, settings: Any, image_shape: tuple[int, int] | tuple[int, int, int], required: BackendCapability) -> bool:
    if selection.backend is None or selection.backend.backend_id != "d3d12_compute":
        return False
    # D3D12 is currently the only backend with screen_tile support; keep it for
    # local-screen tiles even when a later per-tile CPU fallback is required.
    if required & BackendCapability.SCREEN_TILE:
        return False
    info = selection.backend_info or {}
    max_tile_pixels = int(info.get("max_tile_pixels") or D3D12_MVP_MAX_TILE_PIXELS)
    tile_pixels = _estimated_render_tile_pixels(settings, image_shape)
    return max_tile_pixels > 0 and tile_pixels > max_tile_pixels


def begin_render(
    settings: Any,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    mode: Any | None = None,
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
    backends: list[GpuBackend] | None = None,
    refresh: bool = False,
) -> GpuBackendSession:
    selected_mode = _normalize_mode(_setting(settings, "gpu_acceleration", "Off") if mode is None else mode)
    backend_objects = backends if backends is not None else registered_backends()
    probed = probe_backends(backends=backend_objects, include_cpu=False, refresh=refresh)
    selection = select_backend(selected_mode, required_capabilities, backends=backend_objects, probed_backends=probed)
    required = capabilities_to_mask(required_capabilities)
    if _should_try_next_backend_for_tile_limit(selection, settings, image_shape, required):
        fallback_backends = [backend for backend in backend_objects if backend.backend_id != "d3d12_compute"]
        fallback_probed = [info for info in probed if info.get("id") != "d3d12_compute"]
        fallback_selection = select_backend(selected_mode, required_capabilities, backends=fallback_backends, probed_backends=fallback_probed)
        if fallback_selection.backend is not None:
            selection = fallback_selection
    if selection.backend is None:
        return NoOpGpuSession(selection)
    return selection.backend.begin_render(settings, image_shape, force_gpu=selected_mode == "Force GPU")


def end_render(session: GpuBackendSession | None) -> None:
    if session is not None:
        session.end_render()


def process_color_tile(
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
    session: GpuBackendSession | None = None,
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    owned = session is None
    if session is None:
        session = begin_render(settings, rgb_tile.shape[:2], required_capabilities=required_capabilities)
    try:
        return session.process_color_tile(
            rgb_tile,
            alpha_tile,
            background_mask,
            edge_mask,
            probability,
            fringe_mask,
            screen_tile,
            nearest_fg_rgb,
            nearest_fg_valid,
            screen_color,
            settings,
        )
    finally:
        if owned:
            session.end_render()


def process_full_color_tile(
    rgb_tile: np.ndarray,
    alpha_tile: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_tile: np.ndarray | None,
    nearest_inner_rgb: np.ndarray | None,
    nearest_inner_valid: np.ndarray | None,
    screen_color: tuple[int, int, int],
    settings: Any,
    *,
    transition_nearest_rgb: np.ndarray | None = None,
    transition_nearest_valid: np.ndarray | None = None,
    session: GpuBackendSession | None = None,
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    owned = session is None
    if session is None:
        required = capabilities_to_mask(required_capabilities) | BackendCapability.FULL_COLOR_TILE | BackendCapability.RGB_ONLY
        required |= BackendCapability.SCREEN_TILE if screen_tile is not None else BackendCapability.CONSTANT_SCREEN
        session = begin_render(settings, rgb_tile.shape[:2], required_capabilities=required)
    try:
        process_full = getattr(session, "process_full_color_tile", None)
        if process_full is None:
            return _fallback_result(
                "unsupported_capability",
                "Selected GPU backend does not expose full color tile processing; CPU color path remains active.",
                backend=getattr(session, "backend_id", None),
                backend_name=getattr(session, "backend_name", None),
            )
        return process_full(
            rgb_tile,
            alpha_tile,
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
        )
    finally:
        if owned:
            session.end_render()


def native_buffer_from_array(
    name: str,
    array: np.ndarray,
    *,
    expected_channels: int,
    allowed_dtypes: tuple[NativeDType, ...] = (NativeDType.UINT8,),
) -> ImgKeyNativeTileBufferV1:
    if not isinstance(array, np.ndarray):
        raise NativeAbiError(NativeFallbackReason.BAD_DTYPE, f"{name} must be a numpy.ndarray")
    dtype = _native_dtype_for_array(name, array, allowed_dtypes)
    if expected_channels == 1:
        if array.ndim != 2:
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} must have shape HxW")
        height, width = array.shape
        channels = 1
        row_stride, pixel_stride = array.strides
    else:
        if array.ndim != 3 or array.shape[2] != expected_channels:
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} must have shape HxWx{expected_channels}")
        height, width, channels = array.shape
        row_stride, pixel_stride, channel_stride = array.strides
        if channel_stride != 1:
            raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} must have tightly packed channels")
    if width <= 0 or height <= 0:
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} dimensions must be positive")
    if row_stride <= 0 or pixel_stride <= 0:
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} strides must be positive")
    if not array.flags.c_contiguous:
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} must be C-contiguous before native dispatch")
    buffer = ImgKeyNativeTileBufferV1(
        struct_size=ctypes.sizeof(ImgKeyNativeTileBufferV1),
        version=IMGKEY_GPU_ABI_VERSION,
        data=ctypes.c_void_p(int(array.ctypes.data)),
        width=int(width),
        height=int(height),
        channels=int(channels),
        dtype=int(dtype),
        row_stride_bytes=int(row_stride),
        pixel_stride_bytes=int(pixel_stride),
        byte_size=int(array.nbytes),
    )
    validate_native_buffer_v1(name, buffer, expected_channels=expected_channels, allowed_dtypes=allowed_dtypes)
    return buffer


def validate_native_buffer_v1(
    name: str,
    buffer: ImgKeyNativeTileBufferV1 | None,
    *,
    expected_channels: int | None = None,
    allowed_dtypes: tuple[NativeDType, ...] = (NativeDType.UINT8, NativeDType.BOOL8),
) -> None:
    if buffer is None:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, f"{name} buffer pointer is null")
    if int(buffer.struct_size) != ctypes.sizeof(ImgKeyNativeTileBufferV1):
        raise NativeAbiError(NativeFallbackReason.BAD_VERSION, f"{name} buffer struct_size is unsupported")
    if int(buffer.version) != IMGKEY_GPU_ABI_VERSION:
        raise NativeAbiError(NativeFallbackReason.BAD_VERSION, f"{name} buffer version {buffer.version} is unsupported")
    if not buffer.data:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, f"{name} data pointer is null")
    if int(buffer.width) <= 0 or int(buffer.height) <= 0:
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} dimensions must be positive")
    if expected_channels is not None and int(buffer.channels) != int(expected_channels):
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} channel count must be {expected_channels}")
    try:
        dtype = NativeDType(int(buffer.dtype))
    except ValueError as exc:
        raise NativeAbiError(NativeFallbackReason.BAD_DTYPE, f"{name} dtype {buffer.dtype} is unsupported") from exc
    if dtype not in allowed_dtypes:
        raise NativeAbiError(NativeFallbackReason.BAD_DTYPE, f"{name} dtype {buffer.dtype} is unsupported")
    row_stride = int(buffer.row_stride_bytes)
    pixel_stride = int(buffer.pixel_stride_bytes)
    channels = int(buffer.channels)
    if row_stride <= 0 or pixel_stride <= 0:
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} strides must be positive")
    if pixel_stride < max(1, channels):
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} pixel stride is too small for its channel count")
    min_row = (int(buffer.width) - 1) * pixel_stride + max(1, channels)
    if row_stride < min_row:
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} row stride is smaller than the visible row span")
    min_span = (int(buffer.height) - 1) * row_stride + min_row
    if int(buffer.byte_size) < min_span:
        raise NativeAbiError(NativeFallbackReason.BAD_STRIDE, f"{name} byte_size is smaller than the addressed span")


def validate_native_color_tile_inputs(
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
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
) -> NativeColorTileCallV1:
    required = capabilities_to_mask(required_capabilities)
    screen = tuple(int(v) for v in screen_color)
    if len(screen) != 3 or any(v < 0 or v > 255 for v in screen):
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "screen_color must contain three 0..255 values")
    rgb = np.asarray(rgb_tile)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "rgb_tile must have shape HxWx3")
    shape = rgb.shape[:2]
    if nearest_fg_rgb is None or nearest_fg_valid is None:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, "foreground reference inputs are required for native color tile dispatch")
    buffers: dict[str, ImgKeyNativeTileBufferV1] = {
        "rgb": native_buffer_from_array("rgb", rgb, expected_channels=3),
        "alpha": native_buffer_from_array("alpha", np.asarray(alpha_tile), expected_channels=1),
        "background_mask": native_buffer_from_array("background_mask", np.asarray(background_mask), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
        "edge_mask": native_buffer_from_array("edge_mask", np.asarray(edge_mask), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
        "probability": native_buffer_from_array("probability", np.asarray(probability), expected_channels=1),
        "fringe_mask": native_buffer_from_array("fringe_mask", np.asarray(fringe_mask), expected_channels=1),
        "foreground_ref_rgb": native_buffer_from_array("foreground_ref_rgb", np.asarray(nearest_fg_rgb), expected_channels=3),
        "foreground_ref_valid": native_buffer_from_array("foreground_ref_valid", np.asarray(nearest_fg_valid), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
    }
    for name, buffer in buffers.items():
        if (int(buffer.height), int(buffer.width)) != tuple(shape):
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} must match rgb tile dimensions")
    if screen_tile is not None:
        buffers["screen_tile"] = native_buffer_from_array("screen_tile", np.asarray(screen_tile), expected_channels=3)
        if (int(buffers["screen_tile"].height), int(buffers["screen_tile"].width)) != tuple(shape):
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "screen_tile must match rgb tile dimensions")
        required |= BackendCapability.SCREEN_TILE
    else:
        required |= BackendCapability.CONSTANT_SCREEN
    required |= BackendCapability.RGB_ONLY
    params = ImgKeyNativeColorTileParamsV1(
        struct_size=ctypes.sizeof(ImgKeyNativeColorTileParamsV1),
        version=IMGKEY_GPU_ABI_VERSION,
        required_capabilities=int(required),
        status=IMGKEY_GPU_OK,
        fallback_reason=int(NativeFallbackReason.NONE),
        screen_r=int(screen[0]),
        screen_g=int(screen[1]),
        screen_b=int(screen[2]),
        reserved0=0,
        foreground_reference_pull=float(_clip01(_setting(settings, "foreground_reference_pull", 0.65))),
        key_vector_despill=float(_clip01(_setting(settings, "key_vector_despill", 0.75))),
        preserve_foreground_luma=float(_clip01(_setting(settings, "preserve_foreground_luma", 0.85))),
        transition_spill_threshold=float(_setting(settings, "transition_spill_threshold", 0.08)),
        transition_reconstruction_error=float(_setting(settings, "transition_reconstruction_error", 0.08)),
        clip_foreground=float(_clip01(_setting(settings, "clip_foreground", 0.14))),
        transition_alpha_min=int(np.clip(int(_setting(settings, "transition_alpha_min", 2)), 0, 255)),
        transition_alpha_max=int(np.clip(int(_setting(settings, "transition_alpha_max", 253)), 0, 255)),
    )
    _validate_params_struct(params)
    return NativeColorTileCallV1(params=params, buffers=buffers)


def validate_native_full_color_tile_inputs(
    rgb_tile: np.ndarray,
    alpha_tile: np.ndarray,
    background_mask: np.ndarray,
    edge_mask: np.ndarray,
    probability: np.ndarray,
    fringe_mask: np.ndarray,
    screen_tile: np.ndarray | None,
    nearest_inner_rgb: np.ndarray | None,
    nearest_inner_valid: np.ndarray | None,
    transition_ref_rgb: np.ndarray | None,
    transition_ref_valid: np.ndarray | None,
    screen_color: tuple[int, int, int],
    settings: Any,
    *,
    clamp_key_linear: tuple[float, float, float],
    required_capabilities: BackendCapability | int | set[str] | list[str] | tuple[str, ...] | None = None,
) -> NativeColorTileCallV2:
    required = capabilities_to_mask(required_capabilities)
    screen = tuple(int(v) for v in screen_color)
    if len(screen) != 3 or any(v < 0 or v > 255 for v in screen):
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "screen_color must contain three 0..255 values")
    rgb = np.asarray(rgb_tile)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "rgb_tile must have shape HxWx3")
    shape = rgb.shape[:2]
    if nearest_inner_rgb is None or nearest_inner_valid is None or transition_ref_rgb is None or transition_ref_valid is None:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, "nearest and transition reference inputs are required for native full color tile dispatch")
    buffers: dict[str, ImgKeyNativeTileBufferV1] = {
        "rgb": native_buffer_from_array("rgb", rgb, expected_channels=3),
        "alpha": native_buffer_from_array("alpha", np.asarray(alpha_tile), expected_channels=1),
        "background_mask": native_buffer_from_array("background_mask", np.asarray(background_mask), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
        "edge_mask": native_buffer_from_array("edge_mask", np.asarray(edge_mask), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
        "probability": native_buffer_from_array("probability", np.asarray(probability), expected_channels=1),
        "fringe_mask": native_buffer_from_array("fringe_mask", np.asarray(fringe_mask), expected_channels=1),
        "nearest_inner_rgb": native_buffer_from_array("nearest_inner_rgb", np.asarray(nearest_inner_rgb), expected_channels=3),
        "nearest_inner_valid": native_buffer_from_array("nearest_inner_valid", np.asarray(nearest_inner_valid), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
        "transition_ref_rgb": native_buffer_from_array("transition_ref_rgb", np.asarray(transition_ref_rgb), expected_channels=3),
        "transition_ref_valid": native_buffer_from_array("transition_ref_valid", np.asarray(transition_ref_valid), expected_channels=1, allowed_dtypes=(NativeDType.UINT8, NativeDType.BOOL8)),
    }
    for name, buffer in buffers.items():
        if (int(buffer.height), int(buffer.width)) != tuple(shape):
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, f"{name} must match rgb tile dimensions")
    if screen_tile is not None:
        buffers["screen_tile"] = native_buffer_from_array("screen_tile", np.asarray(screen_tile), expected_channels=3)
        if (int(buffers["screen_tile"].height), int(buffers["screen_tile"].width)) != tuple(shape):
            raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "screen_tile must match rgb tile dimensions")
        required |= BackendCapability.SCREEN_TILE
    else:
        required |= BackendCapability.CONSTANT_SCREEN
    required |= BackendCapability.RGB_ONLY | BackendCapability.FULL_COLOR_TILE
    clamp_key = tuple(float(np.clip(v, 0.0, 1.0)) for v in clamp_key_linear)
    if len(clamp_key) != 3:
        raise NativeAbiError(NativeFallbackReason.BAD_SHAPE, "clamp_key_linear must contain three values")
    transition_enabled = bool(_setting(settings, "transition_unmix", True))
    params = ImgKeyNativeColorTileParamsV2(
        struct_size=ctypes.sizeof(ImgKeyNativeColorTileParamsV2),
        version=IMGKEY_GPU_ABI_VERSION,
        required_capabilities=int(required),
        status=IMGKEY_GPU_OK,
        fallback_reason=int(NativeFallbackReason.NONE),
        screen_r=int(screen[0]),
        screen_g=int(screen[1]),
        screen_b=int(screen[2]),
        reserved0=0,
        foreground_reference_pull=float(_clip01(_setting(settings, "foreground_reference_pull", 0.65))),
        key_vector_despill=float(_clip01(_setting(settings, "key_vector_despill", 0.75))),
        preserve_foreground_luma=float(_clip01(_setting(settings, "preserve_foreground_luma", 0.85))),
        transition_spill_threshold=float(_setting(settings, "transition_spill_threshold", 0.08)),
        transition_reconstruction_error=float(_setting(settings, "transition_reconstruction_error", 0.08)),
        clip_foreground=float(_clip01(_setting(settings, "clip_foreground", 0.14))),
        transition_alpha_min=int(np.clip(int(_setting(settings, "transition_alpha_min", 2)), 0, 255)),
        transition_alpha_max=int(np.clip(int(_setting(settings, "transition_alpha_max", 253)), 0, 255)),
        despill=float(_clip01(_setting(settings, "despill", 0.70))),
        decontaminate=float(_clip01(_setting(settings, "decontaminate", 0.50))),
        unmix_amount=float(_clip01(_setting(settings, "unmix_amount", 0.75))),
        edge_color_repair=float(_clip01(_setting(settings, "edge_color_repair", 0.65))),
        inner_color_pull=float(_clip01(_setting(settings, "inner_color_pull", 0.45))),
        fringe_remove=float(_clip01(_setting(settings, "fringe_remove", 0.75))),
        luminance_protect=float(_effective_luminance_protect(settings)),
        clamp_key_r=clamp_key[0],
        clamp_key_g=clamp_key[1],
        clamp_key_b=clamp_key[2],
        transition_enabled=1 if transition_enabled else 0,
        transition_reference_enabled=1 if (transition_enabled and _foreground_reference_radius(settings) > 0) else 0,
    )
    _validate_params_struct(params)
    return NativeColorTileCallV2(params=params, buffers=buffers)


def _validate_params_struct(params: ImgKeyNativeColorTileParamsV1 | ImgKeyNativeColorTileParamsV2 | None) -> None:
    if params is None:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, "params pointer is null")
    expected_size = ctypes.sizeof(type(params))
    if int(params.struct_size) != expected_size:
        raise NativeAbiError(NativeFallbackReason.BAD_VERSION, "params struct_size is unsupported")
    if int(params.version) != IMGKEY_GPU_ABI_VERSION:
        raise NativeAbiError(NativeFallbackReason.BAD_VERSION, f"params version {params.version} is unsupported")


def _native_dtype_for_array(name: str, array: np.ndarray, allowed_dtypes: tuple[NativeDType, ...]) -> NativeDType:
    if array.dtype == np.uint8:
        dtype = NativeDType.UINT8
    elif array.dtype == np.bool_:
        dtype = NativeDType.BOOL8
    else:
        raise NativeAbiError(NativeFallbackReason.BAD_DTYPE, f"{name} must have dtype uint8 or bool8")
    if dtype not in allowed_dtypes:
        raise NativeAbiError(NativeFallbackReason.BAD_DTYPE, f"{name} dtype {array.dtype} is not accepted for this buffer")
    return dtype
