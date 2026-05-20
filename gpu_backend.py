from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum, IntFlag
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


CAPABILITY_NAMES: dict[BackendCapability, str] = {
    BackendCapability.CONSTANT_SCREEN: "constant_screen",
    BackendCapability.SCREEN_TILE: "screen_tile",
    BackendCapability.PERSISTENT_SESSION: "persistent_session",
    BackendCapability.TILE_BATCH: "tile_batch",
    BackendCapability.ALPHA_WRITE: "alpha_write",
    BackendCapability.RGB_ONLY: "rgb_only",
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
    "gpu_exception",
}

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
    ]


@dataclass(slots=True)
class NativeColorTileCallV1:
    params: ImgKeyNativeColorTileParamsV1
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

    def end_render(self) -> None:
        self.ended = True


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
    return [CudaCompatBackend()]


def probe_backends(
    *,
    backends: list[GpuBackend] | None = None,
    include_cpu: bool = True,
    refresh: bool = False,
    cuda_probe: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if backends is None:
        backends = [CudaCompatBackend(cuda_probe=cuda_probe)]
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
        cpu_caps = BackendCapability.CONSTANT_SCREEN | BackendCapability.SCREEN_TILE | BackendCapability.ALPHA_WRITE | BackendCapability.RGB_ONLY
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
    selection = select_backend(selected_mode, required_capabilities, backends=backends, refresh=refresh)
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
    )
    _validate_params_struct(params)
    return NativeColorTileCallV1(params=params, buffers=buffers)


def _validate_params_struct(params: ImgKeyNativeColorTileParamsV1 | None) -> None:
    if params is None:
        raise NativeAbiError(NativeFallbackReason.NULL_POINTER, "params pointer is null")
    if int(params.struct_size) != ctypes.sizeof(ImgKeyNativeColorTileParamsV1):
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
