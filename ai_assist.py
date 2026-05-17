from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


BIREFNET_MODEL_ENV = "IMGKEY_BIREFNET_MODEL"
BIREFNET_ADAPTER_ENV = "IMGKEY_BIREFNET_ADAPTER"
CORRIDORKEY_ADAPTER_ENV = "IMGKEY_CORRIDORKEY_ADAPTER"


@dataclass(frozen=True, slots=True)
class AssistCapability:
    name: str
    available: bool
    message: str
    missing_dependencies: tuple[str, ...] = ()
    missing_configuration: tuple[str, ...] = ()
    model_path: str | None = None
    adapter: str | None = None


@dataclass(frozen=True, slots=True)
class AlphaAssistResult:
    alpha_hint: np.ndarray
    message: str = ""


@dataclass(frozen=True, slots=True)
class CorridorKeyResult:
    foreground_rgb: np.ndarray | None
    alpha: np.ndarray
    processed_rgba: np.ndarray | None
    message: str = ""


class BiRefNetAlphaAssist:
    """Optional BiRefNet seam for coarse alpha-hint generation.

    This class deliberately does not import torch/transformers/weights at module
    import time and never downloads a model. The default app exposes the seam and
    capability reporting only. A user-controlled external adapter may be wired by
    setting ``IMGKEY_BIREFNET_ADAPTER=module:function`` plus a local model path
    in ``IMGKEY_BIREFNET_MODEL``.
    """

    optional_dependencies = ("torch", "torchvision", "transformers")

    def __init__(self, model_path: str | Path | None = None, adapter: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else _path_from_env(BIREFNET_MODEL_ENV)
        self.adapter = _resolve_config(adapter, BIREFNET_ADAPTER_ENV)

    @classmethod
    def from_environment(cls) -> "BiRefNetAlphaAssist":
        return cls()

    def capability(self) -> AssistCapability:
        missing_deps = tuple(name for name in self.optional_dependencies if importlib.util.find_spec(name) is None)
        missing_config: list[str] = []
        model_text: str | None = None

        if self.model_path is None:
            missing_config.append(f"local model path ({BIREFNET_MODEL_ENV})")
        else:
            model_text = str(self.model_path)
            if not self.model_path.exists():
                missing_config.append(f"existing local model path ({model_text})")

        if not self.adapter:
            missing_config.append(f"external adapter ({BIREFNET_ADAPTER_ENV}=module:function)")

        available = not missing_deps and not missing_config
        if available:
            message = "BiRefNet alpha assist is configured through an external adapter; the adapter is imported only when explicitly run and no model download is performed."
        else:
            parts: list[str] = ["BiRefNet alpha assist disabled."]
            if missing_deps:
                parts.append("Missing optional packages: " + ", ".join(missing_deps) + ".")
            if missing_config:
                parts.append("Missing configuration: " + ", ".join(missing_config) + ".")
            parts.append("Install/configure externally only; ImgKey does not bundle PyTorch/CUDA or weights.")
            message = " ".join(parts)
        return AssistCapability(
            name="BiRefNet alpha assist",
            available=available,
            message=message,
            missing_dependencies=missing_deps,
            missing_configuration=tuple(missing_config),
            model_path=model_text,
            adapter=self.adapter or None,
        )

    def generate_hint(self, rgb_u8: np.ndarray, *, max_side: int = 1536) -> AlphaAssistResult:
        capability = self.capability()
        if not capability.available:
            raise RuntimeError(capability.message)
        adapter = _load_adapter_callable(capability.adapter, default_name="generate_alpha_hint")
        rgb = _ensure_rgb_u8(rgb_u8)
        raw = adapter(rgb_u8=rgb, model_path=capability.model_path, max_side=int(max_side))
        if isinstance(raw, AlphaAssistResult):
            return raw
        if isinstance(raw, dict):
            alpha = raw.get("alpha_hint", raw.get("alpha"))
            message = str(raw.get("message", ""))
        else:
            alpha = raw
            message = ""
        return AlphaAssistResult(alpha_hint=_ensure_mask_u8(alpha, rgb.shape[:2], "alpha_hint"), message=message)


class CorridorKeyPlugin:
    """External CorridorKey adapter seam.

    Expected adapter signature:

    ``process(rgb_u8=<HxWx3 uint8>, alpha_hint_u8=<HxW uint8>)``

    Expected return: a ``dict`` with ``alpha`` and optional ``foreground`` and
    ``processed`` arrays, or a ``CorridorKeyResult``. CorridorKey itself, model
    weights, and runtimes are never bundled by ImgKey.
    """

    def __init__(self, adapter: str | None = None) -> None:
        self.adapter = _resolve_config(adapter, CORRIDORKEY_ADAPTER_ENV)

    @classmethod
    def from_environment(cls) -> "CorridorKeyPlugin":
        return cls()

    def capability(self) -> AssistCapability:
        missing_config: list[str] = []
        if not self.adapter:
            missing_config.append(f"external adapter ({CORRIDORKEY_ADAPTER_ENV}=module:function)")
        available = not missing_config
        if available:
            message = "CorridorKey plugin adapter is configured externally and will be imported only when explicitly run; ImgKey will pass RGB + coarse alpha hint and receive FG/alpha/processed outputs."
        else:
            message = (
                "CorridorKey plugin disabled. Missing configuration: "
                + ", ".join(missing_config)
                + ". Do not bundle or redistribute CorridorKey/runtime/model weights without a license decision."
            )
        return AssistCapability(
            name="CorridorKey plugin",
            available=available,
            message=message,
            missing_configuration=tuple(missing_config),
            adapter=self.adapter or None,
        )

    def process(self, rgb_u8: np.ndarray, alpha_hint_u8: np.ndarray) -> CorridorKeyResult:
        capability = self.capability()
        if not capability.available:
            raise RuntimeError(capability.message)
        rgb = _ensure_rgb_u8(rgb_u8)
        hint = _ensure_mask_u8(alpha_hint_u8, rgb.shape[:2], "alpha_hint_u8")
        adapter = _load_adapter_callable(capability.adapter, default_name="process")
        raw = adapter(rgb_u8=rgb, alpha_hint_u8=hint)
        if isinstance(raw, CorridorKeyResult):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("CorridorKey adapter must return CorridorKeyResult or dict")
        alpha = _ensure_mask_u8(raw.get("alpha"), rgb.shape[:2], "alpha")
        foreground = raw.get("foreground")
        if foreground is not None:
            foreground = _ensure_rgb_u8(foreground)
        processed = raw.get("processed")
        if processed is not None:
            processed = _ensure_rgba_u8(processed, rgb.shape[:2], "processed")
        return CorridorKeyResult(
            foreground_rgb=foreground,
            alpha=alpha,
            processed_rgba=processed,
            message=str(raw.get("message", "")),
        )


def check_birefnet_availability(model_path: str | Path | None = None, adapter: str | None = None) -> AssistCapability:
    return BiRefNetAlphaAssist(model_path=model_path, adapter=adapter).capability()


def check_corridorkey_availability(adapter: str | None = None) -> AssistCapability:
    return CorridorKeyPlugin(adapter=adapter).capability()


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _resolve_config(value: str | None, env_name: str) -> str:
    if value is None:
        return os.environ.get(env_name, "").strip()
    return value.strip()


def _load_adapter_callable(adapter: str | None, *, default_name: str) -> Callable[..., Any]:
    if not adapter:
        raise RuntimeError("No external adapter configured")
    module_name, sep, attr = adapter.partition(":")
    module_name = module_name.strip()
    attr = attr.strip() if sep else default_name
    module = importlib.import_module(module_name)
    target = module
    for part in attr.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Configured adapter {adapter!r} is not callable")
    return target


def _ensure_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb_u8 must have shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _ensure_rgba_u8(rgba: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] < 4 or arr.shape[:2] != shape:
        raise ValueError(f"{name} must have shape HxWx4 matching input RGB")
    arr = arr[:, :, :4]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _ensure_mask_u8(mask: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"{name} is required")
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0) * scale
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)
