from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


MANIFEST_PATH = Path(__file__).with_name("birefnet_manifest.json")
OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
SUPPORTED_MODES = frozenset({"global_only", "global_plus_roi"})
SUPPORTED_PRECISIONS = frozenset({"fp16", "float16", "fp32", "float32"})
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

ProgressCallback = Callable[..., Any]
CancelCallback = Callable[[], bool]


class ModelValidationError(ValueError):
    """Raised when a BiRefNet local snapshot path is invalid."""


class BiRefNetInferenceError(RuntimeError):
    """Raised for BiRefNet runtime failures after validation succeeds."""


def load_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in BiRefNet offline manifest without importing AI stacks."""

    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("backend") != "birefnet" or manifest.get("model_family") != "BiRefNet":
        raise ModelValidationError("Manifest is not a BiRefNet-only manifest.")
    return manifest


def validate_model_path(
    model_path: str | Path | None,
    *,
    manifest: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    verify_hashes: bool = True,
    require_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a local BiRefNet snapshot path against the offline manifest.

    Only existing local directories are accepted. URLs, Hugging Face repo IDs,
    empty cache directories, and missing paths are rejected before any torch or
    transformers import can occur.
    """

    text = "" if model_path is None else str(model_path).strip()
    if not text:
        raise ModelValidationError("BiRefNet model path is empty; provide an existing local snapshot directory.")
    if _looks_like_url(text):
        raise ModelValidationError("BiRefNet model path must be a local directory, not a URL.")

    path = Path(text).expanduser()
    if not path.exists() and _looks_like_repo_id(text):
        raise ModelValidationError("BiRefNet model path must be local; Hugging Face repo IDs are not accepted at runtime.")
    if not path.exists():
        raise ModelValidationError(f"BiRefNet model path does not exist: {path}")
    if not path.is_dir():
        raise ModelValidationError(f"BiRefNet model path must be a directory: {path}")

    spec = manifest if manifest is not None else load_manifest(manifest_path)
    required = list(spec.get("expected_layout", {}).get("root_required_files", []))
    optional = list(spec.get("expected_layout", {}).get("root_optional_files", []))
    if not required:
        raise ModelValidationError("BiRefNet manifest has no required files; refusing to validate an unconstrained snapshot.")

    file_results: list[dict[str, Any]] = []
    for entry in required:
        file_results.append(_validate_manifest_file(path, entry, required=True, verify_hashes=verify_hashes, require_hashes=require_hashes))
    optional_results = [
        _validate_manifest_file(path, entry, required=False, verify_hashes=verify_hashes, require_hashes=False)
        for entry in optional
    ]

    config_info = _validate_config_json(path / "config.json", spec)
    _validate_license_metadata(path, spec)

    return {
        "ok": True,
        "backend": "birefnet",
        "model_path": str(path),
        "manifest_source": spec.get("source", {}),
        "required_files": file_results,
        "optional_files": optional_results,
        "config": config_info,
    }


def ensure_rgb_u8(rgb_u8: Any) -> np.ndarray:
    """Return a contiguous HxWx3 uint8 RGB array."""

    arr = np.asarray(rgb_u8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb_u8 must have shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def ensure_alpha_u8(alpha: Any, shape: tuple[int, int]) -> np.ndarray:
    """Return a contiguous HxW uint8 alpha mask matching ``shape``."""

    arr = np.asarray(alpha)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1:] == shape:
            arr = arr[0]
        elif arr.shape[-1] >= 1 and arr.shape[:2] == shape:
            arr = arr[:, :, 0]
        else:
            arr = np.squeeze(arr)
    if arr.shape != shape:
        raise ValueError(f"alpha mask must have shape {shape}, got {arr.shape}")
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr_f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        scale = 255.0 if (arr_f.size == 0 or float(np.nanmax(arr_f)) <= 1.0) else 1.0
        arr = np.clip(arr_f * scale, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def generate_alpha_hint(
    rgb_u8: Any,
    model_path: str | Path | None,
    device: str = "cuda",
    max_side: int = 1536,
    mode: str = "global_plus_roi",
    tile_size: int = 1024,
    tile_overlap: int = 192,
    precision: str = "fp16",
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> dict[str, Any]:
    """Generate a BiRefNet alpha hint from an RGB image.

    Phase 3 supports ``global_only`` inference. ``global_plus_roi`` is accepted
    for API stability and currently returns the global result plus structured ROI
    metadata documenting that Phase 6 classical edge/conflict masks will drive
    real ROI refinement later.
    """

    tile_info = {
        "requested_mode": mode,
        "implemented_mode": "global_only",
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "roi_strategy": "global_fallback_until_phase6_classical_roi_masks",
        "roi_count": 0,
    }

    try:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported BiRefNet mode {mode!r}; supported modes: {sorted(SUPPORTED_MODES)}")
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported BiRefNet precision {precision!r}; supported precisions: {sorted(SUPPORTED_PRECISIONS)}")
        rgb = ensure_rgb_u8(rgb_u8)
        h, w = rgb.shape[:2]
        tile_info["input_shape"] = [int(h), int(w)]
        if h <= 0 or w <= 0:
            raise ValueError("rgb_u8 must be non-empty")
        if _is_cancelled(cancel_callback):
            return _error_result("BiRefNet inference cancelled before validation.", "cancelled", tile_info)

        _notify_progress(progress_callback, 0.05, "Validating local BiRefNet snapshot")
        validation = validate_model_path(model_path)
        if _is_cancelled(cancel_callback):
            return _error_result("BiRefNet inference cancelled before model import.", "cancelled", tile_info)

        _notify_progress(progress_callback, 0.15, "Importing BiRefNet runtime dependencies")
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForImageSegmentation  # type: ignore[import-not-found]
        except Exception as exc:
            return _error_result(
                "BiRefNet dependencies are unavailable. Install the GPU BiRefNet runtime with torch and transformers to run inference. "
                f"Import error: {type(exc).__name__}: {exc}",
                "dependency_unavailable",
                tile_info,
            )

        torch_device, use_fp16, device_error = _resolve_device(torch, device, precision)
        if device_error:
            return _error_result(device_error, "device_unavailable", tile_info)

        inference_shape = _fit_size_for_inference(h, w, int(max_side))
        tile_info["inference_shape"] = [int(inference_shape[0]), int(inference_shape[1])]
        tile_info["precision"] = "fp16" if use_fp16 else "fp32"
        tile_info["device"] = str(torch_device)
        if mode == "global_plus_roi":
            tile_info["roi_note"] = (
                "Phase 3 does not invent ROI crops from raw RGB. Future phases may pass classical edge/conflict masks; "
                "until then global_plus_roi conservatively returns the global BiRefNet pass."
            )

        with _offline_hf_environment():
            _notify_progress(progress_callback, 0.25, "Loading local BiRefNet model")
            if _is_cancelled(cancel_callback):
                return _error_result("BiRefNet inference cancelled before model load.", "cancelled", tile_info)
            model = AutoModelForImageSegmentation.from_pretrained(
                validation["model_path"],
                trust_remote_code=True,
                local_files_only=True,
            )
            model.eval()
            model.to(torch_device)
            if use_fp16 and hasattr(model, "half"):
                model.half()

            _notify_progress(progress_callback, 0.55, "Running global BiRefNet pass")
            tensor = _prepare_tensor(rgb, inference_shape, torch, torch_device, use_fp16)
            if _is_cancelled(cancel_callback):
                return _error_result("BiRefNet inference cancelled before forward pass.", "cancelled", tile_info)
            with torch.inference_mode():
                output = model(tensor)
            pred = _select_prediction_tensor(output, torch)
            alpha_small = _prediction_to_numpy_alpha(pred)

        _notify_progress(progress_callback, 0.85, "Upscaling BiRefNet alpha hint")
        alpha_hint = _resize_alpha_u8(alpha_small, (h, w))
        _notify_progress(progress_callback, 1.0, "BiRefNet alpha hint complete")
        return {
            "ok": True,
            "alpha_hint": alpha_hint,
            "message": "BiRefNet global alpha hint generated from a validated local snapshot.",
            "tile_info": tile_info,
            "model": {
                "backend": "birefnet",
                "path": validation["model_path"],
                "source": validation["manifest_source"],
            },
        }
    except ModelValidationError as exc:
        return _error_result(str(exc), "model_validation_failed", tile_info)
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        return _error_result(f"BiRefNet inference failed: {type(exc).__name__}: {exc}", "inference_failed", tile_info)


def _looks_like_url(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text)) or text.startswith(("hf://", "hf_hub://"))


def _looks_like_repo_id(text: str) -> bool:
    normalized = text.replace("\\", "/")
    if normalized.startswith((".", "~", "/")) or ":" in normalized:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", normalized))


def _validate_manifest_file(
    root: Path,
    entry: dict[str, Any],
    *,
    required: bool,
    verify_hashes: bool,
    require_hashes: bool,
) -> dict[str, Any]:
    rel = str(entry.get("path", "")).strip()
    if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ModelValidationError(f"Invalid manifest file path: {rel!r}")
    path = root / rel
    exists = path.is_file()
    if required and not exists:
        raise ModelValidationError(f"BiRefNet snapshot is missing required file: {rel}")

    expected_sha = entry.get("sha256")
    if require_hashes and required and not expected_sha:
        raise ModelValidationError(f"BiRefNet manifest requires a SHA256 before bundling: {rel}")

    actual_sha = None
    if exists and verify_hashes and expected_sha:
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != str(expected_sha).lower():
            raise ModelValidationError(f"BiRefNet file hash mismatch for {rel}: expected {expected_sha}, got {actual_sha}")

    return {
        "path": rel,
        "required": bool(required),
        "exists": exists,
        "sha256_checked": bool(exists and verify_hashes and expected_sha),
        "sha256": actual_sha,
        "role": entry.get("role"),
    }


def _validate_config_json(config_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelValidationError(f"BiRefNet config.json could not be parsed: {type(exc).__name__}: {exc}") from exc

    expectations = manifest.get("transformers_config_expectations", {})
    expected_arches = set(expectations.get("architectures", ["BiRefNet"]))
    actual_arches = set(config.get("architectures") or [])
    if not (expected_arches & actual_arches):
        raise ModelValidationError(f"BiRefNet config.json must declare architecture {sorted(expected_arches)}, got {sorted(actual_arches)}")

    expected_auto_map = expectations.get("auto_map", {})
    actual_auto_map = config.get("auto_map") or {}
    for key, expected in expected_auto_map.items():
        if actual_auto_map.get(key) != expected:
            raise ModelValidationError(f"BiRefNet config.json auto_map[{key!r}] must be {expected!r}, got {actual_auto_map.get(key)!r}")

    if "BiRefNet" not in json.dumps(config):
        raise ModelValidationError("BiRefNet config.json does not appear to describe BiRefNet.")
    return {
        "architectures": sorted(actual_arches),
        "auto_map": actual_auto_map,
        "bb_pretrained": config.get("bb_pretrained"),
    }


def _validate_license_metadata(root: Path, manifest: dict[str, Any]) -> None:
    layout = manifest.get("expected_layout", {})
    for entry in layout.get("root_required_files", []):
        terms = entry.get("must_contain_any") or []
        if not terms:
            continue
        rel = str(entry.get("path", ""))
        text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        if not any(str(term).lower() in lowered for term in terms):
            raise ModelValidationError(f"BiRefNet snapshot file {rel} is missing expected license/notice metadata.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _offline_hf_environment():
    previous = {name: os.environ.get(name) for name in OFFLINE_ENV_VARS}
    try:
        for name in OFFLINE_ENV_VARS:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fit_size_for_inference(height: int, width: int, max_side: int) -> tuple[int, int]:
    max_side = max(32, int(max_side) if max_side else 1536)
    scale = min(1.0, max_side / float(max(height, width)))
    out_h = max(32, int(round(height * scale)))
    out_w = max(32, int(round(width * scale)))
    out_h = max(32, int(round(out_h / 32.0)) * 32)
    out_w = max(32, int(round(out_w / 32.0)) * 32)
    return out_h, out_w


def _prepare_tensor(rgb: np.ndarray, shape: tuple[int, int], torch_module: Any, torch_device: Any, use_fp16: bool) -> Any:
    resized = _resize_rgb_u8(rgb, shape)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    tensor = torch_module.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device=torch_device)
    return tensor.half() if use_fp16 else tensor.float()


def _resize_rgb_u8(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    target_h, target_w = shape
    if rgb.shape[:2] == (target_h, target_w):
        return np.ascontiguousarray(rgb)
    image = Image.fromarray(rgb, mode="RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.resize((int(target_w), int(target_h)), resample=resample), dtype=np.uint8)


def _resize_alpha_u8(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    target_h, target_w = shape
    alpha_u8 = ensure_alpha_u8(alpha, alpha.shape[:2])
    if alpha_u8.shape == (target_h, target_w):
        return np.ascontiguousarray(alpha_u8)
    image = Image.fromarray(alpha_u8, mode="L")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.resize((int(target_w), int(target_h)), resample=resample), dtype=np.uint8)


def _resolve_device(torch_module: Any, device: str, precision: str) -> tuple[Any, bool, str | None]:
    requested = (device or "cuda").strip().lower()
    try:
        torch_device = torch_module.device(requested)
    except Exception:
        torch_device = requested

    wants_cuda = requested.startswith("cuda")
    if wants_cuda:
        try:
            if not bool(torch_module.cuda.is_available()):
                return torch_device, False, "CUDA was requested for BiRefNet, but torch.cuda.is_available() is false."
        except Exception as exc:
            return torch_device, False, f"CUDA availability check failed: {type(exc).__name__}: {exc}"

    use_fp16 = precision in {"fp16", "float16"} and wants_cuda
    return torch_device, use_fp16, None


def _select_prediction_tensor(output: Any, torch_module: Any) -> Any:
    tensor_type = getattr(torch_module, "Tensor", None)
    if tensor_type is not None and isinstance(output, tensor_type):
        return output
    if isinstance(output, dict):
        for key in ("logits", "pred", "prediction", "out", "output"):
            if key in output:
                return _select_prediction_tensor(output[key], torch_module)
        for value in output.values():
            try:
                return _select_prediction_tensor(value, torch_module)
            except BiRefNetInferenceError:
                continue
    if isinstance(output, (list, tuple)) and output:
        return _select_prediction_tensor(output[-1], torch_module)
    raise BiRefNetInferenceError("BiRefNet model output did not contain a tensor prediction.")


def _prediction_to_numpy_alpha(prediction: Any) -> np.ndarray:
    pred = prediction.float()
    try:
        min_value = float(pred.min().detach().cpu().item())
        max_value = float(pred.max().detach().cpu().item())
    except Exception:
        min_value, max_value = -1.0, 1.0
    if min_value < 0.0 or max_value > 1.0:
        pred = pred.sigmoid()
    pred = pred.detach().float().cpu()
    arr = pred.numpy()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise BiRefNetInferenceError(f"BiRefNet prediction must reduce to HxW alpha, got shape {arr.shape}")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _notify_progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is None:
        return
    try:
        callback(fraction, message)
        return
    except TypeError:
        pass
    try:
        callback(fraction)
        return
    except TypeError:
        callback({"progress": fraction, "message": message})


def _is_cancelled(callback: CancelCallback | None) -> bool:
    if callback is None:
        return False
    try:
        return bool(callback())
    except Exception:
        return False


def _error_result(message: str, code: str, tile_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "alpha_hint": None,
        "message": message,
        "error": {"code": code, "message": message},
        "tile_info": tile_info,
    }


__all__ = [
    "BiRefNetInferenceError",
    "MANIFEST_PATH",
    "ModelValidationError",
    "ensure_alpha_u8",
    "ensure_rgb_u8",
    "generate_alpha_hint",
    "load_manifest",
    "validate_model_path",
]
