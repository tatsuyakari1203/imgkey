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
POSTPROCESS_SETTINGS = {
    "enabled": True,
    "sure_background_threshold": 3,
    "midtone_threshold": 8,
    "midtone_gamma": 0.72,
    "close_radius": 1,
    "dilate_radius": 1,
    "dilate_support_threshold": 24,
    "dilate_max_delta": 18,
    "feather_radius": 2,
}
ROI_SETTINGS = {
    "max_count": 8,
    "max_total_pixels": 8_000_000,
    "min_candidate_pixels": 96,
    "max_rgb_edge_pixels": 20_000_000,
}

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

    ``global_only`` runs a bounded global pass, then applies conservative alpha
    post-processing. ``global_plus_roi`` runs the same global pass, selects a
    capped set of edge/detail ROIs from the global alpha and nearby RGB edges,
    runs high-resolution crop inference for those ROIs, and max-blends the crop
    alpha back into the global alpha before final post-processing.
    """

    tile_info = {
        "requested_mode": mode,
        "implemented_mode": mode if mode in SUPPORTED_MODES else "unvalidated",
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "roi_strategy": "alpha_edge_detail_rois" if mode == "global_plus_roi" else "disabled_global_only",
        "roi_limits": dict(ROI_SETTINGS),
        "roi_count": 0,
        "roi_candidates": 0,
        "rois": [],
        "postprocess": dict(POSTPROCESS_SETTINGS),
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
                "Runs a global pass, selects bounded alpha/RGB-edge detail ROIs, runs crop inference, "
                "max-blends crop alpha into the global alpha, then applies conservative post-processing."
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
            if _is_cancelled(cancel_callback):
                return _error_result("BiRefNet inference cancelled before forward pass.", "cancelled", tile_info)
            alpha_small = _run_model_alpha(rgb, inference_shape, model, torch, torch_device, use_fp16)
            _clear_torch_cache(torch, torch_device)

            _notify_progress(progress_callback, 0.72, "Upscaling BiRefNet alpha hint")
            alpha_hint = _resize_alpha_u8(alpha_small, (h, w))

            if mode == "global_plus_roi":
                _notify_progress(progress_callback, 0.76, "Selecting BiRefNet detail ROIs")
                rois = _select_refinement_rois(alpha_hint, rgb, tile_size=int(tile_size), tile_overlap=int(tile_overlap))
                tile_info["roi_candidates"] = len(rois)
                roi_alpha = alpha_hint
                if rois:
                    roi_alpha = alpha_hint.copy()
                    roi_max_side = max(256, min(int(max_side), max(256, int(tile_size) if tile_size else 1024)))
                    total_rois = len(rois)
                    for idx, roi in enumerate(rois, start=1):
                        if _is_cancelled(cancel_callback):
                            return _error_result(f"BiRefNet inference cancelled before ROI pass {idx}.", "cancelled", tile_info)
                        y0, y1, x0, x1 = roi
                        crop = rgb[y0:y1, x0:x1]
                        crop_shape = _fit_size_for_inference(y1 - y0, x1 - x0, roi_max_side)
                        _notify_progress(progress_callback, 0.76 + 0.16 * (idx - 1) / max(1, total_rois), f"Running BiRefNet ROI pass {idx}/{total_rois}")
                        crop_small = _run_model_alpha(crop, crop_shape, model, torch, torch_device, use_fp16)
                        crop_alpha = _resize_alpha_u8(crop_small, (y1 - y0, x1 - x0))
                        crop_alpha = _postprocess_alpha_u8(crop_alpha)
                        blend_stats = _blend_roi_alpha(roi_alpha, crop_alpha, roi, feather=max(8, min(int(tile_overlap) // 2 if tile_overlap else 32, 96)))
                        tile_info["rois"].append(
                            {
                                "index": idx,
                                "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                                "crop_shape": [int(y1 - y0), int(x1 - x0)],
                                "inference_shape": [int(crop_shape[0]), int(crop_shape[1])],
                                **blend_stats,
                            }
                        )
                        _clear_torch_cache(torch, torch_device)
                tile_info["roi_count"] = len(tile_info["rois"])
                tile_info["implemented_mode"] = "global_plus_roi" if tile_info["roi_count"] else "global_plus_roi_no_candidate"
                alpha_hint = roi_alpha

        _notify_progress(progress_callback, 0.94, "Post-processing BiRefNet alpha hint")
        before_postprocess = alpha_hint
        alpha_hint = _postprocess_alpha_u8(alpha_hint)
        tile_info["postprocess_stats"] = _alpha_change_stats(before_postprocess, alpha_hint)
        _notify_progress(progress_callback, 0.98, "Finalizing BiRefNet alpha hint")
        _notify_progress(progress_callback, 1.0, "BiRefNet alpha hint complete")
        return {
            "ok": True,
            "alpha_hint": alpha_hint,
            "message": "BiRefNet alpha hint generated from a validated local snapshot with conservative detail-preserving post-processing.",
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


def _run_model_alpha(
    rgb: np.ndarray,
    inference_shape: tuple[int, int],
    model: Any,
    torch_module: Any,
    torch_device: Any,
    use_fp16: bool,
) -> np.ndarray:
    tensor = _prepare_tensor(rgb, inference_shape, torch_module, torch_device, use_fp16)
    with torch_module.inference_mode():
        output = model(tensor)
    pred = _select_prediction_tensor(output, torch_module)
    alpha = _prediction_to_numpy_alpha(pred)
    del tensor, output, pred
    return alpha


def _clear_torch_cache(torch_module: Any, torch_device: Any) -> None:
    try:
        device_type = getattr(torch_device, "type", str(torch_device).split(":", 1)[0])
        if str(device_type).lower() == "cuda" and hasattr(torch_module, "cuda"):
            torch_module.cuda.empty_cache()
    except Exception:
        pass


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


def _postprocess_alpha_u8(alpha: Any) -> np.ndarray:
    """Conservatively lift/repair BiRefNet alpha without broad expansion.

    The intent is to counter the common global-pass erosion caused by one-pass
    downscaling while keeping exact background nearly fixed.  Operations only
    raise alpha near existing support; hard background remains zeroed.
    """

    alpha_u8 = ensure_alpha_u8(alpha, np.asarray(alpha).shape[:2])
    if alpha_u8.size == 0:
        return alpha_u8
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return _postprocess_alpha_numpy_fallback(alpha_u8)

    settings = POSTPROCESS_SETTINGS
    sure_bg = int(settings["sure_background_threshold"])
    midtone = int(settings["midtone_threshold"])
    gamma = float(settings["midtone_gamma"])
    close_radius = int(settings["close_radius"])
    dilate_radius = int(settings["dilate_radius"])
    support_threshold = int(settings["dilate_support_threshold"])
    max_delta = int(settings["dilate_max_delta"])
    feather_radius = int(settings["feather_radius"])

    out = alpha_u8.copy()
    mid_mask = (alpha_u8 >= midtone) & (alpha_u8 < 250)
    if np.any(mid_mask):
        lifted = np.power(alpha_u8.astype(np.float32) / 255.0, gamma) * 255.0
        out[mid_mask] = np.maximum(out[mid_mask], np.rint(lifted[mid_mask]).astype(np.uint8))

    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        close_support = cv2.dilate((alpha_u8 >= midtone).astype(np.uint8), kernel) > 0
        out[close_support] = np.maximum(out[close_support], closed[close_support])

    if dilate_radius > 0 and max_delta > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
        dilated = cv2.dilate(out, kernel).astype(np.int16)
        support = cv2.dilate((alpha_u8 >= support_threshold).astype(np.uint8), kernel) > 0
        # Do not create fresh alpha from exact/near-exact background. This keeps
        # massive background expansion out while still thickening eroded edges.
        expansion = support & (alpha_u8 > sure_bg) & (alpha_u8 < 250)
        capped = np.minimum(dilated, out.astype(np.int16) + max_delta)
        out[expansion] = np.maximum(out[expansion].astype(np.int16), capped[expansion]).astype(np.uint8)

    if feather_radius > 0:
        # Bilateral filtering feathers quantized crop/global seams while the
        # max() guard prevents the filter from eroding foreground detail.
        d = feather_radius * 2 + 1
        feathered = cv2.bilateralFilter(out, d=d, sigmaColor=28, sigmaSpace=max(1, feather_radius))
        edge = (out >= midtone) & (out <= 247)
        if np.any(edge):
            blended = np.rint(out.astype(np.float32) * 0.72 + feathered.astype(np.float32) * 0.28).astype(np.uint8)
            out[edge] = np.maximum(out[edge], blended[edge])

    out[alpha_u8 <= sure_bg] = 0
    out[alpha_u8 >= 252] = 255
    return np.ascontiguousarray(out)


def _postprocess_alpha_numpy_fallback(alpha_u8: np.ndarray) -> np.ndarray:
    out = alpha_u8.copy()
    sure_bg = int(POSTPROCESS_SETTINGS["sure_background_threshold"])
    midtone = int(POSTPROCESS_SETTINGS["midtone_threshold"])
    gamma = float(POSTPROCESS_SETTINGS["midtone_gamma"])
    mid_mask = (alpha_u8 >= midtone) & (alpha_u8 < 250)
    if np.any(mid_mask):
        lifted = np.power(alpha_u8.astype(np.float32) / 255.0, gamma) * 255.0
        out[mid_mask] = np.maximum(out[mid_mask], np.rint(lifted[mid_mask]).astype(np.uint8))
    out[alpha_u8 <= sure_bg] = 0
    out[alpha_u8 >= 252] = 255
    return np.ascontiguousarray(out)


def _select_refinement_rois(
    alpha: Any,
    rgb: Any | None = None,
    *,
    tile_size: int = 1024,
    tile_overlap: int = 192,
) -> list[tuple[int, int, int, int]]:
    alpha_u8 = ensure_alpha_u8(alpha, np.asarray(alpha).shape[:2])
    h, w = alpha_u8.shape
    if h <= 0 or w <= 0:
        return []
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return []

    soft_band = (alpha_u8 >= 8) & (alpha_u8 <= 247)
    grad_x = cv2.Sobel(alpha_u8, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_u8, cv2.CV_16S, 0, 1, ksize=3)
    alpha_edge = (np.abs(grad_x) + np.abs(grad_y)) > 24
    candidate = soft_band | alpha_edge

    if rgb is not None and h * w <= int(ROI_SETTINGS["max_rgb_edge_pixels"]):
        try:
            rgb_u8 = ensure_rgb_u8(rgb)
            gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
            rgb_edges = cv2.Canny(gray, 50, 125) > 0
            near_alpha = cv2.dilate((alpha_u8 >= 10).astype(np.uint8), np.ones((9, 9), dtype=np.uint8)) > 0
            candidate |= rgb_edges & near_alpha
        except Exception:
            pass

    if not np.any(candidate):
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    candidate_u8 = cv2.dilate(candidate_u8, kernel, iterations=1)
    contours, _hier = cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    roi_side = max(256, min(1536, int(tile_size) if tile_size else 1024))
    padding = max(16, min(int(tile_overlap) if tile_overlap else 96, roi_side // 4))
    min_pixels = int(ROI_SETTINGS["min_candidate_pixels"])
    boxes: list[tuple[int, int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
        for sy0, sy1, sx0, sx1 in _split_roi_box(y0, y1, x0, x1, roi_side=roi_side, overlap=padding):
            score = int(np.count_nonzero(candidate_u8[sy0:sy1, sx0:sx1]))
            if score < min_pixels:
                continue
            boxes.append((score, sy0, sy1, sx0, sx1))

    boxes.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    total_pixels = 0
    max_count = int(ROI_SETTINGS["max_count"])
    max_total = int(ROI_SETTINGS["max_total_pixels"])
    for _score, y0, y1, x0, x1 in boxes:
        roi = (int(y0), int(y1), int(x0), int(x1))
        if any(_roi_iou(roi, existing) > 0.82 for existing in selected):
            continue
        pixels = max(0, y1 - y0) * max(0, x1 - x0)
        if selected and total_pixels + pixels > max_total:
            continue
        selected.append(roi)
        total_pixels += pixels
        if len(selected) >= max_count:
            break
    return selected


def _split_roi_box(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    roi_side: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    if y1 <= y0 or x1 <= x0:
        return []
    if (y1 - y0) <= roi_side and (x1 - x0) <= roi_side:
        return [(y0, y1, x0, x1)]
    step = max(64, roi_side - max(0, overlap))
    out: list[tuple[int, int, int, int]] = []
    yy = y0
    while yy < y1:
        sy1 = min(y1, yy + roi_side)
        sy0 = max(y0, sy1 - roi_side)
        xx = x0
        while xx < x1:
            sx1 = min(x1, xx + roi_side)
            sx0 = max(x0, sx1 - roi_side)
            out.append((sy0, sy1, sx0, sx1))
            if sx1 >= x1:
                break
            xx += step
        if sy1 >= y1:
            break
        yy += step
    return out


def _roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter <= 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    return float(inter) / float(max(1, area_a + area_b - inter))


def _blend_roi_alpha(full_alpha: np.ndarray, crop_alpha: Any, roi: tuple[int, int, int, int], *, feather: int = 48) -> dict[str, Any]:
    y0, y1, x0, x1 = roi
    target = full_alpha[y0:y1, x0:x1]
    crop = ensure_alpha_u8(crop_alpha, target.shape)
    before = target.copy()
    improved = np.maximum(target, crop)
    if feather > 0:
        weight = _roi_feather_weight(target.shape, feather)
        blended = np.rint(target.astype(np.float32) * (1.0 - weight) + improved.astype(np.float32) * weight).astype(np.uint8)
        target[:, :] = np.maximum(target, blended)
    else:
        target[:, :] = improved
    return _alpha_change_stats(before, target)


def _roi_feather_weight(shape: tuple[int, int], feather: int) -> np.ndarray:
    h, w = shape
    if h <= 0 or w <= 0:
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    dist_y = np.minimum(yy + 1, h - yy)
    dist_x = np.minimum(xx + 1, w - xx)
    dist = np.minimum(dist_y, dist_x).astype(np.float32)
    return np.clip(dist / float(max(1, feather)), 0.0, 1.0)


def _alpha_change_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_i = before.astype(np.int16, copy=False)
    after_i = after.astype(np.int16, copy=False)
    delta = after_i - before_i
    changed = delta != 0
    raised = delta > 0
    return {
        "changed_pixels": int(np.count_nonzero(changed)),
        "raised_pixels": int(np.count_nonzero(raised)),
        "max_raise": int(delta[raised].max()) if np.any(raised) else 0,
        "mean_raise": float(delta[raised].mean()) if np.any(raised) else 0.0,
    }


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
