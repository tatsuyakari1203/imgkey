from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
WORKER_NAME = "imgkey_ai_worker"
SUPPORTED_BACKENDS = frozenset({"birefnet"})
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / ".artifact" / "ai-worker"
DEFAULT_DEVICE = "cuda"
DEFAULT_MODE = "global_plus_roi"
DEFAULT_MAX_SIDE = 1536
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 192
DEFAULT_PRECISION = "fp16"


class WorkerContractError(ValueError):
    def __init__(self, code: str, message: str, *, stage: str = "request", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


def load_request(request_arg: str) -> dict[str, Any]:
    """Load a worker request from a JSON file path, '-' stdin, or inline JSON."""

    if request_arg == "-":
        text = sys.stdin.read()
        source = "stdin"
    else:
        candidate = Path(request_arg).expanduser()
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            source = str(candidate)
        else:
            text = request_arg
            source = "inline"

    try:
        request = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkerContractError(
            "request_json_invalid",
            f"AI worker request must be valid JSON ({source}): {exc.msg} at line {exc.lineno}, column {exc.colno}.",
            details={"source": source, "line": exc.lineno, "column": exc.colno},
        ) from exc

    if not isinstance(request, dict):
        raise WorkerContractError("request_not_object", "AI worker request JSON must be an object.", details={"source": source})
    return request


def run_worker_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run one AI worker request and return a structured JSON response.

    This function deliberately imports the BiRefNet adapter and model runtime only
    inside the request path. Importing ``ai_worker`` itself must remain free of
    torch/transformers/model imports so the default app startup stays lightweight.
    """

    started = _utc_timestamp()
    start_seconds = time.time()
    progress_events: list[dict[str, Any]] = []
    output_dir: Path | None = None
    staging_dir: Path | None = None
    response: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = {
        "started_utc": started,
        "progress": progress_events,
        "request": _summarize_request(request),
        "temp_cleanup": None,
    }

    try:
        if not isinstance(request, dict):
            raise WorkerContractError("request_not_object", "AI worker request JSON must be an object.")

        backend = _require_text(request, "backend")
        if backend not in SUPPORTED_BACKENDS:
            raise WorkerContractError(
                "unsupported_backend",
                f"Unsupported AI backend {backend!r}; supported backends: {sorted(SUPPORTED_BACKENDS)}.",
                details={"backend": backend, "supported_backends": sorted(SUPPORTED_BACKENDS)},
            )

        output_dir, temp_parent = _prepare_output_locations(request)
        staging_dir = Path(tempfile.mkdtemp(prefix="imgkey-ai-worker-", dir=str(temp_parent)))
        diagnostics["output_dir"] = str(output_dir)
        diagnostics["staging_dir"] = str(staging_dir)

        cancel_file = _optional_path(request, "cancel_file_path", "cancel_path")
        if _cancel_requested(cancel_file):
            raise WorkerContractError(
                "cancelled",
                "BiRefNet worker request was cancelled before validation.",
                stage="cancel",
                details={"cancel_file_path": str(cancel_file)},
            )

        input_path = _existing_file_path(request, "input_image_path")
        model_path = _require_text(request, "model_path")
        device = _optional_text(request, "device", DEFAULT_DEVICE)
        mode = _optional_text(request, "mode", DEFAULT_MODE)
        precision = _optional_text(request, "precision", DEFAULT_PRECISION)
        max_side = _optional_int(request, "max_side", DEFAULT_MAX_SIDE, minimum=32)
        tile_size = _optional_int(request, "tile_size", DEFAULT_TILE_SIZE, minimum=32)
        tile_overlap = _optional_int(request, "tile_overlap", DEFAULT_TILE_OVERLAP, minimum=0)

        _append_progress(progress_events, 0.02, "Importing BiRefNet adapter")
        from ai_backends import birefnet_adapter

        if mode not in birefnet_adapter.SUPPORTED_MODES:
            raise WorkerContractError(
                "unsupported_mode",
                f"Unsupported BiRefNet mode {mode!r}; supported modes: {sorted(birefnet_adapter.SUPPORTED_MODES)}.",
                details={"mode": mode, "supported_modes": sorted(birefnet_adapter.SUPPORTED_MODES)},
            )
        if precision not in birefnet_adapter.SUPPORTED_PRECISIONS:
            raise WorkerContractError(
                "unsupported_precision",
                f"Unsupported BiRefNet precision {precision!r}; supported precisions: {sorted(birefnet_adapter.SUPPORTED_PRECISIONS)}.",
                details={"precision": precision, "supported_precisions": sorted(birefnet_adapter.SUPPORTED_PRECISIONS)},
            )

        if _cancel_requested(cancel_file):
            raise WorkerContractError(
                "cancelled",
                "BiRefNet worker request was cancelled before model validation.",
                stage="cancel",
                details={"cancel_file_path": str(cancel_file)},
            )

        _append_progress(progress_events, 0.05, "Validating local BiRefNet snapshot")
        try:
            validation = birefnet_adapter.validate_model_path(model_path)
        except birefnet_adapter.ModelValidationError as exc:
            raise WorkerContractError("model_validation_failed", str(exc), stage="model_validation") from exc
        diagnostics["model_validation"] = _json_safe(validation)

        if _cancel_requested(cancel_file):
            raise WorkerContractError(
                "cancelled",
                "BiRefNet worker request was cancelled before image loading.",
                stage="cancel",
                details={"cancel_file_path": str(cancel_file)},
            )

        _append_progress(progress_events, 0.10, "Loading input image")
        rgb = _load_rgb_image(input_path)
        diagnostics["input_image"] = {
            "path": str(input_path),
            "shape": [int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])],
            "dtype": str(rgb.dtype),
        }

        def progress_callback(fraction: float, message: str | None = None) -> None:
            _append_progress(progress_events, fraction, message or "BiRefNet worker progress")

        def cancel_callback() -> bool:
            return _cancel_requested(cancel_file)

        _append_progress(progress_events, 0.12, "Starting BiRefNet adapter")
        adapter_result = birefnet_adapter.generate_alpha_hint(
            rgb,
            model_path=validation["model_path"],
            device=device,
            max_side=max_side,
            mode=mode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            precision=precision,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        diagnostics["adapter_result"] = _summarize_adapter_result(adapter_result)

        if not adapter_result.get("ok"):
            response = _response_error(
                _classify_adapter_error(adapter_result),
                str(adapter_result.get("message") or "BiRefNet adapter failed."),
                backend="birefnet",
                stage="adapter",
                details={"adapter_error": adapter_result.get("error"), "tile_info": adapter_result.get("tile_info")},
            )
        else:
            alpha = adapter_result.get("alpha_hint")
            alpha_path = _save_alpha_hint(alpha, output_dir=output_dir, staging_dir=staging_dir)
            _append_progress(progress_events, 1.0, "BiRefNet alpha hint written")
            response = _response_ok(
                "BiRefNet completed.",
                backend="birefnet",
                alpha_hint_path=str(alpha_path),
                details={"tile_info": adapter_result.get("tile_info"), "model": adapter_result.get("model")},
            )

    except WorkerContractError as exc:
        response = _response_error(exc.code, str(exc), backend=_safe_backend(request), stage=exc.stage, details=exc.details)
    except MemoryError as exc:
        response = _response_error(
            "out_of_memory",
            f"BiRefNet worker ran out of memory: {exc}",
            backend=_safe_backend(request),
            stage="runtime",
        )
    except Exception as exc:  # pragma: no cover - final worker safety net
        response = _response_error(
            _classify_exception(exc),
            f"BiRefNet worker failed unexpectedly: {type(exc).__name__}: {exc}",
            backend=_safe_backend(request),
            stage="runtime",
            details={"traceback": traceback.format_exc(limit=8)},
        )
    finally:
        cleanup = _cleanup_staging_dir(staging_dir)
        diagnostics["temp_cleanup"] = cleanup
        diagnostics["finished_utc"] = _utc_timestamp()
        diagnostics["duration_seconds"] = max(0.0, time.time() - start_seconds)

    if response is None:  # pragma: no cover - defensive
        response = _response_error("worker_no_response", "AI worker produced no response.", backend=_safe_backend(request), stage="runtime")

    if output_dir is not None:
        _write_diagnostics(response, diagnostics, output_dir)

    return response


def _response_ok(message: str, *, backend: str, alpha_hint_path: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "worker": WORKER_NAME,
        "backend": backend,
        "ok": True,
        "alpha_hint_path": alpha_hint_path,
        "diagnostics_path": None,
        "message": message,
        "error": None,
        "error_code": None,
        "details": _json_safe(details or {}),
    }


def _response_error(
    code: str,
    message: str,
    *,
    backend: str | None,
    stage: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = {
        "code": code,
        "type": _error_type(code),
        "stage": stage,
        "message": message,
        "details": _json_safe(details or {}),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "worker": WORKER_NAME,
        "backend": backend,
        "ok": False,
        "alpha_hint_path": None,
        "diagnostics_path": None,
        "message": message,
        "error": error,
        "error_code": code,
    }


def _error_type(code: str) -> str:
    if code == "cancelled":
        return "cancelled"
    if code in {"dependency_unavailable", "device_unavailable", "out_of_memory"}:
        return "runtime_unavailable"
    if code in {"model_validation_failed", "missing_input", "input_read_failed"}:
        return "input_validation"
    if code.startswith("unsupported_"):
        return "unsupported_contract"
    if code.startswith("request_"):
        return "request_contract"
    return "worker_error"


def _summarize_request(request: Any) -> Any:
    if not isinstance(request, dict):
        return {"type": type(request).__name__}
    keys = (
        "backend",
        "input_image_path",
        "model_path",
        "device",
        "mode",
        "max_side",
        "tile_size",
        "tile_overlap",
        "precision",
        "output_dir",
        "temp_dir",
        "cancel_file_path",
    )
    return {key: _json_safe(request.get(key)) for key in keys if key in request}


def _summarize_adapter_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = {key: _json_safe(value) for key, value in result.items() if key != "alpha_hint"}
    alpha = result.get("alpha_hint")
    if alpha is not None:
        summary["alpha_hint"] = {
            "shape": [int(v) for v in getattr(alpha, "shape", ())],
            "dtype": str(getattr(alpha, "dtype", "unknown")),
        }
    return summary


def _safe_backend(request: Any) -> str | None:
    if isinstance(request, dict):
        backend = request.get("backend")
        return str(backend) if backend is not None else None
    return None


def _require_text(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkerContractError(f"{key}_missing", f"AI worker request field {key!r} must be a non-empty string.")
    return value.strip()


def _optional_text(request: dict[str, Any], key: str, default: str) -> str:
    value = request.get(key, default)
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise WorkerContractError(f"{key}_invalid", f"AI worker request field {key!r} must be a string.")
    return value.strip()


def _optional_int(request: dict[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = request.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerContractError(f"{key}_invalid", f"AI worker request field {key!r} must be an integer.") from exc
    if parsed < minimum:
        raise WorkerContractError(f"{key}_invalid", f"AI worker request field {key!r} must be >= {minimum}.")
    return parsed


def _optional_path(request: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = request.get(key)
        if value:
            return Path(str(value)).expanduser().resolve(strict=False)
    return None


def _existing_file_path(request: dict[str, Any], key: str) -> Path:
    text = _require_text(request, key)
    if _looks_like_url(text):
        raise WorkerContractError("invalid_input_path", "AI worker input_image_path must be a local file path, not a URL.")
    path = Path(text).expanduser().resolve(strict=False)
    if not path.exists():
        raise WorkerContractError("missing_input", f"Input image does not exist: {path}", stage="input_validation", details={"path": str(path)})
    if not path.is_file():
        raise WorkerContractError("invalid_input_path", f"Input image path is not a file: {path}", stage="input_validation", details={"path": str(path)})
    return path


def _prepare_output_locations(request: dict[str, Any]) -> tuple[Path, Path]:
    output_path = _first_path(request, "output_dir", "output_directory") or DEFAULT_OUTPUT_DIR
    temp_path = _first_path(request, "temp_dir", "temp_directory")
    output_dir = output_path.expanduser().resolve(strict=False)
    temp_parent = (temp_path.expanduser().resolve(strict=False) if temp_path is not None else output_dir)

    for label, path in (("output_dir", output_dir), ("temp_dir", temp_parent)):
        if path == ROOT_DIR:
            raise WorkerContractError(
                "unsafe_output_directory",
                f"AI worker {label} cannot be the source root: {ROOT_DIR}",
                stage="output_validation",
                details={label: str(path)},
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_parent.mkdir(parents=True, exist_ok=True)
    return output_dir, temp_parent


def _first_path(request: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = request.get(key)
        if value not in (None, ""):
            return Path(str(value))
    return None


def _load_rgb_image(path: Path) -> Any:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise WorkerContractError(
            "input_read_failed",
            f"Input image could not be read as RGB: {type(exc).__name__}: {exc}",
            stage="input_load",
            details={"path": str(path)},
        ) from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise WorkerContractError("input_read_failed", f"Input image did not decode to HxWx3 RGB: {path}", stage="input_load")
    return rgb


def _save_alpha_hint(alpha: Any, *, output_dir: Path, staging_dir: Path) -> Path:
    try:
        import numpy as np
        from PIL import Image

        alpha_arr = np.asarray(alpha)
        if alpha_arr.ndim != 2:
            raise ValueError(f"alpha hint must be HxW, got shape {alpha_arr.shape}")
        if alpha_arr.dtype != np.uint8:
            alpha_arr = np.clip(alpha_arr, 0, 255).astype(np.uint8)
        token = _file_token()
        staged = staging_dir / f"birefnet_alpha_hint_{token}.png"
        final = output_dir / staged.name
        Image.fromarray(alpha_arr, mode="L").save(staged)
        shutil.move(str(staged), str(final))
        return final
    except Exception as exc:
        raise WorkerContractError(
            "alpha_write_failed",
            f"BiRefNet alpha hint could not be written: {type(exc).__name__}: {exc}",
            stage="output_write",
        ) from exc


def _write_diagnostics(response: dict[str, Any], diagnostics: dict[str, Any], output_dir: Path) -> None:
    try:
        path = output_dir / f"birefnet_worker_diagnostics_{_file_token()}.json"
        response["diagnostics_path"] = str(path)
        payload = {"response": response, "diagnostics": diagnostics}
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        response["diagnostics_path"] = None
        response["diagnostics_error"] = f"diagnostics write failed: {type(exc).__name__}: {exc}"


def _cleanup_staging_dir(staging_dir: Path | None) -> dict[str, Any] | None:
    if staging_dir is None:
        return None
    result = {"staging_dir": str(staging_dir), "removed": False, "error": None}
    try:
        shutil.rmtree(staging_dir, ignore_errors=False)
        result["removed"] = True
    except FileNotFoundError:
        result["removed"] = True
    except Exception as exc:  # pragma: no cover - platform-dependent cleanup edge
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _append_progress(events: list[dict[str, Any]], fraction: float, message: str) -> None:
    events.append({"time_utc": _utc_timestamp(), "fraction": float(fraction), "message": str(message)})


def _cancel_requested(cancel_file: Path | None) -> bool:
    return bool(cancel_file and cancel_file.exists())


def _classify_adapter_error(result: dict[str, Any]) -> str:
    error = result.get("error") if isinstance(result, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    message = str(result.get("message") or (error.get("message") if isinstance(error, dict) else "")).lower()
    if code:
        if _looks_like_oom(message):
            return "out_of_memory"
        return str(code)
    if _looks_like_oom(message):
        return "out_of_memory"
    if "cuda" in message and ("unavailable" in message or "is_available" in message):
        return "device_unavailable"
    if "import" in message or "dependency" in message:
        return "dependency_unavailable"
    if "cancel" in message:
        return "cancelled"
    return "inference_failed"


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if _looks_like_oom(text):
        return "out_of_memory"
    if "cuda" in text and ("unavailable" in text or "is_available" in text):
        return "device_unavailable"
    return "worker_exception"


def _looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "cublas_status_alloc_failed" in lowered


def _looks_like_url(text: str) -> bool:
    return text.startswith(("http://", "https://", "hf://", "hf_hub://"))


def _file_token() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}-{uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def format_human_response(response: dict[str, Any]) -> str:
    status = "ok" if response.get("ok") else "error"
    lines = [f"AI worker {status}: {response.get('message', '')}"]
    if response.get("alpha_hint_path"):
        lines.append(f"alpha_hint_path: {response['alpha_hint_path']}")
    if response.get("diagnostics_path"):
        lines.append(f"diagnostics_path: {response['diagnostics_path']}")
    if response.get("error"):
        error = response["error"]
        lines.append(f"error_code: {error.get('code')} stage={error.get('stage')}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated ImgKey AI worker request.")
    parser.add_argument("--request", required=True, help="request JSON file path, '-' for stdin, or inline JSON object")
    parser.add_argument("--json", action="store_true", dest="json_output", help="print the worker response as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        request = load_request(args.request)
        response = run_worker_request(request)
    except WorkerContractError as exc:
        response = _response_error(exc.code, str(exc), backend=None, stage=exc.stage, details=exc.details)
    except KeyboardInterrupt:
        response = _response_error("cancelled", "AI worker cancelled by KeyboardInterrupt.", backend=None, stage="cancel")
    except Exception as exc:  # pragma: no cover - final CLI safety net
        response = _response_error(
            _classify_exception(exc),
            f"AI worker CLI failed unexpectedly: {type(exc).__name__}: {exc}",
            backend=None,
            stage="cli",
            details={"traceback": traceback.format_exc(limit=8)},
        )

    if args.json_output:
        print(json.dumps(_json_safe(response), indent=2, sort_keys=True))
    else:
        print(format_human_response(response))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
