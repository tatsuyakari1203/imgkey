from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import gpu_backend
import native_toolchain


NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
BACKEND_ID = "compact_cuda_dll"
BACKEND_NAME = "compact CUDA DLL"
KERNEL_SMOKE_SHAPE = (32, 48)


def _base_probe() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "probe": "imgkey_gpu_runtime",
        "backend": {
            "id": BACKEND_ID,
            "name": BACKEND_NAME,
        },
        "backend_registry": {
            "schema_version": 1,
            "backends": [],
            "selected_backend": None,
            "required_capabilities": ["constant_screen", "rgb_only"],
        },
        "status": "unavailable",
        "available": False,
        "reason": "not_run",
        "message": "GPU probe has not run.",
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "cuda_dll": {
            "available": False,
            "load_success": False,
            "status": "not_run",
            "reason": "not_run",
            "message": None,
            "dll_path": None,
            "version": None,
            "device": None,
            "device_index": None,
            "device_count": 0,
            "load_error": None,
            "probe_error": None,
            "last_error": None,
        },
        "cuda": {
            "is_available": False,
            "availability_error": None,
            "device_count": 0,
            "current_device": None,
            "device_name": None,
            "device_capability": None,
            "driver_version": None,
            "cuda_version": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
        },
        "nvidia_smi": {
            "available": False,
            "path": None,
            "error": None,
            "driver_version": None,
            "cuda_version": None,
            "gpus": [],
        },
        "transition_repair_smoke": {
            "ran": False,
            "ok": False,
            "error": None,
            "shape": list(KERNEL_SMOKE_SHAPE),
            "elapsed_ms": None,
            "max_rgb_diff": None,
            "max_mask_diff": None,
        },
        "native_toolchain": {
            "schema_version": native_toolchain.TOOLCHAIN_SCHEMA_VERSION,
            "probe": "imgkey_native_gpu_toolchain",
            "status": "not_run",
            "message": "Native toolchain probe has not run.",
        },
    }


def _command_result(args: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "", "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"command timed out after {timeout_seconds:.1f}s",
        }
    except Exception as exc:  # pragma: no cover - defensive OS boundary
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": None if completed.returncode == 0 else (completed.stderr.strip() or completed.stdout.strip()),
    }


def _parse_int_mib(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def _parse_driver_versions(text: str) -> tuple[str | None, str | None]:
    driver_match = re.search(r"Driver Version:\s*([^\s|]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", text)
    return (
        driver_match.group(1) if driver_match else None,
        cuda_match.group(1) if cuda_match else None,
    )


def _parse_nvidia_smi_rows(stdout: str, fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in csv.reader(line for line in stdout.splitlines() if line.strip()):
        values = [value.strip() for value in raw_row]
        if len(values) < len(fields):
            continue
        item = dict(zip(fields, values))
        rows.append(
            {
                "name": item.get("name") or None,
                "driver_version": item.get("driver_version") or None,
                "memory_total_mib": _parse_int_mib(item.get("memory.total")),
                "memory_free_mib": _parse_int_mib(item.get("memory.free")),
                "compute_capability": item.get("compute_cap") or None,
            }
        )
    return rows


def probe_nvidia_smi(*, timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Return structured nvidia-smi driver/GPU info when the tool is present."""

    path = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "available": False,
        "path": path,
        "error": None,
        "driver_version": None,
        "cuda_version": None,
        "gpus": [],
    }
    if path is None:
        result["error"] = "nvidia-smi was not found on PATH; NVIDIA driver/GPU status could not be queried."
        return result

    summary = _command_result([path], timeout_seconds=timeout_seconds)
    if summary["ok"]:
        driver_version, cuda_version = _parse_driver_versions(summary["stdout"])
        result["driver_version"] = driver_version
        result["cuda_version"] = cuda_version

    fields = ["name", "driver_version", "memory.total", "memory.free", "compute_cap"]
    query = _command_result(
        [path, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        timeout_seconds=timeout_seconds,
    )
    if not query["ok"]:
        fields = ["name", "driver_version", "memory.total", "memory.free"]
        query = _command_result(
            [path, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            timeout_seconds=timeout_seconds,
        )

    if query["ok"]:
        result["gpus"] = _parse_nvidia_smi_rows(query["stdout"], fields)
        if result["gpus"]:
            first = result["gpus"][0]
            result["driver_version"] = result["driver_version"] or first.get("driver_version")
            result["available"] = True
            result["error"] = None
        else:
            result["error"] = "nvidia-smi ran but returned no GPU rows."
    else:
        result["error"] = query["error"] or summary.get("error") or "nvidia-smi query failed."
        result["available"] = summary["ok"]

    return result


def _set_unavailable(result: dict[str, Any], reason: str, message: str) -> dict[str, Any]:
    result["status"] = "unavailable"
    result["available"] = False
    result["reason"] = reason
    result["message"] = message
    return result


def _set_available(result: dict[str, Any], message: str) -> dict[str, Any]:
    result["status"] = "available"
    result["available"] = True
    result["reason"] = None
    result["message"] = message
    return result


def _compute_capability_tuple(value: Any) -> list[int] | None:
    if value is None:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?", str(value))
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return [major, minor]


def _populate_cuda_from_nvidia_smi(result: dict[str, Any]) -> None:
    smi = result.get("nvidia_smi") or {}
    cuda = result["cuda"]
    cuda["driver_version"] = smi.get("driver_version")
    cuda["cuda_version"] = smi.get("cuda_version")
    gpus = smi.get("gpus") or []
    if not gpus:
        return
    first = gpus[0]
    cuda["device_name"] = first.get("name")
    cuda["device_capability"] = _compute_capability_tuple(first.get("compute_capability"))
    if first.get("memory_total_mib") is not None:
        cuda["vram_total_bytes"] = int(first["memory_total_mib"]) * 1024 * 1024
    if first.get("memory_free_mib") is not None:
        cuda["vram_free_bytes"] = int(first["memory_free_mib"]) * 1024 * 1024


def _probe_cuda_dll(*, dll_path: str | None = None) -> dict[str, Any]:
    import gpu_accel

    return gpu_accel.is_available(refresh=True, dll_path=dll_path)


def _apply_cuda_dll_availability(result: dict[str, Any], availability: dict[str, Any]) -> None:
    dll = result["cuda_dll"]
    available = bool(availability.get("available"))
    dll.update(
        {
            "available": available,
            "load_success": bool(availability.get("dll_path")) or available,
            "status": availability.get("status") or ("available" if available else "unavailable"),
            "reason": availability.get("reason"),
            "message": availability.get("message"),
            "dll_path": availability.get("dll_path"),
            "version": availability.get("version"),
            "device": availability.get("device"),
            "device_index": availability.get("device_index"),
            "device_count": int(availability.get("device_count") or 0),
            "load_error": availability.get("load_error"),
            "probe_error": availability.get("probe_error"),
            "last_error": availability.get("last_error"),
        }
    )
    cuda = result["cuda"]
    cuda["is_available"] = available
    cuda["device_count"] = int(availability.get("device_count") or 0)
    cuda["current_device"] = availability.get("device_index") if available else None
    if availability.get("cuda_version"):
        cuda["cuda_version"] = availability.get("cuda_version")
    if not available:
        cuda["availability_error"] = availability.get("message")


def _apply_backend_registry(result: dict[str, Any], availability: dict[str, Any]) -> None:
    try:
        cuda_backend = gpu_backend.CudaCompatBackend(cuda_probe=lambda **_: availability)
        backends = gpu_backend.probe_backends(backends=[cuda_backend], include_cpu=True)
        selection = gpu_backend.select_backend(
            "Auto",
            {"constant_screen", "rgb_only"},
            backends=[cuda_backend],
            probed_backends=backends,
        )
        result["backend_registry"] = {
            "schema_version": 1,
            "backends": backends,
            "selected_backend": selection.as_dict(),
            "required_capabilities": ["constant_screen", "rgb_only"],
        }
    except Exception as exc:  # pragma: no cover - defensive probe isolation
        result["backend_registry"] = {
            "schema_version": 1,
            "backends": [],
            "selected_backend": {
                "mode": "Auto",
                "status": "unavailable",
                "available": False,
                "backend": None,
                "backend_name": None,
                "reason": "backend_probe_failed",
                "message": f"Backend registry probe failed: {type(exc).__name__}: {exc}",
                "required_capabilities": ["constant_screen", "rgb_only"],
                "capabilities": [],
            },
            "required_capabilities": ["constant_screen", "rgb_only"],
        }


def _apply_native_toolchain_probe(result: dict[str, Any]) -> None:
    try:
        result["native_toolchain"] = native_toolchain.probe_native_toolchain()
    except Exception as exc:  # pragma: no cover - defensive probe isolation
        result["native_toolchain"] = {
            "schema_version": native_toolchain.TOOLCHAIN_SCHEMA_VERSION,
            "probe": "imgkey_native_gpu_toolchain",
            "status": "error",
            "message": f"Native toolchain probe failed: {type(exc).__name__}: {exc}",
        }


def _smi_hint(result: dict[str, Any]) -> str:
    smi = result.get("nvidia_smi", {})
    driver = smi.get("driver_version")
    gpu_names = [gpu.get("name") for gpu in smi.get("gpus", []) if gpu.get("name")]
    if gpu_names or driver:
        hint = f" nvidia-smi detected {', '.join(gpu_names) or 'an NVIDIA GPU'}"
        if driver:
            hint += f" with driver {driver}"
        return hint + "."
    if smi.get("error"):
        return f" {smi['error']}"
    return ""


def _unavailable_message(result: dict[str, Any], reason: str) -> str:
    dll_message = (result.get("cuda_dll") or {}).get("message")
    if dll_message:
        return str(dll_message) + _smi_hint(result)
    if reason == "cuda_dll_smoke_failed":
        return "Compact CUDA DLL loaded, but the transition repair kernel smoke test failed." + _smi_hint(result)
    if reason == "cuda_no_device":
        return "Compact CUDA DLL loaded, but it reported no CUDA devices. CPU color path will be used." + _smi_hint(result)
    return "Compact CUDA DLL backend is unavailable. CPU color path will be used." + _smi_hint(result)


def _transition_smoke_fixture(shape: tuple[int, int]) -> tuple[Any, ...]:
    import numpy as np

    h, w = shape
    key_color = (30, 80, 235)
    foreground_color = (226, 28, 20)
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :] = np.asarray(key_color, dtype=np.uint8)
    foreground = np.zeros((h, w, 3), dtype=np.uint8)
    foreground[:, :] = np.asarray(foreground_color, dtype=np.uint8)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    y = np.linspace(-0.10, 0.10, h, dtype=np.float32).reshape(h, 1)
    alpha_f = np.clip((x + y - 0.10) / 0.78, 0.0, 1.0)
    alpha_u8 = np.rint(alpha_f * 255.0).astype(np.uint8)
    rgb = np.clip(np.rint(background.astype(np.float32) * (1.0 - alpha_f[:, :, None]) + foreground.astype(np.float32) * alpha_f[:, :, None]), 0, 255).astype(np.uint8)
    background_mask = alpha_u8 == 0
    edge_mask = (alpha_u8 > 0) & (alpha_u8 < 255)
    probability = np.rint((1.0 - alpha_f) * 255.0).astype(np.uint8)
    fringe = np.where(edge_mask, 180, 0).astype(np.uint8)
    foreground_valid = np.ascontiguousarray((alpha_u8 > 0).astype(np.uint8) * 255)
    settings = SimpleNamespace(
        clip_foreground=0.0,
        transition_alpha_min=2,
        transition_alpha_max=253,
        transition_spill_threshold=0.08,
        transition_reconstruction_error=0.08,
        foreground_reference_pull=0.65,
        key_vector_despill=0.75,
        preserve_foreground_luma=0.85,
    )
    return rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, foreground_valid, key_color, settings


def _run_transition_repair_smoke(*, dll_path: str | None = None) -> dict[str, Any]:
    import numpy as np
    import gpu_accel

    smoke = {
        "ran": True,
        "ok": False,
        "error": None,
        "shape": list(KERNEL_SMOKE_SHAPE),
        "elapsed_ms": None,
        "max_rgb_diff": None,
        "max_mask_diff": None,
    }
    try:
        rgb, alpha_u8, background_mask, edge_mask, probability, fringe, foreground, foreground_valid, key_color, settings = _transition_smoke_fixture(KERNEL_SMOKE_SHAPE)
        transition_strength = gpu_accel.transition_repair_strength_mask_v1(
            rgb,
            alpha_u8,
            background_mask,
            edge_mask,
            probability,
            fringe,
            foreground,
            foreground_valid > 0,
            key_color,
            settings,
        )
        cpu_rgb, cpu_mask = gpu_accel.transition_repair_cpu_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid, key_color, settings)
        start = time.perf_counter()
        dll_rgb, dll_mask = gpu_accel.transition_repair_dll_v1(rgb, alpha_u8, transition_strength, foreground, foreground_valid, key_color, settings, dll_path=dll_path)
        smoke["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        smoke["max_rgb_diff"] = int(np.max(np.abs(cpu_rgb.astype(np.int16) - dll_rgb.astype(np.int16))))
        smoke["max_mask_diff"] = int(np.max(np.abs(cpu_mask.astype(np.int16) - dll_mask.astype(np.int16))))
        smoke["ok"] = smoke["max_rgb_diff"] <= 2 and smoke["max_mask_diff"] <= 2
        if not smoke["ok"]:
            smoke["error"] = f"parity exceeded tolerance: rgb={smoke['max_rgb_diff']} mask={smoke['max_mask_diff']}"
    except Exception as exc:
        smoke["error"] = f"{type(exc).__name__}: {exc}"
    return smoke


def probe_gpu(
    *,
    gpu_accel_probe: Callable[..., dict[str, Any]] | None = None,
    nvidia_smi_probe: Callable[[], dict[str, Any]] | None = None,
    run_kernel_smoke: bool = True,
    dll_path: str | None = None,
) -> dict[str, Any]:
    """Probe compact CUDA DLL availability without importing optional heavy runtimes."""

    result = _base_probe()
    result["nvidia_smi"] = (nvidia_smi_probe or probe_nvidia_smi)()
    _populate_cuda_from_nvidia_smi(result)

    try:
        availability = (gpu_accel_probe or _probe_cuda_dll)(dll_path=dll_path)
    except Exception as exc:
        availability = {
            "available": False,
            "status": "unavailable",
            "reason": "cuda_dll_probe_failed",
            "message": f"Compact CUDA DLL probe failed: {type(exc).__name__}: {exc}. CPU color path will be used.",
            "probe_error": f"{type(exc).__name__}: {exc}",
        }

    _apply_cuda_dll_availability(result, availability)
    _apply_backend_registry(result, availability)
    _apply_native_toolchain_probe(result)
    if not result["cuda_dll"]["available"]:
        reason = str(result["cuda_dll"].get("reason") or "cuda_dll_unavailable")
        return _set_unavailable(result, reason, _unavailable_message(result, reason))

    if run_kernel_smoke:
        result["transition_repair_smoke"] = _run_transition_repair_smoke(dll_path=dll_path)
        if not result["transition_repair_smoke"].get("ok"):
            return _set_unavailable(result, "cuda_dll_smoke_failed", _unavailable_message(result, "cuda_dll_smoke_failed"))

    dll = result["cuda_dll"]
    device = result["cuda"].get("device_name") or dll.get("device") or "CUDA device"
    version = dll.get("version")
    count = int(dll.get("device_count") or 0)
    return _set_available(result, f"Compact CUDA DLL available: {device}; DLL version {version}; {count} CUDA device(s) visible.")


def format_probe_human(result: dict[str, Any]) -> str:
    lines = [f"GPU runtime: {result['status']} - {result['message']}"]
    dll = result.get("cuda_dll", {})
    lines.append(
        "CUDA DLL: "
        f"available={dll.get('available')} "
        f"version={dll.get('version')} "
        f"device_count={dll.get('device_count')} "
        f"path={dll.get('dll_path')}"
    )
    cuda = result.get("cuda", {})
    lines.append(
        "CUDA device: "
        f"available={cuda.get('is_available')} "
        f"device_count={cuda.get('device_count')} "
        f"device={cuda.get('device_name')} "
        f"capability={cuda.get('device_capability')}"
    )
    smi = result.get("nvidia_smi", {})
    lines.append(f"nvidia-smi: available={smi.get('available')} driver={smi.get('driver_version')} cuda={smi.get('cuda_version')}")
    smoke = result.get("transition_repair_smoke", {})
    lines.append(f"transition repair smoke: ran={smoke.get('ran')} ok={smoke.get('ok')} max_rgb_diff={smoke.get('max_rgb_diff')} error={smoke.get('error')}")
    registry = result.get("backend_registry", {})
    selected = registry.get("selected_backend") or {}
    backends = registry.get("backends") or []
    lines.append(
        "backend registry: "
        f"selected={selected.get('backend')} status={selected.get('status')} "
        f"available={selected.get('available')} count={len(backends)}"
    )
    toolchain = result.get("native_toolchain", {})
    decision = toolchain.get("packaging_decision") or {}
    lines.append(
        "native toolchain: "
        f"status={toolchain.get('status')} one_exe={decision.get('status')} "
        f"approved={decision.get('approved')}"
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe ImgKey compact CUDA DLL runtime availability.")
    parser.add_argument("--probe", "--gpu-probe", action="store_true", help="run the GPU runtime probe")
    parser.add_argument("--json", action="store_true", dest="json_output", help="print probe result as JSON")
    parser.add_argument("--dll", dest="dll_path", default=None, help="override imgkey_cuda.dll path for the probe")
    parser.add_argument("--no-kernel-smoke", action="store_true", help="skip the transition repair kernel smoke test")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.probe:
        parser.print_help()
        return 2

    try:
        result = probe_gpu(run_kernel_smoke=not bool(args.no_kernel_smoke), dll_path=args.dll_path)
    except Exception as exc:  # pragma: no cover - final CLI safety net
        result = _base_probe()
        result["status"] = "error"
        result["reason"] = "probe_exception"
        result["message"] = f"GPU probe failed unexpectedly: {type(exc).__name__}: {exc}"

    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_probe_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
