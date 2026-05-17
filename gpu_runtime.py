from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any


NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
MATMUL_SMOKE_SIZE = 64


def _import_torch() -> Any:
    """Import torch lazily; never call this at module import time."""
    return importlib.import_module("torch")


def _base_probe() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "imgkey_gpu_runtime",
        "status": "unavailable",
        "available": False,
        "reason": "not_run",
        "message": "GPU probe has not run.",
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "import_success": False,
            "import_error": None,
            "version": None,
            "cuda_version": None,
        },
        "cuda": {
            "is_available": False,
            "availability_error": None,
            "device_count": 0,
            "current_device": None,
            "device_name": None,
            "device_capability": None,
            "arch_list": [],
            "vram_total_bytes": None,
            "vram_free_bytes": None,
            "device_properties": None,
        },
        "nvidia_smi": {
            "available": False,
            "path": None,
            "error": None,
            "driver_version": None,
            "cuda_version": None,
            "gpus": [],
        },
        "matmul_smoke": {
            "ran": False,
            "ok": False,
            "error": None,
            "device": None,
            "size": MATMUL_SMOKE_SIZE,
            "mean": None,
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


def _torch_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _device_properties_dict(props: Any) -> dict[str, Any]:
    total_memory = _torch_attr(props, "total_memory")
    return {
        "name": _torch_attr(props, "name"),
        "major": _torch_attr(props, "major"),
        "minor": _torch_attr(props, "minor"),
        "multi_processor_count": _torch_attr(props, "multi_processor_count"),
        "total_memory_bytes": int(total_memory) if total_memory is not None else None,
    }


def _run_cuda_matmul_smoke(torch_module: Any, *, device_index: int, size: int) -> dict[str, Any]:
    smoke = {
        "ran": True,
        "ok": False,
        "error": None,
        "device": f"cuda:{device_index}",
        "size": int(size),
        "mean": None,
    }
    try:
        device = torch_module.device(f"cuda:{device_index}") if hasattr(torch_module, "device") else f"cuda:{device_index}"
        no_grad = torch_module.no_grad() if hasattr(torch_module, "no_grad") else _null_context()
        with no_grad:
            x = torch_module.randn((int(size), int(size)), device=device)
            y = x @ x
            smoke["mean"] = float(y.mean().item())
        torch_module.cuda.synchronize(device)
    except Exception as exc:
        smoke["error"] = f"{type(exc).__name__}: {exc}"
    else:
        smoke["ok"] = True
    finally:
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
    return smoke


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _torch_cuda_summary(result: dict[str, Any], torch_module: Any, *, run_matmul: bool, matmul_size: int) -> None:
    version_obj = _torch_attr(torch_module, "version")
    result["torch"].update(
        {
            "import_success": True,
            "import_error": None,
            "version": str(_torch_attr(torch_module, "__version__", "unknown")),
            "cuda_version": _torch_attr(version_obj, "cuda"),
        }
    )

    cuda = _torch_attr(torch_module, "cuda")
    if cuda is None:
        result["cuda"]["availability_error"] = "torch.cuda module is not present."
        return

    try:
        result["cuda"]["arch_list"] = list(cuda.get_arch_list()) if hasattr(cuda, "get_arch_list") else []
    except Exception as exc:
        result["cuda"]["arch_list"] = []
        result["cuda"]["availability_error"] = f"torch.cuda.get_arch_list failed: {type(exc).__name__}: {exc}"

    try:
        is_available = bool(cuda.is_available())
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.is_available failed: {type(exc).__name__}: {exc}"
        is_available = False
    result["cuda"]["is_available"] = is_available

    try:
        result["cuda"]["device_count"] = int(cuda.device_count())
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.device_count failed: {type(exc).__name__}: {exc}"
        result["cuda"]["device_count"] = 0

    if not is_available or result["cuda"]["device_count"] <= 0:
        return

    try:
        device_index = int(cuda.current_device())
    except Exception:
        device_index = 0
    result["cuda"]["current_device"] = device_index

    try:
        result["cuda"]["device_name"] = str(cuda.get_device_name(device_index))
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.get_device_name failed: {type(exc).__name__}: {exc}"

    try:
        capability = cuda.get_device_capability(device_index)
        result["cuda"]["device_capability"] = [int(capability[0]), int(capability[1])]
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.get_device_capability failed: {type(exc).__name__}: {exc}"

    try:
        props = cuda.get_device_properties(device_index)
        props_dict = _device_properties_dict(props)
        result["cuda"]["device_properties"] = props_dict
        if props_dict["total_memory_bytes"] is not None:
            result["cuda"]["vram_total_bytes"] = props_dict["total_memory_bytes"]
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.get_device_properties failed: {type(exc).__name__}: {exc}"

    try:
        free_bytes, total_bytes = cuda.mem_get_info(device_index)
        result["cuda"]["vram_free_bytes"] = int(free_bytes)
        result["cuda"]["vram_total_bytes"] = int(total_bytes)
    except TypeError:
        try:
            free_bytes, total_bytes = cuda.mem_get_info()
            result["cuda"]["vram_free_bytes"] = int(free_bytes)
            result["cuda"]["vram_total_bytes"] = int(total_bytes)
        except Exception as exc:
            result["cuda"]["availability_error"] = f"torch.cuda.mem_get_info failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        result["cuda"]["availability_error"] = f"torch.cuda.mem_get_info failed: {type(exc).__name__}: {exc}"

    if run_matmul:
        result["matmul_smoke"] = _run_cuda_matmul_smoke(torch_module, device_index=device_index, size=matmul_size)


def _unavailable_message(result: dict[str, Any], reason: str) -> str:
    smi = result.get("nvidia_smi", {})
    driver = smi.get("driver_version")
    gpu_names = [gpu.get("name") for gpu in smi.get("gpus", []) if gpu.get("name")]
    smi_hint = ""
    if gpu_names or driver:
        smi_hint = f" nvidia-smi detected {', '.join(gpu_names) or 'an NVIDIA GPU'}"
        if driver:
            smi_hint += f" with driver {driver}"
        smi_hint += "."
    elif smi.get("error"):
        smi_hint = f" {smi['error']}"

    if reason == "torch_import_failed":
        return (
            "PyTorch could not be imported, so ImgKey cannot use CUDA acceleration. "
            "Install the GPU runtime build or a CUDA-enabled PyTorch wheel that matches your NVIDIA driver."
            + smi_hint
        )
    if reason == "cuda_unavailable":
        return (
            "PyTorch imported, but torch.cuda.is_available() is false or no CUDA device was reported. "
            "Use a CUDA-enabled PyTorch build, update the NVIDIA driver, and verify the GPU with nvidia-smi."
            + smi_hint
        )
    if reason == "cuda_matmul_failed":
        return (
            "PyTorch can see CUDA, but the CUDA matmul smoke test failed. "
            "Check driver/runtime compatibility, GPU memory availability, and the installed PyTorch CUDA build."
            + smi_hint
        )
    return "GPU runtime probe did not find a usable CUDA runtime." + smi_hint


def probe_gpu(
    *,
    torch_loader: Callable[[], Any] | None = None,
    nvidia_smi_probe: Callable[[], dict[str, Any]] | None = None,
    run_matmul: bool = True,
    matmul_size: int = MATMUL_SMOKE_SIZE,
) -> dict[str, Any]:
    """Probe PyTorch/CUDA/GPU availability without importing torch at module import time."""
    result = _base_probe()
    result["nvidia_smi"] = (nvidia_smi_probe or probe_nvidia_smi)()

    try:
        torch_module = (torch_loader or _import_torch)()
    except Exception as exc:
        result["torch"]["import_error"] = f"{type(exc).__name__}: {exc}"
        return _set_unavailable(result, "torch_import_failed", _unavailable_message(result, "torch_import_failed"))

    _torch_cuda_summary(result, torch_module, run_matmul=run_matmul, matmul_size=matmul_size)

    if not result["cuda"]["is_available"] or result["cuda"]["device_count"] <= 0:
        return _set_unavailable(result, "cuda_unavailable", _unavailable_message(result, "cuda_unavailable"))

    if run_matmul and not result["matmul_smoke"]["ok"]:
        return _set_unavailable(result, "cuda_matmul_failed", _unavailable_message(result, "cuda_matmul_failed"))

    name = result["cuda"].get("device_name") or "CUDA GPU"
    return _set_available(result, f"CUDA GPU available: {name}; PyTorch CUDA matmul smoke test passed.")


def format_probe_human(result: dict[str, Any]) -> str:
    lines = [f"GPU runtime: {result['status']} - {result['message']}"]
    lines.append(f"PyTorch import: {result['torch']['import_success']} version={result['torch']['version']} cuda={result['torch']['cuda_version']}")
    lines.append(
        "CUDA: "
        f"available={result['cuda']['is_available']} "
        f"device_count={result['cuda']['device_count']} "
        f"device={result['cuda']['device_name']} "
        f"capability={result['cuda']['device_capability']}"
    )
    smi = result.get("nvidia_smi", {})
    lines.append(f"nvidia-smi: available={smi.get('available')} driver={smi.get('driver_version')} cuda={smi.get('cuda_version')}")
    smoke = result.get("matmul_smoke", {})
    lines.append(f"matmul smoke: ran={smoke.get('ran')} ok={smoke.get('ok')} error={smoke.get('error')}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe ImgKey GPU/PyTorch CUDA runtime availability.")
    parser.add_argument("--probe", "--gpu-probe", action="store_true", help="run the GPU runtime probe")
    parser.add_argument("--json", action="store_true", dest="json_output", help="print probe result as JSON")
    parser.add_argument("--matmul-size", type=int, default=MATMUL_SMOKE_SIZE, help="CUDA matmul smoke-test matrix size")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.probe:
        parser.print_help()
        return 2

    try:
        result = probe_gpu(matmul_size=max(1, int(args.matmul_size)))
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
