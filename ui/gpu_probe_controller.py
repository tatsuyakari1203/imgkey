from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QMessageBox


APP_DIR = Path(__file__).resolve().parents[1]
FROZEN_APP = bool(getattr(sys, "frozen", False))
WRITABLE_APP_DIR = Path(sys.executable).resolve().parent if FROZEN_APP else APP_DIR


def gpu_probe_subprocess_command() -> tuple[str, list[str]]:
    if FROZEN_APP:
        return sys.executable, ["--gpu-probe", "--json"]
    return sys.executable, ["-m", "gpu_runtime", "--probe", "--json"]


class GPUProbeController:
    """Owns the lazy GPU probe subprocess boundary."""

    def __init__(self, owner) -> None:
        self.owner = owner
        owner.gpu_probe_process: QProcess | None = None
        owner.last_gpu_probe: dict | None = None

    def show_gpu_status(self, checked: bool = False) -> None:
        del checked
        owner = self.owner
        if self.gpu_probe_running():
            owner.statusBar().showMessage("GPU status probe already running…")
            return
        process = QProcess(owner)
        process.setObjectName("GPUStatusProcess")
        process.setWorkingDirectory(str(WRITABLE_APP_DIR))
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.finished.connect(lambda exit_code, exit_status, proc=process: self.on_gpu_probe_finished(proc, exit_code, exit_status))
        process.errorOccurred.connect(lambda error, proc=process: self.on_gpu_probe_error(proc, error))
        owner.gpu_probe_process = process
        if hasattr(owner, "gpu_probe_status"):
            owner.gpu_probe_status.setText("GPU Status: running probe in subprocess…")
        owner.statusBar().showMessage("GPU status probe running…")
        owner._update_enabled_state()
        command, arguments = gpu_probe_subprocess_command()
        process.start(command, arguments)

    def on_gpu_probe_error(self, process: QProcess, error) -> None:
        owner = self.owner
        if process is not owner.gpu_probe_process:
            return
        if process.state() != QProcess.NotRunning:
            return
        message = process.errorString() or str(error)
        owner.gpu_probe_process = None
        if hasattr(owner, "gpu_probe_status"):
            owner.gpu_probe_status.setText(f"GPU Status: failed to start probe subprocess: {message}")
        owner.statusBar().showMessage("GPU status probe failed to start")
        owner._update_enabled_state()
        process.deleteLater()

    def on_gpu_probe_finished(self, process: QProcess, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        owner = self.owner
        stdout = process_stdout(process)
        stderr = process_stderr(process)
        if process is not owner.gpu_probe_process:
            process.deleteLater()
            return
        owner.gpu_probe_process = None
        result = json_object_from_text(stdout)
        if isinstance(result, dict):
            owner.last_gpu_probe = result
            summary = format_gpu_probe_summary(result)
            if hasattr(owner, "gpu_probe_status"):
                owner.gpu_probe_status.setText(f"GPU Status: {summary}")
            owner.statusBar().showMessage(f"GPU status: {result.get('status', 'unknown')}")
            if not owner._closing:
                QMessageBox.information(owner, "GPU Status", format_gpu_probe_details(result))
        else:
            detail = stderr.strip() or f"process exited {exit_code} status {exit_status}"
            if hasattr(owner, "gpu_probe_status"):
                owner.gpu_probe_status.setText(f"GPU Status: failed - {detail}")
            owner.statusBar().showMessage("GPU status probe failed")
        owner._update_enabled_state()
        process.deleteLater()

    def gpu_probe_running(self) -> bool:
        process = self.owner.gpu_probe_process
        return process is not None and process.state() != QProcess.NotRunning


def process_stdout(process: QProcess) -> str:
    return bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")


def process_stderr(process: QProcess) -> str:
    return bytes(process.readAllStandardError()).decode("utf-8", errors="replace")


def json_object_from_text(text: str) -> dict | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def format_gpu_probe_summary(result: dict) -> str:
    status = result.get("status", "unknown")
    message = str(result.get("message") or "")
    cuda_dll = result.get("cuda_dll") or {}
    device = (result.get("cuda") or {}).get("device_name") or cuda_dll.get("device")
    if device:
        return f"{status} · {device}. {message}"
    return f"{status}. {message}"


def format_gpu_probe_details(result: dict) -> str:
    backend = result.get("backend") or {}
    cuda_dll = result.get("cuda_dll") or {}
    cuda = result.get("cuda") or {}
    smi = result.get("nvidia_smi") or {}
    smoke = result.get("transition_repair_smoke") or {}
    return "\n".join(
        (
            f"GPU runtime: {result.get('status', 'unknown')} - {result.get('message', '')}",
            f"Backend: {backend.get('name', 'compact CUDA DLL')} ({backend.get('id', 'compact_cuda_dll')})",
            f"CUDA DLL: available={cuda_dll.get('available')} version={cuda_dll.get('version')} devices={cuda_dll.get('device_count')} path={cuda_dll.get('dll_path')}",
            f"CUDA device: available={cuda.get('is_available')} device_count={cuda.get('device_count')} device={cuda.get('device_name')} capability={cuda.get('device_capability')}",
            f"nvidia-smi: available={smi.get('available')} driver={smi.get('driver_version')} cuda={smi.get('cuda_version')}",
            f"transition repair smoke: ran={smoke.get('ran')} ok={smoke.get('ok')} max_rgb_diff={smoke.get('max_rgb_diff')} error={smoke.get('error')}",
        )
    )


def message_mentions_gpu_backend(message: str) -> bool:
    lowered = str(message).lower()
    return any(token in lowered for token in ("gpu", "cuda", "dll"))
