from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from keyer import KeySettings, process_key_image, write_png_rgba


class ExportThread(QThread):
    progress = Signal(float, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        path: str,
        rgb: np.ndarray,
        original_alpha: np.ndarray | None,
        settings: KeySettings,
        keep_mask: np.ndarray | None,
        remove_mask: np.ndarray | None,
        alpha_hint: np.ndarray | None,
    ) -> None:
        super().__init__()
        self.path = path
        self.rgb = rgb
        self.original_alpha = original_alpha
        self.settings = settings
        self.keep_mask = keep_mask
        self.remove_mask = remove_mask
        self.alpha_hint = alpha_hint
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            result = process_key_image(
                self.rgb,
                self.settings,
                self.original_alpha,
                keep_mask=self.keep_mask,
                remove_mask=self.remove_mask,
                alpha_hint=self.alpha_hint,
                progress_callback=lambda value, stage: self.progress.emit(value, stage),
                cancel_callback=lambda: self._cancel_requested,
                include_debug=False,
            )
            if self._cancel_requested:
                raise RuntimeError("Processing cancelled")
            write_png_rgba(self.path, result.rgba)
            self.done.emit(self.path)
        except Exception as exc:  # pragma: no cover - UI boundary
            self.failed.emit(str(exc))


class ExportController:
    """Owns full-resolution export worker lifetime and UI status plumbing."""

    def __init__(self, owner) -> None:
        self.owner = owner
        owner.export_thread: ExportThread | None = None

    def export_png(self) -> None:
        owner = self.owner
        if owner.full_rgb is None:
            return
        default = "output_keyed.png"
        if owner.image_path:
            default = str(owner.image_path.with_name(f"{owner.image_path.stem}_keyed.png"))
        path, _ = QFileDialog.getSaveFileName(owner, "Export PNG", default, "PNG image (*.png)")
        if not path:
            return
        settings = owner.current_settings()
        settings.full_res_crop = None
        settings.preview_scale = 1.0
        settings.use_tiling = True
        try:
            alpha_hint = owner._processing_alpha_input(settings, owner.full_rgb.shape[:2])
        except Exception as exc:
            owner.on_failed(str(exc), title="Export setup failed")
            return
        owner.export_thread = ExportThread(
            path,
            owner.full_rgb,
            owner.full_alpha,
            settings,
            owner.keep_mask,
            owner.remove_mask,
            alpha_hint,
        )
        owner.export_thread.progress.connect(owner.on_export_progress)
        owner.export_thread.done.connect(owner.on_export_done)
        owner.export_thread.failed.connect(owner.on_export_failed)
        owner.export_thread.finished.connect(owner._export_finished)
        owner.progress_bar.setValue(0)
        owner.progress_bar.show()
        owner.cancel_export_btn.show()
        owner.open_action.setEnabled(False)
        owner._update_enabled_state()
        owner.statusBar().showMessage("Export started…")
        owner.export_thread.start()

    def cancel_export(self) -> None:
        owner = self.owner
        if owner.export_thread is not None and owner.export_thread.isRunning():
            owner.export_thread.request_cancel()
            owner.statusBar().showMessage("Cancelling export…")

    def on_export_progress(self, value: float, stage: str) -> None:
        owner = self.owner
        owner.progress_bar.setValue(int(np.clip(value, 0.0, 1.0) * 100))
        owner.statusBar().showMessage(f"Export {owner.progress_bar.value()}% · {stage}")

    def on_export_done(self, path: str) -> None:
        owner = self.owner
        owner.progress_bar.setValue(100)
        owner.statusBar().showMessage(f"Saved {Path(path).name}")
        QMessageBox.information(owner, "Export complete", f"Saved transparent PNG:\n{path}")

    def on_export_failed(self, message: str) -> None:
        owner = self.owner
        if "cancel" in message.lower():
            owner.statusBar().showMessage("Export cancelled")
        else:
            if hasattr(owner, "gpu_probe_status") and owner._message_mentions_gpu_backend(message):
                owner.gpu_probe_status.setText(f"GPU Status: error. {message}")
            owner.on_failed(message, title="Export failed")

    def export_finished(self) -> None:
        owner = self.owner
        owner.progress_bar.hide()
        owner.cancel_export_btn.hide()
        owner.export_thread = None
        owner.open_action.setEnabled(True)
        owner._update_enabled_state()
