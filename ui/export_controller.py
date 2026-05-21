from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from imgkey_engine.cache import ProcessCache, ProcessCacheContext, ProcessCacheTransaction
from imgkey_engine.cache_keys import (
    reference_prep_cache_fingerprint,
    runtime_base_matte_cache_fingerprint,
    transition_alpha_cache_fingerprint,
)
from imgkey_engine.image_io import PNG_DEFAULT_COMPRESSION_LEVEL, PNG_FAST_COMPRESSION_LEVEL
from keyer import KeySettings, process_key_image, write_png_rgba


PNG_COMPRESSION_CHOICES = (
    ("Default PNG (level 6 · smaller file)", PNG_DEFAULT_COMPRESSION_LEVEL),
    ("Fast PNG (level 1 · faster save, larger file)", PNG_FAST_COMPRESSION_LEVEL),
)
PNG_COMPRESSION_TOOLTIP = (
    "Default keeps the current PNG compression. Fast PNG saves faster but creates larger lossless files."
)
PNG_PROCESS_FRACTION = 0.94
PNG_ENCODE_START = 0.95


def png_compression_label(level: int) -> str:
    level = int(level)
    for label, choice_level in PNG_COMPRESSION_CHOICES:
        if int(choice_level) == level:
            return label
    return f"PNG compression level {max(0, min(9, level))}"


def png_compression_level_from_owner(owner) -> int:
    combo = getattr(owner, "png_compression", None)
    if combo is None:
        return PNG_DEFAULT_COMPRESSION_LEVEL
    data = combo.currentData()
    if data is None:
        text = combo.currentText().lower()
        return PNG_FAST_COMPRESSION_LEVEL if "fast" in text else PNG_DEFAULT_COMPRESSION_LEVEL
    return int(data)


def format_export_process_stage(stage: str, *, cache_state: str = "cache unavailable") -> str:
    raw = str(stage or "processing").strip()
    lower = raw.lower()
    if lower.startswith("tile "):
        if "d3d12 color render" in lower:
            return f"D3D12 color render · {raw.split(' · ', 1)[0]}"
        if "gpu color render" in lower:
            return f"GPU color render · {raw.split(' · ', 1)[0]}"
        if "gpu fallback" in lower:
            return f"CPU color render (GPU fallback) · {raw.split(' · ', 1)[0]}"
        if "cpu color render" in lower:
            return f"CPU color render · {raw.split(' · ', 1)[0]}"
        return f"Color render · {raw}"
    if "cached matte" in lower or (cache_state == "matte cached" and lower in {"processing", "global matte"}):
        return "Using cached matte"
    if "transition alpha" in lower:
        return f"CPU transition alpha · {raw}"
    if any(
        token in lower
        for token in (
            "sample screen",
            "screen probability",
            "connected background",
            "trimap",
            "screen model",
            "fringe map",
            "inner color map",
            "global matte",
        )
    ):
        return f"CPU global matte · {raw}"
    return raw


def initial_export_stage_text(cache_state: str) -> str:
    if cache_state == "matte cached":
        return "Using cached matte · starting color render"
    if cache_state == "partial matte cache":
        return "Partial matte cache · finishing CPU matte prep"
    if cache_state == "cold global matte":
        return "CPU global matte · cache miss"
    return f"CPU global matte · {cache_state}"


def result_cache_stage_text(cache_info: dict | None) -> str:
    info = cache_info or {}
    if info.get("cache_hit") == "matte":
        return "Using cached matte · color render complete"
    if info.get("base_matte") == "hit":
        return "Partial matte cache · color render complete"
    if info.get("enabled"):
        reason = info.get("cache_miss_reason") or "cache miss"
        return f"CPU global matte complete · {reason}"
    return "CPU global matte complete"


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
        png_compression_level: int = PNG_DEFAULT_COMPRESSION_LEVEL,
        cache_state: str = "cache unavailable",
        process_cache: ProcessCache | None = None,
        cache_context: ProcessCacheContext | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.rgb = rgb
        self.original_alpha = original_alpha
        self.settings = settings
        self.keep_mask = keep_mask
        self.remove_mask = remove_mask
        self.alpha_hint = alpha_hint
        self.png_compression_level = int(png_compression_level)
        self.cache_state = str(cache_state)
        self.process_cache = process_cache
        self.cache_context = cache_context
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _emit_processing_progress(self, value: float, stage: str) -> None:
        self.progress.emit(
            float(np.clip(value, 0.0, 1.0)) * PNG_PROCESS_FRACTION,
            format_export_process_stage(stage, cache_state=self.cache_state),
        )

    def run(self) -> None:
        cache_transaction: ProcessCacheTransaction | None = None
        try:
            self.progress.emit(0.0, initial_export_stage_text(self.cache_state))
            if self.process_cache is not None and self.cache_context is not None:
                cache_transaction = self.process_cache.begin(self.cache_context)
            result = process_key_image(
                self.rgb,
                self.settings,
                self.original_alpha,
                keep_mask=self.keep_mask,
                remove_mask=self.remove_mask,
                alpha_hint=self.alpha_hint,
                progress_callback=self._emit_processing_progress,
                cancel_callback=lambda: self._cancel_requested,
                include_debug=False,
                cache_transaction=cache_transaction,
            )
            self.progress.emit(PNG_PROCESS_FRACTION, result_cache_stage_text(result.cache_info))
            if self._cancel_requested:
                if cache_transaction is not None:
                    cache_transaction.discard()
                raise RuntimeError("Processing cancelled")
            self.progress.emit(PNG_ENCODE_START, f"PNG encode · {png_compression_label(self.png_compression_level)}")
            write_png_rgba(self.path, result.rgba, compression_level=self.png_compression_level)
            if self._cancel_requested:
                if cache_transaction is not None:
                    cache_transaction.discard()
                raise RuntimeError("Processing cancelled")
            if cache_transaction is not None:
                cache_transaction.commit()
            self.progress.emit(1.0, "PNG encode complete")
            self.done.emit(self.path)
        except Exception as exc:  # pragma: no cover - UI boundary
            if cache_transaction is not None:
                cache_transaction.discard()
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
        png_compression_level = png_compression_level_from_owner(owner)
        cache_context = owner._cache_context("full", owner.full_rgb.shape[:2]) if hasattr(owner, "_cache_context") else None
        cache_state = self.full_export_cache_state(settings, cache_context)
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
            png_compression_level,
            cache_state,
            getattr(owner, "process_cache", None),
            cache_context,
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
        owner.statusBar().showMessage(
            f"Export started · {initial_export_stage_text(cache_state)} · "
            f"{png_compression_label(png_compression_level)}"
        )
        owner.export_thread.start()

    def full_export_cache_state(self, settings: KeySettings, cache_context: ProcessCacheContext | None = None) -> str:
        owner = self.owner
        if owner.full_rgb is None:
            return "cache idle"
        cache = getattr(owner, "process_cache", None)
        if cache is None:
            return "cache unavailable"
        if cache_context is None:
            if not hasattr(owner, "_cache_context"):
                return "cache unavailable"
            cache_context = owner._cache_context("full", owner.full_rgb.shape[:2])
        base_key = runtime_base_matte_cache_fingerprint(settings, cache_context.source_key, cache_context.mask_key)
        reference_key = reference_prep_cache_fingerprint(settings, base_key)
        transition_key = transition_alpha_cache_fingerprint(settings, base_key, reference_key)
        status = cache.matte_status(base_key=base_key, reference_key=reference_key, transition_key=transition_key)
        if status.get("transition_alpha"):
            return "matte cached"
        if status.get("base_matte") or status.get("reference_prep"):
            return "partial matte cache"
        return "cold global matte"

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
