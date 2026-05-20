from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal

from keyer import KeyResult, KeySettings, process_key_image


@dataclass(slots=True)
class PreviewJob:
    input_rgb: np.ndarray
    original_alpha: np.ndarray | None
    keep_mask: np.ndarray | None
    remove_mask: np.ndarray | None
    alpha_hint: np.ndarray | None
    settings: KeySettings
    display_rgb: np.ndarray
    display_alpha: np.ndarray | None
    display_scale: float
    crop_origin: tuple[int, int]
    label: str


class PreviewThread(QThread):
    done = Signal(int, object)
    progress = Signal(int, float, str)
    failed = Signal(int, str)

    def __init__(self, generation: int, job: PreviewJob) -> None:
        super().__init__()
        self.generation = generation
        self.job = job
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            result = process_key_image(
                self.job.input_rgb,
                self.job.settings,
                self.job.original_alpha,
                keep_mask=self.job.keep_mask,
                remove_mask=self.job.remove_mask,
                alpha_hint=self.job.alpha_hint,
                progress_callback=lambda value, stage: None
                if self._cancel_requested
                else self.progress.emit(self.generation, value, stage),
                cancel_callback=self._is_cancelled,
            )
            if self._cancel_requested:
                return
            self.done.emit(self.generation, result)
        except Exception as exc:  # pragma: no cover - UI boundary
            if self._cancel_requested and "cancel" in str(exc).lower():
                return
            self.failed.emit(self.generation, str(exc))


class PreviewController:
    """Owns preview worker lifetime while MainWindow owns visible UI state."""

    def __init__(self, owner) -> None:
        self.owner = owner
        owner._preview_generation = 0
        owner._preview_jobs: dict[int, PreviewJob] = {}
        owner._preview_threads: list[PreviewThread] = []
        owner._preview_pending = False
        owner._preview_timer = QTimer(owner)
        owner._preview_timer.setSingleShot(True)
        owner._preview_timer.timeout.connect(owner._start_preview)

    def schedule_preview(self) -> None:
        owner = self.owner
        if owner.full_rgb is None:
            return
        owner.settings = owner.current_settings()
        owner._sync_output_mode_status()
        owner._preview_generation += 1
        owner._preview_jobs.clear()
        self.cancel_preview_threads()
        owner.statusBar().showMessage("Preview queued…")
        owner._preview_timer.start(150)

    def cancel_preview_threads(self) -> None:
        for thread in list(self.owner._preview_threads):
            if thread.isRunning():
                thread.request_cancel()

    def start_preview(self) -> None:
        owner = self.owner
        if owner.full_rgb is None:
            return
        if any(thread.isRunning() for thread in owner._preview_threads):
            owner._preview_pending = True
            return
        try:
            job = self.make_preview_job()
        except Exception as exc:
            owner.on_failed(f"Preview setup failed: {exc}")
            return
        generation = owner._preview_generation
        owner._preview_jobs[generation] = job
        owner._preview_pending = False

        if (
            owner.current_source_rgb is None
            or owner.current_source_rgb.shape != job.display_rgb.shape
            or owner.current_crop_origin != job.crop_origin
            or abs(owner.current_display_scale - job.display_scale) > 1e-6
        ):
            owner.current_result = None
            owner._set_current_source(job.display_rgb, job.display_alpha, job.display_scale, job.crop_origin, job.label)

        thread = PreviewThread(generation, job)
        owner._preview_threads.append(thread)
        thread.done.connect(owner.on_preview_done)
        thread.progress.connect(owner.on_preview_progress)
        thread.failed.connect(owner.on_preview_failed)
        thread.finished.connect(lambda t=thread: owner._forget_preview_thread(t))
        owner.statusBar().showMessage(f"Processing {job.label.lower()} preview…")
        thread.start()

    def make_preview_job(self) -> PreviewJob:
        owner = self.owner
        assert owner.full_rgb is not None
        settings = owner.current_settings()
        if owner.preview_quality.currentText() == "Full Crop":
            if owner._full_crop_rect is None:
                owner._full_crop_rect = owner._current_full_crop()
            crop = owner._full_crop_rect
            x0, y0, x1, y1 = crop
            alpha_hint = owner._processing_alpha_input(settings, owner.full_rgb.shape[:2])
            settings.full_res_crop = crop
            settings.preview_scale = 1.0
            settings.use_tiling = True
            display_rgb = owner.full_rgb[y0:y1, x0:x1].copy()
            display_alpha = None if owner.full_alpha is None else owner.full_alpha[y0:y1, x0:x1].copy()
            return PreviewJob(
                input_rgb=owner.full_rgb,
                original_alpha=owner.full_alpha,
                keep_mask=owner.keep_mask,
                remove_mask=owner.remove_mask,
                alpha_hint=alpha_hint,
                settings=settings,
                display_rgb=display_rgb,
                display_alpha=display_alpha,
                display_scale=1.0,
                crop_origin=(x0, y0),
                label=f"Full crop {x1 - x0}×{y1 - y0}",
            )

        assert owner.proxy_rgb is not None
        settings.full_res_crop = None
        owner._full_crop_rect = None
        settings.preview_scale = owner.proxy_scale
        shape = owner.proxy_rgb.shape[:2]
        alpha_hint = owner._processing_alpha_input(settings, shape)
        return PreviewJob(
            input_rgb=owner.proxy_rgb,
            original_alpha=owner.proxy_alpha,
            keep_mask=resize_mask(owner.keep_mask, shape),
            remove_mask=resize_mask(owner.remove_mask, shape),
            alpha_hint=alpha_hint,
            settings=settings,
            display_rgb=owner.proxy_rgb,
            display_alpha=owner.proxy_alpha,
            display_scale=owner.proxy_scale,
            crop_origin=(0, 0),
            label="Proxy",
        )

    def forget_preview_thread(self, thread: PreviewThread) -> None:
        owner = self.owner
        if thread in owner._preview_threads:
            owner._preview_threads.remove(thread)
        if owner._preview_pending and owner.full_rgb is not None:
            owner._preview_pending = False
            owner._preview_timer.start(0)


def resize_alpha_for_preview(alpha: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if alpha is None:
        return None
    if alpha.shape == shape:
        return alpha.copy()
    return cv2.resize(alpha, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == shape:
        return mask.copy()
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_alpha_hint_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == shape:
        return mask.copy()
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
