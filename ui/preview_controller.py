from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal

from keyer import KeyResult, KeySettings, process_key_image
from imgkey_engine.cache import ProcessCache, ProcessCacheContext, ProcessCacheTransaction
from imgkey_engine.cache_keys import (
    matte_pipeline_settings_fingerprint,
    reference_prep_cache_fingerprint,
    runtime_base_matte_cache_fingerprint,
    transition_alpha_cache_fingerprint,
)


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
    process_cache: ProcessCache | None = None
    cache_context: ProcessCacheContext | None = None
    requested_mode: str = "Proxy"
    work_mode: str = "proxy"
    exact: bool = True
    cache_state: str = "cache unavailable"
    crop_rect: tuple[int, int, int, int] | None = None
    followup_exact: bool = False


@dataclass(slots=True)
class PreviewResult:
    result: KeyResult
    cache_transaction: ProcessCacheTransaction | None = None


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
        cache_transaction = None
        try:
            if self.job.process_cache is not None and self.job.cache_context is not None:
                cache_transaction = self.job.process_cache.begin(self.job.cache_context)
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
                cache_transaction=cache_transaction,
            )
            if self._cancel_requested:
                if cache_transaction is not None:
                    cache_transaction.discard()
                return
            self.done.emit(self.generation, PreviewResult(result, cache_transaction))
        except Exception as exc:  # pragma: no cover - UI boundary
            if cache_transaction is not None:
                cache_transaction.discard()
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
        owner._preview_stage = "exact"
        owner._preview_target_mode = "Proxy"
        owner._preview_draft_only = False
        owner._preview_timer = QTimer(owner)
        owner._preview_timer.setSingleShot(True)
        owner._preview_timer.timeout.connect(owner._start_preview)

    def schedule_preview(self, *, draft: bool = False, debounce_ms: int | None = None) -> None:
        owner = self.owner
        if owner.full_rgb is None:
            return
        next_settings = owner.current_settings()
        if matte_pipeline_settings_fingerprint(next_settings) != matte_pipeline_settings_fingerprint(owner.settings):
            cache = getattr(owner, "process_cache", None)
            if cache is not None:
                cache.clear()
        owner.settings = next_settings
        owner._sync_output_mode_status()
        owner._preview_target_mode = owner.preview_quality.currentText()
        owner._preview_draft_only = bool(draft)
        owner._preview_stage = self._initial_preview_stage(next_settings, draft=draft)
        owner._preview_generation += 1
        owner._preview_jobs.clear()
        owner._preview_pending = False
        self.cancel_preview_threads()
        if hasattr(owner, "_sync_preview_status"):
            owner._sync_preview_status()
        if hasattr(owner, "_update_canvas_hud"):
            owner._update_canvas_hud()
        owner.statusBar().showMessage(self._queued_status_text())
        delay = 150 if debounce_ms is None else max(0, int(debounce_ms))
        owner._preview_timer.start(delay)

    def _initial_preview_stage(self, settings: KeySettings, *, draft: bool) -> str:
        owner = self.owner
        target_mode = owner.preview_quality.currentText()
        if draft:
            return "draft"
        if target_mode == "Full Crop" and not self.full_matte_cache_ready(settings):
            return "draft"
        return "exact"

    def _queued_status_text(self) -> str:
        owner = self.owner
        target = getattr(owner, "_preview_target_mode", owner.preview_quality.currentText())
        stage = getattr(owner, "_preview_stage", "exact")
        if stage == "draft" and target == "Full Crop" and not getattr(owner, "_preview_draft_only", False):
            return "Preview queued · proxy draft first, exact crop after matte cache"
        if stage == "draft":
            return "Preview queued · proxy draft"
        if target == "Full Crop":
            return "Preview queued · exact pinned full-resolution crop"
        return "Preview queued · proxy whole image"

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

        thread = PreviewThread(generation, job)
        owner._preview_threads.append(thread)
        thread.done.connect(owner.on_preview_done)
        thread.progress.connect(owner.on_preview_progress)
        thread.failed.connect(owner.on_preview_failed)
        thread.finished.connect(lambda t=thread: owner._forget_preview_thread(t))
        owner.statusBar().showMessage(f"Processing {job.label.lower()} preview · {job.cache_state}…")
        thread.start()

    def make_preview_job(self) -> PreviewJob:
        owner = self.owner
        assert owner.full_rgb is not None
        settings = owner.current_settings()
        target_mode = getattr(owner, "_preview_target_mode", owner.preview_quality.currentText())
        stage = getattr(owner, "_preview_stage", "exact")
        if stage == "draft":
            return self._make_proxy_job(
                settings,
                requested_mode=target_mode,
                exact=False,
                label="Proxy draft" if target_mode != "Full Crop" else "Proxy draft for exact crop",
                followup_exact=target_mode == "Full Crop" and not getattr(owner, "_preview_draft_only", False),
                preserve_full_crop=True,
            )
        if target_mode == "Full Crop":
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
            cache_state = self.cache_state_for("full", owner.full_rgb.shape[:2], settings)
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
                label=f"Exact full crop {x1 - x0}×{y1 - y0} pinned",
                process_cache=getattr(owner, "process_cache", None),
                cache_context=owner._cache_context("full", owner.full_rgb.shape[:2]) if hasattr(owner, "_cache_context") else None,
                requested_mode=target_mode,
                work_mode="full_crop",
                exact=True,
                cache_state=cache_state,
                crop_rect=crop,
            )

        return self._make_proxy_job(settings, requested_mode=target_mode, exact=True)

    def _make_proxy_job(
        self,
        settings: KeySettings,
        *,
        requested_mode: str,
        exact: bool,
        label: str = "Proxy",
        followup_exact: bool = False,
        preserve_full_crop: bool = False,
    ) -> PreviewJob:
        owner = self.owner
        assert owner.proxy_rgb is not None
        settings.full_res_crop = None
        if not preserve_full_crop:
            owner._full_crop_rect = None
        settings.preview_scale = owner.proxy_scale
        shape = owner.proxy_rgb.shape[:2]
        alpha_hint = owner._processing_alpha_input(settings, shape)
        cache_state = self.cache_state_for("proxy", shape, settings)
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
            label=label,
            process_cache=getattr(owner, "process_cache", None),
            cache_context=owner._cache_context("proxy", owner.proxy_rgb.shape[:2]) if hasattr(owner, "_cache_context") else None,
            requested_mode=requested_mode,
            work_mode="proxy",
            exact=exact,
            cache_state=cache_state,
            crop_rect=None,
            followup_exact=bool(followup_exact),
        )

    def forget_preview_thread(self, thread: PreviewThread) -> None:
        owner = self.owner
        if thread in owner._preview_threads:
            owner._preview_threads.remove(thread)
        if owner._preview_pending and owner.full_rgb is not None:
            owner._preview_pending = False
            owner._preview_timer.start(0)

    def full_matte_cache_ready(self, settings: KeySettings | None = None) -> bool:
        owner = self.owner
        if owner.full_rgb is None:
            return False
        state = self.cache_state_for("full", owner.full_rgb.shape[:2], settings or owner.current_settings())
        return state == "matte cached"

    def cache_state_for_mode(self, mode: str, settings: KeySettings | None = None) -> str:
        owner = self.owner
        if owner.full_rgb is None:
            return "cache idle"
        settings = settings or owner.current_settings()
        if mode == "Full Crop":
            return self.cache_state_for("full", owner.full_rgb.shape[:2], settings)
        if owner.proxy_rgb is None:
            return "cache unavailable"
        return self.cache_state_for("proxy", owner.proxy_rgb.shape[:2], settings)

    def cache_state_for(self, resolution: str, shape: tuple[int, int], settings: KeySettings) -> str:
        owner = self.owner
        cache = getattr(owner, "process_cache", None)
        if cache is None or not hasattr(owner, "_cache_context"):
            return "cache unavailable"
        context = owner._cache_context(resolution, shape)
        base_key = runtime_base_matte_cache_fingerprint(settings, context.source_key, context.mask_key)
        reference_key = reference_prep_cache_fingerprint(settings, base_key)
        transition_key = transition_alpha_cache_fingerprint(settings, base_key, reference_key)
        status = cache.matte_status(base_key=base_key, reference_key=reference_key, transition_key=transition_key)
        if status.get("transition_alpha"):
            return "matte cached"
        if status.get("base_matte") or status.get("reference_prep"):
            return "partial matte cache"
        return "cold global matte"


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
