from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QProcess, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QCursor, QImage, QMouseEvent, QPainter, QPalette, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QSplashScreen,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from keyer import (
    KeyResult,
    KeySettings,
    checkerboard_composite,
    process_key_image,
    read_grayscale_mask,
    read_imported_matte_mask,
    read_image_rgb,
    resize_for_preview,
    write_grayscale_mask,
    write_png_rgba,
)


VIEW_MODES = (
    "Result",
    "Source",
    "Alpha",
    "Imported Matte",
    "Background Mask",
    "Edge Mask",
    "Fringe Mask",
    "Despill Mask",
    "Foreground RGB",
    "Split Compare",
)
BACKGROUND_MODES = ("Checkerboard", "Black", "White", "Gray", "Transparent")
OUTPUT_MODES = ("Classical", "Imported Matte")
APP_DIR = Path(__file__).resolve().parent
FROZEN_APP = bool(getattr(sys, "frozen", False))
WRITABLE_APP_DIR = Path(sys.executable).resolve().parent if FROZEN_APP else APP_DIR
APP_DEFAULT_KEY_MODE = "Blue"
APP_DEFAULT_EDGE_RADIUS = 32
APP_DEFAULT_SETTINGS = KeySettings(
    key_color=(30, 80, 235),
    tolerance=0.45,
    softness=0.01,
    edge_blur=(APP_DEFAULT_EDGE_RADIUS - 1) / 4.0,
    cleanup=0,
    despill=0.70,
    sample_size=10,
    auto_border_sample=True,
    auto_detect_key_color=False,
    clip_background=0.97,
    clip_foreground=0.00,
    matte_gamma=2.20,
    core_strength=0.38,
    edge_refine_radius=APP_DEFAULT_EDGE_RADIUS,
    edge_softness=0.00,
    erode_expand=-8,
    despeckle_min_area=0,
    aggressive_interior_removal=True,
    decontaminate=0.50,
    luminance_restore=0.76,
    fringe_remove=0.75,
    edge_color_repair=0.65,
    inner_color_pull=0.45,
    fringe_band_radius=3,
    transition_unmix=True,
    alpha_recover_strength=0.85,
    key_vector_despill=0.75,
    foreground_reference_pull=0.65,
)


def update_boot_splash(message: str) -> None:
    """Update PyInstaller's boot splash when the frozen onefile build has it."""

    try:
        import pyi_splash  # type: ignore[import-not-found]

        pyi_splash.update_text(str(message))
    except Exception:
        pass


def close_boot_splash() -> None:
    try:
        import pyi_splash  # type: ignore[import-not-found]

        pyi_splash.close()
    except Exception:
        pass


def create_qt_startup_splash() -> QSplashScreen:
    pixmap = QPixmap(560, 300)
    pixmap.fill(QColor("#0B0D10"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.fillRect(0, 0, 560, 300, QColor("#0B0D10"))
    painter.fillRect(0, 0, 560, 5, QColor("#47D18C"))
    painter.setPen(QColor("#E7ECF3"))
    title_font = painter.font()
    title_font.setPointSize(28)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.drawText(34, 88, "ImgKey")
    subtitle_font = painter.font()
    subtitle_font.setPointSize(12)
    subtitle_font.setBold(False)
    painter.setFont(subtitle_font)
    painter.setPen(QColor("#AAB4C2"))
    painter.drawText(36, 122, "Large image chroma keyer")
    painter.drawText(36, 156, "Loading interface and controls…")
    painter.setPen(QColor("#47D18C"))
    painter.drawRoundedRect(36, 196, 488, 12, 6, 6)
    painter.fillRect(38, 198, 180, 8, QColor("#47D18C"))
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    splash.showMessage("Starting ImgKey…", Qt.AlignLeft | Qt.AlignBottom, QColor("#E7ECF3"))
    return splash


def app_default_settings() -> KeySettings:
    return replace(APP_DEFAULT_SETTINGS)


def gpu_probe_subprocess_command() -> tuple[str, list[str]]:
    if FROZEN_APP:
        return sys.executable, ["--gpu-probe", "--json"]
    return sys.executable, ["-m", "gpu_runtime", "--probe", "--json"]


def dispatch_headless_cli(argv: list[str]) -> int | None:
    args = list(argv[1:])
    if not args:
        return None

    if "--gpu-probe" in args or "--imgkey-gpu-probe" in args:
        probe_args = [arg for arg in args if arg != "--imgkey-gpu-probe"]
        if "--gpu-probe" not in probe_args and "--probe" not in probe_args:
            probe_args.insert(0, "--gpu-probe")
        from gpu_runtime import main as gpu_runtime_main

        return gpu_runtime_main(probe_args)

    return None


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


class ImageCanvas(QGraphicsView):
    sampled = Signal(int, int)
    cursor_changed = Signal(int, int)
    cursor_left = Signal()
    view_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ImageCanvas")
        self.setMinimumSize(520, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.source_rgb: np.ndarray | None = None
        self.result: KeyResult | None = None
        self.display_scale = 1.0
        self.crop_origin = (0, 0)
        self.view_mode = "Result"
        self.background_mode = "Checkerboard"
        self.persistent_tool = "Pan"
        self.picker_enabled = False
        self.space_pan_active = False
        self._fit_mode = True
        self._has_pixmap = False
        self._picker_cursor = self._build_picker_cursor()
        self._apply_background()
        self._apply_tool_state()

    def set_images(
        self,
        source_rgb: np.ndarray | None,
        result: KeyResult | None,
        *,
        display_scale: float,
        crop_origin: tuple[int, int],
        reset_view: bool = True,
    ) -> None:
        self.source_rgb = source_rgb
        self.result = result
        self.display_scale = max(float(display_scale), 1e-6)
        self.crop_origin = crop_origin
        self._refresh_pixmap(reset_view=reset_view)

    def set_result(self, result: KeyResult | None) -> None:
        self.result = result
        self._refresh_pixmap(reset_view=False)

    def set_view_mode(self, mode: str) -> None:
        if mode not in VIEW_MODES:
            return
        self.view_mode = mode
        self._refresh_pixmap(reset_view=False)

    def set_background_mode(self, mode: str) -> None:
        if mode not in BACKGROUND_MODES:
            return
        if mode == self.background_mode:
            return
        self.background_mode = mode
        self._apply_background()
        if self._display_depends_on_background():
            self._refresh_pixmap(reset_view=False)
        else:
            self.viewport().update()

    def _display_depends_on_background(self) -> bool:
        return self.view_mode == "Split Compare"

    def set_picker_enabled(self, enabled: bool) -> None:
        self.persistent_tool = "Pick" if enabled else "Pan"
        self.picker_enabled = enabled
        self._apply_tool_state()

    def set_space_pan_active(self, active: bool) -> None:
        if self.space_pan_active == active:
            return
        self.space_pan_active = active
        self._apply_tool_state()

    def fit_to_view(self) -> None:
        if not self._has_pixmap:
            return
        self._fit_mode = True
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.view_changed.emit()

    def set_100_percent(self) -> None:
        if not self._has_pixmap:
            return
        self._fit_mode = False
        self.resetTransform()
        self.view_changed.emit()

    def visible_image_rect(self) -> tuple[int, int, int, int] | None:
        if not self._has_pixmap:
            return None
        polygon = self.mapToScene(self.viewport().rect())
        rect = polygon.boundingRect().intersected(self._pixmap_item.sceneBoundingRect())
        if rect.isEmpty():
            return None
        return self._clamped_rect(rect)

    def _refresh_pixmap(self, *, reset_view: bool) -> None:
        preserve_view = not reset_view and self._has_pixmap
        old_transform = self.transform() if preserve_view else None
        old_center = self.mapToScene(self.viewport().rect().center()) if preserve_view else None
        old_size = self._pixmap_item.pixmap().size() if self._has_pixmap else QSize()
        image = self._display_image()
        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            self._has_pixmap = False
            self.view_changed.emit()
            return
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self._has_pixmap = True
        if reset_view or not preserve_view or (self._fit_mode and old_size != pixmap.size()):
            QTimer.singleShot(0, self.fit_to_view)
        elif old_transform is not None and old_center is not None:
            self.setTransform(old_transform)
            self.centerOn(old_center)
            self.view_changed.emit()

    def _apply_tool_state(self) -> None:
        if self.space_pan_active or self.persistent_tool == "Pan":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().setCursor(self._picker_cursor)

    def _build_picker_cursor(self) -> QCursor:
        try:
            size = 25
            center = size // 2
            pixmap = QPixmap(size, size)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QColor(8, 12, 18))
            painter.drawEllipse(center - 5, center - 5, 10, 10)
            painter.drawLine(center, 1, center, 8)
            painter.drawLine(center, size - 9, center, size - 2)
            painter.drawLine(1, center, 8, center)
            painter.drawLine(size - 9, center, size - 2, center)
            painter.setPen(QColor(245, 248, 255))
            painter.drawEllipse(center - 4, center - 4, 8, 8)
            painter.drawLine(center, 2, center, 8)
            painter.drawLine(center, size - 9, center, size - 3)
            painter.drawLine(2, center, 8, center)
            painter.drawLine(size - 9, center, size - 3, center)
            painter.end()
            if not pixmap.isNull():
                return QCursor(pixmap, center, center)
        except Exception:
            pass
        return QCursor(Qt.CrossCursor)

    def _display_image(self) -> QImage | None:
        if self.source_rgb is None:
            return None
        mode = self.view_mode
        result = self.result
        if mode == "Source" or result is None:
            return rgb_to_qimage(self.source_rgb)
        if mode == "Result":
            return rgba_to_qimage(result.rgba)
        if mode == "Alpha":
            return mask_to_qimage(result.alpha, self.source_rgb.shape[:2])
        if mode == "Imported Matte":
            return mask_to_qimage(result.alpha_hint, self.source_rgb.shape[:2])
        if mode == "Background Mask":
            return mask_to_qimage(result.background_mask, self.source_rgb.shape[:2])
        if mode == "Edge Mask":
            return mask_to_qimage(result.edge_mask, self.source_rgb.shape[:2])
        if mode == "Fringe Mask":
            return mask_to_qimage(result.fringe_mask, self.source_rgb.shape[:2])
        if mode == "Despill Mask":
            return mask_to_qimage(result.despill_mask, self.source_rgb.shape[:2])
        if mode == "Foreground RGB":
            foreground_rgb = result.foreground_rgb
            if foreground_rgb is None:
                foreground_rgb = result.foreground
            if foreground_rgb is None:
                foreground_rgb = result.rgba[:, :, :3]
            return rgb_to_qimage(debug_rgb_to_rgb(foreground_rgb, self.source_rgb.shape[:2]))
        if mode == "Split Compare":
            return rgb_to_qimage(self._split_compare(result))
        return rgb_to_qimage(self.source_rgb)

    def _split_compare(self, result: KeyResult) -> np.ndarray:
        source = self.source_rgb if self.source_rgb is not None else result.rgba[:, :, :3]
        out = source.copy()
        comp = composite_rgba_for_mode(result.rgba, self.background_mode)
        split_x = out.shape[1] // 2
        out[:, split_x:] = comp[:, split_x:]
        out[:, max(0, split_x - 1) : min(out.shape[1], split_x + 1)] = np.array([79, 140, 255], dtype=np.uint8)
        return out

    def _apply_background(self) -> None:
        if self.background_mode == "Checkerboard":
            self._scene.setBackgroundBrush(checker_brush())
        elif self.background_mode == "Black":
            self._scene.setBackgroundBrush(QBrush(QColor("#000000")))
        elif self.background_mode == "White":
            self._scene.setBackgroundBrush(QBrush(QColor("#ffffff")))
        elif self.background_mode == "Gray":
            self._scene.setBackgroundBrush(QBrush(QColor("#777d86")))
        else:
            self._scene.setBackgroundBrush(QBrush(QColor("#101318")))

    def _image_pos_from_event(self, event: QMouseEvent) -> tuple[int, int] | None:
        if not self._has_pixmap:
            return None
        scene_pos = self.mapToScene(event.position().toPoint())
        item_pos = self._pixmap_item.mapFromScene(scene_pos)
        x = int(np.floor(item_pos.x()))
        y = int(np.floor(item_pos.y()))
        pixmap = self._pixmap_item.pixmap()
        if x < 0 or y < 0 or x >= pixmap.width() or y >= pixmap.height():
            return None
        return x, y

    def _clamped_rect(self, rect: QRectF) -> tuple[int, int, int, int]:
        pixmap = self._pixmap_item.pixmap()
        x0 = max(0, min(pixmap.width(), int(np.floor(rect.left()))))
        y0 = max(0, min(pixmap.height(), int(np.floor(rect.top()))))
        x1 = max(0, min(pixmap.width(), int(np.ceil(rect.right()))))
        y1 = max(0, min(pixmap.height(), int(np.ceil(rect.bottom()))))
        if x1 <= x0:
            x1 = min(pixmap.width(), x0 + 1)
        if y1 <= y0:
            y1 = min(pixmap.height(), y0 + 1)
        return x0, y0, x1, y1

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = self._image_pos_from_event(event)
        if pos is None:
            self.cursor_left.emit()
        else:
            self.cursor_changed.emit(pos[0], pos[1])
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.picker_enabled and not self.space_pan_active and event.button() == Qt.LeftButton:
            pos = self._image_pos_from_event(event)
            if pos is not None:
                self.sampled.emit(pos[0], pos[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and self.dragMode() == QGraphicsView.ScrollHandDrag:
            self._fit_mode = False
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.cursor_left.emit()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode and self._has_pixmap:
            QTimer.singleShot(0, self.fit_to_view)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if not self._has_pixmap:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        current_zoom = self.transform().m11()
        factor = 1.22 if delta > 0 else 1 / 1.22
        next_zoom = current_zoom * factor
        if next_zoom < 0.04 or next_zoom > 32.0:
            return
        self._fit_mode = False
        self.scale(factor, factor)
        self.view_changed.emit()
        event.accept()


class SliderRow(QWidget):
    value_changed = Signal()

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        default: float,
        *,
        step: float,
        decimals: int = 2,
        integer: bool = False,
    ) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.default = int(default) if integer else float(default)
        self.step = float(step)
        self.decimals = decimals
        self.integer = integer
        self._steps = max(1, int(round((self.maximum - self.minimum) / self.step)))

        root = QGridLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setHorizontalSpacing(8)
        root.setVerticalSpacing(6)
        self.label = QLabel(title)
        self.label.setObjectName("ControlLabel")
        root.addWidget(self.label, 0, 0, 1, 3)

        self.value_label = QLabel()
        self.value_label.setObjectName("SliderValueLabel")
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        root.addWidget(self.value_label, 0, 3, 1, 2)

        nudge_tooltip = "Adjust by one step. Hold Ctrl or Shift while clicking for 10× step."
        self.minus_btn = QPushButton("−")
        self.minus_btn.setObjectName("StepButton")
        self.minus_btn.setFixedWidth(26)
        self.minus_btn.setToolTip(nudge_tooltip)
        root.addWidget(self.minus_btn, 1, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        root.addWidget(self.slider, 1, 1)

        self.plus_btn = QPushButton("+")
        self.plus_btn.setObjectName("StepButton")
        self.plus_btn.setFixedWidth(26)
        self.plus_btn.setToolTip(nudge_tooltip)
        root.addWidget(self.plus_btn, 1, 2)

        if integer:
            self.spin = QSpinBox()
            self.spin.setRange(int(round(self.minimum)), int(round(self.maximum)))
            self.spin.setSingleStep(max(1, int(round(self.step))))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(self.minimum, self.maximum)
            self.spin.setSingleStep(self.step)
            self.spin.setDecimals(decimals)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(82)
        root.addWidget(self.spin, 1, 3)

        self.reset_btn = QPushButton("↺")
        self.reset_btn.setObjectName("ResetButton")
        self.reset_btn.setFixedWidth(28)
        self.reset_btn.setToolTip("Reset to tuned default")
        root.addWidget(self.reset_btn, 1, 4)
        root.setColumnStretch(1, 1)

        self.minus_btn.clicked.connect(lambda: self.nudge(-1))
        self.plus_btn.clicked.connect(lambda: self.nudge(1))
        self.slider.valueChanged.connect(self._from_slider_changed)
        self.spin.valueChanged.connect(self._from_spin_changed)
        self.reset_btn.clicked.connect(self.reset)
        self.set_value(self.default, emit=False)

    def value(self) -> int | float:
        value = self.spin.value()
        return int(round(value)) if self.integer else float(value)

    def set_value(self, value: float, *, emit: bool = True) -> None:
        slider_pos = self._value_to_pos(value)
        value = self._pos_to_value(slider_pos)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(slider_pos)
        self.spin.setValue(int(round(value)) if self.integer else float(value))
        self._update_value_label(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        if emit:
            self.value_changed.emit()

    def reset(self) -> None:
        self.set_value(self.default)

    def nudge(self, direction: int) -> None:
        modifiers = QApplication.keyboardModifiers()
        multiplier = 10 if modifiers & (Qt.ControlModifier | Qt.ShiftModifier) else 1
        self.set_value(float(self.value()) + (self.step * multiplier * (1 if direction >= 0 else -1)))

    def _from_slider_changed(self, position: int) -> None:
        value = self._pos_to_value(position)
        self.spin.blockSignals(True)
        self.spin.setValue(int(round(value)) if self.integer else value)
        self._update_value_label(value)
        self.spin.blockSignals(False)
        self.value_changed.emit()

    def _from_spin_changed(self, value: float) -> None:
        slider_pos = self._value_to_pos(float(value))
        value = self._pos_to_value(slider_pos)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(slider_pos)
        self.spin.setValue(int(round(value)) if self.integer else float(value))
        self._update_value_label(value)
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        self.value_changed.emit()

    def _clamp(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, float(value)))

    def _value_to_pos(self, value: float) -> int:
        return max(0, min(self._steps, int(round((self._clamp(value) - self.minimum) / self.step))))

    def _pos_to_value(self, position: int) -> float:
        value = self.minimum + max(0, min(self._steps, position)) * self.step
        value = self._clamp(value)
        if self.integer:
            return float(int(round(value)))
        return round(value, self.decimals)

    def _format_value(self, value: float) -> str:
        if self.integer:
            return str(int(round(value)))
        return f"{float(value):.{self.decimals}f}"

    def _update_value_label(self, value: float) -> None:
        self.value_label.setText(self._format_value(value))


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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImgKey — Large Image Keyer")
        self.setAcceptDrops(True)
        self.resize(1360, 840)

        self.full_rgb: np.ndarray | None = None
        self.full_alpha: np.ndarray | None = None
        self.proxy_rgb: np.ndarray | None = None
        self.proxy_alpha: np.ndarray | None = None
        self.proxy_scale = 1.0
        self.current_source_rgb: np.ndarray | None = None
        self.current_source_alpha: np.ndarray | None = None
        self.current_result: KeyResult | None = None
        self.current_display_scale = 1.0
        self.current_crop_origin = (0, 0)
        self.current_preview_label = "Proxy"
        self._full_crop_rect: tuple[int, int, int, int] | None = None
        self.image_path: Path | None = None
        self.keep_mask: np.ndarray | None = None
        self.remove_mask: np.ndarray | None = None
        self.alpha_hint_mask: np.ndarray | None = None
        self.last_sample_rgb: tuple[int, int, int] | None = None
        self.last_cursor_rgb: tuple[int, int, int] | None = None
        self.settings = app_default_settings()

        self._preview_generation = 0
        self._preview_jobs: dict[int, PreviewJob] = {}
        self._preview_threads: list[PreviewThread] = []
        self._preview_pending = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)
        self.export_thread: ExportThread | None = None
        self.gpu_probe_process: QProcess | None = None
        self.last_gpu_probe: dict | None = None
        self._closing = False

        self._build_ui()
        self._apply_theme()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._sync_key_chip()
        self._sync_imported_matte_status()
        self._sync_output_mode_status()
        self._update_enabled_state()

    def _build_ui(self) -> None:
        self.canvas = ImageCanvas()
        self.canvas.sampled.connect(self.pick_from_canvas)
        self.canvas.cursor_changed.connect(self.update_cursor_status)
        self.canvas.cursor_left.connect(lambda: self.cursor_status.setText("Cursor —"))
        self.canvas.view_changed.connect(self._update_canvas_hud)
        canvas_area = self._build_canvas_area()
        self._build_toolbar()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.addWidget(canvas_area)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([920, 430])
        self.setCentralWidget(splitter)
        self._build_status_bar()
        self._update_canvas_hud()

    def _build_canvas_area(self) -> QWidget:
        shell = QFrame()
        shell.setObjectName("CanvasArea")
        shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QGridLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 0, 0)

        self.canvas_hud = QFrame()
        self.canvas_hud.setObjectName("CanvasHud")
        self.canvas_hud.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        hud_layout = QHBoxLayout(self.canvas_hud)
        hud_layout.setContentsMargins(10, 6, 10, 6)
        hud_layout.setSpacing(8)
        self.hud_zoom = QLabel("Zoom —")
        self.hud_zoom.setObjectName("HudText")
        self.hud_preview = QLabel("Preview Proxy")
        self.hud_preview.setObjectName("HudText")
        self.hud_hint = QLabel("Hold Space to pan")
        self.hud_hint.setObjectName("HudHint")
        for label in (self.hud_zoom, self.hud_preview, self.hud_hint):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            hud_layout.addWidget(label)
        layout.addWidget(self.canvas_hud, 0, 0, alignment=Qt.AlignLeft | Qt.AlignTop)
        return shell

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Viewer")
        toolbar.setObjectName("ViewerToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.open_action = QAction("Open", self)
        self.open_action.triggered.connect(self.open_image)
        toolbar.addAction(self.open_action)

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)

        self.pick_action = QAction("Pick", self)
        self.pick_action.setCheckable(True)
        self.pick_action.toggled.connect(self._toggle_picker)
        self.tool_action_group.addAction(self.pick_action)
        toolbar.addAction(self.pick_action)

        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.pan_action.toggled.connect(self._toggle_pan)
        self.tool_action_group.addAction(self.pan_action)
        toolbar.addAction(self.pan_action)

        self.fit_action = QAction("Fit", self)
        self.fit_action.triggered.connect(lambda: self.canvas.fit_to_view())
        toolbar.addAction(self.fit_action)

        self.actual_action = QAction("100%", self)
        self.actual_action.triggered.connect(lambda: self.canvas.set_100_percent())
        toolbar.addAction(self.actual_action)

        toolbar.addSeparator()
        view_label = QLabel("View")
        view_label.setObjectName("CommandBarLabel")
        toolbar.addWidget(view_label)
        self.view_combo = QComboBox()
        self.view_combo.setObjectName("ToolbarCombo")
        self.view_combo.addItems(VIEW_MODES)
        self.view_combo.setToolTip("Debug views include Fringe Mask to show where edge RGB repair is applied; repair changes color, not alpha.")
        self.view_combo.currentTextChanged.connect(self.canvas.set_view_mode)
        toolbar.addWidget(self.view_combo)

        bg_label = QLabel("BG")
        bg_label.setObjectName("CommandBarLabel")
        toolbar.addWidget(bg_label)
        self.background_combo = QComboBox()
        self.background_combo.setObjectName("ToolbarCombo")
        self.background_combo.addItems(BACKGROUND_MODES)
        self.background_combo.currentTextChanged.connect(self.canvas.set_background_mode)
        toolbar.addWidget(self.background_combo)

        toolbar.addSeparator()
        self.export_action = QAction("Export PNG", self)
        self.export_action.triggered.connect(self.export_png)
        toolbar.addAction(self.export_action)
        export_tool_button = toolbar.widgetForAction(self.export_action)
        if export_tool_button is not None:
            export_tool_button.setObjectName("ToolbarPrimaryButton")

        toolbar.addSeparator()
        self.gpu_status_action = QAction("GPU Status", self)
        self.gpu_status_action.setObjectName("GPUStatusAction")
        self.gpu_status_action.triggered.connect(self.show_gpu_status)
        toolbar.addAction(self.gpu_status_action)

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(460)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        panel = QFrame()
        panel.setObjectName("InspectorPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Keyer Inspector")
        title.setObjectName("InspectorTitle")
        layout.addWidget(title)

        presets = QHBoxLayout()
        presets.setSpacing(6)
        for name in ("Fast", "Clean", "High Accuracy"):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            presets.addWidget(btn)
        layout.addLayout(presets)

        self._add_key_section(layout)
        self._add_matte_section(layout)
        self._add_edge_section(layout)
        self._add_color_section(layout)
        self._add_output_section(layout)
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _add_key_section(self, parent: QVBoxLayout) -> None:
        section, layout = self._section("Screen")
        self.key_mode = QComboBox()
        self.key_mode.addItems(("Auto", "Green", "Blue", "Pick"))
        self.key_mode.setCurrentText(APP_DEFAULT_KEY_MODE)
        self.key_mode.currentTextChanged.connect(self._on_key_mode_changed)
        layout.addLayout(label_row("Screen Mode", self.key_mode))

        self.sample_size = QSpinBox()
        self.sample_size.setRange(1, 51)
        self.sample_size.setSingleStep(2)
        self.sample_size.setKeyboardTracking(False)
        self.sample_size.setValue(self.settings.sample_size)
        self.sample_size.valueChanged.connect(self.schedule_preview)
        layout.addLayout(label_row("Sample Size", self.sample_size))

        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        self.color_chip = QLabel()
        self.color_chip.setObjectName("ColorChip")
        self.color_chip.setFixedHeight(26)
        self.color_chip.setMinimumWidth(80)
        color_btn = QPushButton("Color…")
        color_btn.clicked.connect(self.pick_color_dialog)
        chip_row.addWidget(QLabel("Key Color"))
        chip_row.addWidget(self.color_chip, 1)
        chip_row.addWidget(color_btn)
        layout.addLayout(chip_row)

        self.screen_tolerance = SliderRow("Screen Tolerance", 0.01, 0.45, self.settings.tolerance, step=0.01)
        self.screen_softness = SliderRow("Screen Softness", 0.01, 0.35, self.settings.softness, step=0.01)
        self._connect_slider_rows(self.screen_tolerance, self.screen_softness)
        layout.addWidget(self.screen_tolerance)
        layout.addWidget(self.screen_softness)
        parent.addWidget(section)

    def _add_matte_section(self, parent: QVBoxLayout) -> None:
        section, layout = self._section("Matte")
        self.clip_background = SliderRow("Clip Background", 0.40, 0.98, self.settings.clip_background, step=0.01)
        self.clip_foreground = SliderRow("Clip Foreground", 0.00, 0.45, self.settings.clip_foreground, step=0.01)
        self.matte_gamma = SliderRow("Matte Gamma", 0.20, 2.20, self.settings.matte_gamma, step=0.05, decimals=2)
        self.core_strength = SliderRow("Core Strength", 0.00, 1.00, self.settings.core_strength, step=0.01)
        self.despeckle = SliderRow("Despeckle", 0, 1200, self.settings.despeckle_min_area, step=8, integer=True)
        self._connect_slider_rows(
            self.clip_background,
            self.clip_foreground,
            self.matte_gamma,
            self.core_strength,
            self.despeckle,
        )
        for row in (self.clip_background, self.clip_foreground, self.matte_gamma, self.core_strength, self.despeckle):
            layout.addWidget(row)

        self.policy = QComboBox()
        self.policy.addItems(("Connected Background", "Aggressive Interior Removal"))
        self.policy.setCurrentText("Aggressive Interior Removal" if self.settings.aggressive_interior_removal else "Connected Background")
        self.policy.currentTextChanged.connect(self.schedule_preview)
        layout.addLayout(label_row("Interior Policy", self.policy))
        parent.addWidget(section)

    def _add_edge_section(self, parent: QVBoxLayout) -> None:
        section, layout = self._section("Edges")
        default_radius = int(self.settings.edge_refine_radius or round(max(0.0, self.settings.edge_blur) * 4.0 + 1.0))
        self.edge_radius = SliderRow("Edge Radius", 0, 32, default_radius, step=1, integer=True)
        self.edge_softness = SliderRow("Edge Softness", 0.00, 1.00, self.settings.edge_softness, step=0.01)
        self.erode_expand = SliderRow("Erode / Expand", -8, 8, self.settings.erode_expand, step=1, integer=True)
        self._connect_slider_rows(self.edge_radius, self.edge_softness, self.erode_expand)
        layout.addWidget(self.edge_radius)
        layout.addWidget(self.edge_softness)
        layout.addWidget(self.erode_expand)
        parent.addWidget(section)

    def _add_color_section(self, parent: QVBoxLayout) -> None:
        section, layout = self._section("Spill Cleanup")
        self.despill = SliderRow("Despill", 0.00, 1.00, self.settings.despill, step=0.01)
        self.decontaminate = SliderRow("Decontaminate", 0.00, 1.00, self.settings.decontaminate, step=0.01)
        self.fringe_remove = SliderRow("Fringe Remove", 0.00, 1.00, self.settings.fringe_remove, step=0.01)
        self.edge_color_repair = SliderRow("Edge Color Repair", 0.00, 1.00, self.settings.edge_color_repair, step=0.01)
        self.inner_color_pull = SliderRow("Inner Color Pull", 0.00, 1.00, self.settings.inner_color_pull, step=0.01)
        self.fringe_band = SliderRow("Fringe Band", 0, 12, self.settings.fringe_band_radius, step=1, integer=True)
        self.luminance_restore = SliderRow("Luminance Restore", 0.00, 1.00, self.settings.luminance_restore, step=0.01)
        self.transition_unmix = QCheckBox("Transition Unmix")
        self.transition_unmix.setObjectName("TransitionUnmixCheck")
        self.transition_unmix.setChecked(bool(self.settings.transition_unmix))
        self.alpha_recover = SliderRow("Alpha Recover", 0.00, 1.00, self.settings.alpha_recover_strength, step=0.01)
        self.key_vector_despill = SliderRow("Key Vector Despill", 0.00, 1.00, self.settings.key_vector_despill, step=0.01)
        self.foreground_reference_pull = SliderRow("FG Color Pull", 0.00, 1.00, self.settings.foreground_reference_pull, step=0.01)
        self._set_control_tooltip(self.despill, "Legacy spill cleanup signal; higher values reduce key-color cast in repaired edge pixels.")
        self._set_control_tooltip(self.decontaminate, "Blends repaired edge RGB more strongly where the old screen color contaminated the foreground.")
        self._set_control_tooltip(self.fringe_remove, "Strength of key-color fringe/channel removal in soft edge pixels.")
        self._set_control_tooltip(self.edge_color_repair, "Blends reconstructed foreground RGB into the edge; modifies color only, not alpha.")
        self._set_control_tooltip(self.inner_color_pull, "Pulls contaminated edge RGB toward the nearest clean opaque foreground color.")
        self._set_control_tooltip(self.fringe_band, "Pixel width around the alpha edge eligible for fringe color repair.")
        self._set_control_tooltip(self.luminance_restore, "Luminance protection for spill cleanup and edge color repair; higher keeps perceived brightness.")
        self.transition_unmix.setToolTip("Repair semi-transparent graphic transitions by unmixing key-screen color without reducing alpha.")
        self._set_control_tooltip(self.alpha_recover, "Raises plausible transition alpha toward the solved foreground coverage; never lowers alpha.")
        self._set_control_tooltip(self.key_vector_despill, "Removes remaining key-color chroma along the sampled screen vector in repaired transitions.")
        self._set_control_tooltip(self.foreground_reference_pull, "Pulls repaired transition color toward nearby clean foreground while preserving luminance.")
        self.transition_unmix.toggled.connect(self._on_transition_unmix_toggled)
        self._connect_slider_rows(
            self.despill,
            self.decontaminate,
            self.fringe_remove,
            self.edge_color_repair,
            self.inner_color_pull,
            self.fringe_band,
            self.luminance_restore,
            self.alpha_recover,
            self.key_vector_despill,
            self.foreground_reference_pull,
        )
        for row in (
            self.despill,
            self.decontaminate,
            self.fringe_remove,
            self.edge_color_repair,
            self.inner_color_pull,
            self.fringe_band,
            self.luminance_restore,
        ):
            layout.addWidget(row)
        layout.addWidget(self.transition_unmix)
        for row in (self.alpha_recover, self.key_vector_despill, self.foreground_reference_pull):
            layout.addWidget(row)
        self._sync_transition_unmix_controls()
        parent.addWidget(section)

    def _add_output_section(self, parent: QVBoxLayout) -> None:
        section, layout = self._section("Masks & Export")
        self.output_mode = QComboBox()
        self.output_mode.setObjectName("OutputModeCombo")
        self.output_mode.addItems(OUTPUT_MODES)
        self.output_mode.setToolTip("Classical ignores imported mattes; Imported Matte uses a grayscale matte as foreground protection and alpha guidance.")
        self.output_mode.currentTextChanged.connect(self._on_output_mode_changed)
        layout.addLayout(label_row("Output Mode", self.output_mode))

        self.output_mode_status = QLabel("Output: classical chroma keyer")
        self.output_mode_status.setObjectName("HintText")
        self.output_mode_status.setWordWrap(True)
        layout.addWidget(self.output_mode_status)

        self.gpu_status_btn = QPushButton("GPU Status")
        self.gpu_status_btn.setObjectName("GPUStatusButton")
        self.gpu_status_btn.clicked.connect(self.show_gpu_status)
        layout.addWidget(self.gpu_status_btn)

        self.gpu_probe_status = QLabel("GPU Status: not probed")
        self.gpu_probe_status.setObjectName("HintText")
        self.gpu_probe_status.setWordWrap(True)
        layout.addWidget(self.gpu_probe_status)

        self.preview_quality = QComboBox()
        self.preview_quality.addItems(("Proxy", "Full Crop"))
        self.preview_quality.currentTextChanged.connect(self._on_preview_quality_changed)
        layout.addLayout(label_row("Preview", self.preview_quality))

        mask_row = QHBoxLayout()
        self.import_keep_btn = QPushButton("Import Keep")
        self.import_remove_btn = QPushButton("Import Remove")
        self.import_keep_btn.clicked.connect(lambda: self.import_mask("keep"))
        self.import_remove_btn.clicked.connect(lambda: self.import_mask("remove"))
        mask_row.addWidget(self.import_keep_btn)
        mask_row.addWidget(self.import_remove_btn)
        layout.addLayout(mask_row)

        hint_info = QLabel("Optional imported mattes can protect foreground/core areas during preview and export.")
        hint_info.setObjectName("HintText")
        hint_info.setWordWrap(True)
        layout.addWidget(hint_info)

        hint_row = QHBoxLayout()
        hint_row.setSpacing(8)
        self.import_hint_btn = QPushButton("Import Matte")
        self.clear_hint_btn = QPushButton("Clear Matte")
        self.import_hint_btn.clicked.connect(self.import_imported_matte)
        self.clear_hint_btn.clicked.connect(self.clear_imported_matte)
        hint_row.addWidget(self.import_hint_btn)
        hint_row.addWidget(self.clear_hint_btn)
        layout.addLayout(hint_row)

        self.alpha_hint_status = QLabel("Imported matte: none")
        self.alpha_hint_status.setObjectName("HintText")
        self.alpha_hint_status.setWordWrap(True)
        layout.addWidget(self.alpha_hint_status)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        self.export_matte_btn = QPushButton("Export Matte")
        self.export_matte_btn.clicked.connect(self.export_current_matte)
        self.export_btn = QPushButton("Export PNG")
        self.export_btn.setObjectName("PrimaryButton")
        self.export_btn.clicked.connect(self.export_png)
        output_row.addWidget(self.export_matte_btn)
        output_row.addWidget(self.export_btn)
        layout.addLayout(output_row)
        parent.addWidget(section)

    def _section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        section = QFrame()
        section.setObjectName("InspectorSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        return section, layout

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        status.setObjectName("AppStatus")
        self.setStatusBar(status)
        self.file_status = QLabel("No image")
        self.resolution_status = QLabel("—")
        self.scale_status = QLabel("Scale —")
        self.cursor_status = QLabel("Cursor —")
        self.sample_status = QLabel("Sample —")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.cancel_export_btn = QPushButton("Cancel")
        self.cancel_export_btn.clicked.connect(self.cancel_export)
        self.cancel_export_btn.hide()
        status.addWidget(self.file_status, 1)
        status.addPermanentWidget(self.resolution_status)
        status.addPermanentWidget(self.scale_status)
        status.addPermanentWidget(self.sample_status)
        status.addPermanentWidget(self.cursor_status)
        status.addPermanentWidget(self.progress_bar)
        status.addPermanentWidget(self.cancel_export_btn)
        status.showMessage("Ready")

    def _update_canvas_hud(self) -> None:
        if not hasattr(self, "hud_zoom"):
            return
        if getattr(self.canvas, "_has_pixmap", False):
            self.hud_zoom.setText(f"Zoom {self.canvas.transform().m11() * 100:.0f}%")
        else:
            self.hud_zoom.setText("Zoom —")
        if hasattr(self, "preview_quality"):
            preview_mode = self.preview_quality.currentText()
        else:
            preview_mode = "Full Crop" if self.current_preview_label.startswith("Full crop") else "Proxy"
        self.hud_preview.setText(f"Preview {preview_mode}")

    def _connect_slider_rows(self, *rows: SliderRow) -> None:
        for row in rows:
            row.value_changed.connect(self.schedule_preview)

    def _set_control_tooltip(self, row: SliderRow, text: str) -> None:
        row.setToolTip(text)
        for widget in (row.label, row.value_label, row.slider, row.spin):
            widget.setToolTip(text)

    def _on_transition_unmix_toggled(self, checked: bool) -> None:
        del checked
        self._sync_transition_unmix_controls()
        self.schedule_preview()

    def _sync_transition_unmix_controls(self) -> None:
        if not hasattr(self, "transition_unmix"):
            return
        enabled = self.transition_unmix.isChecked()
        for row in (self.alpha_recover, self.key_vector_despill, self.foreground_reference_pull):
            row.setEnabled(enabled)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0B0D10; color: #E7ECF3; font-family: Segoe UI, Inter, Arial; font-size: 13px; }
            QToolBar#ViewerToolbar { background: #0B0D10; border: none; border-bottom: 1px solid #2A3038; padding: 6px 8px; spacing: 6px; }
            QToolBar#ViewerToolbar QToolButton { background: transparent; color: #E7ECF3; border: 1px solid transparent; border-radius: 5px; padding: 6px 10px; font-weight: 600; }
            QToolBar#ViewerToolbar QToolButton:hover { background: #181D26; border-color: #2A3038; }
            QToolBar#ViewerToolbar QToolButton:pressed { background: #101318; border-color: #4F8CFF; }
            QToolBar#ViewerToolbar QToolButton:checked { background: #1E2A42; color: #FFFFFF; border-color: #4F8CFF; }
            QToolBar#ViewerToolbar QToolButton:disabled { color: #5D6775; background: transparent; border-color: transparent; }
            QToolBar#ViewerToolbar QToolButton:focus { border-color: #4F8CFF; }
            QToolBar#ViewerToolbar QToolButton#ToolbarPrimaryButton { background: #4F8CFF; color: #FFFFFF; border-color: #4F8CFF; }
            QToolBar#ViewerToolbar QToolButton#ToolbarPrimaryButton:hover { background: #6CA0FF; border-color: #6CA0FF; }
            QToolBar#ViewerToolbar QToolButton#ToolbarPrimaryButton:pressed { background: #3F74D8; border-color: #3F74D8; }
            QToolBar QLabel#CommandBarLabel { color: #9AA6B6; padding-left: 8px; background: transparent; font-weight: 600; }
            QToolBar::separator { width: 1px; background: #2A3038; margin: 6px 8px; }
            QFrame#CanvasArea { background: #101318; border: none; }
            QGraphicsView#ImageCanvas { background: #101318; border: none; }
            QFrame#CanvasHud { background: rgba(21, 25, 34, 220); border: 1px solid #2A3038; border-radius: 6px; margin: 10px; }
            QLabel#HudText { color: #E7ECF3; background: transparent; font-size: 12px; font-weight: 700; }
            QLabel#HudHint { color: #9AA6B6; background: transparent; font-size: 12px; }
            QSplitter#MainSplitter::handle { background: #2A3038; width: 5px; }
            QSplitter#MainSplitter::handle:hover { background: #3A4350; }
            QScrollArea#InspectorScroll { background: #151922; border: none; border-left: 1px solid #2A3038; }
            QScrollArea#InspectorScroll > QWidget > QWidget { background: #151922; }
            QFrame#InspectorPanel { background: #151922; border: none; }
            QLabel#InspectorTitle { font-size: 18px; font-weight: 800; color: #E7ECF3; padding-bottom: 2px; background: transparent; }
            QFrame#InspectorSection { background: #181D26; border: 1px solid #2A3038; border-radius: 6px; }
            QLabel#SectionTitle { color: #E7ECF3; font-size: 14px; font-weight: 800; background: transparent; }
            QLabel#ControlLabel { color: #9AA6B6; background: transparent; font-size: 13px; }
            QLabel#SliderValueLabel { color: #E7ECF3; background: transparent; font-size: 13px; font-weight: 700; }
            QLabel#HintText { color: #9AA6B6; background: transparent; font-size: 12px; }
            QLabel#ColorChip { border: 1px solid #2A3038; border-radius: 5px; background: #00dc32; }
            QPushButton { background: #202633; color: #E7ECF3; border: 1px solid #2A3038; border-radius: 5px; padding: 6px 10px; font-weight: 600; min-height: 24px; }
            QPushButton:hover { background: #283142; border-color: #3A4350; }
            QPushButton:pressed { background: #151B28; border-color: #4F8CFF; }
            QPushButton:focus { border-color: #4F8CFF; }
            QPushButton:disabled { background: #151922; color: #5D6775; border-color: #202633; }
            QPushButton#PrimaryButton { background: #4F8CFF; color: white; border-color: #4F8CFF; }
            QPushButton#PrimaryButton:hover { background: #6CA0FF; border-color: #6CA0FF; }
            QPushButton#PrimaryButton:pressed { background: #3F74D8; border-color: #3F74D8; }
            QPushButton#StepButton, QPushButton#ResetButton { padding: 3px 0px; color: #9AA6B6; min-height: 24px; min-width: 26px; }
            QPushButton#StepButton:hover, QPushButton#ResetButton:hover { color: #E7ECF3; }
            QPushButton#SecondaryToggleButton { background: transparent; color: #9AA6B6; border-color: #2A3038; text-align: left; }
            QPushButton#SecondaryToggleButton:hover, QPushButton#SecondaryToggleButton:checked { background: #151922; color: #E7ECF3; }
            QFrame#SecondaryPanel { background: #151922; border: 1px solid #2A3038; border-radius: 5px; }
            QCheckBox { background: transparent; color: #E7ECF3; spacing: 8px; padding: 3px 0px; font-weight: 600; }
            QCheckBox:disabled { color: #5D6775; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #101318; color: #E7ECF3; border: 1px solid #2A3038; border-radius: 5px; padding: 5px 8px; min-height: 24px; }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #3A4350; }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4F8CFF; }
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #5D6775; background: #151922; border-color: #202633; }
            QComboBox#ToolbarCombo { min-width: 110px; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { width: 0px; height: 0px; }
            QComboBox QAbstractItemView { background: #151922; color: #E7ECF3; border: 1px solid #2A3038; selection-background-color: #1E2A42; selection-color: #FFFFFF; outline: none; }
            QSlider::groove:horizontal { height: 4px; background: #2A3038; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #4F8CFF; border-radius: 2px; }
            QSlider::handle:horizontal { background: #E7ECF3; border: 1px solid #4F8CFF; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }
            QSlider::handle:horizontal:hover, QSlider::handle:horizontal:pressed { background: #FFFFFF; border-color: #6CA0FF; }
            QSlider:focus { border: none; }
            QStatusBar#AppStatus { background: #0B0D10; border-top: 1px solid #2A3038; color: #9AA6B6; }
            QStatusBar QLabel { color: #9AA6B6; background: transparent; padding: 0 6px; font-size: 12px; }
            QProgressBar { background: #151922; border: 1px solid #2A3038; border-radius: 4px; color: #E7ECF3; text-align: center; height: 14px; }
            QProgressBar::chunk { background: #4F8CFF; border-radius: 3px; }
            QScrollBar:vertical { background: #151922; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #2A3038; border-radius: 4px; min-height: 28px; }
            QScrollBar::handle:vertical:hover { background: #3A4350; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            QScrollBar:horizontal { background: #151922; height: 10px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #2A3038; border-radius: 4px; min-width: 28px; }
            QScrollBar::handle:horizontal:hover { background: #3A4350; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
            """
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() == Qt.Key_Space:
            if event.isAutoRepeat():
                return False
            if event.type() == QEvent.KeyRelease and self.canvas.space_pan_active:
                self.canvas.set_space_pan_active(False)
                event.accept()
                return True
            focus_widget = QApplication.focusWidget()
            focus_inside = focus_widget is not None and (focus_widget is self or self.isAncestorOf(focus_widget))
            if not self.isActiveWindow() and not focus_inside:
                return False
            if self._focus_keeps_space():
                return False
            if self.full_rgb is None:
                return False
            if event.type() == QEvent.KeyPress:
                self.canvas.set_space_pan_active(True)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _focus_keeps_space(self) -> bool:
        widget = QApplication.focusWidget()
        interactive_types = (
            QAbstractSpinBox,
            QComboBox,
            QSlider,
            QAbstractButton,
            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
        )
        while widget is not None:
            if widget is self.canvas or widget is self.canvas.viewport():
                return False
            if isinstance(widget, interactive_types):
                return True
            widget = widget.parentWidget()
        return False

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.load_image(Path(urls[0].toLocalFile()))

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            full_rgb, full_alpha = read_image_rgb(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        self.full_rgb = full_rgb
        self.full_alpha = full_alpha
        self.image_path = path
        self.keep_mask = None
        self.remove_mask = None
        self.alpha_hint_mask = None
        if hasattr(self, "output_mode"):
            self.output_mode.blockSignals(True)
            self.output_mode.setCurrentText("Classical")
            self.output_mode.blockSignals(False)
        self.current_result = None
        self._full_crop_rect = None
        self.last_sample_rgb = None
        self.last_cursor_rgb = None
        self.proxy_rgb, self.proxy_scale = resize_for_preview(self.full_rgb, max_side=1800)
        self.proxy_alpha = resize_alpha_for_preview(self.full_alpha, self.proxy_rgb.shape[:2])
        self.preview_quality.setCurrentText("Proxy")
        self._set_current_source(self.proxy_rgb, self.proxy_alpha, self.proxy_scale, (0, 0), "Proxy", reset_view=True)
        self.canvas.fit_to_view()
        self.file_status.setText(path.name)
        h, w = self.full_rgb.shape[:2]
        self.resolution_status.setText(f"{w}×{h}")
        self.statusBar().showMessage(f"Loaded {path.name}")
        self._sync_imported_matte_status()
        self._sync_output_mode_status()
        self._update_enabled_state()
        self.schedule_preview()

    def current_settings(self) -> KeySettings:
        key_mode = self.key_mode.currentText()
        output_mode = self.output_mode.currentText() if hasattr(self, "output_mode") else "Classical"
        engine_mode = {
            "Imported Matte": "ImportedMatte",
        }.get(output_mode, "GraphicExact")
        radius = int(self.edge_radius.value())
        return KeySettings(
            key_color=self.settings.key_color,
            tolerance=float(self.screen_tolerance.value()),
            softness=float(self.screen_softness.value()),
            edge_blur=max(0.0, (radius - 1) / 4.0),
            cleanup=0,
            mode=engine_mode,
            sample_size=int(self.sample_size.value()),
            auto_border_sample=key_mode != "Pick",
            auto_detect_key_color=key_mode == "Auto",
            clip_background=float(self.clip_background.value()),
            clip_foreground=float(self.clip_foreground.value()),
            matte_gamma=float(self.matte_gamma.value()),
            core_strength=float(self.core_strength.value()),
            edge_refine_radius=radius,
            edge_softness=float(self.edge_softness.value()),
            erode_expand=int(self.erode_expand.value()),
            despeckle_min_area=int(self.despeckle.value()),
            aggressive_interior_removal=self.policy.currentText() == "Aggressive Interior Removal",
            despill=float(self.despill.value()),
            decontaminate=float(self.decontaminate.value()),
            luminance_restore=float(self.luminance_restore.value()),
            fringe_remove=float(self.fringe_remove.value()),
            edge_color_repair=float(self.edge_color_repair.value()),
            inner_color_pull=float(self.inner_color_pull.value()),
            fringe_band_radius=int(self.fringe_band.value()),
            transition_unmix=bool(self.transition_unmix.isChecked()),
            alpha_recover_strength=float(self.alpha_recover.value()),
            key_vector_despill=float(self.key_vector_despill.value()),
            foreground_reference_pull=float(self.foreground_reference_pull.value()),
            luminance_protect=float(self.luminance_restore.value()),
            preview_scale=float(self.current_display_scale),
            use_tiling=True,
        )

    def _processing_alpha_input(self, settings: KeySettings, shape: tuple[int, int]) -> np.ndarray | None:
        if settings.mode == "ImportedMatte":
            return resize_alpha_hint_mask(self.alpha_hint_mask, shape)
        return None

    def schedule_preview(self) -> None:
        if self.full_rgb is None:
            return
        self.settings = self.current_settings()
        self._sync_output_mode_status()
        self._preview_generation += 1
        self._preview_jobs.clear()
        self._cancel_preview_threads()
        self.statusBar().showMessage("Preview queued…")
        self._preview_timer.start(150)

    def _cancel_preview_threads(self) -> None:
        for thread in list(self._preview_threads):
            if thread.isRunning():
                thread.request_cancel()

    def _on_preview_quality_changed(self, mode: str) -> None:
        self._full_crop_rect = self._current_full_crop() if mode == "Full Crop" and self.full_rgb is not None else None
        self.current_result = None
        self._update_canvas_hud()
        self.schedule_preview()

    def _start_preview(self) -> None:
        if self.full_rgb is None:
            return
        if any(thread.isRunning() for thread in self._preview_threads):
            self._preview_pending = True
            return
        try:
            job = self._make_preview_job()
        except Exception as exc:
            self.on_failed(f"Preview setup failed: {exc}")
            return
        generation = self._preview_generation
        self._preview_jobs[generation] = job
        self._preview_pending = False

        if (
            self.current_source_rgb is None
            or self.current_source_rgb.shape != job.display_rgb.shape
            or self.current_crop_origin != job.crop_origin
            or abs(self.current_display_scale - job.display_scale) > 1e-6
        ):
            self.current_result = None
            self._set_current_source(job.display_rgb, job.display_alpha, job.display_scale, job.crop_origin, job.label)

        thread = PreviewThread(generation, job)
        self._preview_threads.append(thread)
        thread.done.connect(self.on_preview_done)
        thread.progress.connect(self.on_preview_progress)
        thread.failed.connect(self.on_preview_failed)
        thread.finished.connect(lambda t=thread: self._forget_preview_thread(t))
        self.statusBar().showMessage(f"Processing {job.label.lower()} preview…")
        thread.start()

    def _make_preview_job(self) -> PreviewJob:
        assert self.full_rgb is not None
        settings = self.current_settings()
        if self.preview_quality.currentText() == "Full Crop":
            if self._full_crop_rect is None:
                self._full_crop_rect = self._current_full_crop()
            crop = self._full_crop_rect
            x0, y0, x1, y1 = crop
            alpha_hint = self._processing_alpha_input(settings, self.full_rgb.shape[:2])
            settings.full_res_crop = crop
            settings.preview_scale = 1.0
            settings.use_tiling = True
            display_rgb = self.full_rgb[y0:y1, x0:x1].copy()
            display_alpha = None if self.full_alpha is None else self.full_alpha[y0:y1, x0:x1].copy()
            return PreviewJob(
                input_rgb=self.full_rgb,
                original_alpha=self.full_alpha,
                keep_mask=self.keep_mask,
                remove_mask=self.remove_mask,
                alpha_hint=alpha_hint,
                settings=settings,
                display_rgb=display_rgb,
                display_alpha=display_alpha,
                display_scale=1.0,
                crop_origin=(x0, y0),
                label=f"Full crop {x1 - x0}×{y1 - y0}",
            )

        assert self.proxy_rgb is not None
        settings.full_res_crop = None
        self._full_crop_rect = None
        settings.preview_scale = self.proxy_scale
        shape = self.proxy_rgb.shape[:2]
        alpha_hint = self._processing_alpha_input(settings, shape)
        return PreviewJob(
            input_rgb=self.proxy_rgb,
            original_alpha=self.proxy_alpha,
            keep_mask=resize_mask(self.keep_mask, shape),
            remove_mask=resize_mask(self.remove_mask, shape),
            alpha_hint=alpha_hint,
            settings=settings,
            display_rgb=self.proxy_rgb,
            display_alpha=self.proxy_alpha,
            display_scale=self.proxy_scale,
            crop_origin=(0, 0),
            label="Proxy",
        )

    def _current_full_crop(self) -> tuple[int, int, int, int]:
        assert self.full_rgb is not None
        h, w = self.full_rgb.shape[:2]
        rect = self.canvas.visible_image_rect()
        if rect is None:
            center_x = w / 2.0
            center_y = h / 2.0
        else:
            x0, y0, x1, y1 = rect
            center_x = self.current_crop_origin[0] + ((x0 + x1) * 0.5) / max(self.current_display_scale, 1e-6)
            center_y = self.current_crop_origin[1] + ((y0 + y1) * 0.5) / max(self.current_display_scale, 1e-6)
        crop_w = min(w, max(320, min(1400, self.canvas.viewport().width())))
        crop_h = min(h, max(240, min(1000, self.canvas.viewport().height())))
        x0 = int(round(center_x - crop_w / 2.0))
        y0 = int(round(center_y - crop_h / 2.0))
        x0 = max(0, min(w - crop_w, x0))
        y0 = max(0, min(h - crop_h, y0))
        return x0, y0, x0 + crop_w, y0 + crop_h

    def _set_current_source(
        self,
        source_rgb: np.ndarray,
        source_alpha: np.ndarray | None,
        display_scale: float,
        crop_origin: tuple[int, int],
        label: str,
        *,
        reset_view: bool = True,
    ) -> None:
        self.current_source_rgb = source_rgb
        self.current_source_alpha = source_alpha
        self.current_display_scale = max(float(display_scale), 1e-6)
        self.current_crop_origin = crop_origin
        self.current_preview_label = label
        self.canvas.set_images(
            source_rgb,
            self.current_result,
            display_scale=self.current_display_scale,
            crop_origin=crop_origin,
            reset_view=reset_view,
        )
        self.scale_status.setText(f"{label} · {self.current_display_scale * 100:.1f}%")
        self._update_canvas_hud()

    def on_preview_progress(self, generation: int, value: float, stage: str) -> None:
        if generation == self._preview_generation:
            self.statusBar().showMessage(f"Preview {int(value * 100):d}% · {stage}")

    def on_preview_done(self, generation: int, result: KeyResult) -> None:
        job = self._preview_jobs.pop(generation, None)
        if generation != self._preview_generation or job is None:
            return
        self.current_result = result
        self._set_current_source(job.display_rgb, job.display_alpha, job.display_scale, job.crop_origin, job.label, reset_view=False)
        if result.screen_color is not None:
            self.color_chip.setToolTip(f"Screen sample used by engine: RGB {result.screen_color}")
            key_rgb = self.settings.key_color
            self.sample_status.setText(f"Key RGB {key_rgb[0]},{key_rgb[1]},{key_rgb[2]} · engine RGB {result.screen_color[0]},{result.screen_color[1]},{result.screen_color[2]}")
            if self.key_mode.currentText() == "Auto":
                r, g, b = result.screen_color
                self.color_chip.setStyleSheet(f"background: rgb({r}, {g}, {b});")
        engine_text = f" · engine RGB {result.screen_color}" if result.screen_color is not None else ""
        self.statusBar().showMessage(f"Preview ready · {job.label}{engine_text}")
        self._update_enabled_state()

    def on_preview_failed(self, generation: int, message: str) -> None:
        self._preview_jobs.pop(generation, None)
        if generation == self._preview_generation:
            self.on_failed(message)

    def _forget_preview_thread(self, thread: PreviewThread) -> None:
        if thread in self._preview_threads:
            self._preview_threads.remove(thread)
        if self._preview_pending and self.full_rgb is not None:
            self._preview_pending = False
            self._preview_timer.start(0)

    def export_png(self) -> None:
        if self.full_rgb is None:
            return
        default = "output_keyed.png"
        if self.image_path:
            default = str(self.image_path.with_name(f"{self.image_path.stem}_keyed.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", default, "PNG image (*.png)")
        if not path:
            return
        settings = self.current_settings()
        settings.full_res_crop = None
        settings.preview_scale = 1.0
        settings.use_tiling = True
        try:
            alpha_hint = self._processing_alpha_input(settings, self.full_rgb.shape[:2])
        except Exception as exc:
            self.on_failed(str(exc), title="Export setup failed")
            return
        self.export_thread = ExportThread(
            path,
            self.full_rgb,
            self.full_alpha,
            settings,
            self.keep_mask,
            self.remove_mask,
            alpha_hint,
        )
        self.export_thread.progress.connect(self.on_export_progress)
        self.export_thread.done.connect(self.on_export_done)
        self.export_thread.failed.connect(self.on_export_failed)
        self.export_thread.finished.connect(self._export_finished)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_export_btn.show()
        self.open_action.setEnabled(False)
        self._update_enabled_state()
        self.statusBar().showMessage("Export started…")
        self.export_thread.start()

    def cancel_export(self) -> None:
        if self.export_thread is not None and self.export_thread.isRunning():
            self.export_thread.request_cancel()
            self.statusBar().showMessage("Cancelling export…")

    def on_export_progress(self, value: float, stage: str) -> None:
        self.progress_bar.setValue(int(np.clip(value, 0.0, 1.0) * 100))
        self.statusBar().showMessage(f"Export {self.progress_bar.value()}% · {stage}")

    def on_export_done(self, path: str) -> None:
        self.progress_bar.setValue(100)
        self.statusBar().showMessage(f"Saved {Path(path).name}")
        QMessageBox.information(self, "Export complete", f"Saved transparent PNG:\n{path}")

    def on_export_failed(self, message: str) -> None:
        if "cancel" in message.lower():
            self.statusBar().showMessage("Export cancelled")
        else:
            self.on_failed(message, title="Export failed")

    def _export_finished(self) -> None:
        self.progress_bar.hide()
        self.cancel_export_btn.hide()
        self.export_thread = None
        self.open_action.setEnabled(True)
        self._update_enabled_state()

    def export_current_matte(self) -> None:
        if self.current_result is None:
            return
        default = "current_matte.png"
        if self.image_path:
            suffix = "crop_matte" if self.current_crop_origin != (0, 0) else "preview_matte"
            default = str(self.image_path.with_name(f"{self.image_path.stem}_{suffix}.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Export current matte", default, "PNG image (*.png)")
        if not path:
            return
        try:
            write_grayscale_mask(path, self.current_result.alpha)
        except Exception as exc:
            QMessageBox.critical(self, "Matte export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved matte {Path(path).name}")

    def import_mask(self, kind: str) -> None:
        if self.full_rgb is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Import {kind} mask",
            "",
            "Mask PNG (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not path:
            return
        try:
            mask = read_grayscale_mask(path, self.full_rgb.shape[:2])
        except Exception as exc:
            QMessageBox.critical(self, "Mask import failed", str(exc))
            return
        if kind == "keep":
            self.keep_mask = mask
            self.statusBar().showMessage(f"Imported keep mask {Path(path).name}")
        else:
            self.remove_mask = mask
            self.statusBar().showMessage(f"Imported remove mask {Path(path).name}")
        self.schedule_preview()

    def import_imported_matte(self) -> None:
        if self.full_rgb is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import matte",
            "",
            "Grayscale mask (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.alpha_hint_mask = read_imported_matte_mask(path, self.full_rgb.shape[:2])
        except Exception as exc:
            QMessageBox.critical(self, "Matte import failed", str(exc))
            return
        self.output_mode.setCurrentText("Imported Matte")
        self._sync_imported_matte_status(Path(path).name)
        self._sync_output_mode_status()
        self.statusBar().showMessage(f"Imported matte {Path(path).name}")
        self._update_enabled_state()
        self.schedule_preview()

    def clear_imported_matte(self) -> None:
        if self.alpha_hint_mask is None:
            return
        self.alpha_hint_mask = None
        if self.output_mode.currentText() == "Imported Matte":
            self.output_mode.setCurrentText("Classical")
        self._sync_imported_matte_status()
        self._sync_output_mode_status()
        self.statusBar().showMessage("Cleared imported matte")
        self._update_enabled_state()
        self.schedule_preview()

    def _sync_imported_matte_status(self, filename: str | None = None) -> None:
        if not hasattr(self, "alpha_hint_status"):
            return
        if self.alpha_hint_mask is None:
            self.alpha_hint_status.setText("Imported matte: none. Import a grayscale mask to protect foreground/core regions before preview/export.")
        else:
            h, w = self.alpha_hint_mask.shape[:2]
            name = f" · {filename}" if filename else ""
            self.alpha_hint_status.setText(f"Imported matte: loaded {w}×{h}{name}; high values protect foreground, low values do not remove background by themselves.")

    def _on_output_mode_changed(self, mode: str) -> None:
        self._sync_output_mode_status()
        if self.full_rgb is not None:
            self.schedule_preview()

    def _sync_output_mode_status(self) -> None:
        if not hasattr(self, "output_mode_status"):
            return
        mode = self.output_mode.currentText() if hasattr(self, "output_mode") else "Classical"
        if mode == "Imported Matte":
            if self.alpha_hint_mask is None:
                text = "Output: Imported Matte selected, but no matte is loaded; processing falls back to classical alpha decisions."
            else:
                h, w = self.alpha_hint_mask.shape[:2]
                text = f"Output: Imported Matte uses grayscale matte {w}×{h} as foreground protection and alpha guidance."
        else:
            text = "Output: Classical mode ignores imported mattes for preview/export."
        self.output_mode_status.setText(text)

    def show_gpu_status(self, checked: bool = False) -> None:
        del checked
        if self._gpu_probe_running():
            self.statusBar().showMessage("GPU status probe already running…")
            return
        process = QProcess(self)
        process.setObjectName("GPUStatusProcess")
        process.setWorkingDirectory(str(WRITABLE_APP_DIR))
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.finished.connect(lambda exit_code, exit_status, proc=process: self._on_gpu_probe_finished(proc, exit_code, exit_status))
        process.errorOccurred.connect(lambda error, proc=process: self._on_gpu_probe_error(proc, error))
        self.gpu_probe_process = process
        if hasattr(self, "gpu_probe_status"):
            self.gpu_probe_status.setText("GPU Status: running probe in subprocess…")
        self.statusBar().showMessage("GPU status probe running…")
        self._update_enabled_state()
        command, arguments = gpu_probe_subprocess_command()
        process.start(command, arguments)

    def _on_gpu_probe_error(self, process: QProcess, error) -> None:
        if process is not self.gpu_probe_process:
            return
        if process.state() != QProcess.NotRunning:
            return
        message = process.errorString() or str(error)
        self.gpu_probe_process = None
        if hasattr(self, "gpu_probe_status"):
            self.gpu_probe_status.setText(f"GPU Status: failed to start probe subprocess: {message}")
        self.statusBar().showMessage("GPU status probe failed to start")
        self._update_enabled_state()
        process.deleteLater()

    def _on_gpu_probe_finished(self, process: QProcess, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        stdout = self._process_stdout(process)
        stderr = self._process_stderr(process)
        if process is not self.gpu_probe_process:
            process.deleteLater()
            return
        self.gpu_probe_process = None
        result = self._json_object_from_text(stdout)
        if isinstance(result, dict):
            self.last_gpu_probe = result
            summary = self._format_gpu_probe_summary(result)
            if hasattr(self, "gpu_probe_status"):
                self.gpu_probe_status.setText(f"GPU Status: {summary}")
            self.statusBar().showMessage(f"GPU status: {result.get('status', 'unknown')}")
            if not self._closing:
                QMessageBox.information(self, "GPU Status", self._format_gpu_probe_details(result))
        else:
            detail = stderr.strip() or f"process exited {exit_code} status {exit_status}"
            if hasattr(self, "gpu_probe_status"):
                self.gpu_probe_status.setText(f"GPU Status: failed - {detail}")
            self.statusBar().showMessage("GPU status probe failed")
        self._update_enabled_state()
        process.deleteLater()

    def _gpu_probe_running(self) -> bool:
        return self.gpu_probe_process is not None and self.gpu_probe_process.state() != QProcess.NotRunning

    def _process_stdout(self, process: QProcess) -> str:
        return bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")

    def _process_stderr(self, process: QProcess) -> str:
        return bytes(process.readAllStandardError()).decode("utf-8", errors="replace")

    def _json_object_from_text(self, text: str) -> dict | None:
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

    def _format_gpu_probe_summary(self, result: dict) -> str:
        status = result.get("status", "unknown")
        message = str(result.get("message") or "")
        device = (result.get("cuda") or {}).get("device_name")
        if device:
            return f"{status} · {device}. {message}"
        return f"{status}. {message}"

    def _format_gpu_probe_details(self, result: dict) -> str:
        cuda = result.get("cuda") or {}
        torch_info = result.get("torch") or {}
        smi = result.get("nvidia_smi") or {}
        smoke = result.get("matmul_smoke") or {}
        return "\n".join(
            (
                f"GPU runtime: {result.get('status', 'unknown')} - {result.get('message', '')}",
                f"PyTorch import: {torch_info.get('import_success')} version={torch_info.get('version')} cuda={torch_info.get('cuda_version')}",
                f"CUDA: available={cuda.get('is_available')} device_count={cuda.get('device_count')} device={cuda.get('device_name')} capability={cuda.get('device_capability')}",
                f"nvidia-smi: available={smi.get('available')} driver={smi.get('driver_version')} cuda={smi.get('cuda_version')}",
                f"matmul smoke: ran={smoke.get('ran')} ok={smoke.get('ok')} error={smoke.get('error')}",
            )
        )

    def set_key_color(self, color: tuple[int, int, int], *, refresh: bool = True) -> None:
        self.settings.key_color = tuple(int(np.clip(c, 0, 255)) for c in color)
        self._sync_key_chip()
        if refresh:
            self.schedule_preview()

    def _sync_key_chip(self) -> None:
        r, g, b = self.settings.key_color
        self.color_chip.setStyleSheet(f"background: rgb({r}, {g}, {b});")
        self.color_chip.setToolTip(f"Key color RGB {r}, {g}, {b}")

    def _on_key_mode_changed(self, mode: str) -> None:
        if mode == "Green":
            self.pan_action.setChecked(True)
            self.set_key_color((0, 220, 50))
        elif mode == "Blue":
            self.pan_action.setChecked(True)
            self.set_key_color((30, 80, 235))
        elif mode == "Pick":
            self.pick_action.setChecked(True)
            self.statusBar().showMessage("Pick mode: click the canvas to sample the screen color")
            self.schedule_preview()
        else:
            self.pan_action.setChecked(True)
            self.schedule_preview()

    def _toggle_picker(self, enabled: bool) -> None:
        if not enabled:
            return
        self.canvas.set_picker_enabled(enabled)
        if enabled and self.key_mode.currentText() != "Pick":
            self.key_mode.blockSignals(True)
            self.key_mode.setCurrentText("Pick")
            self.key_mode.blockSignals(False)
            self.schedule_preview()

    def _toggle_pan(self, enabled: bool) -> None:
        if enabled:
            self.canvas.set_picker_enabled(False)

    def pick_color_dialog(self) -> None:
        r, g, b = self.settings.key_color
        color = QColorDialog.getColor(QColor(r, g, b), self, "Pick key color")
        if color.isValid():
            self.key_mode.setCurrentText("Pick")
            self.set_key_color((color.red(), color.green(), color.blue()))

    def pick_from_canvas(self, x: int, y: int) -> None:
        if self.current_source_rgb is None:
            return
        cursor_color = tuple(int(v) for v in self.current_source_rgb[y, x])
        radius = max(0, int(self.sample_size.value()) // 2)
        y0 = max(0, y - radius)
        y1 = min(self.current_source_rgb.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(self.current_source_rgb.shape[1], x + radius + 1)
        patch = self.current_source_rgb[y0:y1, x0:x1]
        color = tuple(np.median(patch.reshape(-1, 3), axis=0).astype(int).tolist())
        self.last_cursor_rgb = cursor_color
        self.last_sample_rgb = color
        self.key_mode.setCurrentText("Pick")
        self.set_key_color(color)
        full_x, full_y = self._display_to_full(x, y)
        self.sample_status.setText(
            f"Cursor RGB {cursor_color[0]},{cursor_color[1]},{cursor_color[2]} · sampled RGB {color[0]},{color[1]},{color[2]}"
        )
        self.statusBar().showMessage(
            f"Picked sampled RGB {color} from cursor RGB {cursor_color} at {full_x}, {full_y}"
        )

    def update_cursor_status(self, x: int, y: int) -> None:
        if self.current_source_rgb is None:
            return
        if y >= self.current_source_rgb.shape[0] or x >= self.current_source_rgb.shape[1]:
            return
        full_x, full_y = self._display_to_full(x, y)
        cursor_color = tuple(int(v) for v in self.current_source_rgb[y, x])
        r, g, b = cursor_color
        self.last_cursor_rgb = cursor_color
        alpha_text = "—"
        if self.current_result is not None and self.current_result.alpha.shape[:2] == self.current_source_rgb.shape[:2]:
            alpha_text = str(int(self.current_result.alpha[y, x]))
        elif self.current_source_alpha is not None:
            alpha_text = str(int(np.clip(self.current_source_alpha[y, x], 0.0, 1.0) * 255))
        self.cursor_status.setText(f"x {full_x}, y {full_y} · cursor RGB {r},{g},{b} · A {alpha_text}")

    def _display_to_full(self, x: int, y: int) -> tuple[int, int]:
        full_x = int(round(self.current_crop_origin[0] + x / max(self.current_display_scale, 1e-6)))
        full_y = int(round(self.current_crop_origin[1] + y / max(self.current_display_scale, 1e-6)))
        return full_x, full_y

    def apply_preset(self, name: str) -> None:
        presets = {
            "Fast": {
                self.screen_tolerance: 0.20,
                self.screen_softness: 0.07,
                self.clip_background: 0.80,
                self.clip_foreground: 0.16,
                self.edge_radius: 3,
                self.edge_softness: 0.35,
                self.despeckle: 24,
                self.decontaminate: 0.35,
                self.fringe_remove: 0.55,
                self.edge_color_repair: 0.40,
                self.inner_color_pull: 0.20,
                self.fringe_band: 2,
                self.luminance_restore: 0.20,
                self.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
                self.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
                self.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
            },
            "Clean": {
                self.screen_tolerance: 0.18,
                self.screen_softness: 0.08,
                self.clip_background: 0.78,
                self.clip_foreground: 0.14,
                self.edge_radius: 6,
                self.edge_softness: 0.55,
                self.despeckle: 48,
                self.decontaminate: 0.50,
                self.fringe_remove: 0.70,
                self.edge_color_repair: 0.55,
                self.inner_color_pull: 0.35,
                self.fringe_band: 3,
                self.luminance_restore: 0.35,
                self.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
                self.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
                self.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
            },
            "High Accuracy": {
                self.screen_tolerance: APP_DEFAULT_SETTINGS.tolerance,
                self.screen_softness: APP_DEFAULT_SETTINGS.softness,
                self.clip_background: APP_DEFAULT_SETTINGS.clip_background,
                self.clip_foreground: APP_DEFAULT_SETTINGS.clip_foreground,
                self.matte_gamma: APP_DEFAULT_SETTINGS.matte_gamma,
                self.core_strength: APP_DEFAULT_SETTINGS.core_strength,
                self.despeckle: APP_DEFAULT_SETTINGS.despeckle_min_area,
                self.edge_radius: APP_DEFAULT_SETTINGS.edge_refine_radius,
                self.edge_softness: APP_DEFAULT_SETTINGS.edge_softness,
                self.erode_expand: APP_DEFAULT_SETTINGS.erode_expand,
                self.despill: APP_DEFAULT_SETTINGS.despill,
                self.decontaminate: APP_DEFAULT_SETTINGS.decontaminate,
                self.fringe_remove: APP_DEFAULT_SETTINGS.fringe_remove,
                self.edge_color_repair: APP_DEFAULT_SETTINGS.edge_color_repair,
                self.inner_color_pull: APP_DEFAULT_SETTINGS.inner_color_pull,
                self.fringe_band: APP_DEFAULT_SETTINGS.fringe_band_radius,
                self.luminance_restore: APP_DEFAULT_SETTINGS.luminance_restore,
                self.alpha_recover: APP_DEFAULT_SETTINGS.alpha_recover_strength,
                self.key_vector_despill: APP_DEFAULT_SETTINGS.key_vector_despill,
                self.foreground_reference_pull: APP_DEFAULT_SETTINGS.foreground_reference_pull,
            },
        }
        for row, value in presets[name].items():
            row.set_value(value, emit=False)
        self.transition_unmix.blockSignals(True)
        self.transition_unmix.setChecked(bool(APP_DEFAULT_SETTINGS.transition_unmix))
        self.transition_unmix.blockSignals(False)
        self._sync_transition_unmix_controls()
        if name == "High Accuracy":
            self.key_mode.blockSignals(True)
            self.key_mode.setCurrentText(APP_DEFAULT_KEY_MODE)
            self.key_mode.blockSignals(False)
            self.sample_size.blockSignals(True)
            self.sample_size.setValue(APP_DEFAULT_SETTINGS.sample_size)
            self.sample_size.blockSignals(False)
            self.policy.blockSignals(True)
            self.policy.setCurrentText("Aggressive Interior Removal")
            self.policy.blockSignals(False)
            self.pick_action.setChecked(False)
            self.set_key_color(APP_DEFAULT_SETTINGS.key_color, refresh=False)
        self.statusBar().showMessage(f"Preset {name} applied")
        self.schedule_preview()

    def on_failed(self, message: str, *, title: str = "Processing failed") -> None:
        self.statusBar().showMessage(title)
        QMessageBox.critical(self, title, message)

    def _update_enabled_state(self) -> None:
        has_image = self.full_rgb is not None
        has_result = self.current_result is not None
        export_running = self.export_thread is not None and self.export_thread.isRunning()
        gpu_running = self._gpu_probe_running()
        self.export_btn.setEnabled(has_image and not export_running)
        self.export_action.setEnabled(has_image and not export_running)
        self.export_matte_btn.setEnabled(has_result and not export_running)
        self.output_mode.setEnabled(not export_running)
        self.import_keep_btn.setEnabled(has_image and not export_running)
        self.import_remove_btn.setEnabled(has_image and not export_running)
        self.import_hint_btn.setEnabled(has_image and not export_running)
        self.clear_hint_btn.setEnabled(has_image and self.alpha_hint_mask is not None and not export_running)
        self.gpu_status_action.setEnabled(not gpu_running)
        self.gpu_status_btn.setEnabled(not gpu_running)
        self.fit_action.setEnabled(has_image)
        self.actual_action.setEnabled(has_image)
        self.pick_action.setEnabled(has_image)
        self.pan_action.setEnabled(has_image)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.canvas.set_space_pan_active(False)
        self._preview_timer.stop()
        self._preview_pending = False
        if self.export_thread is not None and self.export_thread.isRunning():
            self.export_thread.request_cancel()
            self.export_thread.wait(3000)
        for thread in list(self._preview_threads):
            thread.request_cancel()
            thread.wait(3000)
        if self._gpu_probe_running() and self.gpu_probe_process is not None:
            process = self.gpu_probe_process
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(3000)
            self.gpu_probe_process = None
            process.deleteLater()
        super().closeEvent(event)


def label_row(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    text = QLabel(label)
    text.setObjectName("ControlLabel")
    row.addWidget(text)
    row.addWidget(widget, 1)
    return row


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


def mask_to_rgb(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        h, w = shape or (1, 1)
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.repeat(arr[:, :, None], 3, axis=2)


def blank_rgb_qimage(shape: tuple[int, int] | None = None) -> QImage:
    h, w = shape or (1, 1)
    image = QImage(max(1, int(w)), max(1, int(h)), QImage.Format_RGB888)
    image.fill(QColor(0, 0, 0))
    return image


def mask_to_qimage(mask: np.ndarray | None, shape: tuple[int, int] | None = None) -> QImage:
    if mask is None:
        return blank_rgb_qimage(shape)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()


def debug_rgb_to_rgb(image: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if image is None:
        h, w = shape or (1, 1)
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return mask_to_rgb(arr, shape)
    if arr.ndim != 3 or arr.shape[2] < 3:
        h, w = shape or (1, 1)
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    if shape is not None and arr.shape[:2] != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(rgba[:, :, :4], dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def checker_brush() -> QBrush:
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#161B22"))
    painter = QPainter(pixmap)
    painter.fillRect(0, 0, size // 2, size // 2, QColor("#202633"))
    painter.fillRect(size // 2, size // 2, size // 2, size // 2, QColor("#202633"))
    painter.end()
    return QBrush(pixmap)


def composite_rgba_for_mode(rgba: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Checkerboard":
        return checkerboard_composite(rgba, cell=18)
    if mode == "White":
        bg = np.array([255, 255, 255], dtype=np.float32)
    elif mode == "Gray":
        bg = np.array([119, 125, 134], dtype=np.float32)
    else:
        bg = np.array([0, 0, 0], dtype=np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgb * alpha + bg.reshape(1, 1, 3) * (1.0 - alpha)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    headless_result = dispatch_headless_cli(argv)
    if headless_result is not None:
        return headless_result

    update_boot_splash("Loading Qt runtime…")
    app = QApplication(argv)
    app.setApplicationName("ImgKey")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0B0D10"))
    app.setPalette(palette)
    qt_splash = create_qt_startup_splash()
    qt_splash.show()
    app.processEvents()
    update_boot_splash("Building ImgKey interface…")
    qt_splash.showMessage("Building ImgKey interface…", Qt.AlignLeft | Qt.AlignBottom, QColor("#E7ECF3"))
    app.processEvents()
    window = MainWindow()
    update_boot_splash("Showing ImgKey window…")
    qt_splash.showMessage("Showing ImgKey window…", Qt.AlignLeft | Qt.AlignBottom, QColor("#E7ECF3"))
    window.show()
    app.processEvents()
    QTimer.singleShot(0, lambda splash=qt_splash, win=window: splash.finish(win))
    QTimer.singleShot(0, close_boot_splash)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
