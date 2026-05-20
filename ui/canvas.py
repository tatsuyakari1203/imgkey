from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QSizePolicy

from keyer import KeyResult, checkerboard_composite


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
