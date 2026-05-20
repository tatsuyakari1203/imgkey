from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QEvent, QProcess, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
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
    read_grayscale_mask,
    read_imported_matte_mask,
    read_image_rgb,
    resize_for_preview,
    write_grayscale_mask,
)
from ui.canvas import (
    BACKGROUND_MODES,
    VIEW_MODES,
    ImageCanvas,
    blank_rgb_qimage,
    checker_brush,
    composite_rgba_for_mode,
    debug_rgb_to_rgb,
    mask_to_qimage,
    mask_to_rgb,
    rgb_to_qimage,
    rgba_to_qimage,
)
from ui.export_controller import ExportController, ExportThread
from ui.gpu_probe_controller import (
    GPUProbeController,
    format_gpu_probe_details,
    format_gpu_probe_summary,
    gpu_probe_subprocess_command,
    json_object_from_text,
    message_mentions_gpu_backend,
    process_stderr,
    process_stdout,
)
from ui.preview_controller import (
    PreviewController,
    PreviewJob,
    PreviewThread,
    resize_alpha_for_preview,
    resize_alpha_hint_mask,
    resize_mask,
)
from ui.settings_mapper import (
    APP_DEFAULT_EDGE_RADIUS,
    APP_DEFAULT_KEY_MODE,
    APP_DEFAULT_SETTINGS,
    app_default_settings,
    current_settings_from_window,
    preset_control_values,
    processing_alpha_input,
)
from ui.widgets import SliderRow, label_row


OUTPUT_MODES = ("Classical", "Imported Matte")
GPU_ACCELERATION_MODES = ("Auto", "Off", "Force GPU")


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

        self.preview_controller = PreviewController(self)
        self.export_controller = ExportController(self)
        self.gpu_probe_controller = GPUProbeController(self)
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

        self.gpu_acceleration = QComboBox()
        self.gpu_acceleration.setObjectName("GPUAccelerationCombo")
        self.gpu_acceleration.addItems(GPU_ACCELERATION_MODES)
        gpu_default = str(getattr(self.settings, "gpu_acceleration", "Off") or "Off")
        self.gpu_acceleration.setCurrentText(gpu_default if gpu_default in GPU_ACCELERATION_MODES else "Off")
        self.gpu_acceleration.setToolTip("Optional compact CUDA DLL acceleration. Auto falls back to CPU; Force GPU reports runtime errors clearly.")
        self.gpu_acceleration.currentTextChanged.connect(self._on_gpu_acceleration_changed)
        layout.addLayout(label_row("GPU Acceleration", self.gpu_acceleration))

        self.gpu_status_btn = QPushButton("GPU Status")
        self.gpu_status_btn.setObjectName("GPUStatusButton")
        self.gpu_status_btn.clicked.connect(self.show_gpu_status)
        layout.addWidget(self.gpu_status_btn)

        self.gpu_probe_status = QLabel("GPU Status: acceleration off; CPU path used.")
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
        return current_settings_from_window(self)

    def _processing_alpha_input(self, settings: KeySettings, shape: tuple[int, int]) -> np.ndarray | None:
        return processing_alpha_input(settings, self.alpha_hint_mask, shape)

    def schedule_preview(self) -> None:
        self.preview_controller.schedule_preview()

    def _cancel_preview_threads(self) -> None:
        self.preview_controller.cancel_preview_threads()

    def _on_preview_quality_changed(self, mode: str) -> None:
        self._full_crop_rect = self._current_full_crop() if mode == "Full Crop" and self.full_rgb is not None else None
        self.current_result = None
        self._update_canvas_hud()
        self.schedule_preview()

    def _start_preview(self) -> None:
        self.preview_controller.start_preview()

    def _make_preview_job(self) -> PreviewJob:
        return self.preview_controller.make_preview_job()

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
        self._sync_gpu_usage_status(result.gpu_acceleration)
        self._update_enabled_state()

    def on_preview_failed(self, generation: int, message: str) -> None:
        self._preview_jobs.pop(generation, None)
        if generation == self._preview_generation:
            if hasattr(self, "gpu_probe_status") and self._message_mentions_gpu_backend(message):
                self.gpu_probe_status.setText(f"GPU Status: error. {message}")
            self.on_failed(message)

    def _forget_preview_thread(self, thread: PreviewThread) -> None:
        self.preview_controller.forget_preview_thread(thread)

    def export_png(self) -> None:
        self.export_controller.export_png()

    def cancel_export(self) -> None:
        self.export_controller.cancel_export()

    def on_export_progress(self, value: float, stage: str) -> None:
        self.export_controller.on_export_progress(value, stage)

    def on_export_done(self, path: str) -> None:
        self.export_controller.on_export_done(path)

    def on_export_failed(self, message: str) -> None:
        self.export_controller.on_export_failed(message)

    def _export_finished(self) -> None:
        self.export_controller.export_finished()

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

    def _on_gpu_acceleration_changed(self, mode: str) -> None:
        self._sync_gpu_usage_status({"mode": mode, "status": "pending" if mode != "Off" else "off", "message": None})
        if self.full_rgb is not None:
            self.schedule_preview()

    def _sync_gpu_usage_status(self, info: dict | None = None) -> None:
        if not hasattr(self, "gpu_probe_status"):
            return
        mode = self.gpu_acceleration.currentText() if hasattr(self, "gpu_acceleration") else "Off"
        if info is None:
            if mode == "Off":
                text = "GPU Status: acceleration off; CPU path used."
            else:
                text = f"GPU Status: {mode}; waiting for next preview/export. Use GPU Status to probe the compact CUDA DLL runtime."
            self.gpu_probe_status.setText(text)
            return
        status = str(info.get("status") or "unknown")
        message = str(info.get("message") or "")
        backend = info.get("backend")
        prefix = f"{status}"
        if backend:
            prefix += f" · {backend}"
        if message:
            self.gpu_probe_status.setText(f"GPU Status: {prefix}. {message}")
        elif mode == "Off":
            self.gpu_probe_status.setText("GPU Status: acceleration off; CPU path used.")
        else:
            self.gpu_probe_status.setText(f"GPU Status: {mode} · {prefix}; waiting for next preview/export.")

    def _message_mentions_gpu_backend(self, message: str) -> bool:
        return message_mentions_gpu_backend(message)

    def show_gpu_status(self, checked: bool = False) -> None:
        self.gpu_probe_controller.show_gpu_status(checked)

    def _on_gpu_probe_error(self, process: QProcess, error) -> None:
        self.gpu_probe_controller.on_gpu_probe_error(process, error)

    def _on_gpu_probe_finished(self, process: QProcess, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.gpu_probe_controller.on_gpu_probe_finished(process, exit_code, exit_status)

    def _gpu_probe_running(self) -> bool:
        return self.gpu_probe_controller.gpu_probe_running()

    def _process_stdout(self, process: QProcess) -> str:
        return process_stdout(process)

    def _process_stderr(self, process: QProcess) -> str:
        return process_stderr(process)

    def _json_object_from_text(self, text: str) -> dict | None:
        return json_object_from_text(text)

    def _format_gpu_probe_summary(self, result: dict) -> str:
        return format_gpu_probe_summary(result)

    def _format_gpu_probe_details(self, result: dict) -> str:
        return format_gpu_probe_details(result)

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
        presets = preset_control_values(name, self)
        for row, value in presets.items():
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
        self.gpu_acceleration.setEnabled(not export_running)
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
