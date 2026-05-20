from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


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


def label_row(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    text = QLabel(label)
    text.setObjectName("ControlLabel")
    row.addWidget(text)
    row.addWidget(widget, 1)
    return row
