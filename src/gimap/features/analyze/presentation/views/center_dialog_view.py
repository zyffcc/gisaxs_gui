"""Repeated manual beam-centre adjustment without hiding the curves."""

from PyQt5.QtWidgets import QCheckBox, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from ..center_controls import make_center_step


class CenterDialogView:
    def setup_ui(self, center, center_step_px):
        self.setWindowTitle("Beam Centre")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Direct-beam position in detector pixels: x from the left edge, "
            "y from the top edge of the image as shown.", self,
        )
        note.setWordWrap(True)
        note.setProperty("gimapRole", "muted")
        layout.addWidget(note)
        form = QFormLayout()
        self.x_spin, self.y_spin = [self._coordinate(value) for value in center]
        self.center_step_spin = make_center_step(self, self.x_spin, self.y_spin, center_step_px)
        form.addRow("x (px)", self.x_spin)
        form.addRow("y (px)", self.y_spin)
        form.addRow("Centre step", self.center_step_spin)
        layout.addLayout(form)
        self.live_preview_check = QCheckBox("Live Preview", self)
        self.live_preview_check.setObjectName("analyzeCenterLivePreview")
        self.live_preview_check.setToolTip(
            "Update curves automatically as the beam centre changes; the centre step "
            "controls arrow buttons, keys, and the mouse wheel."
        )
        layout.addWidget(self.live_preview_check)
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self.buttons.button(QDialogButtonBox.Close).setAutoDefault(False)
        self.apply_button = self.buttons.addButton("Apply Centre", QDialogButtonBox.ActionRole)
        self.apply_button.setProperty("gimapRole", "primary")
        self.apply_button.setDefault(True)
        layout.addWidget(self.buttons)

    def _coordinate(self, value):
        spin = QDoubleSpinBox(self)
        spin.setDecimals(4)
        spin.setRange(-100000.0, 100000.0)
        spin.setKeyboardTracking(False)
        spin.setValue(float(value))
        return spin
