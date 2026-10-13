"""A modeless editor that repeatedly applies a session beam centre."""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog

from src.gimap.app.presentation.theme import set_role

from .views.center_dialog_view import CenterDialogView


class CenterDialog(QDialog, CenterDialogView):
    applyCenterRequested = pyqtSignal()

    def __init__(self, center, parent=None, *, center_step_px=1.0):
        super().__init__(parent)
        self.setup_ui(center, center_step_px)
        self.apply_button.clicked.connect(self.applyCenterRequested)
        self.buttons.rejected.connect(self.reject)

    def center(self):
        self.x_spin.interpretText()
        self.y_spin.interpretText()
        return self.x_spin.value(), self.y_spin.value()

    def set_status(self, text, level="info"):
        self.status_label.setText(text)
        set_role(self.status_label, "success" if level == "ok" else level)
