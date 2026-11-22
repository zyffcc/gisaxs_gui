"""Remember the user's first confirmation for each model service."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from src.gimap.app.presentation.i18n import tr, trf
from . import preferences

NOTICE = (
    "The AI assistant sends your instructions, frame information and analysis results to {service}. "
    "If you enable image previews, it also sends an image of the frame. Allow requests to this service?"
)


class ServiceNotice(QDialog):
    def __init__(self, parent, service: str):
        super().__init__(parent)
        self.setWindowTitle(tr("Share data with the AI service?"))
        self.message = QLabel(trf(NOTICE, service=service), self)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setWordWrap(True)
        buttons = QDialogButtonBox(self)
        allow = buttons.addButton(tr("Allow requests"), QDialogButtonBox.AcceptRole)
        cancel = buttons.addButton(tr("Cancel"), QDialogButtonBox.RejectRole)
        allow.setAutoDefault(False)
        cancel.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)
        layout.addWidget(buttons)
        self.setMinimumWidth(400)


def allow_requests(parent, settings, service: str) -> bool:
    accepted = preferences.read(settings, "approved_services")
    if service in accepted:
        return True
    if ServiceNotice(parent, service).exec_() != QDialog.Accepted:
        return False
    preferences.write(settings, "approved_services", [*accepted, service])
    return True
