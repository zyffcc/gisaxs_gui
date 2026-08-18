"""Progress Dialog for Format Converter."""

from __future__ import annotations


from PyQt5.QtCore import QUrl

from PyQt5.QtGui import QDesktopServices

from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
)


from src.gimap.app.presentation import apply_design_system

from src.gimap.app.presentation.assets import app_icon


from .views import (
    ConversionProgressDialogView,
)


class ConversionProgressDialog(QDialog, ConversionProgressDialogView):
    def __init__(self, destination: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.destination = destination
        self.report_path = ""
        self.running = True
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.setModal(False)
        self.job_status.set_actions_visible(details=False)
        self.bar = self.job_status.progress_bar
        self.pause_button = self.job_status.pause_button
        self.cancel_button = self.job_status.cancel_button
        self.open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.destination))
        )
        self.report_button.clicked.connect(self._open_report)
        self.close_button.clicked.connect(self.accept)
        apply_design_system(self)

    def complete(self, report) -> None:
        self.running = False
        self.report_path = report.report_path
        self.bar.setValue(self.bar.maximum())
        if report.cancelled:
            self.title.setText("Conversion cancelled")
            self.job_status.set_state("cancelled", "Conversion cancelled", progress=1.0)
        else:
            self.title.setText("Conversion completed")
            self.job_status.set_state("succeeded", "Conversion completed", progress=1.0)
        self.result.setText(f"{len(report.succeeded)} succeeded\n{len(report.failed)} failed")
        self.pause_button.hide()
        self.cancel_button.hide()
        self.open_button.show()
        self.report_button.setVisible(bool(self.report_path))
        self.close_button.show()

    def fail(self, message: str) -> None:
        self.running = False
        self.title.setText("Conversion could not be completed")
        self.result.setText(message)
        self.job_status.set_state("failed", message, progress=0.0)
        self.pause_button.hide()
        self.cancel_button.hide()
        self.open_button.show()
        self.close_button.show()

    def _open_report(self) -> None:
        if self.report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report_path))

    def closeEvent(self, event) -> None:
        if self.running:
            event.ignore()
            return
        event.accept()
