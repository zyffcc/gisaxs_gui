"""Folder Import Dialog for Format Converter."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QMessageBox,
    QWidget,
)

from src.gimap.app.bootstrap import create_standalone_legacy_context


from src.gimap.app.presentation.assets import app_icon


from .views import (
    FolderImportDialogView,
)


class FolderImportDialog(QDialog, FolderImportDialogView):
    def __init__(self, parent: QWidget | None = None, view_model=None):
        super().__init__(parent)
        self.view_model = view_model or getattr(parent, "view_model", None)
        if self.view_model is None:
            from ..bootstrap import create_format_converter_view_model

            self.view_model = create_format_converter_view_model(create_standalone_legacy_context())
        self.paths: list[str] = []
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.browse_button.clicked.connect(self._browse)
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select input folder", self.path_edit.text()
        )
        if folder:
            self.path_edit.setText(self.view_model.normalize_path(folder))

    def _accept_if_valid(self) -> None:
        if not any(check.isChecked() for check in (self.cbf, self.tiff, self.nxs)):
            QMessageBox.warning(self, "Add Folder", "Select at least one input format.")
            return
        try:
            self.paths = self.view_model.scan_folder(
                self.path_edit.text(),
                include_cbf=self.cbf.isChecked(),
                include_tiff=self.tiff.isChecked(),
                include_nxs=self.nxs.isChecked(),
                recursive=self.recursive.isChecked(),
            )
        except NotADirectoryError:
            QMessageBox.warning(self, "Add Folder", "Please select a valid folder.")
            return
        self.accept()
