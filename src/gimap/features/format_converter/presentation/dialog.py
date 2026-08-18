"""Feature-owned Format Converter dialog and stable public exports."""

from __future__ import annotations


from typing import Optional


from PyQt5.QtCore import Qt, QThread


from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
)

from src.gimap.app.bootstrap import create_standalone_legacy_context


from src.gimap.app.presentation.assets import app_icon


from .views import (
    FormatConverterDialogView,
)

from .folder_import_dialog import FolderImportDialog
from .progress_dialog import ConversionProgressDialog
from .workers import _ConversionWorker, _PreviewWorker

from .bindings.form_setup import FormSetupMixin
from .bindings.input_selection import InputSelectionMixin
from .bindings.preview import PreviewMixin
from .bindings.output_options import OutputOptionsMixin
from .bindings.workflow import WorkflowMixin
from .bindings.conversion import ConversionMixin

__all__ = ["ConversionProgressDialog", "FolderImportDialog", "FormatConverterDialog"]


class FormatConverterDialog(
    FormSetupMixin,
    InputSelectionMixin,
    PreviewMixin,
    OutputOptionsMixin,
    WorkflowMixin,
    ConversionMixin,
    QDialog,
    FormatConverterDialogView,
):
    """Full converter. Inputs are detected; there is no single/batch mode."""

    def __init__(
        self,
        parent: QWidget | None = None,
        current_file: str = "",
        app_context=None,
        view_model=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.app_context = (
            app_context
            or getattr(parent, "app_context", None)
            or getattr(view_model, "app_context", None)
            or create_standalone_legacy_context()
        )
        if view_model is None:
            # Compatibility construction path for the legacy class entry point.
            from ..bootstrap import create_format_converter_view_model

            view_model = create_format_converter_view_model(self.app_context)
        self.view_model = view_model
        self.sources = self.view_model.sources
        self.current_file = current_file
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[_PreviewWorker] = None
        self._preview_request = 0
        self._pending_preview_source = None
        self._conversion_thread: Optional[QThread] = None
        self._conversion_worker: Optional[_ConversionWorker] = None
        self._progress_dialog: Optional[ConversionProgressDialog] = None
        self._conversion_started_at = 0.0
        self._paused = False
        self._bind_form()
        if current_file and self.view_model.supports_input_path(current_file):
            self.add_paths([current_file])
