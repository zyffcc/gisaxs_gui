"""Presentation-only export choices for a fitting curve."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import apply_design_system


@dataclass(frozen=True)
class FittingExportSelection:
    source: str
    preparation: str


class FittingDataExportDialog(QDialog):
    """Make the exported data representation explicit before choosing a path."""

    def __init__(self, sources: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fittingDataExportDialog")
        self.setWindowTitle("Export fitting data")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Choose exactly which curve representation to write. The file header records "
            "the branch, combination, scale and fitting range.",
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.source_combo = QComboBox(self)
        self.source_combo.setObjectName("fittingExportSourceComboBox")
        self.source_combo.addItems(sources)
        self.preparation_combo = QComboBox(self)
        self.preparation_combo.setObjectName("fittingExportPreparationComboBox")
        self.preparation_combo.addItem("Data used for fitting (current range)", "fitting")
        self.preparation_combo.addItem("Prepared full curve", "prepared")
        self.preparation_combo.addItem("Raw signed source curve", "raw")
        form.addRow("Source", self.source_combo)
        form.addRow("Representation", self.preparation_combo)
        layout.addLayout(form)

        self.summary_label = QLabel(self)
        self.summary_label.setObjectName("fittingExportSummaryLabel")
        self.summary_label.setWordWrap(True)
        self.summary_label.setProperty("cardMeta", True)
        layout.addWidget(self.summary_label)
        self.preparation_combo.currentIndexChanged.connect(self._refresh_summary)
        self._refresh_summary()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        apply_design_system(self)

    def selection(self) -> FittingExportSelection:
        return FittingExportSelection(
            source=self.source_combo.currentText(),
            preparation=str(self.preparation_combo.currentData()),
        )

    def _refresh_summary(self) -> None:
        descriptions = {
            "fitting": "Applies the visible branch/combination and the current fitting region.",
            "prepared": "Applies the visible branch/combination over its full valid domain.",
            "raw": "Keeps original signed q coordinates and bypasses branch/combination filtering.",
        }
        self.summary_label.setText(descriptions[str(self.preparation_combo.currentData())])


__all__ = ["FittingDataExportDialog", "FittingExportSelection"]
