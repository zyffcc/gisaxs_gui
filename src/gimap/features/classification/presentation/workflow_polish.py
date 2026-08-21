"""Modern workflow presentation for the Classification page."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QGraphicsView, QLabel, QPushButton


class ClassificationEmptyState(QLabel):
    """Non-interactive guidance shown while the sample preview is empty."""

    def __init__(self, view: QGraphicsView) -> None:
        super().__init__(
            "Add labeled classes and scan files\nto inspect a representative sample.",
            view.viewport(),
        )
        self.view = view
        self.setObjectName("classificationPreviewEmptyState")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        view.viewport().installEventFilter(self)
        self.refresh()

    def eventFilter(self, watched, event):
        if watched is self.view.viewport() and event.type() in (
            QEvent.Resize,
            QEvent.Show,
            QEvent.Paint,
        ):
            self.refresh()
        return False

    def refresh(self) -> None:
        scene = self.view.scene()
        self.setVisible(scene is None or not scene.items())
        self.setGeometry(self.view.viewport().rect().adjusted(24, 24, -24, -24))
        self.raise_()


def polish_classification_workflow(page) -> None:
    """Clarify navigation and visual hierarchy without changing commands."""
    page.titleLabel.setText("Classifier workbench")
    page.subtitleLabel.setText(
        "Import labeled data, define one shared preprocessing pipeline, then compare models."
    )
    for button, text in (
        (page.datasetStepButton, "1  Import dataset"),
        (page.preprocessingStepButton, "2  Preprocess"),
        (page.algorithmsStepButton, "3  Compare models"),
        (page.resultsStepButton, "4  Results"),
    ):
        button.setText(text)

    page.scanImportButton.setText("Scan and import data")
    page.scanImportButton.setProperty("classificationPrimaryAction", True)
    page.addClassButton.setText("Add labeled class")
    page.runComparisonButton.setText("Run model comparison")
    page.runComparisonButton.setProperty("classificationPrimaryAction", True)

    preprocessing_panel = page._preprocessing_panel_ui
    page.preprocessing_continue_button = QPushButton(
        "Continue to model comparison", page.preprocessingStepContent
    )
    page.preprocessing_continue_button.setObjectName("preprocessingContinueButton")
    page.preprocessing_continue_button.setProperty("classificationPrimaryAction", True)
    preprocessing_panel.preprocessingPanelLayout.insertWidget(
        max(0, preprocessing_panel.preprocessingPanelLayout.count() - 1),
        page.preprocessing_continue_button, 0, Qt.AlignRight
    )
    page.preprocessing_continue_button.clicked.connect(
        lambda _checked=False: page.set_step("Algorithms")
    )

    advanced = page.classification_algorithm_advanced
    advanced.setParent(page._experiment_panel_ui.sectionTitle.parentWidget())
    page._experiment_panel_ui.experimentPanelLayout.insertWidget(1, advanced)
    page.algorithmConfigSplitter.setHandleWidth(0)
    page.algorithmConfigSplitter.setSizes([1200])

    page.preview_empty_state = ClassificationEmptyState(page.previewGraphicsView)


__all__ = ["ClassificationEmptyState", "polish_classification_workflow"]
