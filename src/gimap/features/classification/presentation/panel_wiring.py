"""Small wiring helpers for feature-owned Classification panels."""

from __future__ import annotations

from PyQt5.QtWidgets import QFrame, QHeaderView, QWidget

from .views import ClassificationApplyPanelView, ClassificationExplorationPanelView


def create_exploration_panel(page: QWidget) -> QWidget:
    """Create the exploration view and expose its stable binding seam."""
    panel = QFrame(page)
    ui = ClassificationExplorationPanelView()
    ui.setupUi(panel)
    page._exploration_panel_ui = ui
    for name in (
        "embeddingMethodCombo",
        "embeddingColorCombo",
        "runEmbeddingButton",
        "fitEmbeddingButton",
        "clusterMethodCombo",
        "clusterCountSpinBox",
        "minClusterSizeSpinBox",
        "suggestGroupsButton",
        "embeddingScatterView",
        "embeddingGraphicsView",
        "explorationSplitter",
        "explorationSelectionPanel",
        "selectedCountLabel",
        "selectAllEmbeddingButton",
        "clearEmbeddingSelectionButton",
        "labelEdit",
        "assignLabelButton",
        "clearLabelButton",
        "acceptSuggestionsButton",
        "acceptAllSuggestionsButton",
        "explorationSampleLabel",
        "explorationPreviewView",
        "selectionTable",
        "explorationStatusLabel",
    ):
        setattr(page, name, getattr(ui, name))
    page.selectionTable.verticalHeader().setVisible(False)
    page.selectionTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    return panel


def create_apply_panel(page: QWidget) -> QWidget:
    """Create the apply/export view and expose its stable binding seam."""
    panel = QFrame(page)
    ui = ClassificationApplyPanelView()
    ui.setupUi(panel)
    page._apply_panel_ui = ui
    for name in (
        "activePackageLabel",
        "saveActiveModelButton",
        "loadModelButton",
        "predictNewDataButton",
        "predictionTable",
        "exportPredictionsButton",
        "exportResultsButton",
    ):
        setattr(page, name, getattr(ui, name))
    page.predictionTable.verticalHeader().setVisible(False)
    page.predictionTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    return panel


__all__ = ["create_apply_panel", "create_exploration_panel"]
