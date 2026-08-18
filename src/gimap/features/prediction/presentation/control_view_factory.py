"""Compatibility factory that installs the Prediction Python View."""

from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from .views import PredictionPageView


def _expose_view_attributes(ui, view: PredictionPageView) -> None:
    """Preserve the legacy Ui_MainWindow attribute contract."""
    for name, value in vars(view).items():
        if not name.startswith("_"):
            setattr(ui, name, value)


def build_prediction_controls(ui) -> None:
    """Create the feature-owned Prediction page inside the application stack."""
    page = QWidget()
    view = PredictionPageView()
    view.setupUi(page)
    ui._prediction_page_view = view
    ui.gisaxsPredictPage = page
    _expose_view_attributes(ui, view)
    ui.mainWindowWidget.addWidget(page)


def translate_prediction_controls(ui, _translate) -> None:
    """Retranslate the form while retaining the historical helper API."""
    del _translate
    view = getattr(ui, "_prediction_page_view", None)
    if view is not None:
        view.retranslateUi(ui.gisaxsPredictPage)


__all__ = ["build_prediction_controls", "translate_prediction_controls"]
