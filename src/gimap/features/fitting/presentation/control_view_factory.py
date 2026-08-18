"""Compatibility factory that installs the Fitting Python View."""

from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from .views import FittingPageView


def _expose_view_attributes(ui, view: FittingPageView) -> None:
    """Preserve the legacy Ui_MainWindow attribute contract."""
    for name, value in vars(view).items():
        if not name.startswith("_"):
            setattr(ui, name, value)


def build_fitting_controls(ui) -> None:
    """Create the feature-owned Fitting page inside the application stack."""
    page = QWidget()
    view = FittingPageView()
    view.setupUi(page)
    ui._fitting_page_view = view
    ui.gisaxsFittingPage = page
    _expose_view_attributes(ui, view)
    ui.mainWindowWidget.addWidget(page)


def translate_fitting_controls(ui, _translate) -> None:
    """Retranslate the form while retaining the historical helper API."""
    del _translate
    view = getattr(ui, "_fitting_page_view", None)
    if view is not None:
        view.retranslateUi(ui.gisaxsFittingPage)


__all__ = ["build_fitting_controls", "translate_fitting_controls"]
