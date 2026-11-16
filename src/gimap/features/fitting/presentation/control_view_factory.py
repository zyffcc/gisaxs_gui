"""The Fitting page of the application stack: an empty host that ``FittingWorkspace`` fills."""

from __future__ import annotations

from PyQt5.QtWidgets import QWidget


def build_fitting_controls(ui) -> None:
    """Create the Fitting page inside the application stack."""
    page = QWidget()
    page.setObjectName("gisaxsFittingPage")
    ui.gisaxsFittingPage = page
    ui.mainWindowWidget.addWidget(page)


__all__ = ["build_fitting_controls"]
