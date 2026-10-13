"""Theme-token style sheet of the Fitting presentation (see app ``theme``)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.theme import style_widget

FITTING_QSS = Path(__file__).with_name("fitting_theme.qss")


def apply_fitting_style(widget: QWidget) -> None:
    style_widget(widget, FITTING_QSS)


__all__ = ["FITTING_QSS", "apply_fitting_style"]
