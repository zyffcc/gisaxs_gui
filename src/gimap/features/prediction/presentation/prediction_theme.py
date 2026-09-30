"""Theme-token style sheet of the Prediction workbench (see app ``theme``)."""

from pathlib import Path

from PyQt5.QtWidgets import QWidget

from src.gimap.app.presentation.theme import style_widget

PREDICTION_QSS = Path(__file__).with_name("prediction_theme.qss")


def apply_prediction_style(widget: QWidget) -> None:
    style_widget(widget, PREDICTION_QSS)


__all__ = ["PREDICTION_QSS", "apply_prediction_style"]
