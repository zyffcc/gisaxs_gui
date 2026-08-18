"""Prediction-specific visual cards。"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import CollapsibleCardFrame
from src.gimap.app.presentation.layout_primitives import CARD_SPACING, SECTION_MIN_WIDTH, normalize_button
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value


class PredictCard(CollapsibleCardFrame):
    """Collapsible card whose expansion state is scoped to Prediction。"""

    SETTINGS_PREFIX = "gisaxs_predict/cards"

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        *,
        default_expanded: bool = True,
    ) -> None:
        super().__init__(
            title,
            object_name,
            parent,
            default_expanded=default_expanded,
            settings_prefix=self.SETTINGS_PREFIX,
        )
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


class PredictModelLibraryCard(PredictCard):
    """Small browser entry point for remotely hosted prediction models。"""

    MODEL_LIBRARY_URL = "https://syncandshare.desy.de/index.php/s/ZMF7r57KgefPS2W"

    def __init__(self, parent: QWidget | None = None, profile=None) -> None:
        super().__init__(
            "Model Library",
            "predictModelLibraryCard",
            parent,
            default_expanded=False,
        )
        profile = profile or current_profile(parent)
        self.setMinimumHeight(scale_value(54, profile, 46))

        content = QWidget(self.content_widget)
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_SPACING)

        text_column = QWidget(content)
        text_layout = QVBoxLayout(text_column)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)

        description = QLabel(
            "Browse the shared DESY model repository, download a model, then use Model Import.",
            text_column,
        )
        description.setObjectName("predictModelLibraryDescription")
        description.setProperty("cardBody", True)
        description.setWordWrap(True)
        text_layout.addWidget(description)

        url_label = QLabel(self.MODEL_LIBRARY_URL, text_column)
        url_label.setObjectName("predictModelLibraryUrl")
        url_label.setProperty("cardMeta", True)
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_layout.addWidget(url_label)

        self.open_button = QPushButton("Browse Models", content)
        self.open_button.setObjectName("gisaxsPredictBrowseModelLibraryButton")
        self.open_button.setToolTip(self.MODEL_LIBRARY_URL)
        normalize_button(self.open_button, wide=True)
        self.open_button.clicked.connect(self.open_model_library)

        layout.addWidget(text_column, 1)
        layout.addWidget(self.open_button, 0, Qt.AlignVCenter)
        self.add_content(content)

    def open_model_library(self) -> None:
        QDesktopServices.openUrl(QUrl(self.MODEL_LIBRARY_URL))


__all__ = ["PredictCard", "PredictModelLibraryCard"]
