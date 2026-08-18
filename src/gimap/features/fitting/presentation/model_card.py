"""Feature-owned layout for Fitting model parameter controls。"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QBoxLayout, QHBoxLayout, QSizePolicy, QWidget

from src.gimap.app.presentation.layout_primitives import CARD_SPACING
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import CardFrame
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget


class ModelParameterCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Model Parameters", "ModelParameterCard")
        profile = profile or current_profile(ui.centralwidget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        add_button = getattr(ui, "pushButton", None)
        if add_button is not None:
            _detach_from_parent_layout(add_button)
            add_button.setText("+ Add Component")
            add_button.setMinimumWidth(scale_value(220, profile, 190))
            add_button.setMaximumWidth(scale_value(320, profile, 280))
            add_button.setMinimumHeight(scale_value(36, profile, 32))
            add_button.setMaximumHeight(scale_value(40, profile, 36))
            add_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

            self.body_layout.removeWidget(self.title_label)
            header = QWidget(self)
            header.setObjectName("modelParametersHeader")
            header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(CARD_SPACING)
            self.title_label.setParent(header)
            self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            header_layout.addWidget(self.title_label)
            header_layout.addStretch(1)
            header_layout.addWidget(add_button, 0, Qt.AlignRight)
            self.body_layout.insertWidget(0, header)

        _take_widget(ui.gridLayout_24, ui.widget_7)
        ui.widget_7.setMinimumWidth(0)
        ui.widget_7.setMaximumWidth(16777215)
        ui.widget_7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        if ui.widget_7.layout() is not None:
            ui.widget_7.layout().setContentsMargins(0, 0, 0, 0)
            ui.widget_7.layout().setSpacing(0)
        inner_scroll_area = getattr(ui, "scrollArea", None)
        if inner_scroll_area is not None and ui.widget_7.layout() is not None:
            _take_widget(ui.widget_7.layout(), inner_scroll_area)
            content_widget = inner_scroll_area.takeWidget()
            if content_widget is not None:
                content_widget.setParent(ui.widget_7)
                content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                ui.widget_7.layout().addWidget(content_widget)
            inner_scroll_area.deleteLater()
        particle_layout = getattr(ui, "scrollAreaWidgetContents", None)
        particle_layout = particle_layout.layout() if particle_layout is not None else None
        if isinstance(particle_layout, QBoxLayout):
            particle_layout.setDirection(QBoxLayout.TopToBottom)
            particle_layout.setContentsMargins(0, 0, 0, 0)
            particle_layout.setSpacing(scale_value(8, profile, 6))
            particle_layout.setAlignment(Qt.AlignTop)
            for index in range(particle_layout.count()):
                particle_layout.setStretch(index, 0)
        self.body_layout.addWidget(ui.widget_7, 0)
        self.lock_to_natural_height()


__all__ = ["ModelParameterCard"]
