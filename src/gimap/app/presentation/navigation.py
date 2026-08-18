"""Application-shell navigation widgets with no feature dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt5.QtCore import QSize, QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.assets import app_colored_logo_pixmap, app_icon
from src.gimap.app.presentation.layout_primitives import normalize_button


class NavigationSidebar(QWidget):
    """Collapsible app navigation with an always-visible icon rail."""

    collapsedChanged = pyqtSignal(bool)

    SETTINGS_KEY = "main_sidebar_collapsed"
    RAIL_WIDTH = 56
    PANEL_WIDTH = 220
    PANEL_BUTTON_WIDTH = 200
    BUTTON_HEIGHT = 46
    EXPANDED_WIDTH = RAIL_WIDTH + PANEL_WIDTH
    COLLAPSED_WIDTH = RAIL_WIDTH

    PAGE_META = (
        ("Cut & Fitting", "1D_Cut.svg"),
        ("2D Prediction", "Predict.svg"),
        ("Trainset Build", "TraintingSetBuild.svg"),
        ("Classification", "Classification.svg"),
        ("WAXS", "WAXS.svg"),
    )

    def __init__(self, buttons: Sequence[QAbstractButton], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("navigationSidebar")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._buttons = list(buttons)
        self._rail_buttons: list[QToolButton] = []
        self._icon_dir = Path(__file__).resolve().parents[4] / "assets" / "icons"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = QWidget(self)
        self.rail.setObjectName("navigationIconRail")
        self.rail.setFixedWidth(self.RAIL_WIDTH)
        rail_layout = QVBoxLayout(self.rail)
        rail_layout.setContentsMargins(8, 8, 8, 8)
        rail_layout.setSpacing(8)

        self.toggle_button = QToolButton(self.rail)
        self.toggle_button.setObjectName("sidebarCollapseButton")
        self.toggle_button.setToolTip("Collapse sidebar")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setFixedSize(40, 40)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        self.rail_logo_label = QLabel(self.rail)
        self.rail_logo_label.setObjectName("navigationRailLogo")
        self.rail_logo_label.setAlignment(Qt.AlignCenter)
        self.rail_logo_label.setFixedSize(40, 40)
        self.rail_logo_label.setPixmap(app_icon().pixmap(32, 32))
        self.rail_logo_label.setToolTip("GIMaP")
        rail_layout.addWidget(self.rail_logo_label, 0, Qt.AlignHCenter)
        rail_layout.addWidget(self.toggle_button)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        for index, button in enumerate(self._buttons):
            title, icon_name = self._page_meta(index)
            rail_button = QToolButton(self.rail)
            rail_button.setObjectName(f"navigationRailButton{index}")
            rail_button.setProperty("navigationRailButton", True)
            rail_button.setCheckable(True)
            rail_button.setAutoRaise(True)
            rail_button.setFixedSize(40, 40)
            rail_button.setToolTip(title)
            icon_path = self._icon_dir / icon_name
            if icon_path.exists():
                rail_button.setIcon(QIcon(str(icon_path)))
                rail_button.setIconSize(QSize(24, 24))
            else:
                rail_button.setText(title[:1])
            rail_button.clicked.connect(lambda _checked=False, i=index: self.set_active_index(i))
            rail_button.clicked.connect(lambda _checked=False, target=button: target.click())
            self.button_group.addButton(rail_button, index)
            self._rail_buttons.append(rail_button)
            rail_layout.addWidget(rail_button)

            button.toggled.connect(lambda checked, i=index: self._sync_rail_checked(i, checked))
            button.clicked.connect(lambda _checked=False, i=index: self.set_active_index(i))

        rail_layout.addStretch(1)

        self.panel = QWidget(self)
        self.panel.setObjectName("navigationControlPanel")
        self.panel.setFixedWidth(self.PANEL_WIDTH)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(10, 10, 10, 12)
        panel_layout.setSpacing(8)

        self.brand_widget = QWidget(self.panel)
        self.brand_widget.setObjectName("navigationBrand")
        brand_layout = QHBoxLayout(self.brand_widget)
        brand_layout.setContentsMargins(8, 6, 8, 8)
        brand_layout.setSpacing(8)
        self.brand_logo_label = QLabel(self.brand_widget)
        self.brand_logo_label.setObjectName("navigationBrandLogo")
        self.brand_logo_label.setFixedSize(46, 46)
        self.brand_logo_label.setAlignment(Qt.AlignCenter)
        colored_logo = app_colored_logo_pixmap(42, 42)
        if not colored_logo.isNull():
            self.brand_logo_label.setPixmap(colored_logo)
        else:
            self.brand_logo_label.setText("G")
        self.brand_title_label = QLabel(
            '<span style="color: #081b4c;">GIM</span>'
            '<span style="color: #0b9fda;">a</span>'
            '<span style="color: #081b4c;">P</span>',
            self.brand_widget,
        )
        self.brand_title_label.setObjectName("navigationBrandTitle")
        self.brand_title_label.setTextFormat(Qt.RichText)
        self.brand_subtitle_label = QLabel("GISAXS / GIWAXS", self.brand_widget)
        self.brand_subtitle_label.setObjectName("navigationBrandSubtitle")
        brand_text = QWidget(self.brand_widget)
        brand_text_layout = QVBoxLayout(brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(1)
        brand_text_layout.addWidget(self.brand_title_label)
        brand_text_layout.addWidget(self.brand_subtitle_label)
        brand_layout.addWidget(self.brand_logo_label, 0, Qt.AlignVCenter)
        brand_layout.addWidget(brand_text, 1, Qt.AlignVCenter)
        panel_layout.addWidget(self.brand_widget, 0)

        self.controls_scroll = QScrollArea(self.panel)
        self.controls_scroll.setObjectName("navigationControlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_layout.addWidget(self.controls_scroll, 1)

        self.controls_content = QWidget(self.controls_scroll)
        self.controls_content.setObjectName("navigationControlsContent")
        controls_layout = QVBoxLayout(self.controls_content)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        for index, button in enumerate(self._buttons):
            controls_layout.addWidget(self._create_panel_button(index, button), 0, Qt.AlignHCenter)
        controls_layout.addStretch(1)
        self.controls_scroll.setWidget(self.controls_content)

        layout.addWidget(self.rail)
        layout.addWidget(self.panel)

        collapsed = QSettings().value(self.SETTINGS_KEY, False, type=bool)
        self.set_collapsed(collapsed, emit_signal=False)
        self.set_active_index(0)

    def _page_meta(self, index: int) -> tuple[str, str]:
        if index < len(self.PAGE_META):
            return self.PAGE_META[index]
        return (f"Page {index + 1}", "")

    def _create_panel_button(self, index: int, button: QAbstractButton) -> QAbstractButton:
        title, _icon_name = self._page_meta(index)
        button.setParent(self.controls_content)
        button.setProperty("navigationButton", True)
        button.setCheckable(True)
        button.setText(title)
        normalize_button(button, wide=True)
        button.setFixedWidth(self.PANEL_BUTTON_WIDTH)
        button.setMinimumHeight(self.BUTTON_HEIGHT)
        button.setMaximumHeight(self.BUTTON_HEIGHT)
        return button

    def set_active_index(self, index: int) -> None:
        if 0 <= index < len(self._rail_buttons):
            self._rail_buttons[index].setChecked(True)

    def _sync_rail_checked(self, index: int, checked: bool) -> None:
        if checked:
            self.set_active_index(index)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self.is_collapsed())

    def set_collapsed(self, collapsed: bool, *, emit_signal: bool = True) -> None:
        collapsed = bool(collapsed)
        self.apply_layout_state(collapsed)
        self.toggle_button.setArrowType(Qt.RightArrow if collapsed else Qt.LeftArrow)
        self.toggle_button.setToolTip("Expand sidebar" if collapsed else "Collapse sidebar")
        QSettings().setValue(self.SETTINGS_KEY, collapsed)
        if emit_signal:
            self.collapsedChanged.emit(collapsed)

    def apply_layout_state(self, collapsed: bool | None = None) -> None:
        if collapsed is None:
            collapsed = self.is_collapsed()
        collapsed = bool(collapsed)
        width = self.COLLAPSED_WIDTH if collapsed else self.EXPANDED_WIDTH

        if self.layout() is not None:
            self.layout().setContentsMargins(0, 0, 0, 0)
            self.layout().setSpacing(0)
        self.rail.setFixedWidth(self.RAIL_WIDTH)
        self.panel.setFixedWidth(self.PANEL_WIDTH)
        self.panel.setVisible(not collapsed)
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setFixedWidth(width)
        self.updateGeometry()
        self.adjustSize()
        self.update()

    def is_collapsed(self) -> bool:
        return not self.panel.isVisible()


__all__ = ["NavigationSidebar"]
