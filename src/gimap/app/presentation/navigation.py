"""Application navigation: one collapsible sidebar of workspace entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.assets import ICON_ROOT, app_icon
from src.gimap.app.presentation.i18n import tr


@dataclass(frozen=True)
class NavigationItem:
    key: str
    title: str
    icon: str
    section: str
    description: str = ""


NAVIGATION_ITEMS = (
    NavigationItem("home", "Start", "GIMaP Logo.svg", "Workspaces",
                   "Open data and choose what you want to know"),
    NavigationItem("analyze", "Analyze", "Analyze.svg", "Workspaces",
                   "GISAXS and GIWAXS step by step: data, geometry, mask, cuts, results, export"),
    NavigationItem("fitting", "Fitting", "1D_Cut.svg", "Workspaces",
                   "Fit 1D curves with particle models: one curve or an in-situ series"),
    NavigationItem("compare", "Compare", "Compare.svg", "Workspaces",
                   "Runs or samples side by side: odd frames, stages, how fast each changed, which are alike"),
    NavigationItem("predict", "2D Prediction", "Predict.svg", "Labs",
                   "Machine-learning prediction from 2D patterns"),
    NavigationItem("trainset", "Trainset Build", "TraintingSetBuild.svg", "Labs",
                   "Simulate training sets"),
)
"""Workspaces in sidebar order: everyday analysis first, then the
experimental machine-learning tools ("Labs")."""


class NavigationSidebar(QWidget):
    """Workspace switcher: icon + label entries that collapse to an icon rail."""

    pageRequested = pyqtSignal(str)
    collapsedChanged = pyqtSignal(bool)

    EXPANDED_WIDTH = 184
    COLLAPSED_WIDTH = 52
    ICON_SIZE = 20

    def __init__(
        self,
        items: Sequence[NavigationItem] = NAVIGATION_ITEMS,
        parent: QWidget | None = None,
        *,
        collapsed: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("navigationSidebar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.items = tuple(items)
        self._buttons: dict[str, QToolButton] = {}
        self._section_labels: list[QLabel] = []
        self._dividers: list[QFrame] = []
        self._collapsed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 8)
        layout.setSpacing(2)

        brand = QWidget(self)
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(6, 0, 0, 8)
        brand_layout.setSpacing(8)
        self.logo_label = QLabel(brand)
        self.logo_label.setObjectName("navigationLogo")
        self.logo_label.setPixmap(app_icon().pixmap(28, 28))
        self.logo_label.setFixedSize(28, 28)
        self.logo_label.setToolTip("GIMaP")
        self.brand_label = QLabel("GIMaP", brand)
        self.brand_label.setObjectName("navigationBrandTitle")
        brand_layout.addWidget(self.logo_label)
        brand_layout.addWidget(self.brand_label, 1)
        layout.addWidget(brand)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        section = None
        for item in self.items:
            if item.section != section:
                section = item.section
                if self._buttons:
                    divider = QFrame(self)
                    divider.setObjectName("navigationDivider")
                    divider.setFixedHeight(1)
                    self._dividers.append(divider)
                    layout.addWidget(divider)
                label = QLabel(section.upper(), self)
                label.setObjectName("navigationSectionLabel")
                self._section_labels.append(label)
                layout.addWidget(label)
            layout.addWidget(self._create_button(item))
        layout.addStretch(1)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("navigationToggle")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)

        self.set_collapsed(collapsed, emit_signal=False)

    def _create_button(self, item: NavigationItem) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(f"navigation_{item.key}")
        button.setProperty("navigationItem", True)
        button.setCheckable(True)
        button.setAutoRaise(True)
        # Leading spaces separate the label from the icon; a single & is a mnemonic.
        button.setText("  " + item.title.replace("&", "&&"))
        button.setToolTip(f"{item.title}\n{item.description}" if item.description else item.title)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setMinimumHeight(34)
        icon_path = ICON_ROOT / item.icon
        if icon_path.is_file():
            button.setIcon(QIcon(str(icon_path)))
            button.setIconSize(QSize(self.ICON_SIZE, self.ICON_SIZE))
        button.clicked.connect(lambda _checked=False, key=item.key: self._request(key))
        self.button_group.addButton(button)
        self._buttons[item.key] = button
        return button

    def _request(self, key: str) -> None:
        self.set_active(key)
        self.pageRequested.emit(key)

    def button(self, key: str) -> QToolButton:
        return self._buttons[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._buttons)

    def set_active(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def active_key(self) -> str | None:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return None

    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool, *, emit_signal: bool = True) -> None:
        self._collapsed = bool(collapsed)
        style = Qt.ToolButtonIconOnly if self._collapsed else Qt.ToolButtonTextBesideIcon
        for button in self._buttons.values():
            button.setToolButtonStyle(style)
        for label in self._section_labels:
            label.setVisible(not self._collapsed)
        # Collapsed, the rail shows one G: the Start entry's icon is the logo.
        self.logo_label.setVisible(not self._collapsed)
        self.brand_label.setVisible(not self._collapsed)
        # Texts set again at run time go through tr(): the table is applied only when a window is shown.
        self.toggle_button.setText(tr("»") if self._collapsed else tr("«  Collapse"))
        self.toggle_button.setToolTip(tr("Expand sidebar") if self._collapsed else tr("Collapse sidebar"))
        self.setFixedWidth(self.COLLAPSED_WIDTH if self._collapsed else self.EXPANDED_WIDTH)
        if emit_signal:
            self.collapsedChanged.emit(self._collapsed)


__all__ = ["NAVIGATION_ITEMS", "NavigationItem", "NavigationSidebar"]
