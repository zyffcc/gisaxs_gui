"""Responsive Style section for the Trainset page."""

from __future__ import annotations


from PyQt5.QtCore import QTimer, Qt


class ResponsiveStyleMixin:
    """Own the responsive style section."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_responsive_layout)
        QTimer.singleShot(80, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "stack"):
            return
        measured = self.stack.width()
        fallback = self.width() - (self.step_list.width() if hasattr(self, "step_list") else 0) - 42
        content_width = max(measured, fallback)
        if hasattr(self, "impact_responsive_stack"):
            self.impact_responsive_stack.setCurrentIndex(0 if content_width >= 1040 else 1)
        if hasattr(self, "dataset_splitter"):
            # The design form and preview have tested minimums of 480 + 340 px.
            # Keep them side-by-side on a 1280×720 screen; stack only on truly
            # narrow windows where those minimums cannot fit.
            desired_orientation = Qt.Horizontal if content_width >= 820 else Qt.Vertical
            if desired_orientation != getattr(self, "_dataset_splitter_orientation", None):
                self.dataset_splitter.setOrientation(desired_orientation)
                self._dataset_splitter_orientation = desired_orientation
            self.dataset_splitter.setStretchFactor(0, 1)
            self.dataset_splitter.setStretchFactor(1, 1)
            QTimer.singleShot(0, self._balance_dataset_splitter)
        if hasattr(self, "monitor_splitter"):
            self.monitor_splitter.setOrientation(
                Qt.Horizontal if content_width >= 900 else Qt.Vertical
            )
        if hasattr(self, "step_list"):
            self.step_list.setMaximumWidth(218 if self.width() >= 1180 else 190)

    def _balance_dataset_splitter(self) -> None:
        """Give both design panes usable space after Qt finishes the resize pass."""
        if not hasattr(self, "dataset_splitter"):
            return
        first = self.dataset_splitter.widget(0)
        second = self.dataset_splitter.widget(1)
        if self.dataset_splitter.orientation() == Qt.Vertical:
            first.setMinimumSize(0, 220)
            second.setMinimumSize(0, 220)
            available = max(440, self.dataset_splitter.height())
            self.dataset_splitter.setSizes(
                [max(220, int(available * 0.50)), max(220, int(available * 0.50))]
            )
        else:
            first.setMinimumSize(480, 0)
            second.setMinimumSize(340, 0)
            available = max(820, self.dataset_splitter.width())
            self.dataset_splitter.setSizes(
                [max(480, int(available * 0.60)), max(340, int(available * 0.40))]
            )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
                #freshTrainsetBuildPage {
                    background: #eef2f6;
                    color: #1f2937;
                    font-size: 13px;
                }
                #freshTrainsetBuildPage QLabel,
                #freshTrainsetBuildPage QCheckBox,
                #freshTrainsetBuildPage QRadioButton { color: #334155; }
                #pageTitle { color: #0f172a; font-size: 22px; font-weight: 700; }
                #pageSubtitle { color: #64748b; font-size: 13px; }
                #validationBadge, #jobState {
                    background: #eff6ff;
                    color: #1d4ed8;
                    border: 1px solid #bfdbfe;
                    border-radius: 12px;
                    padding: 5px 11px;
                    font-weight: 600;
                }
                #trainsetStepList {
                    background: #ffffff;
                    color: #475569;
                    border: 1px solid #d7dee8;
                    border-radius: 10px;
                    padding: 7px;
                    outline: 0;
                }
                #trainsetStepList::item {
                    color: #475569;
                    padding: 13px 10px;
                    margin: 2px;
                    border-radius: 7px;
                }
                #trainsetStepList::item:hover { background: #f1f5f9; }
                #trainsetStepList::item:selected {
                    background: #eaf3ff;
                    color: #1d4ed8;
                    border: 1px solid #bfdbfe;
                    font-weight: 600;
                }
                #designPreviewCard {
                    background: #ffffff;
                    border: 1px solid #d7dee8;
                    border-radius: 10px;
                }
                #freshTrainsetBuildPage QLabel[sectionTitle="true"] {
                    color: #0f172a;
                    font-size: 16px;
                    font-weight: 700;
                }
                #freshTrainsetBuildPage QLabel[cardBody="true"] { color: #64748b; }
                #freshTrainsetBuildPage QWidget[displayBar="true"] {
                    background: #f8fafc;
                    border: 1px solid #d7dee8;
                    border-radius: 7px;
                }
                #freshTrainsetBuildPage QLabel[infoPanel="true"] {
                    background: #eff6ff;
                    color: #1e40af;
                    border: 1px solid #bfdbfe;
                    border-radius: 7px;
                    padding: 9px;
                }
                #freshTrainsetBuildPage QGroupBox {
                    background: #ffffff;
                    color: #0f172a;
                    border: 1px solid #d7dee8;
                    border-radius: 9px;
                    margin-top: 13px;
                    padding: 12px 9px 9px 9px;
                    font-weight: 600;
                }
                #freshTrainsetBuildPage QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 5px;
                    background: #ffffff;
                    color: #334155;
                }
                #freshTrainsetBuildPage QLineEdit,
                #freshTrainsetBuildPage QSpinBox,
                #freshTrainsetBuildPage QDoubleSpinBox,
                #freshTrainsetBuildPage QComboBox,
                #freshTrainsetBuildPage QTextEdit,
                #freshTrainsetBuildPage QTableWidget {
                    background: #ffffff;
                    color: #111827;
                    border: 1px solid #c6cfdb;
                    border-radius: 6px;
                    padding: 5px;
                    selection-background-color: #dbeafe;
                    selection-color: #1e3a8a;
                }
                #freshTrainsetBuildPage QLineEdit:focus,
                #freshTrainsetBuildPage QSpinBox:focus,
                #freshTrainsetBuildPage QDoubleSpinBox:focus,
                #freshTrainsetBuildPage QComboBox:focus,
                #freshTrainsetBuildPage QTextEdit:focus,
                #freshTrainsetBuildPage QTableWidget:focus { border: 1px solid #60a5fa; }
                #freshTrainsetBuildPage QComboBox QAbstractItemView {
                    background: #ffffff;
                    color: #111827;
                    border: 1px solid #c6cfdb;
                    selection-background-color: #dbeafe;
                }
                #freshTrainsetBuildPage QPushButton {
                    background: #ffffff;
                    color: #334155;
                    border: 1px solid #c6cfdb;
                    border-radius: 6px;
                    padding: 7px 12px;
                    font-weight: 500;
                }
                #freshTrainsetBuildPage QPushButton:hover {
                    background: #f8fafc;
                    border-color: #94a3b8;
                }
                #freshTrainsetBuildPage QPushButton:pressed { background: #eef2f7; }
                #freshTrainsetBuildPage QPushButton:disabled {
                    color: #94a3b8;
                    background: #f1f5f9;
                    border-color: #e2e8f0;
                }
                #freshTrainsetBuildPage QPushButton#primaryAction {
                    background: #2563eb;
                    color: #ffffff;
                    border-color: #2563eb;
                    font-weight: 600;
                }
                #freshTrainsetBuildPage QPushButton#primaryAction:hover { background: #1d4ed8; }
                #freshTrainsetBuildPage QHeaderView::section {
                    background: #f8fafc;
                    color: #475569;
                    border: 0;
                    border-bottom: 1px solid #d7dee8;
                    padding: 7px;
                    font-weight: 600;
                }
                #freshTrainsetBuildPage QTabWidget::pane {
                    background: #ffffff;
                    border: 1px solid #d7dee8;
                    border-radius: 7px;
                    top: -1px;
                }
                #freshTrainsetBuildPage QTabBar::tab {
                    background: #edf1f5;
                    color: #64748b;
                    border: 1px solid #d7dee8;
                    border-bottom: 0;
                    padding: 8px 12px;
                    margin-right: 2px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                }
                #freshTrainsetBuildPage QTabBar::tab:hover { background: #e2e8f0; }
                #freshTrainsetBuildPage QTabBar::tab:selected {
                    background: #ffffff;
                    color: #1d4ed8;
                    font-weight: 600;
                }
                #designStageTabs QTabBar::tab { padding: 8px 7px; }
                #freshTrainsetBuildPage QScrollArea,
                #freshTrainsetBuildPage QStackedWidget { background: transparent; border: 0; }
                """
        )
