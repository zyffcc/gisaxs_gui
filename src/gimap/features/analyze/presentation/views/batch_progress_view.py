"""Static layout of the batch panel above the Results / Series tabs (behaviour: ``bindings/batch_run.py``).

While a batch runs: what is being reduced now, a bar, how many are done, the speed and the time
left, the latest fit, **Pause** and **Stop**, and how many frames are reduced at once (it can be
lowered while running). The frames done appear in the Series tab as they come (the map grows, the
newest frame's curve is drawn below it). When it ends: a summary, **Open Folder** and **Close**.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import FlowLayout


class BatchProgressPanel(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("analyzeBatchPanel")
        self.setProperty("gimapInfoCard", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setSpacing(6)
        self.title_label = QLabel("Batch Export", self)
        self.title_label.setObjectName("batchPanelTitle")
        self.title_label.setProperty("gimapRole", "strong")
        self.elapsed_label = QLabel("", self)
        self.elapsed_label.setObjectName("batchPanelElapsed")
        self.elapsed_label.setProperty("gimapRole", "muted")
        self.at_once_combo = QComboBox(self)
        self.at_once_combo.setObjectName("batchPanelAtOnce")
        self.at_once_combo.setToolTip(
            "How many frames are reduced at the same time. Fewer leaves more of the computer to other "
            "programs; the change applies to the next frames."
        )
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setObjectName("batchPanelPause")
        self.pause_button.setCheckable(True)
        self.pause_button.setToolTip("No new frames are started; the ones being reduced finish first")
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setObjectName("batchPanelStop")
        self.stop_button.setProperty("gimapDangerAction", True)
        self.stop_button.setToolTip(
            "End the batch after the frames being reduced; everything written stays, and the tables are "
            "written for the frames done"
        )
        self.open_button = QPushButton("Open Folder", self)
        self.open_button.setObjectName("batchPanelOpen")
        self.close_button = QPushButton("Close", self)
        self.close_button.setObjectName("batchPanelClose")
        header.addWidget(self.title_label, 1)
        header.addWidget(self.elapsed_label)
        # The controls after the title on one line while there is room; in a narrow panel on a line of their own,
        # wrapping further, so a running batch does not widen the panel (and squeeze the image beside it).
        controls = FlowLayout(spacing=6)
        for widget in (self.at_once_combo, self.pause_button, self.stop_button, self.open_button, self.close_button):
            controls.addWidget(widget)
        self.controls_host = QWidget(self)
        self.controls_host.setObjectName("batchPanelControls")
        self.controls_host.setLayout(controls)
        header.addWidget(self.controls_host)
        layout.addLayout(header)
        self.now_label = QLabel("", self)
        self.now_label.setObjectName("batchPanelNow")
        self.now_label.setWordWrap(True)
        layout.addWidget(self.now_label)
        self.bar = QProgressBar(self)
        self.bar.setObjectName("batchPanelBar")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        layout.addWidget(self.bar)
        self.detail_label = QLabel("", self)
        self.detail_label.setObjectName("batchPanelDetail")
        self.detail_label.setProperty("gimapRole", "muted")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        self.fit_label = QLabel("", self)
        self.fit_label.setObjectName("batchPanelFit")
        self.fit_label.setProperty("gimapRole", "muted")
        self.fit_label.setWordWrap(True)
        self.fit_label.hide()
        layout.addWidget(self.fit_label)
        self.follow_check = QCheckBox("Show the newest frame", self)
        self.follow_check.setObjectName("batchPanelFollow")
        self.follow_check.setChecked(True)
        self.follow_check.setToolTip(
            "In the Series tab, the curve below the map follows the newest frame done; click a row of the "
            "map to look at another one"
        )
        layout.addWidget(self.follow_check)
        self.hide()


__all__ = ["BatchProgressPanel"]
