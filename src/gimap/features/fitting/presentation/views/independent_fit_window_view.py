"""Hand-maintained Python View for the independent fitting curve window."""

from PyQt5 import QtCore, QtWidgets


class IndependentFitWindowView:
    """Create the enlarged curve projection and its display-only controls."""

    def setupUi(self, window):
        window.setObjectName("IndependentFitWindow")
        window.resize(1040, 720)
        self.centralwidget = QtWidgets.QWidget(window)
        self.centralwidget.setObjectName("centralwidget")
        self.fitWindowLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.fitWindowLayout.setContentsMargins(12, 10, 12, 12)
        self.fitWindowLayout.setSpacing(8)

        self.toolbarHost = QtWidgets.QWidget(self.centralwidget)
        self.toolbarHost.setObjectName("toolbarHost")
        self.toolbarHostLayout = QtWidgets.QVBoxLayout(self.toolbarHost)
        self.toolbarHostLayout.setContentsMargins(0, 0, 0, 0)
        self.fitWindowLayout.addWidget(self.toolbarHost)

        self.fitControls = QtWidgets.QWidget(self.centralwidget)
        self.fitControls.setObjectName("fitControls")
        self.fitControlsLayout = QtWidgets.QHBoxLayout(self.fitControls)
        self.fitControlsLayout.setContentsMargins(0, 0, 0, 0)
        self.fitControlsLayout.setSpacing(8)

        self.qViewLabel = QtWidgets.QLabel("q View", self.fitControls)
        self.q_view_combo = QtWidgets.QComboBox(self.fitControls)
        self.q_view_combo.setObjectName("q_view_combo")
        self.fitControlsLayout.addWidget(self.qViewLabel)
        self.fitControlsLayout.addWidget(self.q_view_combo)

        self.layersLabel = QtWidgets.QLabel("Layers", self.fitControls)
        self.curve_mode_combo = QtWidgets.QComboBox(self.fitControls)
        self.curve_mode_combo.setObjectName("curve_mode_combo")
        self.fitControlsLayout.addWidget(self.layersLabel)
        self.fitControlsLayout.addWidget(self.curve_mode_combo)

        self.log_x_cb = QtWidgets.QCheckBox("Log X", self.fitControls)
        self.log_x_cb.setObjectName("log_x_cb")
        self.log_y_cb = QtWidgets.QCheckBox("Log Y", self.fitControls)
        self.log_y_cb.setObjectName("log_y_cb")
        self.normalize_cb = QtWidgets.QCheckBox("Normalize", self.fitControls)
        self.normalize_cb.setObjectName("normalize_cb")
        self.fitControlsLayout.addWidget(self.log_x_cb)
        self.fitControlsLayout.addWidget(self.log_y_cb)
        self.fitControlsLayout.addWidget(self.normalize_cb)

        self.qUnitLabel = QtWidgets.QLabel("Unit", self.fitControls)
        self.q_unit_combo = QtWidgets.QComboBox(self.fitControls)
        self.q_unit_combo.setObjectName("q_unit_combo")
        self.fitControlsLayout.addWidget(self.qUnitLabel)
        self.fitControlsLayout.addWidget(self.q_unit_combo)

        self.yRangeLabel = QtWidgets.QLabel("Y Range", self.fitControls)
        self.y_range_combo = QtWidgets.QComboBox(self.fitControls)
        self.y_range_combo.setObjectName("y_range_combo")
        self.fitControlsLayout.addWidget(self.yRangeLabel)
        self.fitControlsLayout.addWidget(self.y_range_combo)

        self.delete_input_points_cb = QtWidgets.QCheckBox("Delete Points", self.fitControls)
        self.delete_input_points_cb.setObjectName("delete_input_points_cb")
        self.delete_input_points_cb.setToolTip(
            "Enable, then left-click a plotted point to exclude it from AI fitting input."
        )
        self.fitControlsLayout.addWidget(self.delete_input_points_cb)
        self.fitControlsLayout.addStretch(1)
        self.fitWindowLayout.addWidget(self.fitControls)

        self.canvasHost = QtWidgets.QWidget(self.centralwidget)
        self.canvasHost.setObjectName("canvasHost")
        self.canvasHostLayout = QtWidgets.QVBoxLayout(self.canvasHost)
        self.canvasHostLayout.setContentsMargins(0, 0, 0, 0)
        self.fitWindowLayout.addWidget(self.canvasHost, 1)
        window.setCentralWidget(self.centralwidget)
        window.setWindowTitle("GIMaP Curve Viewer")
        QtCore.QMetaObject.connectSlotsByName(window)


__all__ = ["IndependentFitWindowView"]
