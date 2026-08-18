"""Hand-maintained Python View for the prediction heatmap dialog."""


from PyQt5 import QtCore, QtGui, QtWidgets


class DistributionHeatmapDialogView(object):
    def setupUi(self, DistributionHeatmapWindow):
        DistributionHeatmapWindow.setObjectName("DistributionHeatmapWindow")
        DistributionHeatmapWindow.resize(980, 680)
        self.dialog_layout = QtWidgets.QVBoxLayout(DistributionHeatmapWindow)
        self.dialog_layout.setObjectName("dialog_layout")
        self.controls_layout = QtWidgets.QHBoxLayout()
        self.controls_layout.setObjectName("controls_layout")
        self.component_label = QtWidgets.QLabel(DistributionHeatmapWindow)
        self.component_label.setObjectName("component_label")
        self.controls_layout.addWidget(self.component_label)
        self.component_combo = QtWidgets.QComboBox(DistributionHeatmapWindow)
        self.component_combo.setMinimumSize(QtCore.QSize(360, 0))
        self.component_combo.setObjectName("component_combo")
        self.controls_layout.addWidget(self.component_combo)
        self.refresh_btn = QtWidgets.QPushButton(DistributionHeatmapWindow)
        self.refresh_btn.setObjectName("refresh_btn")
        self.controls_layout.addWidget(self.refresh_btn)
        self.dialog_layout.addLayout(self.controls_layout)
        self.status_label = QtWidgets.QLabel(DistributionHeatmapWindow)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("status_label")
        self.dialog_layout.addWidget(self.status_label)
        self.plot_host = QtWidgets.QWidget(DistributionHeatmapWindow)
        self.plot_host.setObjectName("plot_host")
        self.plot_host_layout = QtWidgets.QVBoxLayout(self.plot_host)
        self.plot_host_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_host_layout.setObjectName("plot_host_layout")
        self.dialog_layout.addWidget(self.plot_host)

        self.retranslateUi(DistributionHeatmapWindow)
        QtCore.QMetaObject.connectSlotsByName(DistributionHeatmapWindow)

    def retranslateUi(self, DistributionHeatmapWindow):
        _translate = QtCore.QCoreApplication.translate
        DistributionHeatmapWindow.setWindowTitle(_translate("DistributionHeatmapWindow", "Multi-File Distribution Heatmap"))
        self.component_label.setText(_translate("DistributionHeatmapWindow", "Distribution:"))
        self.refresh_btn.setText(_translate("DistributionHeatmapWindow", "Refresh"))
        self.status_label.setText(_translate("DistributionHeatmapWindow", "Ready"))
