"""Hand-maintained Python View for the prediction parameter trend dialog."""


from PyQt5 import QtCore, QtWidgets


class ParameterTrendDialogView(object):
    def setupUi(self, ParameterTrendWindow):
        ParameterTrendWindow.setObjectName("ParameterTrendWindow")
        ParameterTrendWindow.resize(980, 680)
        self.dialog_layout = QtWidgets.QVBoxLayout(ParameterTrendWindow)
        self.dialog_layout.setObjectName("dialog_layout")
        self.controls_layout = QtWidgets.QHBoxLayout()
        self.controls_layout.setObjectName("controls_layout")
        self.parameter_label = QtWidgets.QLabel(ParameterTrendWindow)
        self.parameter_label.setObjectName("parameter_label")
        self.controls_layout.addWidget(self.parameter_label)
        self.parameter_combo = QtWidgets.QComboBox(ParameterTrendWindow)
        self.parameter_combo.setMinimumSize(QtCore.QSize(240, 0))
        self.parameter_combo.setObjectName("parameter_combo")
        self.controls_layout.addWidget(self.parameter_combo)
        self.refresh_btn = QtWidgets.QPushButton(ParameterTrendWindow)
        self.refresh_btn.setObjectName("refresh_btn")
        self.controls_layout.addWidget(self.refresh_btn)
        self.dialog_layout.addLayout(self.controls_layout)
        self.status_label = QtWidgets.QLabel(ParameterTrendWindow)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("status_label")
        self.dialog_layout.addWidget(self.status_label)
        self.plot_host = QtWidgets.QWidget(ParameterTrendWindow)
        self.plot_host.setObjectName("plot_host")
        self.plot_host_layout = QtWidgets.QVBoxLayout(self.plot_host)
        self.plot_host_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_host_layout.setObjectName("plot_host_layout")
        self.dialog_layout.addWidget(self.plot_host)

        self.retranslateUi(ParameterTrendWindow)
        QtCore.QMetaObject.connectSlotsByName(ParameterTrendWindow)

    def retranslateUi(self, ParameterTrendWindow):
        _translate = QtCore.QCoreApplication.translate
        ParameterTrendWindow.setWindowTitle(_translate("ParameterTrendWindow", "Multi-File Parameter Trend"))
        self.parameter_label.setText(_translate("ParameterTrendWindow", "Parameter:"))
        self.refresh_btn.setText(_translate("ParameterTrendWindow", "Refresh"))
        self.status_label.setText(_translate("ParameterTrendWindow", "Ready"))
