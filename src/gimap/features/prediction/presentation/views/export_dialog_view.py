"""Hand-maintained Python View for the prediction export dialog."""


from PyQt5 import QtCore, QtWidgets


class ExportDialogView(object):
    def setupUi(self, ExportDialog):
        ExportDialog.setObjectName("ExportDialog")
        ExportDialog.setMinimumSize(QtCore.QSize(400, 300))
        ExportDialog.setMaximumSize(QtCore.QSize(400, 300))
        self.dialog_layout = QtWidgets.QVBoxLayout(ExportDialog)
        self.dialog_layout.setObjectName("dialog_layout")
        self.range_box = QtWidgets.QGroupBox(ExportDialog)
        self.range_box.setObjectName("range_box")
        self.range_layout = QtWidgets.QVBoxLayout(self.range_box)
        self.range_layout.setObjectName("range_layout")
        self.all_radio = QtWidgets.QRadioButton(self.range_box)
        self.all_radio.setChecked(True)
        self.all_radio.setObjectName("all_radio")
        self.range_layout.addWidget(self.all_radio)
        self.selected_radio = QtWidgets.QRadioButton(self.range_box)
        self.selected_radio.setObjectName("selected_radio")
        self.range_layout.addWidget(self.selected_radio)
        self.current_radio = QtWidgets.QRadioButton(self.range_box)
        self.current_radio.setObjectName("current_radio")
        self.range_layout.addWidget(self.current_radio)
        self.dialog_layout.addWidget(self.range_box)
        self.type_box = QtWidgets.QGroupBox(ExportDialog)
        self.type_box.setObjectName("type_box")
        self.type_layout = QtWidgets.QVBoxLayout(self.type_box)
        self.type_layout.setObjectName("type_layout")
        self.jsonl_check = QtWidgets.QCheckBox(self.type_box)
        self.jsonl_check.setChecked(True)
        self.jsonl_check.setObjectName("jsonl_check")
        self.type_layout.addWidget(self.jsonl_check)
        self.jpg_check = QtWidgets.QCheckBox(self.type_box)
        self.jpg_check.setObjectName("jpg_check")
        self.type_layout.addWidget(self.jpg_check)
        self.ascii_check = QtWidgets.QCheckBox(self.type_box)
        self.ascii_check.setObjectName("ascii_check")
        self.type_layout.addWidget(self.ascii_check)
        self.dialog_layout.addWidget(self.type_box)
        self.button_box = QtWidgets.QDialogButtonBox(ExportDialog)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.button_box.setObjectName("button_box")
        self.dialog_layout.addWidget(self.button_box)

        self.retranslateUi(ExportDialog)
        QtCore.QMetaObject.connectSlotsByName(ExportDialog)

    def retranslateUi(self, ExportDialog):
        _translate = QtCore.QCoreApplication.translate
        ExportDialog.setWindowTitle(_translate("ExportDialog", "Export Prediction Results"))
        self.range_box.setTitle(_translate("ExportDialog", "Export Range"))
        self.all_radio.setText(_translate("ExportDialog", "All Results"))
        self.selected_radio.setText(_translate("ExportDialog", "Selected Results"))
        self.current_radio.setText(_translate("ExportDialog", "Current Display"))
        self.type_box.setTitle(_translate("ExportDialog", "Export Type"))
        self.jsonl_check.setText(_translate("ExportDialog", "Structured JSONL/NDJSON"))
        self.jpg_check.setText(_translate("ExportDialog", "JPG Images (in folder)"))
        self.ascii_check.setText(_translate("ExportDialog", "1D Curve ASCII files"))
