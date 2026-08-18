"""Hand-maintained layout for the conversion progress dialog."""

from PyQt5 import QtCore, QtWidgets


class ConversionProgressDialogView:
    def setupUi(self, ConversionProgressDialog):
        ConversionProgressDialog.setObjectName("ConversionProgressDialog")
        ConversionProgressDialog.setMinimumSize(QtCore.QSize(570, 0))
        self.dialog_layout = QtWidgets.QVBoxLayout(ConversionProgressDialog)
        self.dialog_layout.setObjectName("dialog_layout")
        self.title = QtWidgets.QLabel(ConversionProgressDialog)
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.title.setObjectName("title")
        self.dialog_layout.addWidget(self.title)
        self.detail = QtWidgets.QLabel(ConversionProgressDialog)
        self.detail.setText("")
        self.detail.setObjectName("detail")
        self.dialog_layout.addWidget(self.detail)
        self.job_status = JobStatus(ConversionProgressDialog)
        self.job_status.setObjectName("job_status")
        self.dialog_layout.addWidget(self.job_status)
        self.time_label = QtWidgets.QLabel(ConversionProgressDialog)
        self.time_label.setStyleSheet("color: #64748b;")
        self.time_label.setObjectName("time_label")
        self.dialog_layout.addWidget(self.time_label)
        self.result = QtWidgets.QLabel(ConversionProgressDialog)
        self.result.setText("")
        self.result.setWordWrap(True)
        self.result.setObjectName("result")
        self.dialog_layout.addWidget(self.result)
        self.button_row = QtWidgets.QHBoxLayout()
        self.button_row.setObjectName("button_row")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.button_row.addItem(spacerItem)
        self.open_button = QtWidgets.QPushButton(ConversionProgressDialog)
        self.open_button.setVisible(False)
        self.open_button.setObjectName("open_button")
        self.button_row.addWidget(self.open_button)
        self.report_button = QtWidgets.QPushButton(ConversionProgressDialog)
        self.report_button.setVisible(False)
        self.report_button.setObjectName("report_button")
        self.button_row.addWidget(self.report_button)
        self.close_button = QtWidgets.QPushButton(ConversionProgressDialog)
        self.close_button.setVisible(False)
        self.close_button.setObjectName("close_button")
        self.button_row.addWidget(self.close_button)
        self.dialog_layout.addLayout(self.button_row)

        self.retranslateUi(ConversionProgressDialog)
        QtCore.QMetaObject.connectSlotsByName(ConversionProgressDialog)

    def retranslateUi(self, ConversionProgressDialog):
        _translate = QtCore.QCoreApplication.translate
        ConversionProgressDialog.setWindowTitle(
            _translate("ConversionProgressDialog", "Format Converter")
        )
        self.title.setText(_translate("ConversionProgressDialog", "Preparing conversion…"))
        self.time_label.setText(_translate("ConversionProgressDialog", "Elapsed: 00:00:00"))
        self.open_button.setText(_translate("ConversionProgressDialog", "Open output folder"))
        self.report_button.setText(_translate("ConversionProgressDialog", "View report"))
        self.close_button.setText(_translate("ConversionProgressDialog", "Close"))


from src.gimap.app.presentation import JobStatus
