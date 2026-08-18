"""Hand-maintained Python View for the independent image window."""


from PyQt5 import QtCore, QtGui, QtWidgets


class IndependentImageWindowView(object):
    def setupUi(self, IndependentMatplotlibWindow):
        IndependentMatplotlibWindow.setObjectName("IndependentMatplotlibWindow")
        IndependentMatplotlibWindow.resize(900, 700)
        self.centralwidget = QtWidgets.QWidget(IndependentMatplotlibWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.viewerContentLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.viewerContentLayout.setContentsMargins(9, 9, 9, 9)
        self.viewerContentLayout.setObjectName("viewerContentLayout")
        IndependentMatplotlibWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(IndependentMatplotlibWindow)
        QtCore.QMetaObject.connectSlotsByName(IndependentMatplotlibWindow)

    def retranslateUi(self, IndependentMatplotlibWindow):
        _translate = QtCore.QCoreApplication.translate
        IndependentMatplotlibWindow.setWindowTitle(_translate("IndependentMatplotlibWindow", "GIMaP Image Viewer - Independent Window (right-click to select)"))
