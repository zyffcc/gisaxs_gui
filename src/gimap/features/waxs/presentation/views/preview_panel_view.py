"""Hand-maintained WAXS preview panel layout."""


from PyQt5 import QtCore, QtGui, QtWidgets


class WaxsPreviewPanelView:
    def setupUi(self, waxsPreviewWidget):
        waxsPreviewWidget.setObjectName("waxsPreviewWidget")
        self.previewWidgetLayout = QtWidgets.QVBoxLayout(waxsPreviewWidget)
        self.previewWidgetLayout.setContentsMargins(0, 0, 0, 0)
        self.previewWidgetLayout.setSpacing(6)
        self.previewWidgetLayout.setObjectName("previewWidgetLayout")
        self.waxsViewTabs = QTabBar(waxsPreviewWidget)
        self.waxsViewTabs.setExpanding(False)
        self.waxsViewTabs.setShape(QTabBar.RoundedNorth)
        self.waxsViewTabs.setObjectName("waxsViewTabs")
        self.previewWidgetLayout.addWidget(self.waxsViewTabs)
        self.viewerHost = QtWidgets.QWidget(waxsPreviewWidget)
        self.viewerHost.setObjectName("viewerHost")
        self.viewerHostLayout = QtWidgets.QVBoxLayout(self.viewerHost)
        self.viewerHostLayout.setContentsMargins(0, 0, 0, 0)
        self.viewerHostLayout.setObjectName("viewerHostLayout")
        self.previewWidgetLayout.addWidget(self.viewerHost)
        self.waxsMetadataLabel = QtWidgets.QLabel(waxsPreviewWidget)
        self.waxsMetadataLabel.setWordWrap(True)
        self.waxsMetadataLabel.setObjectName("waxsMetadataLabel")
        self.previewWidgetLayout.addWidget(self.waxsMetadataLabel)
        self.previewWidgetLayout.setStretch(1, 1)

        self.retranslateUi(waxsPreviewWidget)
        self.waxsViewTabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(waxsPreviewWidget)

    def retranslateUi(self, waxsPreviewWidget):
        _translate = QtCore.QCoreApplication.translate
        self.waxsMetadataLabel.setText(_translate("WaxsPreviewPanel", "No file loaded"))
from PyQt5.QtWidgets import QTabBar
