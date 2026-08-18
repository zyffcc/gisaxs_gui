"""Hand-maintained WAXS configure panel layout."""


from PyQt5 import QtCore, QtWidgets


class WaxsConfigurePanelView:
    def setupUi(self, waxsControlTabs):
        waxsControlTabs.setObjectName("waxsControlTabs")
        self.roiTab = QtWidgets.QWidget()
        self.roiTab.setObjectName("roiTab")
        self.roiTabLayout = QtWidgets.QVBoxLayout(self.roiTab)
        self.roiTabLayout.setContentsMargins(0, 0, 0, 0)
        self.roiTabLayout.setObjectName("roiTabLayout")
        waxsControlTabs.addTab(self.roiTab, "")
        self.integrationTab = QtWidgets.QWidget()
        self.integrationTab.setObjectName("integrationTab")
        self.integrationTabLayout = QtWidgets.QVBoxLayout(self.integrationTab)
        self.integrationTabLayout.setContentsMargins(0, 0, 0, 0)
        self.integrationTabLayout.setObjectName("integrationTabLayout")
        waxsControlTabs.addTab(self.integrationTab, "")

        self.retranslateUi(waxsControlTabs)
        waxsControlTabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(waxsControlTabs)

    def retranslateUi(self, waxsControlTabs):
        _translate = QtCore.QCoreApplication.translate
        waxsControlTabs.setTabText(waxsControlTabs.indexOf(self.roiTab), _translate("WaxsConfigurePanel", "ROI / Cut"))
        waxsControlTabs.setTabText(waxsControlTabs.indexOf(self.integrationTab), _translate("WaxsConfigurePanel", "1D Integration"))
