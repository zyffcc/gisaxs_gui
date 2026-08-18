"""Build the Fitting page shell section."""

from PyQt5 import QtCore, QtWidgets


class PageShellMixin:
    """Own the Fitting page shell widgets."""

    def _setup_page_shell(self, gisaxsFittingPage):
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(gisaxsFittingPage)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.gisaxsFittingPageScrollArea = QtWidgets.QScrollArea(gisaxsFittingPage)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.gisaxsFittingPageScrollArea.sizePolicy().hasHeightForWidth()
        )
        self.gisaxsFittingPageScrollArea.setSizePolicy(sizePolicy)
        self.gisaxsFittingPageScrollArea.setWidgetResizable(True)
        self.gisaxsFittingPageScrollArea.setObjectName("gisaxsFittingPageScrollArea")
        self.gisaxsFittingPageScrollAreaWidgetContents = QtWidgets.QWidget()
        self.gisaxsFittingPageScrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1018, 1005))
        self.gisaxsFittingPageScrollAreaWidgetContents.setObjectName(
            "gisaxsFittingPageScrollAreaWidgetContents"
        )
        self.gridLayout_38 = QtWidgets.QGridLayout(self.gisaxsFittingPageScrollAreaWidgetContents)
        self.gridLayout_38.setObjectName("gridLayout_38")
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_38.addItem(spacerItem, 2, 0, 1, 1)
        self.gisaxsInputBox = QtWidgets.QGroupBox(self.gisaxsFittingPageScrollAreaWidgetContents)
