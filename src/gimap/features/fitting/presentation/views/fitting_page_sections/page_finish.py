"""Build the Fitting page finish section."""

from PyQt5 import QtCore, QtWidgets


class PageFinishMixin:
    """Own the Fitting page finish widgets."""

    def _finish_page_shell(self, gisaxsFittingPage):
        self.gridLayout_38.addWidget(self.fitBox, 1, 0, 1, 1)
        self.gisaxsFittingPageScrollArea.setWidget(self.gisaxsFittingPageScrollAreaWidgetContents)
        self.verticalLayout_19.addWidget(self.gisaxsFittingPageScrollArea)
        self.FittingTextBrowser = QtWidgets.QTextBrowser(gisaxsFittingPage)
        self.FittingTextBrowser.setMinimumSize(QtCore.QSize(0, 0))
        self.FittingTextBrowser.setMaximumSize(QtCore.QSize(16777215, 100))
        self.FittingTextBrowser.setObjectName("FittingTextBrowser")
        self.verticalLayout_19.addWidget(self.FittingTextBrowser)
