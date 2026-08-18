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
        self.FittingTextBrowser.setStyleSheet(
            "QTextBrowser {\n"
            "  background: rgba(255,255,255,0.75);\n"
            "  color: #1d2433;\n"
            "\n"
            "  font-size: 13px;\n"
            "  border: 1px solid rgba(0,0,0,0.06);\n"
            "  border-radius: 14px;\n"
            "  padding: 14px 16px;\n"
            "}\n"
            "\n"
            "QScrollBar:vertical { width: 8px; background: transparent; }\n"
            "QScrollBar::handle:vertical { background: rgba(0,0,0,0.18); border-radius: 4px; }\n"
            "QScrollBar::handle:vertical:hover { background: rgba(0,0,0,0.28); }QTextBrowser {\n"
            "  background: rgba(255,255,255,0.75);\n"
            "  color: #1d2433;\n"
            "\n"
            "  font-size: 13px;\n"
            "  border: 1px solid rgba(0,0,0,0.06);\n"
            "  border-radius: 14px;\n"
            "  padding: 14px 16px;\n"
            "}\n"
            "\n"
            "QScrollBar:vertical { width: 8px; background: transparent; }\n"
            "QScrollBar::handle:vertical { background: rgba(0,0,0,0.18); border-radius: 4px; }\n"
            "QScrollBar::handle:vertical:hover { background: rgba(0,0,0,0.28); }"
        )
        self.FittingTextBrowser.setObjectName("FittingTextBrowser")
        self.verticalLayout_19.addWidget(self.FittingTextBrowser)
