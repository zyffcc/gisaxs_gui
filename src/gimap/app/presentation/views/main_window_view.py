"""Shell widgets of the main window: page stack, menu bar and status bar.

Feature pages replace the ``*PageHost`` placeholders of ``mainWindowWidget``;
the composition root puts the navigation sidebar in front of
``mainContentWidget`` and builds the menus.
"""

from PyQt5 import QtCore, QtWidgets


class MainWindowView(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)

        self.mainContentWidget = QtWidgets.QWidget(self.centralwidget)
        self.mainContentWidget.setObjectName("mainContentWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.mainContentWidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.mainWindowWidget = QtWidgets.QStackedWidget(self.mainContentWidget)
        self.mainWindowWidget.setObjectName("mainWindowWidget")
        self.mainWindowWidget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.trainsetBuildPage = QtWidgets.QWidget()
        self.trainsetBuildPage.setObjectName("trainsetBuildPage")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.trainsetBuildPage)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.mainWindowWidget.addWidget(self.trainsetBuildPage)
        self.gisaxsPredictPageHost = QtWidgets.QWidget()
        self.gisaxsPredictPageHost.setObjectName("gisaxsPredictPageHost")
        self.mainWindowWidget.addWidget(self.gisaxsPredictPageHost)
        self.gisaxsFittingPageHost = QtWidgets.QWidget()
        self.gisaxsFittingPageHost.setObjectName("gisaxsFittingPageHost")
        self.mainWindowWidget.addWidget(self.gisaxsFittingPageHost)
        self.classificationPage = QtWidgets.QWidget()
        self.classificationPage.setObjectName("classificationPage")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.classificationPage)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.mainWindowWidget.addWidget(self.classificationPage)
        self.analyzePageHost = QtWidgets.QWidget()
        self.analyzePageHost.setObjectName("analyzePageHost")
        self.mainWindowWidget.addWidget(self.analyzePageHost)
        self.verticalLayout_2.addWidget(self.mainWindowWidget)
        self.horizontalLayout.addWidget(self.mainContentWidget, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.mainWindowWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GIMaP"))
