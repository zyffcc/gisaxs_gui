"""Hand-maintained model application workspace layout."""

from PyQt5 import QtWidgets


class ClassificationApplyPanelView:
    def setupUi(self, panel):
        panel.setObjectName("classificationApplyPanel")
        self.rootLayout = QtWidgets.QVBoxLayout(panel)
        self.rootLayout.setContentsMargins(0, 0, 0, 0)
        self.rootLayout.setSpacing(10)
        self.modelSummaryFrame = QtWidgets.QFrame(panel)
        self.modelSummaryFrame.setObjectName("applyModelSummaryFrame")
        self.modelSummaryLayout = QtWidgets.QGridLayout(self.modelSummaryFrame)
        self.modelSummaryLayout.setContentsMargins(0, 0, 0, 0)
        self.modelTitle = QtWidgets.QLabel("Model", self.modelSummaryFrame)
        self.activePackageLabel = QtWidgets.QLabel("No model loaded or selected", self.modelSummaryFrame)
        self.activePackageLabel.setObjectName("activePackageLabel")
        self.saveActiveModelButton = QtWidgets.QPushButton("Save active model", self.modelSummaryFrame)
        self.saveActiveModelButton.setObjectName("saveActiveModelButton")
        self.loadModelButton = QtWidgets.QPushButton("Load model", self.modelSummaryFrame)
        self.loadModelButton.setObjectName("loadModelButton")
        self.predictNewDataButton = QtWidgets.QPushButton("Classify new data", self.modelSummaryFrame)
        self.predictNewDataButton.setObjectName("predictNewDataButton")
        self.predictNewDataButton.setProperty("classificationPrimaryAction", True)
        self.predictNewDataButton.setProperty("gimapPrimaryAction", True)
        self.modelSummaryLayout.addWidget(self.modelTitle, 0, 0)
        self.modelSummaryLayout.addWidget(self.activePackageLabel, 0, 1, 1, 3)
        self.modelSummaryLayout.addWidget(self.loadModelButton, 1, 1)
        self.modelSummaryLayout.addWidget(self.saveActiveModelButton, 1, 2)
        self.modelSummaryLayout.addWidget(self.predictNewDataButton, 1, 3)
        self.modelSummaryLayout.setColumnStretch(1, 1)
        self.rootLayout.addWidget(self.modelSummaryFrame)

        self.predictionIntro = QtWidgets.QLabel(
            "New files use the preprocessing and projection saved with the active model.", panel
        )
        self.predictionIntro.setWordWrap(True)
        self.rootLayout.addWidget(self.predictionIntro)
        self.predictionTable = QtWidgets.QTableWidget(panel)
        self.predictionTable.setObjectName("predictionTable")
        self.predictionTable.setColumnCount(4)
        self.predictionTable.setHorizontalHeaderLabels(
            ["File", "Predicted label", "Confidence", "Status"]
        )
        self.predictionTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.predictionTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.rootLayout.addWidget(self.predictionTable, 1)
        self.exportPredictionsButton = QtWidgets.QPushButton("Export prediction CSV", panel)
        self.exportPredictionsButton.setObjectName("exportPredictionsButton")
        self.rootLayout.addWidget(self.exportPredictionsButton, 0)
        self.exportResultsButton = QtWidgets.QPushButton("Export training results", panel)
        self.exportResultsButton.setObjectName("exportResultsButton")
