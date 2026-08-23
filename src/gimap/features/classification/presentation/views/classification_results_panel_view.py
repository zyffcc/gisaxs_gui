"""Hand-maintained training review layout."""

from PyQt5 import QtCore, QtWidgets


class ClassificationResultsPanelView:
    def setupUi(self, panel):
        panel.setObjectName("classificationResultsPanel")
        self.resultsPanelLayout = QtWidgets.QVBoxLayout(panel)
        self.resultsPanelLayout.setContentsMargins(0, 0, 0, 0)
        self.resultsPanelLayout.setSpacing(8)

        self.resultsActionsLayout = QtWidgets.QHBoxLayout()
        self.resultsTitle = QtWidgets.QLabel("Training review", panel)
        self.resultsTitle.setObjectName("resultsTitle")
        self.activeModelTitle = QtWidgets.QLabel("Active model", panel)
        self.activeModelCombo = QtWidgets.QComboBox(panel)
        self.activeModelCombo.setObjectName("activeModelCombo")
        self.setActiveModelButton = QtWidgets.QPushButton("Use this model", panel)
        self.setActiveModelButton.setObjectName("setActiveModelButton")
        self.resultsActionsLayout.addWidget(self.resultsTitle)
        self.resultsActionsLayout.addStretch(1)
        self.resultsActionsLayout.addWidget(self.activeModelTitle)
        self.resultsActionsLayout.addWidget(self.activeModelCombo)
        self.resultsActionsLayout.addWidget(self.setActiveModelButton)
        self.resultsPanelLayout.addLayout(self.resultsActionsLayout)

        self.overviewFrame = QtWidgets.QFrame(panel)
        self.overviewFrame.setObjectName("resultsOverviewFrame")
        self.resultsOverviewLayout = QtWidgets.QGridLayout(self.overviewFrame)
        self.resultsOverviewLayout.setContentsMargins(8, 6, 8, 6)
        titles = (
            ("Best model", "bestModelLabel"),
            ("Macro F1", "bestMacroF1Label"),
            ("Balanced accuracy", "bestBalancedAccuracyLabel"),
            ("Accuracy", "bestAccuracyLabel"),
            ("Samples", "resultSamplesLabel"),
            ("Classes", "resultClassesLabel"),
            ("Validation", "resultValidationLabel"),
        )
        for index, (title, attr) in enumerate(titles):
            row = index // 4
            col = (index % 4) * 2
            self.resultsOverviewLayout.addWidget(
                QtWidgets.QLabel(title, self.overviewFrame), row, col
            )
            value = QtWidgets.QLabel(
                "0" if title in {"Samples", "Classes"} else "-", self.overviewFrame
            )
            value.setObjectName(attr)
            setattr(self, attr, value)
            self.resultsOverviewLayout.addWidget(value, row, col + 1)
        self.resultsOutdatedLabel = QtWidgets.QLabel("", self.overviewFrame)
        self.resultsOutdatedLabel.setObjectName("resultsOutdatedLabel")
        self.resultsOverviewLayout.addWidget(self.resultsOutdatedLabel, 2, 0, 1, 8)
        self.resultsPanelLayout.addWidget(self.overviewFrame)

        self.resultTabs = QtWidgets.QTabWidget(panel)
        self.resultTabs.setDocumentMode(True)
        self.resultTabs.setObjectName("resultTabs")
        self._build_overview_tab()
        self._build_confusion_tab()
        self._build_metrics_tab()
        self._build_misclassified_tab()
        self.resultsPanelLayout.addWidget(self.resultTabs, 1)
        self.resultTabs.setCurrentIndex(0)

    def _build_overview_tab(self):
        self.overviewTab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.overviewTab)
        layout.setContentsMargins(4, 6, 4, 4)
        self.overviewSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.overviewTab)
        self.overviewSplitter.setChildrenCollapsible(False)
        self.overviewSplitter.setObjectName("overviewSplitter")
        leaderboard = QtWidgets.QWidget(self.overviewSplitter)
        leaderboard_layout = QtWidgets.QVBoxLayout(leaderboard)
        leaderboard_layout.setContentsMargins(0, 0, 0, 0)
        leaderboard_layout.addWidget(QtWidgets.QLabel("Model comparison", leaderboard))
        self.resultsTable = QtWidgets.QTableWidget(leaderboard)
        self.resultsTable.setObjectName("resultsTable")
        self.resultsTable.setColumnCount(9)
        self.resultsTable.setHorizontalHeaderLabels(
            [
                "Rank",
                "Algorithm",
                "Accuracy",
                "Balanced accuracy",
                "Macro F1",
                "Weighted F1",
                "Training time",
                "Prediction time",
                "Status",
            ]
        )
        self.resultsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        leaderboard_layout.addWidget(self.resultsTable)
        chart = QtWidgets.QWidget(self.overviewSplitter)
        chart_layout = QtWidgets.QVBoxLayout(chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(QtWidgets.QLabel("Metric comparison", chart))
        self.metricChartLabel = QtWidgets.QLabel("No metrics yet", chart)
        self.metricChartLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.metricChartLabel.setObjectName("metricChartLabel")
        chart_layout.addWidget(self.metricChartLabel)
        layout.addWidget(self.overviewSplitter)
        self.resultTabs.addTab(self.overviewTab, "Overview")

    def _build_confusion_tab(self):
        self.confusionTab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.confusionTab)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Confusion matrix", self.confusionTab))
        header.addStretch(1)
        self.confusionNormalizeCombo = QtWidgets.QComboBox(self.confusionTab)
        self.confusionNormalizeCombo.setObjectName("confusionNormalizeCombo")
        self.confusionNormalizeCombo.addItems(
            ["Raw counts", "Normalize by true class", "Normalize by predicted class"]
        )
        header.addWidget(self.confusionNormalizeCombo)
        layout.addLayout(header)
        self.confusionMatrixTable = QtWidgets.QTableWidget(self.confusionTab)
        self.confusionMatrixTable.setObjectName("confusionMatrixTable")
        layout.addWidget(self.confusionMatrixTable)
        self.resultTabs.addTab(self.confusionTab, "Confusion matrix")

    def _build_metrics_tab(self):
        self.metricsTab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.metricsTab)
        self.perClassTable = QtWidgets.QTableWidget(self.metricsTab)
        self.perClassTable.setObjectName("perClassTable")
        self.perClassTable.setColumnCount(5)
        self.perClassTable.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1-score", "Support"]
        )
        layout.addWidget(self.perClassTable)
        self.resultTabs.addTab(self.metricsTab, "Per-class metrics")

    def _build_misclassified_tab(self):
        self.misclassifiedTab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.misclassifiedTab)
        self.misclassifiedTable = QtWidgets.QTableWidget(self.misclassifiedTab)
        self.misclassifiedTable.setObjectName("misclassifiedTable")
        self.misclassifiedTable.setColumnCount(6)
        self.misclassifiedTable.setHorizontalHeaderLabels(
            ["File", "True label", "Predicted label", "Confidence", "Shape", "Preview"]
        )
        layout.addWidget(self.misclassifiedTable)
        self.resultTabs.addTab(self.misclassifiedTab, "Misclassified")
