"""Hand-maintained Explore and Label workspace layout."""

from PyQt5 import QtCore, QtWidgets

from src.gimap.app.presentation import PlotPanel

from ..components import EmbeddingScatterView


class ClassificationExplorationPanelView:
    def setupUi(self, panel):
        panel.setObjectName("classificationExplorationPanel")
        self.rootLayout = QtWidgets.QVBoxLayout(panel)
        self.rootLayout.setContentsMargins(0, 0, 0, 0)
        self.rootLayout.setSpacing(10)

        self.introLabel = QtWidgets.QLabel(panel)
        self.introLabel.setWordWrap(True)
        self.introLabel.setObjectName("explorationIntroLabel")
        self.rootLayout.addWidget(self.introLabel)

        self.controlsFrame = QtWidgets.QFrame(panel)
        self.controlsFrame.setObjectName("explorationControlsFrame")
        self.controlsLayout = QtWidgets.QVBoxLayout(self.controlsFrame)
        self.controlsLayout.setContentsMargins(0, 0, 0, 0)
        self.controlsLayout.setSpacing(8)
        self.reductionControlsLayout = QtWidgets.QHBoxLayout()
        self.reductionControlsLayout.setSpacing(8)
        self.embeddingMethodLabel = QtWidgets.QLabel("Map method", self.controlsFrame)
        self.embeddingMethodCombo = QtWidgets.QComboBox(self.controlsFrame)
        self.embeddingMethodCombo.setObjectName("embeddingMethodCombo")
        self.embeddingMethodCombo.addItems(["PCA 2D", "UMAP 2D", "t-SNE 2D"])
        self.runEmbeddingButton = QtWidgets.QPushButton("Build 2D map", self.controlsFrame)
        self.runEmbeddingButton.setObjectName("runEmbeddingButton")
        self.runEmbeddingButton.setProperty("classificationPrimaryAction", True)
        self.runEmbeddingButton.setProperty("gimapPrimaryAction", True)
        self.runEmbeddingButton.setToolTip("Build features and compute the selected 2D reduction")
        self.embeddingColorLabel = QtWidgets.QLabel("Color by", self.controlsFrame)
        self.embeddingColorCombo = QtWidgets.QComboBox(self.controlsFrame)
        self.embeddingColorCombo.setObjectName("embeddingColorCombo")
        self.embeddingColorCombo.addItems(
            ["Accepted label", "Suggested group", "Source", "QC status", "Prediction"]
        )
        self.fitEmbeddingButton = QtWidgets.QPushButton("Fit map", self.controlsFrame)
        self.fitEmbeddingButton.setObjectName("fitEmbeddingButton")
        for widget in (
            self.embeddingMethodLabel,
            self.embeddingMethodCombo,
            self.runEmbeddingButton,
            self.embeddingColorLabel,
            self.embeddingColorCombo,
            self.fitEmbeddingButton,
        ):
            self.reductionControlsLayout.addWidget(widget)
        self.reductionControlsLayout.addStretch(1)
        self.controlsLayout.addLayout(self.reductionControlsLayout)

        self.groupingControlsLayout = QtWidgets.QHBoxLayout()
        self.groupingControlsLayout.setSpacing(8)
        self.clusterMethodLabel = QtWidgets.QLabel("Suggest with", self.controlsFrame)
        self.clusterMethodCombo = QtWidgets.QComboBox(self.controlsFrame)
        self.clusterMethodCombo.setObjectName("clusterMethodCombo")
        self.clusterMethodCombo.addItems(["HDBSCAN", "K-Means"])
        self.clusterCountLabel = QtWidgets.QLabel("Groups", self.controlsFrame)
        self.clusterCountSpinBox = QtWidgets.QSpinBox(self.controlsFrame)
        self.clusterCountSpinBox.setObjectName("clusterCountSpinBox")
        self.clusterCountSpinBox.setRange(2, 50)
        self.clusterCountSpinBox.setValue(4)
        self.minClusterSizeLabel = QtWidgets.QLabel("Minimum group", self.controlsFrame)
        self.minClusterSizeSpinBox = QtWidgets.QSpinBox(self.controlsFrame)
        self.minClusterSizeSpinBox.setObjectName("minClusterSizeSpinBox")
        self.minClusterSizeSpinBox.setRange(2, 10000)
        self.minClusterSizeSpinBox.setValue(5)
        self.suggestGroupsButton = QtWidgets.QPushButton("Suggest groups", self.controlsFrame)
        self.suggestGroupsButton.setObjectName("suggestGroupsButton")
        self.suggestGroupsButton.setToolTip(
            "Propose review groups without changing accepted labels"
        )
        for widget in (
            self.clusterMethodLabel,
            self.clusterMethodCombo,
            self.clusterCountLabel,
            self.clusterCountSpinBox,
            self.minClusterSizeLabel,
            self.minClusterSizeSpinBox,
            self.suggestGroupsButton,
        ):
            self.groupingControlsLayout.addWidget(widget)
        self.groupingControlsLayout.addStretch(1)
        self.controlsLayout.addLayout(self.groupingControlsLayout)
        self.rootLayout.addWidget(self.controlsFrame)

        self.explorationSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, panel)
        self.explorationSplitter.setObjectName("explorationSplitter")
        self.explorationSplitter.setChildrenCollapsible(False)
        self.embeddingPlotPanel = PlotPanel(
            "Embedding",
            "Click or drag a box to select samples. Ctrl+wheel zooms; Esc clears selection.",
            self.explorationSplitter,
            empty_title="No reduction yet",
            empty_message="Run PCA or UMAP to inspect structure before training.",
        )
        self.embeddingScatterView = EmbeddingScatterView(self.embeddingPlotPanel)
        self.embeddingGraphicsView = self.embeddingScatterView
        self.embeddingPlotPanel.set_plot_widget(self.embeddingScatterView)

        self.selectionPanel = QtWidgets.QFrame(self.explorationSplitter)
        self.selectionPanel.setProperty("gimapSection", True)
        self.selectionPanel.setObjectName("explorationSelectionPanel")
        self.selectionPanel.setMinimumWidth(300)
        self.selectionPanel.setMaximumWidth(460)
        self.explorationSelectionPanel = self.selectionPanel
        self.selectionLayout = QtWidgets.QVBoxLayout(self.selectionPanel)
        self.selectionLayout.setContentsMargins(12, 10, 12, 12)
        self.selectionLayout.setSpacing(8)
        self.selectionTitle = QtWidgets.QLabel("Review selection", self.selectionPanel)
        self.selectionTitle.setProperty("gimapSectionTitle", True)
        self.selectionLayout.addWidget(self.selectionTitle)
        self.selectedCountLabel = QtWidgets.QLabel("0 samples selected", self.selectionPanel)
        self.selectedCountLabel.setObjectName("selectedCountLabel")
        self.selectionLayout.addWidget(self.selectedCountLabel)
        self.selectionButtonsLayout = QtWidgets.QHBoxLayout()
        self.selectAllEmbeddingButton = QtWidgets.QPushButton("Select all", self.selectionPanel)
        self.clearEmbeddingSelectionButton = QtWidgets.QPushButton("Clear selection", self.selectionPanel)
        self.selectionButtonsLayout.addWidget(self.selectAllEmbeddingButton)
        self.selectionButtonsLayout.addWidget(self.clearEmbeddingSelectionButton)
        self.selectionLayout.addLayout(self.selectionButtonsLayout)
        self.labelEdit = QtWidgets.QLineEdit(self.selectionPanel)
        self.labelEdit.setObjectName("explorationLabelEdit")
        self.labelEdit.setPlaceholderText("Accepted class name")
        self.selectionLayout.addWidget(self.labelEdit)
        self.labelActionsLayout = QtWidgets.QGridLayout()
        self.assignLabelButton = QtWidgets.QPushButton("Apply class label", self.selectionPanel)
        self.assignLabelButton.setObjectName("assignLabelButton")
        self.clearLabelButton = QtWidgets.QPushButton("Keep unlabeled", self.selectionPanel)
        self.clearLabelButton.setObjectName("clearLabelButton")
        self.acceptSuggestionsButton = QtWidgets.QPushButton("Accept selected suggestions", self.selectionPanel)
        self.acceptSuggestionsButton.setObjectName("acceptSuggestionsButton")
        self.acceptAllSuggestionsButton = QtWidgets.QPushButton("Accept all suggestions", self.selectionPanel)
        self.acceptAllSuggestionsButton.setObjectName("acceptAllSuggestionsButton")
        self.labelActionsLayout.addWidget(self.assignLabelButton, 0, 0)
        self.labelActionsLayout.addWidget(self.clearLabelButton, 1, 0)
        self.labelActionsLayout.addWidget(self.acceptSuggestionsButton, 2, 0)
        self.labelActionsLayout.addWidget(self.acceptAllSuggestionsButton, 3, 0)
        self.selectionLayout.addLayout(self.labelActionsLayout)
        self.explorationSampleLabel = QtWidgets.QLabel(
            "Double-click a point to preview a sample", self.selectionPanel
        )
        self.explorationSampleLabel.setObjectName("explorationSampleLabel")
        self.explorationSampleLabel.setWordWrap(True)
        self.selectionLayout.addWidget(self.explorationSampleLabel)
        self.explorationPreviewView = QtWidgets.QGraphicsView(self.selectionPanel)
        self.explorationPreviewView.setObjectName("explorationPreviewView")
        self.explorationPreviewView.setMinimumHeight(180)
        self.selectionLayout.addWidget(self.explorationPreviewView)
        self.selectionTable = QtWidgets.QTableWidget(self.selectionPanel)
        self.selectionTable.setObjectName("explorationSelectionTable")
        self.selectionTable.setColumnCount(4)
        self.selectionTable.setHorizontalHeaderLabels(
            ["File", "Accepted label", "Suggestion", "QC"]
        )
        self.selectionTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.selectionTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.selectionLayout.addWidget(self.selectionTable, 1)
        self.explorationSplitter.setStretchFactor(0, 3)
        self.explorationSplitter.setStretchFactor(1, 1)
        self.rootLayout.addWidget(self.explorationSplitter, 1)

        self.explorationStatusLabel = QtWidgets.QLabel("Import data to begin.", panel)
        self.explorationStatusLabel.setObjectName("explorationStatusLabel")
        self.explorationStatusLabel.setWordWrap(True)
        self.rootLayout.addWidget(self.explorationStatusLabel)
        self.retranslateUi(panel)

    def retranslateUi(self, panel):
        self.introLabel.setText(
            "Build a visual map first. Group suggestions are optional and never become training labels until you accept them."
        )
