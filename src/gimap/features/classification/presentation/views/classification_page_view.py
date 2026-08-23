"""Hand-maintained five-stage Classification workbench shell."""

from PyQt5 import QtCore, QtWidgets

from ..components import ClassificationWorkflowHeader


class ClassificationPageView:
    def setupUi(self, root):
        root.setObjectName("ClassificationPageRoot")
        root.setAcceptDrops(True)
        self.rootLayout = QtWidgets.QVBoxLayout(root)
        self.rootLayout.setContentsMargins(16, 14, 16, 12)
        self.rootLayout.setSpacing(10)
        self._build_header(root)
        self._build_stepper(root)
        self._build_context_bar(root)
        self.workflowStack = QtWidgets.QStackedWidget(root)
        self.workflowStack.setObjectName("classificationWorkflowStack")
        self._build_data_step()
        self._build_prepare_step()
        self._build_explore_step()
        self._build_train_step()
        self._build_apply_step()
        self.rootLayout.addWidget(self.workflowStack, 1)
        self._build_log(root)
        self.retranslateUi(root)
        self.workflowStack.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(root)

    def _build_header(self, root):
        self.classificationHeader = QtWidgets.QWidget(root)
        layout = QtWidgets.QHBoxLayout(self.classificationHeader)
        layout.setContentsMargins(0, 0, 0, 0)
        titles = QtWidgets.QVBoxLayout()
        self.titleLabel = QtWidgets.QLabel(self.classificationHeader)
        self.subtitleLabel = QtWidgets.QLabel(self.classificationHeader)
        self.subtitleLabel.setWordWrap(True)
        titles.addWidget(self.titleLabel)
        titles.addWidget(self.subtitleLabel)
        layout.addLayout(titles, 1)
        self.newSessionButton = QtWidgets.QPushButton(self.classificationHeader)
        self.loadSessionButton = QtWidgets.QPushButton(self.classificationHeader)
        self.saveSessionButton = QtWidgets.QPushButton(self.classificationHeader)
        self.helpButton = QtWidgets.QPushButton(self.classificationHeader)
        for button in (
            self.newSessionButton,
            self.loadSessionButton,
            self.saveSessionButton,
            self.helpButton,
        ):
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
            )
            layout.addWidget(button)
        self.rootLayout.addWidget(self.classificationHeader)

    def _build_context_bar(self, root):
        self.contextBar = QtWidgets.QFrame(root)
        self.contextBar.setObjectName("classificationContextBar")
        layout = QtWidgets.QHBoxLayout(self.contextBar)
        layout.setContentsMargins(10, 6, 10, 6)
        self.dataGroupTitle = QtWidgets.QLabel("Working set", self.contextBar)
        self.dataGroupCombo = QtWidgets.QComboBox(self.contextBar)
        self.dataGroupCombo.setObjectName("classificationDataGroupCombo")
        self.dataGroupCombo.setMinimumWidth(190)
        self.dataGroupSummaryLabel = QtWidgets.QLabel("No compatible data", self.contextBar)
        self.dataGroupSummaryLabel.setObjectName("dataGroupSummaryLabel")
        self.stateBadgeLabel = QtWidgets.QLabel("Waiting for data", self.contextBar)
        self.stateBadgeLabel.setObjectName("stateBadgeLabel")
        self.stateBadgeLabel.setProperty("classificationState", "idle")
        layout.addWidget(self.dataGroupTitle)
        layout.addWidget(self.dataGroupCombo)
        layout.addWidget(self.dataGroupSummaryLabel, 1)
        layout.addWidget(self.stateBadgeLabel)
        self.rootLayout.addWidget(self.contextBar)

    def _build_stepper(self, root):
        self.classificationWorkflowHeader = ClassificationWorkflowHeader(root)
        self.classificationStepper = self.classificationWorkflowHeader
        buttons = self.classificationWorkflowHeader.buttons
        self.datasetStepButton = buttons["Data"]
        self.preprocessingStepButton = buttons["Prepare"]
        self.algorithmsStepButton = buttons["Explore"]
        self.resultsStepButton = buttons["Train"]
        self.applyStepButton = buttons["Apply"]
        self.rootLayout.addWidget(self.classificationWorkflowHeader)

    def _scroll_step(self, prefix):
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName(f"{prefix}StepScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        content.setObjectName(f"{prefix}StepContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(10)
        scroll.setWidget(content)
        setattr(self, f"{prefix}StepScrollArea", scroll)
        setattr(self, f"{prefix}StepContent", content)
        self.workflowStack.addWidget(scroll)
        return content, layout

    def _section(self, parent, prefix, object_name):
        section = QtWidgets.QFrame(parent)
        section.setObjectName(object_name)
        section.setProperty("gimapSection", True)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        title = QtWidgets.QLabel(section)
        title.setProperty("gimapSectionTitle", True)
        description = QtWidgets.QLabel(section)
        description.setProperty("gimapSectionDescription", True)
        description.setWordWrap(True)
        content = QtWidgets.QWidget(section)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(content, 1)
        for suffix, value in (
            ("Section", section),
            ("Title", title),
            ("Description", description),
            ("Content", content),
            ("ContentLayout", content_layout),
        ):
            setattr(self, f"{prefix}{suffix}", value)
        return section

    def _build_data_step(self):
        content, layout = self._scroll_step("dataset")
        self.datasetInspectionSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, content)
        self.datasetInspectionSplitter.setObjectName("datasetInspectionSplitter")
        self.datasetInspectionSplitter.setChildrenCollapsible(False)
        self._section(
            self.datasetInspectionSplitter, "classificationInput", "classificationInputSection"
        )
        self._section(
            self.datasetInspectionSplitter,
            "classificationPreview",
            "classificationPreviewPanel",
        )
        self.classificationPreviewPanel = self.classificationPreviewSection
        layout.addWidget(self.datasetInspectionSplitter)

    def _build_prepare_step(self):
        content, layout = self._scroll_step("preprocessing")
        layout.addWidget(
            self._section(content, "classificationConfigure", "classificationConfigureSection")
        )
        layout.addStretch(1)

    def _build_explore_step(self):
        content, layout = self._scroll_step("algorithms")
        self.classificationExploreSection = QtWidgets.QWidget(content)
        self.classificationExploreSection.setObjectName("classificationExploreSection")
        self.classificationExploreContentLayout = QtWidgets.QVBoxLayout(
            self.classificationExploreSection
        )
        self.classificationExploreContentLayout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.classificationExploreSection, 1)

    def _build_train_step(self):
        content, layout = self._scroll_step("results")
        layout.addWidget(
            self._section(content, "classificationAlgorithm", "classificationAlgorithmSection")
        )
        layout.addWidget(
            self._section(content, "classificationResults", "classificationResultsSection")
        )
        layout.setStretch(1, 1)

    def _build_apply_step(self):
        content, layout = self._scroll_step("apply")
        layout.addWidget(self._section(content, "classificationApply", "classificationApplySection"))
        layout.addWidget(self._section(content, "classificationExport", "classificationExportSection"))
        layout.setStretch(0, 1)

    def _build_log(self, root):
        self.classificationLogSection = QtWidgets.QFrame(root)
        self.classificationLogSection.setObjectName("classificationLogSection")
        layout = QtWidgets.QVBoxLayout(self.classificationLogSection)
        layout.setContentsMargins(10, 8, 10, 8)
        self.logToggleButton = QtWidgets.QToolButton(self.classificationLogSection)
        self.logToggleButton.setCheckable(True)
        self.classificationLogDescription = QtWidgets.QLabel(self.classificationLogSection)
        self.classificationLogDescription.setWordWrap(True)
        self.classificationLogContent = QtWidgets.QWidget(self.classificationLogSection)
        self.classificationLogContentLayout = QtWidgets.QVBoxLayout(self.classificationLogContent)
        self.classificationLogContentLayout.setContentsMargins(0, 0, 0, 0)
        self.logTextBrowser = QtWidgets.QTextBrowser(self.classificationLogContent)
        self.classificationLogContentLayout.addWidget(self.logTextBrowser)
        layout.addWidget(self.logToggleButton)
        layout.addWidget(self.classificationLogDescription)
        layout.addWidget(self.classificationLogContent)
        self.rootLayout.addWidget(self.classificationLogSection)

    def retranslateUi(self, root):
        self.titleLabel.setText("Classifier")
        self.subtitleLabel.setText(
            "Turn mixed 1D curves or 2D images into an auditable map, accepted labels, and a reusable model."
        )
        self.newSessionButton.setText("New")
        self.newSessionButton.setToolTip("Start a new Classification session")
        self.loadSessionButton.setText("Open")
        self.loadSessionButton.setToolTip("Open a saved Classification session")
        self.saveSessionButton.setText("Save")
        self.saveSessionButton.setToolTip("Save this Classification session")
        self.helpButton.setText("?")
        self.helpButton.setToolTip("Show the Classification workflow guide")
        copy = (
            (self.classificationInputTitle, "Import data"),
            (self.classificationInputDescription, "Drop in an unlabeled batch or add labeled folders. Compatible 1D and 2D groups stay separate automatically."),
            (self.classificationPreviewTitle, "Sample preview"),
            (self.classificationPreviewDescription, "Check the selected sample and resolve quality issues before analysis."),
            (self.classificationConfigureTitle, "Feature recipe"),
            (self.classificationConfigureDescription, "Choose how the active 1D or 2D group becomes a comparable feature matrix."),
            (self.classificationAlgorithmTitle, "Choose classifiers"),
            (self.classificationAlgorithmDescription, "Start with the recommended set. Training uses accepted labels only."),
            (self.classificationResultsTitle, "Model results"),
            (self.classificationResultsDescription, "Compare validation scores, inspect mistakes, and choose the model for new data."),
            (self.classificationApplyTitle, "Classify new data"),
            (self.classificationApplyDescription, "Use the active model—or load one—then select unknown files to classify."),
            (self.classificationExportTitle, "Save & export"),
            (self.classificationExportDescription, "Save the active model, experiment results, or prediction table."),
        )
        for label, text in copy:
            label.setText(text)
        self.logToggleButton.setText("Operation log")
        self.classificationLogDescription.setText(
            "Import, reduction, grouping, labeling, training, prediction, and export messages."
        )
