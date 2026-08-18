"""Build the Fitting fit controls section."""

from PyQt5 import QtCore, QtWidgets

from src.gimap.features.fitting.presentation.range_slider import QRangeSlider


class FitControlsMixin:
    """Own the Fitting fit controls widgets."""

    def _setup_fit_controls(self):
        self.curvePlotControlWidget = QtWidgets.QWidget(self.fitBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.curvePlotControlWidget.sizePolicy().hasHeightForWidth())
        self.curvePlotControlWidget.setSizePolicy(sizePolicy)
        self.curvePlotControlWidget.setMinimumSize(QtCore.QSize(420, 0))
        self.curvePlotControlWidget.setMaximumSize(QtCore.QSize(420, 16777215))
        self.curvePlotControlWidget.setStyleSheet(
            "#curvePlotControlWidget {\n"
            "  border: 1px solid rgba(0,0,0,0.12);  /* 细外框 */\n"
            "  padding: 8px;                        /* 让内容不贴边 */\n"
            "  /* 伪“阴影”：用浅色外描边模拟（Qt 没有 box-shadow） */\n"
            "  outline: 6px solid rgba(0,0,0,0.03);\n"
            "  outline-offset: -6px;                /* 紧贴边缘 */\n"
            "}"
        )
        self.curvePlotControlWidget.setObjectName("curvePlotControlWidget")
        self.gridLayout_33 = QtWidgets.QGridLayout(self.curvePlotControlWidget)
        self.gridLayout_33.setObjectName("gridLayout_33")
        self.fitDataPointsNumWidget = QtWidgets.QWidget(self.curvePlotControlWidget)
        self.fitDataPointsNumWidget.setMaximumSize(QtCore.QSize(100000, 16777215))
        self.fitDataPointsNumWidget.setObjectName("fitDataPointsNumWidget")
        self.gridLayout_42 = QtWidgets.QGridLayout(self.fitDataPointsNumWidget)
        self.gridLayout_42.setObjectName("gridLayout_42")
        self.fitInterpolationMethodLabel = QtWidgets.QLabel(self.fitDataPointsNumWidget)
        self.fitInterpolationMethodLabel.setObjectName("fitInterpolationMethodLabel")
        self.gridLayout_42.addWidget(self.fitInterpolationMethodLabel, 2, 0, 1, 1)
        self.fitDataPointsNumLabel = QtWidgets.QLabel(self.fitDataPointsNumWidget)
        self.fitDataPointsNumLabel.setObjectName("fitDataPointsNumLabel")
        self.gridLayout_42.addWidget(self.fitDataPointsNumLabel, 0, 0, 1, 1)
        self.fitDataPointsNumValue = QtWidgets.QSpinBox(self.fitDataPointsNumWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitDataPointsNumValue.sizePolicy().hasHeightForWidth())
        self.fitDataPointsNumValue.setSizePolicy(sizePolicy)
        self.fitDataPointsNumValue.setMaximumSize(QtCore.QSize(80, 20))
        self.fitDataPointsNumValue.setObjectName("fitDataPointsNumValue")
        self.gridLayout_42.addWidget(self.fitDataPointsNumValue, 1, 0, 1, 1)
        self.fitInterpolationMethodValue = QtWidgets.QComboBox(self.fitDataPointsNumWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fitInterpolationMethodValue.sizePolicy().hasHeightForWidth()
        )
        self.fitInterpolationMethodValue.setSizePolicy(sizePolicy)
        self.fitInterpolationMethodValue.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fitInterpolationMethodValue.setObjectName("fitInterpolationMethodValue")
        self.gridLayout_42.addWidget(self.fitInterpolationMethodValue, 3, 0, 1, 1)
        self.gridLayout_33.addWidget(self.fitDataPointsNumWidget, 3, 0, 1, 1)
        self.fitFittingRegionwidget = QtWidgets.QWidget(self.curvePlotControlWidget)
        self.fitFittingRegionwidget.setMaximumSize(QtCore.QSize(16777215, 80))
        self.fitFittingRegionwidget.setStyleSheet(
            "#fitFittingRegionwidget {\n"
            "  border: 1px solid rgba(0,0,0,0.12);  /* 细外框 */\n"
            "  border-radius: 12px;                 /* 圆角大小可改 */\n"
            "  padding: 8px;                        /* 让内容不贴边 */\n"
            "  /* 伪“阴影”：用浅色外描边模拟（Qt 没有 box-shadow） */\n"
            "  outline: 6px solid rgba(0,0,0,0.03);\n"
            "  outline-offset: -6px;                /* 紧贴边缘 */\n"
            "}"
        )
        self.fitFittingRegionwidget.setObjectName("fitFittingRegionwidget")
        self.gridLayout_41 = QtWidgets.QGridLayout(self.fitFittingRegionwidget)
        self.gridLayout_41.setObjectName("gridLayout_41")
        self.fitFittingRegionSlider = QRangeSlider(self.fitFittingRegionwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitFittingRegionSlider.sizePolicy().hasHeightForWidth())
        self.fitFittingRegionSlider.setSizePolicy(sizePolicy)
        self.fitFittingRegionSlider.setMinimumSize(QtCore.QSize(0, 20))
        self.fitFittingRegionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.fitFittingRegionSlider.setObjectName("fitFittingRegionSlider")
        self.gridLayout_41.addWidget(self.fitFittingRegionSlider, 1, 1, 1, 1)
        self.fitFittingRegionValueWidget = QtWidgets.QWidget(self.fitFittingRegionwidget)
        self.fitFittingRegionValueWidget.setMinimumSize(QtCore.QSize(0, 35))
        self.fitFittingRegionValueWidget.setObjectName("fitFittingRegionValueWidget")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.fitFittingRegionValueWidget)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.fitFittingRegionMinValue = QtWidgets.QDoubleSpinBox(self.fitFittingRegionValueWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitFittingRegionMinValue.sizePolicy().hasHeightForWidth())
        self.fitFittingRegionMinValue.setSizePolicy(sizePolicy)
        self.fitFittingRegionMinValue.setMinimumSize(QtCore.QSize(0, 20))
        self.fitFittingRegionMinValue.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fitFittingRegionMinValue.setObjectName("fitFittingRegionMinValue")
        self.horizontalLayout_14.addWidget(self.fitFittingRegionMinValue)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_14.addItem(spacerItem2)
        self.fitFittingRegionMaxValue = QtWidgets.QDoubleSpinBox(self.fitFittingRegionValueWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitFittingRegionMaxValue.sizePolicy().hasHeightForWidth())
        self.fitFittingRegionMaxValue.setSizePolicy(sizePolicy)
        self.fitFittingRegionMaxValue.setMinimumSize(QtCore.QSize(0, 20))
        self.fitFittingRegionMaxValue.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fitFittingRegionMaxValue.setObjectName("fitFittingRegionMaxValue")
        self.horizontalLayout_14.addWidget(self.fitFittingRegionMaxValue)
        self.gridLayout_41.addWidget(self.fitFittingRegionValueWidget, 2, 1, 1, 1)
        self.fitFittingRegionLabel = QtWidgets.QLabel(self.fitFittingRegionwidget)
        self.fitFittingRegionLabel.setMinimumSize(QtCore.QSize(0, 20))
        self.fitFittingRegionLabel.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fitFittingRegionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fitFittingRegionLabel.setObjectName("fitFittingRegionLabel")
        self.gridLayout_41.addWidget(self.fitFittingRegionLabel, 1, 0, 1, 1)
        self.gridLayout_33.addWidget(self.fitFittingRegionwidget, 1, 0, 1, 3)
        self.fitFittingShowWidget = QtWidgets.QWidget(self.curvePlotControlWidget)
        self.fitFittingShowWidget.setMinimumSize(QtCore.QSize(250, 0))
        self.fitFittingShowWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.fitFittingShowWidget.setObjectName("fitFittingShowWidget")
        self.gridLayout_40 = QtWidgets.QGridLayout(self.fitFittingShowWidget)
        self.gridLayout_40.setObjectName("gridLayout_40")
        self.fitBGShowCheckBox = QtWidgets.QCheckBox(self.fitFittingShowWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitBGShowCheckBox.sizePolicy().hasHeightForWidth())
        self.fitBGShowCheckBox.setSizePolicy(sizePolicy)
        self.fitBGShowCheckBox.setObjectName("fitBGShowCheckBox")
        self.gridLayout_40.addWidget(self.fitBGShowCheckBox, 2, 0, 1, 1)
        self.fitDisplayOptionsLabel = QtWidgets.QLabel(self.fitFittingShowWidget)
        self.fitDisplayOptionsLabel.setObjectName("fitDisplayOptionsLabel")
        self.gridLayout_40.addWidget(self.fitDisplayOptionsLabel, 0, 0, 1, 3)
        self.fitResShowCheckBox = QtWidgets.QCheckBox(self.fitFittingShowWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitResShowCheckBox.sizePolicy().hasHeightForWidth())
        self.fitResShowCheckBox.setSizePolicy(sizePolicy)
        self.fitResShowCheckBox.setObjectName("fitResShowCheckBox")
        self.gridLayout_40.addWidget(self.fitResShowCheckBox, 3, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_40.addItem(spacerItem3, 4, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_40.addItem(spacerItem4, 1, 0, 1, 1)
        self.ParticlesNumWidget = QtWidgets.QWidget(self.fitFittingShowWidget)
        self.ParticlesNumWidget.setObjectName("ParticlesNumWidget")
        self.verticalLayout_26 = QtWidgets.QVBoxLayout(self.ParticlesNumWidget)
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.fitParticle1ShowCheckBox = QtWidgets.QCheckBox(self.ParticlesNumWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitParticle1ShowCheckBox.sizePolicy().hasHeightForWidth())
        self.fitParticle1ShowCheckBox.setSizePolicy(sizePolicy)
        self.fitParticle1ShowCheckBox.setObjectName("fitParticle1ShowCheckBox")
        self.verticalLayout_26.addWidget(self.fitParticle1ShowCheckBox)
        self.fitParticle2ShowCheckBox = QtWidgets.QCheckBox(self.ParticlesNumWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitParticle2ShowCheckBox.sizePolicy().hasHeightForWidth())
        self.fitParticle2ShowCheckBox.setSizePolicy(sizePolicy)
        self.fitParticle2ShowCheckBox.setObjectName("fitParticle2ShowCheckBox")
        self.verticalLayout_26.addWidget(self.fitParticle2ShowCheckBox)
        self.fitParticle3ShowCheckBox = QtWidgets.QCheckBox(self.ParticlesNumWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitParticle3ShowCheckBox.sizePolicy().hasHeightForWidth())
        self.fitParticle3ShowCheckBox.setSizePolicy(sizePolicy)
        self.fitParticle3ShowCheckBox.setObjectName("fitParticle3ShowCheckBox")
        self.verticalLayout_26.addWidget(self.fitParticle3ShowCheckBox)
        self.gridLayout_40.addWidget(self.ParticlesNumWidget, 1, 1, 4, 2)
        self.gridLayout_33.addWidget(self.fitFittingShowWidget, 3, 1, 1, 2)
        self.fitGraphicsView = QtWidgets.QGraphicsView(self.curvePlotControlWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fitGraphicsView.sizePolicy().hasHeightForWidth())
        self.fitGraphicsView.setSizePolicy(sizePolicy)
        self.fitGraphicsView.setMinimumSize(QtCore.QSize(400, 300))
        self.fitGraphicsView.setMaximumSize(QtCore.QSize(400, 16777215))
        self.fitGraphicsView.setStyleSheet(
            "QGraphicsView {\n"
            "    background: transparent;\n"
            "    border: 1px solid #888888; \n"
            "    border-radius: 4px;  \n"
            "}"
        )
        self.fitGraphicsView.setObjectName("fitGraphicsView")
        self.gridLayout_33.addWidget(self.fitGraphicsView, 0, 0, 1, 1)
        self.gridLayout_24.addWidget(self.curvePlotControlWidget, 0, 3, 6, 1)
        self.fitMethodWidget = QtWidgets.QWidget(self.fitBox)
        self.fitMethodWidget.setMinimumSize(QtCore.QSize(0, 100))
        self.fitMethodWidget.setObjectName("fitMethodWidget")
        self.gridLayout_37 = QtWidgets.QGridLayout(self.fitMethodWidget)
        self.gridLayout_37.setObjectName("gridLayout_37")
        self.fitMethodLabel = QtWidgets.QLabel(self.fitMethodWidget)
        self.fitMethodLabel.setMinimumSize(QtCore.QSize(0, 20))
        self.fitMethodLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fitMethodLabel.setObjectName("fitMethodLabel")
        self.gridLayout_37.addWidget(self.fitMethodLabel, 0, 0, 1, 1)
        self.FittingAutoFittingButton = QtWidgets.QPushButton(self.fitMethodWidget)
        self.FittingAutoFittingButton.setMinimumSize(QtCore.QSize(50, 20))
        self.FittingAutoFittingButton.setObjectName("FittingAutoFittingButton")
        self.gridLayout_37.addWidget(self.FittingAutoFittingButton, 2, 3, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_37.addItem(spacerItem5, 2, 2, 1, 1)
        self.fitMethodValue = QtWidgets.QComboBox(self.fitMethodWidget)
        self.fitMethodValue.setObjectName("fitMethodValue")
        self.fitMethodValue.addItem("")
        self.fitMethodValue.addItem("")
        self.fitMethodValue.addItem("")
        self.fitMethodValue.addItem("")
        self.fitMethodValue.addItem("")
        self.gridLayout_37.addWidget(self.fitMethodValue, 0, 1, 1, 3)
        spacerItem6 = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_37.addItem(spacerItem6, 1, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_37.addItem(spacerItem7, 2, 1, 1, 1)
        self.FittingClearFittingButton_2 = QtWidgets.QPushButton(self.fitMethodWidget)
        self.FittingClearFittingButton_2.setMinimumSize(QtCore.QSize(50, 20))
        self.FittingClearFittingButton_2.setObjectName("FittingClearFittingButton_2")
        self.gridLayout_37.addWidget(self.FittingClearFittingButton_2, 2, 0, 1, 1)
        self.FittingAutoKButton = QtWidgets.QPushButton(self.fitMethodWidget)
        self.FittingAutoKButton.setMinimumSize(QtCore.QSize(50, 20))
        self.FittingAutoKButton.setObjectName("FittingAutoKButton")
        self.gridLayout_37.addWidget(self.FittingAutoKButton, 1, 0, 1, 1)
        self.gridLayout_24.addWidget(self.fitMethodWidget, 4, 1, 2, 2)
        self.widget_8 = QtWidgets.QWidget(self.fitBox)
        self.widget_8.setMinimumSize(QtCore.QSize(0, 40))
        self.widget_8.setObjectName("widget_8")
        self.gridLayout_32 = QtWidgets.QGridLayout(self.widget_8)
        self.gridLayout_32.setObjectName("gridLayout_32")
        self.fitSigmaResLabel = QtWidgets.QLabel(self.widget_8)
        self.fitSigmaResLabel.setObjectName("fitSigmaResLabel")
        self.gridLayout_32.addWidget(self.fitSigmaResLabel, 0, 0, 1, 1)
        self.fitSigmaResValue = QtWidgets.QDoubleSpinBox(self.widget_8)
        self.fitSigmaResValue.setMinimumSize(QtCore.QSize(0, 20))
        self.fitSigmaResValue.setObjectName("fitSigmaResValue")
        self.gridLayout_32.addWidget(self.fitSigmaResValue, 0, 1, 1, 1)
        self.fitNuResLabel = QtWidgets.QLabel(self.widget_8)
        self.fitNuResLabel.setObjectName("fitNuResLabel")
        self.gridLayout_32.addWidget(self.fitNuResLabel, 0, 2, 1, 1)
        self.fitNuResValue = QtWidgets.QDoubleSpinBox(self.widget_8)
        self.fitNuResValue.setMinimumSize(QtCore.QSize(0, 20))
        self.fitNuResValue.setObjectName("fitNuResValue")
        self.gridLayout_32.addWidget(self.fitNuResValue, 0, 3, 1, 1)
        self.gridLayout_24.addWidget(self.widget_8, 3, 0, 1, 3)
