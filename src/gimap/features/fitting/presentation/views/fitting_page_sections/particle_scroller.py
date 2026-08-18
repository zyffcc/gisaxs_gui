"""Build the Fitting particle scroller section."""

from PyQt5 import QtCore, QtWidgets


class ParticleScrollerMixin:
    """Own the Fitting particle scroller widgets."""

    def _finish_particle_scroller(self):
        self.addModelButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(30)
        sizePolicy.setVerticalStretch(30)
        sizePolicy.setHeightForWidth(self.addModelButton.sizePolicy().hasHeightForWidth())
        self.addModelButton.setSizePolicy(sizePolicy)
        self.addModelButton.setMinimumSize(QtCore.QSize(30, 30))
        self.addModelButton.setMaximumSize(QtCore.QSize(30, 30))
        self.addModelButton.setObjectName("addModelButton")
        self.horizontalLayout_18.addWidget(self.addModelButton)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_18.addItem(spacerItem1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_17.addWidget(self.scrollArea)
        self.gridLayout_24.addWidget(self.widget_7, 2, 0, 1, 3)
