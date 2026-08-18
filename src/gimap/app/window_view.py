"""Application shell View composition for feature-owned pages."""

from PyQt5 import QtCore

from src.gimap.app.presentation.views import MainWindowView
from src.gimap.features.fitting.presentation import (
    build_fitting_controls,
    translate_fitting_controls,
)
from src.gimap.features.prediction.presentation import (
    build_prediction_controls,
    translate_prediction_controls,
)


class ApplicationWindowView(MainWindowView):
    """Install feature-owned Fitting and Prediction controls into shell hosts."""

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self._remove_legacy_feature_host("gisaxsPredictPageHost")
        self._remove_legacy_feature_host("gisaxsFittingPageHost")
        build_prediction_controls(self)
        self._move_page(self.gisaxsPredictPage, 1)
        build_fitting_controls(self)
        self._move_page(self.gisaxsFittingPage, 2)
        self.retranslateUi(MainWindow)
        self.mainWindowWidget.setCurrentIndex(2)
        self.gisaxsPredictImageShowTabWidget.setCurrentIndex(1)
        self.fitParticleStackWidget_1.setCurrentIndex(1)
        self.fitParticleStackWidget_2.setCurrentIndex(0)
        self.fitParticleStackWidget_3.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow) -> None:
        """Translate the shell and any transitional feature-owned controls."""
        super().retranslateUi(MainWindow)
        translate = QtCore.QCoreApplication.translate
        if hasattr(self, "gisaxsPredictPredictButton"):
            translate_prediction_controls(self, translate)
        if hasattr(self, "FittingManualFittingButton"):
            translate_fitting_controls(self, translate)

    def _remove_legacy_feature_host(self, attribute_name: str) -> None:
        host = getattr(self, attribute_name)
        self.mainWindowWidget.removeWidget(host)
        host.setParent(None)
        host.deleteLater()
        delattr(self, attribute_name)

    def _move_page(self, page, index: int) -> None:
        self.mainWindowWidget.removeWidget(page)
        self.mainWindowWidget.insertWidget(index, page)


__all__ = ["ApplicationWindowView"]
