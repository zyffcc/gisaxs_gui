"""Responsive sizing for generated Prediction controls。"""

from __future__ import annotations

from PyQt5.QtWidgets import QPushButton, QSizePolicy

from src.gimap.app.presentation.layout_primitives import normalize_button, normalize_input
from src.gimap.app.presentation.responsive_layout import scale_value


def apply_prediction_control_style(ui, profile) -> None:
    """Preserve the existing Prediction control sizing and labels。"""

    ui.widget_2.setMinimumWidth(0)
    ui.widget_2.setMaximumWidth(16777215)
    ui.gisaxsPredictImageShowWidget.setMinimumWidth(0)
    ui.gisaxsPredictImageShowWidget.setMaximumWidth(16777215)

    for view in (ui.gisaxsImageGraphicsView, ui.predict2dGraphicsView):
        view.setMinimumSize(
            scale_value(360, profile, 300),
            scale_value(280, profile, 220),
        )
        view.setMaximumSize(16777215, 16777215)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    for controls in (
        ui.gisaxsImageParametersWidget,
        ui.predict2dParameterWidget,
    ):
        controls.setMinimumWidth(scale_value(340, profile, 300))
        controls.setMaximumWidth(scale_value(420, profile, 360))
        controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    ui.gisaxsPredictImageShowTabWidget.setMinimumHeight(scale_value(430, profile, 340))
    ui.gisaxsPredictImageShowTabWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    ui.gisaxsPredictImageShowWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    ui.predictStatusScrollArea.setVisible(False)
    ui.predictStatusScrollArea.setMaximumHeight(0)
    ui.predictStatusTextBrowser.setMinimumHeight(scale_value(130, profile, 110))
    ui.predictStatusTextBrowser.setMaximumHeight(scale_value(180, profile, 150))
    ui.predictStatusTextBrowser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    ui.gisaxsPredictExportFolderButton.setVisible(False)
    ui.gisaxsPredictExportFolderValue.setVisible(False)
    ui.gisaxsPredictEditButton.setText("Edit Config")
    if not hasattr(ui, "gisaxsPredictReloadConfigButton"):
        ui.gisaxsPredictReloadConfigButton = QPushButton("Reload Config")
        ui.gisaxsPredictReloadConfigButton.setObjectName("gisaxsPredictReloadConfigButton")
    ui.gisaxsPredictReloadConfigButton.setText("Reload Config")
    ui.gisaxsPredictModelImportButton.setText("Import Model")
    ui.gisaxsPredictPredictButton.setText("Predict")
    ui.gisaxsImageExportButton.setText("Export...")
    ui.predict2dExportButton.setText("Export...")
    ui.gisaxsPredictEveryValue.setPlaceholderText("1")
    ui.gisaxsPredictStackValue.setPlaceholderText("e.g. 5-15")

    for button in (
        ui.gisaxsPredictChooseGisaxsFileButton,
        ui.gisaxsPredictChooseFolderButton,
        ui.gisaxsPredictEditButton,
        ui.gisaxsPredictReloadConfigButton,
        ui.gisaxsPredictModelImportButton,
        ui.gisaxsPredictPredictButton,
        ui.gisaxsPredictImportimagesButton,
        ui.gisaxsImageExportButton,
        ui.predict2dExportButton,
    ):
        normalize_button(button, wide=button in (ui.gisaxsPredictPredictButton,))
    for button in (ui.gisaxsImageExportButton, ui.predict2dExportButton):
        button.setMinimumWidth(scale_value(120, profile, 104))
        button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

    for widget in (
        ui.gisaxsPredictChooseGisaxsFileValue,
        ui.gisaxsPredictChooseFolderValue,
        ui.gisaxsPredictStackValue,
        ui.gisaxsPredictEveryValue,
        ui.gisaxsImageShowingValue,
        ui.gisaxsImageColormapCombox,
        ui.predict2dLabelCombox,
        ui.gisaxsPredictModuleSelectCombox,
        ui.gisaxsPredictFrameworkCombox,
        ui.gisaxsImageVminValue,
        ui.gisaxsImageVmaxValue,
        ui.predict2dVminValue,
        ui.predict2dVmaxValue,
    ):
        normalize_input(widget)


__all__ = ["apply_prediction_control_style"]
