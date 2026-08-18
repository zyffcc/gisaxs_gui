"""Composition root for the feature-free presentation menu manager."""

from .presentation.menu_manager import MenuManager as PresentationMenuManager


def _create_geometry_calibration_dialog(parent):
    from src.gimap.features.calibration.presentation.dialog import (
        GeometryCalibrationDialog,
    )

    return GeometryCalibrationDialog(parent)


def _create_format_converter_dialog(parent, *, current_file=""):
    from src.gimap.features.format_converter.presentation.dialog import (
        FormatConverterDialog,
    )

    return FormatConverterDialog(parent, current_file=current_file)


class MenuManager(PresentationMenuManager):
    def __init__(self, main_window, *, settings):
        super().__init__(
            main_window,
            settings=settings,
            calibration_dialog_factory=_create_geometry_calibration_dialog,
            format_converter_dialog_factory=_create_format_converter_dialog,
        )

__all__ = ["MenuManager"]
