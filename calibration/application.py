"""Calibration application 旧入口的兼容门面。"""

from __future__ import annotations

from src.gimap.app.bootstrap import create_standalone_legacy_context
from src.gimap.features.calibration.infrastructure.adapters import SettingsGeometryAdapter

from .models import CalibrationResult


def apply_calibration_result(
    result: CalibrationResult,
    main_window=None,
) -> dict[str, float]:
    """保留旧 API；参数写入由新 infrastructure adapter 负责。"""
    context = create_standalone_legacy_context()
    geometry = SettingsGeometryAdapter(context.settings).apply(result)
    candidate = result.selected_candidate
    if main_window is not None:
        page = getattr(getattr(main_window, "components", None), "waxs_page", None)
        if page is not None:
            controls = {
                "center_x_spin": candidate.center_x_px,
                "center_y_spin": candidate.center_y_px,
                "distance_spin": candidate.distance_mm,
                "pixel_x_spin": result.pixel_size_x_m * 1e6,
                "pixel_y_spin": result.pixel_size_y_m * 1e6,
                "wavelength_spin": result.wavelength_angstrom,
            }
            for name, value in controls.items():
                widget = getattr(page, name, None)
                if widget is not None:
                    widget.setValue(float(value))
            if hasattr(page, "refresh_view"):
                page.refresh_view()
        if hasattr(main_window, "statusbar"):
            main_window.statusbar.showMessage(
                "Geometry calibration applied: center "
                f"({candidate.center_x_px:.2f}, {candidate.center_y_px:.2f}), "
                f"distance {candidate.distance_mm:.2f} mm"
            )
    return geometry
