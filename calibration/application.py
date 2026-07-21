from __future__ import annotations

from core.global_params import global_params

from .models import CalibrationResult


def apply_calibration_result(result: CalibrationResult, main_window=None) -> dict[str, float]:
    candidate = result.selected_candidate
    geometry = {
        "distance": float(candidate.distance_mm),
        "pixel_size_x": float(result.pixel_size_x_m * 1e6),
        "pixel_size_y": float(result.pixel_size_y_m * 1e6),
        "beam_center_x": float(candidate.center_x_px),
        "beam_center_y": float(candidate.center_y_px),
    }
    for key, value in geometry.items():
        global_params.set_parameter("detector", key, value)
        global_params.set_parameter("fitting", f"detector.{key}", value)
    global_params.set_parameter("detector", "rotation_deg", float(candidate.detector_rotation_deg))
    global_params.set_parameter("beam", "wavelength", float(result.wavelength_angstrom / 10.0))
    global_params.set_parameter("beam", "energy_kev", float(result.energy_kev))
    global_params.set_parameter("system", "geometry_calibration", {
        "source_image": result.source_image,
        "timestamp": result.calibration_timestamp,
        "standard": candidate.standard_key,
        "confidence": candidate.confidence,
        "residual_px": candidate.rms_residual_px,
    })

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
                f"Geometry calibration applied: center ({candidate.center_x_px:.2f}, {candidate.center_y_px:.2f}), "
                f"distance {candidate.distance_mm:.2f} mm"
            )
    return geometry
