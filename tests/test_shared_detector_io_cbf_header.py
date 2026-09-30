"""Optional CBF header geometry lines (Wavelength, Detector_distance, Beam_xy)."""

from __future__ import annotations

import pytest

from src.gimap.shared.detector_io.metadata import extract_cbf_metadata

PILATUS_HEADER = """# Detector: PILATUS 2M, S/N 24-0104, EMBL
# 2024-10-26T01:38:33.207
# Pixel_size 172e-6 m x 172e-6 m
# Silicon sensor, thickness 0.000320 m
# Exposure_time 0.9950000 s
# Threshold_setting: 6000 eV
"""

GEOMETRY_LINES = (
    "# Wavelength 1.0332 A\n# Detector_distance 4.20000 m\n# Beam_xy (737.50, 1400.25) pixels\n"
)


def _metadata(extra: str = "") -> dict:
    return extract_cbf_metadata(
        {"_array_data.header_contents": PILATUS_HEADER + extra}, (1679, 1475)
    )


def test_header_without_geometry_lines() -> None:
    metadata = _metadata()
    assert metadata["header_wavelength_angstrom"] is None
    assert metadata["header_distance_m"] is None
    assert metadata["header_beam_xy_px"] is None
    assert metadata["pixel_size_x_m"] == pytest.approx(172e-6)


def test_geometry_lines_are_reported_separately() -> None:
    metadata = _metadata(GEOMETRY_LINES)
    assert metadata["header_wavelength_angstrom"] == pytest.approx(1.0332)
    assert metadata["header_distance_m"] == pytest.approx(4.2)
    assert metadata["header_beam_xy_px"] == pytest.approx((737.5, 1400.25))


def test_geometry_lines_do_not_replace_primary_fields() -> None:
    """Header values can be stale; energy look-up and calibration seeding must not use them."""
    metadata = _metadata(GEOMETRY_LINES)
    assert metadata["energy_kev"] is None
    assert metadata["wavelength_angstrom"] is None
    assert metadata["distance_m"] is None
    assert metadata["beam_center_x_px"] is None
    assert metadata["beam_center_y_px"] is None


def test_explicit_energy_still_sets_the_primary_wavelength() -> None:
    metadata = _metadata("# Energy 12.0 keV\n# Wavelength 1.5418 A\n")
    assert metadata["energy_kev"] == pytest.approx(12.0)
    assert metadata["wavelength_angstrom"] == pytest.approx(12.39842 / 12.0, rel=1e-5)
    assert metadata["header_wavelength_angstrom"] == pytest.approx(1.5418)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("# Wavelength 0.10332 nm", 1.0332),
        ("# wavelength 1.54180 A", 1.5418),
        ("# Wavelength 1.0332 Å", 1.0332),
        ("   # Wavelength 1.0332 A", 1.0332),
        ("# Wavelength 1.0332", None),
        ("# Wavelength 1.0332 m", None),
        ("# Wavelength 0 A", None),
    ],
)
def test_wavelength_needs_a_known_unit(line: str, expected: float | None) -> None:
    value = _metadata(line + "\n")["header_wavelength_angstrom"]
    assert value == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("# Detector_distance 250.0 mm", 0.25),
        ("\t# Detector_distance 0.2 m", 0.2),
        ("# Detector_distance 0.2", None),
        ("# Detector_distance 0.2 inch", None),
    ],
)
def test_distance_needs_a_known_unit(line: str, expected: float | None) -> None:
    value = _metadata(line + "\n")["header_distance_m"]
    assert value == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("# Beam_xy (737.50, 1400.25) pixels", (737.5, 1400.25)),
        ("# Beam_xy (737, 1400)", (737.0, 1400.0)),
        ("# Beam_xy (0.1268, 0.2408) m", None),
        ("# Beam_xy 737 1400", None),
    ],
)
def test_beam_xy_is_read_only_in_pixels(line: str, expected) -> None:
    value = _metadata(line + "\n")["header_beam_xy_px"]
    assert value == (pytest.approx(expected) if expected is not None else None)
