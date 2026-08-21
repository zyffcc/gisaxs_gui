from src.gimap.features.fitting.application import (
    LoadDetectorSettings,
    SaveDetectorSettings,
)
from src.gimap.features.fitting.domain import (
    DetectorSettings,
    energy_to_wavelength,
    wavelength_to_energy,
)
from src.gimap.integrations.state import InMemorySettingsRepository


def test_detector_settings_round_trip_preserves_legacy_keys() -> None:
    repository = InMemorySettingsRepository()
    expected = DetectorSettings(
        distance=1840.5,
        grazing_angle=0.32,
        wavelength=0.0103,
        beam_center_x=512.25,
        beam_center_y=498.75,
        pixel_size_x=75.0,
        pixel_size_y=75.0,
        show_q_axis=True,
        horizontal_q_axis="qr",
    )

    SaveDetectorSettings(repository).execute(expected)

    assert LoadDetectorSettings(repository).execute() == expected
    assert repository.get("fitting", "detector.distance") == 1840.5
    assert repository.get("beam", "wavelength") == 0.0103
    assert repository.get("fitting", "detector.horizontal_q_axis") == "qr"


def test_detector_energy_conversion_is_numerically_reversible() -> None:
    wavelength = 0.015

    assert energy_to_wavelength(wavelength_to_energy(wavelength)) == wavelength
