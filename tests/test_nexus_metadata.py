"""Header values of NeXus / HDF5 files in GIMaP's units (``shared/detector_io/nexus_metadata.py``).

The ``units`` attribute decides; a value without one is read as before (by its magnitude); a value in
an unknown unit is left out.  The real P03 Lambda file states its exposure in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gimap.shared.detector_io import extract_nexus_metadata, forget_layout, load_detector_image, open_nexus
from src.gimap.shared.detector_io.metadata import extract_nxs_metadata

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = ROOT / "TestSAXSdata" / "lmbdp03" / "jg_gisaxs_12nm_insitu_00001r3_00001_m01.nxs"
DET = "/entry/instrument/detector"
BEAM = "/entry/instrument/beam"


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_layout()
    yield
    forget_layout()


def _file(path: Path, values: dict, det: str = DET, frames: int = 3) -> Path:
    """A three-frame stack with header datasets ``{path: value}`` or ``{path: (value, units)}``."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{det}/data", data=np.ones((frames, 16, 24), dtype=np.int32))
        for name, value in values.items():
            value, units = value if isinstance(value, tuple) else (value, None)
            dataset = handle.create_dataset(name, data=value)
            if units is not None:
                dataset.attrs["units"] = units
    return path


def _header(tmp_path: Path, values: dict, frame=None, det: str = DET, entry: str = "/entry") -> dict:
    path = _file(tmp_path / f"header_{len(list(tmp_path.iterdir()))}.nxs", values, det)
    with open_nexus(path) as handle:
        return extract_nexus_metadata(handle, det, entry, frame=frame)


@pytest.mark.skipif(not LAMBDA.is_file(), reason="TestSAXSdata/lmbdp03 is not in this checkout")
def test_the_real_lambda_file_gives_its_exposure_in_seconds() -> None:
    with open_nexus(LAMBDA) as handle:
        assert handle[f"{DET}/count_time"].attrs["units"] == "millisecond"  # what the file says: 998 ms
        header = extract_nxs_metadata(handle)
    assert header["exposure_time_s"] == pytest.approx(0.998)
    assert header["energy_kev"] == pytest.approx(12.0) and header["wavelength_angstrom"] == pytest.approx(1.0332, abs=1e-4)
    assert header["pixel_size_x_m"] == pytest.approx(55e-6) and header["pixel_size_y_m"] == pytest.approx(55e-6)
    assert header["detector_name"] == "Lambda" and header["distance_m"] is None  # the translation is not a distance
    assert header["timestamp"] == "2024-10-25T11:31:59.360787+0200"  # the root attribute file_time
    assert load_detector_image(LAMBDA).metadata["exposure_time_s"] == pytest.approx(0.998)


@pytest.mark.parametrize("value, units, kev", [
    (12000.0, "eV", 12.0),
    (12.0, "keV", 12.0),
    (105.0, "keV", 105.0),            # high energy: not divided by 1000 because it is above 100
    (0.9, "keV", 0.9),
    (1.9226e-15, "J", 12.0),
    (np.array([12.4]), b"keV", 12.4),  # a one-element array, the unit stored as bytes
    (12000.0, None, 12.0),            # no unit: by magnitude, as before
    (12.0, None, 12.0),
    (12.0, "furlong", None),          # a unit GIMaP does not know: left out, not guessed
])
def test_energy_follows_its_units(tmp_path: Path, value, units, kev) -> None:
    header = _header(tmp_path, {f"{BEAM}/incident_energy": (value, units)})
    if kev is None:
        assert header["energy_kev"] is None and header["wavelength_angstrom"] is None
    else:
        assert header["energy_kev"] == pytest.approx(kev, rel=1e-4)
        assert header["wavelength_angstrom"] == pytest.approx(12.398419843320026 / kev, rel=1e-4)


@pytest.mark.parametrize("value, units, angstrom", [
    (1.0332, "angstrom", 1.0332),
    (1.0332, "Å", 1.0332),
    (0.10332, "nm", 1.0332),
    (103.32, "pm", 1.0332),
    (1.0332e-10, "m", 1.0332),
    (1.0332e-10, None, 1.0332),  # no unit: metres when tiny, as before
    (1.0332, None, 1.0332),
    (np.array([1.0332, 1.0332, 1.0332]), "angstrom", 1.0332),
])
def test_wavelength_follows_its_units(tmp_path: Path, value, units, angstrom) -> None:
    header = _header(tmp_path, {f"{BEAM}/incident_wavelength": (value, units)})
    assert header["wavelength_angstrom"] == pytest.approx(angstrom) and header["energy_kev"] is None


@pytest.mark.parametrize("name, value, units, metres", [
    ("distance", 3.2, "m", 3.2),
    ("distance", 3200.0, "mm", 3.2),
    ("distance", 320.0, "cm", 3.2),
    ("detector_distance", 3200.0, "mm", 3.2),  # the NXmx / Dectris name
    ("detector_distance", 3.2, None, 3.2),
])
def test_distance_follows_its_units_and_both_names(tmp_path: Path, name, value, units, metres) -> None:
    assert _header(tmp_path, {f"{DET}/{name}": (value, units)})["distance_m"] == pytest.approx(metres)


@pytest.mark.parametrize("units, metres", [("micrometer", 75e-6), ("um", 75e-6), ("µm", 75e-6), ("mm", 75e-3), (None, 75.0)])
def test_pixel_size_follows_its_units(tmp_path: Path, units, metres) -> None:
    header = _header(tmp_path, {f"{DET}/x_pixel_size": (75.0, units), f"{DET}/y_pixel_size": ([75.0], units)})
    assert header["pixel_size_x_m"] == pytest.approx(metres) and header["pixel_size_y_m"] == pytest.approx(metres)


@pytest.mark.parametrize("units, metres", [
    ("meters", 3.2), ("metres", 3.2), ("Meter", 3.2), (" m ", 3.2), ("millimeters", 3.2e-3), ("millimetres", 3.2e-3),
    ("centimeters", 3.2e-2), ("micrometers", 3.2e-6), ("microns", 3.2e-6), ("nanometers", 3.2e-9),
    ("ms", None), ("cms", None), ("furlongs", None),  # a symbol is never read as the plural of another one
])
def test_plural_unit_names_are_the_unit(tmp_path: Path, units: str, metres) -> None:
    """udunits, which NeXus refers to, accepts ``meters``: such a header pre-fills the geometry like ``m`` does."""
    header = _header(tmp_path, {f"{DET}/distance": (3.2, units), f"{DET}/x_pixel_size": ([3.2], units)})
    if metres is None:
        assert header["distance_m"] is None and header["pixel_size_x_m"] is None
    else:
        assert header["distance_m"] == pytest.approx(metres) and header["pixel_size_x_m"] == pytest.approx(metres)


@pytest.mark.parametrize("path, value, units, key, expected", [
    (f"{BEAM}/incident_wavelength", 0.10332, "nanometers", "wavelength_angstrom", 1.0332),
    (f"{BEAM}/incident_wavelength", 1.0332, "angstroms", "wavelength_angstrom", 1.0332),
    (f"{BEAM}/incident_wavelength", 1.0332, "Ångström", "wavelength_angstrom", 1.0332),
    (f"{BEAM}/incident_wavelength", 1.0332e-10, "meters", "wavelength_angstrom", 1.0332),
    (f"{BEAM}/incident_energy", 12000.0, "electronvolts", "energy_kev", 12.0),
    (f"{BEAM}/incident_energy", 12000.0, "electron volt", "energy_kev", 12.0),
    (f"{BEAM}/incident_energy", 12.0, "kiloelectronvolts", "energy_kev", 12.0),
    (f"{BEAM}/incident_energy", 12.0, "MeV", "energy_kev", None),  # not known here: left out, not read as meV or keV
    (f"{DET}/count_time", 2.0, "hours", "exposure_time_s", 7200.0),
    (f"{DET}/count_time", 0.5, "secs", "exposure_time_s", 0.5),
    (f"{DET}/count_time", 998.0, "milliseconds", "exposure_time_s", 0.998),
])
def test_spelled_out_units_of_every_quantity(tmp_path: Path, path: str, value, units: str, key: str, expected) -> None:
    header = _header(tmp_path, {path: (value, units)})
    assert header[key] is None if expected is None else header[key] == pytest.approx(expected, rel=1e-6)


def test_exposure_and_time_stamps_of_the_frame_asked(tmp_path: Path) -> None:
    values = {
        f"{DET}/count_time": (np.array([100.0, 500.0, 1000.0]), "ms"),
        f"{DET}/collection/timestamps": (np.array([0.0, 1500.0, 3000.0]), "millisecond"),
        "/entry/start_time": "2024-10-25T11:31:59+02:00",
    }
    assert _header(tmp_path, values)["exposure_time_s"] == pytest.approx(0.1)  # no frame asked: the first
    second = _header(tmp_path, values, frame=1)
    assert second["exposure_time_s"] == pytest.approx(0.5) and second["frame_time_s"] == pytest.approx(1.5)
    assert second["timestamp"] == "2024-10-25T11:31:59+02:00"
    assert _header(tmp_path, values, frame=2)["exposure_time_s"] == pytest.approx(1.0)
    stamps = np.array([b"2024-10-25T11:32:00", b"2024-10-25T11:32:01", b"2024-10-25T11:32:02"])
    named = _header(tmp_path, {f"{DET}/timestamps": stamps, "/entry/start_time": "2024-10-25T11:31:59"}, frame=2)
    assert named["timestamp"] == "2024-10-25T11:32:02" and named["frame_time_s"] is None
    # Through the reader: the header values are those of the frame that is read.
    path = _file(tmp_path / "per_frame.h5", values)
    assert [load_detector_image(path, frame_idx=k).metadata["exposure_time_s"] for k in range(3)] == pytest.approx([0.1, 0.5, 1.0])


def test_a_detector_group_under_another_name_keeps_its_header(tmp_path: Path) -> None:
    det = "/scan_0001/instrument/lambda"
    values = {
        f"{det}/x_pixel_size": (55.0, "um"),
        f"{det}/y_pixel_size": (55.0, "um"),
        f"{det}/count_time": (998.0, "millisecond"),
        f"{det}/description": "Lambda",
        f"{det}/collection/beam_energy": (12000.0, "eV"),
        f"{det}/distance": (4.5, "m"),
        f"{det}/beam_center_x": (100.0, "pixel"),
        f"{det}/beam_center_y": (5.5e-3, "m"),  # a length: divided by the pixel size
        "/scan_0001/start_time": "2025-01-01T00:00:00",
    }
    header = _header(tmp_path, values, det=det, entry="/scan_0001")
    assert header["detector_name"] == "Lambda" and header["energy_kev"] == pytest.approx(12.0)
    assert header["pixel_size_x_m"] == pytest.approx(55e-6) and header["exposure_time_s"] == pytest.approx(0.998)
    assert header["distance_m"] == pytest.approx(4.5) and header["timestamp"] == "2025-01-01T00:00:00"
    assert header["beam_center_x_px"] == 100.0 and header["header_beam_center_px"] == pytest.approx((100.0, 100.0))
    # The reader finds the group from the dataset; the fixed /entry paths alone would find nothing.
    path = _file(tmp_path / "renamed.nxs", values, det)
    image = load_detector_image(path)
    assert (image.detector_name, image.energy_kev, image.distance_m) == ("Lambda", pytest.approx(12.0), pytest.approx(4.5))
    assert image.metadata["exposure_time_s"] == pytest.approx(0.998)
    with open_nexus(path) as handle:
        fixed = extract_nxs_metadata(handle)
    assert fixed["detector_name"] is None and fixed["energy_kev"] is None


def test_the_classic_paths_and_plain_numbers_read_as_before(tmp_path: Path) -> None:
    """What ``extract_nxs_metadata`` returned before units were honoured, for files without units."""
    path = _file(tmp_path / "classic.nxs", {
        f"{DET}/collection/beam_energy": [12000.0],
        f"{DET}/x_pixel_size": [75e-6],
        f"{DET}/y_pixel_size": 75e-6,
        f"{DET}/distance": 2.5,
        f"{DET}/beam_center_x": [999.0],
        f"{DET}/beam_center_y": [888.0],
        f"{DET}/count_time": [0.5],
        f"{DET}/local_name": "pilatus2m",
        "/entry/start_time": b"2024-01-02T03:04:05",
    })
    with h5py.File(path, "r") as handle:
        header = extract_nxs_metadata(handle)
    assert header == {
        "energy_kev": 12.0, "wavelength_angstrom": pytest.approx(1.0332016536100022), "detector_name": "pilatus2m",
        "pixel_size_x_m": 75e-6, "pixel_size_y_m": 75e-6, "distance_m": 2.5,
        "beam_center_x_px": 999.0, "beam_center_y_px": 888.0, "header_beam_center_px": (999.0, 888.0),
        "exposure_time_s": 0.5, "timestamp": "2024-01-02T03:04:05", "frame_time_s": None,
    }
    empty = _file(tmp_path / "empty.nxs", {})
    with h5py.File(empty, "r") as handle:
        assert all(value is None for value in extract_nxs_metadata(handle).values())
