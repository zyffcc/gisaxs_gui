"""Invented scan folders and a fake of the scan header for the Sputter page (no real data): shared by
``test_sputter_feature.py``, ``test_sputter_page.py``, ``test_sputter_main_screen.py`` and ``test_sputter_live.py``."""

from __future__ import annotations

from pathlib import Path

from src.gimap.features.sputter.application import ScanInfo


def scan_folder(root: Path, name: str = "zz_invented_scan_00001", frames: int = 3) -> Path:
    """A folder of invented mini-CBF frames (a text header and a few bytes; no real data)."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, frames + 1):
        (folder / f"{name}_{index:05d}.cbf").write_bytes(cbf_header().encode("latin-1") + b"\x0c\x1a\x04\xd5\x00\x00")
    return folder


def cbf_header(*, serial: str = "99-0001", stamp: str = "2030-01-02T03:04:05.678", flat: str = "FF_x99_E9000_T4500.tif",
               extra: str = "") -> str:
    return ("###CBF: VERSION 1.5\n\ndata_invented\n\n_array_data.header_convention \"PILATUS_1.2\"\n"
            "_array_data.header_contents\n;\n"
            f"# Detector: PILATUS 1M, S/N {serial}, Invented\n# {stamp}\n# Pixel_size 172e-6 m x 172e-6 m\n"
            "# Silicon sensor, thickness 0.000450 m\n# Exposure_time 0.0300000 s\n# Exposure_period 0.0400000 s\n"
            "# Tau = 100.0e-09 s\n# Count_cutoff 123456 counts\n# Threshold_setting: 4500 eV\n"
            f"# Flat_field: {flat}\n# Trim_file: x99_E9000_T4500.bin\n{extra};\n\n_array_data.data\n;\n"
            "--CIF-BINARY-FORMAT-SECTION--\n")


class FakeInspector:
    def __init__(self, **values):
        self.values = values
        self.calls = []

    def inspect(self, folder):
        self.calls.append(Path(folder))
        values = dict(dict(frames=3, detector="PILATUS 1M", serial="99-0001", pixel_um=172.0, exposure_s=0.03,
                           period_s=0.04, timestamp="2030-01-02T03:04:05", flat_field="FF_x99_E9000_T4500.tif",
                           energy_kev_hint=9.0, energy_hint_source="header"), **self.values)
        return ScanInfo(Path(folder), **values)
