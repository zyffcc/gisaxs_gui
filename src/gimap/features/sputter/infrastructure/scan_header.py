"""Look at one scan folder: how many CBF frames, and what the first frame's header says.

Only the text header of the first frame is read (a few kB; no pixel data): detector and serial number,
pixel size, exposure time and period, time stamp and the flat-field file. The X-ray energy is a hint, with
its source: a ``Wavelength`` line of the header (``header``), else the ``beam_energy`` of a NeXus file of the
same scan in a sibling detector folder (``nexus``), else the ``E<eV>`` in the flat-field or trim file name
(``flat_field``: the detector's calibration energy, which need not be the beam's; the page only shows it).
A folder without frames whose one subfolder holds them (a scan folder with ``p2m/``) is taken as that
subfolder.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..application import NO_FRAMES, ScanInfo

HEADER_BYTES = 65536
_NUMBER = r"([0-9.]+(?:[eE][+-]?\d+)?)"


def header_text(path: Path) -> str:
    """The text header of a mini-CBF file (up to the binary section)."""
    with open(path, "rb") as stream:
        data = stream.read(HEADER_BYTES)
    text = data.decode("latin-1")
    end = text.find("--CIF-BINARY-FORMAT-SECTION--")
    return text if end < 0 else text[:end]


def _search(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text, re.IGNORECASE | re.MULTILINE)


def _float(pattern: str, text: str) -> Optional[float]:
    found = _search(pattern, text)
    try:
        return float(found.group(1)) if found else None
    except ValueError:
        return None


def _frames(folder: Path) -> tuple[Path, list]:
    frames = sorted(folder.glob("*.cbf"))
    if frames or not folder.is_dir():
        return folder, frames
    holders = [sub for sub in sorted(folder.iterdir()) if sub.is_dir() and next(sub.glob("*.cbf"), None) is not None]
    if len(holders) == 1:
        return holders[0], sorted(holders[0].glob("*.cbf"))
    return folder, []


def _nexus_energy(frame: Path) -> Optional[float]:
    """keV from the NeXus file of another detector of the same scan (``<scan>/<detector>/*.nxs``)."""
    try:
        from src.gimap.shared.detector_io.metadata import extract_nxs_metadata
        from src.gimap.shared.detector_io.nexus_files import open_nexus
    except ImportError:  # h5py missing
        return None
    for candidate in sorted(frame.parent.parent.glob("*/*.nxs"))[:8]:
        try:
            with open_nexus(candidate) as handle:
                energy = extract_nxs_metadata(handle).get("energy_kev")
        except Exception:  # noqa: BLE001 - an unreadable companion is no hint
            continue
        if energy:
            return float(energy)
    return None


class CbfHeaderInspector:
    def inspect(self, folder: Path) -> ScanInfo:
        folder, frames = _frames(Path(folder))
        if not frames:
            return ScanInfo(folder, problem=NO_FRAMES)
        try:
            text = header_text(frames[0])
        except OSError as exc:
            return ScanInfo(folder, frames=len(frames), first_frame=frames[0].name, problem=str(exc))
        detector = _search(r"^#\s*Detector:\s*([^,\r\n]+)(?:,\s*S/N\s*([^,\r\n]+))?", text)
        stamp = _search(r"^#\s*(\d{4}-\d{2}-\d{2}T[^\r\n]+)", text)
        flat = _search(r"^#\s*Flat_field:\s*(\S+)", text)
        trim = _search(r"^#\s*Trim_file:\s*(\S+)", text)
        pixel = _float(r"Pixel_size\s+" + _NUMBER + r"\s*m", text)
        energy, source = None, ""
        wavelength = _search(r"^#\s*Wavelength\s+" + _NUMBER + r"\s*(A|Å|nm)\b", text)
        if wavelength:
            value = float(wavelength.group(1))
            energy, source = (12.398419843 / value if wavelength.group(2) != "nm" else 1.239841984 / value), "header"
        if energy is None:
            energy = _nexus_energy(frames[0])
            source = "nexus" if energy else ""
        if energy is None:
            for name in (flat.group(1) if flat else "", trim.group(1) if trim else ""):
                found = re.search(r"_E(\d{4,6})(?:_|\.|$)", name)
                if found:
                    energy, source = int(found.group(1)) / 1000.0, "flat_field"
                    break
        return ScanInfo(
            folder=folder, frames=len(frames), first_frame=frames[0].name,
            detector=detector.group(1).strip() if detector else "",
            serial=(detector.group(2) or "").strip() if detector else "",
            pixel_um=pixel * 1e6 if pixel else None,
            exposure_s=_float(r"Exposure_time\s+" + _NUMBER + r"\s*s", text),
            period_s=_float(r"Exposure_period\s+" + _NUMBER + r"\s*s", text),
            timestamp=stamp.group(1).strip() if stamp else "",
            flat_field=flat.group(1) if flat else "",
            energy_kev_hint=energy, energy_hint_source=source)


__all__ = ["CbfHeaderInspector", "NO_FRAMES", "header_text"]
