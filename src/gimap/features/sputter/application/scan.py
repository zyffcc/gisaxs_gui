"""What one scan folder says about itself, and where each geometry value on the page comes from.

A mini-CBF header (PILATUS) names the detector, its serial number, the pixel size, the exposure and the
time stamp; it rarely holds the X-ray energy, and never the sample-detector distance, the incidence angle,
the beam centre or the sample roll. The energy can be read from a ``Wavelength`` line of the header or from
a companion NeXus file of the same scan (``beam_energy``). The ``E<eV>`` in the flat-field file name is the
detector's calibration energy, not necessarily the beam's: it is only shown beside the field, never filled in.

Every geometry value comes, in this order, from
1. the values last used for this same folder;
2. a geometry preset the user saved (matched by the detector serial number, the flat-field file name and
   the date of the scan);
3. (the energy only) a ``Wavelength`` line of the header or the scan's NeXus file;
4. the values used last time for any folder (“please check”) — not the sample roll, which belongs to one
   set-up: without 1 or 2 it is 0, said as assumed.

A value the user chose for this folder or this beamtime is not overridden by a file (a different energy of
the files is shown next to it). The mask, too, belongs to one set-up: it is taken from this folder's values
only; the polymer and the time window come from the values used last.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

GROUPS = ("xray", "sdd", "alpha", "beam", "roll")
"""The geometry fields of the page that carry a source: X-rays (energy or wavelength), distance, incidence,
beam centre (column and row), sample roll."""
GROUP_KEYS = {"xray": ("energy_kind", "energy"), "sdd": ("sdd",), "alpha": ("alpha",),
              "beam": ("beam_col", "beam_row"), "roll": ("roll",)}
SAMPLE_KEYS = ("metal", "polymer", "mask", "window")
FOLDER_ONLY_KEYS = ("rate", "t0")
"""Kept for the folder only: a deposition rate belongs to one scan."""
FOLDER_SAMPLE_KEYS = ("mask",)
"""Taken from this folder's values only (a detector mask belongs to one set-up), never from the last used."""
MAX_FOLDERS = 300
HINT_TOLERANCE = 0.005
"""A header energy hint that differs from the value used by more than this share is shown next to it."""

SOURCE_TEXT = {
    "header": "from the file header",
    "nexus": "from the scan's NeXus file",
    "folder": "as last used for this folder",
    "preset": "from preset {name}",
    "last": "last used — please check",
    "missing": "not in the files — please enter",
    "assumed": "not in the files: 0 assumed — please check",
    "auto": "empty: estimated from the frames",
    "entered": "entered",
}
HINT_TEXT = {
    "header": "the file header suggests {energy} keV",
    "nexus": "the scan's NeXus file suggests {energy} keV",
    "flat_field": ("the flat-field file name says E = {energy} keV: the detector's calibration energy, not "
                   "necessarily the beam's"),
}
"""What the files say about the energy, beside the value used (by ``ScanInfo.energy_hint_source``)."""
FILLING_HINTS = ("header", "nexus")
"""Energy hints that fill the field; a flat-field file name is only shown beside it."""
SUMMARY_HINT = {"header": "header: {energy} keV", "nexus": "NeXus file: {energy} keV",
                "flat_field": "flat field: E = {energy} keV"}
NO_FRAMES = "No CBF frames in this folder."


@dataclass(frozen=True)
class ScanInfo:
    folder: Path
    frames: int = 0
    first_frame: str = ""
    detector: str = ""
    serial: str = ""
    pixel_um: Optional[float] = None
    exposure_s: Optional[float] = None
    period_s: Optional[float] = None
    timestamp: str = ""
    flat_field: str = ""
    energy_kev_hint: Optional[float] = None
    energy_hint_source: str = ""        # "header" (a Wavelength line), "nexus" or "flat_field" (the file name)
    problem: str = ""

    @property
    def date(self) -> str:
        """YYYY-MM-DD of the time stamp, or ""."""
        return self.timestamp[:10] if len(self.timestamp) >= 10 and self.timestamp[4] == "-" else ""


@dataclass(frozen=True)
class Source:
    kind: str                   # a key of SOURCE_TEXT
    name: str = ""              # the preset's name
    hint_kev: Optional[float] = None   # an energy of the files that differs from the value used
    hint_source: str = ""       # where that energy comes from (a key of HINT_TEXT)

    def template(self) -> str:
        return SOURCE_TEXT.get(self.kind, "")


@dataclass
class Prefill:
    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)


def folder_key(folder) -> str:
    return os.path.normcase(os.path.abspath(str(folder)))


def _has(values: dict, group: str) -> bool:
    keys = GROUP_KEYS[group]
    if group == "beam":  # an empty beam centre (automatic estimate) is a value too
        return all(key in values for key in keys)
    return all(values.get(key) not in (None, "") for key in keys)


def _in_range(date: str, low, high) -> bool:
    if not low and not high:
        return True
    if not date:
        return False
    return (not low or str(low) <= date) and (not high or date <= str(high))


def matching_preset(info: ScanInfo, presets) -> Optional[dict]:
    """The user's geometry preset for this scan: same detector serial, same flat-field file (when the preset
    names one) and the scan's date in its range; the narrowest range wins, then the newest preset."""
    found = []
    for order, preset in enumerate(presets or []):
        if not isinstance(preset, dict) or not preset.get("serial") or preset.get("serial") != info.serial:
            continue
        if preset.get("flat_field") and preset["flat_field"] != info.flat_field:
            continue
        if not _in_range(info.date, preset.get("date_from"), preset.get("date_to")):
            continue
        found.append(((_span_days(preset), -order), preset))
    if not found:
        return None
    return min(found, key=lambda item: item[0])[1]


def _span_days(preset: dict) -> int:
    """Days the preset's date range spans (an open range counts as very wide)."""
    try:
        return (date.fromisoformat(str(preset["date_to"])) - date.fromisoformat(str(preset["date_from"]))).days
    except (KeyError, TypeError, ValueError):
        return 10 ** 6


def _copy(values: dict, source: dict, group: str) -> None:
    for key in GROUP_KEYS[group]:
        values[key] = source.get(key)


def prefill(info: ScanInfo, memory: dict) -> Prefill:
    """The page's values for ``info`` and the source of each geometry group."""
    memory = memory if isinstance(memory, dict) else {}
    folders = memory.get("folders") if isinstance(memory.get("folders"), dict) else {}
    folder = folders.get(folder_key(info.folder)) or {}
    last = memory.get("last") if isinstance(memory.get("last"), dict) else {}
    preset = matching_preset(info, memory.get("presets"))
    hint, hint_source = info.energy_kev_hint, info.energy_hint_source or "header"
    out = Prefill()
    for group in GROUPS:
        if _has(folder, group):
            _copy(out.values, folder, group); out.sources[group] = Source("folder")
        elif preset is not None and _has(preset, group):
            _copy(out.values, preset, group); out.sources[group] = Source("preset", name=str(preset.get("name", "")))
        elif group == "xray" and hint and hint_source in FILLING_HINTS:
            out.values.update(energy_kind="energy", energy=hint)
            out.sources[group] = Source(hint_source)
        elif group != "roll" and _has(last, group):
            _copy(out.values, last, group); out.sources[group] = Source("last")
        elif group == "beam":
            out.values.update(beam_col=None, beam_row=None); out.sources[group] = Source("auto")
        elif group == "roll":  # a roll belongs to one set-up: not carried from another folder
            out.values["roll"] = 0.0; out.sources[group] = Source("assumed")
        else:
            out.sources[group] = Source("missing")
    xray = out.sources.get("xray")
    if xray is not None and hint and xray.kind not in FILLING_HINTS:
        used = energy_kev(out.values.get("energy_kind"), out.values.get("energy"))
        if not used or abs(used - hint) > HINT_TOLERANCE * hint:
            out.sources["xray"] = Source(xray.kind, xray.name, hint_kev=hint, hint_source=hint_source)
    if out.values.get("beam_col") in (None, "") and out.sources.get("beam") and out.sources["beam"].kind != "auto":
        out.sources["beam"] = Source("auto")
    for key in SAMPLE_KEYS:
        if folder.get(key) not in (None, ""):
            out.values[key] = folder[key]
        elif key in FOLDER_SAMPLE_KEYS:
            out.values[key] = ""  # a detector mask belongs to one set-up: not carried from another folder
        elif last.get(key) not in (None, ""):
            out.values[key] = last[key]
    for key in FOLDER_ONLY_KEYS:
        out.values[key] = folder.get(key)
    return out


def energy_kev(kind, value) -> Optional[float]:
    """The photon energy (keV) of a value given as energy (keV) or wavelength (nm)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value if kind != "wavelength" else 1.239841984 / value


def remember(memory: dict, folder, values: dict) -> dict:
    """``memory`` with ``values`` kept for ``folder`` and as the last used ones (a new dict)."""
    memory = dict(memory) if isinstance(memory, dict) else {}
    folders = dict(memory.get("folders") or {})
    key = folder_key(folder)
    folders.pop(key, None)
    folders[key] = dict(values)
    while len(folders) > MAX_FOLDERS:
        folders.pop(next(iter(folders)))
    memory["folders"] = folders
    memory["last"] = {k: v for k, v in values.items() if k not in FOLDER_ONLY_KEYS}
    return memory


def make_preset(name: str, info: ScanInfo, values: dict, date_from: str, date_to: str) -> dict:
    """A geometry preset from the page's values, for scans of this detector and flat field in the date range."""
    preset = {"name": name.strip(), "serial": info.serial, "flat_field": info.flat_field,
              "date_from": date_from, "date_to": date_to}
    for group in GROUPS:
        for key in GROUP_KEYS[group]:
            preset[key] = values.get(key)
    return preset


def add_preset(memory: dict, preset: dict) -> dict:
    """``memory`` with ``preset`` added (a preset of the same name is replaced)."""
    memory = dict(memory) if isinstance(memory, dict) else {}
    presets = [p for p in (memory.get("presets") or []) if isinstance(p, dict) and p.get("name") != preset["name"]]
    memory["presets"] = presets + [preset]
    return memory


__all__ = ["FILLING_HINTS", "GROUPS", "GROUP_KEYS", "HINT_TEXT", "NO_FRAMES", "Prefill", "SOURCE_TEXT", "SUMMARY_HINT",
           "ScanInfo", "Source", "add_preset", "energy_kev", "folder_key", "make_preset", "matching_preset", "prefill",
           "remember"]
