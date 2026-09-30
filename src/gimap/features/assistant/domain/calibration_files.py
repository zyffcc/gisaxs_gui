"""Recognise calibration material by name and content (no file access here).

Used when a frame has no geometry: which nearby files are calibration results
(pyFAI ``.poni``, GIMaP calibration ``.json``), images of a calibration
standard, or logs that may state the energy, distance or beam centre.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

HC_KEV_ANGSTROM = 12.398419843320026

STANDARD_NAMES = {
    "agbh": "silver behenate (AgBH)",
    "lab6": "lanthanum hexaboride (LaB6)",
    "ceo2": "cerium dioxide (CeO2)",
    "lab6_ceo2": "LaB6 + CeO2 mixture",
    "si": "silicon powder (Si)",
    "al2o3": "corundum (Al2O3)",
    "cr2o3": "chromium oxide (Cr2O3)",
}
SUPPORTED_STANDARDS = ("agbh", "lab6", "ceo2", "lab6_ceo2")
"""Standards GIMaP's calibration can fit; the others are recognised but need another tool."""

_TOKEN_ALIASES = {
    "agbh": ("agbh", "agbe", "agbeh", "agb", "behenate", "silverbehenate", "agbehenate"),
    "lab6": ("lab6", "hexaboride", "lanthanumhexaboride"),
    "ceo2": ("ceo2", "ceo", "ceria", "ceriumoxide", "ceriumdioxide"),
    "si": ("si", "silicon", "sipowder"),
    "al2o3": ("al2o3", "corundum", "alumina"),
    "cr2o3": ("cr2o3",),
}
_SUBSTRING_ALIASES = {
    "agbh": ("behenate", "agbh", "agbeh"),
    "lab6": ("lab6", "hexaboride"),
    "ceo2": ("ceo2",),
    "al2o3": ("al2o3", "corundum"),
    "cr2o3": ("cr2o3",),
}
CALIBRATION_WORDS = ("calib", "kalib", "poni", "standard")
CALIBRATION_TOKENS = ("cal", "std", "stds", "geometry", "geom", "setup")
LOG_WORDS = ("log", "fio", "param", "meta", "setup", "info", "notes", "readme", "scan", "beamtime", "exp", "config")

IMAGE_SUFFIXES = (".cbf", ".tif", ".tiff", ".nxs", ".h5", ".hdf5", ".edf")
READABLE_IMAGE_SUFFIXES = (".cbf", ".tif", ".tiff", ".nxs")
"""Image formats GIMaP reads (and can calibrate from)."""
TEXT_SUFFIXES = (".log", ".txt", ".fio", ".dat", ".json", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".md", ".csv", ".xml", ".par", ".prm")

POINT_OF_NORMAL_INCIDENCE = "poni"
GIMAP_CALIBRATION = "gimap_calibration"
STANDARD_IMAGE = "standard_image"
LOG = "log"

_HINT_WORDS = re.compile(
    r"distance|sdd|s[_-]?d[_-]?d|det[_ -]?dist|detector|wavelength|lambda|energy|kev|\bev\b|beam[_ ]?cent|"
    r"cent(er|re)[_ ]?[xy]|\bbc[xy]\b|beam[_ ]?[xy]|poni|pixel|incidence|alpha[_ ]?i|\bai\b|omega|\bom\b|"
    r"theta|calib|agbh|agbe|behenate|lab6|ceo2|standard",
    re.IGNORECASE,
)


def tokens(name: str) -> list[str]:
    """Lower-case alphanumeric tokens of a file name (``AgBH_calib-01`` → agbh, calib, 01)."""
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


WEAK_STANDARDS = ("si", "al2o3", "cr2o3")
"""Also common as substrates or samples: named as a standard only next to a calibration word."""
_CONTEXT_TOKENS = ("powder", "std", "standard", "calib", "calibration", "cal", "nist", "srm640", "srm676", "srm674")


def standard_from_name(name: str) -> Optional[str]:
    """The calibration standard a file name (or relative path) refers to, or ``None``."""
    first = _single_standard(name)
    if first in ("lab6", "ceo2"):
        rest = name.lower().replace("lab6", " ").replace("hexaboride", " ") if first == "lab6" else name.lower().replace("ceo2", " ")
        if _single_standard(rest) in ("lab6", "ceo2") and _single_standard(rest) != first:
            return "lab6_ceo2"
    return first


_MODULE = re.compile(r"^(?P<stem>.+)_m(?P<module>\d{2})\.nxs$", re.IGNORECASE)


def module_series(name: str) -> Optional[tuple[str, int]]:
    """``(stem, module)`` of one file of a multi-module NeXus series (``…_m01.nxs``), else ``None``."""
    match = _MODULE.match(name)
    return (match.group("stem"), int(match.group("module"))) if match else None


RANK_WORDS = ("giwaxs", "waxs")
FINAL_WORDS = ("final", "best", "good", "redone")


def name_preference(name: str) -> int:
    """How much a calibration's name suggests it is the one used: 'final' > 'redone' > plain."""
    lowered = name.lower()
    if "final" in lowered or "best" in lowered or "good" in lowered:
        return 2
    return 1 if "redone" in lowered or "redo" in lowered else 0


def _single_standard(name: str) -> Optional[str]:
    words = tokens(name)
    context = any(word in _CONTEXT_TOKENS or word.startswith("calib") for word in words)
    for key, aliases in _TOKEN_ALIASES.items():
        if key in WEAK_STANDARDS and not context:
            continue
        if any(word in aliases or word.rstrip("0123456789") in aliases for word in words):
            return key
    squashed = "".join(words)
    for key, aliases in _SUBSTRING_ALIASES.items():
        if key in WEAK_STANDARDS and not context:
            continue
        if any(alias in squashed for alias in aliases):
            return key
    return None


def mentions_calibration(name: str) -> bool:
    lowered = name.lower()
    words = tokens(name)
    return any(word in lowered for word in CALIBRATION_WORDS) or any(word in CALIBRATION_TOKENS for word in words)


def looks_like_log(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in LOG_WORDS)


def suffix(name: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", name)
    return match.group(1).lower() if match else ""


def classify(name: str) -> Optional[str]:
    """``poni``, ``gimap_calibration`` (a json named like a calibration), ``standard_image``, ``log``."""
    ending = suffix(name)
    if ending == ".poni":
        return POINT_OF_NORMAL_INCIDENCE
    if ending == ".json" and mentions_calibration(name):
        return GIMAP_CALIBRATION
    if ending in IMAGE_SUFFIXES and (standard_from_name(name) or mentions_calibration(name)):
        return STANDARD_IMAGE
    if ending in TEXT_SUFFIXES and (ending in (".log", ".fio") or looks_like_log(name) or mentions_calibration(name)):
        return LOG
    return None


def text_hints(text: str, *, limit: int = 60) -> list[str]:
    """Lines of a text file that mention geometry, energy, incidence angle or a calibrant."""
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and len(line) <= 400 and _HINT_WORDS.search(line):
            lines.append(line)
            if len(lines) >= limit:
                break
    return lines


@dataclass(frozen=True)
class PoniGeometry:
    distance_mm: float
    """Sample to beam-centre distance (the direct beam, not the PONI, when the detector is tilted)."""
    beam_center_x_px: float
    beam_center_y_px: float
    """Canonical pixels: pixel j spans [j, j+1]."""
    pixel_size_x_m: Optional[float]
    pixel_size_y_m: Optional[float]
    wavelength_angstrom: Optional[float]
    tilt_deg: float
    detector: str
    notes: tuple[str, ...]


def _float(value) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


DETECTOR_PIXELS_M = (
    ("pilatus", 172e-6), ("eiger", 75e-6), ("perkin", 200e-6), ("lambda", 55e-6), ("mar3450", 100e-6),
    ("mar345", 100e-6), ("jungfrau", 75e-6),
)
"""Pixel sizes of detectors a ``.poni`` file may name instead of listing ``pixel1``/``pixel2``."""


def parse_poni(text: str) -> PoniGeometry:
    """A pyFAI ``.poni`` file as a direct-beam geometry (pyFAI's ``Geometry.getFit2D`` conversion).

    The PONI is where the detector normal through the sample meets the detector, not the direct
    beam: with rotations ``rot1`` (about the vertical axis) and ``rot2`` (about the horizontal
    axis) the beam lands at ``poni2 − L·tan(rot1)`` and ``poni1 + L·tan(rot2)/cos(rot1)``.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    pixel1 = _float(values.get("pixelsize1"))
    pixel2 = _float(values.get("pixelsize2"))
    config = values.get("detector_config")
    if config and (pixel1 is None or pixel2 is None):
        found1 = re.search(r'"pixel1"\s*:\s*([0-9.eE+-]+)', config)
        found2 = re.search(r'"pixel2"\s*:\s*([0-9.eE+-]+)', config)
        pixel1 = pixel1 if pixel1 is not None else (_float(found1.group(1)) if found1 else None)
        pixel2 = pixel2 if pixel2 is not None else (_float(found2.group(1)) if found2 else None)
    if not pixel1 or not pixel2:
        name = values.get("detector", "").lower()
        known = next((size for key, size in DETECTOR_PIXELS_M if key in name), None)
        pixel1, pixel2 = pixel1 or known, pixel2 or known
    distance = _float(values.get("distance"))
    poni1, poni2 = _float(values.get("poni1")), _float(values.get("poni2"))
    if distance is None or poni1 is None or poni2 is None or not pixel1 or not pixel2:
        raise ValueError("The .poni file lacks Distance, Poni1, Poni2 or the pixel size.")
    rot1 = _float(values.get("rot1")) or 0.0
    rot2 = _float(values.get("rot2")) or 0.0
    cos_tilt = math.cos(rot1) * math.cos(rot2)
    center_x = (poni2 - distance * math.tan(rot1)) / pixel2
    center_y = (poni1 + distance * math.tan(rot2) / math.cos(rot1)) / pixel1
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))
    wavelength = _float(values.get("wavelength"))
    notes = []
    if tilt > 0.5:
        notes.append(f"The detector is tilted by {tilt:.2f}°; GIMaP's geometry is flat, so q is approximate far from the beam.")
    return PoniGeometry(
        distance_mm=distance / cos_tilt * 1e3,
        beam_center_x_px=center_x,
        beam_center_y_px=center_y,
        pixel_size_x_m=pixel2,
        pixel_size_y_m=pixel1,
        wavelength_angstrom=wavelength * 1e10 if wavelength else None,
        tilt_deg=tilt,
        detector=values.get("detector", ""),
        notes=tuple(notes),
    )


READABLE_SUFFIXES = tuple(dict.fromkeys((*IMAGE_SUFFIXES, ".poni", *TEXT_SUFFIXES)))
"""Kinds of file the assistant may read: detector images, calibrations, text and logs."""
PRIVATE_FOLDERS = frozenset({
    "appdata", "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "library", "etc", "proc", "sys",
})


def private_name(name: str) -> bool:
    """A hidden (``.ssh``) or system (``AppData``, ``Windows`` …) file or folder name."""
    return (name.startswith(".") and name not in (".", "..")) or name.lower() in PRIVATE_FOLDERS


def private_path(path: str, base: Optional[str] = None) -> bool:
    """Whether ``path`` goes through a hidden or system folder below ``base``.

    Only the part below ``base`` (a folder the frame lives in, or one the user
    named) counts: data kept under ``AppData\\Local\\Temp`` stays readable, while
    a search never wanders into ``.ssh`` or ``AppData`` from there.
    """
    text = str(path).replace("\\", "/")
    if base:
        root = str(base).replace("\\", "/").rstrip("/")
        if text.lower() == root.lower() or text.lower().startswith(root.lower() + "/"):
            text = text[len(root):]
    return any(private_name(part) for part in text.split("/") if part)


_QUOTED = re.compile(r'"([^"\r\n]+)"|\'([^\'\r\n]+)\'|“([^”\r\n]+)”|‘([^’\r\n]+)’|「([^」\r\n]+)」|『([^』\r\n]+)』')
_WINDOWS_PATH = re.compile(r'(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])[^\r\n"<>|?*]*')
_POSIX_PATH = re.compile(r'(?<![\w:])/(?:[^\s"<>|?*/]+/)+[^\s"<>|?*/]*')
_TRAILING = " \t.,;:!?)]}，。；：！？、）】」』"


def path_candidates(text: str) -> list[list[str]]:
    """Paths a person may have written in free text, one list per mention, longest first.

    A Windows path can contain spaces and run into the following words
    (``D:\\data\\calib\\LaB6.cbf 是标样``), so each mention offers shorter
    versions, cut at spaces and non-ASCII characters; the first that exists on
    disk is the one meant.
    """
    groups: list[list[str]] = []
    for match in _QUOTED.finditer(text or ""):
        value = next(group for group in match.groups() if group)
        if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|/)", value.strip()):
            groups.append([value.strip()])
    for pattern in (_WINDOWS_PATH, _POSIX_PATH):
        for match in pattern.finditer(text or ""):
            raw = match.group(0).rstrip(_TRAILING)
            if len(raw) < 4:
                continue
            options = [raw]
            for index in range(len(raw) - 1, 2, -1):
                if raw[index].isspace() or ord(raw[index]) > 127 or raw[index] in _TRAILING:
                    shorter = raw[:index].rstrip(_TRAILING)
                    if len(shorter) >= 3 and shorter not in options:
                        options.append(shorter)
                if len(options) >= 160:
                    break
            groups.append(options)
    unique: list[list[str]] = []
    seen: set[str] = set()
    for options in groups:
        if options[0] not in seen:
            seen.add(options[0])
            unique.append(options)
    return unique


def energy_kev(wavelength_angstrom: Optional[float]) -> Optional[float]:
    return HC_KEV_ANGSTROM / wavelength_angstrom if wavelength_angstrom else None


def wavelength_angstrom(energy: Optional[float]) -> Optional[float]:
    return HC_KEV_ANGSTROM / energy if energy else None


__all__ = [
    "CALIBRATION_WORDS",
    "GIMAP_CALIBRATION",
    "HC_KEV_ANGSTROM",
    "IMAGE_SUFFIXES",
    "LOG",
    "POINT_OF_NORMAL_INCIDENCE",
    "PoniGeometry",
    "READABLE_IMAGE_SUFFIXES",
    "STANDARD_IMAGE",
    "STANDARD_NAMES",
    "SUPPORTED_STANDARDS",
    "TEXT_SUFFIXES",
    "WEAK_STANDARDS",
    "classify",
    "energy_kev",
    "looks_like_log",
    "mentions_calibration",
    "module_series",
    "name_preference",
    "PRIVATE_FOLDERS",
    "READABLE_SUFFIXES",
    "parse_poni",
    "path_candidates",
    "private_name",
    "private_path",
    "standard_from_name",
    "suffix",
    "text_hints",
    "tokens",
    "wavelength_angstrom",
]
