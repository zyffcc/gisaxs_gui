"""Recognise calibration material by name and content (no file access here).

Used when a frame has no geometry: which nearby files are calibration results
(pyFAI ``.poni``, GIMaP calibration ``.json``), images of a calibration
standard, or logs that may state the energy, distance or beam centre.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.gimap.shared.geometry.detector_tilt import DetectorTilt
from src.gimap.shared.geometry.poni import DETECTOR_PIXELS_M, read_poni  # noqa: F401  (re-exported)

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

HDF5_IMAGE_SUFFIXES = (".nxs", ".nx", ".nx5", ".h5", ".hdf5", ".hdf")
"""Every suffix of a NeXus / HDF5 file (the shared detector reader's list; no file access here)."""
IMAGE_SUFFIXES = (".cbf", ".tif", ".tiff", *HDF5_IMAGE_SUFFIXES, ".edf")
READABLE_IMAGE_SUFFIXES = IMAGE_SUFFIXES
"""Image formats GIMaP reads (and can calibrate from): one list for the search, the command line and the window,
so a format the Calibration window fits (``.edf``) is never refused by the tools."""
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


FAR_FOLDER = (99, 1)
"""``folder_place`` of a file with no folder in common with the frame (another drive)."""


def folder_place(frame_folder: str, path: str) -> tuple[int, int]:
    """Where the folder of ``path`` lies seen from ``frame_folder``, as ``(up, below)``: the levels up from
    the frame's folder to the folder both lie in, and 1 when ``path`` lies in a folder below that one, else 0.
    Sorted, nearest first: (0, 0) the frame's folder, (0, 1) a subfolder of it, (1, 0) the parent folder,
    (1, 1) a sibling folder (or a folder in one), (2, 0) two levels up, (2, 1) a folder there … Both paths
    absolute and spelled alike (``os.path.normcase``); nothing is read."""
    here = [part for part in re.split(r"[\\/]+", frame_folder) if part]
    there = [part for part in re.split(r"[\\/]+", path) if part][:-1]
    common = 0
    for mine, theirs in zip(here, there):
        if mine != theirs:
            break
        common += 1
    if common == 0:
        return FAR_FOLDER
    return len(here) - common, int(len(there) > common)


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
    tilt: Optional[DetectorTilt] = None
    """What the flat geometry above leaves out (the ``.poni``'s own distance, PONI, rotations and pixel sizes);
    ``None`` when the file has no rotation. Analyze estimates from it how far |q| is off (``detector_tilt``)."""
    sample_roll_deg: float = 0.0
    sample_roll_measurement: Optional[dict] = None


def parse_poni(text: str, *, frame_shape=None) -> PoniGeometry:
    """A pyFAI ``.poni`` file as a direct-beam geometry (pyFAI's ``Geometry.getFit2D`` conversion).

    The PONI is where the detector normal through the sample meets the detector, not the direct
    beam: with rotations ``rot1`` (about the vertical axis) and ``rot2`` (about the horizontal
    axis) the beam lands at ``poni2 − L·tan(rot1)`` and ``poni1 + L·tan(rot2)/cos(rot1)``. The file is read by
    ``shared/geometry/poni.read_poni`` (the reader of Analyze's Load .poni… too).
    """
    stated = read_poni(text, frame_shape=frame_shape)
    poni = stated.tilt
    # The conversion is the tilt's own (``DetectorTilt.direct_beam_px``), so Analyze's estimate of what the tilt
    # costs compares the .poni with exactly this flat geometry.
    center_x, center_y = poni.direct_beam_px()
    tilt = poni.tilt_deg
    notes = []
    if tilt > 0.5:
        notes.append(f"The detector is tilted by {tilt:.2f}°; the tilt is part of the geometry (pyFAI's rotations).")
    return PoniGeometry(
        distance_mm=poni.beam_distance_m * 1e3,
        beam_center_x_px=center_x,
        beam_center_y_px=center_y,
        pixel_size_x_m=poni.pixel2_m,
        pixel_size_y_m=poni.pixel1_m,
        wavelength_angstrom=stated.wavelength_angstrom,
        tilt_deg=tilt,
        detector=stated.detector,
        notes=tuple(notes),
        tilt=poni if stated.tilted or poni.orientation != 3 else None,
        sample_roll_deg=stated.sample_roll_deg or 0.0,
        sample_roll_measurement=stated.sample_roll_measurement,
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
# Like a Windows path, a POSIX path may hold spaces and run into the words after it, so it is taken to the end
# of its line (``path_candidates`` offers the shorter versions).  It starts at a slash that begins a folder
# name and follows neither a word (and/or), a colon nor another slash (https://…); the look-ahead lets a
# second path of the same line start a mention of its own.
_POSIX_PATH = re.compile(r'(?<![\w:/])(?=(/[^\s"<>|?*/]+/[^\r\n"<>|?*]*))')
_TRAILING = " \t.,;:!?)]}，。；：！？、）】」』"
MAX_PATH_OPTIONS = 160
"""Versions of one mention that are offered: the shortest ones. A mention with more places to cut is a
path followed by a long sentence on its line, and the path ends near its start."""


def path_candidates(text: str) -> list[list[str]]:
    """Paths a person may have written in free text, one list per mention, longest first.

    A path can contain spaces and run into the following words
    (``D:\\data\\calib\\LaB6.cbf 是标样``, ``/data/my run/LaB6.cbf 是标样``), so each
    mention offers shorter versions, cut at spaces and non-ASCII characters; the
    first that exists on disk is the one meant. The places to cut are counted from
    the start of the mention (at most ``MAX_PATH_OPTIONS``): however long the line
    goes on after a path, the path itself is among the options.
    """
    groups: list[list[str]] = []
    for match in _QUOTED.finditer(text or ""):
        value = next(group for group in match.groups() if group)
        if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|/)", value.strip()):
            groups.append([value.strip()])
    for pattern in (_WINDOWS_PATH, _POSIX_PATH):
        for match in pattern.finditer(text or ""):
            raw = match.group(match.lastindex or 0).rstrip(_TRAILING)
            if len(raw) < 4:
                continue
            cuts: list[str] = []  # shortest first
            for index in range(3, len(raw)):
                if raw[index].isspace() or ord(raw[index]) > 127 or raw[index] in _TRAILING:
                    shorter = raw[:index].rstrip(_TRAILING)
                    if len(shorter) >= 3 and shorter not in cuts:
                        cuts.append(shorter)
                if len(cuts) >= MAX_PATH_OPTIONS:
                    break
            else:  # every place to cut was taken: the whole mention is the first option
                cuts.append(raw)
            groups.append(cuts[::-1])
    unique: list[list[str]] = []
    seen: set[str] = set()
    for options in groups:
        if options[0] not in seen:
            seen.add(options[0])
            unique.append(options)
    return unique


_NAME = r"A-Za-z0-9_.+()\-"
_NAMED_FILE = re.compile(
    rf"(?<![{_NAME}])([A-Za-z0-9_][{_NAME}]*\.(?:poni|json|cbf|tiff?|nxs|h5|hdf5|edf))(?![A-Za-z0-9_+()\-]|\.[A-Za-z0-9_])",
    re.IGNORECASE,
)


def file_name_mentions(text: str) -> list[str]:
    """Names of files that may hold a calibration, as free text writes them, each once in the order written.

    ``calibration LaB6_take2.poni`` names a file without saying where it is; the last part of a path
    counts too (``calib/LaB6_take2.poni``). Only the endings of calibration files and detector images,
    so that ``0.2 deg`` or ``12.4 keV`` is never a name; a name with spaces needs its path in quotes.
    """
    names: list[str] = []
    for match in _NAMED_FILE.finditer(text or ""):
        if match.group(1).lower() not in (name.lower() for name in names):
            names.append(match.group(1))
    return names


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
    "FAR_FOLDER",
    "classify",
    "energy_kev",
    "folder_place",
    "looks_like_log",
    "mentions_calibration",
    "module_series",
    "name_preference",
    "PRIVATE_FOLDERS",
    "READABLE_SUFFIXES",
    "parse_poni",
    "path_candidates",
    "file_name_mentions",
    "private_name",
    "private_path",
    "standard_from_name",
    "suffix",
    "text_hints",
    "tokens",
    "wavelength_angstrom",
]
