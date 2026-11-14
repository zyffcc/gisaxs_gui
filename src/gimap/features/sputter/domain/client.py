"""The result of the client analysis package (pure Python and NumPy): time windows, their solutions and images.

The package analyses one in-situ scan window by window. For every window it writes several solutions (*entries*):
the candidates of its network, ranked, and — when it refined — the refined solution. Each entry holds the quantities
of the main screen (grain radius R, grain height H, centre distance D, the gap between grains, the width of the
log-normal radius distribution, the coverage; a quantity a model family does not have is missing) and its fit
deviation (how far the model image is from the measured one: lower is better). ``pick`` names the entry with the
lowest fit deviation, ``smooth`` the one a smooth evolution over the whole scan selects: the one shown first.
A window the package could not use is ``{t, error}``.

Two kinds of result differ from that. On a porous substrate (``ClientResult.porous``) the entries have no height,
distance, gap or coverage: they hold the share of the metal in resolvable islands, the share in the cap layer, the
vertical and the lateral correlation length of the metal network in the pores and the film share (``POROUS_KEYS``),
and their radius is a relative size. A series of *static* frames (``ClientResult.static``: the package is called
once per sample measured after its deposition) has its samples in the place of the time windows: each has a label,
its nominal thickness — the position on the series' axis, ``ClientResult.x_label`` — and one entry, whose quantities
may come with a range (``ClientEntry.ranges``).

The images of a window are arrays over the exit angle and q_y (``WindowPatterns``): the measured count rate (NaN:
no data) and the model of every entry on the same cells.

What the user chooses is a ``{window key: entry label}`` mapping that holds only the windows where the choice is not
the window's default (``ClientWindow.default_label``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .real_space import SKETCH_KEYS

MAIN_FOLDER = "main"
MAIN_FILE = "main.json"
ANALYSIS_FILE = "analysis.json"
QUANTITIES = (("R_nm", "grain radius R", "nm"), ("H_nm", "grain height H", "nm"), ("D_nm", "centre distance D", "nm"),
              ("gap_nm", "gap between grains", "nm"), ("width", "width of the radius distribution", ""),
              ("coverage", "coverage", ""), ("amount_nm", "metal amount", "nm"),
              ("f_islands", "share of the metal in resolvable islands", ""),
              ("cap_share", "share of the metal in the cap layer", ""),
              ("depth_nm", "vertical correlation length of the metal network", "nm"),
              ("xi_nm", "lateral correlation length of the metal network", "nm"),
              ("film_share", "share of the scattering that leaves through the metal film", ""))
"""The quantities of an entry (key, the package's label, unit), used when a result does not list its own."""
POROUS_KEYS = ("f_islands", "cap_share", "depth_nm", "xi_nm", "film_share")
"""Quantities only a result on a porous substrate has."""
STATIC_KIND = "static"
NO_MAIN = "{folder} holds no result of the analysis (main/main.json)."
BAD_MAIN = "The result in {folder} cannot be read: {reason}"
NO_ARRAYS = "The images of window {window} are not in {folder}."
BAD_ARRAYS = "The images of window {window} cannot be read: {reason}"
NO_SOLUTION = "The analysis gave no solution for this window."
CLIENT_RESULT_TEMPLATES = (NO_MAIN, BAD_MAIN, NO_ARRAYS, BAD_ARRAYS, NO_SOLUTION)


def number(value) -> Optional[float]:
    """A finite float, or None (missing, text, NaN, a boolean)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def window_key(time_s: float) -> str:
    """The name of a window in the choices: its time as the package writes it in file names (``w_12``)."""
    return f"{float(time_s):g}"


@dataclass(frozen=True)
class ClientQuantity:
    key: str
    label: str = ""
    unit: str = ""


@dataclass(frozen=True)
class ClientEntry:
    """One solution of a window: a candidate of the network or the refined one."""

    label: str
    kind: str = "candidate"
    rank: Optional[int] = None
    weight: Optional[float] = None
    """The network's weight of a candidate (None for a refined entry)."""
    fit_deviation: Optional[float] = None
    values: dict = field(default_factory=dict)
    """Quantity key -> value; a quantity the model family does not have is absent."""
    ranges: dict = field(default_factory=dict)
    """Quantity key -> (low, high), where the package gives a range."""
    source: str = ""

    def value(self, key: str) -> Optional[float]:
        return self.values.get(key)

    def range(self, key: str) -> Optional[tuple]:
        return self.ranges.get(key)


@dataclass(frozen=True)
class ClientWindow:
    time_s: float
    since_onset_s: Optional[float] = None
    nominal_nm: Optional[float] = None
    """The nominal deposit at this time (nm), from the deposition rate."""
    image: str = ""
    arrays: str = ""
    entries: tuple = ()
    pick: str = ""
    smooth: str = ""
    path: str = ""
    error: str = ""
    """Why the package has no solution for this window ("" when it has)."""
    label: str = ""
    """The name of the sample (a series of static frames; "" for a time window)."""
    sample: str = ""
    substrate: str = ""
    reference: str = ""
    frames: Optional[int] = None
    """How many frames of the sample were used (static frames)."""

    @property
    def key(self) -> str:
        """The name of the window in the choices and in its files: the sample's label, else the time."""
        return self.label or window_key(self.time_s)

    def labels(self) -> list:
        return [entry.label for entry in self.entries]

    def entry(self, label: str) -> Optional[ClientEntry]:
        return next((entry for entry in self.entries if entry.label == label), None)

    def default_label(self) -> str:
        """The entry shown first: the smooth evolution's, else the lowest fit deviation, else the first."""
        labels = self.labels()
        for label in (self.smooth, self.pick):
            if label in labels:
                return label
        ranked = [entry for entry in self.entries if entry.fit_deviation is not None]
        if ranked:
            return min(ranked, key=lambda entry: entry.fit_deviation).label
        return labels[0] if labels else ""

    def chosen_label(self, choices: Optional[dict] = None) -> str:
        """The user's choice for this window when it names one of its entries, else the default."""
        label = (choices or {}).get(self.key)
        return label if label in self.labels() else self.default_label()

    def chosen(self, choices: Optional[dict] = None) -> Optional[ClientEntry]:
        return self.entry(self.chosen_label(choices))


@dataclass(frozen=True)
class ClientResult:
    folder: Path
    """The output folder of the analysis (it holds ``analysis.json`` and ``main/``)."""
    status: str = ""
    scan: str = ""
    family: str = ""
    metal: str = ""
    substrate: str = ""
    mode: str = ""
    note: str = ""
    wanted: tuple = ()
    windows: tuple = ()
    files: dict = field(default_factory=dict)
    steps: tuple = ()
    kind: str = ""
    """``static``: a series of samples measured after their deposition ("" for an in-situ scan)."""
    x_label: str = ""
    """What the series' axis is, as the package says it ("" for an in-situ scan: the time)."""
    x_unit: str = ""

    @property
    def static(self) -> bool:
        return self.kind == STATIC_KIND

    @property
    def porous(self) -> bool:
        """The entries hold the quantities of a porous substrate (their radius is a relative size)."""
        return any(key in entry.values for window in self.windows for entry in window.entries for key in POROUS_KEYS)

    def position(self, window: ClientWindow) -> float:
        """Where a window is on the result's axis: the nominal thickness of a static sample, else the time."""
        if self.static and window.nominal_nm is not None:
            return window.nominal_nm
        return window.time_s

    def file(self, key: str) -> Optional[Path]:
        """A result file the package names (``report_md``, ``report_png`` …) when it is there."""
        value = self.files.get(key)
        path = Path(value) if isinstance(value, str) and value else None
        return path if path is not None and path.is_file() else None

    def series(self, key: str, choices: Optional[dict] = None) -> tuple:
        """(positions, values) of a quantity over the windows, for the chosen entry of each (NaN where it is
        missing)."""
        positions, values = [], []
        for window in self.windows:
            entry = window.chosen(choices)
            value = entry.value(key) if entry is not None else None
            positions.append(self.position(window))
            values.append(math.nan if value is None else value)
        return positions, values

    def spread(self, key: str, choices: Optional[dict] = None) -> tuple:
        """(lows, highs) of the range of a quantity over the windows (NaN where the package gives none)."""
        lows, highs = [], []
        for window in self.windows:
            entry = window.chosen(choices)
            found = entry.range(key) if entry is not None else None
            lows.append(math.nan if found is None else found[0])
            highs.append(math.nan if found is None else found[1])
        return lows, highs

    def nearest(self, position: float) -> Optional[int]:
        """The index of the window nearest on the result's axis (None without windows)."""
        if not self.windows:
            return None
        return min(range(len(self.windows)), key=lambda index: abs(self.position(self.windows[index]) - position))


@dataclass(frozen=True)
class WindowPatterns:
    """The images of one window on the cells (exit angle x q_y): ``measured`` and one ``models[i]`` per label."""

    q: np.ndarray
    af: np.ndarray
    measured: np.ndarray
    models: np.ndarray
    labels: tuple

    def model(self, label: str) -> Optional[np.ndarray]:
        return self.models[self.labels.index(label)] if label in self.labels else None


def _quantities(listed) -> tuple:
    found = []
    for item in listed if isinstance(listed, list) else ():
        if isinstance(item, dict) and isinstance(item.get("key"), str) and item["key"]:
            found.append(ClientQuantity(item["key"], str(item.get("label") or ""), str(item.get("unit") or "")))
    return tuple(found) or tuple(ClientQuantity(*item) for item in QUANTITIES)


def _entry(record: dict, keys) -> Optional[ClientEntry]:
    label = record.get("label")
    if not isinstance(label, str) or not label:
        return None
    values = {key: number(record.get(key)) for key in keys}
    rank = number(record.get("rank"))
    ranges = {}
    listed = record.get("ranges") if isinstance(record.get("ranges"), dict) else {}
    for key, pair in listed.items():
        if key in keys and isinstance(pair, (list, tuple)) and len(pair) == 2:
            low, high = number(pair[0]), number(pair[1])
            if low is not None and high is not None and low <= high:
                ranges[key] = (low, high)
    return ClientEntry(label, str(record.get("kind") or "candidate"), None if rank is None else int(rank),
                       number(record.get("weight")), number(record.get("fit_deviation")),
                       {key: value for key, value in values.items() if value is not None}, ranges,
                       str(record.get("source") or ""))


def _file_name(value) -> str:
    """A plain file name of the result's ``main`` folder ("" for anything else: a path is never followed)."""
    name = str(value) if isinstance(value, str) else ""
    return name if name and Path(name).name == name and name not in (".", "..") else ""


def _window(record, keys) -> Optional[ClientWindow]:
    if not isinstance(record, dict):
        return None
    time_s = number(record.get("t"))
    if time_s is None:
        return None
    label = _file_name(record.get("label"))  # the sample of a static series: it names the files of the window
    if record.get("error"):
        return ClientWindow(time_s, error=str(record["error"]), label=label)
    entries = [entry for entry in (_entry(item, keys) for item in record.get("entries") or ()
                                   if isinstance(item, dict)) if entry is not None]
    frames = number(record.get("frames"))
    seen, unique = set(), []
    for entry in entries:  # a label names one entry
        if entry.label not in seen:
            seen.add(entry.label)
            unique.append(entry)
    return ClientWindow(time_s, number(record.get("t_rel")), number(record.get("h_nom")),
                        _file_name(record.get("image")), _file_name(record.get("arrays")), tuple(unique),
                        str(record.get("pick") or ""), str(record.get("smooth") or ""), str(record.get("path") or ""),
                        "" if unique else NO_SOLUTION, label, str(record.get("sample") or ""),
                        str(record.get("substrate") or ""), str(record.get("reference") or ""),
                        None if frames is None else int(frames))


def client_result(folder: Path, main: dict, analysis: Optional[dict] = None) -> ClientResult:
    """The result described by ``main/main.json`` (and ``analysis.json`` when it is there). Raises ``ValueError``
    when ``main`` is not such a description."""
    if not isinstance(main, dict) or not isinstance(main.get("windows"), list):
        raise ValueError(BAD_MAIN.format(folder=folder, reason="no list of windows"))
    analysis = analysis if isinstance(analysis, dict) else {}
    wanted = _quantities(main.get("wanted"))
    keys = [quantity.key for quantity in wanted]
    keys += [key for key in SKETCH_KEYS if key not in keys]  # not shown on a tile: for the real-space picture
    windows = sorted((window for window in (_window(record, keys) for record in main["windows"])
                      if window is not None), key=lambda window: window.time_s)
    files = analysis.get("files") if isinstance(analysis.get("files"), dict) else {}
    steps = analysis.get("steps") if isinstance(analysis.get("steps"), list) else []
    axis = main.get("x") if isinstance(main.get("x"), dict) else {}
    seen, unique = set(), []
    for window in windows:  # a key names one window (two samples of a series never share a label)
        if window.key not in seen:
            seen.add(window.key)
            unique.append(window)
    return ClientResult(
        Path(folder), str(analysis.get("status") or ""), str(main.get("scan") or analysis.get("label") or ""),
        str(main.get("family") or analysis.get("family") or ""), str(main.get("metal") or analysis.get("metal") or ""),
        str(analysis.get("substrate") or ""), str(analysis.get("mode") or ""),
        str(main.get("note") or analysis.get("note") or ""), wanted, tuple(unique),
        {str(key): value for key, value in files.items()}, tuple(step for step in steps if isinstance(step, dict)),
        str(main.get("kind") or analysis.get("kind") or ""), str(axis.get("label") or ""), str(axis.get("unit") or ""))


def window_patterns(window: ClientWindow, arrays) -> WindowPatterns:
    """The images of ``window`` from the arrays of its file (a mapping with ``q``, ``af``, ``measured``, ``models``
    and ``labels``). Raises ``ValueError`` when the arrays do not fit together."""
    try:
        q = np.asarray(arrays["q"], dtype=float).ravel()
        af = np.asarray(arrays["af"], dtype=float).ravel()
        measured = np.asarray(arrays["measured"], dtype=float)
        models = np.asarray(arrays["models"], dtype=float)
        labels = tuple(str(label) for label in np.asarray(arrays["labels"]).ravel())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(BAD_ARRAYS.format(window=window.key, reason=f"{type(exc).__name__}: {exc}")) from exc
    if models.ndim == 2:
        models = models[None]
    shape = (af.size, q.size)
    if q.size < 2 or af.size < 2 or measured.shape != shape or models.shape[1:] != shape \
            or models.shape[0] != len(labels):
        raise ValueError(BAD_ARRAYS.format(window=window.key, reason="the arrays do not have one shape"))
    return WindowPatterns(q, af, measured, models, labels)


__all__ = ["ANALYSIS_FILE", "BAD_ARRAYS", "BAD_MAIN", "CLIENT_RESULT_TEMPLATES", "ClientEntry", "ClientQuantity",
           "ClientResult", "ClientWindow", "MAIN_FILE", "MAIN_FOLDER", "NO_ARRAYS", "NO_MAIN", "NO_SOLUTION",
           "POROUS_KEYS", "QUANTITIES", "STATIC_KIND", "WindowPatterns", "client_result", "window_key",
           "window_patterns"]
