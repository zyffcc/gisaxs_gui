"""A GIMaP project: what is open in Analyze and Fitting, in one file, to reopen a sample as it was left.

``<name>.gimap`` is JSON: the frames listed in Analyze with its whole set-up (mode, geometry, αi,
masks and corrections, cuts), Single analysis (the curve, its halves, fitting range, left-out
points, the model and the method), In-situ series (the folder of curves, the frames, how each
starts) and Compare (its settings and series: curve files by path, maps from Analyze in
``<name>.compare.npz`` next to the project). Paths are kept as they are; what is no longer there is
reported when the project opens.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_FORMAT = "gimap-project"
PROJECT_VERSION = 1
PROJECT_SUFFIX = ".gimap"
PROJECT_FILTER = "GIMaP project (*.gimap);;All files (*)"


def collect(components, page: str = "", path=None) -> dict:
    """The project of the open window (``path``: where it is saved, for Compare's maps next to it)."""
    workspace = components.fitting_workspace
    data = {
        "format": PROJECT_FORMAT, "version": PROJECT_VERSION, "saved": datetime.now().isoformat(timespec="seconds"),
        "page": page,
        "analyze": components.analyze_page.project_state(),
        "fitting": {"single": workspace.fit_page.project_state(), "series": workspace.series_page.project_state()},
    }
    compare = getattr(components, "compare_page", None)
    if compare is not None:
        data["compare"] = compare.project_state(path)
    return data


def save(components, path, page: str = "") -> Path:
    path = Path(path)
    if path.suffix.lower() != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    path.write_text(json.dumps(collect(components, page, path), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != PROJECT_FORMAT:
        raise ValueError("not a GIMaP project")
    if int(data.get("version", 0)) > PROJECT_VERSION:
        raise ValueError("made by a newer GIMaP")
    return data


def apply(components, data: dict, path=None) -> list[str]:
    """Open a project in the window; returns notes on what could not be restored."""
    notes: list[str] = []
    workspace = components.fitting_workspace
    notes += components.analyze_page.apply_project_state(data.get("analyze") or {})
    fitting = data.get("fitting") or {}
    notes += workspace.fit_page.apply_project_state(fitting.get("single") or {})
    notes += workspace.series_page.apply_project_state(fitting.get("series") or {})
    compare = getattr(components, "compare_page", None)
    if compare is not None and data.get("compare"):
        notes += compare.apply_project_state(data["compare"], path)
    return notes


__all__ = ["PROJECT_FILTER", "PROJECT_FORMAT", "PROJECT_SUFFIX", "apply", "collect", "read", "save"]
