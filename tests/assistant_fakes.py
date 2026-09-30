"""Scripted Claude turns and an in-memory Analyze workbench for the assistant tests."""

from __future__ import annotations

import copy
import itertools
from typing import Callable, Optional, Union

import numpy as np

from src.gimap.features.assistant.application import CurveData, LlmTurn, LlmUsage, ToolCall

# Synthetic GIWAXS film: lamellar (100)/(200) out of plane, an isotropic ring, π–π in plane.
PEAKS = (
    # q, FWHM, radial height, in-plane height, out-of-plane height
    (0.40, 0.030, 300.0, 20.0, 900.0),
    (0.80, 0.035, 60.0, 5.0, 180.0),
    (1.20, 0.040, 100.0, 100.0, 100.0),
    (1.65, 0.080, 120.0, 300.0, 10.0),
)
Q = np.linspace(0.1, 2.2, 700)
_ids = itertools.count(1)


def _gauss(x, center, fwhm):
    return np.exp(-4.0 * np.log(2.0) * (x - center) ** 2 / fwhm**2)


def _curve(key: str, x: np.ndarray, clean: np.ndarray, *, pixels: float, label: str, seed: int) -> CurveData:
    rng = np.random.default_rng(seed)
    counts = np.full_like(x, pixels)
    sigma = np.sqrt(clean / counts)
    y = clean + rng.normal(0.0, sigma)
    return CurveData(key, key.replace("_", " "), x, y, sigma, counts, label)


def q_curve(key: str, column: int, *, pixels: float = 400.0, seed: int = 1) -> CurveData:
    clean = 40.0 + 400.0 * np.exp(-Q / 0.25)
    for peak in PEAKS:
        clean = clean + peak[column] * _gauss(Q, peak[0], peak[1])
    return _curve(key, Q, clean, pixels=pixels, label="q (Å⁻¹)", seed=seed)


def chi_curve(q_center: float, *, seed: int = 4) -> CurveData:
    chi = np.arange(-89.5, 90.0, 1.0)
    if abs(q_center - 0.40) < 0.05:  # out of plane: around the surface normal
        clean = 60.0 + 900.0 * _gauss(chi, 0.0, 25.0)
    elif abs(q_center - 1.65) < 0.1:  # in plane
        clean = 60.0 + 300.0 * (_gauss(chi, -90.0, 30.0) + _gauss(chi, 90.0, 30.0))
    else:
        clean = np.full_like(chi, 160.0)
    return _curve("azimuthal", chi, clean, pixels=80.0, label="χ (°)", seed=seed)


class FakeWorkbench:
    """The AnalysisWorkbench port without Qt: records every action."""

    def __init__(self, *, measurement: Optional[str] = "giwaxs"):
        self.measurement = measurement
        self.calls: list[tuple] = []
        self.chi_window = (0.38, 0.42)
        self.exports: list[str] = []
        self.valid_range = (None, None)

    def status(self) -> dict:
        curves = [{"key": key, "points": len(Q), "x_range": [0.1, 2.2]} for key in ("radial", "in_plane", "out_of_plane")]
        curves.append({"key": "azimuthal", "points": 180, "x_range": [-89.5, 89.5]})
        return {
            "file": "synthetic.tif",
            "path": "C:/data/synthetic.tif",
            "measurement": self.measurement,
            "geometry": {"wavelength_A": 1.0, "incidence_deg": 0.2},
            "giwaxs": {
                "in_plane_half_width_deg": 10.0,
                "out_of_plane_half_width_deg": 10.0,
                "chi_q_window": list(self.chi_window),
            },
            "curves": curves,
        }

    def _record(self, *call) -> dict:
        self.calls.append(call)
        return self.status()

    def set_mode(self, mode):
        self.measurement = "giwaxs" if mode in ("giwaxs", "auto") else "gisaxs"
        return self._record("set_mode", mode)

    def set_incidence(self, degrees):
        return self._record("set_incidence", degrees)

    def set_sector_widths(self, in_plane_deg, out_of_plane_deg):
        return self._record("set_sector_widths", in_plane_deg, out_of_plane_deg)

    def set_radial_bins(self, bins):
        return self._record("set_radial_bins", bins)

    def set_custom_sector(self, chi, q_range):
        return self._record("set_custom_sector", chi, q_range)

    def set_q_box(self, q_parallel, qz):
        return self._record("set_q_box", q_parallel, qz)

    def set_chi_window(self, q_low, q_high):
        self.chi_window = (q_low, q_high)
        return self._record("set_chi_window", q_low, q_high)

    def set_valid_range(self, minimum, maximum):
        self.valid_range = (minimum, maximum)
        return self._record("set_valid_range", minimum, maximum)

    def curve(self, key):
        if key == "radial":
            return q_curve("radial", 2, pixels=4000.0, seed=1)
        if key == "in_plane":
            return q_curve("in_plane", 3, seed=2)
        if key == "out_of_plane":
            return q_curve("out_of_plane", 4, seed=3)
        if key == "azimuthal":
            return chi_curve(sum(self.chi_window) / 2.0)
        return None

    def show(self, view=None, lower_profile=None):
        self.calls.append(("show", view, lower_profile))

    def export_curves(self):
        self.exports = ["C:/data/gimap_analysis/synthetic_radial.csv"]
        self.calls.append(("export_curves",))
        return list(self.exports)

    def preview_png(self, max_size=900):
        return b"\x89PNG fake"


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(f"toolu_{next(_ids):03d}", name, arguments)


def turn(*calls: ToolCall, text: str = "", stop: Optional[str] = None, usage: LlmUsage = LlmUsage(100, 20)) -> LlmTurn:
    content = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": item.id, "name": item.name, "input": item.input} for item in calls
    ]
    return LlmTurn(
        stop_reason=stop or ("tool_use" if calls else "end_turn"),
        text=text,
        tool_calls=tuple(calls),
        content=tuple(content),
        usage=usage,
        model="fake-claude",
    )


def report(*items: tuple[str, str], summary: str = "Synthetic film analysed.") -> ToolCall:
    return call(
        "submit_report",
        summary=summary,
        items=[
            {"item": item, "status": status, "findings": f"{item} {status}", "evidence": "tools", "reason": ""}
            for item, status in items
        ],
        caveats=["Scherrer sizes are lower bounds."],
        suggestions=[],
    )


Step = Union[LlmTurn, Exception, Callable[[list], LlmTurn]]


class ScriptedLlm:
    """Plays back prepared turns and keeps every request it received."""

    model = "fake-claude"

    def __init__(self, turns: list[Step]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def respond(self, *, system, tools, messages, cancelled=None, progress=None) -> LlmTurn:
        self.requests.append({"system": system, "tools": tools, "messages": copy.deepcopy(list(messages))})
        if not self.turns:
            raise AssertionError("The script has no more turns")
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        if progress is not None:
            progress("thinking", "Looking at the curves.")
        return item


def tool_results(messages: list) -> list[dict]:
    """The tool_result blocks of the last user message."""
    return [block for block in messages[-1]["content"] if block.get("type") == "tool_result"]


__all__ = [
    "FakeWorkbench",
    "PEAKS",
    "ScriptedLlm",
    "call",
    "chi_curve",
    "q_curve",
    "report",
    "tool_results",
    "turn",
]
