"""Detector frames for Fitting ▸ In-situ series, cut as Analyze cuts them (a bridge of the composition root).

Fitting fits curves and Analyze makes them. A series of detector frames — a folder of CBF / TIFF / EDF files, or the
frames of NeXus / HDF5 sources — is listed in In-situ series, and each frame is cut with Analyze's set-up as it is
when the frames are listed (``recipe``): the request Analyze's Batch Export builds (``AnalyzeViewModel.request_for``:
instrument profile, αi, beam centre, masks and corrections, the widths of the cuts), cut where the frame on screen is
cut (``series_cut``, when the pin is on), summed by Analyze's Sum N, without the q map. ``cut`` (thread-safe) analyses
the frame with ``AnalyzeViewModel.analyze`` and gives the curve and observation record Analyze writes into
``<stem>_fit_input.dat`` (``fit_input_curve`` / ``fit_input_observation``), in memory: nothing is written beside the
data. The recipe's ``record`` (Analyze's settings record and where the cut is pinned) goes into the series record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from src.gimap.features.analyze.application.export_record import fit_input_curve, fit_input_observation
from src.gimap.features.analyze.application.models import AnalysisRequest
from src.gimap.features.analyze.application.settings_file import settings_record
from src.gimap.features.analyze.infrastructure.adapters.frames import DetectorIoFrameSource
from src.gimap.features.analyze.presentation.frames_model import with_cut
from src.gimap.features.fitting.application import InSituSourceFrame

NOT_OPEN = "Analyze is not open."
NOT_SET_UP = ("Analyze has no frame set up yet: open one frame of this series in Analyze, set its geometry, masks "
              "and cut, then take Analyze's set-up here.")
NOTHING_TO_FIT = "Nothing to fit: this frame has no usable curve."


@dataclass(frozen=True)
class FrameRecipe:
    """Analyze's set-up for every frame of a series (immutable: safe in the worker threads)."""

    request: AnalysisRequest
    series_cut: Optional[dict]
    halves: str
    sum_count: int
    record: dict


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


class SeriesFrames:
    """``view_model()``: Analyze's view model (``None`` while Analyze is not built)."""

    def __init__(self, view_model: Callable[[], object]):
        self._view_model = view_model
        self._source = DetectorIoFrameSource()

    def recipe(self) -> tuple[Optional[FrameRecipe], str]:
        """Analyze's set-up now (GUI thread), or ``None`` and why not."""
        model = self._view_model()
        if model is None:
            return None, NOT_OPEN
        if model.state.analysis is None or model.current_path is None:
            return None, NOT_SET_UP
        request = replace(model.request_for(Path("."), summed=()), with_map=False)
        choices, _destination, _subfolder = model.batch_preferences()
        halves = str(getattr(choices, "fit_halves", "mean") or "mean")
        cut = model.series_cut()
        settings = model.current_settings()
        sum_count = max(1, int(settings.sum_count or 1))
        record = _plain(dict(
            analyze_settings=settings_record(settings), series_cut=cut, halves=halves,
            sum_count=sum_count, set_up_on=dict(path=model.current_path, frame=int(model.state.frame_index)),
            incidence_deg=getattr(model.state.analysis.geometry, "incidence_deg", None),
            kind=model.state.analysis.kind))
        return FrameRecipe(request, cut, halves, sum_count, record), ""

    def list(self, folder) -> list[tuple[str, tuple]]:
        """``(token, summed frames)`` of every item of the series in ``folder``: every frame of its detector sources
        in natural order (``InSituSourceFrame.token``: the path, ``…::gimap-frame=N`` past the first frame of a
        source), with Analyze's Sum N in consecutive groups (the first frame names the group)."""
        frames = []
        for path in self._source.expand([Path(folder)]):
            try:
                count = self._source.frame_count(path)
            except (OSError, ValueError):
                count = 1  # read when it is cut: the frame then says why it cannot be
            frames.extend((Path(path), index) for index in range(max(1, int(count))))
        return [(InSituSourceFrame(path, index).token, ()) for path, index in frames]

    @staticmethod
    def grouped(items, sum_count: int) -> list[tuple[str, tuple]]:
        """``items`` summed by ``sum_count`` in consecutive groups (a last group may hold fewer frames)."""
        if sum_count <= 1:
            return list(items)
        out = []
        for start in range(0, len(items), sum_count):
            group = [InSituSourceFrame.from_token(token) for token, _summed in items[start:start + sum_count]]
            out.append((group[0].token, tuple((frame.path, frame.frame_index) for frame in group[1:])))
        return out

    def cut(self, recipe: FrameRecipe, token: str, summed=()) -> dict:
        """The fit-input curve of one frame (q in Å⁻¹, I, σ, pixels per point) and its observation record."""
        model = self._view_model()
        if model is None:
            raise RuntimeError(NOT_OPEN)
        frame = InSituSourceFrame.from_token(token)
        request = replace(recipe.request, path=frame.path, frame_index=frame.frame_index,
                          summed_frames=tuple((Path(path), int(index)) for path, index in summed))
        analysis = model.analyze(with_cut(request, recipe.series_cut))
        curve = fit_input_curve(analysis)
        if curve is None:
            raise ValueError(NOTHING_TO_FIT)
        return dict(q=curve.x, intensity=curve.intensity, sigma=curve.sigma, pixels=curve.pixels,
                    observation=fit_input_observation(analysis, recipe.halves))


__all__ = ["FrameRecipe", "NOTHING_TO_FIT", "NOT_OPEN", "NOT_SET_UP", "SeriesFrames"]
