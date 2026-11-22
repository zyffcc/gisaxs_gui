"""An In-situ series in a project: the fits of its frames (each frame's result with its errors, χ²ᵣ, flags, points,
halves and unit of q, or why it failed), the start model and the series' settings — kept with File ▸ Save Project and
shown again once the series' curves are listed (``FitSeriesPage._restore_series_choices``). A frame whose file is no
longer at its place in the listing is not restored; the status line says how many."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf

from ...application.fit_state import (
    STATE_SCHEMA,
    frame_from_dict,
    frame_to_dict,
    settings_from_dict,
    settings_to_dict,
)
from ...application.single_fit import model_from_dict, model_to_dict

RESTORED = "{count} fitted frames of the project are shown again."
RESTORED_SOME = "{count} fitted frames of the project are shown again; {missing} are not, their files are no longer listed."
NOT_READ = "The fits of the series in the project could not be read: {reason}"


def _same_file(listed: str, saved: str) -> bool:
    if listed == saved:
        return True
    try:
        return Path(listed.split("::gimap", 1)[0]).resolve() == Path(saved.split("::gimap", 1)[0]).resolve() \
            and listed.partition("::gimap")[2] == saved.partition("::gimap")[2]
    except (OSError, ValueError):
        return False


class FitSeriesProjectMixin:
    """Needs the series page's ``fits``, ``paths``, ``running``, ``_start_model``, ``_settings``, ``_unit``, ``_curves``,
    ``_fill_trend_choices``, ``_rebuild_results``, ``_draw_trend``, ``refresh``, ``_status``."""

    def series_results_state(self) -> dict | None:
        if self.running or not self.fits or self._start_model is None:
            return None
        return {"schema": STATE_SCHEMA, "start_model": model_to_dict(self._start_model),
                "settings": settings_to_dict(self._settings), "unit": self._unit,
                "frames": [frame_to_dict(frame) for _index, frame in sorted(self.fits.items())]}

    def restore_series_results(self, data) -> None:
        """The fits of a project, once its curves are listed again (module docstring)."""
        if not isinstance(data, dict) or self.running:
            return
        try:
            start = model_from_dict(data["start_model"])
            settings = settings_from_dict(data.get("settings") or {})
            frames = [frame_from_dict(item) for item in data.get("frames") or ()]
        except (ValueError, KeyError, TypeError) as exc:
            self._status(lambda reason=str(exc): trf(NOT_READ, reason=reason), "warning")
            return
        kept = {frame.index: self.with_curve_time(frame) for frame in frames
                if 0 <= frame.index < len(self.paths) and _same_file(str(self.paths[frame.index]), str(frame.path))}
        if not kept:
            return
        self._start_model, self._settings = start, settings
        self._unit = str(data.get("unit") or self._unit)
        self.fits, self._curves = kept, {}
        if any(frame.acquisition for frame in kept.values()):
            self.trend_axis_combo.show()
        self._fill_trend_choices()
        self._rebuild_results()
        self._draw_trend()
        self.refresh()
        missing = len(frames) - len(kept)
        if missing:
            self._status(lambda: trf(RESTORED_SOME, count=len(kept), missing=missing), "warning")
        else:
            self._status(lambda: trf(RESTORED, count=len(kept)))
        self._log(tr(RESTORED).format(count=len(kept)))


__all__ = ["NOT_READ", "RESTORED", "RESTORED_SOME", "FitSeriesProjectMixin"]
