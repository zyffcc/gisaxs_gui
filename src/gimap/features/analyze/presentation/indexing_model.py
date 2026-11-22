"""The phases of the Phases window and the overlay computed from them (one per Analyze page).

It lives on the page, not in the window: the automatic analysis and the command line (``--phase``) give phases
without the window ever opening, and the overlay stays when the window is closed."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence
from weakref import ref

from ..application.indexing import (
    IndexingSettings,
    PhaseOverlay,
    PhaseSpec,
    assign_peaks,
    indexing_record,
    overlays,
    phase_from_cif,
    with_color,
)


class IndexingModel:
    def __init__(self):
        self.phases: list[PhaseSpec] = []
        self.settings = IndexingSettings()
        self._cache: Optional[tuple] = None
        self._assigned = None
        self.last_pole = None
        """The last pole figure fitted (``application.pole_figure.PoleFigure``)."""

    # -- phases --------------------------------------------------------------------------

    def add(self, spec: PhaseSpec) -> int:
        """Add ``spec`` in the next colour of ``PHASE_COLORS``; returns its index."""
        self.phases.append(with_color(spec, len(self.phases)))
        self._cache = None
        return len(self.phases) - 1

    def add_cif(self, path, *, orientation: str = "powder", indices=(0, 0, 1)) -> int:
        return self.add(phase_from_cif(path, orientation=orientation, indices=tuple(indices)))

    def replace(self, index: int, **changes) -> None:
        self.phases[index] = replace(self.phases[index], **changes)
        self._cache = None

    def remove(self, index: int) -> None:
        del self.phases[index]
        self._cache = None

    def clear(self) -> None:
        self.phases.clear()
        self._cache = None

    def set_settings(self, **changes) -> None:
        self.settings = replace(self.settings, **changes)
        self._cache = None

    @property
    def active(self) -> bool:
        return any(phase.visible for phase in self.phases)

    # -- results -------------------------------------------------------------------------

    def overlays(self, geometry) -> tuple[PhaseOverlay, ...]:
        """The overlay of every phase for ``geometry`` (cached while nothing changes); errors of one phase are
        raised (a q_max too large for its cell)."""
        key = (tuple(self.phases), self.settings, geometry)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, overlays(self.phases, self.settings, geometry))
        return self._cache[1]

    def visible_overlays(self, geometry) -> tuple[PhaseOverlay, ...]:
        return tuple(overlay for overlay in self.overlays(geometry) if overlay.spec.visible)

    def assign(self, peaks: Sequence[dict], geometry=None, *, transmission=False) -> dict:
        return assign_peaks(peaks, self.phases, self.settings.q_max, geometry=geometry, transmission=transmission)

    def remember_assignment(self, analysis, assignment, orientations=()):
        if analysis is not None:
            self._assigned = (ref(analysis), tuple(self.phases), assignment, orientations)

    def cell_inputs(self, analysis):
        if self._assigned and self._assigned[0]() is analysis and self._assigned[1] == tuple(self.phases):
            return self._assigned[2:]
        return None, ()

    def record(self, geometry) -> dict:
        return indexing_record(self.phases, self.settings, self.overlays(geometry))


__all__ = ["IndexingModel"]
