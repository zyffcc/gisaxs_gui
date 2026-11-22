"""The Phases window operated for another component (the automatic analysis, the command line, Process with AI).

``set_phases`` replaces the phases of the page's ``IndexingModel`` (CIF files, or cells typed as dictionaries) and
draws the overlay; ``assign_peaks`` assigns rings of I(q) to their reflections; ``pole_figure`` fits the pole
figure of the ring in the I(χ) window. All return plain dictionaries (English, full precision) and must be called
on the GUI thread."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional, Sequence

from ..application import TRANSMISSION
from ..application.indexing import PLANE, POWDER, PhaseSpec, phase_from_cif
from ..application.pole_figure import CHI_MAX, GAUSSIAN, pole_figure


def phase_of(entry, index: int) -> PhaseSpec:
    """A ``PhaseSpec`` from a CIF path or ``{"cif" | "cell", "name", "centring", "orientation", "indices"}``."""
    if isinstance(entry, (str, Path)):
        return phase_from_cif(entry)
    entry = dict(entry)
    orientation = str(entry.get("orientation") or (PLANE if entry.get("indices") else POWDER))
    indices = tuple(int(value) for value in (entry.get("indices") or (0, 0, 1)))
    if entry.get("cif"):
        return phase_from_cif(entry["cif"], orientation=orientation, indices=indices)
    cell = tuple(float(value) for value in entry["cell"])
    if len(cell) == 3:
        cell = (*cell, 90.0, 90.0, 90.0)
    return PhaseSpec(str(entry.get("name") or f"phase {index + 1}"), cell, str(entry.get("centring") or "P"),
                     orientation=orientation, indices=indices)


class IndexingAutomationMixin:
    """Needs ``self._page`` (the Analyze page, with ``indexing`` and ``indexing_changed``)."""

    def set_orientation_min_exit(self, degrees: float, done) -> None:
        page = self._page
        options = page.view_model.state.corrections
        page.view_model.set_orientation_options(options.lorentz_band_px, degrees, options.orientation_polarization)
        page._sync_normalisation_widgets()
        self.reanalyse(done)

    def set_phases(self, phases: Sequence) -> dict:
        """Replace the phases; returns ``{"phases": [records]}`` (ValueError for a file or cell that cannot be read)."""
        page = self._page
        specs = [phase_of(entry, index) for index, entry in enumerate(phases or ())]
        page.indexing.clear()
        for spec in specs:
            page.indexing.add(spec)
        page.indexing_changed()
        return self.indexing_phases()

    def indexing_phases(self) -> dict:
        return {"phases": [spec.record() for spec in self._page.indexing.phases if spec.visible]}

    def assign_peaks(self, peaks: Sequence[dict]) -> dict:
        """``{"phases", "rule", "assignments"}`` for peaks ``{"q", "q_err", "fwhm"}`` (no phases: empty lists)."""
        analysis = self._page.view_model.state.analysis
        model = self._page.indexing
        assignment = model.assign(list(peaks), geometry=getattr(analysis, "geometry", None),
                                  transmission=getattr(analysis, "kind", None) == TRANSMISSION)
        model.remember_assignment(analysis, assignment)
        return assignment

    def pole_figure(self, background: Optional[float] = None, model: str = GAUSSIAN,
                    chi0_deg: Optional[float] = None, chi_max: float = CHI_MAX) -> dict:
        """The pole figure of the ring in the I(χ) window and its fit, as a record."""
        return self.store_pole(self.prepare_pole(background, model, chi0_deg, chi_max)())

    def prepare_pole(self, background=None, model=GAUSSIAN, chi0_deg=None, chi_max=CHI_MAX):
        """Snapshot the selected ring on the GUI thread. The fit can run on the worker."""
        return partial(pole_figure, self._page.view_model.state.analysis, model=model, chi0_deg=chi0_deg,
                       background=background, chi_max=chi_max)

    def prepare_crystal_orientation(self, assignment, check_cancel):
        """Capture the displayed frame and its phases before the fit runs on the worker."""
        from ..application.crystal_orientation import frame_crystal_orientation

        model = self._page.view_model
        specs = tuple(spec for spec in self._page.indexing.phases if spec.visible)
        if [spec.record() for spec in specs] != assignment["phases"]:
            raise ValueError("The visible phases changed after peak assignment.")
        return partial(frame_crystal_orientation, model.state.analysis, specs,
                       frame_source=model.frames, assignment=assignment, check_cancel=check_cancel)

    def prepare_cell_refinement(self, assignment, orientations, max_peaks, check_cancel):
        from ..application.cell_refinement import frame_cell_refinement

        page = self._page
        page.indexing.remember_assignment(page.view_model.state.analysis, assignment, orientations)
        return partial(frame_cell_refinement, page.view_model.state.analysis, tuple(page.indexing.phases),
                       page.indexing.settings, assignment=assignment, orientations=orientations,
                       max_peaks=max_peaks, check_cancel=check_cancel)

    def store_pole(self, result) -> dict:
        self._page.indexing.last_pole = result
        return result.record()


__all__ = ["IndexingAutomationMixin", "phase_of"]
