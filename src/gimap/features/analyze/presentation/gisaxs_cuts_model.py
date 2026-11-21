"""The person's GISAXS cuts in the Analyze set-up (``AnalyzeState.gisaxs.regions``), kept in q.

Mixed into ``AnalyzeViewModel``; the table, the editor and the drawing are ``bindings/gisaxs_cuts.py`` and
``bindings/gisaxs_cut_display.py``. ``cut_home`` remembers the set-up (detector and geometry, ``CutSetup``) the cuts
were made on or last kept for, so a frame of another set-up asks before they are used on it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..application import gisaxs_cut_tools


class GisaxsCutsModelMixin:
    """Needs ``state`` (``gisaxs``, ``giwaxs``)."""

    cut_home: Optional[tuple] = None
    """``(path, CutSetup)`` of the frame the cuts were last made or kept on (``None``: not yet seen)."""

    @property
    def gisaxs_cuts(self) -> tuple:
        return tuple(self.state.gisaxs.regions)

    def set_gisaxs_cuts(self, cuts) -> None:
        self.state.gisaxs = replace(self.state.gisaxs, regions=tuple(cuts))

    def add_gisaxs_cut(self, cut) -> int:
        """Append a cut; returns its index."""
        self.set_gisaxs_cuts((*self.state.gisaxs.regions, cut))
        return len(self.state.gisaxs.regions) - 1

    def update_gisaxs_cut(self, index: int, cut) -> None:
        cuts = list(self.state.gisaxs.regions)
        cuts[int(index)] = cut
        self.set_gisaxs_cuts(cuts)

    def remove_gisaxs_cut(self, index: int) -> None:
        cuts = list(self.state.gisaxs.regions)
        del cuts[int(index)]
        self.set_gisaxs_cuts(cuts)

    def person_cut_count(self) -> int:
        """The cuts the person defined, GISAXS and GIWAXS (the ones a frame of another set-up asks about)."""
        return len(self.state.gisaxs.regions) + len(self.state.giwaxs.regions)

    def drop_person_cuts(self) -> None:
        """Only the automatic cuts stay (GISAXS cuts and GIWAXS regions removed)."""
        self.set_gisaxs_cuts(())
        self.state.giwaxs = replace(self.state.giwaxs, regions=())

    def cut_setup_changes(self, analysis) -> list[str]:
        """What ``analysis`` has different from the set-up the cuts were made on (``[]``: nothing to ask). A frame of
        the same file, or any frame while there are no cuts of the person's, becomes the new home."""
        setup = gisaxs_cut_tools.cut_setup(analysis)
        home = self.cut_home
        if setup is None:
            return []
        if home is None or not self.person_cut_count() or home[0] == analysis.path:
            self.cut_home = (analysis.path, setup)
            return []
        changes = gisaxs_cut_tools.setup_changes(home[1], setup)
        if not changes:
            self.cut_home = (analysis.path, setup)
        return changes

    def keep_cuts_for(self, analysis) -> None:
        """The person keeps the cuts for this frame's set-up."""
        setup = gisaxs_cut_tools.cut_setup(analysis)
        if setup is not None:
            self.cut_home = (analysis.path, setup)


__all__ = ["GisaxsCutsModelMixin"]
