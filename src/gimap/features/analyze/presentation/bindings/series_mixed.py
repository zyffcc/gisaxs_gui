"""Rows of a Series map that are not alike: frames at other incidence angles, frames of other exposure times.

A run that switches between incidence angles gives frames whose cut at the same detector rows is another exit
angle: the map holds one angle (``application/series_angles.py``) and keeps the other rows aside
(``_series_aside``); the αi list beside the sentence above the map shows another angle, or every row. Every redraw
sorts the rows again (``_series_split_angles``), so frames written while watching join the map of their angle: the
split before is kept and the rows that arrived are added to it (``application.series_angles.AngleSplit``).

Rows of different exposure times, with the intensity not divided by the exposure, differ by their exposure alone:
a note under the sentence says so, and its link turns on the exposure from the file header and builds the map again.
"""

from __future__ import annotations

from operator import is_

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.live_rows import row_identity
from ...application.series_angles import ALL, AngleSplit, same_incidence

ANGLE_ITEM = "αi {angle}° · {n} frames"
ALL_ITEM = "All angles · {n} frames"
NO_ANGLE_ITEM = "No αi · {n} frames"
ANGLE_WORDS = "αi {angle}° only ({n} frames at other angles: choose them in the αi list)"
EXPOSURE_NOTE = "These frames have different exposure times ({times} s) and their intensities are not divided by them."
EXPOSURE_LINK = "Divide by the exposure time of the file header"
EXPOSURE_HTML = '{note} <a href="exposure">{link}</a>'


class SeriesMixedMixin:
    """Needs ``series_angle_combo``, ``series_exposure_label``, ``exposure_combo``, ``_series_rows``, ``_series_row``,
    ``_series_built``, ``view_model``, ``build_series_map`` and ``_normalisation_changed``."""

    def _connect_series_mixed(self) -> None:
        self._series_forget_angles()
        self.series_angle_combo.activated.connect(self._series_angle_chosen)
        self.series_exposure_label.linkActivated.connect(lambda _link: self.series_divide_by_exposure())

    def _series_forget_angles(self) -> None:
        """A new map: no rows aside, its angle chosen afresh."""
        self._series_aside: list = []
        self._series_angle_choice = None
        """``None``: the angle of most rows (``default_incidence``); else ``ALL`` or the angle chosen in the list."""
        self._series_angle_shown = ALL
        self._series_angles_kept = None
        self.series_angle_combo.hide()
        self.series_exposure_label.hide()

    # -- one angle per map ---------------------------------------------------------------------------------------

    def _series_split_angles(self) -> None:
        """Before the map is drawn: the rows of its angle in ``_series_rows`` (the row picked stays picked; one at
        another angle gives way to the last row of the map), the others in ``_series_aside``."""
        rows = self._series_rows
        picked = row_identity(rows[self._series_row]) if 0 <= self._series_row < len(rows) else None
        kept = getattr(self, "_series_angles_kept", None)
        if kept is None:
            kept = self._series_angles_kept = AngleSplit()
        split = kept.split(rows, getattr(self, "_series_aside", []), self._series_built.get("numbers"),
                           self._series_angle_choice)
        self._series_rows, self._series_aside, self._series_angle_shown = split.kept, split.aside, split.angle
        self._series_row = split.places.get(picked, max(0, len(split.kept) - 1))
        self._series_built["incidence"] = split.line
        self._series_fill_angles(split.groups, split.rows)

    def _series_fill_angles(self, groups, rows: int) -> None:
        """The αi list: ``groups`` (``incidence_groups`` of every row), ``rows`` of all angles."""
        combo = self.series_angle_combo
        with QSignalBlocker(combo):
            combo.clear()
            for angle, count in groups:
                text = (trf(NO_ANGLE_ITEM, n=count) if angle is None
                        else trf(ANGLE_ITEM, angle=f"{angle:.3g}", n=count))
                combo.addItem(text, angle)
            combo.addItem(trf(ALL_ITEM, n=rows), ALL)
            shown = self._series_angle_shown
            index = next((i for i in range(combo.count()) if (combo.itemData(i) == ALL) == (shown == ALL)
                          and (shown == ALL or same_incidence(combo.itemData(i), shown))), combo.count() - 1)
            combo.setCurrentIndex(index)
        combo.setVisible(len(groups) > 1)

    def _series_angle_chosen(self, index: int) -> None:
        """Another angle in the αi list: its map (the stages and traces of its rows)."""
        self._series_angle_choice = self.series_angle_combo.itemData(index)
        if self._series_map is None:
            return
        self._series_row = -1  # (the last row of the angle's map)
        self._clear_stages()
        self._show_series_map(keep_view=False, keep_q=True)
        self._find_stages()

    def _series_angle_words(self) -> str:
        """Said in the sentence above a map of one angle."""
        angle, aside = self._series_angle_shown, getattr(self, "_series_aside", [])
        if angle == ALL or angle is None or not aside:
            return ""
        return trf(ANGLE_WORDS, angle=f"{angle:.3g}", n=len(aside))

    # -- exposure times ------------------------------------------------------------------------------------------

    def _series_check_exposures(self) -> None:
        """A note when the rows were measured with different exposure times and the intensity is not divided by
        them (the exposure of the normalisation is off)."""
        label = self.series_exposure_label
        times = sorted(self._series_exposures())
        exposure = self.view_model.state.corrections.normalisation.exposure
        if len(times) < 2 or exposure != "off":
            label.hide()
            return
        listed = ", ".join(f"{value:g}" for value in times[:4]) + (" …" if len(times) > 4 else "")
        label.setText(trf(EXPOSURE_HTML, note=trf(EXPOSURE_NOTE, times=listed), link=tr(EXPOSURE_LINK)))
        label.show()

    def _series_exposures(self) -> set:
        """The exposure times of the rows; those of the rows checked before kept, while they are the same rows (a live
        map checks at every drawing: 40 ms of the window at 20 000 rows)."""
        rows = self._series_rows
        kept = getattr(self, "_series_exposures_kept", None)
        if kept is not None and len(rows) >= len(kept[0]) and all(map(is_, rows, kept[0])):
            seen, times = kept[0], set(kept[1])
        else:
            seen, times = [], set()
        times |= {round(float(row[3].exposure_s), 6) for row in rows[len(seen):] if getattr(row[3], "exposure_s", None)}
        self._series_exposures_kept = (list(rows), frozenset(times))
        return times

    def series_divide_by_exposure(self) -> None:
        """Mask & corrections ▸ Normalisation ▸ Exposure: from the file header, and the map built again with it."""
        index = self.exposure_combo.findData("header")
        if index < 0:
            return
        self.exposure_combo.setCurrentIndex(index)
        self._normalisation_changed()
        self.series_exposure_label.hide()
        if self._series_map is not None and not self.batch_running():
            self.build_series_map()


__all__ = ["SeriesMixedMixin"]
