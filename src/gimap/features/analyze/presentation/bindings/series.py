"""Behaviour of the Series tab: which frames a map takes, building it, and opening its rows.

Build Map reduces every frame (or group of summed frames) with the current settings through the
batch runner (``bindings/batch_run.py``) without writing files: several frames at once when the
series is long, the batch panel with progress, Pause and Stop, and the map redrawn as rows arrive.
A Batch Export fills the same map live.

A map is one series seen one way, and says so: every frame is cut where the frame on screen is
(``view_model.series_cut``; “Find the cut again in every frame” in the options turns that off), a
Sum N that Run Automatic Analysis set for its own result is switched off first, and what the map was
built with is kept with it (``_series_built``). The picture, the frame axis and the title line are in
``series_display.py``, picking rows and x windows in ``series_pick.py``, saving in ``series_exports.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ...application import GISAXS, BatchChoices, SeriesMap
from ..series_texts import (
    CUT_OF_SERIES, HOW_TO, HOW_TO_SUMS, LISTED_FILES, LISTED_FRAMES, LISTED_SUMS, LISTED_SUMS_SHORT, MAP_DONE_SUMS,
    MAP_DONE_SUMS_FAILED, NEEDS_GEOMETRY, NO_FRAMES_IN_RANGE, SUM_SWITCHED_OFF, cut_sentence, map_group, map_title,
    range_words,
)
from ..texts import curve_title
from ..views.series_view import (
    SERIES_EMPTY_ACTION, SERIES_EMPTY_TEXT, SERIES_EMPTY_TITLE, SERIES_WAITING_ACTION, SERIES_WAITING_TEXT,
    SERIES_WAITING_TITLE,
)
from .series_display import SeriesDisplayMixin
from .series_exports import SeriesExportsMixin
from .series_pick import SeriesPickMixin
from .series_fit_results import SeriesFitResultsMixin

DEFAULT_CURVE = {GISAXS: "horizontal"}
"""The curve a series map starts with; GIWAXS and others: the radial I(q)."""
REDRAW_EVERY = 5
"""Rows added between redraws of the growing map."""


class SeriesMixin(SeriesDisplayMixin, SeriesPickMixin, SeriesExportsMixin, SeriesFitResultsMixin):
    """Needs the Series widgets (``views/series_view.py``), ``view_model``, ``tasks`` and ``_status``."""

    def _connect_series(self) -> None:
        self._series_map: Optional[SeriesMap] = None
        self._series_name: Optional[str] = None
        """The map's own name (from the files of its rows), kept when the file list changes."""
        self._series_rows: list = []
        self._series_failures: list[str] = []
        self._series_total = 0
        self._series_row = 0
        self._series_note = ""
        """Said with the line of a finished map (a Sum N of the automatic analysis that was switched off)."""
        self._series_done_text = None
        """``(text shown, how to say it again)`` of the status line of the map built last (a language switch)."""
        # Every way files are listed (opened, dropped, folder watching) tells a map of the earlier list.
        self.file_list.model().rowsInserted.connect(lambda *_args: self._series_list_grew())
        self.series_build_button.clicked.connect(self.build_series_map)
        self.series_open_button.clicked.connect(lambda: self.open_series_row())
        self.series_map_view.horizontalBandChanged.connect(lambda *_args: self._batch_unfollow())
        self.series_watch_follow_check.toggled.connect(self._watch_series_follow_toggled)
        self.series_empty.actionRequested.connect(self._series_card_action)
        self.series_each_cut_check.toggled.connect(self._series_each_cut_toggled)
        for spin in (self.series_first_spin, self.series_last_spin):
            spin.valueChanged.connect(lambda _value: self._series_range_changed())
        self.right_tabs.currentIndexChanged.connect(lambda _index: self._series_show_info())  # (the cut may be another)
        self._connect_series_display()
        self._connect_series_pick()
        self._connect_series_exports()
        self._connect_series_fit_results()

    # -- curves offered ----------------------------------------------------------------

    def _refresh_series_curves(self, analysis) -> None:
        """The curves of the frame on screen, the mode's usual one first chosen."""
        reduction = analysis.reduction if analysis is not None else None
        curves = [curve for curve in (reduction.curves if reduction is not None else ()) if not curve.is_empty]
        current = self.series_curve_combo.currentData()
        with QSignalBlocker(self.series_curve_combo):
            self.series_curve_combo.clear()
            for curve in curves:
                self.series_curve_combo.addItem(curve_title(curve.title), curve.key)
            keys = [curve.key for curve in curves]
            wanted = current if current in keys else DEFAULT_CURVE.get(getattr(analysis, "kind", None), "radial")
            if wanted in keys:
                self.series_curve_combo.setCurrentIndex(keys.index(wanted))
        popup = self.series_curve_combo.view()  # the combo may be narrower than a name: its list shows them whole
        popup.setMinimumWidth(popup.sizeHintForColumn(0) + 2 * popup.frameWidth() + 24 if curves else 0)
        files = len(self.view_model.state.files)
        frames = max(0, int(self.view_model.listed_frames()))
        several = frames > 1
        ready = bool(curves) and several
        self._series_ready = ready
        self.series_controls.setVisible(several)  # one frame: only the explanation
        self.series_options.setVisible(several)
        if files:  # (an empty list would hold a range set before at “1 to 1” for the next one)
            listed = max(1, int(self.view_model.total_frame_count()))
            self.series_first_spin.setMaximum(listed)
            self.series_last_spin.setMaximum(listed)
        # The card shown until there is a map never says “One frame” while several are listed; its button
        # does what the card asks for: the frames of a folder, or the map.
        card = ((SERIES_WAITING_TITLE, SERIES_WAITING_TEXT, SERIES_WAITING_ACTION if ready else "") if several
                else (SERIES_EMPTY_TITLE, SERIES_EMPTY_TEXT, SERIES_EMPTY_ACTION))
        self.series_empty.set_content(tr(card[0]), tr(card[1]), tr(card[2]) if card[2] else "")
        self.series_build_button.setEnabled(ready and not self._series_queue and not self.batch_running())
        self.series_empty.action_button.setEnabled(not several or self.series_build_button.isEnabled())
        if self._series_map is None and not getattr(self, "_batch_live", False):
            if not curves:  # (no frame shown: nothing to say of it)
                self._series_say(NEEDS_GEOMETRY if analysis is not None else "")
            elif ready:
                self._series_say_listed(analysis, files, frames)
            else:
                self._series_say("")
        else:
            self._series_show_info()  # (the cut's check box as the setting is now)
        self._series_mark_stale()

    def _series_say_listed(self, analysis, files: int, frames: int) -> None:
        """What Build Map would stack — frames, or sums of N frames (with a last sum of fewer) — and where the
        frames are cut when the curve chosen is a cut."""
        model = self.view_model
        cut = self._series_cut_matters(getattr(analysis, "kind", None), self.series_curve_combo.currentData())
        if model.sum_count > 1 and model.sum_set_by != "automatic":  # (an automatic sum is switched off for a map)
            summary = model.batch_group_summary()
            values = dict(frames=summary["frames"], groups=summary["groups"], n=model.sum_count)
            if summary["short_count"]:
                self._series_say(LISTED_SUMS_SHORT, cut=cut, last=summary["short_last"], **values)
            else:
                self._series_say(LISTED_SUMS, cut=cut, **values)
        elif files > 1:
            self._series_say(LISTED_FILES, cut=cut, n=files)
        else:
            self._series_say(LISTED_FRAMES, cut=cut, n=frames)

    def _series_say(self, template: str, *, cut: bool = False, **values) -> None:
        """The sentence under the Series controls: ``template`` (English) filled with ``values`` in the interface
        language, kept so a switch of the language says it again (``_series_language``). ``cut``: followed by
        where the frames of the series are cut (``view_model.series_cut_text``)."""
        self._series_info = (template, values, bool(cut))
        self.series_info_label.set_one_line(template in (HOW_TO, HOW_TO_SUMS))  # the how-to of a map; a notice: whole
        self._series_show_info()

    def _series_show_info(self) -> None:
        """The sentence — with the frame range Build Map takes when it is not every frame —, the title of the
        folded options (which says that range too), and the check box of the cut as the setting is now (the Batch
        Export dialog has the same check box)."""
        with QSignalBlocker(self.series_each_cut_check):
            self.series_each_cut_check.setChecked(not self.view_model.pin_series_cuts)
        template, values, cut = getattr(self, "_series_info", ("", {}, False))
        text = (trf(template, **values) if values else tr(template)) if template else ""
        where = cut_sentence(self.view_model.series_cut_text()) if cut else ""
        if text and where:
            text += " " + trf(CUT_OF_SERIES, cut=where)
        if template in (LISTED_FILES, LISTED_FRAMES, LISTED_SUMS, LISTED_SUMS_SHORT):  # (a range is never silent)
            limited = range_words(self.series_first_spin.value(), self.series_last_spin.value())
            text += f" {limited}" if limited else ""
        self.series_info_label.setText(text)
        self.series_options.toggle_button.setText(
            range_words(self.series_first_spin.value(), self.series_last_spin.value(), options=True))

    def _series_range_changed(self) -> None:
        """Options ▸ Frames: said under the buttons and in the title of the folded options at once."""
        self._series_show_info()

    def _series_each_cut_toggled(self, each: bool) -> None:
        """Options ▸ “Find the cut again in every frame”: off (the usual way) pins the cut of the series."""
        self.view_model.set_pin_series_cuts(not each)
        self._series_show_info()
        self._series_mark_stale()  # (the pin is part of the settings a map was built with)
        refresh = getattr(self, "refresh_batch_entry", None)  # the Batch Export entry may say which cut it takes
        if refresh is not None:
            refresh()

    def _series_card_action(self) -> None:
        """The card's button: the frames of a folder while one frame is listed, else the map."""
        if getattr(self, "_series_ready", False):
            self.build_series_map()
        else:
            self.open_folder()

    def _series_language(self, analysis) -> None:
        """After a switch of the interface language: the curve names, the sentence under the controls, the map's
        title line, the titles of the frame and trace plots and the line of the finished map again (nothing is
        reduced again)."""
        self._refresh_series_curves(analysis)
        self._series_show_info()
        self._series_view_tips()
        self._series_fit_language()
        series = self._series_map
        if series is None:
            return
        self.series_map_view.title_label.setText(self._series_title(series))
        self._series_draw_frame_axis(series)
        self._series_mark_short_sums(series)
        self._series_describe()
        self._series_pick_row(self._series_row + 0.5, move_band=False)
        self._series_redraw_trace()
        done = self._series_done_text
        if done is not None and self.status_text() == done[0]:  # still the line of this map: in the new language
            text = done[1]()
            self._series_done_text = (text, done[1])
            self._status(text, self.status_level())

    def _series_keep_curve(self, key: str) -> None:
        """The curve a map is about to stack: its key, and its English title from the frame on screen. The map
        is named after that curve from then on, whatever the combo shows later (another choice not built yet,
        a frame of another kind on screen, another language)."""
        self._series_key = key
        analysis = self.view_model.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        curves = reduction.curves if reduction is not None else ()
        self._series_curve_title = next((curve.title for curve in curves if curve.key == key), "")
        # What the run that fills the map is built with (its frames, sum, cut and settings), taken now: the
        # frame on screen, the list and the settings may all be others by the time the map is read or saved.
        run = getattr(self, "_batch_run", None)
        self._series_run = run
        self._series_built = dict(
            self.view_model.series_built_with(), kind=getattr(analysis, "kind", None),
            every=int(getattr(getattr(run, "choices", None), "every", 1) or 1),
            first=int(self.series_first_spin.value()), last=int(self.series_last_spin.value()),
        )
        with self.view_model.only_frames(self._series_built["first"], self._series_built["last"] or None):
            tail = self.view_model.batch_tail()
        self._series_built["watch_tail"] = tuple(tail)
        if self.view_model.watch.active and self.batch_running():
            self._watch_series_align_groups()

    def _series_curve_name(self) -> str:
        """The name of the curve the map holds, in the interface language (the combo's, when it is unknown)."""
        return curve_title(getattr(self, "_series_curve_title", "")) or self.series_curve_combo.currentText()

    def _series_title(self, series) -> str:
        return map_title(self._series_curve_name(), series)

    # -- building ----------------------------------------------------------------------

    @property
    def _series_queue(self) -> list:
        """The frames of a map being built still to start (a map-only batch, see ``bindings/batch_run.py``)."""
        return self._batch if getattr(self, "_batch_map_only", False) else []

    def build_series_map(self) -> None:
        """Reduce every listed frame (every n-th) and stack the chosen curve — through the batch runner, so
        a long series uses several processes, shows its progress and can be paused or stopped."""
        model = self.view_model
        key = self.series_curve_combo.currentData()
        if self.batch_running():
            self._status(tr("A batch is running: its frames appear in this map as they are done."), "warning")
            return
        self._series_note = ""
        if key is not None and model.sum_set_by == "automatic" and model.sum_count > 1:
            # The sum the automatic analysis made its result of is not the person's Sum N for the series.
            switched = model.sum_count
            self.sum_spin.setValue(1)  # (the frame on screen follows: read again as one frame)
            if model.sum_count != 1:
                model.set_sum_count(1)
            self._series_note = trf(SUM_SWITCHED_OFF, n=switched)
        first, last = self.series_first_spin.value(), self.series_last_spin.value()
        with model.only_frames(first, last or None):  # (the batch runner asks for its frames as it starts)
            requests = model.batch_requests()[:: max(1, self.series_step_spin.value())]
            if key is None or len(requests) < 2:
                limited = first > 1 or last
                self._status(trf(NO_FRAMES_IN_RANGE, first=first, last=last or model.listed_frames()) if limited
                             else tr("A series map needs at least two frames with a geometry."), "warning")
                return
            remembered = model.batch_preferences()[0]
            choices = BatchChoices(tables=False, every=self.series_step_spin.value(), speed=remembered.speed)
            self.series_export_button.hide()
            self._clear_stages()
            self.run_batch(Path("."), choices, stem="series", map_only=True, live_key=key)
        if self._series_note:
            self._status(self._series_note, "warning")
            show_toast(self.window(), self._series_note, level="warning")

    def cancel_series_map(self) -> None:
        """Stop after the frames being reduced; the rows so far stay in the map."""
        if getattr(self, "_batch_map_only", False):
            self.cancel_batch()

    def _reset_series(self) -> None:
        """The file list was cleared: no map, the explanation of the empty tab again.

        A map being built is stopped and its late rows are dropped; a running Batch Export goes on
        writing its files, but no longer draws its frames into the map."""
        self._stop_watch_series()
        self._reset_series_fit()
        if getattr(self, "_batch_map_only", False) and self.batch_running():
            self._series_dropped = True  # ``_series_finished`` then says nothing
            self.cancel_series_map()
        self._batch_live = False
        self._series_map, self._series_name = None, None
        self._series_rows = []
        self._series_row = 0
        self._series_done_text = None
        self._clear_stages()
        self._series_mark_short_sums(None)
        if self.series_wide_button.isChecked():
            self.series_wide_button.setChecked(False)  # the image gets its room back
        for widget in (self.series_map_view, self.series_plots, self.series_export_button, self.series_compare_button,
                       self.series_title_label, self.series_stale_label, self.series_wide_button,
                       self.series_view_combo):
            widget.hide()
        if not self.view_model.state.files:  # Clear: a frame range set for the list before ends with it
            for spin, value in ((self.series_first_spin, 1), (self.series_last_spin, 0)):
                with QSignalBlocker(spin):
                    spin.setValue(value)
        self.series_empty.show()
        stretch = self.series_host.itemAt(self.series_stretch_index)
        if stretch is not None and stretch.spacerItem() is not None:
            self.series_host.setStretch(self.series_stretch_index, 1)
        self._series_say("")
        # The tab as the list is now: no file (Clear) — the first explanation again, no controls, no curve of
        # the files cleared to choose; files still listed — what Build Map would stack.
        self._refresh_series_curves(self.view_model.state.analysis)

    def _series_list_grew(self, *, existing: bool = False) -> None:
        """Files were added: a map built before still holds for its own frames, and says so."""
        if (self.view_model.watch.active and not existing and self._series_rows and self._series_built
                and not self._watch_series_corrected()
                and not self.view_model.series_settings_changed(self._series_built)
                and self.series_curve_combo.currentData() == self._series_key):
            return  # the watch appends complete rows; a closed frame range remains its chosen range
        series = self._series_map
        if series is not None and not getattr(self, "_batch_live", False):
            self._series_say("Map of the earlier list ({n} frames) — Build Map again to include the new files", n=series.rows)

    def _series_drop_file(self, path, *, summed_across: bool = False) -> None:
        """A file left the list (not while a batch runs): its rows leave the map, which is drawn again, and its
        stages are found again; fewer than two rows left: no map. With frames summed across files
        (``summed_across``) every later group changes, so the map stays and says it is of the earlier list."""
        series = self._series_map
        if series is None:
            return
        key = str(path).casefold()
        kept = [row for row in self._series_rows if str(row[4][0]).casefold() != key]
        if summed_across:
            self._series_say("Map of the earlier list ({n} frames) — Build Map again without the removed file",
                             n=series.rows)
            return
        if len(kept) == len(self._series_rows):
            return  # none of its frames is in the map (every n-th frame)
        if len(kept) < 2:
            self._reset_series()
            return
        self._series_rows = kept
        self._clear_stages()
        self._show_series_map(keep_view=False, keep_q=True)
        self._find_stages()

    def _series_finished(self, failures) -> None:
        """A map-only batch ended: the whole map, a note of what failed, Export."""
        if getattr(self, "_series_dropped", False):  # the list was cleared while it ran: nothing to show
            self._series_dropped = False
            self._series_rows = []
            self.series_build_button.setEnabled(len(self.view_model.state.files) > 1)
            return
        self.series_build_button.setEnabled(True)
        if not self._series_rows:
            self._status(trf("No frame gave the curve: {reasons}", reasons="; ".join(failures[:3])), "error")
            self._series_say("No frame gave the curve.")
            return
        self._show_series_map(keep_view=False)
        note = self._series_note

        def said() -> str:  # (said again in the interface language after a switch, see ``_series_language``)
            values = dict(n=len(self._series_rows), curve=self._series_curve_name())
            failed = dict(failed=len(failures), names=", ".join(failures[:3]))
            group = map_group(self._series_map) if self._series_map is not None else 1
            if group > 1:  # rows that are sums are no frames
                text = (trf(MAP_DONE_SUMS_FAILED, group=group, **failed, **values) if failures
                        else trf(MAP_DONE_SUMS, group=group, **values))
            else:
                text = (trf("Series map: {n} frames of {curve}; {failed} failed: {names}", **failed, **values)
                        if failures else trf("Series map: {n} frames of {curve}", **values))
            return f"{text} · {note}" if note else text

        text = said()
        level = "warning" if failures or note else "ok"
        self._status(text, level)
        self._series_done_text = (text, said)
        show_toast(self.window(), text, level=level, action=(tr("Export CSV…"), self.export_series_csv))
        self._find_stages()

    # -- opening a row -----------------------------------------------------------------

    def open_series_row(self, row: Optional[int] = None) -> bool:
        """Show the frame of a map row in Analyze (its file, and its frame of a series), cut where the map
        was cut: a frame whose own automatic cut lies elsewhere would not show the curve of its row."""
        series = self._series_map
        row = self._series_row if row is None else int(row)
        if series is None or not 0 <= row < len(series.refs):
            return False
        path, frame_index = series.refs[row]
        keys = [str(item).casefold() for item in self.view_model.state.files]
        if str(path).casefold() not in keys:
            self._status(tr("That frame is no longer listed"), "warning")
            return False
        index = keys.index(str(path).casefold())
        cut = self._series_built.get("cut")
        shown = (self.file_list.currentRow() == index and int(frame_index) == self.view_model.state.frame_index
                 and self.view_model.state.analysis is not None)
        held = self.view_model.held_cut(path, int(frame_index))  # (opened from a map before: cut as that map was)
        self.view_model.show_with_cut(path, int(frame_index), cut)
        if shown and held == (dict(cut) if cut else None):
            return True  # on screen already, cut as this map was
        return self.show_frame(index, int(frame_index))


__all__ = ["SeriesMixin"]
