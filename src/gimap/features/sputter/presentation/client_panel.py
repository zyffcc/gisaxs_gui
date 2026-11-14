"""The main screen: one time window of an Analyze result, and the solution chosen for it (a mixin of ``SputterPage``).

A result is a list of time windows; each has several solutions (the network's candidates and, when the analysis
refined, the refined one). The page shows one window at a time — the slider, the arrows or a click on the evolution
choose it — and one solution of it: the measured image beside that solution's model image (or, with the switch
Patterns | Real Space, the solution drawn in real space: ``real_space_view.py``), its values on the tiles
and its fit deviation. The solution shown first is the window's default (the one the smooth evolution over the scan
selects, else the lowest fit deviation). Choosing another one is kept per window in ``main/choices.json`` next to the
result (only where it differs from the default) and read again when the result is opened; the evolution plots follow
the choices. A quantity a solution does not have hides its tile; a window without a solution says why instead of its
images. Nothing here grades a solution: when the data cannot tell solutions apart, they are simply all offered.

Two kinds of result are shown with the same parts. On a porous substrate the tiles are the island radius (said to be
a relative size), the width, the island, cap and film shares and the two correlation lengths; the evolution draws
those. A series of static samples has one sample in the place of a time window: the selector names the sample and
its nominal thickness, the evolution runs along the series' axis, and no text speaks of time; a range the analysis
gives for a value is in the tooltip of its tile.
"""

from __future__ import annotations

from typing import Optional

from src.gimap.app.presentation.components import SegmentedControl
from src.gimap.app.presentation.i18n import tr

from ..application import ClientResult, ClientWindow
from .client_texts import (CHOICE_NOT_SAVED, EMPTY_ACTION, EMPTY_MESSAGE, EMPTY_TITLE, PACKAGE_ACTION, PACKAGE_TITLE,
                           RELATIVE_RADIUS_TIP, RESULT_NAME, SAMPLE_CONTROLS, SAMPLE_ERROR, WINDOW_CONTROLS,
                           WINDOW_ERROR, entry_name, entry_segment, entry_tip, range_text, shown_number, window_text,
                           window_tip)
from .evolution_view import RELATIVE_RADIUS, EvolutionPlot
from .pattern_view import PatternPair
from .real_space_view import RealSpacePicture
from .texts import message_text, text
from .views.sputter_main_view import ENTRY_HINT, TILES

TILE_DIGITS = {"width": 2, "coverage": 2, "f_islands": 2, "cap_share": 2, "film_share": 2}
"""Significant digits of a tile's value (3 where not named)."""
PATTERN_CACHE = 8


class ClientPanelMixin:
    """Needs the widgets of ``views/sputter_main_view.py``, ``service`` and ``_notify``."""

    def _init_client_panel(self) -> None:
        self.client_result: Optional[ClientResult] = None
        self.client_choices: dict = {}
        self.client_window_index = 0
        self.entry_control: Optional[SegmentedControl] = None
        self._entry_labels: tuple = ()
        self._pattern_cache: dict = {}
        self._empty_kind = ""
        self.patterns = PatternPair(self.pattern_layout)
        self.real_space = RealSpacePicture(self.space_layout)
        self.image_switch.currentIndexChanged.connect(self._show_image_kind)
        self.evolution = EvolutionPlot(self.evolution_layout, self.select_window)
        self.window_slider.valueChanged.connect(self.select_window)
        self.window_previous.clicked.connect(lambda: self.select_window(self.client_window_index - 1))
        self.window_next.clicked.connect(lambda: self.select_window(self.client_window_index + 1))
        self.default_button.clicked.connect(self._back_to_default)
        self.client_empty.actionRequested.connect(self._empty_action)
        self.entry_hint.set_full_text(tr(ENTRY_HINT))
        self._show_result_kind()
        self._show_client_empty()

    # -- the empty states ----------------------------------------------------------------------------------------
    def _show_client_empty(self) -> None:
        """Without a result: how to get one — or, without the analysis package, how to give it."""
        if self.client_result is not None:
            self.result_stack.setCurrentIndex(1)
            return
        client = self.service.client
        problem = client.package().problem if client is not None else ""
        self._empty_kind = "package" if problem else "result"
        if problem:
            self.client_empty.set_content(tr(PACKAGE_TITLE), message_text(problem), tr(PACKAGE_ACTION))
        else:
            self.client_empty.set_content(tr(EMPTY_TITLE), tr(EMPTY_MESSAGE), tr(EMPTY_ACTION))
        self.result_stack.setCurrentIndex(0)

    def _empty_action(self) -> None:
        if self._empty_kind == "package":
            self._choose_package()
        else:
            self._choose_client_result()

    # -- a result ------------------------------------------------------------------------------------------------
    def show_client_result(self, result: Optional[ClientResult], *, sample: str = "") -> None:
        """Show ``result`` with the choices kept for it (None: the empty state). ``sample``: the sample of a static
        series to show first (else the first window that has a solution)."""
        if result is not None and not result.windows:
            result = None  # nothing to show: the empty state
        self.client_result = result
        self._pattern_cache = {}
        self.client_choices = {}
        has_report = result is not None and result.file("report_md") is not None
        self.report_button.setEnabled(has_report)
        self.result_folder_button.setEnabled(result is not None)
        self._show_result_kind()
        if result is None:
            self.evolution.draw(None, None, None)
            self.patterns.show(None, "", "")
            self._show_client_empty()
            return
        self.client_choices = self.service.client.choices(result)
        named = "/".join(part for part in (result.metal, result.substrate) if part) or "?"  # the metal on its substrate
        self.result_name.setText(RESULT_NAME.format(scan=result.scan or result.folder.name, sample=named,
                                                    family=result.family or "?"))
        self.result_name.setToolTip(str(result.folder))
        self.window_slider.blockSignals(True)
        self.window_slider.setRange(0, len(result.windows) - 1)
        first = next((index for index, window in enumerate(result.windows) if window.entries), 0)
        if sample:
            first = next((index for index, window in enumerate(result.windows) if window.label == sample), first)
        self.window_slider.setValue(first)
        self.window_slider.blockSignals(False)
        self.client_window_index = first
        self.result_stack.setCurrentIndex(1)
        self._draw_evolution()
        self._show_window()

    def _show_result_kind(self) -> None:
        """What depends on the kind of result: the words of the selector and the caption of the radius."""
        result = self.client_result
        static = result is not None and result.static
        porous = result is not None and result.porous
        previous, following, shown = (tr(words) for words in (SAMPLE_CONTROLS if static else WINDOW_CONTROLS))
        self.window_previous.setToolTip(previous)
        self.window_next.setToolTip(following)
        self.window_slider.setToolTip(shown)
        for key, caption, _unit, _tip in TILES:
            name = self.tile_captions[key]
            name.set_full_text(tr(RELATIVE_RADIUS if porous and key == "R_nm" else caption))
            name.setToolTip("")  # the tile's own tooltip explains the quantity

    def client_window(self) -> Optional[ClientWindow]:
        result = self.client_result
        if result is None or not 0 <= self.client_window_index < len(result.windows):
            return None
        return result.windows[self.client_window_index]

    def client_entry_label(self) -> str:
        """The label of the solution shown for the window shown ("" without one)."""
        window = self.client_window()
        return window.chosen_label(self.client_choices) if window is not None else ""

    # -- which window --------------------------------------------------------------------------------------------
    def select_window(self, index: int) -> None:
        result = self.client_result
        if result is None:
            return
        index = max(0, min(len(result.windows) - 1, int(index)))
        if index == self.client_window_index and self.window_slider.value() == index:
            return
        self.client_window_index = index
        if self.window_slider.value() != index:
            self.window_slider.blockSignals(True)
            self.window_slider.setValue(index)
            self.window_slider.blockSignals(False)
        self.evolution.mark(index)
        self._show_window()

    def _show_window(self) -> None:
        window, result = self.client_window(), self.client_result
        if window is None:
            return
        count = len(result.windows)
        self.window_label.setText(window_text(window, result.static))
        self.window_label.setToolTip(window_tip(window, self.client_window_index, count, result.static))
        self.window_previous.setEnabled(self.client_window_index > 0)
        self.window_next.setEnabled(self.client_window_index < count - 1)
        self._fill_entries(window)
        self._show_entry()

    # -- which solution ------------------------------------------------------------------------------------------
    def _fill_entries(self, window: ClientWindow) -> None:
        """The selector of this window's solutions: built again only when the window has other entries."""
        labels = tuple(window.labels())
        if labels != self._entry_labels or self.entry_control is None:
            if self.entry_control is not None:
                self.entry_layout.removeWidget(self.entry_control)
                self.entry_control.hide()  # gone at once: Qt deletes it on the next turn of the event loop
                self.entry_control.deleteLater()
            control = SegmentedControl(self.client_view)
            control.setObjectName("sputterEntries")
            for entry in window.entries:
                control.addItem(entry_segment(entry), entry.label)
            control.activated.connect(lambda index, control=control: self.select_entry(control.itemData(index)))
            self.entry_layout.addWidget(control)
            self.entry_control, self._entry_labels = control, labels
        control = self.entry_control
        for index, entry in enumerate(window.entries):
            control.button(index).setText(entry_segment(entry))
            control.setItemToolTip(index, entry_tip(entry))
        control.setVisible(bool(labels))
        self.entry_caption.setVisible(bool(labels))
        self.entry_hint.setVisible(bool(labels))
        chosen = control.findData(window.chosen_label(self.client_choices)) if labels else -1
        if chosen >= 0:
            control.blockSignals(True)
            control.setCurrentIndex(chosen)
            control.blockSignals(False)

    def select_entry(self, label: str) -> None:
        """Show solution ``label`` of the window shown and keep the choice with the result."""
        window, result = self.client_window(), self.client_result
        if window is None or label not in window.labels():
            return
        if label == window.chosen_label(self.client_choices):
            return
        self.client_choices = self.service.client.with_choice(window, label, self.client_choices)
        try:
            self.service.client.save_choices(result, self.client_choices)
        except OSError as exc:  # the choice still holds while the result is shown
            self._notify(text(CHOICE_NOT_SAVED, reason=str(exc)), error=True)
        self._fill_entries(window)
        self._show_entry()
        self._draw_evolution()

    def _back_to_default(self) -> None:
        window = self.client_window()
        if window is not None:
            self.select_entry(window.default_label())

    def _show_entry(self) -> None:
        """The images (or the real-space picture), the tiles and the fit deviation of the solution shown."""
        window, result = self.client_window(), self.client_result
        entry = window.chosen(self.client_choices) if window is not None else None
        porous = result is not None and result.porous
        self.default_button.setVisible(entry is not None and entry.label != window.default_label())
        for key, _caption, unit, tip in TILES:
            value = entry.value(key) if entry is not None else None
            digits = TILE_DIGITS.get(key, 3)
            self.tile_values[key].setText(shown_number(value, digits))
            words = [tr(RELATIVE_RADIUS_TIP if porous and key == "R_nm" else tip)]
            found = range_text(entry.range(key) if entry is not None else None, digits)
            if found:
                words.append(f"{found} {unit}".rstrip())
            self.tiles[key].setToolTip("\n".join(words))
            self.tiles[key].setVisible(value is not None)
        deviation = entry.fit_deviation if entry is not None else None
        self.deviation_value.setText(shown_number(deviation))
        self.deviation_caption.setVisible(deviation is not None)
        self.deviation_value.setVisible(deviation is not None)
        self._draw_space()
        if entry is None:
            reason = message_text(window.error) if window is not None and window.error else ""
            static = result is not None and result.static
            self.patterns.show_message(text(SAMPLE_ERROR if static else WINDOW_ERROR, reason=reason))
            return
        try:
            patterns = self._patterns(window)
        except (OSError, ValueError) as exc:
            self.patterns.show_message(message_text(exc))
            return
        self.patterns.show(patterns, entry.label, entry_name(entry.label))

    def _show_image_kind(self, _index: int = 0) -> None:
        """The patterns or the solution in real space, as the switch says."""
        space = self.image_switch.currentData() == "space"
        self.pattern_host.setCurrentIndex(1 if space else 0)
        if space:
            self._draw_space()

    def _draw_space(self) -> None:
        """The chosen solution of the window shown, in real space (drawn only while that picture is shown)."""
        if self.pattern_host.currentIndex() != 1:
            return
        window, result = self.client_window(), self.client_result
        entry = window.chosen(self.client_choices) if window is not None else None
        if entry is None:
            reason = message_text(window.error) if window is not None and window.error else ""
            static = result is not None and result.static
            self.real_space.show_message(text(SAMPLE_ERROR if static else WINDOW_ERROR, reason=reason))
            return
        substrate = getattr(window, "substrate", "") or result.substrate
        self.real_space.show(entry.values, result.metal, substrate, entry_name(entry.label))

    def _patterns(self, window: ClientWindow):
        """The images of a window (the last few are kept: going back and forth does not read them again)."""
        cached = self._pattern_cache.get(window.key)
        if cached is None:
            cached = self.service.client.patterns(self.client_result, window)
            while len(self._pattern_cache) >= PATTERN_CACHE:
                self._pattern_cache.pop(next(iter(self._pattern_cache)))
            self._pattern_cache[window.key] = cached
        return cached

    def _draw_evolution(self) -> None:
        shown = self.client_window_index if self.client_window() is not None else None
        self.evolution.draw(self.client_result, self.client_choices, shown)

    # -- theme and language ---------------------------------------------------------------------------------------
    def redraw_client_plots(self) -> None:
        self.patterns.redraw()
        self.real_space.redraw()
        self.evolution.redraw()

    def refresh_client_panel_language(self) -> None:
        self.entry_hint.set_full_text(tr(ENTRY_HINT))
        self._show_result_kind()
        if self.client_result is None:
            self._show_client_empty()
            return
        self._entry_labels = ()  # the names of the entries are composed again
        self._show_window()
        self.redraw_client_plots()  # every title and axis title again, not only the model's


__all__ = ["ClientPanelMixin", "TILE_DIGITS"]
