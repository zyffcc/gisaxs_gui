"""What can be done from the Results step of the single-curve page.

* next: Refine These Values (after a prediction or a list of solutions), Fit Again (the last method, with Model as it
  is now), Fit Every Curve in This Folder… (In-situ series, each curve from this model);
* the values: select rows of the table to fix them or let them fit (right-click does the same), or to see their
  range in Model — the next fit uses them;
* what the fit did by itself (values at an edge it resolved, ``domain.fit_edges`` / ``fit1d_edges``);
* the folds: other solutions or compositions to choose from, the constraints and ranges fit1d used.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QAction

from src.gimap.app.presentation.i18n import tr, trf

from .fit1d_review import note_text

DONE = "Done automatically:"
FIXED = "{names}: fixed in Model. Fit Again to keep them there."
FREED = "{names}: the next fit changes them. Fit Again to fit them."
NOT_IN_MODEL = "The values selected are not in Model (a global term of fit1d): set them in Model."
RANGE_SHOWN = "The range of {name} is shown in Model: change its min or max there, then Fit Again."
OTHER_SOLUTIONS = "Other solutions ({count})"
OTHER_COMPOSITIONS = "Other compositions ({count})"
FIT_INPUT_SUFFIX = "_fit_input.dat"


class ResultActionsMixin:
    """Needs the page's widgets, ``session``, ``run_fit``, ``set_model``, ``show_step`` and ``seriesRequested``."""

    def setup_result_actions(self) -> None:
        self._row_paths = []
        self._last_method = None
        self.next_refine_button.clicked.connect(lambda: self.run_fit("local"))
        self.fit_again_button.clicked.connect(lambda: self.run_fit(self._last_method or self.method()))
        self.series_button.clicked.connect(self.fit_every_curve_of_folder)
        self.fix_button.clicked.connect(lambda: self.set_selected_free(False))
        self.free_button.clicked.connect(lambda: self.set_selected_free(True))
        self.range_button.clicked.connect(self.show_selected_range)
        self.parameters_table.itemSelectionChanged.connect(self._values_selected)
        # After Copy Rows / Copy Table (``enable_table_copy``): the same as the buttons under the table.
        separator = QAction(self.parameters_table)
        separator.setSeparator(True)
        self.parameters_table.addAction(separator)
        self._value_actions = []
        for text, slot, button in (("Fix", lambda: self.set_selected_free(False), self.fix_button),
                                   ("Let It Fit", lambda: self.set_selected_free(True), self.free_button),
                                   ("Range…", self.show_selected_range, self.range_button)):
            action = QAction(tr(text), self.parameters_table)
            action.triggered.connect(slot)
            self.parameters_table.addAction(action)
            self._value_actions.append((action, text, button))
        self._values_selected()

    # -- the values --------------------------------------------------------------------------------

    def selected_paths(self) -> list:
        """The Model's parameters of the rows selected in the table (rows that are not in Model are left out)."""
        rows = sorted({index.row() for index in self.parameters_table.selectedIndexes()})
        return [self._row_paths[row] for row in rows if row < len(self._row_paths) and self._row_paths[row]]

    def _values_selected(self) -> None:
        paths = self.selected_paths()
        model = self.session.model
        free = [model.get(path).free for path in paths if self._in_model(path)]
        self.fix_button.setEnabled(any(free))
        self.free_button.setEnabled(bool(free) and not all(free))
        self.range_button.setEnabled(len(paths) == 1)
        for action, text, button in getattr(self, "_value_actions", ()):
            action.setEnabled(button.isEnabled())
            action.setText(tr(text))

    def _in_model(self, path) -> bool:
        try:
            self.session.model.get(path)
        except (KeyError, IndexError, TypeError):
            return False
        return True

    def set_selected_free(self, free: bool) -> None:
        """Fix the selected values (``free=False``) or let the next fit change them."""
        paths = [path for path in self.selected_paths() if self._in_model(path)]
        if not paths:
            self._status(tr(NOT_IN_MODEL), "warning")
            return
        model = self.session.model
        for path in paths:
            model = model.with_parameter(path, free=free)
        names = ", ".join(self._path_name(path) for path in paths)
        self.set_model(model, message=lambda: trf(FREED if free else FIXED, names=names))

    def show_selected_range(self) -> None:
        paths = self.selected_paths()
        if len(paths) != 1 or not self._in_model(paths[0]):
            self._status(tr(NOT_IN_MODEL), "warning")
            return
        self.show_ranges_check.setChecked(True)
        self.show_step("model")
        name = self._path_name(paths[0])
        self._status(lambda: trf(RANGE_SHOWN, name=name), "info")

    # -- next ---------------------------------------------------------------------------------------

    def fit_every_curve_of_folder(self) -> None:
        """In-situ series on the folder of this curve: the files like it (Analyze's fit inputs, or the same suffix)."""
        curve = self.session.curve
        if curve is None or not curve.path:
            return
        path = Path(curve.path)
        pattern = f"*{FIT_INPUT_SUFFIX}" if path.name.endswith(FIT_INPUT_SUFFIX) else f"*{path.suffix}"
        self.seriesRequested.emit(str(path.parent), pattern)

    def render_result_actions(self) -> None:
        session = self.session
        native = session.native_result if session.result is None else None
        fitted = session.result is not None or native is not None or bool(session.solutions)
        running = self.fit_running()
        self.next_refine_button.setVisible(bool(native is not None or (session.solutions and session.result is None)))
        self.fit_again_button.setVisible(fitted)
        self.series_button.setVisible(fitted and bool(session.curve is not None and session.curve.path)
                                      and bool(session.model.components))
        for button in (self.next_refine_button, self.fit_again_button, self.series_button):
            button.setEnabled(not running)
        if native is not None:
            notes = list((native.get("fit1d_bounds") or {}).get("actions") or ())
        elif self._fit_notes and self._fit_notes.get("result") is session.result and session.result is not None:
            notes = [note.to_dict() for note in self._fit_notes["actions"]]
        else:
            notes = []
        self.actions_label.setText("\n".join([tr(DONE)] + ["• " + note_text(note, self._path_name) for note in notes]))
        self.actions_label.setVisible(bool(notes))
        candidates = self.fill_fit1d_candidates(native)
        solutions = bool(session.solutions)
        self.alternatives.setVisible(candidates or solutions)
        if candidates:
            self.alternatives.toggle_button.setText(trf(OTHER_COMPOSITIONS, count=len(self._fit1d_candidates)))
        elif solutions:
            self.alternatives.toggle_button.setText(trf(OTHER_SOLUTIONS, count=len(session.solutions)))
        if solutions and session.result is None and not self.alternatives.is_expanded():
            self.alternatives.set_expanded(True)  # choosing among them is what they are for
        self.constraints_section.setVisible(native is not None)
        if native is not None:
            self.fit1d_constraints_label.setText(self.fit1d_constraints_text(native))
        has_values = bool(self._row_paths) and fitted
        for widget in (self.fix_button, self.free_button, self.range_button, self.values_hint):
            widget.setVisible(has_values)
        self._values_selected()


__all__ = ["ResultActionsMixin"]
