"""Choose whole computed solutions, never unrelated per-parameter range endpoints."""
from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import FitModel
from ...application.single_fit1d import in_model_order, is_auto


class EquivalentSolutionsMixin:
    def setup_equivalent_solutions(self):
        self._equivalent_solutions = []
        self._equivalent_curve = None
        self.solution_selector.activated.connect(self.choose_equivalent_solution)

    def keep_equivalent_solutions(self, value, selection, curve):
        self._equivalent_curve = curve
        self._equivalent_selection = selection
        native = value["native"]
        self._equivalent_base = FitModel() if is_auto(native.get("settings") or {}) else self.session.model
        self._equivalent_solutions = [dict(native=native, mapping_row=value["mapping_row"], delta_chi2=0)]
        self._equivalent_solutions.extend(value.get("equivalents") or [])

    def refresh_equivalent_selector(self):
        combo = self.solution_selector
        blocker = QSignalBlocker(combo)
        combo.clear()
        native = self.session.native_result
        if native is None and self.session.solutions:
            for index, solution in enumerate(self.session.solutions):
                combo.addItem(f"{index + 1}: {solution.label} · χ² {solution.chi2:.3g}")
            selected = next((i for i, s in enumerate(self.session.solutions) if s.model == self.session.model), 0)
            combo.setCurrentIndex(selected)
            combo.setEnabled(not self.fit_running() and self.session.solutions_are_current())
            self.solution_selector_table.setVisible(len(self.session.solutions) > 1)
            return
        same = (self._equivalent_curve is self.session.curve and native is not None and self.session.result is None
                and any(choice["native"] is native for choice in self._equivalent_solutions))
        choices = self._equivalent_solutions if same else []
        selected = 0
        for index, choice in enumerate(choices):
            title = tr("Best solution") if index == 0 else trf("Equivalent solution {index}", index=index)
            combo.addItem(trf("{solution} · Δχ² {delta}", solution=title, delta=f"{choice['delta_chi2']:.3g}"))
            if choice["native"] is native:
                selected = index
        combo.setCurrentIndex(selected)
        combo.setEnabled(not self.fit_running() and self.session.native_is_current())
        self.solution_selector_table.setVisible(len(choices) > 1)
        del blocker

    def choose_equivalent_solution(self, index):
        if self.session.native_result is None:
            if not self.fit_running() and self.session.solutions_are_current() and 0 <= index < len(self.session.solutions):
                self.solutions_table.selectRow(index)
                self.use_selected_solution()
            return
        if (self.fit_running() or not self.session.native_is_current()
                or self._equivalent_curve is not self.session.curve
                or not 0 <= index < len(self._equivalent_solutions)):
            return
        choice = self._equivalent_solutions[index]
        native = choice["native"]
        native["selected_solution"] = index
        row = choice["mapping_row"]
        if self._equivalent_base.components:
            row = in_model_order(row, self._equivalent_base)
        self._take_fit1d(native, row, self._equivalent_base, self._equivalent_selection)
