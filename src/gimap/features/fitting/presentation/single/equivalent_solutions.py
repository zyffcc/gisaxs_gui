"""Choose whole computed solutions (those of Predict parameters, or of Analyze), never unrelated per-parameter range
endpoints.

Each entry names the solution (numbered where two have the same composition), its χ²ᵣ, probability and flags in
short (their sentences in the entry's tooltip), and says when it is incomplete (Predict was stopped). After a fit that
started from a solution (Refine These Values) the first entry is that fit — “Refined from solution 1 (Refine)” — and
it is the one shown: the values in the table are the fit's, not the solution's."""
from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker, Qt

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared import fit_names

from .solution_texts import numbered_labels, refined_entry, selector_text

FLAGS_TIP = "{label}: {flags}"


class EquivalentSolutionsMixin:
    def setup_equivalent_solutions(self):
        self.solution_selector.activated.connect(self.choose_equivalent_solution)

    def _refined_shown(self):
        """The fit that started from a solution, while it holds (``FitSession.refined_from``)."""
        session = self.session
        if session.refined_from and session.result is not None and session.result_is_current():
            return session.refined_from
        return None

    def refresh_equivalent_selector(self):
        combo = self.solution_selector
        blocker = QSignalBlocker(combo)
        combo.clear()
        solutions = self.session.solutions
        labels = numbered_labels(solutions)
        refined = self._refined_shown() if solutions else None
        if refined:
            combo.addItem(refined_entry(refined, self.session.result), None)
        for index, solution in enumerate(solutions):
            combo.addItem(selector_text(index, solution, labels[index]), index)
            many = len(solution.model.components) > 1
            flags = "; ".join(trf(template, **values) for template, values in
                              fit_names.grouped(solution.flags, many, " ↔ ", tr))
            if flags:
                combo.setItemData(combo.count() - 1, trf(FLAGS_TIP, label=labels[index], flags=flags), Qt.ToolTipRole)
        if refined:
            combo.setCurrentIndex(0)
        elif solutions:
            combo.setCurrentIndex(next((i for i, s in enumerate(solutions) if s.model == self.session.model), 0))
        combo.setEnabled(bool(solutions) and not self.fit_running() and self.session.solutions_are_current())
        self.solution_selector_table.setVisible(len(solutions) > 1 or bool(refined))
        del blocker

    def choose_equivalent_solution(self, index):
        number = self.solution_selector.itemData(index)
        if number is None:  # the fit from a solution: it is in Model already
            return
        if not self.fit_running() and self.session.solutions_are_current() and 0 <= number < len(self.session.solutions):
            self.solutions_table.selectRow(number)
            self.use_selected_solution()


__all__ = ["FLAGS_TIP", "EquivalentSolutionsMixin"]
