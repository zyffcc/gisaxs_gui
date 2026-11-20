"""Choose whole computed solutions (those of Predict parameters, or of Analyze), never unrelated per-parameter range
endpoints."""
from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker


class EquivalentSolutionsMixin:
    def setup_equivalent_solutions(self):
        self.solution_selector.activated.connect(self.choose_equivalent_solution)

    def refresh_equivalent_selector(self):
        combo = self.solution_selector
        blocker = QSignalBlocker(combo)
        combo.clear()
        solutions = self.session.solutions
        for index, solution in enumerate(solutions):
            combo.addItem(f"{index + 1}: {solution.label} · χ² {solution.chi2:.3g}")
        selected = next((i for i, s in enumerate(solutions) if s.model == self.session.model), 0)
        if solutions:
            combo.setCurrentIndex(selected)
        combo.setEnabled(bool(solutions) and not self.fit_running() and self.session.solutions_are_current())
        self.solution_selector_table.setVisible(len(solutions) > 1)
        del blocker

    def choose_equivalent_solution(self, index):
        if not self.fit_running() and self.session.solutions_are_current() and 0 <= index < len(self.session.solutions):
            self.solutions_table.selectRow(index)
            self.use_selected_solution()


__all__ = ["EquivalentSolutionsMixin"]
