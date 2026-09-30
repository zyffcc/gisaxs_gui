"""The four steps of fitting one curve, shown like the steps of Analyze: Curve → Model → Fit → Results.

The rail sits above the Fitting controls. Each step says where things stand — the curve loaded (its
points and q range), the components chosen, whether a model is fitted (running, done, needs
attention), what can be exported — and a click goes there: the curve card, the Components tab, the
fitting methods (1D Predict), the plot and its Export buttons. Choosing a tab moves the rail along.

The states are read from what is on the page (the curve, the component choosers, the fitted curve,
the Fit step of the workflow) about twice a second while Fitting is visible: every way of fitting —
the AI, the physical fit, Global Search, Local Refine, Plot Current Model — shows up without each of
them reporting to the rail.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.components import StepRail
from src.gimap.app.presentation.i18n import tr

STEPS = (("curve", "Curve"), ("model", "Model"), ("fit", "Fit"), ("results", "Results"))
TAB_STEPS = {0: "model", 1: "model", 2: "fit", 3: "fit"}
MODEL_TAB, FIT_TAB = 0, 3
REFRESH_MS = 500


class FitSteps:
    def __init__(self, workspace, binding: Callable[[], Optional[object]]):
        self.workspace = workspace
        self._binding = binding
        content = workspace.controls_content
        self.rail = StepRail(STEPS, content)
        self.rail.setObjectName("fittingStepRail")
        content.layout().insertWidget(0, self.rail)
        self.rail.stepChosen.connect(self.go)
        self.tabs = workspace.fitting_controls_card.mode_tabs
        self.tabs.currentChanged.connect(self._tab_changed)
        self._timer = QTimer(self.rail)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.rail.set_current("curve")
        self.refresh(force=True)

    def go(self, key: str) -> None:
        """Show the controls of a step."""
        workspace = self.workspace
        controls = workspace.controls_scroll_area
        if key == "curve":
            controls.ensureWidgetVisible(workspace.curve_card)
        elif key in ("model", "fit"):
            self.tabs.setCurrentIndex(MODEL_TAB if key == "model" else FIT_TAB)
            controls.ensureWidgetVisible(self.tabs)
        elif key == "results":
            workspace.fitting_plot_card.set_expanded(True)
            export = getattr(workspace.ui, "FittingExportButton", None)
            if export is not None:
                workspace.results_scroll_area.ensureWidgetVisible(export)
        self.rail.set_current(key)

    def _tab_changed(self, index: int) -> None:
        step = TAB_STEPS.get(int(index))
        if step is not None:
            self.rail.set_current(step)

    def refresh(self, *, force: bool = False) -> None:
        if not force and not self.rail.isVisible():
            return
        binding = self._binding()
        curve = getattr(binding, "current_1d_data", None) if binding is not None else None
        detail = ""
        if curve is not None:
            card = self.workspace.curve_card
            points = card.detail_label.text().split(" · ")[0]
            detail = " · ".join(part for part in (card.name_label.text(), points) if part)
        self.rail.set_state("curve", "ok" if curve is not None else "pending",
                            detail or tr("Open a curve, or Send to Fitting in Analyze"))
        shapes = []
        if binding is not None and hasattr(binding, "_collect_active_particles"):
            try:
                shapes = binding._collect_active_particles()[0]
            except (AttributeError, RuntimeError):
                shapes = []
        drawn = drawn_solution(binding)
        if drawn:
            self.rail.set_state("model", "ok", tr("{model} (the solution drawn)").format(model=drawn))
        elif shapes:
            self.rail.set_state("model", "ok", components_text(shapes))
        else:
            self.rail.set_state("model", "warn" if curve is not None else "pending",
                                tr("Choose at least one component (Components)"))
        status = ""
        view_model = getattr(binding, "fitting_view_model", None)
        if view_model is not None:
            try:
                step = view_model.state.workflow.step("fit")
                status, message = step.status, step.message
            except (AttributeError, StopIteration):
                status, message = "", ""
        fitted = bool(getattr(binding, "has_fitting_data", False) or getattr(binding, "_has_fitting_data", False))
        if status == "running":
            self.rail.set_state("fit", "busy", tr("Fitting …"))
        elif status == "error":
            self.rail.set_state("fit", "error", message or tr("The fit needs attention (see Log)"))
        elif fitted:
            self.rail.set_state("fit", "ok", tr("A model is drawn on the data"))
        else:
            self.rail.set_state("fit", "pending", tr("Fit curve, Physical fit (no AI) or Global Search"))
        if fitted:
            self.rail.set_state("results", "ok", tr("Export Data… (fit and parameters) or Export Plot…"))
        else:
            self.rail.set_state("results", "pending", tr("The fitted curve, its parameters and the plot"))

    def dispose(self) -> None:
        self._timer.stop()


def drawn_solution(binding) -> str:
    """The model of a drawn workflow solution (1D Predict, or Analyze ▸ Results ▸ Show in Fitting), else ""."""
    fitting = getattr(binding, "fitting", None)
    meta = (fitting.get("meta") or {}) if isinstance(fitting, dict) else {}
    if meta.get("source") != "native_v5":
        return ""
    return str((meta.get("candidate") or {}).get("combination") or "").replace("_", " ")


def components_text(shapes) -> str:
    """“Sphere ×3”, “Sphere + Cylinder ×2”: the components in the order first chosen."""
    counts: dict[str, int] = {}
    for shape in shapes:
        name = str(shape).title()
        counts[name] = counts.get(name, 0) + 1
    return " + ".join(name if count == 1 else f"{name} ×{count}" for name, count in counts.items())


__all__ = ["FitSteps", "STEPS", "components_text", "drawn_solution"]
