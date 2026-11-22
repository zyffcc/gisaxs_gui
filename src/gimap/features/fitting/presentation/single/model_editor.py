"""The model of one curve as cards: one per particle component, one for background and resolution, one for the
spheres in grazing incidence, and one for links and constraints.

Each parameter is one line (``parameter_row``): name (unit), value, ± error of the last fit, min and max (when ranges
are shown), fit (free or fixed) and its link button. A card's header chooses the family, switches the interference
between neighbours S(q) on or off, and removes the component; under its parameters, the size distribution and (with
S(q)) the interference function and the coupling. Every edit emits ``edited`` with the new model, its links applied;
values typed outside their range widen the range.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.i18n import tr, trf

from ...application.fit_uncertainty import linked, prune_links, renumbered
from ...application.single_fit import (
    BASE_GLOBALS,
    FAMILIES,
    GRAZING,
    INTERFERENCES,
    NETWORK_FAMILIES,
    RESOLUTION,
    SIZE_LAWS,
    Component,
    FitModel,
    new_component,
    with_interference,
)
from ...application.substrate_level import with_substrate
from .model_links import add_constraint, links_card, remove_constraint
from .parameter_row import ParameterRow, header, label_text
from .substrate_row import SubstrateControls

MANUAL_ONLY = "For Refine and Search: Predict parameters does not propose this family"
DROPPED = "Links or constraints that no longer hold were removed: {items}"
SIZES_TIP = ("Choose the size distribution for this population. The parameter labels give its location and width. "
             "Results also show the mean radius and relative spread by number and by intensity.")
RESOLUTION_CHOICES = {"constant": "Constant σ_q", "file": "Resolution from the file"}
WEIGHT_CHOICES = {"file": "σ from the file", "relative": "Relative (σ = I)", "poisson": "Poisson (σ = √I)",
                  "none": "None (fit ln I)"}
INTERFERENCE_TIP = ("1D paracrystal: neighbours along a line (Hosemann). 2D hexagonal paracrystal: a disordered "
                    "hexagonal lattice in the plane, averaged over its orientation. Hard spheres: the Percus–Yevick "
                    "liquid of radius R_HS and volume fraction φ. The last two are for Refine and Search only.")
COUPLING_TIP = ("LMA: every size sits among neighbours of its own size. DA: sizes and positions are independent, so "
                "the size spread adds scattering that the spacing does not modulate.")


def _narrow(box: QComboBox) -> None:
    """A combo that does not ask for the width of its longest item (the Model panel is narrow)."""
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(10)


class FitModelEditor(QWidget):
    edited = pyqtSignal(object)
    """The model after an edit by the person (``FitModel``)."""
    message = pyqtSignal(str, str)
    """(text, level): a link or constraint that could not be made, or that was removed."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("fitModelEditor")
        self.model = FitModel()
        self.show_ranges = False
        self._errors: Mapping = {}
        self._at_bounds: tuple = ()
        self._rows: dict[tuple, ParameterRow] = {}
        self._shape: tuple = ()
        self._links_card: Optional[QFrame] = None
        self._links_shape: tuple = ()
        self._substrate: Optional[SubstrateControls] = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._rebuild()

    # -- from the page ------------------------------------------------------------------

    def set_model(self, model: FitModel, errors: Optional[Mapping] = None, at_bounds=()) -> None:
        self.model = model
        self._errors, self._at_bounds = dict(errors or {}), tuple(at_bounds)
        if self._structure(model) != self._shape:
            self._rebuild()
        else:
            self._refresh_rows()

    def set_show_ranges(self, on: bool) -> None:
        self.show_ranges = bool(on)
        for row in self._rows.values():
            row.show_range(self.show_ranges)
        for title in self.findChildren(QLabel, "fitRangeTitle"):
            title.setVisible(self.show_ranges)

    def refresh_language(self) -> None:
        """After a switch of the interface language: the cards again (names with their units, families, tips)."""
        self._rebuild()
        self._refresh_rows()

    def add_component(self, family: str) -> None:
        radius = self.model.components[-1].value("R") if self.model.components else None
        component = new_component(family, radius=radius * 2 if radius else None)
        self._emit(replace(self.model, components=(*self.model.components, component)))

    def path_title(self, path: tuple) -> str:
        """“1·Sphere σR/R”, “Background”: a parameter in the link menu."""
        owner, key = path
        if owner == "globals":
            return label_text(key)
        component = self.model.components[owner]
        return f"{owner + 1}·{tr(FAMILIES[component.family][0])} {label_text(key, component)}"

    # -- edits --------------------------------------------------------------------------

    def change(self, path: tuple, **changes) -> None:
        self._errors = {key: value for key, value in self._errors.items() if key != path}
        self._emit(self.model.with_parameter(path, **changes), rebuild=False)

    def link(self, path: tuple, expression: Optional[str]) -> None:
        """Link ``path`` to ``expression`` (``None``: unlink; it keeps its value, fixed)."""
        try:
            model = linked(self.model, path, expression)
        except ValueError as exc:
            self.message.emit(trf("Could not link: {reason}", reason=str(exc)), "warning")
            return
        self._errors = {}
        self._emit(model)

    def set_contact(self, on: bool) -> None:
        self._emit(replace(self.model, contact=bool(on)))

    def set_substrate(self, material: str, delta: float = 0.0, beta: float = 0.0) -> None:
        """The substrate under the particles: a material of the table, ``custom`` δ and β, or none (``""``)."""
        try:
            model = with_substrate(self.model, material, delta, beta)
        except ValueError as exc:
            self.message.emit(str(exc), "warning")
            return
        self._emit(model, rebuild=False)

    def add_constraint(self, text: str) -> None:
        try:
            model = add_constraint(self.model, text)
        except ValueError as exc:
            self.message.emit(trf("Could not add the constraint: {reason}", reason=str(exc)), "warning")
            return
        self._emit(model)

    def remove_constraint(self, index: int) -> None:
        self._emit(remove_constraint(self.model, index))

    def _replace_component(self, index: int, component: Optional[Component]) -> None:
        model = self.model
        if component is None:
            model = renumbered(model, index)  # links to later components move down; to this one they go
        components = list(model.components)
        if component is None:
            del components[index]
        else:
            components[index] = component
        self._errors = {}
        self._emit(replace(model, components=tuple(components)))

    def _family(self, index: int, family: str) -> None:
        old = self.model.components[index]
        if family == old.family:
            return
        fresh = new_component(family, structure=old.structure, law=old.law, interference=old.interference)
        params = {key: old.params.get(key, value) for key, value in fresh.params.items()}
        self._replace_component(index, replace(fresh, params=params, coupling=old.coupling))

    def _interference(self, index: int, kind: str) -> None:
        if kind != self.model.components[index].interference:
            self._replace_component(index, with_interference(self.model.components[index], kind))

    def _emit(self, model: FitModel, *, rebuild: bool = True) -> None:
        model, dropped = prune_links(model)  # linked values follow; links that lost a name go
        if dropped:
            self.message.emit(trf(DROPPED, items="; ".join(dropped)), "warning")
        self.model = model
        if rebuild and self._structure(model) != self._shape:
            self._rebuild()
        else:
            self._refresh_rows()
        self.edited.emit(model)

    def _refresh_rows(self) -> None:
        for row in self._rows.values():
            row.refresh(self._errors, self._at_bounds)
        if self._substrate is not None:
            self._substrate.refresh()
        if self._links_signature(self.model) != self._links_shape:
            self._replace_links_card()

    # -- cards --------------------------------------------------------------------------

    @staticmethod
    def _structure(model: FitModel) -> tuple:
        return (tuple((c.family, c.structure, c.interference, c.heights, c.law, c.coupling) for c in model.components),
                model.resolution_mode, model.weighting)

    @staticmethod
    def _links_signature(model: FitModel) -> tuple:
        ties = tuple((path, p.tie) for path, p in model.parameters() if p.tie)
        return ties, tuple(model.inequalities), model.contact

    def _rebuild(self) -> None:
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows = {}
        self._shape = self._structure(self.model)
        for index, component in enumerate(self.model.components):
            self._layout.addWidget(self._component_card(index, component))
        if not self.model.components:
            empty = QLabel(tr("No particle yet: add one above, or let Fit ▸ Predict parameters find them."), self)
            empty.setWordWrap(True)
            empty.setProperty("gimapRole", "muted")
            self._layout.addWidget(empty)
        self._layout.addWidget(self._globals_card())
        self._layout.addWidget(self._grazing_card())
        self._links_card = links_card(self)
        self._links_shape = self._links_signature(self.model)
        self._layout.addWidget(self._links_card)
        self.set_show_ranges(self.show_ranges)

    def _replace_links_card(self) -> None:
        card = links_card(self)
        if self._links_card is not None:
            self._layout.replaceWidget(self._links_card, card)
            self._links_card.setParent(None)
            self._links_card.deleteLater()
        else:
            self._layout.addWidget(card)
        self._links_card, self._links_shape = card, self._links_signature(self.model)

    def _card(self, name: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName(name)
        card.setProperty("gimapInfoCard", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        return card, layout

    def _grid(self, card: QFrame, keys, owner) -> QGridLayout:
        """One line per parameter of ``owner`` under the column titles."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(3)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(2, 1)  # the ± column takes what is left (its text is cut with “…”, the tooltip has it)
        header(grid, card)
        for column in (3, 4):
            title = grid.itemAtPosition(0, column)
            if title is not None and title.widget() is not None:
                title.widget().setObjectName("fitRangeTitle")
        for line, key in enumerate(keys):
            self._rows[(owner, key)] = ParameterRow(self, grid, line, (owner, key), card)
        return grid

    def _component_card(self, index: int, component: Component) -> QFrame:
        card, layout = self._card(f"fitComponent_{index}")
        top = QHBoxLayout()
        title = QLabel(f"{index + 1}", card)
        title.setProperty("gimapRole", "strong")
        family = QComboBox(card)
        family.setObjectName(f"fitFamily_{index}")
        _narrow(family)
        for key, (name, _keys) in FAMILIES.items():
            family.addItem(tr(name), key)
            if key not in NETWORK_FAMILIES:
                family.setItemData(family.count() - 1, tr(MANUAL_ONLY), Qt.ToolTipRole)
        family.setCurrentIndex(family.findData(component.family))
        family.currentIndexChanged.connect(lambda _i, box=family, at=index: self._family(at, box.currentData()))
        structure = QCheckBox("S(q)", card)
        structure.setObjectName(f"fitStructure_{index}")
        structure.setToolTip(tr("Interference between neighbours (a structure factor): choose its kind below the "
                                "parameters"))
        structure.setChecked(component.structure)
        structure.toggled.connect(lambda on, at=index: self._replace_component(
            at, replace(self.model.components[at], structure=bool(on))))
        remove = QToolButton(card)
        remove.setObjectName(f"fitRemove_{index}")
        remove.setText("×")
        remove.setAutoRaise(True)
        remove.setToolTip(tr("Remove this component"))
        remove.clicked.connect(lambda _checked=False, at=index: self._replace_component(at, None))
        top.addWidget(title)
        top.addWidget(family, 1)
        top.addWidget(structure)
        top.addWidget(remove)
        layout.addLayout(top)
        layout.addLayout(self._grid(card, component.keys(), index))
        laws = {key: label for key, label in SIZE_LAWS.items()
                if key not in ("lognormal", "number_gauss") or key == component.law}
        layout.addLayout(self._choice(card, "Sizes", f"fitSizes_{index}", laws, component.law, index, "law",
                                      SIZES_TIP))
        if component.structure:
            kinds = {key: name for key, (name, _keys) in INTERFERENCES.items()}
            layout.addLayout(self._choice(card, "Interference", f"fitInterference_{index}", kinds,
                                          component.interference, index, "interference", INTERFERENCE_TIP))
            layout.addLayout(self._choice(card, "Coupling", f"fitCoupling_{index}",
                                          {"lma": "LMA: P·S", "da": "DA: P + |⟨F⟩|²(S − 1)"}, component.coupling,
                                          index, "coupling", COUPLING_TIP))
        return card

    def _choice(self, card, label: str, name: str, items: dict, current: str, index: Optional[int], field: str,
                tip: str) -> QHBoxLayout:
        """One labelled choice for a population or the whole manual model."""
        row = QHBoxLayout()
        box = QComboBox(card)
        box.setObjectName(name)
        _narrow(box)
        for key, text in items.items():
            box.addItem(tr(text), key)
        box.setToolTip(tr(tip))
        box.setCurrentIndex(max(0, box.findData(current)))
        if index is None:
            box.currentIndexChanged.connect(lambda: self._emit(replace(self.model, **{field: box.currentData()})))
        elif field == "interference":
            box.currentIndexChanged.connect(lambda _i, at=index, combo=box: self._interference(at, str(combo.currentData())))
        else:
            box.currentIndexChanged.connect(lambda _i, at=index, combo=box, key=field: self._replace_component(
                at, replace(self.model.components[at], **{key: str(combo.currentData())})))
        row.addWidget(QLabel(tr(label), card))
        row.addWidget(box, 1)
        return row

    def _globals_card(self) -> QFrame:
        card, layout = self._card("fitGlobals")
        title = QLabel(tr("Background and beam tail"), card)
        title.setProperty("gimapRole", "strong")
        title.setToolTip(tr("I(q) = BG + k·[Σ particles + A/(1 + (|q|/w)^ν)]: the peak is the direct and "
                            "reflected beam's tail and the resolution near q = 0"))
        layout.addWidget(title)
        keys = BASE_GLOBALS + (RESOLUTION if self.model.resolution_mode == "constant" else ())
        layout.addLayout(self._grid(card, keys, "globals"))
        layout.addLayout(self._choice(card, "Resolution", "fitResolutionMode", RESOLUTION_CHOICES,
                                      self.model.resolution_mode, None, "resolution_mode",
                                      "Refine and Search smear the particle terms using constant σ_q or each point’s dQ."))
        layout.addLayout(self._choice(card, "Weights", "fitWeighting", WEIGHT_CHOICES, self.model.weighting, None,
                                      "weighting", "Refine and Search use these weights. Without a file σ, the default "
                                      "fits ln I with equal weights."))
        return card

    def _grazing_card(self) -> QFrame:
        card, layout = self._card("fitGrazing")
        title = QLabel(tr("Spheres in grazing incidence"), card)
        title.setProperty("gimapRole", "strong")
        title.setToolTip(tr("αi = αf = 0: spheres at q_z = 0 (effective interface approximation, no substrate needed). "
                            "With αi, αf and λ of the cut: direct and exit-reflected path, with a free effective "
                            "reflection coefficient r = |r|·e^(i·arg r); free |r| and arg r to fit it. "
                            "With a known substrate as well: the four-term DWBA. Upright cylinders are not affected."))
        layout.addWidget(title)
        layout.addLayout(self._grid(card, GRAZING, "globals"))
        self._substrate = SubstrateControls(self, card)
        layout.addLayout(self._substrate.layout)
        return card

    def row(self, path: tuple) -> Optional[ParameterRow]:
        return self._rows.get(path)


__all__ = ["FitModelEditor"]
