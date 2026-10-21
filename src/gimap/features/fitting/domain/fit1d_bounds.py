"""The parameter ranges of fit1d: what one curve can measure, the Model's constraints, and parameters at an edge.

fit1d (the gisaxs1d kit) searches each population in a box tied to the curve's own q window
(``research.cond_v2.codec``): R in [0.5/q_max, 5/q_min], D in [2.002 R, 40 R], σR/R in [0.03, 0.5] and
w/D in [0.02, 0.5]. It accepts a fixed value or a range for any of them (``research.fit1d.constraints``:
``{"pop1": {"R_nm": 2.5}, "pop2": {"D_nm": [5, 8]}, "background": 0}``), populations numbered by type
(spheres before vertical cylinders, then in the order given).

Here:

* ``model_constraints``: the Model's fixed values and the ranges a user changed, in the kit's units —
  R, D in nm; σR/R as is; Fitting's σD/D is the kit's w/D divided by sqrt(2π) (the spread with the same
  paracrystal damping, as the adapter maps the kit's result back), so w/D = σD/D · sqrt(2π);
* ``range_notes``: before a fit, a user range reaching beyond what the curve can resolve (a narrower one needs
  no word: a value that stops at its edge is resolved by the fit, ``fit1d_edges``);
* what happens to a parameter that stops at an edge after the fit is in ``fit1d_edges``.

Messages are English templates with ``{fields}`` (``Note``); the interface translates the template.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Optional

from .fit_model import INFO, FitModel

SQRT_2PI = math.sqrt(2.0 * math.pi)
TYPE_CODE = {"sphere": 1, "vertical_cylinder": 3}
"""Fitting's family → the kit's type code (its populations are numbered in the order of these codes)."""
KIT_KEYS = {"R": "R_nm", "sigma_R": "sigma_R_rel", "D": "D_nm", "sigma_D": "w_rel"}
MODEL_KEYS = {kit: model for model, kit in KIT_KEYS.items()}
TO_KIT = {"sigma_D": SQRT_2PI}
"""Model value × factor = the kit's value (only σD/D differs: w/D = σD/D · sqrt(2π))."""
LIMITS = {"R_nm": (1e-3, 1e6), "D_nm": (1e-3, 1e7), "sigma_R_rel": (0.02, 0.6), "w_rel": (1e-3, 1.0)}
"""What the kit accepts as a fixed value or a range (``research.fit1d.constraints.LIMITS``)."""
WIDTHS = {"sigma_R_rel": (0.03, 0.5), "w_rel": (0.02, 0.5)}
"""The kit's default boxes of the relative widths (``research.cond_v2.codec``)."""
D_PER_R = (2.002, 40.0)
"""The kit's default box of D: from touching particles (2R) to 40 R."""
UNITS = {"R_nm": "nm", "D_nm": "nm", "sigma_R_rel": "", "w_rel": ""}
"""Units of the kit's parameters (a note puts a space before a unit that is there: ``{unit}`` is " nm" or "")."""
LABELS = {"R_nm": "R", "D_nm": "D", "sigma_R_rel": "σR/R", "w_rel": "w/D"}

BEYOND = ("{name}: your range reaches {side_value}{unit}, beyond what this curve can resolve "
          "({measurable_low}–{measurable_high}{unit}); values out there are not determined by it.")

def number(value: float) -> str:
    """Three significant digits, without an exponent for lengths up to 10⁶ nm."""
    return f"{value:.0f}" if 1000 <= abs(value) < 1e6 else f"{value:.3g}"


def unit_text(key: str) -> str:
    return f" {UNITS[key]}" if UNITS.get(key) else ""


@dataclass(frozen=True)
class Note:
    template: str
    values: dict

    @property
    def text(self) -> str:
        return self.template.format(**self.values)

    def to_dict(self) -> dict:
        return dict(template=self.template, values=dict(self.values), text=self.text)


@dataclass(frozen=True)
class Constraint:
    """One constraint passed to the kit. ``target``: ``pop1``, ``pop2``, … or ``background``; ``key``: the kit's
    parameter (``R_nm`` …, empty for the background); ``component``: the Model's component (None: global);
    ``source``: ``model`` (fixed or a range in Model) or ``widened`` (after a fit stopped at a default edge)."""

    target: str
    key: str
    kind: str
    low: float
    high: float
    component: Optional[int] = None
    source: str = "model"

    @property
    def name(self) -> str:
        return f"{self.target}.{self.key}" if self.key else self.target

    @property
    def text(self) -> str:
        unit = UNITS.get(self.key, "")
        if self.kind == "fixed":
            return f"{self.name} = {self.low:.4g} {unit}".strip()
        return f"{self.name} in {self.low:.4g}–{self.high:.4g} {unit}".strip()

    def to_dict(self) -> dict:
        return {**asdict(self), "name": self.name, "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Constraint":
        return cls(str(data["target"]), str(data.get("key") or ""), str(data["kind"]), float(data["low"]),
                   float(data["high"]), None if data.get("component") is None else int(data["component"]),
                   str(data.get("source") or "model"))


def population_order(families: Iterable[str]) -> list[int]:
    """The Model's component index of pop1, pop2, … (spheres first, then vertical cylinders, each in Model order)."""
    families = list(families)
    return sorted(range(len(families)), key=lambda index: (TYPE_CODE[families[index]], index))


def measurable(q_min: float, q_max: float) -> dict:
    """What a curve on [q_min, q_max] (nm⁻¹) can measure: R (the kit's default box) and D (the structure peak
    2π/D inside the window), in nm."""
    return {"R_nm": (0.5 / q_max, 5.0 / q_min), "D_nm": (2.0 * math.pi / q_max, 2.0 * math.pi / q_min)}


def _limited(key: str, name: str, low: float, high: float, kind: str) -> tuple[float, float]:
    lowest, highest = LIMITS[key]
    if kind == "fixed":
        if not lowest <= low <= highest:
            raise ValueError(f"{name} fixed at {low:.4g} is outside what fit1d accepts ({lowest:g}–{highest:g}): "
                             "free it or change the value")
        return low, low
    low, high = max(low, lowest), min(high, highest)  # a range is clipped to what the kit accepts
    if low >= high:
        raise ValueError(f"{name}: the range is outside what fit1d accepts ({lowest:g}–{highest:g})")
    return low, high


def model_constraints(model: FitModel, *, structure_factor: bool = True) -> list[Constraint]:
    """The Model's fixed values and the ranges a user changed (default ranges are left to the kit, which ties
    them to the curve). D and σD/D only with a structure factor (GISAXS), and only for a component that has one.
    Raises ValueError for a family fit1d has not or a fixed value it does not accept."""
    families = [component.family for component in model.components]
    if any(family not in TYPE_CODE for family in families):
        raise ValueError("fit1d needs 1–4 particles in Model: Sphere or Vertical cylinder")
    constraints = []
    for number, index in enumerate(population_order(families), start=1):
        component = model.components[index]
        keys = ["R", "sigma_R"] + (["D", "sigma_D"] if structure_factor and component.structure else [])
        for key in keys:
            parameter, info = component.params[key], INFO[key]
            factor = TO_KIT.get(key, 1.0)
            name = f"pop{number}.{KIT_KEYS[key]}"
            if not parameter.free:
                low, high = _limited(KIT_KEYS[key], name, parameter.value * factor, parameter.value * factor, "fixed")
                constraints.append(Constraint(f"pop{number}", KIT_KEYS[key], "fixed", low, high, index))
            elif (parameter.lower, parameter.upper) != (info.lower, info.upper):
                upper = parameter.upper if math.isfinite(parameter.upper) else LIMITS[KIT_KEYS[key]][1] / factor
                low, high = _limited(KIT_KEYS[key], name, parameter.lower * factor, upper * factor, "range")
                constraints.append(Constraint(f"pop{number}", KIT_KEYS[key], "range", low, high, index))
    background = model.globals["background"]
    if not background.free:
        if background.value < 0:
            raise ValueError("background fixed below 0: fit1d needs a background ≥ 0")
        constraints.append(Constraint("background", "", "fixed", background.value, background.value))
    return constraints


def as_kit(constraints: Iterable[Constraint]) -> dict:
    """The kit's ``constraints`` argument."""
    out: dict = {}
    for item in constraints:
        value = item.low if item.kind == "fixed" else [item.low, item.high]
        if item.target == "background":
            out["background"] = value
        else:
            out.setdefault(item.target, {})[item.key] = value
    return out


def range_notes(constraints: Iterable[Constraint], q_min: float, q_max: float) -> list[Note]:
    """Before a fit: a range of R or D narrower than the curve can measure, or reaching beyond it."""
    window = measurable(q_min, q_max)
    notes = []
    for item in constraints:
        if item.kind != "range" or item.key not in window:
            continue
        low, high = window[item.key]
        values = dict(name=item.name, low=number(item.low), high=number(item.high), unit=unit_text(item.key),
                      measurable_low=number(low), measurable_high=number(high))
        if item.high > 2 * high or item.low < low / 2:
            side = item.high if item.high > 2 * high else item.low
            notes.append(Note(BEYOND, {**values, "side_value": number(side)}))
    return notes


__all__ = [
    "BEYOND", "Constraint", "D_PER_R", "KIT_KEYS", "LABELS", "LIMITS", "MODEL_KEYS", "Note", "TO_KIT",
    "UNITS", "WIDTHS", "as_kit", "measurable", "model_constraints", "number", "population_order", "range_notes",
    "unit_text",
]
