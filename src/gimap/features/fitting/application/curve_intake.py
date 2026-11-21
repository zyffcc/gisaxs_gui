"""What Fitting says of a curve as it takes it in, the same on the Fit page, in the 1D Predict job and in the
analyses that fit one cut: the unit of q it assumed when the file states none (and whether the q range that gives is
plausible), the σ it estimated for a file without one, and the points with a negative or zero intensity.

Every remark is an ``IntakeNote``: a ``kind``, an English ``template`` with its ``values`` (records and logs take
``text``; the pages translate the template). The σ of a curve without one is one estimate everywhere
(``relative_sigma``: the proposal network's own, ``domain.posterior.infer``): a relative error from the scatter of
ln I about its local trend, at least 1 %. ln I needs I > 0, so such a curve is fitted without its negative and zero
points; with a σ in the file they stay, weighted by it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..domain.posterior.infer import _noise
from .curve_units import FROM_SETTING

Q_MAX_PLAUSIBLE = 10.0
"""nm⁻¹ (1 Å⁻¹): a GISAXS / SAXS cut does not reach beyond it."""
Q_MAX_TOO_SMALL = 0.02
"""nm⁻¹ (0.002 Å⁻¹): a whole cut below it resolves nothing GIMaP's particles are."""
UNIT_NAMES = {"angstrom": "Å⁻¹", "nm": "nm⁻¹", "A^-1": "Å⁻¹", "nm^-1": "nm⁻¹"}
_OTHER = {"Å⁻¹": "nm⁻¹", "nm⁻¹": "Å⁻¹"}
_EXAMPLE = {"Å⁻¹": "# q in 1/A", "nm⁻¹": "# q in 1/nm"}

HOW_TO_CHANGE = ("Set “q unit of files without one in their header” to {other} (Fit page: Curve ▸ File; 1D Predict: "
                 "Parameters…), or state the unit in the file's header, for example “{example}”.")
UNIT_TOO_LARGE = ("The file states no unit of q and was read in {unit}: |q| then reaches {q_max} nm⁻¹ ({q_max_a} Å⁻¹), "
                  "beyond what a GISAXS or SAXS cut covers. If its q is in {other}, every length found (R, D) is 10× "
                  "too small. " + HOW_TO_CHANGE)
UNIT_TOO_SMALL = ("The file states no unit of q and was read in {unit}: |q| then reaches only {q_max} nm⁻¹ "
                  "({q_max_a} Å⁻¹), less than a GISAXS or SAXS cut covers. If its q is in {other}, every length found "
                  "(R, D) is 10× too large. " + HOW_TO_CHANGE)
SIGMA_ESTIMATED = ("No σ in the file: a relative σ of {percent} % is used, estimated from the scatter of ln I about "
                   "its trend")
NONPOSITIVE_KEPT = ("{count} of the {points} fitted points have a negative or zero intensity ({negative} negative, "
                    "{zero} zero): they are kept and weighted by their σ. The model cannot go below zero, so a "
                    "background subtracted too far shows as a poor χ².")
NONPOSITIVE_LEFT_OUT = ("{count} points with a negative or zero intensity ({negative} negative, {zero} zero) were "
                        "left out: without σ the fit is on ln I, which they do not have.")
ROWS_LEFT_OUT = "{count} rows of the file could not be read as numbers and were left out."


@dataclass(frozen=True)
class IntakeNote:
    kind: str
    """``q_unit`` / ``sigma_estimated`` / ``nonpositive`` / ``unreadable_rows``."""
    template: str
    values: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.template.format(**self.values)

    def as_dict(self) -> dict:
        """As a record keeps it: ``kind``, ``text`` and the values (counts and numbers as data)."""
        return {"kind": self.kind, "text": self.text, **self.values}


def relative_sigma(q, intensity) -> np.ndarray:
    """σ of the points of a curve given without one (|q| > 0, I > 0): Predict parameters' own estimate."""
    q, y = np.abs(np.asarray(q, float)).ravel(), np.asarray(intensity, float).ravel()
    return _noise(q, y)


def sigma_level(q, intensity) -> Optional[float]:
    """The relative σ ``relative_sigma`` gives these points (one number: σ_i = level · I_i); None without points."""
    q, y = np.abs(np.asarray(q, float)).ravel(), np.asarray(intensity, float).ravel()
    keep = np.isfinite(q) & np.isfinite(y) & (q > 0) & (y > 0)
    if not keep.any():
        return None
    return float(np.median(_noise(q[keep], y[keep]) / y[keep]))


def nonpositive(intensity) -> tuple[int, int]:
    """(negative, zero) among the finite intensities."""
    y = np.asarray(intensity, float).ravel()
    y = y[np.isfinite(y)]
    return int(np.count_nonzero(y < 0)), int(np.count_nonzero(y == 0))


def unit_note(q_nm, unit: str, unit_from: str) -> Optional[IntakeNote]:
    """The remark on a unit the file did not state (``unit_from`` is ``FROM_SETTING``) when the q range it gives is
    implausible for a GISAXS / SAXS cut; ``q_nm``: the curve's q in nm⁻¹ as read."""
    if unit_from != FROM_SETTING or unit not in UNIT_NAMES:
        return None
    q = np.abs(np.asarray(q_nm, float).ravel())
    q = q[np.isfinite(q)]
    if not q.size:
        return None
    q_max = float(q.max())
    if Q_MAX_TOO_SMALL <= q_max <= Q_MAX_PLAUSIBLE:
        return None
    shown = UNIT_NAMES[unit]
    other = _OTHER[shown]
    return IntakeNote("q_unit", UNIT_TOO_LARGE if q_max > Q_MAX_PLAUSIBLE else UNIT_TOO_SMALL,
                      dict(unit=shown, other=other, q_max=f"{q_max:.3g}", q_max_a=f"{q_max / 10:.3g}",
                           example=_EXAMPLE[other]))


def sigma_note(level: Optional[float]) -> Optional[IntakeNote]:
    """The remark on an estimated σ (``level``: ``sigma_level``); None when the file has σ."""
    if level is None:
        return None
    return IntakeNote("sigma_estimated", SIGMA_ESTIMATED, dict(percent=f"{100 * level:.3g}"))


def nonpositive_note(negative: int, zero: int, points: int, *, left_out: bool) -> Optional[IntakeNote]:
    """The remark on negative and zero intensities: ``left_out`` (no σ) or kept among the ``points`` fitted."""
    count = int(negative) + int(zero)
    if not count:
        return None
    values = dict(count=count, negative=int(negative), zero=int(zero), points=int(points), left_out=bool(left_out))
    return IntakeNote("nonpositive", NONPOSITIVE_LEFT_OUT if left_out else NONPOSITIVE_KEPT, values)


def rows_note(count: int) -> Optional[IntakeNote]:
    return IntakeNote("unreadable_rows", ROWS_LEFT_OUT, dict(count=int(count))) if count else None


TEMPLATES = {"q_unit": (UNIT_TOO_LARGE, UNIT_TOO_SMALL), "sigma_estimated": (SIGMA_ESTIMATED,),
             "nonpositive": (NONPOSITIVE_LEFT_OUT, NONPOSITIVE_KEPT), "unreadable_rows": (ROWS_LEFT_OUT,)}


def note_from_dict(record) -> Optional[IntakeNote]:
    """The note a record's entry (``IntakeNote.as_dict``) was made of, so that a page can say it in its language;
    None for anything else."""
    if not isinstance(record, dict):
        return None
    values = {key: value for key, value in record.items() if key not in ("kind", "text")}
    for template in TEMPLATES.get(record.get("kind"), ()):
        try:
            if template.format(**values) == record.get("text"):
                return IntakeNote(record["kind"], template, values)
        except (KeyError, IndexError, ValueError):
            continue
    return None


def intake_notes(q_nm, *, unit: str = "", unit_from: str = "", level: Optional[float] = None, negative: int = 0,
                 zero: int = 0, points: int = 0, left_out: bool = False, unreadable: int = 0) -> list[IntakeNote]:
    """Every remark on a curve, the unit first (it changes every length)."""
    notes = [unit_note(q_nm, unit, unit_from), sigma_note(level),
             nonpositive_note(negative, zero, points, left_out=left_out), rows_note(unreadable)]
    return [note for note in notes if note is not None]


__all__ = ["HOW_TO_CHANGE", "IntakeNote", "NONPOSITIVE_KEPT", "NONPOSITIVE_LEFT_OUT", "Q_MAX_PLAUSIBLE",
           "Q_MAX_TOO_SMALL", "ROWS_LEFT_OUT", "SIGMA_ESTIMATED", "UNIT_TOO_LARGE", "UNIT_TOO_SMALL", "intake_notes",
           "nonpositive", "nonpositive_note", "note_from_dict", "relative_sigma", "rows_note", "sigma_level", "sigma_note", "unit_note"]
