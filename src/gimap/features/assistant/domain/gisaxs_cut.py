"""The horizontal GISAXS cut I(qy): which halves to use, the curve to fit, the dominant spacing.

Which halves and the curve of a fit are decided in ``shared/signed_cut.py`` — Fitting and Analyze prepare
their cuts with the same functions — and are named here as before (``choose_halves``, ``fit_curve``,
``HalvesChoice``). The spacing of the side maximum is the automatic analysis' own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.gimap.shared.signed_cut import (
    MAX_FIT_POINTS,
    MIN_COVERAGE,
    MIN_REACH,
    MISMATCH_LIMIT,
    SPECULAR_POINTS,
    HalvesChoice,
    choose_halves,
    fit_curve,
)

HALVES_SENTENCES = (
    "Only the {where} half is on the detector.",
    "The {side} half reaches only |qy| = {reach} Å⁻¹ (the {other} half {other_reach} Å⁻¹), so the {used} half is used.",
    "The {side} half has points in only {coverage} of its |qy| range (detector gaps or the beam-stop shadow; the "
    "{other} half: {other_coverage}), so the {used} half is used.",
    "The halves differ by {percent} % (median) up to |qy| = {q} Å⁻¹ even after the symmetry correction (a shadow, an "
    "absorber or real in-plane anisotropy): both are kept and the {better} half, the better covered one, is fitted.",
    "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both "
    "exist, which halves the noise{extend}.",
    "agree within {percent} %",
    "; beyond {q} Å⁻¹ the longer {side} half continues alone to {reach} Å⁻¹",
    "mean of both halves up to |qy| = {q} Å⁻¹",
    "then the {side} half alone to {q} Å⁻¹",
    "the {side} half",
    "{group} neighbouring points merged ({count} points for the fit)",
)
"""The sentences of ``choose_halves`` and ``fit_curve`` as the Results tab knows them: it shows a sentence of
one of these forms in the interface language (``presentation/guided_words.SENTENCES``), so the wording in
``shared/signed_cut.py`` has to stay in these forms (``tests/test_signed_cut.py`` compares them). The
sentences about a half that have no value in them ("the qy < 0 half has no points", "… is too sparse to
average with") need no form: they are entries of the translation table as they are written."""
HOLES_SENTENCE = ("{averaged} points averaged, {single} from one half alone ({inside} of them inside that range, "
                  "where the other half has no points)")
"""The part of ``fit_curve``'s note that counts the points of a mean when a half has a hole or begins later
than the other. Its translation is in the table and its form is listed in
``presentation/guided_words.SENTENCES``, so the Results tab says it in the interface language."""


@dataclass(frozen=True)
class Spacing:
    q: float
    """|qy| of the side maximum (Å⁻¹)."""
    distance_nm: float
    """2π / q: the dominant in-plane distance."""
    snr: float
    kind: str = "maximum"
    """``maximum`` (a resolved side peak) or ``shoulder`` (a change of slope only: a hint, not a measurement)."""


SHOULDER_MIN_SNR = 5.0


def spacing_from_peaks(peaks, *, q_min: float) -> Optional[Spacing]:
    """The strongest side maximum away from qy = 0 as a distance 2π/q; else the strongest shoulder."""
    side = [peak for peak in peaks if abs(peak.q) >= q_min and "spike" not in peak.flags]
    resolved = [peak for peak in side if "broad" not in peak.flags and "fit_failed" not in peak.flags]
    if resolved:
        best, kind = max(resolved, key=lambda peak: peak.snr), "maximum"
    else:
        shoulders = [peak for peak in side if peak.snr >= SHOULDER_MIN_SNR]
        if not shoulders:
            return None
        best, kind = max(shoulders, key=lambda peak: peak.snr), "shoulder"
    q = abs(float(best.q))
    return Spacing(q, 2.0 * math.pi / q / 10.0, float(best.snr), kind)


__all__ = ["HALVES_SENTENCES", "HOLES_SENTENCE", "HalvesChoice", "MAX_FIT_POINTS", "MIN_REACH", "MISMATCH_LIMIT", "MIN_COVERAGE",
           "SPECULAR_POINTS", "Spacing", "choose_halves", "fit_curve", "spacing_from_peaks"]
