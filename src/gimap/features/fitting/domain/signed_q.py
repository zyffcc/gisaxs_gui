"""Explicit preparation rules for signed q branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


QBranch = Literal["both", "positive", "negative"]
QCombination = Literal["separate", "fold", "average"]


@dataclass(frozen=True)
class SignedQPreparation:
    """A traceable curve produced from a signed-q source curve."""

    q: np.ndarray
    intensity: np.ndarray
    branch: QBranch
    combination: QCombination
    source_point_count: int
    source_sign: np.ndarray | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        intensity = np.asarray(self.intensity, dtype=float).reshape(-1)
        if q.size != intensity.size:
            raise ValueError("Prepared q and intensity must have the same length")
        if q.size < 2:
            raise ValueError("At least two prepared q points are required")
        source_sign = self.source_sign
        if source_sign is None:
            source_sign = np.sign(q)
        source_sign = np.asarray(source_sign, dtype=np.int8).reshape(-1)
        if source_sign.size != q.size:
            raise ValueError("Prepared q and source_sign must have the same length")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "source_sign", source_sign)


def _finite_sorted_pairs(
    q_values, intensity_values
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    q = np.asarray(q_values, dtype=float).reshape(-1)
    intensity = np.asarray(intensity_values, dtype=float).reshape(-1)
    count = min(q.size, intensity.size)
    q = q[:count]
    intensity = intensity[:count]
    finite = np.isfinite(q) & np.isfinite(intensity)
    q = q[finite]
    intensity = intensity[finite]
    source_sign = np.sign(q).astype(np.int8, copy=False)
    order = np.argsort(q, kind="mergesort")
    return q[order], intensity[order], source_sign[order], count


def _average_branches(q: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = q > 0
    negative = q < 0
    q_positive = q[positive]
    i_positive = intensity[positive]
    q_negative = np.abs(q[negative])[::-1]
    i_negative = intensity[negative][::-1]
    if q_positive.size < 2 or q_negative.size < 2:
        raise ValueError("Average ±q requires at least two points on each branch")

    lower = max(float(q_positive.min()), float(q_negative.min()))
    upper = min(float(q_positive.max()), float(q_negative.max()))
    if not lower < upper:
        raise ValueError("Positive and negative q branches do not overlap")
    coordinates = np.unique(
        np.concatenate(
            (
                q_positive[(q_positive >= lower) & (q_positive <= upper)],
                q_negative[(q_negative >= lower) & (q_negative <= upper)],
            )
        )
    )
    if coordinates.size < 2:
        raise ValueError("Average ±q overlap contains fewer than two coordinates")
    averaged = 0.5 * (
        np.interp(coordinates, q_positive, i_positive)
        + np.interp(coordinates, q_negative, i_negative)
    )
    return coordinates, averaged


def prepare_signed_q_curve(
    q_values,
    intensity_values,
    *,
    branch: QBranch = "both",
    combination: QCombination = "separate",
) -> SignedQPreparation:
    """Select and combine q branches without depending on plotting or Qt.

    ``separate`` preserves the sign of q. ``fold`` maps selected points to
    ``abs(q)`` while preserving both measurements. ``average`` interpolates the
    positive and mirrored negative branches on their shared domain and averages
    their intensities.
    """

    if branch not in ("both", "positive", "negative"):
        raise ValueError(f"Unsupported q branch: {branch}")
    if combination not in ("separate", "fold", "average"):
        raise ValueError(f"Unsupported q combination: {combination}")
    if combination == "average" and branch != "both":
        raise ValueError("Average ±q requires the Both branches selection")

    q, intensity, source_sign, source_count = _finite_sorted_pairs(
        q_values, intensity_values
    )
    if branch == "positive":
        keep = q > 0
    elif branch == "negative":
        keep = q < 0
    else:
        keep = np.ones(q.shape, dtype=bool)
    q = q[keep]
    intensity = intensity[keep]
    source_sign = source_sign[keep]

    if combination == "average":
        q, intensity = _average_branches(q, intensity)
        source_sign = np.zeros(q.shape, dtype=np.int8)
    elif combination == "fold":
        nonzero = q != 0
        q = np.abs(q[nonzero])
        intensity = intensity[nonzero]
        source_sign = source_sign[nonzero]
        order = np.argsort(q, kind="mergesort")
        q = q[order]
        intensity = intensity[order]
        source_sign = source_sign[order]

    if q.size < 2:
        raise ValueError(
            f"Not enough points after q preparation ({branch}, {combination})"
        )
    return SignedQPreparation(
        q,
        intensity,
        branch,
        combination,
        source_count,
        source_sign,
    )


__all__ = [
    "QBranch",
    "QCombination",
    "SignedQPreparation",
    "prepare_signed_q_curve",
]
