"""The measured points of Single analysis as the plot draws them (no Qt)."""

from __future__ import annotations

from ...application.single_fit import prepare_curve

DATA_COLOR, SECOND_HALF_COLOR = "#2563eb", "#0891b2"
"""The fitted points; and the q < 0 half when both halves of a cut are shown on |q| (two colours)."""
BOTH_HALVES = (("measured (+q)", "positive", DATA_COLOR), ("measured (−q)", "negative", SECOND_HALF_COLOR))
"""Names without “<”: a plot's legend reads its entries as rich text."""


def measured_series(session, data) -> list[tuple]:
    """``(name, q, I, colour)`` of the points that are fitted (``data``: ``session.data()``): one series —
    or, for both halves of a cut on |q|, each half as its own series in its own colour, so that a difference
    between the halves is seen (they are fitted together)."""
    if data is None or not data.q.size:
        return []
    curve = session.curve
    if session.side == "both" and curve is not None and curve.signed:
        try:
            halves = [(name, prepare_curve(curve, side, session.q_range, session.excluded), color)
                      for name, side, color in BOTH_HALVES]
        except ValueError:
            halves = []
        if halves and sum(half.q.size for _name, half, _color in halves) == data.q.size:
            return [(name, half.q, half.intensity, color) for name, half, color in halves if half.q.size]
    return [("measured", data.q, data.intensity, DATA_COLOR)]


__all__ = ["BOTH_HALVES", "DATA_COLOR", "SECOND_HALF_COLOR", "measured_series"]
