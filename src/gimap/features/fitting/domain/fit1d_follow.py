"""Following the previous frame of a series with fit1d: among solutions that fit equally well, the one that continues
the previous frame.

One curve of a series often has several solutions that fit it equally well (another R with another D, a broad
distribution against a second population). Fitted one by one, a series then jumps between them from frame to frame.
The kit's own series tracker (``research.series1d.tracker``) keeps the solution that continues the previous frame
unless a fresh one is clearly better: lower χ² by more than the same-basin tolerance max(25, 0.02 N)·max(1, χ²/N).
Its tracker works on detector frames only, so for a folder of curves the same rule is applied here:

* the frame is fitted freely (the composition and constraints of the series), and once more with every parameter of
  every population kept in a window around the previous frame's value (``window``: R and D within × / ÷ ``WINDOW``,
  the relative widths within ± ``WINDOW``; the user's fixed values stay, a user range is intersected) — a window
  edge is resolved like a default range (``fit1d_edges``: widened when a value stops there);
* the continuing solution is kept when its χ² is at most the free one's plus the tolerance (``same_basin``);
  else the free one is: the sample changed (``JUMPED``).
"""

from __future__ import annotations

from typing import Iterable

from .fit1d_bounds import LIMITS, Constraint, Note

WINDOW = {"R_nm": 1.5, "D_nm": 1.5, "sigma_R_rel": 0.15, "w_rel": 0.15}
FOLLOWED = ("Followed the previous frame: its continuation fits as well as a free fit (χ² {follow} against {free}, "
            "tolerance {tolerance}).")
JUMPED = ("Did not follow the previous frame: a free fit is better by more than the tolerance (χ² {free} against "
          "{follow}, tolerance {tolerance}) — the sample changed here.")


def window(previous: Iterable[dict], constraints: Iterable[Constraint]) -> list[Constraint]:
    """``constraints`` with a window around every value of the previous frame's populations (the kit's result
    ``populations``: ``name``, ``R_nm``, ``sigma_R_rel``, ``D_nm``, ``w_rel``, ``inactive``)."""
    given = {item.name: item for item in constraints}
    out = list(given.values())
    for population in previous or ():
        if population.get("inactive"):
            continue
        for key, width in WINDOW.items():
            value = population.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            name = f"{population['name']}.{key}"
            mine = given.get(name)
            if mine is not None and mine.kind == "fixed":
                continue
            lowest, highest = LIMITS[key]
            low, high = (value / width, value * width) if key in ("R_nm", "D_nm") else (value - width, value + width)
            low, high = max(low, lowest), min(high, highest)
            component = None
            if mine is not None:
                component = mine.component
                if max(low, mine.low) < min(high, mine.high):
                    low, high = max(low, mine.low), min(high, mine.high)
                else:
                    continue  # the window lies outside the user's range: the user's range stays
            out = [item for item in out if item.name != name]
            out.append(Constraint(population["name"], key, "range", low, high, component, "follow"))
    return out


def tolerance(chi2_free: float, points: int) -> float:
    """The same-basin tolerance of the kit's tracker: max(25, 0.02 N)·max(1, χ²/N)."""
    points = max(int(points), 1)
    return max(25.0, 0.02 * points) * max(1.0, chi2_free / points)


def same_basin(chi2_free: float, chi2_follow: float, points: int) -> bool:
    return chi2_follow - chi2_free <= tolerance(chi2_free, points)


def follow_note(chi2_free: float, chi2_follow: float, points: int) -> Note:
    values = dict(free=f"{chi2_free:.4g}", follow=f"{chi2_follow:.4g}",
                  tolerance=f"{tolerance(chi2_free, points):.3g}")
    return Note(FOLLOWED if same_basin(chi2_free, chi2_follow, points) else JUMPED, values)


__all__ = ["FOLLOWED", "JUMPED", "WINDOW", "follow_note", "same_basin", "tolerance", "window"]
