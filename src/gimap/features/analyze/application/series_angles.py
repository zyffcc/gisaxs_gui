"""Rows of a series measured at different incidence angles: one map per angle.

A map stacks one cut of every frame at the same detector rows. When the frames of a run switch between incidence
angles (a reference angle and short looks at others between spray pulses), those rows are another exit angle — and
another q_z — at every αi, so rows of different angles do not belong in one picture. The angle of a row is its
nominal αi (``incidence_record.nominal_incidence``: typed, from the file header or side file, from the profile).
Angles closer than ``TOLERANCE_DEG`` are one angle (a motor reads 0.399998 for 0.4).

By default a map holds the angle most of its rows were measured at, when that angle has at least two rows and the
rows have more than one angle; a scan through αi (every angle once) stays one map of all its rows.
"""

from __future__ import annotations

from typing import Optional, Sequence

TOLERANCE_DEG = 0.005
ALL = "all"
"""The choice of a map of every row, whatever its angle."""
INCIDENCE_LINE = "frames at alpha_i = {angle:g} deg only; {n} frames at other angles left out ({others})"
"""What a series export says of a map of one angle."""


def row_incidence(row) -> Optional[float]:
    """The nominal αi of a row ``(x, intensity, x_label, label, ref)`` (``RowLabel.incidence_deg``)."""
    value = getattr(row[3], "incidence_deg", None)
    return None if value is None else float(value)


def same_incidence(first: Optional[float], second: Optional[float]) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return abs(float(first) - float(second)) <= TOLERANCE_DEG


def incidence_groups(rows: Sequence) -> list[tuple[Optional[float], int]]:
    """``(angle, rows)`` of every angle among ``rows``, by angle (rows without one last)."""
    groups: list[list] = []
    for row in rows:
        angle = row_incidence(row)
        for group in groups:
            if same_incidence(group[0], angle):
                group[1] += 1
                break
        else:
            groups.append([angle, 1])
    return sorted(((angle, count) for angle, count in groups), key=lambda group: (group[0] is None, group[0] or 0.0))


def default_incidence(rows: Sequence):
    """The angle a map of ``rows`` holds unless one is chosen: the angle of most rows when the rows have several
    angles and that one has two rows or more (ties: the first row's), else ``ALL``."""
    groups = incidence_groups(rows)
    if len(groups) < 2:
        return ALL
    most = max(count for _angle, count in groups)
    if most < 2:
        return ALL
    first = row_incidence(rows[0])
    tied = [angle for angle, count in groups if count == most]
    return next((angle for angle in tied if same_incidence(angle, first)), tied[0])


def split_rows(rows: Sequence, angle) -> tuple[list, list]:
    """``(kept, aside)``: the rows at ``angle`` (``ALL``: every row), and the others."""
    if angle == ALL:
        return list(rows), []
    kept, aside = [], []
    for row in rows:
        (kept if same_incidence(row_incidence(row), angle) else aside).append(row)
    return kept, aside


def incidence_line(angle, aside: Sequence) -> str:
    """The export line of a map of one angle (empty for a map of every row)."""
    if angle == ALL or angle is None or not aside:
        return ""
    others = ", ".join(f"{value:g} deg: {count}" if value is not None else f"no angle: {count}"
                       for value, count in incidence_groups(aside))
    return INCIDENCE_LINE.format(angle=float(angle), n=len(aside), others=others)


__all__ = ["ALL", "TOLERANCE_DEG", "default_incidence", "incidence_groups", "incidence_line", "row_incidence",
           "same_incidence", "split_rows"]
