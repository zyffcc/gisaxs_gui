"""Translate stored beam-centre conventions into the canonical one.

Canonical (see :class:`DetectorGeometry`): pixel corners at integers, row 0 at
the top, so the centre of pixel ``(row i, column j)`` is ``(j + 0.5, i + 0.5)``.

Other conventions still met in stored data:

* **index** – 0-based pixel-centre indices, row 0 at the top (``numpy`` indexing).
  Used by Calibration results and Trainset configs.
* **fitting** – 0-based pixel-centre indices with the row counted from the
  *bottom* of the image.  The former Cut & Fitting page stored its beam centre
  this way; Analyze converts such a stored geometry once into an instrument
  profile (“Use Previous Geometry”).
"""

from __future__ import annotations


def canonical_from_index_center(x_px: float, y_px: float) -> tuple[float, float]:
    return float(x_px) + 0.5, float(y_px) + 0.5


def index_center_from_canonical(x_px: float, y_px: float) -> tuple[float, float]:
    return float(x_px) - 0.5, float(y_px) - 0.5


def canonical_from_fitting_center(
    x_px: float, y_px_from_bottom: float, rows: int
) -> tuple[float, float]:
    return float(x_px) + 0.5, float(rows) - float(y_px_from_bottom) - 0.5
