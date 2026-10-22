"""The unit of q a curve file is read in: one rule for all of Fitting — Single analysis, In-situ series and 1D Predict.

The unit the file's header states wins (``CurveRepository.stated_q_unit``: Analyze's ``_fit_input.dat`` state 1/A);
the setting “q unit of files without one in their header” is for the files that state none. A header that states
both units is refused (``ValueError``): such a file is neither opened nor fitted. Where the unit came from is kept
with the curve (``CurveData.q_unit_from``, ``Curve.unit_from``) and written into the records, in English; the pages
show it translated.
"""

from __future__ import annotations

from typing import Callable, Optional

FROM_FILE = "stated in the file"
FROM_SETTING = "not stated in the file: the setting for files without one"
FROM_FRAME = "cut from a detector frame as Analyze cuts it"
FROM_REQUEST = "the current curve, sent in this unit"
"""Where the unit a curve's q was read in comes from (``q_unit_from`` of curves and records)."""


def q_unit_read(stated_unit: Optional[Callable], path, setting: str) -> tuple[str, str]:
    """``(unit, where it came from)`` of the file ``path``: the unit its header states (``FROM_FILE``), else
    ``setting`` (``FROM_SETTING``). ``stated_unit`` reads the header (``"angstrom"`` / ``"nm"`` / ``None``; it raises
    ``OSError`` for a file that cannot be read and ``ValueError`` for one that states both units); ``None``: no
    reader, the setting."""
    stated = None if stated_unit is None else stated_unit(path)
    return (setting, FROM_SETTING) if stated is None else (stated, FROM_FILE)


__all__ = ["FROM_FILE", "FROM_FRAME", "FROM_REQUEST", "FROM_SETTING", "q_unit_read"]
