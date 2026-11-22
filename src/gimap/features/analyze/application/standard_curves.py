"""Read the measured and reference I(q) tables for an absolute-scale fit."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np

from src.gimap.shared.nxcansas import q_unit


def read_standard_curve(path: Path, unit: str, *, reference: bool = False):
    """Read q, I and optional σ (required for reference files). GIMaP's fourth column is pixel count."""
    rows = []
    gimap_table = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#"):
            if line.startswith("# GIMaP Analyze"):
                gimap_table = True
            if not reference and line.startswith("# intensity:") and "normalised I =" in line:
                raise ValueError("Export the standard curve again with frame normalisation off.")
            if gimap_table and line.startswith("# columns:") and not line.startswith("# columns: q ("):
                raise ValueError("Choose a radial I(q) curve with q as the first column.")
            continue
        if line:
            rows.append(line)
    if not rows:
        raise ValueError("The curve file contains no data.")
    delimiter = "," if "," in rows[0] else None
    first = rows[0].split(delimiter)[0].strip()
    try:
        float(first)
    except ValueError:
        # q or q with a unit, including the q_(1/A) of GIMaP's space-separated export.
        if not re.match(r"^q(?:$|\s|\(|/|\[|_(?:\(|1|nm|ang|a[-^]|å))", first, flags=re.IGNORECASE):
            raise ValueError("Choose a radial I(q) curve with q as the first column.") from None
        rows = rows[1:]
    if not rows:
        raise ValueError("The curve file contains no data.")
    if gimap_table and q_unit(unit) != "angstrom":
        raise ValueError("GIMaP radial curves use Å⁻¹. Select that q unit.")
    try:
        table = np.loadtxt(StringIO("\n".join(rows)), delimiter=delimiter, ndmin=2)
    except ValueError:
        raise ValueError("Check the numeric columns and separators in the curve file.") from None
    if reference and table.shape[1] != 3:
        raise ValueError("The reference file needs three columns for q, intensity and standard uncertainty.")
    if not reference and table.shape[1] not in (2, 3, 4):
        raise ValueError("The measured curve needs q and intensity, with optional uncertainty and pixel count.")
    if table.shape[1] == 4:
        table = table[table[:, 3] > 0]
    units = q_unit(unit)
    if units not in ("nm", "angstrom"):
        raise ValueError("Select Å⁻¹ or nm⁻¹ for the q unit.")
    q = table[:, 0] * (0.1 if units == "nm" else 1.0)
    return q, table[:, 1], table[:, 2] if table.shape[1] >= 3 else None
