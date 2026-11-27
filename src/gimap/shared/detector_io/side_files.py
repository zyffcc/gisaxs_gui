"""The text file a beamline writes beside each image with the motors and counters of that exposure.

ALS 7.3.3 writes ``<image stem>.txt`` next to every EDF: three numbers, then ``Motors: n`` and ``name: value`` lines
for the motors, the digital inputs (``DIOs``) and the analog inputs and counters (``AIs``). The image header itself
holds no incidence angle and no monitor, so they are read from there:

* ``Sample Alpha Stage`` — the grazing incidence angle αi (°), as the motor reads it;
* ``Izero``, ``I1``, ``GiSAXS Beamstop Counter``, ``Beam Current`` — counters and the ring current, offered as
  monitors for the normalisation.

A ``.txt`` of the same stem is taken only when it has the ``Motors:`` line and one of these names, so a note of the
user that happens to share the image's name is not read as a side file.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

ALPHA = "Sample Alpha Stage"
MONITORS = ("Izero", "I1", "GiSAXS Beamstop Counter", "Beam Current")
_LINE = re.compile(r"^\s*([A-Za-z][^:=\n]*?)\s*:\s*(\S+)\s*$", re.MULTILINE)
_MOTORS = re.compile(r"^\s*Motors\s*:\s*\d+\s*$", re.MULTILINE)
MAX_BYTES = 256 * 1024


def side_file(image: Path) -> Optional[Path]:
    """The side file of ``image`` (``<stem>.txt`` in its folder), or ``None``."""
    for suffix in (".txt", ".TXT"):
        candidate = Path(image).with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _number(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_side_values(path: Path) -> Optional[dict[str, float]]:
    """The ``name: number`` lines of a side file, or ``None`` when it is not one (no ``Motors:`` line, none of the
    names above, too large or unreadable)."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        text = path.read_text(encoding="latin-1")
    except OSError:
        return None
    if not _MOTORS.search(text):
        return None
    values: dict[str, float] = {}
    for name, value in _LINE.findall(text):
        number = _number(value)
        if number is not None:
            values.setdefault(name.strip(), number)  # (a name listed twice: its first value)
    if ALPHA not in values and not any(name in values for name in MONITORS):
        return None
    return values


def side_file_metadata(image: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """What the side file of ``image`` adds to ``metadata``: ``side_files`` (its path, recorded with the frame's
    files), ``incidence_deg`` with its source (unless the image header gave one) and ``side_monitors``."""
    path = side_file(image)
    values = read_side_values(path) if path is not None else None
    if values is None:
        return {}
    added: dict[str, Any] = {"side_files": [str(path)]}
    alpha = values.get(ALPHA)
    if metadata.get("incidence_deg") is None and alpha is not None and 0.0 < alpha < 90.0:
        added["incidence_deg"] = alpha
        added["incidence_source"] = f"side file {path.name}: {ALPHA}"
    monitors = {name: values[name] for name in MONITORS if name in values}
    if monitors:
        added["side_monitors"] = monitors
    return added


__all__ = ["ALPHA", "MONITORS", "read_side_values", "side_file", "side_file_metadata"]
