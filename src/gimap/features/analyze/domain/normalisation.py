"""Normalisation of a frame: exposure time, monitor, transmission, thickness and an absolute scale.

After every other correction each pixel is multiplied by one constant of the frame,

    I_norm = A · I / (t · M · T · d),

``t`` the exposure time (s), ``M`` a monitor value (counts of an ion chamber or diode, a flux or the ring current),
``T`` the transmission of the sample (0 < T ≤ 1), ``d`` its thickness and ``A`` the scale factor. Thickness is entered
in mm and converted to cm for a secondary-standard calibration. Each term is optional (1 when it is off);
none is on by default. ``t`` and ``M`` are taken from the file header or typed; ``T`` and ``d`` are typed. A
sum of frames is divided by the sum of their exposure times and of their monitor values.

The normalisation only scales: q, the shape of every curve and the error model do not change. For photon counts
the variance of a pixel multiplied by ``c`` is ``c × value`` (Poisson carried through the factor, as for the flat
field: ``BinnedMean.add``); a frame whose σ comes from the scatter of its pixels keeps that model, and the scatter
scales with ``c`` by itself. Either way σ of every point is ``c`` times what it was.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

OFF, HEADER, VALUE = "off", "header", "value"
SOURCES = (OFF, HEADER, VALUE)
EXPOSURE_KEY = "exposure_time_s"
"""The reader's name of the exposure time in a frame's metadata (CBF, EDF, NeXus)."""
APPLIED_AS = ("I_norm = A * I / (t * M * T * d), after every other correction (one constant per frame); "
              "q unchanged; sigma multiplied by the same constant (photon counts: Poisson variance c * value)")
NO_EXPOSURE = "The header of {name} gives no exposure time: type it or turn the exposure off."
NO_MONITOR = "The header of {name} has no monitor value “{key}”: choose another or type it."
BAD_VALUE = "{what} must be a positive number."
BAD_TRANSMISSION = "The transmission must be above 0 and at most 1."
NEED_THICKNESS = "Enter the sample thickness to use a standard calibration."
ABSOLUTE_UNIT = "cm⁻¹"
"""Why a frame cannot be normalised (templates: ``presentation/texts.py`` translates them)."""

_MONITOR_NAME = re.compile(
    r"^(?:monitor\w*|mon\d*|i0|i_0|io|ion_?chamber\w*|intensity[01]|flux|photon_?flux|beam_?current|"
    r"ring_?current|sr_?current|srcur|diode\w*|pin_?diode\w*|count_?monitor)$", re.IGNORECASE)
"""Header names that can hold a monitor value (ESRF SAXS keywords, Pilatus ``Flux``, ion chambers, ring current)."""
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_CBF_LINE = re.compile(rf"^[ \t]*#[ \t]*([A-Za-z][\w.\-]*)[ \t:=]+({_NUMBER})", re.MULTILINE)


def mm_to_cm(thickness_mm: float) -> float:
    return float(thickness_mm) / 10.0


@dataclass(frozen=True)
class Normalisation:
    """What a frame is divided by; the defaults change nothing."""

    exposure: str = OFF
    """``off``, ``header`` (the exposure time the file states) or ``value`` (``exposure_s``)."""
    exposure_s: Optional[float] = None
    monitor: str = OFF
    """``off``, ``header`` (the header value named ``monitor_key``) or ``value`` (``monitor_value``)."""
    monitor_key: Optional[str] = None
    monitor_value: Optional[float] = None
    transmission: Optional[float] = None
    thickness_mm: Optional[float] = None
    absolute_factor: Optional[float] = None
    standard_calibration: Optional[dict] = field(default=None, hash=False)

    def __post_init__(self) -> None:
        for name in ("exposure", "monitor"):
            if getattr(self, name) not in SOURCES:
                raise ValueError(f"Unknown {name} source {getattr(self, name)!r}; expected one of {SOURCES}.")
        if self.transmission is not None and not 0.0 < float(self.transmission) <= 1.0:
            raise ValueError(BAD_TRANSMISSION)
        for what, value in (("The exposure time", self.exposure_s), ("The monitor value", self.monitor_value),
                            ("The thickness", self.thickness_mm), ("The absolute-scale factor", self.absolute_factor)):
            if value is not None and not (math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(BAD_VALUE.format(what=what))
        if self.exposure == VALUE and self.exposure_s is None:
            raise ValueError(BAD_VALUE.format(what="The exposure time"))
        if self.monitor == VALUE and self.monitor_value is None:
            raise ValueError(BAD_VALUE.format(what="The monitor value"))
        if self.monitor == HEADER and not self.monitor_key:
            raise ValueError("A monitor from the header needs the name of its header value.")
        if self.standard_calibration is not None:
            record = self.standard_calibration
            if self.thickness_mm is None:
                raise ValueError(NEED_THICKNESS)
            if (not isinstance(record, dict) or self.absolute_factor is None
                    or record.get("factor") != self.absolute_factor
                    or not isinstance(record.get("sigma_factor"), (int, float))
                    or not math.isfinite(record["sigma_factor"]) or record["sigma_factor"] < 0
                    or not all(record.get(key) for key in ("standard", "source_path", "q_range_A_inv", "date_utc"))):
                raise ValueError("Invalid standard calibration.")

    @property
    def is_identity(self) -> bool:
        return (self.exposure == OFF and self.monitor == OFF and self.transmission is None
                and self.thickness_mm is None and self.absolute_factor is None)


def _number(text: Any) -> Optional[float]:
    try:
        value = float(str(text).split()[0])
    except (ValueError, IndexError):
        return None
    return value if math.isfinite(value) else None


def header_values(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """What the header of a frame offers for the normalisation: ``{"exposure_time_s": t or None, "monitors":
    {name: value}}``. Monitors: a NeXus monitor the reader found (named by its place in the file) and the header
    values whose names hold monitors (EDF keywords, ``# Flux`` of a Pilatus CBF)."""
    metadata = metadata or {}
    exposure = _number(metadata.get(EXPOSURE_KEY)) if metadata.get(EXPOSURE_KEY) is not None else None
    monitors: dict[str, float] = {}
    if metadata.get("monitor") is not None and _number(metadata["monitor"]) is not None:
        monitors[str(metadata.get("monitor_source") or "monitor")] = _number(metadata["monitor"])
    header = metadata.get("header") or {}
    if isinstance(header, Mapping):
        for key, text in header.items():
            if _MONITOR_NAME.match(str(key)) and _number(text) is not None:
                monitors[str(key)] = _number(text)
        contents = str(header.get("_array_data.header_contents", ""))  # a mini-CBF: “# Flux 1.2e12”
        for key, text in _CBF_LINE.findall(contents):
            if _MONITOR_NAME.match(key) and _number(text) is not None:
                monitors.setdefault(key, _number(text))
    return {EXPOSURE_KEY: exposure if exposure and exposure > 0 else None, "monitors": monitors}


def summed_header_values(values: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The header values of a sum of frames: exposure times and monitors added; a value that one of the frames
    lacks is not there for the sum."""
    if not values:
        return {EXPOSURE_KEY: None, "monitors": {}}
    exposures = [item.get(EXPOSURE_KEY) for item in values]
    exposure = None if any(value is None for value in exposures) else float(sum(exposures))
    keys = set.intersection(*(set((item.get("monitors") or {}).keys()) for item in values))
    monitors = {key: float(sum(item["monitors"][key] for item in values)) for key in sorted(keys)}
    return {EXPOSURE_KEY: exposure, "monitors": monitors}


@dataclass(frozen=True)
class NormalisationUse:
    """The constant a frame was multiplied by, each term with its value and where it came from."""

    factor: float
    terms: tuple[tuple[str, float, str], ...]
    """Each applied (name, value, source). The thickness term is in cm for a fitted standard, mm otherwise."""
    unit: str
    """The unit of the normalised intensity (``intensity_unit``)."""
    standard_calibration: Optional[dict] = field(default=None, hash=False)

    def term(self, name: str) -> Optional[tuple[float, str]]:
        return next(((value, source) for key, value, source in self.terms if key == name), None)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"applied_as": APPLIED_AS, "factor": float(self.factor), "intensity_unit": self.unit}
        for name, value, source in self.terms:
            record[name] = {"value": float(value), "source": source}
        if self.standard_calibration is not None:
            record["standard_calibration"] = self.standard_calibration
        return record


def intensity_unit(norm: Normalisation) -> str:
    """Absolute cross section for a measured calibration, or the units of the selected manual terms."""
    if norm.standard_calibration is not None:
        return ABSOLUTE_UNIT
    per = ["pixel"]
    if norm.exposure != OFF:
        per.append("s")
    if norm.monitor != OFF:
        per.append("monitor")
    if norm.thickness_mm is not None:
        per.append("mm")
    unit = "counts/pixel" if len(per) == 1 else "counts/(" + "·".join(per) + ")"
    return unit if norm.absolute_factor is None else f"{unit} × {float(norm.absolute_factor):.6g}"


def normalisation_use(norm: Normalisation, header: Mapping[str, Any], name: str = "this frame") -> NormalisationUse:
    """The constant of a frame (``header``: ``header_values`` of the frame, or ``summed_header_values`` of a sum).
    ``ValueError`` (``NO_EXPOSURE``, ``NO_MONITOR``) when the header lacks a value it is to be taken from."""
    terms: list[tuple[str, float, str]] = []
    if norm.exposure == HEADER:
        exposure = header.get(EXPOSURE_KEY)
        if not exposure:
            raise ValueError(NO_EXPOSURE.format(name=name))
        terms.append(("exposure_time_s", float(exposure), "header (exposure_time_s)"))
    elif norm.exposure == VALUE:
        terms.append(("exposure_time_s", float(norm.exposure_s), "typed"))
    if norm.monitor == HEADER:
        value = (header.get("monitors") or {}).get(norm.monitor_key)
        if value is None or not value > 0:
            raise ValueError(NO_MONITOR.format(name=name, key=norm.monitor_key))
        terms.append(("monitor", float(value), f"header ({norm.monitor_key})"))
    elif norm.monitor == VALUE:
        terms.append(("monitor", float(norm.monitor_value), "typed"))
    if norm.transmission is not None:
        terms.append(("transmission", float(norm.transmission), "typed"))
    if norm.thickness_mm is not None:
        if norm.standard_calibration is None:
            terms.append(("thickness_mm", float(norm.thickness_mm), "typed"))
        else:
            terms.append(("thickness_cm", mm_to_cm(norm.thickness_mm), "typed"))
    if norm.absolute_factor is not None:
        source = "typed" if norm.standard_calibration is None else "secondary standard"
        terms.append(("absolute_factor", float(norm.absolute_factor), source))
    divisor = math.prod(value for key, value, _source in terms if key != "absolute_factor")
    factor = float(norm.absolute_factor or 1.0) / divisor
    return NormalisationUse(factor, tuple(terms), intensity_unit(norm), norm.standard_calibration)


NORMALISATION_TEMPLATES = (NO_EXPOSURE, NO_MONITOR, BAD_VALUE, BAD_TRANSMISSION)

__all__ = [
    "ABSOLUTE_UNIT", "APPLIED_AS", "BAD_TRANSMISSION", "BAD_VALUE", "EXPOSURE_KEY", "HEADER", "NORMALISATION_TEMPLATES", "NO_EXPOSURE",
    "NO_MONITOR", "Normalisation", "NormalisationUse", "OFF", "SOURCES", "VALUE", "header_values", "intensity_unit",
    "mm_to_cm", "normalisation_use", "summed_header_values",
]
