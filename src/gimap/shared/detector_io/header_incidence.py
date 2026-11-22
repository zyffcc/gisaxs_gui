"""Sample incidence explicitly named in detector headers, normalized to degrees."""

import math
import re

import h5py
import numpy as np

from .nexus_datasets import _text


_UNITS = {"°": 1., "deg": 1., "degree": 1., "degrees": 1.,
          "rad": 180./math.pi, "radian": 180./math.pi, "radians": 180./math.pi,
          "mrad": .18/math.pi, "milliradian": .18/math.pi, "milliradians": .18/math.pi}
_CBF_LINE = re.compile(
    r"^[ \t]*#[ \t]*(?P<name>(?:grazing[ _])?inciden(?:ce|t)[ _]?angle|angle[ _]of[ _]incidence|alpha_i|α_i)"
    r"[ \t:=]+(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)[ \t]*(?P<units>[^\s,;]+)",
    re.MULTILINE | re.IGNORECASE)


def _degrees(value, units):
    scale = _UNITS.get(_text(units).strip().rstrip(".").casefold())
    if scale is None:
        return None
    try:
        degrees = float(value) * scale
    except (TypeError, ValueError, OverflowError):
        return None
    return degrees if 0. < degrees < 90. else None


def _common(candidates):
    if not candidates:
        return {}
    first, source = candidates[0]
    if all(math.isclose(first, value, rel_tol=1e-9, abs_tol=1e-9) for value, _ in candidates):
        return {"incidence_deg": first, "incidence_source": source}
    return {}


def cbf_incidence(contents):
    candidates = []
    for match in _CBF_LINE.finditer(contents):
        value = _degrees(match["value"], match["units"])
        if value is not None:
            candidates.append((value, match["name"]))
    return _common(candidates)


def nexus_incidence(handle, entry, frame):
    """Read named angular quantities from this entry's sample. A vector is indexed by frame."""
    group = handle.get(entry or "/")
    if not isinstance(group, h5py.Group):
        return {}
    candidates = []
    for name in group:
        node = group.get(name)
        if not isinstance(node, h5py.Group):
            continue
        if name != "sample" and _text(node.attrs.get("NX_class", "")) != "NXsample":
            continue
        for field in ("incident_angle", "incidence_angle"):
            dataset = node.get(field)
            if not isinstance(dataset, h5py.Dataset) or dataset.size == 0:
                continue
            index = 0 if frame is None else int(frame)
            try:
                if dataset.size == 1:
                    raw = np.asarray(dataset[()]).item()
                elif 0 <= index < dataset.shape[0] and dataset.size == dataset.shape[0]:
                    raw = np.asarray(dataset[index]).item()
                else:
                    continue
            except (OSError, TypeError, ValueError):
                continue
            value = _degrees(raw, dataset.attrs.get("units"))
            if value is not None:
                candidates.append((value, dataset.name))
    return _common(candidates)
