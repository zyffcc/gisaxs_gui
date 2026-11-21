"""Save Data and Fit as NXcanSAS (``shared.nxcansas``): the points fitted and the model, two SASentry groups.

Q is |q| in 1/nm as Fitting fits it (Fitting's own unit; the file says it on Q). The first entry holds the points
fitted with their σ as Idev (none when the curve has no σ), the second the model at the same q, without Idev. The
record that Save Data and Fit writes beside the file (``fit_record``: the model, the fit and how it was made) is
also in each entry, as the JSON ``data`` of an NXnote in SASprocess.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from src.gimap.shared.export_text import export_text
from src.gimap.shared.nxcansas import ARBITRARY, PER_NM, SasCurve, write_nxcansas
from src.gimap.shared.version import SOFTWARE as RELEASE

from ..domain.fit_engine import FitData
from ..domain.fit_model import FitModel, evaluate


def fit_curves(model: FitModel, data: FitData, record: dict, name: str, date: Optional[str] = None) -> list[SasCurve]:
    """The data fitted and the model of ``model`` at its q, as two NXcanSAS curves named after the curve ``name``."""
    q = np.asarray(data.q, dtype=float)
    text = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str)
    common = dict(q_unit=PER_NM, intensity_unit=ARBITRARY, run=export_text(name), sample=export_text(name),
                  process="GIMaP Fitting", record=text, date=date or "")
    described = f"Saved by {RELEASE} (Fitting, Save Data and Fit). Q: |q| in 1/nm, as fitted."
    return [
        SasCurve(title=export_text(f"{name}: the points fitted"), q=q, intensity=np.asarray(data.intensity, dtype=float),
                 sigma=None if data.sigma is None else np.asarray(data.sigma, dtype=float),
                 description=described + " I: the points fitted; Idev: their sigma.", **common),
        SasCurve(title=export_text(f"{name}: the model"), q=q, intensity=np.asarray(evaluate(model, q), dtype=float),
                 description=described + " I: the model fitted (the record names it and its parameters).", **common),
    ]


def write_curves(path, curves: list[SasCurve]):
    """The curves as one NXcanSAS file at ``path`` (``shared.nxcansas.write_nxcansas``)."""
    return write_nxcansas(path, curves)


__all__ = ["fit_curves", "write_curves"]
