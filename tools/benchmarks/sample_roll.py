"""Measure absolute sample roll on the three public grazing-incidence reference frames."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
from src.gimap.features.analyze.application.sample_roll import measure_frame_roll
from src.gimap.features.analyze.domain import Corrections, giwaxs_maps
from src.gimap.features.analyze.infrastructure.adapters import DetectorIoFrameSource

DATA = ROOT / "tests" / "data" / "external"


def _signed_means(analysis, geometry):
    """Strongest MAPbI3 ring, in the original crystal orientation benchmark window."""
    maps = giwaxs_maps(analysis.shape, geometry)
    ring = analysis.valid & maps.above_horizon & (maps.q >= .9888) & (maps.q <= 1.0057)
    chi, signal = maps.chi_deg[ring], analysis.data[ring].astype(float)
    sides = []
    for sign in (-1, 1):
        n = np.histogram(sign*chi, bins=np.arange(91))[0]
        total = np.histogram(sign*chi, bins=np.arange(91), weights=signal)[0]
        profile = np.divide(total, n, where=n > 0, out=np.full(90, np.nan))
        sides.append((profile, float(total[84:86].sum()/n[84:86].sum())))
    return sides


def measure_public(name):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    info = manifest[name]
    corrections = Corrections()
    if info.get("mask"):
        corrections = replace(corrections, mask_path=str(DATA / info["mask"]))
    request = AnalysisRequest(DATA / info["frame"], mode=info["kind"], incidence_deg=info["incidence_deg"],
                              corrections=corrections, with_map=False, with_record=False)
    frames = DetectorIoFrameSource()
    analysis = AnalyzeFrame(frames, profiles=None)(request)
    if analysis.geometry is None:
        raise RuntimeError(f"Missing detector geometry on {name}.")
    result = measure_frame_roll(analysis, frames)
    record = result.as_dict()
    if name == "giwaxs_p08_mapi" and result.deg is not None:
        before = _signed_means(analysis, analysis.geometry.with_sample_roll(0.))
        after = _signed_means(analysis, analysis.geometry.with_sample_roll(result.deg))
        db = before[1][0] - before[0][0]
        da = after[1][0] - after[0][0]
        common = np.isfinite(db) & np.isfinite(da)
        record["strongest_ring"] = {
            "q_range_A_inv": [.9888, 1.0057],
            "counts_at_abs_chi_84_to_86_before": [entry[1] for entry in before],
            "counts_at_abs_chi_84_to_86_after": [entry[1] for entry in after],
            "rms_signed_difference_before": float(np.sqrt(np.mean(db[common]**2))),
            "rms_signed_difference_after": float(np.sqrt(np.mean(da[common]**2))),
            "common_one_degree_bins": int(common.sum()),
        }
    return record


if __name__ == "__main__":
    print(json.dumps({name: measure_public(name) for name in (
        "giwaxs_p08_mapi", "giwaxs_xeuss_pbte", "gisaxs_galaxi")}, indent=2, allow_nan=False))
