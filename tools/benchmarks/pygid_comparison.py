"""GIMaP against pygid on the two GIWAXS frames of tests/data/external that come from pygid's usage examples.

Per pixel: q∥ and q_z of GIMaP against pygid's (``CoordMaps``: detector → lab → sample frame → q_xy, q_z), with the
flips of pygid's own examples. pygid places pixel i at i·p from the PONI, pyFAI and GIMaP at (i + ½)·p (the PONI is
measured from the detector's corner), so the comparison is made twice: as given, and with GIMaP's direct beam moved
by half a pixel along both axes, which is pygid's convention.

Rings: the angle-mean of pygid's polar map (``det2pol_gid``, 0–90° from the surface, uncorrected) against GIMaP's
I(q) of the q∥ > 0 side (uncorrected); each ring's centre is fitted with a Gaussian on a straight line in both curves.
``--published DIR`` adds the polar maps that pygid's authors published with the examples (Zenodo 10.5281/zenodo.17466183,
``*_result.h5``, made with pygid 0.1.16).

pygid is not a dependency of GIMaP: run with an interpreter that has it.
usage: python tools/benchmarks/pygid_comparison.py [--json FILE] [--published DIR]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pyfai_comparison import EXTERNAL, _geometries  # noqa: E402
from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps, reduce_giwaxs  # noqa: E402
from src.gimap.features.analyze.domain.validity import valid_pixels  # noqa: E402
from src.gimap.shared.detector_io import load_detector_image  # noqa: E402
from src.gimap.shared.geometry.q_mapping import grazing_q_map  # noqa: E402

FRAMES = {  # pygid's examples: flips, and the published result file
    "giwaxs_p08_mapi": dict(flipud=True, fliplr=False, published="S121_MAI_A2_00841_result.h5", window=0.025),
    "giwaxs_xeuss_pbte": dict(flipud=True, fliplr=False, published="PbTe_3010_result.h5", window=0.04),
}
EDGE = 0.98
"""Rings beyond this fraction of pygid's largest q are left out (the map ends there)."""


def _pygid():
    logging.disable(logging.INFO)
    return importlib.import_module("pygid")


def _params(key: str, entry: dict, incidence: float, shape):
    pygid = _pygid()
    mask = str(EXTERNAL / entry["mask"]) if "mask" in entry else None
    flips = FRAMES[key]
    return pygid.ExpParams(poni_path=str(EXTERNAL / entry["poni"]), mask_path=mask, fliplr=flips["fliplr"],
                           flipud=flips["flipud"], transp=False, ai=incidence,
                           img_dim=list(shape))


def _pixel_q(key: str, entry: dict, incidence: float, shape):
    """pygid's q_xy and q_z of every pixel, in GIMaP's orientation (row 0 at the top)."""
    params = _params(key, entry, incidence, shape)
    maps = _pygid().CoordMaps(params)
    maps.img_dim = params.img_dim
    params._calc_k_()
    lab = maps._p_to_q_lab_("full")
    q_xy, q_z = maps._q_smpl_to_q_giwaxs_(maps._q_lab_to_q_smpl_(lab, ai=incidence))
    flips = FRAMES[key]
    for flip, axis in ((flips["flipud"], 0), (flips["fliplr"], 1)):
        if flip:
            q_xy, q_z = np.flip(q_xy, axis), np.flip(q_z, axis)
    assert q_xy.shape == tuple(shape)
    return np.asarray(q_xy, float), np.asarray(q_z, float)


def _polar(key: str, entry: dict, incidence: float, shape):
    """pygid's polar map of the frame: q, and the mean over the angles of every q (NaN where no angle has data)."""
    params = _params(key, entry, incidence, shape)
    maps = _pygid().CoordMaps(params, angular_range=(0, 90))
    conversion = _pygid().Conversion(matrix=maps, path=str(EXTERNAL / entry["frame"]))
    with tempfile.TemporaryDirectory() as folder:  # pygid may write beside the working directory
        here = os.getcwd()
        os.chdir(folder)
        try:
            q, _angles, image = conversion.det2pol_gid(return_result=True)
        finally:
            os.chdir(here)
    return np.asarray(q, float), _angle_mean(np.asarray(image)[0])


def _angle_mean(image):
    image = np.where(image > 0, image, np.nan)
    with np.errstate(invalid="ignore"):
        counts = np.sum(np.isfinite(image), axis=0)
        return np.where(counts > 0, np.nansum(image, axis=0) / np.maximum(counts, 1), np.nan)


def _centre(q, intensity, q0, window):
    from scipy.optimize import curve_fit

    keep = (q > q0 - window) & (q < q0 + window) & np.isfinite(intensity)

    def peak(x, height, centre, width, base, slope):
        return height * np.exp(-0.5 * ((x - centre) / width) ** 2) + base + slope * (x - q0)

    y = intensity[keep]
    start = [y.max() - np.median(y), q0, window / 4, np.median(y), 0.0]
    return float(curve_fit(peak, q[keep], y, p0=start, maxfev=20000)[0][1])


def _rings(q_ref, i_ref, q_gimap, i_gimap, window):
    from scipy.signal import find_peaks

    found, _ = find_peaks(np.nan_to_num(i_ref), prominence=0.08 * np.nanmax(i_ref))
    out = []
    for q0 in q_ref[found]:
        if q0 < 0.3 or q0 > EDGE * np.nanmax(q_ref[np.isfinite(i_ref)]):
            continue
        try:
            out.append(dict(q=round(float(q0), 4),
                            gimap_minus_pygid=_centre(q_gimap, i_gimap, q0, window) - _centre(q_ref, i_ref, q0, window)))
        except RuntimeError:
            continue
    return out


def compare(key: str, entry: dict, published: Path | None) -> dict:
    frame = load_detector_image(EXTERNAL / entry["frame"])
    data = np.asarray(frame.data, dtype=np.float64)
    shape = data.shape[:2]
    valid = valid_pixels(frame.data, frame.mask)
    if "mask" in entry:
        valid &= np.asarray(load_detector_image(EXTERNAL / entry["mask"]).data) == 0
    gimap, _reference, incidence = _geometries(key, entry, shape)
    shifted = replace(gimap, beam_center_x_px=gimap.beam_center_x_px + 0.5,
                      beam_center_y_px=gimap.beam_center_y_px + 0.5)
    q_xy, q_z = _pixel_q(key, entry, incidence, shape)
    pixels = {}
    for label, geometry in (("as_given", gimap), ("half_pixel", shifted)):
        grazing = grazing_q_map(shape, geometry)
        pixels[label] = dict(q_parallel_max_abs=float(np.max(np.abs(np.abs(grazing.q_parallel) - np.abs(q_xy)))),
                             qz_max_abs=float(np.max(np.abs(grazing.qz - q_z))),
                             same_sign=bool(np.all(np.sign(grazing.q_parallel[np.abs(q_xy) > 1e-6])
                                                   == np.sign(q_xy[np.abs(q_xy) > 1e-6]))))

    maps = giwaxs_maps(shape, gimap)
    side = valid & maps.above_horizon & (maps.q_parallel > 0)
    radial = reduce_giwaxs(data.astype(np.float32), side, gimap, with_map=False, counts=False).curve("radial")
    q_gimap, i_gimap = np.asarray(radial.x, float), np.asarray(radial.intensity, float)
    window = FRAMES[key]["window"]
    q_ref, i_ref = _polar(key, entry, incidence, shape)
    result = dict(frame=key, shape=list(shape), tilt_deg=round(float(gimap.tilt_deg), 3), incidence_deg=incidence,
                  pixels=pixels, polar_step=float(np.median(np.diff(q_ref))),
                  rings=_rings(q_ref, i_ref, q_gimap, i_gimap, window))
    if published is not None and (published / FRAMES[key]["published"]).exists():
        h5py = importlib.import_module("h5py")
        with h5py.File(published / FRAMES[key]["published"], "r") as source:
            q_pub = source["entry_0001/data/q_gid_pol"][()]
            i_pub = _angle_mean(source["entry_0001/data/img_gid_pol"][0])
            version = source["entry_0001/process/version"][()]
        result["published"] = dict(pygid=version.decode() if isinstance(version, bytes) else str(version),
                                   q_grid_max_abs_vs_this_pygid=float(np.max(np.abs(q_pub - q_ref)))
                                   if q_pub.shape == q_ref.shape else None,
                                   rings=_rings(q_pub, i_pub, q_gimap, i_gimap, window))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--published", type=Path, help="folder with pygid's *_result.h5 of the examples")
    a = ap.parse_args()
    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
    version = importlib.metadata.version("pygid") if hasattr(importlib, "metadata") else "?"
    rows = [compare(key, manifest[key], a.published) for key in FRAMES]
    for row in rows:
        print(json.dumps(row))
    if a.json:
        a.json.write_text(json.dumps(dict(pygid=version, frames=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    import importlib.metadata  # noqa: F401  (for the version above)

    main()
