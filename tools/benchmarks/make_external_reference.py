"""Write tests/data/external/reference_values.json: what pyFAI and pygid give at fixed pixels of the public frames.

For every frame of tests/data/external, at 9 pixels (the corners, the middle, two off-diagonal points and one 10
pixels above the direct beam): pyFAI's |q| and 2θ, and q∥, q_z of pyFAI's fiber units (sample orientation 4, the one
that matches GIMaP's axes, ``pyfai_comparison.py``); for the two pygid example frames, pygid's q_xy and q_z (its
own pixel convention, pixel i at i·p). ``tests/test_external_reference.py`` checks GIMaP against these numbers, so
the comparison stays checked without pyFAI or pygid installed.

Run with an interpreter that has pyFAI and pygid:
usage: python tools/benchmarks/make_external_reference.py
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pyfai_comparison import EXTERNAL, _geometries, _integrator  # noqa: E402
from pygid_comparison import FRAMES as PYGID_FRAMES, _pixel_q  # noqa: E402
from src.gimap.shared.detector_io import load_detector_image  # noqa: E402

OUT = EXTERNAL / "reference_values.json"
FRACTIONS = ((0.02, 0.02), (0.02, 0.98), (0.98, 0.02), (0.98, 0.98), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25))


def pixels(shape, gimap) -> list[tuple[int, int]]:
    rows, columns = shape
    out = [(int(f * (rows - 1)), int(g * (columns - 1))) for f, g in FRACTIONS]
    out.append((int(np.clip(round(gimap.beam_center_y_px) - 10, 0, rows - 1)),
                int(np.clip(round(gimap.beam_center_x_px), 0, columns - 1))))
    return out


def main() -> None:
    units = importlib.import_module("pyFAI.units")
    manifest = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for key, entry in manifest.items():
        shape = load_detector_image(EXTERNAL / entry["frame"]).data.shape[:2]
        gimap, reference, incidence = _geometries(key, entry, shape)
        integrator = _integrator(reference, shape)
        where = pixels(shape, gimap)
        index = tuple(np.array(where).T)
        q = np.asarray(integrator.center_array(shape, unit="q_A^-1", scale=True), float)
        two_theta = np.degrees(np.asarray(integrator.center_array(shape, unit="2th_rad", scale=False), float))
        fiber = {}
        for name, label in (("qip_A^-1", "q_parallel"), ("qoop_A^-1", "qz")):
            unit = units.get_unit_fiber(name, incident_angle=incidence, tilt_angle=0.0, sample_orientation=4,
                                        angle_unit="deg")
            fiber[label] = np.asarray(_integrator(reference, shape).array_from_unit(unit=unit, typ="center",
                                                                                   scale=True), float)
        row = dict(incidence_deg=incidence, pixels=[list(p) for p in where],
                   pyfai=dict(q=q[index].tolist(), two_theta_deg=two_theta[index].tolist(),
                              q_parallel=fiber["q_parallel"][index].tolist(), qz=fiber["qz"][index].tolist()))
        if key in PYGID_FRAMES:
            q_xy, q_z = _pixel_q(key, entry, incidence, shape)
            row["pygid"] = dict(q_xy=q_xy[index].tolist(), q_z=q_z[index].tolist())
        frames[key] = row
    versions = {name: importlib.metadata.version(name) for name in ("pyFAI", "pygid")}
    OUT.write_text(json.dumps(dict(made_by="tools/benchmarks/make_external_reference.py", versions=versions,
                                   units="q in 1/A, angles in degrees; pixels are (row, column), row 0 at the top",
                                   frames=frames), indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
