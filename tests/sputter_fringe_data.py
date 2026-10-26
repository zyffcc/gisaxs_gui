"""Invented Sputter result folders for the fringe analysis (``utils/sputter_fringes``): binned vertical cuts of a
growing metal layer, with Poisson noise, two detector module gaps, a partly covered top edge and a few narrow rows
without pixels, as the toolkit writes them (``binned/run.json``, ``binned/chunk_NNNNN.npz`` / ``.json``,
``structure_physical.csv`` with ``beam_off_sample``). No real data: the geometry and the values are made up.

Intensity per (alpha_f, q_y): A(q_y) q_z^-p [b + F(Q)], with F = <sin^2(Q H / 2)> the fringe of a uniform layer of
height H (averaged over a Gaussian spread of H; minima at Q H = 2 pi n) and Q = kz(alpha_i) + kz(alpha_f) inside the
metal (refraction with the metal's critical angle); without metal (H = 0) F = 0, a cut without fringes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

ENERGY_EV = 12000.0
ALPHA_I = 0.4
GAPS = ((0.66, 0.70), (1.12, 1.16))
"""Invented module gaps (deg): no exposure there."""
EMPTY_ROWS = (0.853, 1.302)
"""Invented narrow native rows without a pixel centre (no exposure; they must not split a range)."""
CHUNK = 32


def _grid():
    from utils.sputter_fringes.loader import toolkit_grid

    return toolkit_grid()


def critical_angle(metal: str) -> float:
    from utils.sputter_fringes.optics import bulk_critical_angle_deg

    return bulk_critical_angle_deg(metal, ENERGY_EV)


def fringe(q: np.ndarray, height: float, spread: float = 0.08) -> np.ndarray:
    """<sin^2(Q H' / 2)> over H' ~ N(H, spread H): the layer's fringe without the form factor's envelope, minima at
    Q H = 2 pi n (0 without metal)."""
    if height <= 0:
        return np.zeros_like(q)
    return 0.5 - 0.5 * np.cos(q * height) * np.exp(-0.5 * (q * spread * height) ** 2)


def write_result(folder: Path, height: Callable[[float], float], *, metal: str = "Cu", windows: int = 170,
                 seconds: float = 0.5, level: float = 1.0e5, floor: float = 0.5, slope: float = 3.0, beam_off=(),
                 seed: int = 1, spread: float = 0.08) -> Path:
    """A result folder whose window i (time (i + 0.5) seconds) shows a layer of height ``height(t)`` nm."""
    from utils.sputter_fringes.optics import q_inside, qz_vacuum, wavenumber

    grid = _grid()
    edges, q_edges = grid.native_edges, grid.q_edges
    af = (edges[:-1] + edges[1:]) / 2
    qy = (q_edges[:-1] + q_edges[1:]) / 2
    k = wavenumber(ENERGY_EV)
    ac = critical_angle(metal)
    qz = qz_vacuum(af, ALPHA_I, k)
    q = q_inside(af, ALPHA_I, ac, k)
    width = np.diff(edges) / 0.01
    exposure_row = width.copy()
    for lo, hi in GAPS:
        exposure_row[(af > lo) & (af < hi)] = 0.0
    for row in EMPTY_ROWS:
        exposure_row[np.argmin(np.abs(af - row))] = 0.0
    exposure_row[af > 1.48] *= 0.4
    exposure_row[af > 1.49] = 0.0
    E1 = np.broadcast_to((exposure_row[:, None] * seconds * 0.2 * np.ones(len(qy))[None, :]), (len(af), len(qy)))
    shape_qy = (qy / qy[0]) ** -2.5
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    binned = folder / "binned"
    binned.mkdir(parents=True, exist_ok=True)
    chunks, rows_all = [], []
    times = (np.arange(windows) + 0.5) * seconds
    for start in range(0, windows, CHUNK):
        Y = np.zeros((min(CHUNK, windows - start), 2, len(af), len(qy)), np.uint64)
        E = np.zeros(Y.shape)
        rows = []
        for i, t in enumerate(times[start:start + CHUNK]):
            dim = 0.3 if (start + i) in beam_off else 1.0
            cut = (floor + fringe(q, float(height(t)), spread)) * qz ** -slope
            rate = level * dim * cut[:, None] * shape_qy[None, :]
            for side in (0, 1):
                E[i, side] = E1
                Y[i, side] = rng.poisson(rate * E1)
            index = start + i
            rows.append({"frame": f"bin_{index:06d}", "frame_id": index, "time_s": float(t),
                         "first_time_s": float(t - seconds / 2), "last_time_s": float(t + seconds / 2),
                         "raw_frames": 5, "nominal_bin_seconds": seconds})
        name = f"chunk_{len(chunks):05d}"
        buffer = io.BytesIO()
        np.savez(buffer, Y=Y, E=E)
        data = buffer.getvalue()
        (binned / f"{name}.npz").write_bytes(data)
        meta = json.dumps({"rows": rows}).encode("utf-8")
        (binned / f"{name}.json").write_bytes(meta)
        chunks.append({"file": f"{name}.npz", "sha256": hashlib.sha256(data).hexdigest(),
                       "metadata_sha256": hashlib.sha256(meta).hexdigest()})
        rows_all += rows
    run = {"status": "completed", "bins": windows, "chunks": chunks, "metal": metal, "model_metal": "Ag",
           "geometry": {"energy_ev": ENERGY_EV, "sdd_mm": 3000.0, "pixel_mm": 0.172, "alpha_i_deg": ALPHA_I,
                        "db_col": 800.0, "db_row": 1300.0, "sample_roll_deg": 0.0}}
    (binned / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    with (folder / "structure_physical.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "beam_off_sample"])
        for index, row in enumerate(rows_all):
            writer.writerow([row["frame"], row["time_s"], "true" if index in beam_off else "false"])
    return folder


def linear(rate: float, start: float) -> Callable[[float], float]:
    """H = rate (t - start) after ``start``, 0 before (no metal)."""
    return lambda t: max(rate * (t - start), 0.0)


def true_order(af: float, height: float, metal: str = "Cu") -> float:
    """n = H Q / (2 pi) of a minimum at ``af`` (close to an integer for a minimum of the layer)."""
    from utils.sputter_fringes.optics import q_inside, wavenumber

    return height * float(q_inside(af, ALPHA_I, critical_angle(metal), wavenumber(ENERGY_EV))) / (2 * math.pi)


__all__ = ["ALPHA_I", "ENERGY_EV", "GAPS", "critical_angle", "fringe", "linear", "true_order", "write_result"]
