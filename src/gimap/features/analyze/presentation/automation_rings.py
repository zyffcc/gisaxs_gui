"""The rings the automatic analysis draws on a GIWAXS map, split by what their pixels are (``automation.py``)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

RING_STATES = (
    ("measured", dict(color="white", linewidth=1.4, linestyle="-")),
    ("in a shadow", dict(color="#ff9f1c", linewidth=2.2, linestyle="-")),
    ("not measured", dict(color="#e63946", linewidth=1.2, linestyle=(0, (3, 2)))),
)


def _draw_rings(axes, image: np.ndarray, extent: tuple, rings: Sequence[dict]) -> None:
    """Each ring on the map, split by what its pixels are (χ from the surface normal, ±90° in plane)."""
    q0, q1, z0, z1 = extent
    rows, columns = image.shape
    chi = np.arange(-90.0, 90.001, 0.5)
    drawn = set()
    for ring in rings:
        q = float(ring["q"])
        x, z = q * np.sin(np.radians(chi)), q * np.cos(np.radians(chi))
        column = np.floor((x - q0) / max(q1 - q0, 1e-12) * columns).astype(int)
        row = np.floor((z1 - z) / max(z1 - z0, 1e-12) * rows).astype(int)  # row 0 is the top (largest qz)
        inside = (column >= 0) & (column < columns) & (row >= 0) & (row < rows)
        pixel = np.full(chi.shape, np.nan)
        pixel[inside] = image[row[inside], column[inside]]
        state = np.where(np.isfinite(pixel) & (pixel > 0), 0, 2)
        for low, high in ring.get("shadowed") or ():
            state[(state == 0) & (np.abs(chi) >= float(low)) & (np.abs(chi) <= float(high))] = 1
        state[~inside] = -1
        start = 0
        for index in range(1, chi.size + 1):
            if index < chi.size and state[index] == state[start]:
                continue
            if state[start] >= 0:
                name, style = RING_STATES[int(state[start])]
                segment = slice(start, min(index + 1, chi.size))
                axes.plot(x[segment], z[segment], label=None if name in drawn else name, **style)
                drawn.add(name)
            start = index
        if ring.get("label"):
            seen = np.flatnonzero(state == 0)  # label a measured part, near χ = 30° where possible
            at = int(seen[np.argmin(np.abs(chi[seen] - 30.0))]) if seen.size else int(np.argmin(np.abs(chi - 45.0)))
            axes.annotate(
                str(ring["label"]), (x[at], z[at]), color="white", fontsize=8, xytext=(4, 4),
                textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, linewidth=0),
            )
    axes.set_xlim(q0, q1)
    axes.set_ylim(z0, z1)
    if drawn:
        axes.legend(loc="upper right", fontsize=7, facecolor="#303030", labelcolor="white", framealpha=0.8)


__all__ = ["RING_STATES", "_draw_rings"]
