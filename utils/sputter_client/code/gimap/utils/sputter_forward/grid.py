"The toolkit's native (alpha_f, |q_y|) grid and the fine evaluation grid of the model."

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

UTILS = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NativeGrid:
    native_edges: np.ndarray
    q_edges: np.ndarray

    @property
    def native_af(self):
        return (self.native_edges[:-1] + self.native_edges[1:]) / 2

    @property
    def q(self):
        return (self.q_edges[:-1] + self.q_edges[1:]) / 2


def native_grid(toolkit=None) -> NativeGrid:
    "The toolkit's grid (``sputtergisaxs.linecut_v3.NATIVE_EDGES``, ``tool_qinput.Q_EDGES``) via the fringe tool."
    if str(UTILS) not in sys.path:
        sys.path.insert(0, str(UTILS))
    from sputter_fringes.loader import toolkit_grid
    g = toolkit_grid(toolkit)
    return NativeGrid(np.asarray(g.native_edges, float), np.asarray(g.q_edges, float))


class FineGrid:
    def __init__(self, native: NativeGrid, k: float, sdd_mm: float, alpha_i_deg: float, pixel_mm: float = 0.172,
                 daf: float = 0.0025, nsub_q: int = 2, nblk: int = 16):
        self.native = native
        self.k = k
        self.alpha_i = alpha_i_deg
        self.daf = daf
        self.af = np.arange(0.005, native.native_edges[-1] + daf / 2, daf)
        qe = native.q_edges
        self.q = (qe[:-1, None] + (np.arange(nsub_q)[None, :] + 0.5) / nsub_q * np.diff(qe)[:, None]).ravel()
        self.nsub_q = nsub_q
        self.pix_deg = float(np.degrees(pixel_mm / sdd_mm))
        W = np.zeros((len(native.native_af), len(self.af)))
        for r in range(len(native.native_af)):
            lo, hi = native.native_edges[r], native.native_edges[r + 1]
            xs = np.linspace(lo, hi, 9)[:-1] + (hi - lo) / 16
            j = np.clip(np.searchsorted(self.af, xs) - 1, 0, len(self.af) - 2)
            fr = (xs - self.af[j]) / daf
            np.add.at(W[r], j, (1 - fr) / len(xs))
            np.add.at(W[r], j + 1, fr / len(xs))
        self.W = W
        edges = np.linspace(0, len(self.af), nblk + 1).astype(int)
        self.blocks = [(edges[b], edges[b + 1]) for b in range(nblk)]
        afc = np.array([self.af[(a + b) // 2] for a, b in self.blocks])
        self.qpar_b = self.qpar(afc[:, None], self.q[None, :])

    def qpar(self, af_deg, qy):
        af = np.radians(af_deg); ai = np.radians(self.alpha_i)
        phi = np.arcsin(np.clip(qy / (self.k * np.cos(af)), -1, 1))
        qx = self.k * (np.cos(af) * np.cos(phi) - np.cos(ai))
        return np.sqrt(qx * qx + qy * qy)

    def to_native(self, I_fine, psf_px):
        if psf_px > 0:
            I_fine = gaussian_filter1d(I_fine, psf_px * self.pix_deg / self.daf, axis=0, mode="nearest")
        I = self.W @ I_fine
        return I.reshape(len(self.native.native_af), len(self.native.q), self.nsub_q).mean(-1)

    def per_block(self, fn):
        out = np.empty((len(self.af), len(self.q)))
        for b, (a0, a1) in enumerate(self.blocks):
            out[a0:a1] = fn(slice(a0, a1), self.qpar_b[b])
        return out


class PointGrid:
    'Explicit (alpha_f row, q_par per pixel) grid, e.g. a BornAgain detector: one block per row, no binning.'

    def __init__(self, af_deg, qpar, alpha_i_deg):
        self.af = np.asarray(af_deg, float)
        self.q = np.arange(qpar.shape[1])
        self.alpha_i = alpha_i_deg
        self.blocks = [(i, i + 1) for i in range(len(self.af))]
        self.qpar_b = np.asarray(qpar, float)

    def per_block(self, fn):
        return FineGrid.per_block(self, fn)


__all__ = ["FineGrid", "NativeGrid", "PointGrid", "native_grid"]
