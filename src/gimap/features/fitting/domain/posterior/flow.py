"""An exported proposal network evaluated in numpy: a 1D CNN that summarises the curve, and a conditional normalising
flow of rational-quadratic spline couplings (Durkan et al. 2019) over the parameters z of ``encoding``.

The flow maps z to u ~ N(0, I) layer by layer; each coupling leaves some of the coordinates as they are and moves the
others by monotone splines on [−B, B] (identity outside) whose knots an MLP computes from the unchanged coordinates and
the curve's context. log q(z | curve) = log N(u) + Σ log|∂u/∂z|. Sampling runs the layers backwards.

The architecture and the arrays come from ``tools/ml1d/train_flow.py`` (PyTorch); a test compares the two on a stored
network. Files: NAME.npz (the arrays) and NAME.json (architecture, encoding, training record, SHA-256 of the npz).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import encoding

MIN_BIN = 1e-3
MIN_DERIVATIVE = 1e-3
FORMAT = "gimap_ml1d_flow_v1"


def _silu(x):
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _linear(arrays, name, x):
    return x @ arrays[f"{name}.weight"].T + arrays[f"{name}.bias"]


def _conv(arrays, name, x):
    """Conv1d with 'same' padding (cross-correlation, as PyTorch): x (B, C, L)."""
    w, b = arrays[f"{name}.weight"], arrays[f"{name}.bias"]
    k = w.shape[2]
    padded = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=2)  # (B, C, L, K)
    return np.einsum("bclk,ock->bol", windows, w, optimize=True) + b[None, :, None]


def knots(raw, bound: float, bins: int):
    """(x knots, y knots, derivatives) of the splines from the conditioner's output (…, 3·bins − 1)."""
    def edges(unnormalised):
        width = MIN_BIN + (1.0 - MIN_BIN * bins) * _softmax(unnormalised)
        cum = np.concatenate([np.zeros(width.shape[:-1] + (1,)), np.cumsum(width, axis=-1)], axis=-1)
        cum = 2.0 * bound * cum - bound
        cum[..., 0], cum[..., -1] = -bound, bound
        return cum

    inner = MIN_DERIVATIVE + _softplus(raw[..., 2 * bins:] + np.log(np.expm1(1.0 - MIN_DERIVATIVE)))
    ones = np.ones(inner.shape[:-1] + (1,))
    return edges(raw[..., :bins]), edges(raw[..., bins:2 * bins]), np.concatenate([ones, inner, ones], axis=-1)


def _bin(locations, values):
    index = np.sum(values[..., None] >= locations[..., 1:-1], axis=-1)
    return index[..., None]


def spline(x, raw, bound: float, bins: int, inverse: bool = False):
    """(y, log|dy/dx|) of the monotone rational-quadratic splines, identity outside [−bound, bound]."""
    inside = np.abs(x) < bound
    xs, ys, d = knots(raw, bound, bins)
    xc = np.clip(x, -bound, bound)
    k = _bin(ys if inverse else xs, xc)
    take = lambda a: np.take_along_axis(a, k, axis=-1)[..., 0]  # noqa: E731
    x0, w = take(xs), take(np.diff(xs, axis=-1))
    y0, h = take(ys), take(np.diff(ys, axis=-1))
    d0, d1 = take(d), take(d[..., 1:])
    s = h / w
    if inverse:
        dy = xc - y0
        a = h * (s - d0) + dy * (d1 + d0 - 2 * s)
        b = h * d0 - dy * (d1 + d0 - 2 * s)
        c = -s * dy
        t = (2 * c) / (-b - np.sqrt(np.maximum(b * b - 4 * a * c, 0.0)))
        out = t * w + x0
    else:
        t = (xc - x0) / w
        out = y0 + h * (s * t * t + d0 * t * (1 - t)) / (s + (d1 + d0 - 2 * s) * t * (1 - t))
    denominator = s + (d1 + d0 - 2 * s) * t * (1 - t)
    log_det = 2 * np.log(s) + np.log(d1 * t * t + 2 * s * t * (1 - t) + d0 * (1 - t) ** 2) - 2 * np.log(denominator)
    if inverse:
        log_det = -log_det
    return np.where(inside, out, x), np.where(inside, log_det, 0.0)


class Network:
    """An exported network: ``context`` of curves, then ``log_prob`` of parameters or ``sample``."""

    def __init__(self, arrays: dict, meta: dict):
        if meta.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} network")
        if tuple(meta["names"]) != encoding.NAMES or meta["points"] != encoding.POINTS:
            raise ValueError("The network was trained on another encoding of the parameters")
        self.arrays = {k: np.asarray(v, float) for k, v in arrays.items()}
        self.meta = meta
        self.arch = meta["architecture"]
        self.masks = [np.asarray(m, int) for m in meta["transformed"]]

    @classmethod
    def load(cls, path, check: bool = True) -> "Network":
        """NAME.npz with NAME.json beside it; ``check``: refuse arrays whose SHA-256 is not the recorded one."""
        path = Path(path).with_suffix(".npz")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if check:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != meta.get("sha256"):
                raise ValueError(f"{path.name} is not the file the network was exported as (SHA-256 differs)")
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        return cls(arrays, meta)

    def context(self, features, conditions) -> np.ndarray:
        """The flow's context of curves (``encoding.features``, ``encoding.conditions``): (B, context)."""
        a = self.arrays
        x = np.asarray(features, float)
        for i in range(len(self.arch["channels"])):
            x = _silu(_conv(a, f"encoder.stages.{i}.a", x))
            x = _silu(_conv(a, f"encoder.stages.{i}.b", x))
            x = 0.5 * (x[..., 0::2] + x[..., 1::2])
        x = _silu(_linear(a, "encoder.dense", x.reshape(x.shape[0], -1)))
        x = _linear(a, "encoder.embed", x)
        return _silu(_linear(a, "encoder.join", np.concatenate([x, np.asarray(conditions, float)], axis=1)))

    def _raw(self, layer: int, identity, context):
        a = self.arrays
        x = _silu(_linear(a, f"layers.{layer}.inp", np.concatenate([identity, context], axis=1)))
        for j in range(self.arch["blocks"] - 1):
            x = _silu(_linear(a, f"layers.{layer}.hidden.{j}", x))
        x = _linear(a, f"layers.{layer}.out", x)
        return x.reshape(x.shape[0], -1, 3 * self.arch["bins"] - 1)

    def _split(self, layer):
        transformed = self.masks[layer]
        kept = np.setdiff1d(np.arange(len(encoding.NAMES)), transformed)
        return kept, transformed

    def forward(self, z, context):
        """(u, Σ log|∂u/∂z|) of parameters z (B, D) given the context (B, C)."""
        u = np.array(z, float)
        total = np.zeros(u.shape[0])
        for layer in range(len(self.masks)):
            kept, moved = self._split(layer)
            raw = self._raw(layer, u[:, kept], context)
            u[:, moved], log_det = spline(u[:, moved], raw, self.arch["bound"], self.arch["bins"])
            total += log_det.sum(axis=1)
        return u, total

    def inverse(self, u, context):
        """(z, Σ log|∂z/∂u|) of base points u (B, D)."""
        z = np.array(u, float)
        total = np.zeros(z.shape[0])
        for layer in reversed(range(len(self.masks))):
            kept, moved = self._split(layer)
            raw = self._raw(layer, z[:, kept], context)
            z[:, moved], log_det = spline(z[:, moved], raw, self.arch["bound"], self.arch["bins"], inverse=True)
            total += log_det.sum(axis=1)
        return z, total

    def log_prob(self, z, context) -> np.ndarray:
        """log q(z | curve) of each row."""
        u, log_det = self.forward(z, context)
        return -0.5 * np.sum(u * u, axis=1) - 0.5 * u.shape[1] * np.log(2 * np.pi) + log_det

    def sample(self, context, count: int, rng: Optional[np.random.Generator] = None):
        """(z (count, D), log q(z)) for one curve's context (1, C)."""
        rng = rng or np.random.default_rng()
        u = rng.standard_normal((count, len(encoding.NAMES)))
        z, log_det = self.inverse(u, np.repeat(np.asarray(context, float)[:1], count, axis=0))
        return z, -0.5 * np.sum(u * u, axis=1) - 0.5 * u.shape[1] * np.log(2 * np.pi) - log_det


__all__ = ["FORMAT", "Network", "knots", "spline"]
