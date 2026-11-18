"""An exported proposal network evaluated in numpy.

* A 1D CNN summarises the curve (``encoding.features``) with the cut's conditions into h;
* a head gives the probabilities of the compositions (``encoding.COMPOSITIONS``) from h;
* a conditional normalising flow of rational-quadratic spline couplings (Durkan et al. 2019) gives the density of z
  (``encoding``) given h and a composition: the flow maps z to u ~ N(0, I) layer by layer; each coupling leaves some
  coordinates as they are and moves the others by monotone splines on [−B, B] (identity outside) whose knots an MLP
  computes from the unchanged coordinates and the context. Coordinates a composition does not have are set to 0 before
  every MLP and never moved, so the density of the active ones, log N(u_active) + Σ log|∂u/∂z|, is exact.

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
FORMAT = "gimap_ml1d_flow_v2"


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


def spline(x, raw, bound: float, bins: int, inverse: bool = False):
    """(y, log|dy/dx|) of the monotone rational-quadratic splines, identity outside [−bound, bound]."""
    inside = np.abs(x) < bound
    xs, ys, d = knots(raw, bound, bins)
    xc = np.clip(x, -bound, bound)
    k = np.sum(xc[..., None] >= (ys if inverse else xs)[..., 1:-1], axis=-1)[..., None]
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
    """An exported network: ``summary`` of curves, ``composition_log_probs``, then for a composition ``context`` and
    ``log_prob`` of parameters or ``sample``."""

    def __init__(self, arrays: dict, meta: dict):
        if meta.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} network")
        if tuple(meta["names"]) != encoding.NAMES or meta["points"] != encoding.POINTS or \
                meta["compositions"] != len(encoding.COMPOSITIONS) or meta.get("encoding") != encoding.VERSION:
            raise ValueError("The network was trained on another encoding of the parameters")
        self.arrays = {k: np.asarray(v, float) for k, v in arrays.items()}
        self.meta = meta
        self.arch = meta["architecture"]
        self.moved = [np.asarray(m, int) for m in meta["transformed"]]
        self.kept = [np.setdiff1d(np.arange(encoding.DIMS), m) for m in self.moved]

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

    def summary(self, features, conditions) -> np.ndarray:
        """h of curves (``encoding.features``, ``encoding.conditions``): (B, context)."""
        a = self.arrays
        x = np.asarray(features, float)
        for i in range(len(self.arch["channels"])):
            x = _silu(_conv(a, f"encoder.stages.{i}.a", x))
            x = _silu(_conv(a, f"encoder.stages.{i}.b", x))
            x = 0.5 * (x[..., 0::2] + x[..., 1::2])
        x = _silu(_linear(a, "encoder.dense", x.reshape(x.shape[0], -1)))
        x = _linear(a, "encoder.embed", x)
        return _silu(_linear(a, "encoder.join", np.concatenate([x, np.asarray(conditions, float)], axis=1)))

    def composition_log_probs(self, h) -> np.ndarray:
        """log p(composition | curve) (B, len(COMPOSITIONS)) under the training prior of compositions."""
        logits = _linear(self.arrays, "head.out", _silu(_linear(self.arrays, "head.hidden", h)))
        logits = logits - logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def context(self, h, composition) -> np.ndarray:
        """The flow's context of curves h (B, C) for compositions ``composition`` (B,) or one for all."""
        h = np.asarray(h, float)
        code = encoding.composition_code(np.broadcast_to(np.asarray(composition, int), (h.shape[0],)))
        return _silu(_linear(self.arrays, "context", np.concatenate([h, code], axis=1)))

    def _raw(self, layer: int, identity, context):
        a = self.arrays
        x = _silu(_linear(a, f"layers.{layer}.inp", np.concatenate([identity, context], axis=1)))
        for j in range(self.arch["blocks"] - 1):
            x = _silu(_linear(a, f"layers.{layer}.hidden.{j}", x))
        x = _linear(a, f"layers.{layer}.out", x)
        return x.reshape(x.shape[0], -1, 3 * self.arch["bins"] - 1)

    def _flow(self, x, context, active, inverse: bool):
        x = np.where(active, np.asarray(x, float), 0.0)
        total = np.zeros(x.shape[0])
        order = reversed(range(len(self.moved))) if inverse else range(len(self.moved))
        for layer in order:
            kept, moved = self.kept[layer], self.moved[layer]
            raw = self._raw(layer, x[:, kept], context)
            y, log_det = spline(x[:, moved], raw, self.arch["bound"], self.arch["bins"], inverse=inverse)
            on = active[:, moved]
            x[:, moved] = np.where(on, y, 0.0)
            total += np.sum(np.where(on, log_det, 0.0), axis=1)
        return x, total

    def forward(self, z, context, active):
        """(u, Σ log|∂u/∂z|) of parameters z (B, DIMS) given contexts (B, C) and active coordinates (B, DIMS)."""
        return self._flow(z, context, np.asarray(active, bool), inverse=False)

    def inverse(self, u, context, active):
        """(z, Σ log|∂z/∂u|) of base points u (B, DIMS); inactive coordinates come back as 0."""
        return self._flow(u, context, np.asarray(active, bool), inverse=True)

    def log_prob(self, z, context, active) -> np.ndarray:
        """log q(z_active | curve, composition) of each row."""
        active = np.asarray(active, bool)
        u, log_det = self.forward(z, context, active)
        return np.sum(np.where(active, -0.5 * u * u - 0.5 * np.log(2 * np.pi), 0.0), axis=1) + log_det

    def sample(self, context, composition: int, count: int, rng: Optional[np.random.Generator] = None):
        """(z (count, DIMS), log q(z)) for one curve's context (1, C) of composition ``composition``."""
        rng = rng or np.random.default_rng()
        active = np.repeat(encoding.active(composition), count, axis=0)
        u = np.where(active, rng.standard_normal((count, encoding.DIMS)), 0.0)
        z, log_det = self.inverse(u, np.repeat(np.asarray(context, float)[:1], count, axis=0), active)
        base = np.sum(np.where(active, -0.5 * u * u - 0.5 * np.log(2 * np.pi), 0.0), axis=1)
        return z, base - log_det


__all__ = ["FORMAT", "Network", "knots", "spline"]
