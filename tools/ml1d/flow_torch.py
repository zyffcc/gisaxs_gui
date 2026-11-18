"""The proposal network in PyTorch, for training; posterior/flow.py evaluates the exported arrays in numpy.

Same names and layout as the numpy version: ``encoder.stages.{i}.a|b`` (Conv1d), ``encoder.dense``, ``encoder.embed``,
``encoder.join``, ``head.hidden``, ``head.out``, ``context``, ``layers.{l}.inp``, ``layers.{l}.hidden.{j}``,
``layers.{l}.out`` (Linear).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.gimap.features.fitting.domain.posterior import encoding
from src.gimap.features.fitting.domain.posterior.flow import FORMAT, MIN_BIN, MIN_DERIVATIVE

ARCHITECTURE = {"channels": [32, 64, 96, 128], "kernel": 5, "dense": 512, "summary": 256, "context": 256,
                "layers": 10, "hidden": 256, "blocks": 2, "bins": 10, "bound": 6.0, "head": 256}


def transformed_sets(layers: int, dims: int, seed: int) -> list[list[int]]:
    """Which coordinates each coupling moves: a random half, alternating with its complement."""
    rng = np.random.default_rng(seed)
    out = []
    for layer in range(layers):
        if layer % 2 == 0:
            half = np.sort(rng.permutation(dims)[: dims // 2])
        else:
            half = np.setdiff1d(np.arange(dims), out[-1])
        out.append([int(i) for i in half])
    return out


class Stage(nn.Module):
    def __init__(self, c_in, c_out, kernel):
        super().__init__()
        self.a = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.b = nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2)

    def forward(self, x):
        return F.avg_pool1d(F.silu(self.b(F.silu(self.a(x)))), 2)


class Encoder(nn.Module):
    def __init__(self, arch):
        super().__init__()
        c_in, stages = encoding.CHANNELS, []
        for c in arch["channels"]:
            stages.append(Stage(c_in, c, arch["kernel"]))
            c_in = c
        self.stages = nn.ModuleList(stages)
        length = encoding.POINTS // 2 ** len(arch["channels"])
        self.dense = nn.Linear(c_in * length, arch["dense"])
        self.embed = nn.Linear(arch["dense"], arch["summary"])
        self.join = nn.Linear(arch["summary"] + encoding.CONDITIONS, arch["summary"])

    def forward(self, features, conditions):
        x = features
        for stage in self.stages:
            x = stage(x)
        x = self.embed(F.silu(self.dense(x.flatten(1))))
        return F.silu(self.join(torch.cat([x, conditions], dim=1)))


class Head(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.hidden = nn.Linear(arch["summary"], arch["head"])
        self.out = nn.Linear(arch["head"], len(encoding.COMPOSITIONS))

    def forward(self, h):
        return self.out(F.silu(self.hidden(h)))


class Coupling(nn.Module):
    def __init__(self, kept: int, moved: int, arch):
        super().__init__()
        self.inp = nn.Linear(kept + arch["context"], arch["hidden"])
        self.hidden = nn.ModuleList([nn.Linear(arch["hidden"], arch["hidden"]) for _ in range(arch["blocks"] - 1)])
        self.out = nn.Linear(arch["hidden"], moved * (3 * arch["bins"] - 1))
        nn.init.zeros_(self.out.weight)  # the identity at the start
        nn.init.zeros_(self.out.bias)

    def forward(self, identity, context):
        x = F.silu(self.inp(torch.cat([identity, context], dim=1)))
        for layer in self.hidden:
            x = F.silu(layer(x))
        return self.out(x)


def knots(raw, bound: float, bins: int):
    def edges(unnormalised):
        width = MIN_BIN + (1.0 - MIN_BIN * bins) * torch.softmax(unnormalised, dim=-1)
        cum = F.pad(torch.cumsum(width, dim=-1), (1, 0))
        cum = 2.0 * bound * cum - bound
        return torch.cat([torch.full_like(cum[..., :1], -bound), cum[..., 1:-1], torch.full_like(cum[..., :1], bound)],
                         dim=-1)

    inner = MIN_DERIVATIVE + F.softplus(raw[..., 2 * bins:] + math.log(math.expm1(1.0 - MIN_DERIVATIVE)))
    ones = torch.ones_like(inner[..., :1])
    return edges(raw[..., :bins]), edges(raw[..., bins:2 * bins]), torch.cat([ones, inner, ones], dim=-1)


def spline(x, raw, bound: float, bins: int, inverse: bool = False):
    inside = x.abs() < bound
    xs, ys, d = knots(raw, bound, bins)
    xc = x.clamp(-bound, bound)
    locations = ys if inverse else xs
    k = (xc[..., None] >= locations[..., 1:-1]).sum(dim=-1, keepdim=True)

    def take(a):
        return torch.gather(a, -1, k)[..., 0]

    x0, w = take(xs), take(torch.diff(xs, dim=-1))
    y0, h = take(ys), take(torch.diff(ys, dim=-1))
    d0, d1 = take(d), take(d[..., 1:])
    s = h / w
    if inverse:
        dy = xc - y0
        a = h * (s - d0) + dy * (d1 + d0 - 2 * s)
        b = h * d0 - dy * (d1 + d0 - 2 * s)
        c = -s * dy
        t = (2 * c) / (-b - torch.sqrt((b * b - 4 * a * c).clamp_min(0.0)))
        out = t * w + x0
    else:
        t = (xc - x0) / w
        out = y0 + h * (s * t * t + d0 * t * (1 - t)) / (s + (d1 + d0 - 2 * s) * t * (1 - t))
    denominator = s + (d1 + d0 - 2 * s) * t * (1 - t)
    log_det = (2 * torch.log(s) + torch.log(d1 * t * t + 2 * s * t * (1 - t) + d0 * (1 - t) ** 2)
               - 2 * torch.log(denominator))
    if inverse:
        log_det = -log_det
    return torch.where(inside, out, x), torch.where(inside, log_det, torch.zeros_like(log_det))


class Flow(nn.Module):
    def __init__(self, arch=None, seed: int = 0):
        super().__init__()
        self.arch = dict(arch or ARCHITECTURE)
        dims = encoding.DIMS
        self.transformed = transformed_sets(self.arch["layers"], dims, seed)
        self._kept = [[i for i in range(dims) if i not in moved] for moved in self.transformed]
        self.encoder = Encoder(self.arch)
        self.head = Head(self.arch)
        self.context = nn.Linear(self.arch["summary"] + encoding.SLOTS * encoding.SLOT_CODE, self.arch["context"])
        self.layers = nn.ModuleList([Coupling(len(k), len(m), self.arch)
                                     for k, m in zip(self._kept, self.transformed)])
        self.register_buffer("codes", torch.as_tensor(encoding.composition_code(np.arange(len(encoding.COMPOSITIONS)))))
        self.register_buffer("actives", torch.as_tensor(encoding.active(np.arange(len(encoding.COMPOSITIONS)))))

    def flow_context(self, h, composition):
        return F.silu(self.context(torch.cat([h, self.codes[composition].to(h.dtype)], dim=1)))

    def _raw(self, layer, identity, context):
        raw = self.layers[layer](identity, context)
        return raw.view(raw.shape[0], -1, 3 * self.arch["bins"] - 1)

    def _flow(self, x, context, active, inverse: bool):
        x = torch.where(active, x, torch.zeros_like(x))
        total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        order = reversed(range(len(self.transformed))) if inverse else range(len(self.transformed))
        for layer in order:
            moved, kept = self.transformed[layer], self._kept[layer]
            y, log_det = spline(x[:, moved], self._raw(layer, x[:, kept], context), self.arch["bound"],
                                self.arch["bins"], inverse=inverse)
            on = active[:, moved]
            x = x.clone()
            x[:, moved] = torch.where(on, y, torch.zeros_like(y))
            total = total + torch.where(on, log_det, torch.zeros_like(log_det)).sum(dim=1)
        return x, total

    def forward_z(self, z, context, active):
        return self._flow(z, context, active, inverse=False)

    def inverse_u(self, u, context, active):
        return self._flow(u, context, active, inverse=True)

    def losses(self, features, conditions, composition, z):
        """(−log q(z_active | curve, composition), cross-entropy of the composition head) per row."""
        h = self.encoder(features, conditions)
        active = self.actives[composition]
        u, log_det = self.forward_z(z, self.flow_context(h, composition), active)
        base = torch.where(active, -0.5 * u * u - 0.5 * math.log(2 * math.pi), torch.zeros_like(u)).sum(dim=1)
        return -(base + log_det), F.cross_entropy(self.head(h), composition, reduction="none")

    def export(self, extra: dict) -> tuple[dict, dict]:
        """(arrays, metadata) for posterior/flow.py; the caller adds the SHA-256 of the written npz."""
        arrays = {k: v.detach().cpu().float().numpy() for k, v in self.state_dict().items()
                  if k not in ("codes", "actives")}
        meta = {"format": FORMAT, "names": list(encoding.NAMES), "points": encoding.POINTS,
                "compositions": len(encoding.COMPOSITIONS), "architecture": self.arch,
                "transformed": self.transformed, **extra}
        return arrays, meta
