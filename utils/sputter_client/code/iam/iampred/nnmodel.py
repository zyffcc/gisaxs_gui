'Mixture-density network of the predictor: window features -> K candidate solutions with weights.'

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 0.01
BN_EPS = 1e-5


def to_z(x, lb, ub):
    'Fitter coordinates -> unbounded network coordinates (squashed logit of the position inside the bounds).'
    u = np.clip((np.asarray(x, float) - lb) / (ub - lb), 0.0, 1.0)
    v = EPS + (1 - 2 * EPS) * u
    return np.log(v / (1 - v))


def from_z(z, lb, ub):
    v = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    u = np.clip((v - EPS) / (1 - 2 * EPS), 0.0, 1.0)
    return lb + u * (ub - lb)


def build(n_in_ch, n_scalar, n_out, k, width=32):
    import torch
    from torch import nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class MDN(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.f = nn.Sequential(blk(n_in_ch, w), nn.MaxPool2d(2), blk(w, 2 * w), nn.MaxPool2d(2), blk(2 * w, 4 * w),
                                   nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten())
            self.s = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU())
            self.h = nn.Sequential(nn.Linear(4 * w * 18 + 64, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512),
                                   nn.ReLU(), nn.Linear(512, k * (1 + 2 * n_out)))
            self.k, self.d = k, n_out

        def forward(self, x, s):
            o = self.h(torch.cat([self.f(x), self.s(s)], 1))
            k, d = self.k, self.d
            logit = o[:, :k]
            mu = o[:, k:k + k * d].reshape(-1, k, d)
            ls = o[:, k + k * d:].reshape(-1, k, d).clamp(-7, 3)
            return logit, mu, ls

    return MDN()


def nll(logit, mu, ls, y, wdim=None):
    'Mixture negative log likelihood per sample (y: (n, d) normalised targets; wdim: optional per-dim weights).'
    import torch
    lp = -0.5 * ((y[:, None, :] - mu) / ls.exp()) ** 2 - ls - 0.9189385332
    if wdim is not None:
        lp = lp * wdim[None, None, :]
    return -torch.logsumexp(torch.log_softmax(logit, 1) + lp.sum(2), 1)


class _Base:
    'Normalisation, candidates and intervals shared by both back ends; subclasses implement _forward(xa, sa) ->'

    def _setup(self, meta):
        self.meta = meta
        for key in ("xm", "xs", "sm", "ss", "tm", "ts", "lb", "ub"):
            setattr(self, key, np.asarray(meta[key], float))

    def prep(self, X, S):
        X = np.array(X, np.float32, copy=True)
        nc = X.shape[1] - 1
        X[:, :nc] = (X[:, :nc] - self.xm[None, :, None, None]) / self.xs[None, :, None, None]
        X[:, :nc] *= X[:, nc:nc + 1]
        return X, ((np.asarray(S, float) - self.sm) / self.ss).astype(np.float32)

    def raw(self, X, S, batch=256):
        xa, sa = self.prep(X, S)
        W, MU, SD = [], [], []
        for b in range(0, len(xa), batch):
            lg, mu, ls = self._forward(xa[b:b + batch], sa[b:b + batch])
            lg = lg - lg.max(1, keepdims=True)
            w = np.exp(lg); w /= w.sum(1, keepdims=True)
            W.append(w); MU.append(mu); SD.append(np.exp(ls))
        w = np.concatenate(W)
        mu = np.concatenate(MU) * self.ts + self.tm
        sd = np.concatenate(SD) * self.ts
        return w, mu, sd

    def candidates(self, X, S):
        'Per sample: candidates sorted by weight, each {weight, params (fitter units), reported, reported_sd}.'
        w, mu, sd = self.raw(X, S)
        m = self.meta
        npar = len(m["params"])
        out = []
        for i in range(len(w)):
            cands = []
            for j in np.argsort(-w[i]):
                x = from_z(mu[i, j, :npar], self.lb, self.ub)
                cands.append(dict(weight=float(w[i, j]), x=x, params=dict(zip(m["params"], map(float, x))),
                                  reported={nm: float(mu[i, j, npar + q]) for q, nm in enumerate(m["reported"])},
                                  reported_sd={nm: float(sd[i, j, npar + q]) for q, nm in enumerate(m["reported"])}))
            out.append(cands)
        return out

    def intervals(self, X, S, n=400, seed=0):
        'Mixture quantiles (2.5, 16, 50, 84, 97.5 %) of the reported quantities per sample, by sampling.'
        w, mu, sd = self.raw(X, S)
        rng = np.random.default_rng(seed)
        npar = len(self.meta["params"])
        q = np.zeros((len(w), len(self.meta["reported"]), 5))
        for i in range(len(w)):
            comp = rng.choice(len(w[i]), size=n, p=w[i] / w[i].sum())
            draw = mu[i, comp, npar:] + sd[i, comp, npar:] * rng.normal(size=(n, mu.shape[2] - npar))
            q[i] = np.percentile(draw, [2.5, 16, 50, 84, 97.5], axis=0).T
        return q


class TorchPredictor(_Base):
    def __init__(self, path, device="cpu"):
        import torch
        ck = torch.load(path, map_location=device, weights_only=False)
        self._setup(ck["meta"])
        m = self.meta
        self.net = build(m["n_in_ch"], m["n_scalar"], len(m["targets"]), m["k"], m.get("width", 32))
        self.net.load_state_dict(ck["state"])
        self.net.to(device).eval()
        self.device = device

    def _forward(self, xa, sa):
        import torch
        with torch.no_grad():
            lg, mu, ls = self.net(torch.tensor(xa).to(self.device), torch.tensor(sa).to(self.device))
        return lg.cpu().numpy().astype(float), mu.cpu().numpy().astype(float), ls.cpu().numpy().astype(float)


Predictor = TorchPredictor


def _conv3(x, W, b):
    '3x3 convolution, padding 1: x (n, ci, h, w), W (co, ci, 3, 3).'
    n, ci, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((n, ci, 3, 3, h, w), x.dtype)
    for i in range(3):
        for j in range(3):
            cols[:, :, i, j] = xp[:, :, i:i + h, j:j + w]
    return np.einsum("nckluv,ockl->nouv", cols, W, optimize=True) + b[None, :, None, None]


def _bn(x, g, b, m, v):
    return (x - m[None, :, None, None]) / np.sqrt(v[None, :, None, None] + BN_EPS) * g[None, :, None, None] + b[None, :, None, None]


def _maxpool2(x):
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, :2 * h2, :2 * w2].reshape(n, c, h2, 2, w2, 2).max((3, 5))


def _adaptive_avg(x, oh, ow):
    n, c, h, w = x.shape
    out = np.empty((n, c, oh, ow), x.dtype)
    for i in range(oh):
        a0, a1 = (i * h) // oh, -((-(i + 1) * h) // oh)
        for j in range(ow):
            b0, b1 = (j * w) // ow, -((-(j + 1) * w) // ow)
            out[:, :, i, j] = x[:, :, a0:a1, b0:b1].mean((2, 3))
    return out


class NumpyPredictor(_Base):
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self._setup(json.loads(str(z["meta"])))
        self.p = {k: z[k].astype(np.float64) for k in z.files if k != "meta"}

    def raw(self, X, S, batch=32):
        return super().raw(X, S, batch)

    def _block(self, x, pre):
        p = self.p
        for c, bn in ((f"{pre}.0", f"{pre}.1"), (f"{pre}.3", f"{pre}.4")):
            x = _conv3(x, p[f"{c}.weight"], p[f"{c}.bias"])
            x = np.maximum(_bn(x, p[f"{bn}.weight"], p[f"{bn}.bias"], p[f"{bn}.running_mean"], p[f"{bn}.running_var"]), 0)
        return x

    def _forward(self, xa, sa):
        p = self.p
        x = xa.astype(np.float64)
        x = _maxpool2(self._block(x, "f.0"))
        x = _maxpool2(self._block(x, "f.2"))
        x = _adaptive_avg(self._block(x, "f.4"), 3, 6).reshape(len(x), -1)
        s = np.maximum(sa.astype(np.float64) @ p["s.0.weight"].T + p["s.0.bias"], 0)
        h = np.concatenate([x, s], 1)
        h = np.maximum(h @ p["h.0.weight"].T + p["h.0.bias"], 0)
        h = np.maximum(h @ p["h.3.weight"].T + p["h.3.bias"], 0)
        o = h @ p["h.5.weight"].T + p["h.5.bias"]
        k, d = self.meta["k"], len(self.meta["targets"])
        return o[:, :k], o[:, k:k + k * d].reshape(-1, k, d), np.clip(o[:, k + k * d:].reshape(-1, k, d), -7, 3)


def _head(path):
    "Head type stored in the model meta ('flow' for iampred.flow; absent for the mixture-density network)."
    if str(path).endswith(".npz"):
        return json.loads(str(np.load(path, allow_pickle=False)["meta"])).get("head")
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["meta"].get("head")


def load(path, device="cpu"):
    if _head(path) == "flow":
        from . import flow
        return flow.FlowPredictor(path)
    return NumpyPredictor(path) if str(path).endswith(".npz") else TorchPredictor(path, device)


SAME_META = ("family", "params", "lb", "ub", "reported", "scalars", "features")


class Ensemble:
    'Several networks of one family (2026-11-21, results/iam/seedcmp_20261121: the same training with other seeds).'

    def __init__(self, members, k):
        self.members, self.k = members, int(k)
        self.meta = members[0].meta
        self.n_members = len(members)

    def candidates(self, X, S):
        per = [P.candidates(X, S) for P in self.members]
        return [[dict(c, weight=c["weight"] / self.n_members, member=m) for m, cs in enumerate(win) for c in cs[:self.k]]
                for win in zip(*per)]

    def intervals(self, X, S, n=400, seed=0):
        if all(hasattr(P, "samples") for P in self.members):
            nm = max(1, n // self.n_members)
            per = [P.samples(X, S, n=nm, seed=seed + i) for i, P in enumerate(self.members)]
            return np.array([np.percentile(np.concatenate([rs for _, rs in win]), [2.5, 16, 50, 84, 97.5], axis=0).T
                             for win in zip(*per)])
        return np.mean([P.intervals(X, S, n=n, seed=seed) for P in self.members], axis=0)


def member_paths(nn_dir):
    'The networks of a family folder: model.npz first, then model_*.npz in name order (a seed ensemble).'
    nn_dir = Path(nn_dir)
    extra = sorted(p for p in nn_dir.glob("model_*.npz"))
    return [nn_dir / "model.npz"] + extra


def load_many(paths, k, device="cpu"):
    'One path: that network (as load); several: an Ensemble, after checking that they are networks of the same'
    paths = [str(p) for p in paths]
    if len(paths) == 1:
        return load(paths[0], device)
    return ensemble_of([load(p, device) for p in paths], k, paths)


def ensemble_of(members, k, names=None):
    names = names or [str(i) for i in range(len(members))]
    def same(key, a, b):
        if key in ("lb", "ub") and a is not None and b is not None and len(a) == len(b):
            return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-12, atol=1e-12))
        return a == b
    for nm, P in zip(names[1:], members[1:]):
        bad = [key for key in SAME_META if not same(key, P.meta.get(key), members[0].meta.get(key))]
        if bad:
            raise SystemExit(f"networks of an ensemble differ in {bad}: {names[0]} / {nm}")
    return Ensemble(members, k)
