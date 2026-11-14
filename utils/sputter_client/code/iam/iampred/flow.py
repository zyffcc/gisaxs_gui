'Posterior head of the predictor (posterior model v1, 2026-11-13): window features -> a distribution over the full'

from __future__ import annotations

import json

import numpy as np

from . import nnmodel as NM

SMAX = 2.0
MERGE_TOL = dict(ln=0.10, cov=0.04)
MIN_WEIGHT = 0.02


def make_masks(npar, layers, seed=0):
    'Alternating random halves: layer 2j uses a random half, layer 2j + 1 its complement.'
    rng = np.random.default_rng(seed)
    out = []
    for l in range(layers):
        if l % 2 == 0:
            b = np.zeros(npar)
            b[rng.permutation(npar)[: npar // 2]] = 1.0
        else:
            b = 1.0 - out[-1]
        out.append(b)
    return np.array(out, np.float32)


def build(n_in_ch, n_scalar, npar, nrep, k, masks, width=32, hidden=128, ctx=64):
    import torch
    from torch import nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.f = nn.Sequential(blk(n_in_ch, w), nn.MaxPool2d(2), blk(w, 2 * w), nn.MaxPool2d(2), blk(2 * w, 4 * w),
                                   nn.AdaptiveAvgPool2d((3, 6)), nn.Flatten())
            self.s = nn.Sequential(nn.Linear(2 * n_scalar, 64), nn.ReLU())
            self.h = nn.Sequential(nn.Linear(4 * w * 18 + 64, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.ReLU())
            self.base = nn.Linear(512, k * (1 + 2 * npar))
            self.cp = nn.Linear(512, ctx)
            self.c = nn.ModuleList([nn.Sequential(nn.Linear(npar + ctx, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                                  nn.ReLU(), nn.Linear(hidden, 2 * npar)) for _ in range(len(masks))])
            for m in self.c:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
            self.register_buffer("masks", torch.tensor(np.asarray(masks, np.float32)))
            self.r = nn.Sequential(nn.Linear(npar + 2 * n_scalar, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, nrep))
            self.k, self.d = k, npar

        def context(self, x, s):
            return self.h(torch.cat([self.f(x), self.s(s)], 1))

        def base_params(self, c):
            o = self.base(c)
            k, d = self.k, self.d
            return o[:, :k], o[:, k:k + k * d].reshape(-1, k, d), o[:, k + k * d:].reshape(-1, k, d).clamp(-7, 3)

        def inverse(self, z, c):
            'Data -> base coordinates; returns (u, sum of the log scales).'
            cp, v, ld = self.cp(c), z, 0.0
            for l in reversed(range(len(self.c))):
                b = self.masks[l]
                st = self.c[l](torch.cat([v * b, cp], 1))
                sc = SMAX * torch.tanh(st[:, :self.d])
                v = v * b + (1 - b) * ((v - st[:, self.d:]) * torch.exp(-sc))
                ld = ld + ((1 - b) * sc).sum(1)
            return v, ld

        def nll(self, x, s, y):
            c = self.context(x, s)
            u, ld = self.inverse(y, c)
            logit, mu, ls = self.base_params(c)
            lp = (-0.5 * ((u[:, None, :] - mu) / ls.exp()) ** 2 - ls - 0.9189385332).sum(2)
            return -(torch.logsumexp(torch.log_softmax(logit, 1) + lp, 1) - ld)

        def reported(self, y, s):
            return self.r(torch.cat([y, s], 1))

    return Net()


def _lin(p, key, x):
    return x @ p[f"{key}.weight"].T + p[f"{key}.bias"]


class FlowPredictor(NM.NumpyPredictor):
    'model.pt or model.npz of the flow head; numpy inference.'

    def __init__(self, path):
        if str(path).endswith(".npz"):
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            self.p = {k: z[k].astype(np.float64) for k in z.files if k != "meta"}
        else:
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            meta = ck["meta"]
            self.p = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in ck["state"].items() if v.dtype.is_floating_point}
        self._setup(meta)
        self.rm, self.rs = np.asarray(meta["rm"], float), np.asarray(meta["rs"], float)
        self.masks = np.asarray(meta["masks"], float)
        self.opt = np.array([nm in meta.get("optional_scalars", []) for nm in meta["scalars"]], bool)


    def prep(self, X, S):
        X = np.array(X, np.float32, copy=True)
        nc = X.shape[1] - 1
        X[:, :nc] = (X[:, :nc] - self.xm[None, :, None, None]) / self.xs[None, :, None, None]
        X[:, :nc] *= X[:, nc:nc + 1]
        S = np.asarray(S, float)
        given = np.isfinite(S)
        if np.any(~given & ~self.opt[None, :]):
            raise ValueError("a required scalar input is missing (NaN): " + str([n for n, o in zip(self.meta["scalars"], self.opt) if not o]))
        sn = np.where(given, (np.where(given, S, 0.0) - self.sm) / self.ss, 0.0)
        return X, np.concatenate([sn, given.astype(float)], 1).astype(np.float32)

    def _context(self, xa, sa):
        p = self.p
        x = xa.astype(np.float64)
        x = NM._maxpool2(self._block(x, "f.0"))
        x = NM._maxpool2(self._block(x, "f.2"))
        x = NM._adaptive_avg(self._block(x, "f.4"), 3, 6).reshape(len(x), -1)
        s = np.maximum(_lin(p, "s.0", sa.astype(np.float64)), 0)
        h = np.maximum(_lin(p, "h.0", np.concatenate([x, s], 1)), 0)
        return np.maximum(_lin(p, "h.3", h), 0)

    def _base(self, c):
        o = _lin(self.p, "base", c)
        k, d = self.meta["k"], len(self.meta["params"])
        lg = o[:, :k] - o[:, :k].max(1, keepdims=True)
        w = np.exp(lg)
        return w / w.sum(1, keepdims=True), o[:, k:k + k * d].reshape(-1, k, d), np.exp(np.clip(o[:, k + k * d:].reshape(-1, k, d), -7, 3))

    def _flow(self, u, cp):
        'Base -> data coordinates. u (n, m, d), cp (n, ctx).'
        p, d = self.p, u.shape[-1]
        v = u
        cpb = np.broadcast_to(cp[:, None, :], u.shape[:2] + (cp.shape[1],))
        for l in range(len(self.masks)):
            b = self.masks[l]
            h = np.maximum(_lin(p, f"c.{l}.0", np.concatenate([v * b, cpb], -1)), 0)
            h = np.maximum(_lin(p, f"c.{l}.2", h), 0)
            st = _lin(p, f"c.{l}.4", h)
            v = v * b + (1 - b) * (v * np.exp(SMAX * np.tanh(st[..., :d])) + st[..., d:])
        return v

    def _rep(self, zn, sa):
        'Reported quantities (own units) of normalised parameter vectors zn (n, m, d).'
        p = self.p
        sb = np.broadcast_to(sa[:, None, :].astype(np.float64), zn.shape[:2] + (sa.shape[1],))
        h = np.maximum(_lin(p, "r.0", np.concatenate([zn, sb], -1)), 0)
        h = np.maximum(_lin(p, "r.2", h), 0)
        return _lin(p, "r.4", h) * self.rs + self.rm

    def sample(self, X, S, n=256, seed=0, batch=32):
        'Per window: mixture weights (k), candidate vectors = flow(mixture means) (k, d) normalised, n posterior'
        xa, sa = self.prep(X, S)
        rng = np.random.default_rng(seed)
        out = []
        for b0 in range(0, len(xa), batch):
            c = self._context(xa[b0:b0 + batch], sa[b0:b0 + batch])
            w, mu, sd = self._base(c)
            cp = _lin(self.p, "cp", c)
            comp = np.stack([rng.choice(w.shape[1], size=n, p=wi) for wi in w])
            u = np.take_along_axis(mu, comp[:, :, None], 1) + np.take_along_axis(sd, comp[:, :, None], 1) * rng.normal(size=(len(w), n, mu.shape[2]))
            zc, zs = self._flow(mu, cp), self._flow(u, cp)
            sb = sa[b0:b0 + batch]
            rc, rsamp = self._rep(zc, sb), self._rep(zs, sb)
            for i in range(len(w)):
                out.append(dict(w=w[i], zc=zc[i], zs=zs[i], comp=comp[i], rc=rc[i], rs=rsamp[i]))
        return out

    def _x(self, zn):
        return NM.from_z(zn * self.ts + self.tm, self.lb, self.ub)

    def candidates(self, X, S, n=256, seed=0):
        m = self.meta
        names, rep = m["params"], m["reported"]
        merge = [rep.index(q) for q in m.get("merge", ["lnR", "lnD", "cov", "lnT"]) if q in rep]
        out = []
        for d in self.sample(X, S, n=n, seed=seed):
            k = len(d["w"])
            sdk = np.array([d["rs"][d["comp"] == j].std(0) if np.sum(d["comp"] == j) > 3 else d["rs"].std(0) for j in range(k)])
            kept = []
            for j in np.argsort(-d["w"]):
                for q in kept:
                    i = q[0]
                    tol = np.array([max(MERGE_TOL["ln"] if rep[r].startswith("ln") else MERGE_TOL.get(rep[r], 0.0),
                                        0.5 * min(sdk[i, r], sdk[j, r])) for r in merge])
                    if merge and np.all(np.abs(d["rc"][i, merge] - d["rc"][j, merge]) <= tol):
                        q[1] += float(d["w"][j])
                        break
                else:
                    kept.append([int(j), float(d["w"][j])])
            kept = sorted(kept, key=lambda q: -q[1])
            kept = [q for i, q in enumerate(kept) if i == 0 or q[1] >= MIN_WEIGHT]
            cands = []
            for j, wj in kept:
                x = self._x(d["zc"][j])
                cands.append(dict(weight=wj, x=x, params=dict(zip(names, map(float, x))),
                                  reported={nm: float(d["rc"][j, r]) for r, nm in enumerate(rep)},
                                  reported_sd={nm: float(sdk[j, r]) for r, nm in enumerate(rep)}))
            out.append(cands)
        return out

    def intervals(self, X, S, n=400, seed=0):
        q = [np.percentile(d["rs"], [2.5, 16, 50, 84, 97.5], axis=0).T for d in self.sample(X, S, n=n, seed=seed)]
        return np.array(q)

    def samples(self, X, S, n=256, seed=0):
        'Posterior samples per window: (fitter-unit parameter vectors (n, d), reported (n, nrep)).'
        return [(self._x(d["zs"]), d["rs"]) for d in self.sample(X, S, n=n, seed=seed)]

def check(model_pt, shard, n=64):
    'Consistency of the two code paths on n windows of a shard: the torch network maps the true parameter vectors'
    import torch
    ck = torch.load(model_pt, map_location="cpu", weights_only=False)
    m = ck["meta"]
    net = build(m["n_in_ch"], m["n_scalar"], len(m["params"]), len(m["reported"]), m["k"], np.array(m["masks"]),
                m.get("width", 32), m.get("hidden", 128))
    net.load_state_dict(ck["state"])
    net.eval()
    P = FlowPredictor(model_pt)
    z = np.load(shard, allow_pickle=False)
    X, S = z["X"][:n].astype(np.float32), z["S"][:n].astype(np.float32)
    xa, sa = P.prep(X, S)
    y = (NM.to_z(z["P"][:n], P.lb, P.ub) - P.tm) / P.ts
    with torch.no_grad():
        c = net.context(torch.tensor(xa), torch.tensor(sa))
        u, _ = net.inverse(torch.tensor(y.astype(np.float32)), c)
    cn = P._context(xa, sa)
    y2 = P._flow(u.numpy().astype(float)[:, None, :], _lin(P.p, "cp", cn))[:, 0, :]
    return float(np.max(np.abs(c.numpy() - cn))), float(np.max(np.abs(y - y2)))


if __name__ == "__main__":
    import sys
    dc, dz = check(sys.argv[1], sys.argv[2])
    print(f"flow check: max |context torch - numpy| {dc:.2e}, max |z - flow(inverse(z))| {dz:.2e}")
