"""Fast variable-projection refinement for the frozen multishape_v1 physics.

Same model as physics.np_particle (default 64/128 quadrature), two speed-ups:

1. Random-cylinder orientation average. The radial factor depends on q only
   through q' = q*sqrt(1-c^2) and the height factor through q'' = q*c, so each
   size-averaged factor is tabulated once on a fine 1D grid and interpolated at
   all (angle, q) pairs, instead of evaluating J1/sinc on the full
   size x angle x q grid.
2. Component caching for finite differences. Each slot column is
   radial/height form factor x structure factor; a perturbation recomputes only
   the factor it touches (D, sigma_D -> structure factor only; h, sigma_h ->
   height factor only; logR -> radial factor and structure factor, because the
   D codec depends on R).

Accuracy against physics.np_particle is checked by check_accuracy().
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import j1

from research.multishape_v1 import physics
from research.conditional_fit.codec import LEGACY

TABLE_MIN, TABLE_MAX = 512, 4096
POWERLAW_N = (1.0, 4.0)   # exponent range of the optional low-q power law (lowq='powerlaw')


def powerlaw_exponent(u):
    """Exponent n of the low-q power law C*(q/q_min)^-n encoded in u[31]."""
    return POWERLAW_N[0]+(POWERLAW_N[1]-POWERLAW_N[0])*float(np.clip(u[31], 0, 1))


def powerlaw_u(n):
    """u[31] for a power-law exponent n (inverse of powerlaw_exponent)."""
    return float(np.clip((n-POWERLAW_N[0])/(POWERLAW_N[1]-POWERLAW_N[0]), 0, 1))
TABLE_STEP = 0.04   # grid spacing times the largest size in the distribution (nm^-1 * nm)


def _table_points(qmax, size_max):
    """Adaptive 1D table: spacing ~ TABLE_STEP / size_max resolves the oscillation of the factor."""
    return int(np.clip(np.ceil(qmax*size_max/TABLE_STEP), TABLE_MIN, TABLE_MAX))


def _radial(x):
    safe = np.where(np.abs(x) < 1e-4, 1.0, x)
    return np.where(np.abs(x) < 1e-4, 1-x*x/8+x**4/192, 2*j1(safe)/safe)


def _sphere_amp(x):
    safe = np.where(np.abs(x) < .1, 1.0, x)
    amp = 3*(np.sin(safe)-safe*np.cos(safe))/safe**3
    return np.where(np.abs(x) < .1, 1-x*x/10+x**4/280-x**6/15120, amp)


class Physics:
    """Per-q-grid evaluator of the separable slot factors."""

    def __init__(self, q, order=physics.DEFAULT_ORDER, angles=physics.DEFAULT_ANGLES, qz=0.0):
        """q is the in-plane momentum (q_par; |q| for SAXS). qz is the fixed out-of-plane
        momentum of a Yoneda cut: spheres and randomly oriented cylinders see
        |q| = sqrt(q_par^2 + qz^2); vertical-cylinder cross sections and structure
        factors depend on q_par only (the qz factor of a vertical cylinder is a
        constant along the cut and is absorbed by its free amplitude)."""
        self.q = np.abs(np.asarray(q, float))
        self.qz = float(qz)
        self.qt = np.hypot(self.q, self.qz)
        self.order = order
        xa, wa = physics.quadrature(angles)
        self.c, self.ow = (xa+1)/2, wa/2
        self.s = np.sqrt(1-self.c**2)
        self.qmax = float(self.qt.max())*1.000001
        self.qs = self.qt[None, :]*self.s[:, None]   # [angles, Q]
        self.qc = self.qt[None, :]*self.c[:, None]

    def radial(self, typ, r, sr):
        rr, wr = physics._gaussian(r, sr, self.order)
        if typ == 1:
            return wr @ (_sphere_amp(rr[:, None]*self.qt[None, :])**2)
        if typ == 3:
            return wr @ (_radial(rr[:, None]*self.q[None, :])**2)
        grid = np.linspace(0.0, self.qmax, _table_points(self.qmax, rr.max()))
        table = wr @ (_radial(rr[:, None]*grid[None, :])**2)
        return np.interp(self.qs, grid, table)                # fm [angles, Q]

    def height(self, h, sh):
        hh, wh = physics._gaussian(h, sh, self.order)
        grid = np.linspace(0.0, self.qmax, _table_points(self.qmax, hh.max()))
        table = wh @ (np.sinc(hh[:, None]*grid[None, :]/(2*np.pi))**2)
        return np.interp(self.qc, grid, table)                 # hm [angles, Q]

    def structure(self, distance, sd):
        q = self.q
        lp = -np.pi*q*q*(distance*sd)**2
        phi = np.exp(lp)
        sf = -np.expm1(2*lp)/np.maximum(np.expm1(lp)**2+4*phi*np.sin(0.5*q*distance)**2, 1e-30)
        return np.where(q == 0, 2*np.pi*sd*sd, sf)

    def combine(self, typ, radial, height):
        if typ == 2:
            return self.ow @ (radial*height)
        return radial


class FastFit:
    """Bounded variable projection with per-factor caching.

    The linear coefficients (A1..A4, B, C) are solved by ridged NNLS with the
    serialized measurement sigma, exactly as identifiability.run.Fit.solve.
    """

    def __init__(self, q, y, sigma, types, order=physics.DEFAULT_ORDER, angles=physics.DEFAULT_ANGLES,
                 *, qz=0.0, sf_on=True, res_on=True, codec=LEGACY, lowq='lorentz'):
        """sf_on=False sets S(q)=1 for every population (dilute SAXS); res_on=False drops the
        additive low-q term. Both remove the corresponding coordinates from the fit.
        lowq selects the additive low-q column C*g(q): 'lorentz' (default, the "resolution function"
        1/(1+(q/sigma_res)^nu), u[30] -> sigma_res, u[31] -> nu through the codec) or 'powerlaw'
        ((q/q_min)^-n with n = POWERLAW_N[0] + (POWERLAW_N[1]-POWERLAW_N[0]) u[31]; u[30] unused), a
        Porod/aggregate background that can replace an aggregate population."""
        self.q, self.y, self.sigma = (np.asarray(v, float) for v in (q, y, sigma))
        self.types = np.asarray(types, int)
        self.ph = Physics(self.q, order, angles, qz=qz)
        self.sf_on, self.res_on = bool(sf_on), bool(res_on)
        if lowq not in ('lorentz', 'powerlaw'):
            raise ValueError(f"lowq must be 'lorentz' or 'powerlaw', not {lowq!r}")
        self.lowq = lowq
        self.q_ref = float(np.min(self.q[self.q > 0])) if np.any(self.q > 0) else 1.0
        self.codec = codec
        active = physics.active_mask(self.types).astype(bool)
        active[[6, 13, 20, 27, 28, 29]] = False
        if not self.sf_on:
            for j in range(4):
                active[[7*j+4, 7*j+5]] = False
        if not self.res_on:
            active[[30, 31]] = False
        elif lowq == 'powerlaw':
            active[30] = False
        self.coords = np.flatnonzero(active)
        self.evaluations = 0

    # --- slot state -------------------------------------------------------
    def slot_state(self, u, j):
        p = self.codec.decode(u)
        typ = int(self.types[j])
        radial = self.ph.radial(typ, p['R'][j], p['sigma_R'][j])
        height = self.ph.height(p['h'][j], p['sigma_h'][j]) if typ == 2 else None
        sf = self.ph.structure(p['D'][j], p['sigma_D'][j]) if self.sf_on else np.ones_like(self.q)
        return dict(radial=radial, height=height, sf=sf, ff=self.ph.combine(typ, radial, height))

    def resolution(self, u):
        if not self.res_on:
            return np.zeros_like(self.q)
        if self.lowq == 'powerlaw':
            return (np.maximum(self.q, self.q_ref)/self.q_ref)**(-powerlaw_exponent(u))
        p = self.codec.decode(u)
        return 1/(1+(self.q/p['sigma_res'])**p['nu_res'])

    def full(self, u):
        states = [self.slot_state(u, j) if self.types[j] else None for j in range(4)]
        return states, self.resolution(u)

    def matrix_from(self, states, res):
        m = np.zeros((len(self.q), 6))
        for j, st in enumerate(states):
            if st is not None:
                m[:, j] = st['ff']*st['sf']
        m[:, 4] = 1.0
        m[:, 5] = res
        return m

    def perturbed_column(self, u2, j, k, st):
        """Column j after changing slot coordinate k (0..5) of u; reuses cached factors."""
        typ = int(self.types[j])
        p = self.codec.decode(u2)
        radial, height, sf = st['radial'], st['height'], st['sf']
        if k in (0, 1):
            radial = self.ph.radial(typ, p['R'][j], p['sigma_R'][j])
        if k in (2, 3):
            height = self.ph.height(p['h'][j], p['sigma_h'][j])
        if k in (0, 4, 5) and self.sf_on:
            sf = self.ph.structure(p['D'][j], p['sigma_D'][j])
        ff = st['ff'] if k in (4, 5) else self.ph.combine(typ, radial, height)
        return ff*sf

    # --- linear solve (identical to identifiability.run.Fit.solve) ---------
    def solve(self, m):
        design, response = m/self.sigma[:, None], self.y/self.sigma
        scale = np.linalg.norm(design, axis=0)
        valid = scale > 1e-25
        coef = np.zeros(6)
        if valid.any():
            ridge = 1e-8
            a = np.vstack((design[:, valid]/scale[valid], np.sqrt(ridge)*np.eye(valid.sum())))
            coef[valid] = nnls(a, np.r_[response, np.zeros(valid.sum())], maxiter=2000)[0]/scale[valid]
        prediction = m@coef
        return coef, prediction, (prediction-self.y)/self.sigma

    # --- optimizer ----------------------------------------------------------
    def run(self, u0, *, fixed=(), lower=None, upper=None, max_nfev=60, fd_step=1e-4, cov_step=1e-5, jac_mode='forward'):
        """Refine nonlinear coordinates from u0. lower/upper: {u index: bound} narrowing [0,1].
        cov_step: step of the central-difference Jacobian used for cov_u at the solution (0/None: the last
        one-sided LM Jacobian, the behaviour before 2026-10-01). jac_mode: 'forward' (one-sided LM Jacobian,
        default) or 'central' (central differences with fd_step in every LM iteration; 2x the model evaluations)."""
        u0 = np.asarray(u0, float).copy()
        fixed = {int(v) for v in fixed}
        coords = np.array([c for c in self.coords if c not in fixed], int)
        lo, hi = np.zeros(32), np.ones(32)
        for key, value in (lower or {}).items():
            lo[key] = value
        for key, value in (upper or {}).items():
            hi[key] = value
        u0 = np.clip(u0, lo, hi)
        width = np.maximum(hi[coords]-lo[coords], 1e-12)
        cache = {}

        def to_u(x):
            u = u0.copy()
            u[coords] = lo[coords]+x*width
            return u

        def state(x):
            key = x.tobytes()
            if key not in cache:
                u = to_u(x)
                states, res = self.full(u)
                m = self.matrix_from(states, res)
                cache.clear()
                cache[key] = (u, states, res, m, *self.solve(m))
                self.evaluations += 1
            return cache[key]

        def fun(x):
            return state(x)[6]

        def jac(x):
            if jac_mode == 'central':
                return jac_central(x, fd_step)
            u, states, res, m, _, _, r0 = state(x)
            out = np.empty((len(r0), len(x)))
            for i, c in enumerate(coords):
                step = fd_step if x[i] <= .5 else -fd_step
                u2 = u.copy()
                u2[c] += step*width[i]
                m2 = m.copy()
                if c >= 28:
                    m2[:, 5] = self.resolution(u2)
                else:
                    j, k = divmod(int(c), 7)
                    m2[:, j] = self.perturbed_column(u2, j, k, states[j])
                out[:, i] = (self.solve(m2)[2]-r0)/step
                self.evaluations += 1
            return out

        def jac_central(x, h=cov_step):
            """Central-difference Jacobian at the solution (one-sided only where x +- h leaves [0, 1]); the
            one-sided LM Jacobian with fd_step 1e-4 biases the covariance of weakly constrained coordinates."""
            u, states, res, m, _, _, r0 = state(x)
            out = np.empty((len(r0), len(x)))
            for i, c in enumerate(coords):
                sides = []
                for sign in (1.0, -1.0):
                    if not 0.0 <= x[i]+sign*h <= 1.0:
                        sides.append(None)
                        continue
                    u2 = u.copy()
                    u2[c] += sign*h*width[i]
                    m2 = m.copy()
                    if c >= 28:
                        m2[:, 5] = self.resolution(u2)
                    else:
                        j, k = divmod(int(c), 7)
                        m2[:, j] = self.perturbed_column(u2, j, k, states[j])
                    sides.append(self.solve(m2)[2])
                    self.evaluations += 1
                plus, minus = sides
                if plus is not None and minus is not None:
                    out[:, i] = (plus-minus)/(2*h)
                elif plus is not None:
                    out[:, i] = (plus-r0)/h
                elif minus is not None:
                    out[:, i] = (r0-minus)/h
                else:
                    out[:, i] = 0.0
            return out

        started = time.perf_counter()
        if len(coords):
            # Start strictly inside the box: scipy's trf rescales v = x - lb by the Jacobian scale, and a
            # start exactly on a bound (warm starts, 2026-10-01) underflows v to 0 and yields NaN steps.
            x0 = np.clip((u0[coords]-lo[coords])/width, 1e-9, 1-1e-9)
            result = least_squares(fun, x0, jac=jac, bounds=(0., 1.), max_nfev=max_nfev,
                                   x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8)
            x, status = result.x, int(result.status)
            jac_x = jac_central(x) if cov_step else result.jac
        else:
            x, status, jac_x = np.zeros(0), 1, np.zeros((len(self.y), 0))
        u, states, res, m, coef, prediction, residual = state(x)
        chi2 = float(residual@residual)
        # Gauss-Newton covariance of the free nonlinear coordinates (amplitudes profiled out),
        # scaled by the reduced chi2; coordinates at a bound are reported but not trusted.
        n_linear = int(np.sum(coef > 0))
        dof = max(len(self.y)-len(coords)-n_linear, 1)
        j_u = jac_x[:len(self.y)]/width[None, :] if len(coords) else jac_x
        cov_u = np.linalg.pinv(j_u.T@j_u)*chi2/dof if len(coords) else np.zeros((0, 0))
        at_bound = [int(c) for c, xi in zip(coords, x) if xi < 1e-6 or xi > 1-1e-6]
        return dict(u=u, coef=coef, prediction=prediction, chi2=chi2, status=status,
                    evaluations=self.evaluations, seconds=time.perf_counter()-started,
                    coords=coords.tolist(), cov_u=cov_u, at_bound=at_bound, dof=dof)


def check_accuracy(n=200, seed=0):
    """Max lnRMSE between FastFit columns and physics.np_particle over random u and grids."""
    from research.multishape_v1.data import native_grid
    rng = np.random.default_rng(seed)
    worst = {1: 0.0, 2: 0.0, 3: 0.0}
    for _ in range(n):
        q, mask, _, _ = native_grid(rng)
        q = q[mask]
        u = rng.uniform(size=32)
        p = physics.decode(u)
        ph = Physics(q)
        for typ in (1, 2, 3):
            ref = physics.np_particle(q, typ, p['R'][0], p['sigma_R'][0], p['h'][0], p['sigma_h'][0],
                                      p['D'][0], p['sigma_D'][0])
            radial = ph.radial(typ, p['R'][0], p['sigma_R'][0])
            height = ph.height(p['h'][0], p['sigma_h'][0]) if typ == 2 else None
            fast = ph.combine(typ, radial, height)*ph.structure(p['D'][0], p['sigma_D'][0])
            err = float(np.sqrt(np.mean((np.log(fast)-np.log(ref))**2)))
            worst[typ] = max(worst[typ], err)
    return worst
