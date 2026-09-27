"""The diagnostic ablations must really preserve the frozen forward equation."""

import numpy as np
import pytest
from tools.audit_cbf_causes import Model


@pytest.mark.parametrize(
    "types,gates", [([1], [True]), ([2], [False]), ([1, 2, 3], [True, False, True])]
)
def test_train_exact_matches_frozen_reference(types, gates):
    q = np.linspace(0.001, 4, 80)
    y = np.linspace(100, 1, 80)
    model = Model(dict(q=q, y=y, count=np.ones(80) * 6), "train_exact", types, gates)
    z = (model.lo + model.hi) / 2
    a, parts = model.decode(z)
    from referenced_forward64 import referenced_forward
    from flow_codec import COMBOS

    k = len(types)
    params = np.zeros((1, 1, 4, 6))
    weights = np.zeros((1, 1, 4))
    d = np.zeros((1, 1, 4))
    for i, p in enumerate(parts):
        params[0, 0, i] = [
            np.log(p["R"]) / np.log(100),
            (p["sigma_R"] - 0.02) / 0.88,
            np.log(p["h"] / 2) / np.log(250),
            (p["sigma_h"] - 0.02) / 0.88,
            np.log(p["D"] / 3) / np.log(500 / 3),
            (p["sigma_D"] - 0.05) / 0.85,
        ]
        weights[0, 0, i] = a.get(f"w{i}", 0)
        d[0, 0, i] = float(gates[i])
    glob = np.array(
        [
            [
                (a["bg"] - np.log(1e-6)) / np.log(1e4),
                (a["rs"] - np.log(0.007)) / np.log(0.013 / 0.007),
                (a["nu"] - 5) / 5,
                (a["rc"] - np.log(10)) / np.log(100),
            ]
        ]
    )[:, None, :]
    combo = np.flatnonzero(np.all(COMBOS == np.array(types + [0] * (4 - k)), axis=1))[0]
    candidate = dict(
        params=params,
        weights=weights,
        d=d,
        globals=glob,
        res=np.ones((1, 1)),
        combos=np.array([[combo]]),
    )
    np.testing.assert_allclose(
        model.predict(z), referenced_forward(candidate, 0, 0, q, q) * model.norm, rtol=1e-11
    )


def test_gauss_orientation_matches_independent_adaptive_integral():
    from scipy.integrate import quad
    from tools.audit_gauss_forward import component, radial

    r, sr, h, sh = 1.6, 0.23, 12.0, 0.25

    def average(mu, width, fn):
        lo = max(mu - 4 * width, 0)
        hi = mu + 4 * width
        norm = quad(lambda v: np.exp(-0.5 * ((v - mu) / width) ** 2), lo, hi, epsabs=1e-11)[0]
        return (
            quad(lambda v: np.exp(-0.5 * ((v - mu) / width) ** 2) * fn(v), lo, hi, epsabs=1e-11)[0]
            / norm
        )

    expected = []
    for q in (0.1, 1.0, 4.0):

        def integrand(u):
            a = average(
                r, r * sr, lambda rad: float(radial(np.array(q * rad * np.sqrt(1 - u * u)))) ** 2
            )
            b = average(h, h * sh, lambda hh: np.sinc(q * hh * u / (2 * np.pi)) ** 2)
            return a * b

        expected.append(quad(integrand, 0, 1, epsabs=1e-10, epsrel=1e-9)[0])
    actual = component(
        np.array([0.1, 1.0, 4.0]), 2, r, sr, h, sh, 4, 0.1, False, order=48, orientation_order=96
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)
