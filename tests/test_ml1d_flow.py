"""The proposal network's numpy side: the encoding of curves and parameters, and the flow against PyTorch (stored)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain.posterior import encoding
from src.gimap.features.fitting.domain.posterior.flow import Network, spline

DATA = Path(__file__).resolve().parent / "data" / "ml1d"
TOOLS = Path(__file__).resolve().parents[1] / "tools" / "ml1d"


def test_z_round_trip_and_jacobian():
    rng = np.random.default_rng(1)
    ratio = rng.uniform(10, 300, 50)
    lo, hi = encoding.bounds(ratio)
    values = lo + rng.uniform(0.01, 0.99, lo.shape) * (hi - lo)
    z = encoding.to_z(values, ratio)
    back, log_jacobian = encoding.from_z(z, ratio)
    assert np.allclose(back, values, rtol=1e-10, atol=1e-12)
    step = 1e-6
    numeric = (encoding.to_z(values + step, ratio) - encoding.to_z(values - step, ratio)) / (2 * step)
    assert np.allclose(log_jacobian, np.log(numeric), atol=1e-6)


def test_features_do_not_depend_on_the_scale():
    rng = np.random.default_rng(2)
    intensity = np.exp(rng.normal(size=(3, encoding.POINTS)))
    sigma = 0.1 * intensity
    mask = rng.uniform(size=intensity.shape) > 0.1
    one = encoding.features(intensity, sigma, mask)
    other = encoding.features(1e4 * intensity, 1e4 * sigma, mask)
    assert one.shape == (3, encoding.CHANNELS, encoding.POINTS)
    assert np.allclose(one, other, atol=1e-6)
    assert np.all(one[:, 0][~mask] == 0) and np.all(one[:, 2] == mask)


def test_conditions_and_active_coordinates():
    c = encoding.conditions(["sphere", "vertical_cylinder"], [1, 0], [10, 300], [0.5, 0.0], [0.0, 0.01])
    assert c.shape == (2, encoding.CONDITIONS)
    assert list(c[:, 0]) == [1, 0] and list(c[:, 1]) == [0, 1]
    assert c[1, 3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        encoding.conditions(["cone"], [0], [10])
    active = encoding.active([True, False])
    assert active[0].all() and not active[1, list(encoding.STRUCTURE_DIMS)].any()


def test_binning_keeps_the_counts_weighting():
    q = np.linspace(0.01, 1.0, 2000)
    intensity = 1.0 / q
    sigma = 0.05 * intensity
    out_i, out_s, mask = encoding.bin_curve(q, intensity, sigma, q_max=1.0, ratio=100)
    centres = encoding.grid(100)
    assert mask[centres > 0.012].all()
    assert np.allclose(out_i[mask] * centres[mask], 1.0, rtol=0.03)
    assert np.all(out_s[mask] <= sigma.max())


def test_spline_inverts_and_is_monotone():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(200, 1, 3 * 6 - 1))
    x = np.sort(rng.uniform(-4, 4, 200))[:, None]
    y, log_det = spline(x, raw[:1].repeat(200, axis=0), 3.0, 6)
    assert np.all(np.diff(y[:, 0]) > 0)
    back, inverse_det = spline(y, raw[:1].repeat(200, axis=0), 3.0, 6, inverse=True)
    assert np.allclose(back, x, atol=1e-9)
    assert np.allclose(log_det, -inverse_det, atol=1e-9)
    h = 1e-6
    numeric = (spline(x + h, raw[:1].repeat(200, axis=0), 3.0, 6)[0] - spline(x - h, raw[:1].repeat(200, axis=0), 3.0,
                                                                              6)[0]) / (2 * h)
    inside = np.abs(x) < 3.0 - 1e-5
    assert np.allclose(np.log(numeric[inside]), log_det[inside], atol=1e-5)


def test_numpy_flow_equals_pytorch():
    """The stored network (tools/ml1d/make_flow_fixture.py, random weights) gives PyTorch's numbers in numpy."""
    network = Network.load(DATA / "flow_fixture.npz")
    with np.load(DATA / "flow_fixture_reference.npz") as data:
        ref = {k: data[k] for k in data.files}
    context = network.context(ref["features"], ref["conditions"])
    assert np.allclose(context, ref["context"], rtol=1e-9, atol=1e-12)
    assert np.allclose(network.log_prob(ref["z"], context), ref["log_prob"], rtol=1e-9, atol=1e-9)
    sample, log_det = network.inverse(ref["u"], context)
    assert np.allclose(sample, ref["sample"], rtol=1e-9, atol=1e-9)
    assert np.allclose(log_det, ref["log_det"], rtol=1e-9, atol=1e-9)
    u, forward_det = network.forward(sample, context)
    assert np.allclose(u, ref["u"], atol=1e-8) and np.allclose(forward_det, -log_det, atol=1e-8)


def test_samples_carry_their_density():
    network = Network.load(DATA / "flow_fixture.npz")
    with np.load(DATA / "flow_fixture_reference.npz") as data:
        context = network.context(data["features"][:1], data["conditions"][:1])
    z, log_q = network.sample(context, 64, np.random.default_rng(4))
    assert z.shape == (64, len(encoding.NAMES))
    assert np.allclose(log_q, network.log_prob(z, np.repeat(context, 64, axis=0)), atol=1e-8)


def test_a_changed_network_file_is_refused(tmp_path):
    for name in ("flow_fixture.npz", "flow_fixture.json"):
        shutil.copy(DATA / name, tmp_path / name)
    meta = json.loads((tmp_path / "flow_fixture.json").read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    (tmp_path / "flow_fixture.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        Network.load(tmp_path / "flow_fixture.npz")


def test_simulator_rows_are_reproducible_and_in_the_prior():
    sys.path.insert(0, str(TOOLS))
    try:
        import simulate
    finally:
        sys.path.remove(str(TOOLS))
    prior = simulate.load_prior(TOOLS / "prior_k1.json")
    one, two = simulate.shard(prior, 40, 7), simulate.shard(prior, 40, 7)
    assert all(np.array_equal(one[k], two[k]) for k in one)
    lo, hi = encoding.bounds(one["ratio"])
    assert np.all((one["values"] >= lo - 1e-5) & (one["values"] <= hi + 1e-5))
    assert np.all(np.isfinite(one["term"])) and np.all(one["term"] > 0)
    features, conditions, z, active = simulate.inputs(one, np.arange(40), prior, np.random.default_rng(0))
    assert features.shape == (40, encoding.CHANNELS, encoding.POINTS) and np.all(np.isfinite(features))
    assert conditions.shape == (40, encoding.CONDITIONS) and np.all(np.isfinite(z))
