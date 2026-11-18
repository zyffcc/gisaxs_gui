"""The proposal network's numpy side: the encoding of curves, compositions and parameters, and the flow against
PyTorch (stored)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain.cut_model import model, resolution
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior.flow import Network, spline

DATA = Path(__file__).resolve().parent / "data" / "ml1d"
TOOLS = Path(__file__).resolve().parents[1] / "tools" / "ml1d"


def _random_values(count, seed):
    rng = np.random.default_rng(seed)
    composition = rng.integers(0, len(E.COMPOSITIONS), count)
    ratio = np.exp(rng.uniform(np.log(10), np.log(300), count))
    return composition, ratio, E.from_unit(rng.uniform(0.001, 0.999, (count, E.DIMS)), ratio, composition)


def test_compositions_and_their_values():
    assert len(E.COMPOSITIONS) == 6 + 36 + 216 and E.DIMS == 22
    index = E.composition_index([("sphere", True), ("cylinder", False)])
    assert E.composition_types(index) == (("sphere", True), ("cylinder", False))
    assert E.composition_name(index) == "sphere (S) + random cylinder"
    on = E.active(index)[0]
    names = np.array(E.NAMES)
    assert set(names[on]) == {"R_1", "sigma_R_1", "D_over_2R_1", "sigma_nn_1", "R_2", "sigma_R_2", "h_over_2R_2",
                              "sigma_h_2", *E.GLOBAL_NAMES}
    assert len(E.select(("sphere", "vertical_cylinder"), 2, structure=True)) == 2 + 4
    assert len(E.select(max_populations=1)) == 6 and len(E.select()) == len(E.COMPOSITIONS)
    code = E.composition_code(index)[0].reshape(E.SLOTS, E.SLOT_CODE)
    assert list(code[0]) == [1, 1, 0, 0, 1] and list(code[1]) == [1, 0, 0, 1, 0] and not code[2].any()


def test_values_round_trip_ordered_and_inside_the_prior():
    composition, ratio, values = _random_values(500, 1)
    z = E.to_z(values, ratio, composition)
    back = E.from_z(z, ratio, composition)
    on = E.active(composition)
    assert np.allclose(back[on], values[on], rtol=1e-10) and np.all(np.isnan(back[~on])) and np.all(z[~on] == 0)
    radii = values[:, 0:E.GLOBAL.start:E.PER_SLOT]
    k = np.array([len(E.COMPOSITIONS[c]) for c in composition])
    for row, n in zip(radii, k):
        assert np.all(np.diff(row[:n]) >= 0) and row[0] >= 0.5
    lo, hi = E.bounds(ratio)
    inside = (np.log(values) >= lo - 1e-9) & (np.log(values) <= hi + 1e-9)
    assert np.all(inside[:, [0, 6, 12]][on[:, [0, 6, 12]]])
    for i, c in enumerate(composition):
        for slot, (family, _s) in enumerate(E.composition_types(c)):
            if family == "cylinder":
                r = values[i, slot * E.PER_SLOT]
                assert r <= E.CYLINDER_RADIUS + 1e-9 and 2 * r * values[i, slot * E.PER_SLOT + 2] <= E.CYLINDER_HEIGHT + 1e-6


def test_prior_density_is_the_logistic_one():
    composition, ratio, values = _random_values(50, 2)
    z = E.to_z(values, ratio, composition)
    on = E.active(composition)
    expected = np.sum(np.where(on, -np.log1p(np.exp(-z)) - np.log1p(np.exp(z)), 0.0), axis=1)
    assert np.allclose(E.log_prior(z, composition), expected)


def test_features_do_not_depend_on_the_scale():
    rng = np.random.default_rng(2)
    intensity = np.exp(rng.normal(size=(3, E.POINTS)))
    sigma = 0.1 * intensity
    mask = rng.uniform(size=intensity.shape) > 0.1
    one = E.features(intensity, sigma, mask)
    other = E.features(1e4 * intensity, 1e4 * sigma, mask)
    assert one.shape == (3, E.CHANNELS, E.POINTS)
    assert np.allclose(one, other, atol=1e-6)
    assert np.all(one[:, 0][~mask] == 0) and np.all(one[:, 2] == mask)


def test_window_and_binning():
    q = np.linspace(0.02, 2.0, 397)
    q_max, ratio = E.window(q)
    assert q_max == 2.0 and ratio == pytest.approx(100.0)
    with pytest.raises(ValueError):
        E.window(np.linspace(0.5, 2.0, 50))
    intensity = 1.0 / q
    out_i, out_s, mask = E.bin_curve(q, intensity, 0.05 * intensity, q_max, ratio)
    centres = q_max * E.grid(ratio)
    assert mask[centres > 0.2].all() and not mask.all()  # low-q bins narrower than the point spacing stay empty
    assert np.allclose(out_i[mask] * centres[mask], 1.0, rtol=0.03)


def test_error_basis_is_a_partition_of_unity():
    x = np.linspace(0, 1, 101)
    basis = E.error_basis(x)
    assert basis.shape == (E.ERROR_BASIS, 101) and np.allclose(basis.sum(axis=0), 1.0) and np.all(basis >= 0)


def test_resolution_on_a_tabulated_grid_agrees_with_the_direct_convolution():
    population = model.Population("sphere", 8.0, 0.1, distance=30.0, distance_spread=0.1)
    q = np.geomspace(0.01, 1.0, 200)
    u = np.linspace(-1.4, 1.4, 56001)
    on_u = model.unit_terms([population], model.Settings(), np.abs(u) + 1e-9)[0]
    table_q = np.concatenate([np.linspace(0.002, 0.01, 40)[:-1], np.geomspace(0.01, 1.0, 3000),
                              np.linspace(1.0, 1.2, 60)[1:]])
    table = model.unit_terms([population], model.Settings(), table_q)
    for sigma in (0.003, 0.01, 0.03):
        kernel = np.exp(-0.5 * ((q[:, None] - u[None, :]) / sigma) ** 2)
        direct = kernel @ on_u / kernel.sum(axis=1)
        assert np.allclose(resolution.tabulated(q, table_q, table, sigma)[0], direct, rtol=1e-3)


def test_spline_inverts_and_is_monotone():
    rng = np.random.default_rng(3)
    raw = np.repeat(rng.normal(size=(1, 1, 3 * 6 - 1)), 200, axis=0)
    x = np.sort(rng.uniform(-4, 4, 200))[:, None]
    y, log_det = spline(x, raw, 3.0, 6)
    assert np.all(np.diff(y[:, 0]) > 0)
    back, inverse_det = spline(y, raw, 3.0, 6, inverse=True)
    assert np.allclose(back, x, atol=1e-9) and np.allclose(log_det, -inverse_det, atol=1e-9)
    h = 1e-6
    numeric = (spline(x + h, raw, 3.0, 6)[0] - spline(x - h, raw, 3.0, 6)[0]) / (2 * h)
    inside = np.abs(x) < 3.0 - 1e-5
    assert np.allclose(np.log(numeric[inside]), log_det[inside], atol=1e-5)


def _fixture():
    network = Network.load(DATA / "flow_fixture.npz")
    with np.load(DATA / "flow_fixture_reference.npz") as data:
        ref = {k: data[k] for k in data.files}
    return network, ref


def test_numpy_flow_equals_pytorch():
    """The stored network (tools/ml1d/make_flow_fixture.py, random weights) gives PyTorch's numbers in numpy."""
    network, ref = _fixture()
    h = network.summary(ref["features"], ref["conditions"])
    assert np.allclose(h, ref["summary"], rtol=1e-9, atol=1e-12)
    assert np.allclose(network.composition_log_probs(h), ref["composition_log_probs"], rtol=1e-9, atol=1e-9)
    context = network.context(h, ref["composition"])
    assert np.allclose(context, ref["context"], rtol=1e-9, atol=1e-12)
    active = E.active(ref["composition"])
    assert np.allclose(network.log_prob(ref["z"], context, active), ref["log_prob"], rtol=1e-9, atol=1e-9)
    sample, log_det = network.inverse(ref["u"], context, active)
    assert np.allclose(sample, ref["sample"], rtol=1e-9, atol=1e-9)
    assert np.allclose(log_det, ref["log_det"], rtol=1e-9, atol=1e-9)
    u, forward_det = network.forward(sample, context, active)
    assert np.allclose(u, np.where(active, ref["u"], 0.0), atol=1e-8) and np.allclose(forward_det, -log_det, atol=1e-8)


def test_inactive_coordinates_change_nothing():
    network, ref = _fixture()
    context = network.context(network.summary(ref["features"], ref["conditions"]), ref["composition"])
    active = E.active(ref["composition"])
    other = np.where(active, ref["z"], 7.0)
    assert np.allclose(network.log_prob(other, context, active), network.log_prob(ref["z"], context, active))


def test_samples_carry_their_density():
    network, ref = _fixture()
    context = network.context(network.summary(ref["features"][:1], ref["conditions"][:1]), 17)
    z, log_q = network.sample(context, 17, 64, np.random.default_rng(4))
    active = np.repeat(E.active(17), 64, axis=0)
    assert z.shape == (64, E.DIMS) and np.all(z[~active] == 0)
    assert np.allclose(log_q, network.log_prob(z, np.repeat(context, 64, axis=0), active), atol=1e-8)


def test_a_changed_network_file_is_refused(tmp_path):
    for name in ("flow_fixture.npz", "flow_fixture.json"):
        shutil.copy(DATA / name, tmp_path / name)
    meta = json.loads((tmp_path / "flow_fixture.json").read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    (tmp_path / "flow_fixture.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        Network.load(tmp_path / "flow_fixture.npz")


def test_the_network_that_comes_with_gimap_is_the_one_its_record_names(monkeypatch):
    """The shipped weights pass their SHA-256 check, were trained in this encoding, and propose for a curve."""
    from src.gimap.features.fitting.domain.posterior import network

    monkeypatch.delenv(network.VARIABLE, raising=False)
    assert network.path() == network.FOLDER / f"{network.DEFAULT}.npz" and network.available()
    net = Network.load(network.path())  # refuses weights that differ from the record's SHA-256
    meta = net.meta
    assert meta["encoding"] == E.VERSION and meta["names"] == list(E.NAMES)
    assert meta["compositions"] == len(E.COMPOSITIONS) and meta["points"] == E.POINTS
    text = network.path().with_suffix(".json").read_text(encoding="utf-8")
    assert "/data/" not in text and "\\\\" not in text  # no path of the computer it was trained on
    assert (network.FOLDER / "MODEL_CARD.md").is_file()
    q = np.geomspace(0.05, 2.0, 300)
    curve = model.population_terms("sphere", q, radius=4.0, radius_spread=0.1)[0] + 1e-3
    q_max, ratio = E.window(q)
    binned = E.bin_curve(q, curve, 0.05 * curve, q_max, ratio)
    h = net.summary(E.features(*(np.asarray(a)[None] for a in binned)), E.conditions(ratio, 0.0))
    probabilities = np.exp(net.composition_log_probs(h))[0]
    assert np.isfinite(h).all() and abs(probabilities.sum() - 1) < 1e-6


def test_simulator_rows_are_reproducible_and_usable():
    sys.path.insert(0, str(TOOLS))
    try:
        import simulate
    finally:
        sys.path.remove(str(TOOLS))
    prior = simulate.load_prior(TOOLS / "prior_v1.json")
    one, two = simulate.shard(prior, 30, 7), simulate.shard(prior, 30, 7)
    assert all(np.array_equal(one[k], two[k], equal_nan=k == "values") for k in one)
    used = E.active(one["composition"].astype(int))[:, 0:E.GLOBAL.start:E.PER_SLOT]
    assert np.all((one["term"].max(axis=2) > 0) == used) and np.all(np.isfinite(one["term"]))
    features, conditions, composition, z, active = simulate.inputs(one, np.arange(30), prior,
                                                                  np.random.default_rng(0))
    assert features.shape == (30, E.CHANNELS, E.POINTS) and np.all(np.isfinite(features))
    assert conditions.shape == (30, E.CONDITIONS) and np.all(np.isfinite(z)) and np.all(z[~active] == 0)
