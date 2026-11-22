"""Manual fit weights and resolution follow the measured rows through intake and selection."""

import numpy as np
import pytest

from src.gimap.features.fitting.application import LoadCurve, LoadCurveRequest
from src.gimap.features.fitting.application.single_fit import Curve, FitData, prepare_points
from src.gimap.features.fitting.infrastructure.adapters import LocalCurveRepository
from src.gimap.shared.nxcansas import PER_M, SasCurve, write_nxcansas


def load(path):
    outcome = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(path, "angstrom", keep_missing_sigma=True))
    assert outcome.succeeded, outcome.error
    raw = outcome.value
    return Curve.from_arrays(raw.q, raw.intensity, raw.error, name="example", path=str(path),
                             unit=raw.q_source_unit, dq=raw.dq)


@pytest.mark.parametrize("filename,sep,resolution", [
    ("example.dat", " ", "dQ"), ("example.csv", ",", "sigma_q (1/nm)"),
])
def test_named_resolution_is_never_intensity_sigma(tmp_path, filename, sep, resolution):
    path = tmp_path / filename
    path.write_text("\n".join(sep.join(row) for row in [
        ("# q (1/A)", "I", resolution), ("0.2", "12", "0.003"), ("-0.1", "20", "0.002"),
        ("0.1", "10", "0.004"), ("0.3", "30", "0.001")]), encoding="utf-8")
    curve = load(path)
    assert curve.sigma is None
    factor = 10 if filename.endswith(".dat") else 1
    fit = prepare_points(curve, q_range=(0.5, 2), resolution="file").data
    np.testing.assert_allclose(fit.q, [1, 2])
    np.testing.assert_allclose(fit.intensity, [15, 12])
    np.testing.assert_allclose(fit.dq, np.array([np.sqrt((0.002**2 + 0.004**2) / 2), 0.003]) * factor)
    assert prepare_points(curve).data.dq is None


def test_weight_and_resolution_columns_read_separately_in_any_order(tmp_path):
    from src.gimap.features.fitting.application.curve_sources import sources_from_table

    path = tmp_path / "permuted.dat"
    path.write_text("# I dQ (1/A) q (1/nm) dI\n3 0.003 0.9 1\n4 0.004 0.7 2\n", encoding="utf-8")
    curve = load(path)
    fit = prepare_points(curve, resolution="file").data
    np.testing.assert_allclose(fit.q, [0.7, 0.9])
    np.testing.assert_allclose(fit.dq, [0.04, 0.03])
    np.testing.assert_allclose(fit.sigma, [2, 1])
    source, = sources_from_table(LocalCurveRepository().inspect_table(path), "single")
    assert (source.columns.q, source.columns.intensity, source.columns.sigma, source.columns.dq) == (2, 0, 3, 1)


@pytest.mark.parametrize("cell", ["", "n/a"])
def test_missing_sigma_columns_remain_available_to_other_weighting_choices(tmp_path, cell):
    path = tmp_path / "missing.csv"
    path.write_text(f"q,I,sigma,dQ\n.2,12,{cell},.01\n.1,25,5,.02\n-.2,14,2,.03\n-.1,16,4,.04\n", encoding="utf-8")
    curve = load(path)
    # File weighting keeps the single measured half at |q|=2, as before alternate weights existed.
    np.testing.assert_array_equal(prepare_points(curve).data.intensity, [20.5, 14])
    alternate = prepare_points(curve, weighting="poisson").data
    np.testing.assert_allclose(alternate.intensity, [20.5, 13])
    np.testing.assert_allclose(alternate.sigma, np.sqrt([20.5, 13]))
    path.write_text("q,I,sigma,model,residual\n.1,25,nan,25,0\n.2,16,nan,16,0\n", encoding="utf-8")
    data = prepare_points(load(path)).data
    np.testing.assert_array_equal(data.intensity, [25, 16])
    assert data.sigma is None


def test_qdev_units_mask_resolution_attribute_and_zero_width(tmp_path):
    import h5py

    path = tmp_path / "synthetic.h5"
    q = np.array([2e9, 1e9, 3e9, 4e9])
    write_nxcansas(path, [SasCurve("one", q, np.arange(4.) + 5, q_unit=PER_M, dq=np.array([2e7, 0., 3e7, 4e7]))])
    with h5py.File(path, "r+") as root:
        group = root["sasentry01/sasdata01"]
        group.move("Qdev", "beam_width")
        group["Q"].attrs["resolutions"] = "beam_width"
        group["mask"][2] = 1
    fit = prepare_points(load(path), resolution="file").data
    np.testing.assert_allclose(fit.q, [1, 2, 4])
    np.testing.assert_allclose(fit.dq, [0, 0.02, 0.04])
    assert fit.sigma is None


@pytest.mark.parametrize("mode,want", [("file", [9, 8, 7]), ("relative", [16, 4]),
                                      ("poisson", [4, 2]), ("none", None)])
def test_weighting_uses_the_fitted_observation(mode, want):
    curve = Curve.from_arrays([3, 1, 2], [4, -2, 16], [7, 9, 8], name="example", unit="nm", dq=[0.3, 0.1, 0.2])
    data = prepare_points(curve, weighting=mode, resolution="file").data
    np.testing.assert_array_equal(data.q, [1, 2, 3] if mode == "file" else [2, 3])
    np.testing.assert_allclose(data.dq, data.q / 10)
    assert data.weight_mode == mode
    if want is None:
        assert data.sigma is None
    else:
        np.testing.assert_array_equal(data.sigma, want)


def test_bad_resolution_matters_only_on_the_fitted_rows(tmp_path):
    path = tmp_path / "bad_resolution.csv"
    path.write_text("q,I,dI,dQ\n.1,1,3,0\n.2,2,1,bad\n.3,3,2,-1\n.4,4,1,.04\n", encoding="utf-8")
    curve = load(path)
    assert prepare_points(curve).points == 4
    with pytest.raises(ValueError, match="finite, non-negative dQ"):
        prepare_points(curve, resolution="file")
    points = prepare_points(curve, excluded=(2, 3), resolution="file")
    np.testing.assert_allclose(points.data.dq, [0, 0.4])
    assert FitData.prepare([1, 2], [3, 4], None).sigma is None
    with pytest.raises(ValueError, match="needs a dQ"):
        prepare_points(Curve.from_arrays([1, 2], [3, 4], name="example"), resolution="file")


def test_chain_compares_predictions_with_the_manual_weights_resolution_and_law(monkeypatch):
    from dataclasses import replace
    from src.gimap.features.fitting.application import series_chain
    from src.gimap.features.fitting.application.series_fit import FrameFit, SeriesSettings
    from src.gimap.features.fitting.application.single_fit import FitModel, FitResult, new_component, number_size

    model = FitModel((new_component("sphere", radius=10, law="lognormal_median"),),
                     weighting="poisson", resolution_mode="file")
    proposal = FitModel((new_component("sphere", radius=6, law="number_gauss"),))
    calls = []

    def fit_frame(index, path, curve, start, settings, *args):
        calls.append((settings.method, start))
        result_model = proposal if settings.method == "predict" else start
        chi = (5, 0.1, 7)[len(calls) - 1]
        result = FitResult(result_model, "local", "sigma", 20, 2, chi, 0.1, 1, 0.0, True, False, "")
        return FrameFit(index, path, result, points=20)

    monkeypatch.setattr(series_chain, "fit_frame", fit_frame)
    frame = series_chain.chained_fit(0, "", None, model, model, SeriesSettings(), None, new_stage=True,
                                     predict=lambda *_: proposal)
    assert [method for method, start in calls] == ["local", "predict", "local"]
    seed = calls[-1][1]
    assert seed.resolution_mode == "file" and seed.weighting == "poisson"
    assert seed.components[0].law == "lognormal_median"
    np.testing.assert_allclose(number_size(seed.components[0]), number_size(proposal.components[0]))
    assert frame.start_from == "single" and frame.result.chi2_reduced == 5
    # The same objectives still require only the original fit and the Predict restart.
    calls.clear()
    legacy = replace(proposal, weighting="file")
    series_chain.chained_fit(0, "", None, legacy, legacy, SeriesSettings(), None, new_stage=True,
                             predict=lambda *_: proposal)
    assert len(calls) == 2
