"""Scientific regressions for native CBF observations and experimental fits."""

import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import validate_options
from src.gimap.features.fitting.domain.cbf_observations import column_observations
from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import (
    fit_candidates,
    forward,
)


def test_native_columns_use_preprocessing_mask_and_partial_counts():
    image = np.full((3, 30), 9.0)
    image[:, 13:18] = np.nan
    image[0, 1] = np.nan
    qmesh = np.broadcast_to(np.arange(30), image.shape)
    original = image.copy()
    q, y, sigma, meta = column_observations(image, qmesh, (0, 2, 0, 29))
    assert not set(range(13, 18)) & set(q)
    assert len(q) == 25 and meta["invalid_columns"] == 5
    assert y[q == 1] == 9
    np.testing.assert_allclose(sigma[q == 1], np.sqrt(18) / 2)
    np.testing.assert_array_equal(image, original)


def test_zero_count_is_observation_and_selection_edge_is_not_hardware_gap():
    image = np.zeros((2, 20))
    qmesh = np.broadcast_to(np.arange(20), image.shape)
    selection = qmesh >= 4
    q, y, sigma, _ = column_observations(
        image, qmesh, (0, 1, 0, 19), selection_mask=selection
    )
    assert len(q) == 16 and q[0] == 4
    assert not y.any() and np.all(sigma == 0.5)


def synthetic_item(scale=1):
    q = np.linspace(0.02, 3, 100)
    cs = [dict(type_id=1, amplitude=420, params=dict(R=1.8, sigma_R=0.18, D=3.9, sigma_D=0.13))]
    globals_ = dict(background=0.8, resolution_amplitude=50000, sigma_Res=0.008, nu_Res=2.1)
    y = forward(q, cs, globals_) * scale
    return dict(
        q=q, observed=y, sigma=np.hypot(0.1 * y, scale), sign=1, side="positive", normalizer=y.max()
    )


def test_known_composition_resolution_and_forward_reconstruction():
    options = validate_options(
        dict(method="experimental", components=[1], sigma_res=0.008, nu_res=2.1)
    )
    item = synthetic_item()
    rows = fit_candidates(item, options, lambda *a: None, lambda: False)
    row = rows[0]
    assert row["best_log_rmse"] < 0.005
    assert row["global_params"]["sigma_Res"] == 0.008
    assert row["global_params"]["nu_Res"] == 2.1
    assert [c["type_id"] for c in row["components"]] == [1]
    np.testing.assert_allclose(
        forward(item["q"], row["components"], row["global_params"]), row["native_fit"], rtol=1e-12
    )
    scaled = fit_candidates(synthetic_item(100), options, lambda *a: None, lambda: False)[0]
    np.testing.assert_allclose(np.array(scaled["native_fit"]) / 100, row["native_fit"], rtol=1e-4)


def test_cancellation_and_prior_validation():
    with pytest.raises(ValueError):
        validate_options(dict(method="model", nu_res=2))
    options = validate_options(dict(method="experimental", nu_res=2))
    with pytest.raises(RuntimeError, match="cancelled"):
        fit_candidates(synthetic_item(), options, lambda *a: None, lambda: True)


def test_requested_repeated_components_keep_distinct_amplitudes():
    item = synthetic_item()
    options = validate_options(
        dict(method="experimental", components=[1, 1], sigma_res=0.008, nu_res=2.1)
    )
    row = fit_candidates(item, options, lambda *a: None, lambda: False)[0]
    assert [c["type_id"] for c in row["components"]] == [1, 1]
    assert all("amplitude" in c for c in row["components"])
    np.testing.assert_allclose(
        forward(item["q"], row["components"], row["global_params"]), row["native_fit"], rtol=1e-12
    )
    assert row["best_log_rmse"] < 0.005


def test_text_batch_persists_results_and_continues_after_invalid_file(tmp_path):
    import json
    from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import run_workflow_job

    item = synthetic_item()
    valid = tmp_path / "valid.csv"
    np.savetxt(valid, np.column_stack([item["q"], item["observed"], item["sigma"]]), delimiter=",")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("bad data")
    result = run_workflow_job(
        dict(
            output_dir=str(tmp_path / "output"),
            files=[str(invalid), str(valid)],
            options=dict(method="experimental", components=[1], sigma_res=0.008, nu_res=2.1),
        ),
        lambda *a: None,
        lambda: False,
    )
    assert [r["status"] for r in result["records"]] == ["failed", "complete"]
    rows = json.loads((tmp_path / "output/top20_candidates.json").read_text())
    assert rows[0]["best_source"] == "experimental_physical"
    assert rows[0]["best_log_rmse"] < 0.005
    assert rows[0]["curve_quality_passed"] is None
    assert rows[0]["curve_logrmse_target"] is None
    assert "measurement noise" in rows[0]["curve_quality_note"]
