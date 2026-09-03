from __future__ import annotations

import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observed_curve_io import (
    prepare_observed_curve,
    prepare_observed_text,
    read_numeric_text,
    write_reference_input,
)


def _curve(count=40):
    q = np.geomspace(1.0e-3, 1.0, count)
    intensity = 10.0 + np.exp(-q * 3.0) * 100.0
    return q, intensity


def test_missing_uncertainty_is_encoder_only_and_not_acceptance_evidence():
    q, intensity = _curve()
    result = prepare_observed_curve(
        q,
        intensity,
        curve_id="real-1",
        q_unit="nm^-1",
        missing_relative_sigma=0.02,
    )

    assert result.observed.sigma_log is None
    assert result.provenance["exact_acceptance_has_measured_sigma_log"] is False
    assert result.provenance["encoder_uncertainty"]["source"] == "imputed_relative"
    valid = result.preprocessed.point_mask
    np.testing.assert_allclose(
        result.preprocessed.sigma[valid] / result.preprocessed.intensity[valid], 0.02
    )


def test_measured_absolute_and_log_sigma_have_same_prepared_contract():
    q, intensity = _curve()
    sigma_log = np.linspace(0.01, 0.03, q.size)
    absolute = intensity * sigma_log

    from_absolute = prepare_observed_curve(
        q,
        intensity,
        absolute,
        curve_id="absolute",
        q_unit="nm^-1",
        sigma_kind="absolute",
    )
    from_log = prepare_observed_curve(
        q / 10.0,
        intensity,
        sigma_log,
        curve_id="log",
        q_unit="angstrom^-1",
        sigma_kind="log",
    )

    np.testing.assert_allclose(from_absolute.observed.q, from_log.observed.q)
    np.testing.assert_allclose(
        from_absolute.observed.sigma_log, from_log.observed.sigma_log
    )
    assert from_log.provenance["q_to_nm_inverse_scale"] == 10.0
    assert from_log.provenance["exact_acceptance_has_measured_sigma_log"] is True


def test_text_loader_allows_one_header_and_records_source(tmp_path):
    path = tmp_path / "Cut_Data.txt"
    q, intensity = _curve()
    lines = ["q, intensity"] + [f"{x:.12g}, {y:.12g}" for x, y in zip(q, intensity)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = read_numeric_text(path)
    result = prepare_observed_text(
        path,
        curve_id="cut-data",
        q_unit="nm^-1",
    )

    assert table.shape == (q.size, 2)
    assert result.observed.q.size == q.size
    assert result.provenance["text_source"]["path"] == str(path.resolve())
    assert len(result.provenance["text_source"]["sha256"]) == 64


def test_text_loader_rejects_malformed_rows_and_duplicate_q(tmp_path):
    malformed = tmp_path / "bad.txt"
    malformed.write_text("q y\n0.1 2\nnot data\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_numeric_text(malformed)

    q, intensity = _curve()
    q[5] = q[4]
    with pytest.raises(ValueError, match="duplicates"):
        prepare_observed_curve(
            q,
            intensity,
            curve_id="duplicate",
            q_unit="nm^-1",
        )


def test_reference_input_is_exclusive_and_omits_imputed_sigma(tmp_path):
    q, intensity = _curve()
    prepared = prepare_observed_curve(
        q,
        intensity,
        curve_id="real-no-sigma",
        q_unit="nm^-1",
    )
    target = tmp_path / "curve.npz"

    npz_path, manifest_path = write_reference_input(prepared, target)

    with np.load(npz_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"q", "intensity"}
        np.testing.assert_allclose(archive["q"], prepared.observed.q)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "gisaxs.posterior_v8.observed_curve_input/v1"
    assert manifest["provenance"]["exact_acceptance_has_measured_sigma_log"] is False
    with pytest.raises(FileExistsError):
        write_reference_input(prepared, target)


@pytest.mark.parametrize(
    ("sigma,kind,message"),
    [
        (None, "absolute", "absent exactly"),
        (np.ones(40), "missing", "absent exactly"),
        (None, "unknown", "sigma_kind"),
    ],
)
def test_uncertainty_contract_rejects_ambiguous_combinations(sigma, kind, message):
    q, intensity = _curve()
    with pytest.raises(ValueError, match=message):
        prepare_observed_curve(
            q,
            intensity,
            sigma,
            curve_id="bad",
            q_unit="nm^-1",
            sigma_kind=kind,
        )
