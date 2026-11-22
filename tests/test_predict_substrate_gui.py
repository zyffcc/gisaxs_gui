"""Substrate input follows both Predict entry points, their saved settings, and the fitted model report."""

import numpy as np
import pytest

from src.gimap.features.fitting.application.single_fit import FitData, FitModel, evaluate, fit_record, new_component
from src.gimap.features.fitting.application.single_predict import predict_request
from src.gimap.features.fitting.application.substrate_level import batch_geometry, prediction_model
from src.gimap.features.fitting.infrastructure.reproduce_fit import predict_options
from src.gimap.features.fitting.presentation.predict_substrate import PredictSubstrateDialog
from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog


def test_typed_film_in_predict_and_batch_uses_the_same_geometry():
    dialog = PredictSubstrateDialog({}, batch=True)
    try:
        assert dialog.cut() == {}
        dialog.support.setCurrentIndex(dialog.support.findData("custom"))
        dialog.beam_unit.setCurrentIndex(dialog.beam_unit.findData("energy"))
        dialog.beam.setText("12")
        dialog.incidence.setText("0.45")
        dialog.exit.setText("0.18")
        dialog.delta.setText("3.4e-6")
        dialog.beta.setText("2.2e-8")
        dialog.coated.setChecked(True)
        dialog.thickness.setText("90")
        dialog.layer_delta.setText("1.6e-6")
        dialog.layer_beta.setText("5e-10")
        assert dialog.ok.isEnabled()
        cut = dialog.cut()
        assert cut["grazing"]["wavelength"] == pytest.approx(1.239841984 / 12)
        assert cut["substrate"]["layers"] == [{"thickness_nm": 90., "delta": 1.6e-6, "beta": 5e-10}]
        dialog.thickness.clear()
        assert not dialog.ok.isEnabled()
    finally:
        dialog.deleteLater()
    batch = WorkflowV5Dialog(options={"predict_cut": cut, "cut_qz": 0.4}, settings_only=True)
    try:
        assert not batch.cut_qz.isEnabled()
        saved = batch.options()["predict_cut"]
        assert saved == cut
    finally:
        batch.deleteLater()
    model = prediction_model(FitModel((new_component("vertical_cylinder", radius=5., heights=True),)), cut)
    q = np.linspace(0.1, 1.2, 30)
    data = FitData(q, evaluate(model, q), np.ones_like(q))
    options = predict_options(predict_request(data, model, "model").payload)
    settings = batch_geometry(saved)
    assert settings["substrate"] == options["substrate"]
    assert settings["qz"] == options["qz"] and settings["qz_minus"] == options["qz_minus"]
    assert options["upright_heights"]["1"] == [model.components[0].value(k) for k in ("h", "sigma_h")]
    record = fit_record(model, data, None, side="mean", q_range=None, result=None, include_recipe=False)
    assert record["sphere_model"].startswith("four-term DWBA on 90 nm")
    assert record["model"]["substrate"] == cut["substrate"]


def test_substrate_needs_two_angles_and_the_material_table_needs_an_in_range_energy():
    dialog = PredictSubstrateDialog({"substrate": {"material": "Si"},
                                    "grazing": {"alpha_i": 0.35, "wavelength": 0.1}})
    try:
        assert not dialog.ok.isEnabled() and "both cut angles" in dialog.note.text()
        dialog.exit.setText("0.14")
        assert dialog.ok.isEnabled()
        dialog.beam.setText("1.0")
        assert not dialog.ok.isEnabled() and "4.9–30 keV" in dialog.note.text()
        dialog.support.setCurrentIndex(dialog.support.findData(""))
        assert dialog.ok.isEnabled() and "substrate" not in dialog.cut()
    finally:
        dialog.deleteLater()
