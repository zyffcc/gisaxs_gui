from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import tuning_checkpoint_model_v5 as loader
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.tuning_checkpoint_runtime_v5 import V5RetainedFullCheckpoint


def _result_fixture(tmp_path, *, full_epochs=2, warmup_epochs=1):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_v5 import (
        V5_GROUPED_TRAINER_SCHEMA, V5_GROUPED_TRAINER_VERSION,
        _checkpoint_selection_payload, _full_checkpoint_relative_path,
    )
    rows = []
    for number in range(1, full_epochs + 1):
        relative = _full_checkpoint_relative_path(epoch=warmup_epochs + number, phase_epoch=number)
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"unit-checkpoint-{number}".encode())
        path.chmod(0o400)
        rows.append({"epoch": warmup_epochs + number, "phase_epoch": number,
                     "relative_path": relative.as_posix(), "file_sha256": sha256(path.read_bytes()).hexdigest(),
                     "weights_sha256": "a" * 64})
    core = {"schema": V5_GROUPED_TRAINER_SCHEMA, "version": V5_GROUPED_TRAINER_VERSION,
            "status": "complete", "full_checkpoints": rows,
            "checkpoint_selection": _checkpoint_selection_payload(),
            "paper_model_status": {"full_training_completed": True,
                "paper_checkpoint_candidates_eligible_for_external_selection": True,
                "full_checkpoint_count": full_epochs, "expected_full_checkpoint_count": full_epochs,
                "paper_model_eligible": False}}
    path = tmp_path / "result_manifest.json"
    def publish():
        value = dict(core)
        value["result_sha256"] = sha256(loader.canonical_json(core).encode()).hexdigest()
        if path.exists():
            path.chmod(0o600)
        path.write_text(json.dumps(value))
        path.chmod(0o400)
        return dict(result_path=path, expected_result_file_sha256=sha256(path.read_bytes()).hexdigest(),
                    expected_result_sha256=value["result_sha256"], expected_full_epochs=full_epochs,
                    expected_warmup_epochs=warmup_epochs)
    return core, path, publish


def test_result_reader_retains_all_epochs_in_order_without_selecting(tmp_path):
    core, _, publish = _result_fixture(tmp_path)
    args = publish()
    values = loader.read_v5_retained_checkpoints_from_training_result(**args)
    assert tuple(value.full_epoch for value in values) == (1, 2)
    assert tuple(value.checkpoint_artifact_sha256 for value in values) == tuple(
        row["file_sha256"] for row in core["full_checkpoints"])
    assert all(value.training_result_sha256 == args["expected_result_sha256"] for value in values)


@pytest.mark.parametrize("drift", ["missing", "order", "epoch", "bool_epoch", "path", "absolute_path",
                                  "warmup_only", "ineligible", "selected", "count", "bool_count",
                                  "weights", "schema", "failed", "extra_record_field", "duplicate"])
def test_result_reader_rejects_rehashed_incomplete_or_invalid_inventory(tmp_path, drift):
    core, _, publish = _result_fixture(tmp_path)
    row = core["full_checkpoints"][0]
    if drift == "missing":
        core["full_checkpoints"].pop()
    elif drift == "order":
        core["full_checkpoints"].reverse()
    elif drift in ("epoch", "bool_epoch"):
        row["phase_epoch"] = 2 if drift == "epoch" else True
    elif drift in ("path", "absolute_path"):
        row["relative_path"] = "../best.keras" if drift == "path" else str(tmp_path / "best.keras")
    elif drift == "warmup_only":
        core["paper_model_status"]["full_training_completed"] = False
    elif drift == "ineligible":
        core["paper_model_status"]["paper_checkpoint_candidates_eligible_for_external_selection"] = False
    elif drift == "selected":
        core["checkpoint_selection"]["paper_selected_checkpoint"] = "best.keras"
    elif drift in ("count", "bool_count"):
        core["paper_model_status"]["full_checkpoint_count"] = 1 if drift == "count" else True
    elif drift == "weights":
        row["weights_sha256"] = "not-a-hash"
    elif drift == "schema":
        core["schema"] += "-unknown"
    elif drift == "failed":
        core["status"] = "failed"
    elif drift == "extra_record_field":
        row["injected"] = True
    else:
        second = core["full_checkpoints"][1]
        path = tmp_path / second["relative_path"]
        path.chmod(0o600)
        path.write_bytes((tmp_path / row["relative_path"]).read_bytes())
        path.chmod(0o400)
        second["file_sha256"] = row["file_sha256"]
    with pytest.raises(ValueError):
        loader.read_v5_retained_checkpoints_from_training_result(**publish())


@pytest.mark.parametrize("drift", ["file_hash", "result_hash", "manifest_writable", "checkpoint_writable",
                                  "checkpoint_replaced", "checkpoint_symlink", "hardlink", "schedule"])
def test_result_reader_rejects_actual_file_or_external_binding_drift(tmp_path, drift):
    core, path, publish = _result_fixture(tmp_path)
    args = publish()
    checkpoint = tmp_path / core["full_checkpoints"][0]["relative_path"]
    if drift == "file_hash":
        args["expected_result_file_sha256"] = "f" * 64
    elif drift == "result_hash":
        args["expected_result_sha256"] = "f" * 64
    elif drift == "manifest_writable":
        path.chmod(0o600)
    elif drift == "checkpoint_writable":
        checkpoint.chmod(0o600)
    elif drift == "checkpoint_replaced":
        checkpoint.chmod(0o600)
        checkpoint.write_bytes(b"different-model")
        checkpoint.chmod(0o400)
    elif drift == "checkpoint_symlink":
        other = checkpoint.with_suffix(".original")
        checkpoint.rename(other)
        checkpoint.symlink_to(other)
    elif drift == "hardlink":
        import os
        os.link(checkpoint, checkpoint.with_suffix(".linked"))
    else:
        args["expected_warmup_epochs"] += 1
    with pytest.raises(ValueError):
        loader.read_v5_retained_checkpoints_from_training_result(**args)


def test_result_reader_detects_manifest_replacement_during_checkpoint_replay(tmp_path, monkeypatch):
    _, path, publish = _result_fixture(tmp_path)
    args = publish()
    original = loader.read_only_identity
    def replace(selected, name):
        if name == "retained checkpoint" and selected.name.startswith("full_epoch_0002"):
            other = path.with_suffix(".replacement")
            other.write_bytes(path.read_bytes())
            other.chmod(0o400)
            other.replace(path)
        return original(selected, name)
    monkeypatch.setattr(loader, "read_only_identity", replace)
    with pytest.raises(RuntimeError, match="training result changed"):
        loader.read_v5_retained_checkpoints_from_training_result(**args)


def _case(tmp_path):
    path = tmp_path.resolve() / "model.keras"
    path.write_bytes(b"unit-test-checkpoint")
    path.chmod(0o400)
    model = SimpleNamespace(weights=[SimpleNamespace(
        name="test-weight", numpy=lambda: np.array([1.0], dtype=np.float32))])
    checkpoint = V5RetainedFullCheckpoint(
        full_epoch=1, checkpoint_path=path,
        checkpoint_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
        checkpoint_weights_sha256=loader.model_weights_sha256(model),
        training_result_sha256="b" * 64,
    )
    return checkpoint, model


def test_load_checks_actual_weights_and_stable_sealed_file(tmp_path, monkeypatch):
    checkpoint, model = _case(tmp_path)
    monkeypatch.setattr(loader, "_load_graph", lambda path: model)
    assert loader.load_v5_verified_tuning_checkpoint(checkpoint) is model
    model.weights[0].numpy = lambda: np.array([2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="weights differ"):
        loader.load_v5_verified_tuning_checkpoint(checkpoint)


def test_writable_checkpoint_rejected_before_deserialization(tmp_path, monkeypatch):
    checkpoint, _ = _case(tmp_path)
    checkpoint.checkpoint_path.chmod(0o600)
    monkeypatch.setattr(loader, "_load_graph", lambda path: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="read-only"):
        loader.load_v5_verified_tuning_checkpoint(checkpoint)


def test_changed_checkpoint_rejected_after_loading(tmp_path, monkeypatch):
    checkpoint, model = _case(tmp_path)

    def mutate(path):
        path.chmod(0o600)
        path.write_bytes(b"different-model")
        path.chmod(0o400)
        return model

    monkeypatch.setattr(loader, "_load_graph", mutate)
    with pytest.raises(RuntimeError, match="changed during"):
        loader.load_v5_verified_tuning_checkpoint(checkpoint)


def test_real_v5_graph_round_trip_preserves_weights_and_predictions(tmp_path):
    from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_model_v5 import (
        _small_model, _inputs, _sphere_bounds, _resolution_bounds,
    )
    model = _small_model()
    inputs = _inputs((_sphere_bounds(),), 0, resolution_bounds=_resolution_bounds())
    expected = model(inputs, training=False)
    core, _, publish = _result_fixture(tmp_path, full_epochs=1)
    row = core["full_checkpoints"][0]
    path = tmp_path / row["relative_path"]
    # Replace only this test's placeholder with an actual serialized V5 graph.
    path.chmod(0o600)
    path.unlink()
    model.save(path)
    path.chmod(0o400)
    row["file_sha256"] = sha256(path.read_bytes()).hexdigest()
    row["weights_sha256"] = loader.model_weights_sha256(model)
    (checkpoint,) = loader.read_v5_retained_checkpoints_from_training_result(**publish())
    loaded = loader.load_v5_verified_tuning_checkpoint(checkpoint)
    actual = loaded(inputs, training=False)
    for name in expected:
        np.testing.assert_array_equal(expected[name].numpy(), actual[name].numpy())
