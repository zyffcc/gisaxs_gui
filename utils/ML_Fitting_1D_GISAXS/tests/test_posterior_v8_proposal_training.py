from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import proposal_training as training_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.dataset import (
    DATASET_GENERATOR_VERSION,
    DATASET_SCHEMA_VERSION,
    PILOT_PHASE,
    SPLIT_CODE,
    NumpyShard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training import (
    HISTORY_FILE,
    HISTORY_SCHEMA,
    MANIFEST_FILE,
    MODEL_FILE,
    RUN_MANIFEST_SCHEMA,
    ProposalTrainingConfig,
    inspect_phase2_shards,
    train_proposal,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal import build_parser
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    active_dimension_mask_for,
)


def _fake_shard(tmp_path: Path, *, nan_input: bool = False):
    path = tmp_path / "phase2-00000.npz"
    payload = b"validated-fake-phase2-shard"
    path.write_bytes(payload)
    path.with_suffix(".json").write_text('{"sidecar":"test"}\n', encoding="utf-8")
    sample_count, max_points = 10, 16
    rng = np.random.default_rng(17)
    x = rng.normal(size=(sample_count, max_points, 3)).astype(np.float32)
    if nan_input:
        x[0, 0, 0] = np.nan
    active = np.broadcast_to(
        np.asarray(active_dimension_mask_for(0, 0), dtype=np.bool_),
        (sample_count, 26),
    ).copy()
    target = np.full((sample_count, 26), 0.5, dtype=np.float32)
    target[:, 0] = np.linspace(0.25, 0.75, sample_count)
    target[:, 1] = np.linspace(0.7, 0.3, sample_count)
    low = np.full_like(target, 0.5)
    high = np.full_like(target, 0.5)
    low[active] = 0.0
    high[active] = 1.0
    arrays = {
        "x": x,
        "point_mask": np.ones((sample_count, max_points), dtype=np.bool_),
        "global_features": rng.normal(size=(sample_count, 5)).astype(np.float32),
        "topology_id": np.zeros(sample_count, dtype=np.int32),
        "branch_pattern_id": np.zeros(sample_count, dtype=np.int32),
        "target_unit": target,
        "active_dimension_mask": active,
        "branch_low": low,
        "branch_high": high,
        "assigned_split": np.asarray(
            [SPLIT_CODE["train"]] * 4
            + [SPLIT_CODE["validation"]] * 3
            + [SPLIT_CODE["calibration"]]
            + [SPLIT_CODE["test"]] * 2,
            dtype=np.uint8,
        ),
        "recipe_index": np.arange(sample_count, dtype=np.int64),
        "view_index": np.zeros(sample_count, dtype=np.int16),
    }
    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "phase": PILOT_PHASE,
        "versions": {"test": "v1"},
        "config": {"test": True},
        "preprocessing_contract": {"max_points": max_points},
        "pilot_limitations": ["test-fixture"],
        "source_sha256_aggregate": "a" * 64,
        "npz_sha256": sha256(payload).hexdigest(),
    }
    shard = NumpyShard(path, MappingProxyType(metadata), MappingProxyType(arrays))
    return path, shard


def _small_config(**changes):
    values = {
        "epochs": 1,
        "global_batch_size": 2,
        "learning_rate": 2.0e-4,
        "seed": 73,
        "max_points": 16,
        "width": 4,
        "encoder_blocks": 1,
        "mixture_components": 2,
        "shuffle_buffer": 8,
        "gradient_clip_norm": 5.0,
        "checkpoint_keep": 2,
        "steps_per_epoch": 1,
        "validation_steps": None,
    }
    values.update(changes)
    return ProposalTrainingConfig(**values)


def test_shard_audit_records_content_checksums_and_row_level_splits(tmp_path):
    path, shard = _fake_shard(tmp_path)
    audit = inspect_phase2_shards([path], shard_loader=lambda _: shard)

    assert audit.train_count == 4
    assert audit.validation_count == 3
    assert audit.calibration_count == 1
    assert audit.test_count == 2
    assert audit.max_points == 16
    assert audit.payload["fingerprint_sha256"]
    assert audit.payload["shards"][0]["npz_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert (
        audit.payload["shards"][0]["metadata_sha256"]
        == sha256(path.with_suffix(".json").read_bytes()).hexdigest()
    )


def test_cpu_training_writes_best_model_versioned_manifest_history_and_resumes(tmp_path):
    shard_path, shard = _fake_shard(tmp_path)
    loader = lambda _: shard
    output = tmp_path / "run"
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])

    result = train_proposal(
        [shard_path],
        output,
        _small_config(),
        shard_loader=loader,
        strategy=strategy,
    )

    assert result.completed_epochs == 1
    assert result.best_epoch == 1
    assert np.isfinite(result.best_validation_loss)
    assert result.model_path == output / MODEL_FILE
    assert result.model_path.is_file()
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    history = json.loads((output / HISTORY_FILE).read_text())
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA
    assert manifest["dataset_audit"]["split_counts"]["train"] == 4
    assert manifest["dataset_audit"]["split_counts"]["validation"] == 3
    assert manifest["dataset_audit"]["split_counts"]["calibration"] == 1
    assert manifest["dataset_audit"]["split_counts"]["test"] == 2
    assert manifest["runtime"]["strategy"] == "MirroredStrategy"
    assert history["schema_version"] == HISTORY_SCHEMA
    assert history["status"] == "complete"
    assert history["completed_epochs"] == 1
    assert history["epochs"][0]["validation_steps"] == 2
    assert history["epochs"][0]["validation_examples"] == 3
    assert history["epochs"][0]["validation_examples_not_used"] == 0
    assert all(
        np.isfinite(value)
        for split in ("train", "validation")
        for value in history["epochs"][0][split].values()
    )
    loaded = tf.keras.models.load_model(result.model_path, safe_mode=True)
    assert loaded.name == "posterior_v8_proposal"
    cold = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training "
                "import load_trained_proposal_model; "
                "print(load_trained_proposal_model(sys.argv[1]).name)"
            ),
            str(result.model_path),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert cold.returncode == 0, cold.stderr
    assert cold.stdout.strip() == "posterior_v8_proposal"

    resumed = train_proposal(
        [shard_path],
        output,
        _small_config(),
        shard_loader=loader,
        strategy=strategy,
        resume=True,
    )
    assert resumed.completed_epochs == 1
    assert resumed.best_validation_loss == pytest.approx(result.best_validation_loss)


def test_mixed_precision_uses_loss_scaling_and_completes_a_real_cpu_step(tmp_path):
    shard_path, shard = _fake_shard(tmp_path)

    result = train_proposal(
        [shard_path],
        tmp_path / "mixed-run",
        _small_config(mixed_precision=True),
        shard_loader=lambda _: shard,
        strategy=tf.distribute.MirroredStrategy(devices=["/cpu:0"]),
    )

    history = json.loads(result.history_path.read_text())
    assert history["status"] == "complete"
    assert history["epochs"][0]["train"]["optimizer_update_applied"] == 1.0
    assert np.isfinite(history["epochs"][0]["train"]["loss"])


def test_failed_later_epoch_restores_model_optimizer_and_ignores_orphan_checkpoint(
    tmp_path, monkeypatch
):
    shard_path, shard = _fake_shard(tmp_path)
    output = tmp_path / "resume-run"
    config = _small_config(epochs=2)
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    real_run_epoch = training_module._run_epoch
    calls = 0

    def fail_on_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated worker interruption")
        return real_run_epoch(*args, **kwargs)

    monkeypatch.setattr(training_module, "_run_epoch", fail_on_second_epoch)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train_proposal(
            [shard_path],
            output,
            config,
            shard_loader=lambda _: shard,
            strategy=strategy,
        )
    interrupted = json.loads((output / HISTORY_FILE).read_text())
    assert interrupted["status"] == "failed"
    assert interrupted["completed_epochs"] == 1
    orphan = output / "checkpoints" / "epoch-000002"
    orphan.mkdir()
    (orphan / "partial").write_bytes(b"incomplete")

    monkeypatch.setattr(training_module, "_run_epoch", real_run_epoch)
    resumed = train_proposal(
        [shard_path],
        output,
        config,
        shard_loader=lambda _: shard,
        strategy=strategy,
        resume=True,
    )

    completed = json.loads(resumed.history_path.read_text())
    assert completed["status"] == "complete"
    assert completed["completed_epochs"] == 2
    assert completed["epochs"][0]["optimizer_iterations"] == 1
    assert completed["epochs"][1]["optimizer_iterations"] == 2
    assert (output / "checkpoints" / "epoch-000002" / "ckpt.index").is_file()


def test_existing_output_and_resume_config_mismatch_fail_closed(tmp_path):
    shard_path, shard = _fake_shard(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep me")

    with pytest.raises(FileExistsError, match="not empty"):
        train_proposal([shard_path], output, _small_config(), shard_loader=lambda _: shard)
    with pytest.raises(ValueError, match="not owned"):
        train_proposal(
            [shard_path],
            output,
            _small_config(),
            shard_loader=lambda _: shard,
            overwrite=True,
        )
    assert (output / "unrelated.txt").read_text() == "keep me"


def test_nonfinite_training_tensor_fails_and_records_failure(tmp_path):
    shard_path, shard = _fake_shard(tmp_path, nan_input=True)
    output = tmp_path / "run"

    with pytest.raises((tf.errors.InvalidArgumentError, FloatingPointError), match="NaN|Inf"):
        train_proposal([shard_path], output, _small_config(), shard_loader=lambda _: shard)

    history = json.loads((output / HISTORY_FILE).read_text())
    assert history["status"] == "failed"
    assert history["completed_epochs"] == 0
    assert "NaN" in history["failure"]["message"] or "Inf" in history["failure"]["message"]


def test_config_and_cli_reject_unsafe_or_ambiguous_modes():
    with pytest.raises(ValueError, match="at least two"):
        ProposalTrainingConfig(checkpoint_keep=1)
    with pytest.raises(ValueError, match="finite and positive"):
        ProposalTrainingConfig(learning_rate=np.nan)
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--shards",
                "one.npz",
                "--output-dir",
                "run",
                "--resume",
                "--overwrite",
            ]
        )
