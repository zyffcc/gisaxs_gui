from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.dataset import (
    DATASET_GENERATOR_VERSION,
    DATASET_SCHEMA_VERSION,
    PILOT_PHASE,
    RANGE_GENERATOR_VERSION,
    SPLIT_CODE,
    NumpyShard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.local_target import (
    LOCAL_TARGET_COORDINATE_SEMANTICS,
    TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.local_training_audit import (
    LOCAL_RUN_MANIFEST_SCHEMA,
    load_local_trained_proposal_model,
    validate_local_training_run,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.local_training_data import (
    run_hybrid_validation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v2 import LOCAL_PROPOSAL_MODEL_NAME
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training import (
    load_trained_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training_v2 import (
    LocalProposalTrainingConfig,
    train_local_proposal,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal_v2 import build_parser
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    active_dimension_mask_for,
)


def _fake_shard(tmp_path: Path):
    path = tmp_path / "phase2-local-test-00000.npz"
    payload = b"validated-fake-local-target-phase2-shard"
    path.write_bytes(payload)
    path.with_suffix(".json").write_text('{"sidecar":"test"}\n', encoding="utf-8")
    sample_count, max_points = 10, 16
    rng = np.random.default_rng(31)
    active = np.broadcast_to(
        np.asarray(active_dimension_mask_for(0, 0), dtype=np.bool_),
        (sample_count, 26),
    ).copy()
    target = np.full((sample_count, 26), 0.5, dtype=np.float32)
    target[:, 0] = np.linspace(0.25, 0.75, sample_count)
    target[:, 1] = np.linspace(0.7, 0.3, sample_count)
    low = np.full_like(target, 0.5)
    high = np.full_like(target, 0.5)
    low[active], high[active] = 0.0, 1.0
    arrays = {
        "x": rng.normal(size=(sample_count, max_points, 3)).astype(np.float32),
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
        "versions": {"range_generator": RANGE_GENERATOR_VERSION},
        "config": {"test": True},
        "preprocessing_contract": {"max_points": max_points},
        "pilot_limitations": ["test-fixture"],
        "source_sha256_aggregate": "a" * 64,
        "npz_sha256": sha256(payload).hexdigest(),
    }
    return path, NumpyShard(
        path,
        MappingProxyType(metadata),
        MappingProxyType(arrays),
    )


def _config(**changes):
    values = {
        "epochs": 1,
        "global_batch_size": 2,
        "learning_rate": 2.0e-4,
        "seed": 83,
        "max_points": 16,
        "width": 4,
        "encoder_blocks": 1,
        "mixture_components": 2,
        "shuffle_buffer": 8,
        "gradient_clip_norm": 5.0,
        "checkpoint_keep": 2,
        "steps_per_epoch": 1,
        "allow_truth_centered_range_pilot": True,
    }
    values.update(changes)
    return LocalProposalTrainingConfig(**values)


def test_truth_centered_v3_requires_explicit_engineering_scope(tmp_path):
    path, shard = _fake_shard(tmp_path)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="truth-centered"):
        train_local_proposal(
            [path],
            output,
            _config(allow_truth_centered_range_pilot=False),
            shard_loader=lambda _: shard,
            strategy=tf.distribute.MirroredStrategy(devices=["/cpu:0"]),
        )
    assert not output.exists()


def test_v2_manifest_loader_and_hybrid_validation_tail_are_exact(tmp_path):
    path, shard = _fake_shard(tmp_path)
    output = tmp_path / "local-run"
    result = train_local_proposal(
        [path],
        output,
        _config(),
        shard_loader=lambda _: shard,
        strategy=tf.distribute.MirroredStrategy(devices=["/cpu:0"]),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    epoch = history["epochs"][0]
    assert manifest["schema_version"] == LOCAL_RUN_MANIFEST_SCHEMA
    assert manifest["model_name"] == LOCAL_PROPOSAL_MODEL_NAME
    assert manifest["coordinate_contract"]["model_output"] == (
        LOCAL_TARGET_COORDINATE_SEMANTICS
    )
    assert manifest["range_construction_semantics"] == (
        TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS
    )
    assert manifest["scientific_scope"] == (
        "engineering_smoke_only_truth_centered_ranges"
    )
    assert epoch["validation_examples"] == 3
    assert epoch["validation_distributed_examples"] == 2
    assert epoch["validation_coordinator_tail_examples"] == 1
    assert epoch["validation_examples_not_used"] == 0
    loaded = load_local_trained_proposal_model(output)
    assert loaded.name == LOCAL_PROPOSAL_MODEL_NAME
    with pytest.raises(ValueError, match="incompatible"):
        load_trained_proposal_model(result.model_path)

    manifest["scientific_scope"] = "bounds_first_training"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific scope"):
        validate_local_training_run(output)
    manifest["scientific_scope"] = "engineering_smoke_only_truth_centered_ranges"
    manifest["coordinate_contract"]["model_output"] = "global_unit"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate_contract"):
        validate_local_training_run(output)


def test_hybrid_validation_uses_exact_example_weighting_without_repeating_tail():
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"value": tf.constant([1.0, 2.0, 3.0])},
            {"topology_id": tf.constant([0, 0, 0], tf.int32)},
        )
    ).batch(2)

    metrics, _, audit = run_hybrid_validation(
        dataset,
        strategy=strategy,
        global_batch_size=2,
        expected_examples=3,
        distributed_step=lambda _inputs, _labels: {"loss": tf.constant(10.0)},
        coordinator_step=lambda _inputs, _labels: {"loss": tf.constant(40.0)},
    )

    assert metrics["loss"] == pytest.approx((2 * 10.0 + 40.0) / 3)
    assert audit == {
        "validation_steps": 2,
        "validation_examples": 3,
        "validation_distributed_examples": 2,
        "validation_coordinator_tail_examples": 1,
    }


def test_v2_cli_has_explicit_pilot_gate_and_no_partial_validation_mode():
    parsed = build_parser().parse_args(
        [
            "--shards",
            "one.npz",
            "--output-dir",
            "run",
            "--allow-truth-centered-range-pilot",
        ]
    )
    assert parsed.allow_truth_centered_range_pilot
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--shards",
                "one.npz",
                "--output-dir",
                "run",
                "--validation-steps",
                "1",
            ]
        )
