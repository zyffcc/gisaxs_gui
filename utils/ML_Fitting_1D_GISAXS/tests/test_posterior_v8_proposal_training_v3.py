from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    LOCAL_TARGET_SEMANTICS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_shards import (
    BoundsFirstShardConfig,
    BoundsFirstShardSpec,
    SPLIT_CODE,
    recipe_group_id_for,
    split_for_recipe_group,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_model_contract import (
    BOUNDS_MODEL_COORDINATE_CONTRACT,
    BOUNDS_PROPOSAL_MODEL_NAME,
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_INPUT_KEYS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_training_audit import (
    BOUNDS_RUN_MANIFEST_SCHEMA,
    BOUNDS_SCIENTIFIC_SCOPE,
    _validate_loaded_model_contract,
    load_bounds_trained_proposal_model,
    validate_bounds_training_run,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_training_data import (
    bounds_training_data,
    epoch_dataset,
    inspect_bounds_first_shards,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_training_steps import (
    run_hybrid_validation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import branch_pattern_id
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_bounds_first_shards import (
    build_shard,
    load_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.local_training_audit import (
    LOCAL_RUN_MANIFEST_SCHEMA,
    load_local_trained_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v3 import (
    build_bounds_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import topology_id_for
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training import (
    load_trained_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training_audit import (
    RUN_MANIFEST_SCHEMA,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training_v3 import (
    BoundsProposalTrainingConfig,
    train_bounds_proposal,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal_v3 import build_parser
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v2 import (
    LOCAL_TRAINING_OBJECTIVE_VERSION,
    LocalTrainingObjectiveConfig,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    active_dimension_mask_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v3 import (
    BOUNDS_OBJECTIVE_TARGET_COORDINATE_SEMANTICS,
    BOUNDS_TRAINING_OBJECTIVE_VERSION,
    BoundsTrainingObjectiveConfig,
    compute_bounds_training_objective,
)


def _recipe_count_with_required_splits(config) -> int:
    observed = set()
    for index in range(128):
        observed.add(split_for_recipe_group(recipe_group_id_for(config, index)))
        if {"train", "tuning_validation"}.issubset(observed) and observed.intersection(
            {"calibration", "test"}
        ):
            return index + 1
    raise RuntimeError("test seed did not expose required split coverage")


@pytest.fixture(scope="module")
def real_v4_shard(tmp_path_factory):
    root = tmp_path_factory.mktemp("posterior-v8-bounds-v3")
    config = BoundsFirstShardConfig(20260903, "k1", 2, 128)
    count = _recipe_count_with_required_splits(config)
    artifact = build_shard(root, config, BoundsFirstShardSpec(0, 0, count))
    return artifact.npz_path


def _config(**changes):
    values = {
        "epochs": 1,
        "global_batch_size": 2,
        "learning_rate": 2.0e-4,
        "seed": 97,
        "max_points": 1000,
        "width": 4,
        "encoder_blocks": 1,
        "mixture_components": 2,
        "shuffle_buffer": 8,
        "gradient_clip_norm": 5.0,
        "checkpoint_keep": 2,
        "steps_per_epoch": 1,
    }
    values.update(changes)
    return BoundsProposalTrainingConfig(**values)


def test_v3_objective_has_native_bounds_identity_and_distinct_config_type():
    assert BOUNDS_OBJECTIVE_TARGET_COORDINATE_SEMANTICS == LOCAL_TARGET_SEMANTICS
    assert BOUNDS_TRAINING_OBJECTIVE_VERSION != LOCAL_TRAINING_OBJECTIVE_VERSION
    assert isinstance(BoundsTrainingObjectiveConfig(), LocalTrainingObjectiveConfig)
    with pytest.raises(TypeError, match="BoundsTrainingObjectiveConfig"):
        compute_bounds_training_objective({}, {}, LocalTrainingObjectiveConfig())


def test_v3_model_and_objective_reject_permutation_duplicate_branch():
    topology_id = topology_id_for(("sphere", "sphere"))
    pattern_id = branch_pattern_id((True, False, False, False), False)
    active = np.asarray(active_dimension_mask_for(topology_id, pattern_id), dtype=np.float32)
    embedding = np.empty(78, dtype=np.float32)
    embedding[0::3] = np.where(active > 0.5, 0.0, 0.5)
    embedding[1::3] = np.where(active > 0.5, 1.0, 0.5)
    embedding[2::3] = active
    topology = np.zeros((1, 34), dtype=np.float32)
    topology[0, topology_id] = 1.0
    inputs = {
        "x": np.zeros((1, 16, 3), dtype=np.float32),
        "point_mask": np.ones((1, 16), dtype=np.bool_),
        "global_features": np.zeros((1, 5), dtype=np.float32),
        "branch_topology": topology,
        "branch_d_present": np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32),
        "branch_resolution_present": np.zeros((1, 1), dtype=np.float32),
        "bounds_embedding": embedding[np.newaxis],
        "active_dimension_mask": active[np.newaxis],
        "varying_dimension_mask": active[np.newaxis],
    }
    model = build_bounds_proposal_model(
        max_points=16, width=4, encoder_blocks=1, mixture_components=2
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="canonical"):
        model(inputs, training=False)

    outputs = {
        "topology_logits": tf.zeros((1, 34)),
        "branch_pattern_logits": tf.zeros((1, 34, 32)),
        "mixture_logits": tf.zeros((1, 2)),
        "mixture_loc": tf.zeros((1, 2, 26)),
        "mixture_logscale": tf.zeros((1, 2, 26)),
    }
    labels = {
        "topology_id": np.asarray([topology_id], np.int32),
        "branch_pattern_id": np.asarray([pattern_id], np.int32),
        "target_local": np.full((1, 26), 0.5, np.float32),
        "active_dimension_mask": active[np.newaxis],
        "varying_dimension_mask": active[np.newaxis],
    }
    with pytest.raises(tf.errors.InvalidArgumentError, match="canonical"):
        compute_bounds_training_objective(outputs, labels, BoundsTrainingObjectiveConfig())


def test_v3_adapter_uses_native_local_target_and_blocks_holdout_splits(real_v4_shard):
    shard = load_shard(real_v4_shard)
    audit = inspect_bounds_first_shards([real_v4_shard])
    assert audit.train_count > 0
    assert audit.tuning_validation_count > 0
    assert audit.calibration_count + audit.test_count > 0
    assert audit.payload["calibration_test_rows_consumed"] == 0
    assert audit.payload["consumed_splits"] == ["train", "tuning_validation"]
    with pytest.raises(ValueError, match="only consume"):
        bounds_training_data(shard, split="calibration")
    inputs, labels = bounds_training_data(shard, split="train")
    assert tuple(inputs) == MODEL_INPUT_KEYS
    assert "branch_low" not in inputs and "branch_high" not in inputs
    assert np.array_equal(inputs["active_dimension_mask"], labels["active_dimension_mask"])
    assert np.array_equal(inputs["varying_dimension_mask"], labels["varying_dimension_mask"])
    fixed_or_inactive = labels["varying_dimension_mask"] == 0
    assert np.all(labels["target_local"][fixed_or_inactive] == 0.5)

    all_active = shard.arrays["active_dimension_mask"]
    all_varying = shard.arrays["local_varying_mask"]
    assert np.any(all_active & ~all_varying), "fixture must exercise a fixed active axis"
    group_to_splits = {}
    for group, code in zip(shard.arrays["recipe_group_id"], shard.arrays["assigned_split"]):
        group_to_splits.setdefault(bytes(group), set()).add(int(code))
    assert all(len(values) == 1 for values in group_to_splits.values())


def test_discrete_heads_are_curve_only_and_bounds_context_is_physical(real_v4_shard):
    shard = load_shard(real_v4_shard)
    inputs, _ = bounds_training_data(shard, split="train")
    if inputs["x"].shape[0] < 2:
        pytest.skip("fixture unexpectedly has fewer than two training rows")
    paired = {name: np.asarray(value[:2]).copy() for name, value in inputs.items()}
    paired["x"][1] = paired["x"][0]
    paired["point_mask"][1] = paired["point_mask"][0]
    paired["global_features"][1] = paired["global_features"][0]
    model = build_bounds_proposal_model(
        max_points=1000, width=4, encoder_blocks=1, mixture_components=2
    )
    outputs = model(paired, training=False)
    np.testing.assert_array_equal(outputs["topology_logits"][0], outputs["topology_logits"][1])
    np.testing.assert_array_equal(
        outputs["branch_pattern_logits"][0],
        outputs["branch_pattern_logits"][1],
    )
    assert model.name == BOUNDS_PROPOSAL_MODEL_NAME
    assert model.posterior_v8_model_version == BOUNDS_PROPOSAL_MODEL_VERSION
    assert model.posterior_v8_branch_catalog_version == MODEL_BRANCH_CATALOG_VERSION
    assert model.posterior_v8_component_slots_version == MODEL_COMPONENT_SLOTS_VERSION
    assert {value.name.split(":", 1)[0] for value in model.inputs} == set(MODEL_INPUT_KEYS)

    invalid = {name: value[:1].copy() for name, value in inputs.items()}
    present = np.flatnonzero(invalid["bounds_embedding"][0, 2::3] == 1)
    axis = int(present[0])
    invalid["bounds_embedding"][0, 3 * axis] = 0.9
    invalid["bounds_embedding"][0, 3 * axis + 1] = 0.1
    with pytest.raises(tf.errors.InvalidArgumentError, match="lower bounds"):
        model(invalid, training=False)


def test_dataset_epoch_counts_only_authorized_rows(real_v4_shard):
    audit = inspect_bounds_first_shards([real_v4_shard])
    config = _config(global_batch_size=3, steps_per_epoch=1)
    validation = epoch_dataset(audit, config, "tuning_validation", epoch=0)
    seen = sum(int(labels["topology_id"].shape[0]) for _, labels in validation)
    assert seen == audit.tuning_validation_count
    with pytest.raises(ValueError, match="only consume"):
        epoch_dataset(audit, config, "test", epoch=0)


def test_real_v4_cpu_one_step_save_load_and_tamper_guard(real_v4_shard, tmp_path):
    output = tmp_path / "bounds-v3-run"
    result = train_bounds_proposal(
        [real_v4_shard],
        output,
        _config(),
        strategy=tf.distribute.MirroredStrategy(devices=["/cpu:0"]),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BOUNDS_RUN_MANIFEST_SCHEMA
    assert manifest["objective_version"] == BOUNDS_TRAINING_OBJECTIVE_VERSION
    assert manifest["objective_version"] != LOCAL_TRAINING_OBJECTIVE_VERSION
    assert manifest["branch_catalog_contract"] == {
        "version": MODEL_BRANCH_CATALOG_VERSION,
        "physical_branch_count": 418,
        "wire_pattern_dimension": 32,
        "noncanonical_pattern_logits": (
            "excluded_by_training_objective_and_canonical_inference_ranking"
        ),
    }
    assert manifest["component_slots_version"] == MODEL_COMPONENT_SLOTS_VERSION
    assert manifest["scientific_scope"] == BOUNDS_SCIENTIFIC_SCOPE
    assert manifest["coordinate_contract"] == dict(BOUNDS_MODEL_COORDINATE_CONTRACT)
    assert manifest["claim_limits"] == {
        "production_model": False,
        "paper_holdout": False,
        "ood_generalization": False,
        "no_solution_classification": False,
    }
    assert manifest["training_split_contract"]["calibration_test_rows_consumed"] == 0
    assert history["epochs"][0]["calibration_examples_used"] == 0
    assert history["epochs"][0]["test_examples_used"] == 0
    loaded = load_bounds_trained_proposal_model(output)
    assert loaded.name == BOUNDS_PROPOSAL_MODEL_NAME
    assert loaded.posterior_v8_model_version == BOUNDS_PROPOSAL_MODEL_VERSION
    assert loaded.posterior_v8_branch_catalog_version == MODEL_BRANCH_CATALOG_VERSION
    assert loaded.posterior_v8_component_slots_version == MODEL_COMPONENT_SLOTS_VERSION
    wrong_shape = build_bounds_proposal_model(
        max_points=16, width=4, encoder_blocks=1, mixture_components=3
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        _validate_loaded_model_contract(wrong_shape, manifest)
    with pytest.raises(ValueError):
        load_trained_proposal_model(output)
    with pytest.raises(ValueError):
        load_local_trained_proposal_model(output)

    original = result.manifest_path.read_text(encoding="utf-8")
    manifest["coordinate_contract"]["model_output"] = "global_unit"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="coordinate_contract"):
            validate_bounds_training_run(output)
    finally:
        result.manifest_path.write_text(original, encoding="utf-8")
    validate_bounds_training_run(output)
    for old_schema in (RUN_MANIFEST_SCHEMA, LOCAL_RUN_MANIFEST_SCHEMA):
        old_manifest = json.loads(original)
        old_manifest["schema_version"] = old_schema
        result.manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="schema_version"):
                load_bounds_trained_proposal_model(output)
        finally:
            result.manifest_path.write_text(original, encoding="utf-8")


def test_hybrid_validation_weights_unique_short_tail():
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"value": tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])},
            {"topology_id": tf.constant([0, 0, 0, 0, 0], tf.int32)},
        )
    ).batch(4)
    metrics, _, audit = run_hybrid_validation(
        dataset,
        strategy=strategy,
        global_batch_size=4,
        expected_examples=5,
        distributed_step=lambda _inputs, _labels: {"loss": tf.constant(10.0)},
        coordinator_step=lambda _inputs, _labels: {"loss": tf.constant(50.0)},
    )
    assert metrics["loss"] == pytest.approx(18.0)
    assert audit == {
        "validation_steps": 2,
        "validation_examples": 5,
        "validation_distributed_examples": 4,
        "validation_coordinator_tail_examples": 1,
    }


def test_v3_cli_has_no_truth_centered_or_partial_validation_escape_hatch():
    parsed = build_parser().parse_args(["--shards", "one.npz", "--output-dir", "run"])
    assert parsed.output_dir == Path("run")
    for forbidden in (
        "--allow-truth-centered-range-pilot",
        "--validation-steps",
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--shards", "one.npz", "--output-dir", "run", forbidden])
