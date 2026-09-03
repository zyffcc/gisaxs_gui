from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    build_formal_sobol_grouped_shard_v5 as formal_builder,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_formal_sobol_grouped_shard_v5 import (
    build_formal_sobol_grouped_shard,
    main,
    plan_formal_sobol_grouped_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    V5GroupedDataset,
    read_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_data_v5 import (
    V5GroupedTrainingConfig,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_shards_v5 import (
    inspect_v5_grouped_training,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_coordinates_v5 import (
    v5_sobol_recipe_design,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_recipe_v5 import (
    V5_SOBOL_CLEAN_RECIPE_SCHEMA,
    V5_SOBOL_CLEAN_RECIPE_VERSION,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.split_design_v5 import (
    V5SplitCounts,
    V5SplitPlan,
)


def _plan() -> V5SplitPlan:
    return V5SplitPlan.create(
        V5SplitCounts(
            train=5,
            tuning_validation=2,
            calibration=1,
            test=1,
            reference=1,
            ood_topology=1,
            ood_range_width=1,
            ood_weak_component=1,
            ood_acquisition_policy=1,
        ),
        start_index=3,
        guard_band=2,
    )


def test_start_and_shard_index_resolve_only_the_requested_split_window():
    plan = _plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    block = next(value for value in plan.blocks if value.name == "train")

    start = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split="train",
        start=1,
        count=3,
        view_indices=(0, 2),
    )
    assert start.selected_indices == tuple(range(block.start + 1, block.start + 4))
    assert start.selection["sobol_index_runs"] == [
        {"start": block.start + 1, "stop": block.start + 4}
    ]
    assert start.selection["sobol_indices_contiguous"] is True

    indexed = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split="train",
        shard_index=2,
        count=2,
    )
    assert indexed.selected_indices == (block.start + 4,)
    assert indexed.split_offset == 4
    assert indexed.selection["requested_count"] == 2
    assert indexed.selection["actual_count"] == 1


def test_formal_shard_round_trip_is_exclusive_and_manifest_binds_split_and_sobol(tmp_path):
    plan = _plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    selected = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split="train",
        start=0,
        count=1,
        view_indices=(0,),
    )
    output = tmp_path / "formal-train-000000.gvd5"

    dataset, receipt = build_formal_sobol_grouped_shard(selected, output)
    replay, replay_receipt = read_v5_grouped_dataset(output)
    assert replay.manifest == dataset.manifest
    assert replay_receipt.artifact_sha256 == receipt.artifact_sha256
    metadata = replay.manifest["shard_metadata"]
    assert metadata["formal_single_split_sobol_shard"] is True
    assert metadata["split_inventory"]["split_ids"] == ["train"]
    assert metadata["split_inventory"]["mode"] == "single_split"
    assert metadata["sobol_parent_inventory"]["split_plan_sha256"] == plan.sha256
    assert metadata["sobol_parent_inventory"]["sobol_design_sha256"] == design.sha256
    assert (
        metadata["formal_sobol_selection"]["sobol_index_runs"]
        == metadata["sobol_parent_inventory"]["sobol_index_runs"]
    )

    tampered = deepcopy(dict(replay.manifest))
    selection = tampered["shard_metadata"]["formal_sobol_selection"]
    selection["view_indices"] = [999]
    selection_core = dict(selection)
    selection_core.pop("selection_sha256")
    selection["selection_sha256"] = sha256(
        canonical_json(selection_core).encode("utf-8")
    ).hexdigest()
    manifest_core = dict(tampered)
    manifest_core.pop("manifest_sha256")
    tampered["manifest_sha256"] = sha256(canonical_json(manifest_core).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="observation views"):
        V5GroupedDataset(tampered, replay.arrays)

    with pytest.raises(FileExistsError, match="overwrite"):
        build_formal_sobol_grouped_shard(selected, output)


def test_direct_sobol_train_and_validation_shards_pass_training_identity_audit(tmp_path):
    plan = _plan()
    design = v5_sobol_recipe_design(scramble_seed=20260903)
    paths = {}
    for split in ("train", "tuning_validation"):
        selected = plan_formal_sobol_grouped_shard(
            plan=plan,
            design=design,
            target_split=split,
            start=0,
            count=1,
            view_indices=(0,),
        )
        path = tmp_path / f"formal-{split}.gvd5"
        build_formal_sobol_grouped_shard(selected, path)
        paths[split] = path

    _, audit = inspect_v5_grouped_training(
        paths["train"],
        paths["tuning_validation"],
        V5GroupedTrainingConfig(
            warmup_epochs=1,
            full_epochs=0,
            recipes_per_replica=1,
            steps_per_epoch=1,
        ),
        replicas=1,
    )

    assert audit.recipe_generator_identity[0] == V5_SOBOL_CLEAN_RECIPE_SCHEMA
    assert audit.recipe_generator_identity[1] == V5_SOBOL_CLEAN_RECIPE_VERSION
    assert audit.recipe_generator_identity[2].endswith("make_mixed_model")


def test_dry_run_writes_nothing_and_ood_remains_fail_closed(tmp_path, capsys):
    plan = _plan()
    design = v5_sobol_recipe_design(scramble_seed=17)
    plan_path = tmp_path / "split-plan.json"
    design_path = tmp_path / "sobol-design.json"
    output = tmp_path / "never-written.gvd5"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    design_path.write_text(design.to_json(), encoding="utf-8")

    assert (
        main(
            (
                "--split-plan",
                str(plan_path),
                "--sobol-design",
                str(design_path),
                "--target-split",
                "train",
                "--shard-index",
                "0",
                "--count",
                "2",
                "--view-indices",
                "0,3",
                "--output",
                str(output),
                "--dry-run",
            )
        )
        == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert audit["dry_run"] is True
    assert audit["actual_count"] == 2
    assert not output.exists()

    with pytest.raises(ValueError, match="OOD design is fail-closed"):
        plan_formal_sobol_grouped_shard(
            plan=plan,
            design=design,
            target_split="ood",
            start=0,
            count=1,
        )


def test_slurm_array_wrapper_requires_worker_and_dust_output():
    wrapper = (
        Path(__file__).parents[1] / "PosteriorV8" / "slurm" / "v5_formal_sobol_dataset_cpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-0" in wrapper
    assert "SLURM_JOB_ID" in wrapper
    assert "max-wgs*" in wrapper
    assert "/data/dust/user/zhaiyufe/*" in wrapper
    assert "--shard-index" in wrapper
    assert "--dry-run" in wrapper


def test_actual_build_is_forbidden_on_maxwell_login_node(tmp_path, monkeypatch):
    plan = _plan()
    design = v5_sobol_recipe_design(scramble_seed=19)
    selected = plan_formal_sobol_grouped_shard(
        plan=plan,
        design=design,
        target_split="train",
        start=0,
        count=1,
    )
    monkeypatch.setattr(formal_builder.socket, "gethostname", lambda: "max-wgs01.desy.de")
    with pytest.raises(RuntimeError, match="forbidden on the Maxwell login node"):
        build_formal_sobol_grouped_shard(selected, tmp_path / "forbidden.gvd5")
