from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import array_sha256
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    V5_GROUPED_DATASET_SCHEMA,
    V5_GROUPED_DATASET_VERSION,
    clean_array,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_job_staging_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_job_staging_v5 import (
    build_job_staging_proof,
    finalize_staging_proof,
    stage_training_inputs,
    validate_staging_proof,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_contract_v5 import (
    V5K1TrainingArtifact,
    build_v5_k1_training_inventory,
    canonical_json,
    write_v5_k1_training_inventory,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_dataset_audit_v5 import (
    k1_parent_set_sha256,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_plan_v5 import (
    K1_TRAINING_REQUIRED_SOURCE_FILES,
    V5K1TrainingChainConfig,
    build_v5_k1_training_chain_plan,
    fingerprint_v5_k1_training_source,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_runtime_v5 import (
    collect_v5_k1_training_handoff,
    run_v5_k1_training_seed,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.launch_k1_training_chain_v5 import (
    launch_v5_k1_training_chain,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _minimal_source(root: Path) -> Path:
    required = {
        Path("AGENTS.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("utils/__init__.py"),
        Path("src/gimap/example.py"),
        Path("docs/architecture/placeholder.md"),
        Path("docs/research/placeholder.md"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/study_protocol.py"),
        Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8/launch_k1_phase_a_dag_v5.py"),
        *K1_TRAINING_REQUIRED_SOURCE_FILES,
    }
    for index, relative in enumerate(sorted(required)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-source-{index}\n", encoding="utf-8")
    return root


def _grouped_fixture(path: Path, marker: str) -> tuple[str, str, tuple[str, ...]]:
    recipes = np.asarray(
        [
            canonical_json(
                {
                    "branch_pattern_id": branch.pattern_id,
                    "query": {"topology": [branch.shape]},
                }
            )
            for branch in K1_PHASE_C_BRANCHES
        ]
    )
    patterns = np.asarray(
        [branch.pattern_id for branch in K1_PHASE_C_BRANCHES], dtype=np.int32
    )
    splits = np.asarray([marker] * len(K1_PHASE_C_BRANCHES))
    group_ids = tuple(
        f"{marker}-clean-parent-{index:03d}" for index in range(len(K1_PHASE_C_BRANCHES))
    )
    groups = np.asarray(group_ids)
    arrays = {
        clean_array("recipe_canonical_json"): recipes,
        clean_array("target_pattern_id"): patterns,
        clean_array("split_id"): splits,
        clean_array("clean_group_id"): groups,
    }
    core = {
        "dataset_schema": V5_GROUPED_DATASET_SCHEMA,
        "dataset_version": V5_GROUPED_DATASET_VERSION,
        "fixture_marker": marker,
        "counts": {"clean_recipes": len(K1_PHASE_C_BRANCHES)},
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": array_sha256(name, value),
            }
            for name, value in arrays.items()
        },
    }
    manifest_sha = sha256(canonical_json(core).encode()).hexdigest()
    manifest = {**core, "manifest_sha256": manifest_sha}
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("manifest.json", canonical_json(manifest))
        for name, array in arrays.items():
            encoded = BytesIO()
            np.save(encoded, array, allow_pickle=False)
            archive.writestr(f"arrays/{name}.npy", encoded.getvalue())
    return _sha256_file(path), manifest_sha, group_ids


@dataclass(frozen=True)
class _Fixture:
    dust: Path
    source_root: Path
    source_archive: Path
    archive_sha256: str
    inventory: Path
    inventory_file_sha256: str
    train_dataset: Path

    def config(
        self,
        name: str,
        *,
        mode: str = "engineering_e1",
        seeds: tuple[int, ...] = (101,),
        full_epochs: int = 0,
    ) -> V5K1TrainingChainConfig:
        return V5K1TrainingChainConfig(
            source_root=self.source_root,
            source_archive=self.source_archive,
            expected_source_archive_sha256=self.archive_sha256,
            input_inventory=self.inventory,
            expected_inventory_file_sha256=self.inventory_file_sha256,
            run_root=self.dust / "runs" / name,
            mode=mode,
            model_seeds=seeds,
            warmup_epochs=1,
            full_epochs=full_epochs,
            recipes_per_replica=1,
            validation_recipes_per_batch=1,
            width=8,
            encoder_blocks=1,
            mixture_components=2,
            mixed_precision=False,
        )


@pytest.fixture
def chain_fixture(tmp_path: Path) -> _Fixture:
    dust = tmp_path / "data" / "dust" / "user" / "zhaiyufe"
    dust.mkdir(parents=True)
    staging = _minimal_source(dust / "source-staging")
    archive = dust / "source.tar"
    source_identity = build_source_snapshot(staging, archive)
    source_root = dust / "source-extracted"
    extract_source_snapshot(
        archive,
        source_root,
        expected_sha256=source_identity["archive_sha256"],
    )

    data = dust / "data"
    data.mkdir()
    train = data / "train.gvd5"
    tuning = data / "tuning.gvd5"
    train_sha, train_manifest_sha, train_groups = _grouped_fixture(train, "train")
    tuning_sha, tuning_manifest_sha, tuning_groups = _grouped_fixture(
        tuning, "tuning_validation"
    )
    counts = tuple((branch.branch_id, 1) for branch in K1_PHASE_C_BRANCHES)
    fingerprint = fingerprint_v5_k1_training_source(source_root)
    inventory = build_v5_k1_training_inventory(
        source_archive_sha256=source_identity["archive_sha256"],
        source_bundle_sha256=fingerprint["bundle_sha256"],
        train_artifacts=(
            V5K1TrainingArtifact(
                path=str(train),
                role="train",
                split_id="train",
                artifact_sha256=train_sha,
                manifest_sha256=train_manifest_sha,
                clean_parent_count=12,
                branch_counts=counts,
            ),
        ),
        tuning_artifacts=(
            V5K1TrainingArtifact(
                path=str(tuning),
                role="tuning_validation",
                split_id="tuning_validation",
                artifact_sha256=tuning_sha,
                manifest_sha256=tuning_manifest_sha,
                clean_parent_count=12,
                branch_counts=counts,
            ),
        ),
        train_parent_set_sha256=k1_parent_set_sha256(train_groups),
        tuning_parent_set_sha256=k1_parent_set_sha256(tuning_groups),
        train_tuning_disjointness_receipt_sha256="3" * 64,
        k1_phase_c_disjointness_receipt_sha256="4" * 64,
    )
    inventory_path = data / "k1-input-inventory.json"
    write_v5_k1_training_inventory(inventory_path, inventory)
    return _Fixture(
        dust=dust,
        source_root=source_root,
        source_archive=archive,
        archive_sha256=source_identity["archive_sha256"],
        inventory=inventory_path,
        inventory_file_sha256=_sha256_file(inventory_path),
        train_dataset=train,
    )


def _publish_plan(plan: dict[str, object]) -> Path:
    layout = plan["layout"]
    Path(layout["run_root"]).mkdir(parents=True)
    for name in ("logs", "audit", "models"):
        Path(layout[name]).mkdir()
    target = Path(layout["plan"])
    target.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _job_stage(
    fixture: _Fixture,
    plan: dict[str, object],
    plan_path: Path,
    *,
    job_id: str,
    worker: str,
    array_index: int | None = None,
    scratch_root: Path | None = None,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    scratch = fixture.dust.parent / "node-local-scratch" if scratch_root is None else scratch_root
    scratch.mkdir(exist_ok=True)
    prefix = (
        f"gisaxs-v5-k1-seed-{job_id}-{array_index}-"
        if worker == "training_seed"
        else f"gisaxs-v5-k1-collect-{job_id}-"
    )
    root = scratch / f"{prefix}fixture"
    root.mkdir(mode=0o700)
    local_plan = root / "k1-training-plan.json"
    shutil.copyfile(plan_path, local_plan)
    local_plan.chmod(0o400)
    local_archive = root / "source-snapshot.tar"
    shutil.copyfile(fixture.source_archive, local_archive)
    local_archive.chmod(0o400)
    local_source = root / "source"
    extract_source_snapshot(
        local_archive,
        local_source,
        expected_sha256=plan["source"]["archive_sha256"],
    )
    environment = {"SLURM_JOB_ID": job_id, "SLURM_TMPDIR": str(scratch)}
    return root, local_plan, local_source, local_archive, environment


def _rehashed(plan: dict[str, object]) -> dict[str, object]:
    result = dict(plan)
    result.pop("plan_sha256")
    return {
        **result,
        "plan_sha256": sha256(canonical_json(result).encode()).hexdigest(),
    }


def test_engineering_dry_run_is_write_free_and_freezes_gpu_array(chain_fixture: _Fixture):
    result = launch_v5_k1_training_chain(
        chain_fixture.config("dry-run"),
        allowed_root=chain_fixture.dust,
    )
    plan = result["plan"]

    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    assert not Path(plan["layout"]["run_root"]).exists()
    assert plan["configuration"]["model_seeds"] == [101]
    assert plan["slurm"]["training_array_spec"] == "0-0"
    assert plan["execution_gate"] == {
        "submission_allowed": False,
        "blockers": ["job_local_staging_security_audit_not_closed"],
        "formal_chain_complete": False,
        "login_node_work": "hash_contract_path_checks_plan_publication_and_sbatch_only",
        "training_compute": "Slurm_GPU_worker_only",
        "tuning_handoff_compute": "Slurm_CPU_worker_only",
    }
    assert "--array=0-0" in result["submission_preview"]["training"]
    assert "TRAINING_ARRAY_JOB_ID" in " ".join(
        result["submission_preview"]["tuning_handoff"]
    )
    assert plan["source"]["archive_tree_binding"]["verified"] is True


def test_formal_requires_five_seeds_and_stays_blocked_on_missing_adapters(
    chain_fixture: _Fixture,
):
    with pytest.raises(ValueError, match="at least five"):
        build_v5_k1_training_chain_plan(
            chain_fixture.config(
                "too-few", mode="formal_multiseed", seeds=(1, 2, 3, 4), full_epochs=1
            ),
            allowed_root=chain_fixture.dust,
        )

    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config(
            "formal",
            mode="formal_multiseed",
            seeds=(11, 12, 13, 14, 15),
            full_epochs=1,
        ),
        allowed_root=chain_fixture.dust,
    )

    assert plan["resource_estimate"]["training_job_count"] == 5
    assert plan["resource_estimate"]["maximum_allocated_gpu_hours"] == 240
    assert plan["execution_gate"]["submission_allowed"] is False
    assert "input inventory has no fully promoted search supervision" in plan[
        "execution_gate"
    ]["blockers"]
    assert any(
        "V5TuningCheckpointEvaluation" in blocker
        for blocker in plan["execution_gate"]["blockers"]
    )
    with pytest.raises(RuntimeError, match="fail-closed"):
        launch_v5_k1_training_chain(
            chain_fixture.config(
                "formal-submit",
                mode="formal_multiseed",
                seeds=(11, 12, 13, 14, 15),
                full_epochs=1,
            ),
            submit=True,
            allowed_root=chain_fixture.dust,
            hostname="max-wgs",
            environment={},
        )


def test_tiny_fixture_seed_and_collection_dry_runs_do_not_import_tensorflow_or_write(
    chain_fixture: _Fixture,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("runtime-dry"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)

    seed = run_v5_k1_training_seed(
        plan_path,
        0,
        expected_plan_sha256=plan["plan_sha256"],
        dry_run=True,
        allowed_root=chain_fixture.dust,
    )
    handoff = collect_v5_k1_training_handoff(
        plan_path,
        expected_plan_sha256=plan["plan_sha256"],
        dry_run=True,
        allowed_root=chain_fixture.dust,
    )

    assert seed["status"] == "checked_dry_run"
    assert len(seed["input_verification"]["verified_artifacts"]) == 2
    assert not Path(plan["seed_runs"][0]["output"]).exists()
    assert handoff["paper_checkpoint_selection_complete"] is False
    assert not Path(plan["layout"]["tuning_handoff"]).exists()


def test_runtime_rejects_dataset_bytes_changed_after_plan(chain_fixture: _Fixture):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("dataset-drift"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    chain_fixture.train_dataset.write_bytes(chain_fixture.train_dataset.read_bytes() + b"drift")

    with pytest.raises(RuntimeError, match="dataset bytes changed"):
        run_v5_k1_training_seed(
            plan_path,
            0,
            expected_plan_sha256=plan["plan_sha256"],
            dry_run=True,
            allowed_root=chain_fixture.dust,
        )


def test_seed_inputs_are_exclusive_node_local_read_only_copies(
    chain_fixture: _Fixture, monkeypatch,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("node-local-stage"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    root, local_plan, local_source, local_archive, environment = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="701",
        worker="training_seed",
        array_index=0,
    )
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(
            local_source
            / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"
        ),
    )
    common = build_job_staging_proof(
        plan,
        plan_path=local_plan,
        staging_root=root,
        source_root=local_source,
        source_archive=local_archive,
        worker_kind="training_seed",
        array_index=0,
        environment=environment,
        allowed_root=chain_fixture.dust,
    )
    staged = stage_training_inputs(
        plan, staging_root=root, allowed_root=chain_fixture.dust
    )

    assert Path(common["staging_root"]).parent == Path(environment["SLURM_TMPDIR"])
    original_paths = set((*plan["inputs"]["train_datasets"], *plan["inputs"]["validation_datasets"]))
    local_paths = {
        *staged["local_inputs"]["train_datasets"],
        *staged["local_inputs"]["validation_datasets"],
    }
    assert original_paths.isdisjoint(local_paths)
    assert all(Path(value).is_relative_to(root) for value in local_paths)
    assert all(Path(value).stat().st_mode & 0o222 == 0 for value in local_paths)
    before = {value: _sha256_file(Path(value)) for value in local_paths}
    chain_fixture.train_dataset.write_bytes(chain_fixture.train_dataset.read_bytes() + b"drift")
    assert {value: _sha256_file(Path(value)) for value in local_paths} == before

    with pytest.raises(FileExistsError, match="reuse"):
        stage_training_inputs(plan, staging_root=root, allowed_root=chain_fixture.dust)


def test_staging_rejects_symlinked_frozen_input(chain_fixture: _Fixture):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("symlink-stage"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    root, _, _, _, _ = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="702",
        worker="training_seed",
        array_index=0,
    )
    backup = chain_fixture.train_dataset.with_name("train-identical-backup.gvd5")
    shutil.copyfile(chain_fixture.train_dataset, backup)
    chain_fixture.train_dataset.unlink()
    chain_fixture.train_dataset.symlink_to(backup)

    with pytest.raises(ValueError, match="symlink"):
        stage_training_inputs(plan, staging_root=root, allowed_root=chain_fixture.dust)


def test_staging_rejects_shared_dust_as_slurm_scratch(
    chain_fixture: _Fixture, monkeypatch,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("dust-scratch"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    scratch = chain_fixture.dust / "shared-scratch"
    root, local_plan, local_source, local_archive, environment = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="704",
        worker="collector",
        scratch_root=scratch,
    )
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(local_source / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"),
    )

    with pytest.raises(ValueError, match="Slurm scratch"):
        build_job_staging_proof(
            plan,
            plan_path=local_plan,
            staging_root=root,
            source_root=local_source,
            source_archive=local_archive,
            worker_kind="collector",
            array_index=None,
            environment=environment,
            allowed_root=chain_fixture.dust,
        )


def test_non_dry_workers_fail_closed_without_complete_private_staging(
    chain_fixture: _Fixture,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("missing-stage"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)

    with pytest.raises(RuntimeError, match="complete job-private staging"):
        run_v5_k1_training_seed(
            plan_path,
            0,
            expected_plan_sha256=plan["plan_sha256"],
            allowed_root=chain_fixture.dust,
            hostname="max-gpu001",
            environment={"SLURM_JOB_ID": "703"},
        )


def test_persisted_staging_proof_claims_remain_fail_closed():
    with pytest.raises(RuntimeError, match="job_local_staging_security_audit_not_closed"):
        finalize_staging_proof(
            {}, staged_inputs=None, trainer_manifest=None
        )
    with pytest.raises(RuntimeError, match="job_local_staging_security_audit_not_closed"):
        validate_staging_proof(
            {}, {}, worker_kind="collector", expected_array_index=None
        )


def test_collection_is_fail_closed_before_reading_seed_bindings(
    chain_fixture: _Fixture,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("collect-blocked"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    with pytest.raises(RuntimeError, match="blocked and cannot execute"):
        collect_v5_k1_training_handoff(
            plan_path,
            expected_plan_sha256=plan["plan_sha256"],
            dry_run=False,
            allowed_root=chain_fixture.dust,
            hostname="max-gpu001",
            environment={"SLURM_JOB_ID": "123"},
            job_staging_root=chain_fixture.dust.parent / "unused-stage",
            job_source_root=chain_fixture.dust.parent / "unused-source",
            job_source_archive=chain_fixture.dust.parent / "unused.tar",
        )

    assert not Path(plan["layout"]["tuning_handoff"]).exists()


def test_rehashed_plan_cannot_change_resource_or_output_contract(chain_fixture: _Fixture):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("tampered-plan"), allowed_root=chain_fixture.dust
    )
    plan["slurm"]["training_resources"]["gpus_per_task"] = 1
    plan = _rehashed(plan)

    with pytest.raises(ValueError, match="Slurm resource"):
        validate_v5_k1_training_chain_plan(plan)


def test_source_archive_and_extracted_tree_are_reverified(chain_fixture: _Fixture):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("source-drift"), allowed_root=chain_fixture.dust
    )
    required = (
        chain_fixture.source_root
        / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"
    )
    required.chmod(0o644)

    with pytest.raises(ValueError, match="read-only"):
        replay_v5_k1_training_chain_fingerprints(plan, allowed_root=chain_fixture.dust)


def test_engineering_submit_is_fail_closed_before_publication_or_sbatch(
    chain_fixture: _Fixture,
):
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]):
        calls.append(tuple(argv))
        raise AssertionError("blocked K1 chain must not invoke sbatch")

    config = chain_fixture.config("submitted")
    with pytest.raises(RuntimeError, match="job_local_staging_security_audit_not_closed"):
        launch_v5_k1_training_chain(
            config,
            submit=True,
            runner=runner,
            allowed_root=chain_fixture.dust,
            hostname="max-wgs.desy.de",
            environment={},
        )

    assert calls == []
    assert not config.run_root.exists()


def test_reused_run_root_and_wrong_explicit_archive_sha_fail_closed(
    chain_fixture: _Fixture,
):
    reused = chain_fixture.config("reused")
    reused.run_root.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="reuse"):
        build_v5_k1_training_chain_plan(reused, allowed_root=chain_fixture.dust)

    wrong = replace(
        chain_fixture.config("wrong-sha"),
        expected_source_archive_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="explicit expectation"):
        build_v5_k1_training_chain_plan(wrong, allowed_root=chain_fixture.dust)


@pytest.mark.parametrize(
    ("wrapper_name", "heavy_command"),
    (
        ("v5_k1_training_seed_gpu4.sbatch", " train-seed \\\n"),
        ("v5_k1_training_collect_cpu.sbatch", " collect \\\n"),
    ),
)
def test_k1_workers_verify_node_local_source_and_plan_before_use(
    wrapper_name: str, heavy_command: str,
):
    wrapper = (
        Path(__file__).parents[1] / "PosteriorV8" / "slurm" / wrapper_name
    ).read_text(encoding="utf-8")

    assert "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}" in wrapper
    assert "job-private staging cannot use the shared dust tree" in wrapper
    assert "mktemp -d" in wrapper
    assert "job-cache/" not in wrapper
    assert "POSTERIOR_V8_JOB_PLAN" in wrapper
    assert "POSTERIOR_V8_JOB_SOURCE_ARCHIVE" in wrapper
    assert "POSTERIOR_V8_JOB_SOURCE_ROOT" in wrapper
    assert "runtime was not" not in wrapper
    assert 'cd "$POSTERIOR_V8_SOURCE_ROOT"' not in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert "python -I -c" in wrapper
    assert '--plan "$POSTERIOR_V8_JOB_PLAN"' in wrapper
    assert '--job-staging-root "$POSTERIOR_V8_JOB_STAGING_ROOT"' in wrapper
    assert "POSTERIOR_V8_JOB_ARCHIVE_SHA256" not in wrapper
    assert '--expected-sha256 "$POSTERIOR_V8_FROZEN_ARCHIVE_SHA256"' in wrapper
    assert '--expected-manifest-sha256 "$POSTERIOR_V8_FROZEN_MANIFEST_SHA256"' in wrapper
    assert (
        '--expected-source-tree-sha256 "$POSTERIOR_V8_FROZEN_SOURCE_TREE_SHA256"'
        in wrapper
    )
    assert wrapper.index("local_plan_value, _ = checked_plan(") < wrapper.index(
        "    extract \\")
    assert wrapper.index("verify-extracted") < wrapper.index(
        'cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"'
    )
    assert wrapper.index("verify-extracted") < wrapper.index(heavy_command)
