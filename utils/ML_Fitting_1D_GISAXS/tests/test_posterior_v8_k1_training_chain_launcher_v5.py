from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
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
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import job_local_input_capability_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_staging_files_v5
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.job_local_input_capability_v5 import (
    _mint_job_local_input_capability,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_input_closure_v5 import (
    rehash_staged_artifacts,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_job_staging_v5 import (
    build_job_staging_proof,
    freeze_staging_tree,
    stage_training_inputs,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_staging_receipt_v5 import (
    finalize_staging_proof,
    validate_staging_proof,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_training_v5 import (
    V5JobLocalInputCapability,
    _job_local_capability_post_validation_payload,
    _validate_job_local_input_capability,
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
    V5CommandResult,
    launch_v5_k1_training_chain,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.package_source_snapshot_v5 import (
    build_source_snapshot,
    extract_source_snapshot,
)


WRAPPER_MINT_BYTES = b"fixture-wrapper-mint-state-0001x"


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


def _fixture_with_sidecar_evidence(fixture: _Fixture) -> _Fixture:
    raw = json.loads(fixture.inventory.read_text(encoding="utf-8"))
    artifacts = {}
    for role in ("train", "tuning_validation"):
        original = raw["artifacts"][role][0]
        marker = role.replace("_validation", "")
        sidecar = fixture.inventory.parent / f"{marker}-search.gsv5"
        manifest_core = {"fixture": marker, "kind": "frozen_search_sidecar"}
        manifest = {
            **manifest_core,
            "manifest_sha256": sha256(canonical_json(manifest_core).encode()).hexdigest(),
        }
        with zipfile.ZipFile(sidecar, "x") as archive:
            archive.writestr("manifest.json", canonical_json(manifest))
        evidence = fixture.inventory.parent / f"{marker}-executor.bin"
        evidence.write_bytes(f"{marker}-executor-evidence\n".encode())
        receipt = sidecar.with_suffix(".evidence-receipt.json")
        receipt.write_text(
            json.dumps(
                {
                    "branch_evidence": [
                        {
                            "relative_path": evidence.name,
                            "artifact_sha256": _sha256_file(evidence),
                        }
                    ]
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts[role] = V5K1TrainingArtifact(
            path=original["path"],
            role=role,
            split_id=original["split_id"],
            artifact_sha256=original["artifact_sha256"],
            manifest_sha256=original["manifest_sha256"],
            clean_parent_count=original["clean_parent_count"],
            branch_counts=tuple(
                (branch.branch_id, original["branch_counts"][branch.branch_id])
                for branch in K1_PHASE_C_BRANCHES
            ),
            sidecar_path=str(sidecar),
            sidecar_artifact_sha256=_sha256_file(sidecar),
            sidecar_manifest_sha256=manifest["manifest_sha256"],
            evidence_receipt_path=str(receipt),
            evidence_receipt_sha256=_sha256_file(receipt),
            full_training_eligible=False,
        )
    inventory = build_v5_k1_training_inventory(
        source_archive_sha256=raw["source_archive_sha256"],
        source_bundle_sha256=raw["source_bundle_sha256"],
        train_artifacts=(artifacts["train"],),
        tuning_artifacts=(artifacts["tuning_validation"],),
        train_parent_set_sha256=raw["splits"]["train_parent_set_sha256"],
        tuning_parent_set_sha256=raw["splits"]["tuning_parent_set_sha256"],
        train_tuning_disjointness_receipt_sha256=raw["splits"][
            "train_tuning_disjointness_receipt_sha256"
        ],
        k1_phase_c_disjointness_receipt_sha256=raw["splits"][
            "k1_phase_c_disjointness_receipt_sha256"
        ],
    )
    path = fixture.inventory.parent / "k1-input-inventory-with-sidecars.json"
    write_v5_k1_training_inventory(path, inventory)
    return replace(
        fixture,
        inventory=path,
        inventory_file_sha256=_sha256_file(path),
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
    scratch = fixture.dust / "private-job-scratch" if scratch_root is None else scratch_root
    scratch.mkdir(exist_ok=True)
    prefix = (
        f"gisaxs-v5-k1-seed-{job_id}-{array_index}-"
        if worker == "training_seed"
        else f"gisaxs-v5-k1-collect-{job_id}-"
    )
    job_tmp = scratch / f"{prefix}fixture0"
    job_tmp.mkdir(mode=0o700)
    root = job_tmp / "staging"
    root.mkdir(mode=0o700)
    (job_tmp / "runtime-cache").mkdir(mode=0o700)
    wrapper_mint = job_tmp / ".posterior-v8-wrapper-mint"
    wrapper_mint.write_bytes(WRAPPER_MINT_BYTES)
    assert wrapper_mint.stat().st_size == 32
    wrapper_mint.chmod(0o400)
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
    environment = {
        "SLURM_JOB_ID": job_id,
        "TMPDIR": str(scratch),
        "POSTERIOR_V8_SCRATCH_BASE": str(scratch),
        "POSTERIOR_V8_JOB_TMP_ROOT": str(job_tmp),
    }
    if worker == "training_seed":
        environment["SLURM_ARRAY_TASK_ID"] = str(array_index)
    return root, local_plan, local_source, local_archive, environment


def _rehashed(plan: dict[str, object]) -> dict[str, object]:
    result = dict(plan)
    result.pop("plan_sha256")
    return {
        **result,
        "plan_sha256": sha256(canonical_json(result).encode()).hexdigest(),
    }


def _minted_training_capability(
    fixture: _Fixture,
    monkeypatch,
    *,
    name: str,
    job_id: str,
):
    fixture = _fixture_with_sidecar_evidence(fixture)
    plan = build_v5_k1_training_chain_plan(
        fixture.config(name), allowed_root=fixture.dust
    )
    plan_path = _publish_plan(plan)
    root, local_plan, local_source, local_archive, environment = _job_stage(
        fixture,
        plan,
        plan_path,
        job_id=job_id,
        worker="training_seed",
        array_index=0,
    )
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(local_source / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"),
    )
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    common = build_job_staging_proof(
        plan,
        plan_path=local_plan,
        staging_root=root,
        source_root=local_source,
        source_archive=local_archive,
        worker_kind="training_seed",
        array_index=0,
        environment=environment,
        allowed_root=fixture.dust,
    )
    staged = stage_training_inputs(
        plan, staging_root=root, allowed_root=fixture.dust
    )
    freeze_staging_tree(root)
    capability = _mint_job_local_input_capability(
        plan,
        common,
        staged,
        environment=environment,
        allowed_root=fixture.dust,
    )
    arguments = tuple(
        value
        for role in (
            "train_datasets",
            "validation_datasets",
            "train_sidecars",
            "validation_sidecars",
        )
        for value in staged["local_inputs"][role]
    )
    return fixture, plan, common, staged, capability, arguments, environment


def _capability_closure_target(
    name: str,
    fixture: _Fixture,
    plan: dict[str, object],
    common: dict[str, object],
    staged: dict[str, object],
) -> Path:
    by_kind = {
        (scope, item["kind"]): Path(item[f"{scope}_path"])
        for item in staged["artifacts"]
        for scope in ("original", "job_local")
    }
    local_source = Path(common["source"]["job_local_source_root"])
    first_source = next(iter(plan["source"]["required_file_sha256"]))
    targets = {
        "original_inventory": Path(plan["input_inventory"]["path"]),
        "job_local_inventory": Path(staged["inventory"]["job_local_path"]),
        "original_parent": by_kind[("original", "grouped_parent")],
        "job_local_parent": by_kind[("job_local", "grouped_parent")],
        "original_sidecar": by_kind[("original", "frozen_search_sidecar")],
        "job_local_sidecar": by_kind[("job_local", "frozen_search_sidecar")],
        "original_evidence_receipt": by_kind[
            ("original", "task_bound_search_evidence_receipt")
        ],
        "job_local_evidence_receipt": by_kind[
            ("job_local", "task_bound_search_evidence_receipt")
        ],
        "original_executor_evidence": by_kind[("original", "executor_evidence")],
        "job_local_executor_evidence": by_kind[("job_local", "executor_evidence")],
        "original_plan": Path(common["plan"]["original_path"]),
        "job_local_plan": Path(common["plan"]["job_local_path"]),
        "original_source_archive": Path(common["source"]["original_archive_path"]),
        "job_local_source_archive": Path(common["source"]["job_local_archive_path"]),
        "original_source_tree": fixture.source_root / first_source,
        "job_local_source_manifest": local_source / "SOURCE-MANIFEST.json",
        "job_local_source_tree": local_source / first_source,
    }
    return targets[name]


def _append_drift(path: Path) -> None:
    path.chmod(0o600)
    with path.open("ab") as stream:
        stream.write(b"post-mint-drift")


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
    gate = plan["execution_gate"]
    assert gate["submission_allowed"] is True
    assert gate["blockers"] == []
    assert gate["formal_chain_complete"] is False
    assert gate["job_local_input_security"] == {
        "status": "closed_by_live_wrapper_minted_capability_v2",
        "capability_is_process_live_and_single_use": True,
        "serialized_capability_or_receipt_authorizes_training": False,
        "wrapper_created_owner_private_job_tmp_root_required": True,
        "wrapper_mint_is_exclusive_one_shot_and_live_consumed": True,
        "capability_has_no_public_factory": True,
        "mint_token_seal_and_digest_are_never_audited": True,
        "shared_scratch_base_alone_authorizes_training": False,
        "original_and_job_local_bytes_rehashed_pre_and_post_training": True,
        "source_archive_manifest_and_tree_reverified_pre_and_post_training": True,
        "trainer_arguments_must_equal_verified_local_copy_set": True,
        "output_remains_restricted_to_user_dust": True,
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


@pytest.mark.parametrize("hostname", ["max-wgs", "max-wgs001.desy.de",
                                      "max-fs-display006", "max-fs-display.desy.de"])
@pytest.mark.parametrize("operation", ["train", "collect"])
def test_login_hosts_reject_training_and_collection_before_input_io(tmp_path, hostname, operation):
    missing = tmp_path / "nonexistent-plan.json"
    arguments = dict(hostname=hostname, environment={"SLURM_JOB_ID": "12345"})
    with pytest.raises(RuntimeError, match="login node"):
        if operation == "train":
            run_v5_k1_training_seed(missing, 0, **arguments)
        else:
            collect_v5_k1_training_handoff(missing, **arguments)
    assert not list(tmp_path.iterdir())


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

    assert Path(common["job_tmp_root"]).parent == Path(environment["TMPDIR"])
    assert Path(common["staging_root"]) == Path(common["job_tmp_root"]) / "staging"
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


@pytest.mark.parametrize("attack", ("mutate", "replace"))
def test_stable_file_hash_rejects_mutation_or_same_byte_path_replacement(
    tmp_path: Path, monkeypatch, attack: str,
):
    target = tmp_path / "frozen.bin"
    target.write_bytes(b"frozen-byte-identity")
    displaced = tmp_path / "displaced.bin"
    real_sha256 = sha256

    class ReplacingDigest:
        def __init__(self):
            self._inner = real_sha256()
            self._replaced = False

        def update(self, chunk: bytes) -> None:
            self._inner.update(chunk)
            if not self._replaced:
                self._replaced = True
                if attack == "replace":
                    target.rename(displaced)
                    target.write_bytes(displaced.read_bytes())
                else:
                    with target.open("ab") as stream:
                        stream.write(b"mutation-race")

        def hexdigest(self) -> str:
            return self._inner.hexdigest()

    monkeypatch.setattr(k1_staging_files_v5, "sha256", ReplacingDigest)

    with pytest.raises(RuntimeError, match="changed|replaced"):
        k1_staging_files_v5.file_sha256(target, "mutation-race fixture")


def test_staging_rejects_scratch_outside_dust(
    chain_fixture: _Fixture, monkeypatch,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("dust-scratch"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    scratch = chain_fixture.dust.parent / "outside-dust-scratch"
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


@pytest.mark.parametrize(
    "attack",
    (
        "missing_or_reused_mint",
        "writable_mint",
        "hardlinked_mint",
        "unexpected_prior_content",
        "wrong_root_mode",
        "symlinked_runtime_cache",
        "symlinked_job_root",
    ),
)
def test_staging_rejects_forged_or_reused_tmp_roots(
    chain_fixture: _Fixture, monkeypatch, attack: str,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config(f"tmp-root-attack-{attack}"),
        allowed_root=chain_fixture.dust,
    )
    plan_path = _publish_plan(plan)
    root, local_plan, local_source, local_archive, environment = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="714",
        worker="training_seed",
        array_index=0,
    )
    job_tmp = root.parent
    mint = job_tmp / ".posterior-v8-wrapper-mint"
    if attack == "missing_or_reused_mint":
        mint.unlink()
    elif attack == "writable_mint":
        mint.chmod(0o600)
    elif attack == "hardlinked_mint":
        (job_tmp.parent / "attacker-mint-link").hardlink_to(mint)
    elif attack == "unexpected_prior_content":
        (job_tmp / "attacker-content").write_text("unexpected", encoding="utf-8")
    elif attack == "wrong_root_mode":
        job_tmp.chmod(0o750)
    elif attack == "symlinked_runtime_cache":
        (job_tmp / "runtime-cache").rmdir()
        (job_tmp / "runtime-cache").symlink_to(job_tmp.parent, target_is_directory=True)
    else:
        moved = job_tmp.with_name(f"{job_tmp.name}-real")
        job_tmp.rename(moved)
        job_tmp.symlink_to(moved, target_is_directory=True)
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(local_source / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"),
    )
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        build_job_staging_proof(
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


def test_staging_rejects_arbitrary_existing_tmp_directory(
    chain_fixture: _Fixture, monkeypatch,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("arbitrary-tmp-root"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    root, local_plan, local_source, local_archive, environment = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="715",
        worker="training_seed",
        array_index=0,
    )
    arbitrary = root.parent.with_name("attacker-existing-directory")
    root.parent.rename(arbitrary)
    environment["POSTERIOR_V8_JOB_TMP_ROOT"] = str(arbitrary)
    root = arbitrary / "staging"
    local_plan = root / local_plan.name
    local_source = root / "source"
    local_archive = root / local_archive.name
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(local_source / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"),
    )
    with pytest.raises(ValueError, match="Slurm job binding"):
        build_job_staging_proof(
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


def test_job_local_capability_is_opaque_single_use_and_exact_argument_bound(
    chain_fixture: _Fixture, monkeypatch,
):
    (
        fixture,
        plan,
        common,
        staged,
        capability,
        arguments,
        environment,
    ) = _minted_training_capability(
        chain_fixture, monkeypatch, name="opaque-capability", job_id="706"
    )

    assert not hasattr(V5JobLocalInputCapability, "create")
    assert "_mint_job_local_input_capability" not in job_local_input_capability_v5.__all__
    with pytest.raises(TypeError, match="opaque"):
        V5JobLocalInputCapability()
    with pytest.raises(TypeError, match="subclassed"):
        class ForgedCapability(V5JobLocalInputCapability):
            pass
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(capability)
    with pytest.raises(TypeError):
        copy.copy(capability)
    with pytest.raises(TypeError):
        copy.deepcopy(capability)
    forged = object.__new__(V5JobLocalInputCapability)
    forged._nonce = b"forged"
    assert forged is not capability
    assert forged != capability
    assert hash(capability) == hash(capability)
    assert repr(capability) == "<V5JobLocalInputCapability opaque>"
    with pytest.raises(RuntimeError, match="not minted"):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            forged,
            phase="pre_training",
        )
    audit = capability.audit_payload()
    serialized_audit = canonical_json(audit)
    assert audit["scratch_base_device"] == common["scratch_base_device"]
    assert audit["scratch_base_inode"] == common["scratch_base_inode"]
    assert audit["job_tmp_root_owner_private"] is True
    assert audit["staging_root_owner_private"] is True
    assert (
        audit["authorization"]["shared_scratch_base_alone_authorizes_use"]
        is False
    )
    assert WRAPPER_MINT_BYTES.hex() not in serialized_audit
    assert sha256(WRAPPER_MINT_BYTES).hexdigest() not in serialized_audit
    assert not {
        "nonce",
        "seal",
        "seal_key",
        "token",
        "token_sha256",
        "wrapper_mint_live_identity",
    }.intersection(audit)
    audit["trainer_argument_local_paths"] = ["/tmp/substituted"]
    assert capability.audit_payload()["trainer_argument_local_paths"] == list(arguments)
    with pytest.raises(TypeError, match="unsupported type"):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            audit,
            phase="pre_training",
        )
    with pytest.raises((RuntimeError, ValueError), match="mint|prior content"):
        _mint_job_local_input_capability(
            plan,
            common,
            staged,
            environment=environment,
            allowed_root=fixture.dust,
        )
    with pytest.raises(RuntimeError, match="other than"):
        _validate_job_local_input_capability(
            (Path(arguments[0]), *tuple(Path(value) for value in arguments[:-1])),
            capability,
            phase="pre_training",
        )

    expected_rehash = capability.audit_payload()["pre_training_rehash_sha256"]
    assert _validate_job_local_input_capability(
        tuple(Path(value) for value in arguments),
        capability,
        phase="pre_training",
    ) == expected_rehash
    with pytest.raises(RuntimeError, match="single-use"):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            capability,
            phase="pre_training",
        )
    assert _validate_job_local_input_capability(
        tuple(Path(value) for value in arguments),
        capability,
        phase="post_training",
    ) == expected_rehash
    with pytest.raises(RuntimeError, match="single-use"):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            capability,
            phase="post_training",
        )


@pytest.mark.parametrize(
    "tamper_target",
    (
        "original_inventory",
        "job_local_inventory",
        "original_parent",
        "job_local_parent",
        "original_sidecar",
        "job_local_sidecar",
        "original_evidence_receipt",
        "job_local_evidence_receipt",
        "original_executor_evidence",
        "job_local_executor_evidence",
        "original_plan",
        "job_local_plan",
        "original_source_archive",
        "job_local_source_archive",
        "original_source_tree",
        "job_local_source_manifest",
        "job_local_source_tree",
    ),
)
def test_job_local_capability_post_validation_detects_toctou_across_full_closure(
    chain_fixture: _Fixture, monkeypatch, tamper_target: str,
):
    (
        fixture,
        plan,
        common,
        staged,
        capability,
        arguments,
        _,
    ) = _minted_training_capability(
        chain_fixture,
        monkeypatch,
        name=f"toctou-{tamper_target}",
        job_id="707",
    )
    _validate_job_local_input_capability(
        tuple(Path(value) for value in arguments),
        capability,
        phase="pre_training",
    )
    _append_drift(
        _capability_closure_target(tamper_target, fixture, plan, common, staged)
    )

    with pytest.raises(
        (RuntimeError, ValueError),
        match=(
            "changed|identity|read-only|archive|binding|invalid|manifest|reproduce|"
            "valid JSON|byte-for-byte"
        ),
    ):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            capability,
            phase="post_training",
        )


@pytest.mark.parametrize(
    "tamper_target",
    (
        "original_parent",
        "job_local_sidecar",
        "original_plan",
        "job_local_source_manifest",
    ),
)
def test_job_local_capability_pre_validation_replays_full_closure(
    chain_fixture: _Fixture, monkeypatch, tamper_target: str,
):
    fixture, plan, common, staged, capability, arguments, _ = (
        _minted_training_capability(
            chain_fixture,
            monkeypatch,
            name=f"pre-toctou-{tamper_target}",
            job_id="717",
        )
    )
    _append_drift(
        _capability_closure_target(tamper_target, fixture, plan, common, staged)
    )
    with pytest.raises((RuntimeError, ValueError)):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            capability,
            phase="pre_training",
        )


@pytest.mark.parametrize("link_attack", ("hardlink", "symlink"))
def test_job_local_capability_rejects_linked_local_input(
    chain_fixture: _Fixture, monkeypatch, link_attack: str,
):
    _, _, common, staged, capability, arguments, _ = _minted_training_capability(
        chain_fixture,
        monkeypatch,
        name=f"local-{link_attack}",
        job_id="718",
    )
    target = Path(staged["local_inputs"]["train_datasets"][0])
    if link_attack == "hardlink":
        (Path(common["scratch_base"]) / "attacker-hardlink").hardlink_to(target)
    else:
        target.parent.chmod(0o700)
        original = target.with_name("renamed-parent.gvd5")
        target.rename(original)
        target.symlink_to(original)
    with pytest.raises((RuntimeError, ValueError), match="hard link|symlink"):
        _validate_job_local_input_capability(
            tuple(Path(value) for value in arguments),
            capability,
            phase="pre_training",
        )


def test_training_staging_receipt_binds_capability_and_pre_post_rehash(
    chain_fixture: _Fixture, monkeypatch,
):
    (
        fixture,
        plan,
        common,
        staged,
        capability,
        arguments,
        environment,
    ) = _minted_training_capability(
        chain_fixture, monkeypatch, name="training-proof", job_id="708"
    )
    paths = tuple(Path(value) for value in arguments)
    _validate_job_local_input_capability(
        paths, capability, phase="pre_training"
    )
    _validate_job_local_input_capability(
        paths, capability, phase="post_training"
    )
    attestation = _job_local_capability_post_validation_payload(capability)
    trainer_rows = [
        row
        for row in attestation["capability"]["original_to_local_inputs"]
        if row["kind"] in {"grouped_parent", "frozen_search_sidecar"}
    ]
    manifest_core = {
        "status": "complete",
        "plan_sha256": "9" * 64,
        "input_artifacts": [
            {
                "path": row["job_local_path"],
                "artifact_sha256": row["sha256"],
            }
            for row in trainer_rows
        ],
        "job_local_input_capability": attestation,
    }
    manifest = {
        **manifest_core,
        "result_sha256": sha256(canonical_json(manifest_core).encode()).hexdigest(),
    }
    post_rehash = rehash_staged_artifacts(
        staged,
        plan=plan,
        allowed_root=fixture.dust,
    )
    staged["trainer_staging_capability"] = capability.audit_payload()
    proof = finalize_staging_proof(
        common,
        plan=plan,
        staged_inputs=staged,
        trainer_manifest=manifest,
        trainer_capability=capability,
        post_training_rehash_sha256=post_rehash,
        allowed_root=fixture.dust,
    )

    assert proof["input_binding"]["pre_post_byte_identity_equal"] is True
    assert proof["input_binding"]["serialized_receipt_authorizes_training"] is False
    assert validate_staging_proof(
        proof,
        plan,
        worker_kind="training_seed",
        expected_array_index=0,
        require_live_staging=True,
        environment=environment,
        allowed_root=fixture.dust,
    ) == proof


def test_collector_staging_receipt_is_strict_but_never_training_authority(
    chain_fixture: _Fixture, monkeypatch,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("collector-proof"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    root, local_plan, local_source, local_archive, environment = _job_stage(
        chain_fixture,
        plan,
        plan_path,
        job_id="705",
        worker="collector",
    )
    monkeypatch.setattr(
        k1_job_staging_v5,
        "__file__",
        str(local_source / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_job_staging_v5.py"),
    )
    common = build_job_staging_proof(
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
    freeze_staging_tree(root)
    proof = finalize_staging_proof(
        common,
        plan=plan,
        staged_inputs=None,
        trainer_manifest=None,
        allowed_root=chain_fixture.dust,
    )

    assert proof["input_binding"] is None
    assert proof["trainer_result"] is None
    assert validate_staging_proof(
        proof,
        plan,
        worker_kind="collector",
        expected_array_index=None,
        require_live_staging=True,
        environment=environment,
        allowed_root=chain_fixture.dust,
    ) == proof
    tampered = copy.deepcopy(proof)
    tampered["job_tmp_root"] = "/tmp/gisaxs-v5-k1-collect-705-attacker-existing"
    core = dict(tampered)
    core.pop("proof_sha256")
    tampered["proof_sha256"] = sha256(canonical_json(core).encode()).hexdigest()
    with pytest.raises(ValueError, match="wrapper-created temporary root"):
        validate_staging_proof(
            tampered,
            plan,
            worker_kind="collector",
            expected_array_index=None,
            allowed_root=chain_fixture.dust,
        )


def test_collection_requires_wrapper_created_job_tmp_root_before_seed_bindings(
    chain_fixture: _Fixture,
):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("collect-blocked"), allowed_root=chain_fixture.dust
    )
    plan_path = _publish_plan(plan)
    unused_stage = chain_fixture.dust.parent / "unused-stage"
    unused_stage.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="did not export"):
        collect_v5_k1_training_handoff(
            plan_path,
            expected_plan_sha256=plan["plan_sha256"],
            dry_run=False,
            allowed_root=chain_fixture.dust,
            hostname="max-gpu001",
            environment={"SLURM_JOB_ID": "123"},
            job_staging_root=unused_stage,
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


@pytest.mark.parametrize("module", [
    "tuning_checkpoint_model_v5.py", "tuning_search_recorder_v5.py",
    "tuning_checkpoint_runtime_v5.py", "tuning_checkpoint_summary_io_v5.py",
    "tuning_trace_artifact_v5.py", "tuning_lossless_emission_store_v5.py",
    "paper_representative_history_v5.py", "paper_representative_history_store_v5.py",
    "paper_checkpoint_selector_v5.py", "paper_budget_evaluator_v5.py",
])
def test_training_source_requires_tuning_evidence_modules(tmp_path, module):
    relative = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8") / module
    assert relative in K1_TRAINING_REQUIRED_SOURCE_FILES
    for required in K1_TRAINING_REQUIRED_SOURCE_FILES:
        if required != relative:
            target = tmp_path / required
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# source inventory fixture\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=module):
        fingerprint_v5_k1_training_source(tmp_path)


def test_source_archive_and_extracted_tree_are_reverified(chain_fixture: _Fixture):
    plan = build_v5_k1_training_chain_plan(
        chain_fixture.config("source-drift"), allowed_root=chain_fixture.dust
    )
    verifier_relative = (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"
    )
    assert verifier_relative in plan["source"]["required_file_sha256"]
    required = (
        chain_fixture.source_root
        / "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"
    )
    required.chmod(0o644)

    with pytest.raises((RuntimeError, ValueError), match="changed|read-only"):
        replay_v5_k1_training_chain_fingerprints(plan, allowed_root=chain_fixture.dust)


def test_engineering_submit_is_enabled_only_after_security_contract_is_closed(
    chain_fixture: _Fixture,
):
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]):
        calls.append(tuple(argv))
        return V5CommandResult(0, f"{700 + len(calls)}\n", "")

    config = chain_fixture.config("submitted")
    receipt = launch_v5_k1_training_chain(
        config,
        submit=True,
        runner=runner,
        allowed_root=chain_fixture.dust,
        hostname="max-wgs.desy.de",
        environment={},
    )

    assert receipt["status"] == "submitted"
    assert receipt["job_ids"] == {
        "training_seed_array": "701",
        "tuning_handoff": "702",
    }
    assert len(calls) == 2
    assert config.run_root.is_dir()


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

    assert 'export POSTERIOR_V8_SCRATCH_BASE="/data/dust/user/zhaiyufe"' in wrapper
    assert "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}" not in wrapper
    verifier_relative = (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"
    )
    assert Path(verifier_relative) in K1_TRAINING_REQUIRED_SOURCE_FILES
    assert "POSTERIOR_V8_JOB_TMP_ROOT=\"$(mktemp -d" in wrapper
    assert "export POSTERIOR_V8_JOB_TMP_ROOT" in wrapper
    assert ".posterior-v8-wrapper-mint" in wrapper
    assert "os.O_CREAT" in wrapper and "os.O_EXCL" in wrapper
    assert 'getattr(os, "O_NOFOLLOW", 0)' in wrapper
    assert 'getattr(os, "O_CLOEXEC", 0)' in wrapper
    assert "stream.write(os.urandom(32))" in wrapper
    assert "os.fsync(stream.fileno())" in wrapper
    assert "os.fchmod(stream.fileno(), 0o400)" in wrapper
    assert 'POSTERIOR_V8_JOB_STAGING_ROOT="$POSTERIOR_V8_JOB_TMP_ROOT/staging"' in wrapper
    assert 'POSTERIOR_V8_RUNTIME_CACHE_ROOT="$POSTERIOR_V8_JOB_TMP_ROOT/runtime-cache"' in wrapper
    assert 'Path(os.environ["POSTERIOR_V8_JOB_TMP_ROOT"])' in wrapper
    assert "wrapper job-private temporary root is not fresh owner-private scratch" in wrapper
    assert 'Path(os.environ["POSTERIOR_V8_SCRATCH_BASE"])' in wrapper
    assert 'export TMPDIR="$POSTERIOR_V8_RUNTIME_CACHE_ROOT"' in wrapper
    assert "mktemp -d" in wrapper
    assert "job-cache/" not in wrapper
    assert "POSTERIOR_V8_JOB_PLAN" in wrapper
    assert "POSTERIOR_V8_JOB_SOURCE_ARCHIVE" in wrapper
    assert "POSTERIOR_V8_JOB_SOURCE_ROOT" in wrapper
    assert "runtime was not" not in wrapper
    assert 'cd "$POSTERIOR_V8_SOURCE_ROOT"' not in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert '"$POSTERIOR_V8_PYTHON" -I -c' in wrapper
    assert '/data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2/bin/python' in wrapper
    assert 'conda run' not in wrapper
    assert 'readonly POSTERIOR_V8_PYTHON=' in wrapper
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


@pytest.mark.parametrize(
    "wrapper_name",
    ("v5_k1_training_seed_gpu4.sbatch", "v5_k1_training_collect_cpu.sbatch"),
)
def test_wrapper_mint_exclusive_create_never_overwrites_existing_path(
    tmp_path: Path, wrapper_name: str,
):
    wrapper = (
        Path(__file__).parents[1] / "PosteriorV8" / "slurm" / wrapper_name
    ).read_text(encoding="utf-8")
    marker = '<<\'PY\'\n'
    mint_program = wrapper.split(marker, 1)[1].split("\nPY\n", 1)[0]
    existing = tmp_path / ".posterior-v8-wrapper-mint"
    original = b"attacker-preexisting-content"
    existing.write_bytes(original)

    result = subprocess.run(
        [sys.executable, "-I", "-", str(existing)],
        input=mint_program,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert existing.read_bytes() == original
    fresh = tmp_path / ".posterior-v8-wrapper-mint-fresh"
    created = subprocess.run(
        [sys.executable, "-I", "-", str(fresh)],
        input=mint_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0
    assert created.stdout == ""
    assert created.stderr == ""
    assert fresh.stat().st_size == 32
    assert fresh.stat().st_mode & 0o777 == 0o400
