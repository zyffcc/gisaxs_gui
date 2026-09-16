"""Publication orchestration checks; not production search or model acceptance."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import stat

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_search_training_inputs_v5 as runtime
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_search_projection_evidence_v5 import (
    V5K1SearchProjectionEvidence,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_contract_v5 import (
    build_v5_k1_training_inventory, canonical_json, validate_v5_k1_training_inventory,
)
from test_posterior_v8_k1_training_chain_contract_v5 import _artifact, _identity_authorization


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "MAXWELL_DUST_ROOT", tmp_path)
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "max-worker-test")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    authorization = _identity_authorization()
    train, tune = _artifact("train", "01"), _artifact("tuning_validation", "23")
    mappings = tuple({
        "array_task_id": index, "role": item.role,
        "source_path": f"/data/dust/user/zhaiyufe/original/{original}.gvd5",
        "source_artifact_sha256": original[0] * 64,
        "source_manifest_sha256": original[1] * 64,
        "projected_path": item.path, "projected_artifact_sha256": item.artifact_sha256,
        "projected_manifest_sha256": item.manifest_sha256,
        "clean_parent_count": 12, "ordered_recipe_branch_split_group_arrays_equal": True,
    } for index, (item, original) in enumerate(((train, "cd"), (tune, "ef"))))
    projection = V5K1SearchProjectionEvidence(
        plan_sha256="4" * 64, inventory_file_sha256="5" * 64,
        completion_file_sha256="6" * 64,
        original_identity_authorization_sha256=authorization.sha256, mappings=mappings,
    )
    inventory = build_v5_k1_training_inventory(
        source_archive_sha256="7" * 64, source_bundle_sha256="8" * 64,
        train_artifacts=(train,), tuning_artifacts=(tune,),
        train_parent_set_sha256="1" * 64, tuning_parent_set_sha256="2" * 64,
        train_tuning_disjointness_receipt_sha256="3" * 64,
        k1_phase_c_disjointness_receipt_sha256="4" * 64,
        identity_authorization=authorization, projection_evidence=projection,
    )
    paths = {}
    for name in ("balanced_dataset_completion", "train_tuning_receipt", "phase_c_completion", "three_way_receipt"):
        path = tmp_path / (name + ".json")
        path.write_text("{}\n")
        path.chmod(0o400)
        paths[name] = path
    checked = {
        "training_inventory": inventory, "projection_evidence": projection.to_payload(),
        "original_identity_authorization_sha256": authorization.sha256,
        "original_identity_publication_files": {
            name: runtime.read_only_identity(path, name) for name, path in paths.items()
        },
    }
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")
    calls = []

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return deepcopy(checked)

    monkeypatch.setattr(runtime, "read_v5_k1_search_training_inputs", read)
    return plan, paths, checked, calls


def test_publishes_valid_inventory_then_sealed_completion(tmp_path, prepared):
    plan, paths, checked, calls = prepared
    root = tmp_path / "fresh"
    result = runtime.publish_v5_k1_search_training_inputs(
        plan, output_root=root, identity_publication_paths=paths, expected_plan_sha256="4" * 64,
    )
    assert calls == [(plan, {"identity_publication_paths": paths, "expected_plan_sha256": "4" * 64})]
    inventory = Path(result["inventory"]["path"])
    completion = Path(result["completion"]["path"])
    assert validate_v5_k1_training_inventory(json.loads(inventory.read_text())) == checked["training_inventory"]
    payload = json.loads(completion.read_text())
    supplied = payload.pop("completion_sha256")
    assert sha256(canonical_json(payload).encode()).hexdigest() == supplied
    assert payload["inventory"] == runtime.read_only_identity(inventory, "inventory")
    assert payload["completion_written_last"] is True
    assert payload["gradient_training_authorized"] is False
    assert payload["phase_c_disjointness_authorized"] is False
    assert payload["scientific_acceptance_evidence"] is False
    assert completion.stat().st_mtime_ns >= inventory.stat().st_mtime_ns
    for path in (inventory, completion):
        assert stat.S_IMODE(path.stat().st_mode) == 0o400 and path.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        runtime.publish_v5_k1_search_training_inputs(plan, output_root=root, identity_publication_paths=paths)
    assert len(calls) == 1


@pytest.mark.parametrize("host,job", [("max-wgs", "123"), ("max-fs-display006", "123"), ("worker", "")])
def test_rejects_nonworker_before_read_or_write(tmp_path, prepared, monkeypatch, host, job):
    plan, paths, _, calls = prepared
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: host)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    with pytest.raises(RuntimeError):
        runtime.publish_v5_k1_search_training_inputs(plan, output_root=tmp_path / "fresh", identity_publication_paths=paths)
    assert calls == [] and not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("failure", ["missing_inventory", "missing_originals", "projection", "reader"])
def test_input_failure_does_not_create_publication(tmp_path, prepared, monkeypatch, failure):
    plan, paths, checked, _ = prepared
    if failure == "missing_inventory":
        checked["training_inventory"] = None
    elif failure == "missing_originals":
        checked["original_identity_publication_files"] = None
    elif failure == "projection":
        checked["projection_evidence"]["evidence_sha256"] = "f" * 64
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("actual collection replay failed")
        monkeypatch.setattr(runtime, "read_v5_k1_search_training_inputs", fail)
    with pytest.raises((ValueError, RuntimeError)):
        runtime.publish_v5_k1_search_training_inputs(plan, output_root=tmp_path / "fresh", identity_publication_paths=paths)
    assert not (tmp_path / "fresh").exists()


def test_drift_during_inventory_seal_leaves_no_completion(tmp_path, prepared, monkeypatch):
    plan, paths, _, _ = prepared
    seal = runtime._seal_json

    def drift(path, payload):
        result = seal(path, payload)
        paths["phase_c_completion"].chmod(0o600)
        return result

    monkeypatch.setattr(runtime, "_seal_json", drift)
    root = tmp_path / "fresh"
    with pytest.raises(ValueError, match="read-only"):
        runtime.publish_v5_k1_search_training_inputs(plan, output_root=root, identity_publication_paths=paths)
    assert (root / "training-input-inventory-v3.json").exists()
    assert not (root / "training-input-completion-v1.json").exists()


@pytest.mark.parametrize("historical", [False, True])
def test_cli_forwards_every_required_binding(monkeypatch, historical):
    captured = {}
    def publish(plan, **kwargs):
        captured.update(plan=plan, **kwargs)
        return {"gradient_training_authorized": False}
    monkeypatch.setattr(runtime, "publish_v5_k1_search_training_inputs", publish)
    names = ("plan", "output-root", "expected-plan-sha256", "expected-plan-file-sha256",
             "expected-inventory-file-sha256", "expected-completion-file-sha256", "source-root",
             "source-archive", "local-sobol-schedule-path", "calibration-path",
             "balanced-dataset-completion", "train-tuning-receipt", "phase-c-completion", "three-way-receipt")
    if historical:
        names += ("balanced-dataset-launch-plan-path",)
    assert runtime.main([value for name in names for value in ("--" + name, name)]) == 0
    assert captured["plan"] == "plan"
    assert captured["local_sobol_schedule_path"] == "local-sobol-schedule-path"
    assert len(captured["identity_publication_paths"]) == 4
    assert len(captured) == 11 + int(historical)
    if historical:
        assert captured["balanced_dataset_launch_plan_path"] == "balanced-dataset-launch-plan-path"
