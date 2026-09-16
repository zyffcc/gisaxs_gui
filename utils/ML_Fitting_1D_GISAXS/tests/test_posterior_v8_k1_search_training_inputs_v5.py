"""Input bridge boundaries; scientific receipt replay has its own integration tests."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

from PosteriorV8 import k1_search_training_inputs_v5 as bridge


@pytest.fixture
def row_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "MAXWELL_DUST_ROOT", tmp_path)
    row = {"role": "train", "recipe_count": 1,
           "branch_id": bridge.K1_PHASE_C_BRANCHES[0].branch_id,
           "evidence_receipt_sha256": "a" * 64}
    for key, hash_key in (("projected_parent_path", "projected_parent_artifact_sha256"),
                          ("sidecar_path", "sidecar_artifact_sha256"),
                          ("evidence_receipt_path", "evidence_receipt_file_sha256")):
        p = tmp_path / key
        p.write_bytes(key.encode())
        p.chmod(0o400)
        row[key] = str(p)
        row[hash_key] = sha256(p.read_bytes()).hexdigest()
    receipt = SimpleNamespace(file_sha256=row["evidence_receipt_file_sha256"], manifest={
        "receipt_sha256": row["evidence_receipt_sha256"],
        "parent": {"artifact_sha256": row["projected_parent_artifact_sha256"], "manifest_sha256": "b" * 64},
        "sidecar": {"artifact_sha256": row["sidecar_artifact_sha256"], "manifest_sha256": "c" * 64},
    })
    calls = []

    def read_receipt(*args, **kwargs):
        calls.append(kwargs)
        return receipt

    monkeypatch.setattr(bridge, "read_v5_search_evidence_receipt", read_receipt)
    monkeypatch.setattr(bridge, "_artifact_audit", lambda value: {"clean_group_ids": ["group-a"]})
    return row, receipt, calls


def test_descriptor_uses_receipt_file_hash_and_role(row_fixture):
    row, receipt, calls = row_fixture
    artifact, groups = bridge._checked_row(row)
    assert artifact.evidence_receipt_sha256 == receipt.file_sha256
    assert artifact.evidence_receipt_sha256 != row["evidence_receipt_sha256"]
    assert artifact.manifest_sha256 == "b" * 64
    assert artifact.sidecar_manifest_sha256 == "c" * 64
    assert groups == ("group-a",)
    assert calls[0]["require_training_eligible"] is True
    assert calls[0]["expected_consumer_role"] == "gradient_training"


@pytest.mark.parametrize("field", ["projected_parent_artifact_sha256", "sidecar_artifact_sha256", "evidence_receipt_file_sha256", "evidence_receipt_sha256"])
def test_wrong_identity_rejected(row_fixture, field):
    row, _, _ = row_fixture
    row[field] = "d" * 64
    with pytest.raises(ValueError, match="differs|identity"):
        bridge._checked_row(row)


def test_audit_failure_cannot_grant_descriptor(row_fixture, monkeypatch):
    row, _, _ = row_fixture

    def fail(value):
        raise ValueError("actual branch counts differ")

    monkeypatch.setattr(bridge, "_artifact_audit", fail)
    with pytest.raises(ValueError, match="actual branch"):
        bridge._checked_row(row)


def test_mutation_during_audit_rejected(row_fixture, monkeypatch):
    from pathlib import Path

    row, _, _ = row_fixture

    def mutate(value):
        Path(row["sidecar_path"]).chmod(0o600)
        return {"clean_group_ids": ["group-a"]}

    monkeypatch.setattr(bridge, "_artifact_audit", mutate)
    with pytest.raises((ValueError, RuntimeError), match="read.only|changed"):
        bridge._checked_row(row)


def test_collection_replay_failure_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("sealed collection invalid")

    monkeypatch.setattr(bridge, "replay_v5_k1_balanced_full_search_collection", fail)
    with pytest.raises(ValueError, match="sealed collection invalid"):
        bridge.read_v5_k1_search_training_inputs("plan")


@pytest.fixture
def collection_fixture(monkeypatch):
    rows = [{"array_task_id": 0, "role": "train", "ids": ("train-a", "train-b")},
            {"array_task_id": 1, "role": "tuning_validation", "ids": ("tune-a",)}]
    checked = {"plan": {"parents": rows, "plan_sha256": "d" * 64}, "inventory": {"tasks": rows},
               "inventory_file_identity": {"sha256": "e" * 64},
               "completion_file_identity": {"sha256": "f" * 64}}
    calls = []

    def replay(path, **binding):
        calls.append((path, binding))
        return checked

    monkeypatch.setattr(bridge, "replay_v5_k1_balanced_full_search_collection", replay)
    monkeypatch.setattr(bridge, "_checked_row", lambda row: (SimpleNamespace(role=row["role"]), row["ids"]))
    monkeypatch.setattr(bridge, "_source_projection_mapping",
                        lambda selected, artifact: {
                            "array_task_id": selected["array_task_id"], "role": artifact.role,
                            "source_path": f"/source/{selected['array_task_id']}",
                            "projected_path": f"/projected/{selected['array_task_id']}",
                            "source_artifact_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                            "projected_artifact_sha256": "c" * 64, "projected_manifest_sha256": "d" * 64,
                            "clean_parent_count": len(selected["ids"]),
                            "ordered_recipe_branch_split_group_arrays_equal": True,
                        })
    return rows, checked, calls


def test_complete_preparation_preserves_binding_without_authorizing(collection_fixture):
    _, _, calls = collection_fixture
    result = bridge.read_v5_k1_search_training_inputs("frozen-plan", expected_plan_sha256="f" * 64)
    assert calls == [("frozen-plan", {"expected_plan_sha256": "f" * 64})] * 2
    assert result["parent_set_sha256"]["train"] == bridge.k1_parent_set_sha256(["train-a", "train-b"])
    assert result["collection_inventory_file_sha256"] == "e" * 64
    assert len(result["artifacts"]["tuning_validation"]) == 1
    assert result["training_inventory"] is None
    assert [r["array_task_id"] for r in result["projection_evidence"]["mappings"]] == [0, 1]
    for claim in ("writes_performed", "gradient_training_authorized",
                  "phase_c_disjointness_authorized", "scientific_acceptance_evidence"):
        assert result[claim] is False


@pytest.mark.parametrize("ids,message", [(("train-a", "train-a"), "unique"),
                                         ((), "non-empty")])
def test_duplicate_or_empty_population_rejected(collection_fixture, ids, message):
    rows, _, _ = collection_fixture
    rows[0]["ids"] = ids
    with pytest.raises(ValueError, match=message):
        bridge.read_v5_k1_search_training_inputs("frozen-plan")


def test_train_tuning_overlap_rejected(collection_fixture):
    rows, _, _ = collection_fixture
    rows[1]["ids"] = ("train-b",)
    with pytest.raises(ValueError, match="overlap"):
        bridge.read_v5_k1_search_training_inputs("frozen-plan")


def test_final_collection_replay_drift_rejected(collection_fixture, monkeypatch):
    from copy import deepcopy

    _, checked, _ = collection_fixture
    changed = deepcopy(checked)
    changed["inventory_file_identity"]["sha256"] = "f" * 64
    values = iter((checked, changed))
    monkeypatch.setattr(bridge, "replay_v5_k1_balanced_full_search_collection",
                        lambda *args, **kwargs: next(values))
    with pytest.raises(RuntimeError, match="collection changed"):
        bridge.read_v5_k1_search_training_inputs("frozen-plan")


def test_original_identity_rechecked_after_final_collection(collection_fixture, monkeypatch):
    _, checked, _ = collection_fixture
    authorization = SimpleNamespace(sha256="a" * 64)
    events = []

    def replay(*args, **kwargs):
        events.append("collection")
        return checked

    def validate(actual, collection, mappings, parent_hashes):
        assert actual is authorization
        assert collection is checked
        assert len(mappings) == 2
        assert set(parent_hashes) == {"train", "tuning_validation"}
        events.append("identity")
        if events.count("identity") == 2:
            raise ValueError("original identity changed after collection replay")

    monkeypatch.setattr(bridge, "replay_v5_k1_balanced_full_search_collection", replay)
    monkeypatch.setattr(bridge, "_validate_original_identity", validate)
    with pytest.raises(ValueError, match="original identity changed"):
        bridge.read_v5_k1_search_training_inputs(
            "frozen-plan", identity_authorization=authorization,
        )
    assert events == ["collection", "identity", "collection", "identity"]


@pytest.mark.parametrize("drift", [None, "mode", "file", "plan_hash", "replayed_identity", "mutation", "missing"])
def test_original_publication_replay_binds_four_actual_files(tmp_path, monkeypatch, drift):
    from dataclasses import replace
    from hashlib import sha256
    from test_posterior_v8_k1_training_chain_contract_v5 import _identity_authorization

    authorization = bridge.V5K1TrainingIdentityAuthorization.from_payload(
        _identity_authorization().to_payload(),
    )
    paths, hashes = {}, {}
    for name in ("balanced_dataset_completion", "train_tuning_receipt", "phase_c_completion", "three_way_receipt"):
        p = tmp_path / (name + ".json")
        p.write_bytes(name.encode())
        p.chmod(0o400)
        paths[name] = p
        hashes[name + "_file_sha256"] = sha256(p.read_bytes()).hexdigest()
    authorization = replace(authorization, **hashes)
    plan = {"identity_authorization": authorization.to_payload(),
            "identity_authorization_sha256": authorization.sha256}
    monkeypatch.setattr(bridge, "MAXWELL_DUST_ROOT", tmp_path)
    calls = []

    def issue(**arguments):
        calls.append(arguments)
        if drift == "mutation":
            paths["phase_c_completion"].chmod(0o600)
        return replace(authorization, source_archive_sha256="9" * 64) if drift == "replayed_identity" else authorization

    monkeypatch.setattr(bridge, "issue_v5_k1_training_identity_authorization_from_files", issue)
    if drift == "mode":
        paths["phase_c_completion"].chmod(0o600)
    if drift == "file":
        p = paths["phase_c_completion"]
        p.chmod(0o600)
        p.write_bytes(b"changed")
        p.chmod(0o400)
    if drift == "plan_hash":
        plan["identity_authorization_sha256"] = "9" * 64
    if drift == "missing":
        paths.pop("phase_c_completion")
    if drift is not None:
        with pytest.raises((ValueError, RuntimeError), match="identity|publication|must be read-only"):
            bridge._replay_original_publications(plan, paths)
        return
    actual, identities = bridge._replay_original_publications(plan, paths)
    assert actual.to_payload() == authorization.to_payload()
    assert set(identities) == set(paths)
    assert len(calls) == 1
    for name, path in paths.items():
        assert calls[0][name + "_path"] == path
        assert identities[name]["sha256"] == hashes[name + "_file_sha256"]


@pytest.mark.parametrize("drift", [False, True])
def test_preparation_replays_original_files_before_and_after_shards(collection_fixture, monkeypatch, drift):
    from test_posterior_v8_k1_training_chain_contract_v5 import _identity_authorization

    authorization = bridge.V5K1TrainingIdentityAuthorization.from_payload(_identity_authorization().to_payload())
    paths, calls = {"fixture": "paths"}, []
    _, checked, _ = collection_fixture
    checked["plan"]["source"] = {"archive_sha256": "d" * 64, "bundle_sha256": "e" * 64}
    builds = []

    def build(**arguments):
        assert len(calls) == 2
        builds.append(arguments)
        return {"fixture_inventory": True}

    monkeypatch.setattr(bridge, "build_v5_k1_training_inventory", build)
    before = {"phase_c_completion": {"inode": 1, "sha256": "a" * 64}}

    def replay(plan, actual_paths):
        assert actual_paths is paths
        calls.append(plan)
        after = {"phase_c_completion": {"inode": 2, "sha256": "a" * 64}}
        return authorization, after if drift and len(calls) == 2 else before

    monkeypatch.setattr(bridge, "_replay_original_publications", replay)
    monkeypatch.setattr(bridge, "_validate_original_identity", lambda *args: None)
    if drift:
        with pytest.raises(RuntimeError, match="changed across"):
            bridge.read_v5_k1_search_training_inputs("plan", identity_publication_paths=paths)
    else:
        result = bridge.read_v5_k1_search_training_inputs("plan", identity_publication_paths=paths)
        assert result["original_identity_publication_files"] == before
        assert result["original_identity_authorization_sha256"] == authorization.sha256
        assert result["gradient_training_authorized"] is False
        assert result["training_inventory"] == {"fixture_inventory": True}
        assert builds[0]["source_archive_sha256"] == "d" * 64
        assert builds[0]["source_bundle_sha256"] == "e" * 64
        assert builds[0]["identity_authorization"] is authorization
        assert builds[0]["projection_evidence"].to_payload() == result["projection_evidence"]
        assert builds[0]["train_tuning_disjointness_receipt_sha256"] == authorization.train_tuning_receipt_sha256
        assert builds[0]["k1_phase_c_disjointness_receipt_sha256"] == authorization.three_way_receipt_sha256
    assert len(builds) == (0 if drift else 1)
    assert len(calls) == 2


def test_original_publications_use_real_issuer_and_sealed_receipt_chain(tmp_path, monkeypatch):
    from test_posterior_v8_k1_training_identity_runtime_v5 import _write_publications

    names = ("balanced_dataset_completion", "train_tuning_receipt", "phase_c_completion", "three_way_receipt")
    paths = dict(zip(names, _write_publications(tmp_path), strict=True))
    authorization = bridge.issue_v5_k1_training_identity_authorization_from_files(
        source_archive_sha256="a" * 64, source_bundle_sha256="b" * 64,
        **{name + "_path": path for name, path in paths.items()},
    )
    plan = {"identity_authorization": authorization.to_payload(),
            "identity_authorization_sha256": authorization.sha256}
    monkeypatch.setattr(bridge, "MAXWELL_DUST_ROOT", tmp_path)
    actual, identities = bridge._replay_original_publications(plan, paths)
    assert actual.to_payload() == authorization.to_payload()
    assert len(identities) == 4
    duplicate = tmp_path / "duplicate-receipt.json"
    duplicate.write_bytes(paths["train_tuning_receipt"].read_bytes())
    duplicate.chmod(0o400)
    paths["train_tuning_receipt"] = duplicate
    with pytest.raises(ValueError, match="points at another train/tuning receipt"):
        bridge._replay_original_publications(plan, paths)


def test_actual_forced_recipe_counts_override_branch_pure_claim(row_fixture, tmp_path, monkeypatch):
    from pathlib import Path

    from test_posterior_v8_k1_training_dataset_audit_forced_v5 import _write_role_artifact
    from PosteriorV8.k1_balanced_dataset_plan_v5 import build_v5_k1_balanced_dataset_plan
    from PosteriorV8.k1_training_chain_dataset_audit_v5 import _artifact_audit, _checked_audit_arrays

    row, receipt, _ = row_fixture
    plan = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=101, tuning_master_scramble_seed=303,
        train_parents_per_branch=1, tuning_parents_per_branch=1,
    )
    actual, _ = _write_role_artifact(tmp_path, plan, "train")
    p = Path(actual["path"])
    p.chmod(0o400)
    manifest, _ = _checked_audit_arrays(p)
    row["projected_parent_path"] = str(p)
    row["projected_parent_artifact_sha256"] = sha256(p.read_bytes()).hexdigest()
    row["recipe_count"] = 12
    receipt.manifest["parent"] = {
        "artifact_sha256": row["projected_parent_artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    monkeypatch.setattr(bridge, "_artifact_audit", _artifact_audit)
    with pytest.raises(ValueError, match="branch counts disagree"):
        bridge._checked_row(row)


@pytest.mark.parametrize("difference", [None, "recipe", "file", "manifest"])
def test_source_projection_mapping_uses_actual_sealed_arrays(tmp_path, monkeypatch, difference):
    from pathlib import Path

    from test_posterior_v8_k1_training_dataset_audit_forced_v5 import _write_role_artifact
    from PosteriorV8.k1_balanced_dataset_plan_v5 import build_v5_k1_balanced_dataset_plan
    from PosteriorV8.k1_training_chain_dataset_audit_v5 import _checked_audit_arrays

    monkeypatch.setattr(bridge, "MAXWELL_DUST_ROOT", tmp_path)
    bindings = []
    for i, seed in enumerate((101, 102 if difference == "recipe" else 101)):
        directory = tmp_path / str(i)
        directory.mkdir()
        plan = build_v5_k1_balanced_dataset_plan(
            train_master_scramble_seed=seed, tuning_master_scramble_seed=303,
            train_parents_per_branch=1, tuning_parents_per_branch=1,
        )
        actual, _ = _write_role_artifact(directory, plan, "train")
        path = Path(actual["path"])
        path.chmod(0o400)
        manifest, _ = _checked_audit_arrays(path)
        bindings.append({"path": str(path), "artifact_sha256": sha256(path.read_bytes()).hexdigest(),
                         "manifest_sha256": manifest["manifest_sha256"]})
    selected = {"array_task_id": 7, "parent": bindings[0]}
    artifact = SimpleNamespace(**bindings[1], role="train", clean_parent_count=12)
    if difference == "file":
        selected["parent"]["artifact_sha256"] = "f" * 64
    elif difference == "manifest":
        selected["parent"]["manifest_sha256"] = "f" * 64
    if difference is not None:
        with pytest.raises(ValueError, match="differs"):
            bridge._source_projection_mapping(selected, artifact)
    else:
        result = bridge._source_projection_mapping(selected, artifact)
        assert result["source_path"] != result["projected_path"]
        assert result["ordered_recipe_branch_split_group_arrays_equal"] is True
        assert result["array_task_id"] == 7


@pytest.mark.parametrize("drift", [None, "plan", "artifact", "manifest", "count", "groups", "type"])
def test_original_authorization_must_match_source_population(drift):
    from test_posterior_v8_k1_training_chain_contract_v5 import _identity_authorization

    authorization = bridge.V5K1TrainingIdentityAuthorization.from_payload(
        _identity_authorization().to_payload()
    )
    checked = {"plan": {"identity_authorization_sha256": authorization.sha256}}
    mappings = []
    groups = {}
    for role in ("train", "tuning_validation"):
        population = authorization.population(role)
        mappings.append({"role": role, "source_artifact_sha256": population.artifact_sha256s[0],
                         "source_manifest_sha256": population.manifest_sha256s[0],
                         "clean_parent_count": population.clean_parent_count})
        groups[role] = population.clean_group_set_sha256
    if drift == "plan":
        checked["plan"]["identity_authorization_sha256"] = "f" * 64
    elif drift == "artifact":
        mappings[0]["source_artifact_sha256"] = "f" * 64
    elif drift == "manifest":
        mappings[0]["source_manifest_sha256"] = "f" * 64
    elif drift == "count":
        mappings[0]["clean_parent_count"] += 1
    elif drift == "groups":
        groups["train"] = "f" * 64
    elif drift == "type":
        authorization = authorization.to_payload()
    if drift is None:
        bridge._validate_original_identity(authorization, checked, mappings, groups)
    else:
        with pytest.raises((TypeError, ValueError), match="identity|population"):
            bridge._validate_original_identity(authorization, checked, mappings, groups)


@pytest.mark.parametrize("drift", [None, "claim", "integer_claim", "hash", "order", "extra", "duplicate", "false_equality"])
def test_projection_evidence_round_trip_is_strict(collection_fixture, drift):
    value = bridge.read_v5_k1_search_training_inputs("frozen-plan")["projection_evidence"]
    cls = bridge.V5K1SearchProjectionEvidence
    if drift is None:
        checked = cls.from_payload(value)
        assert checked.to_payload() == value
        value["mappings"][0]["clean_parent_count"] = 999
        assert checked.mappings[0]["clean_parent_count"] == 2
        with pytest.raises(TypeError):
            checked.mappings[0]["clean_parent_count"] = 999
        return
    if drift == "claim":
        value["gradient_training_authorized"] = True
    elif drift == "integer_claim":
        value["gradient_training_authorized"] = 0
    elif drift == "hash":
        value["evidence_sha256"] = "0" * 64
    elif drift == "order":
        value["mappings"].reverse()
    elif drift == "extra":
        value["unexpected_authority"] = True
    elif drift == "duplicate":
        value["mappings"][1]["source_path"] = value["mappings"][0]["source_path"]
    else:
        value["mappings"][0]["ordered_recipe_branch_split_group_arrays_equal"] = False
    with pytest.raises(ValueError):
        cls.from_payload(value)
