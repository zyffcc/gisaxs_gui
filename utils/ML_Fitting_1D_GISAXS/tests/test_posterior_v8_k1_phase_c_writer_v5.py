from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import stat

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    V5AmplitudeQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    V5BoundsQuery,
    full_range_axis_designs,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_contract_v5 import (
    K1_PHASE_C_METHOD_IDS,
    v5_k1_phase_c_contract_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_filesystem_replay_v5 import (
    V5K1PhaseCFilesystemReplayAdapter,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
    build_v5_k1_phase_c_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_production_writer_v5 import (
    write_v5_k1_phase_c_lossless_snapshot,
    write_v5_k1_phase_c_production_snapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_writer_capability_v5 import (
    _V5K1PhaseCWriterCapability,
    verify_v5_k1_phase_c_writer_receipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_writer_contract_v5 import (
    V5K1PhaseCParentWriterInput,
    V5K1PhaseCWriterSnapshot,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_contract_v5 import (
    V5TopologyQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_phase_c_filesystem_replay_v5 import (
    _write_snapshot,
)


def _topology_query() -> V5TopologyQuery:
    bounds = GuiComponentBounds(
        shape=SPHERE,
        R=ClosedInterval(5.0, 30.0),
        sigma_R=ClosedInterval(0.5, 3.0),
    )
    geometry = V5BoundsQuery.create(
        query_seed=19,
        generation_attempt=0,
        component_bounds=(bounds,),
        resolution_presence_policy="absent",
        resolution_bounds=None,
        axis_designs=full_range_axis_designs((bounds,), None),
    )
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0),
        k=ClosedInterval(0.1, 10.0),
        component_intensities=(ClosedInterval(0.01, 10.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    return V5TopologyQuery(geometry=geometry, amplitude=amplitude)


def _different_topology_query() -> V5TopologyQuery:
    first = GuiComponentBounds(
        shape=SPHERE,
        R=ClosedInterval(5.0, 30.0),
        sigma_R=ClosedInterval(0.5, 3.0),
    )
    second = GuiComponentBounds(
        shape=SPHERE,
        R=ClosedInterval(8.0, 40.0),
        sigma_R=ClosedInterval(0.8, 4.0),
    )
    geometry = V5BoundsQuery.create(
        query_seed=23,
        generation_attempt=0,
        component_bounds=(first, second),
        resolution_presence_policy="absent",
        resolution_bounds=None,
        axis_designs=full_range_axis_designs((first, second), None),
    )
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0),
        k=ClosedInterval(0.1, 10.0),
        component_intensities=(
            ClosedInterval(0.01, 10.0),
            ClosedInterval(0.01, 10.0),
        ),
        resolution_presence_policy="absent",
        int_res=None,
    )
    return V5TopologyQuery(geometry=geometry, amplitude=amplitude)


def _writer_snapshot(tmp_path: Path):
    plan = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    source_manifest, source_sha, _ = _write_snapshot(tmp_path / "source", plan=plan)
    source = V5K1PhaseCFilesystemReplayAdapter(
        source_manifest, expected_manifest_file_sha256=source_sha
    )
    bundle = source.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    snapshot = V5K1PhaseCWriterSnapshot(
        plan=plan,
        contract=v5_k1_phase_c_contract_payload(),
        bundle=bundle,
        parents=(
            V5K1PhaseCParentWriterInput(
                evidence=bundle.parents[0], topology_queries=(_topology_query(),)
            ),
        ),
    )
    return plan, snapshot


def _unlock(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
    ):
        path.chmod(0o700)
    root.chmod(0o700)


def _write_canonical(path: Path, payload: dict[str, object]) -> str:
    encoded = canonical_json(payload).encode("utf-8")
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(encoded)
    path.chmod(0o400)
    return sha256(encoded).hexdigest()


def _promote_for_capability_unit_test(result, plan):
    """Build a verifier fixture; production code still uses the frozen plan builder."""

    root = result.output_root
    pre_promotion = V5K1PhaseCFilesystemReplayAdapter(
        result.manifest_path,
        expected_manifest_file_sha256=result.manifest_file_sha256,
    ).load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    binding = pre_promotion.artifact_binding
    global_hashes = {
        "source_archive": binding.source_archive_sha256,
        "source_manifest": binding.source_manifest_sha256,
        "cross_platform_gate_claim": binding.cross_platform_gate_claim_sha256,
        "phase_a_launch_receipt": binding.phase_a_launch_receipt_sha256,
        "model_artifact": binding.model_artifact_sha256,
        "model_weights": binding.model_weights_sha256,
        "model_training_result": binding.model_training_result_sha256,
        "split_receipt": sha256(b"upstream-split-receipt").hexdigest(),
    }
    _unlock(root)
    manifest = json.loads(result.manifest_path.read_bytes())
    manifest["formal"] = True
    manifest_core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256(canonical_json(manifest_core).encode("utf-8")).hexdigest()
    manifest_file_sha = _write_canonical(result.manifest_path, manifest)
    parent_sha = manifest["parents"][0]["clean_parent_sha256"]
    parent_roles = (
        "search_sidecar",
        "search_evidence_receipt",
        "reference_bank",
        "reference_trace",
        *(f"method_trace:{method_id}" for method_id in K1_PHASE_C_METHOD_IDS),
        *sorted(
            {
                f"representative_payload:{emission.candidate_id}"
                for parent in pre_promotion.parents
                for method in parent.methods
                for emission in method.trace.candidate_emissions
            }
        ),
    )
    upstream = [
        {
            "scope": "global",
            "parent_sha256": None,
            "role": role,
            "file_sha256": file_sha,
        }
        for role, file_sha in global_hashes.items()
    ] + [
        {
            "scope": "parent",
            "parent_sha256": parent_sha,
            "role": role,
            "file_sha256": sha256(f"parent:{role}".encode()).hexdigest(),
        }
        for role in parent_roles
    ]
    upstream.sort(
        key=lambda value: (
            str(value["scope"]),
            str(value["parent_sha256"]),
            str(value["role"]),
            str(value["file_sha256"]),
        )
    )
    receipt = json.loads(result.receipt_path.read_bytes())
    receipt.update(
        production_eligible=True,
        formal=True,
        manifest_file_sha256=manifest_file_sha,
        manifest_sha256=manifest["manifest_sha256"],
        upstream_file_count=len(upstream),
        upstream_provenance=upstream,
        upstream_provenance_sha256=sha256(canonical_json(upstream).encode("utf-8")).hexdigest(),
    )
    receipt_core = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    receipt["receipt_sha256"] = sha256(canonical_json(receipt_core).encode("utf-8")).hexdigest()
    _write_canonical(result.receipt_path, receipt)
    for directory in sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    root.chmod(0o500)
    return replace(plan, formal=True, total_parent_count=1)


def _rewrite_manifest(path: Path, mutate) -> str:
    payload = json.loads(path.read_bytes())
    mutate(payload)
    core = {name: value for name, value in payload.items() if name != "manifest_sha256"}
    payload["manifest_sha256"] = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return _write_canonical(path, payload)


def _rewrite_bound_file(result, role: str, mutate) -> str:
    manifest = json.loads(result.manifest_path.read_bytes())
    row = next(value for value in manifest["files"] if value["role"] == role)
    path = result.output_root / row["relative_path"]
    payload = json.loads(path.read_bytes())
    mutate(payload)
    row_sha = _write_canonical(path, payload)

    def update(value):
        selected = next(item for item in value["files"] if item["file_id"] == row["file_id"])
        selected["sha256"] = row_sha

    return _rewrite_manifest(result.manifest_path, update)


def test_writer_round_trips_lossless_files_and_remains_nonclaiming_for_fixture(
    tmp_path,
):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        assert result.production_eligible is False
        files = [value for value in output.rglob("*") if value.is_file()]
        assert files
        assert all(
            stat.S_IMODE(value.stat().st_mode) == 0o400 and value.stat().st_nlink == 1
            for value in files
        )
        assert result.receipt_path.stat().st_mtime_ns >= max(
            value.stat().st_mtime_ns for value in files if value != result.receipt_path
        )
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path,
            expected_manifest_file_sha256=result.manifest_file_sha256,
        )
        replay = adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
        assert replay.parents[0].methods[0].trace.candidate_emissions[
            0
        ].payload.parameter.exact_intensity.tolist() == [1.0]
        with pytest.raises(ValueError, match="eligible frozen formal"):
            verify_v5_k1_phase_c_writer_receipt(result.receipt_path)
        with pytest.raises(ValueError, match="partial, fixture"):
            write_v5_k1_phase_c_production_snapshot(snapshot, tmp_path / "forbidden")
    finally:
        _unlock(output)


def test_writer_never_overwrites_an_existing_output_root(tmp_path):
    _, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        receipt_bytes = result.receipt_path.read_bytes()
        with pytest.raises(FileExistsError, match="refusing to reuse"):
            write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        assert result.receipt_path.read_bytes() == receipt_bytes
    finally:
        _unlock(output)


def test_writer_failure_never_publishes_completion_receipt(tmp_path):
    _, snapshot = _writer_snapshot(tmp_path)
    bad_parent = replace(snapshot.parents[0], topology_queries=(_different_topology_query(),))
    escaped = replace(snapshot, parents=(bad_parent,))
    output = tmp_path / "partial"
    try:
        with pytest.raises(ValueError, match="escaped the raw distance context"):
            write_v5_k1_phase_c_lossless_snapshot(escaped, output)
        assert (output / "manifest.json").exists()
        assert not (output / "writer-completion.json").exists()
    finally:
        _unlock(output)


def test_consumer_rejects_hardlinked_writer_artifact(tmp_path):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    external_link = tmp_path / "external-hardlink.json"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        raw = next(
            value
            for value in output.rglob("*.json")
            if value.name not in {"manifest.json", "writer-completion.json"}
        )
        os.link(raw, external_link)
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path,
            expected_manifest_file_sha256=result.manifest_file_sha256,
        )
        with pytest.raises(ValueError, match="uniquely linked"):
            adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    finally:
        external_link.unlink(missing_ok=True)
        _unlock(output)


def test_consumer_detects_hardlink_added_and_removed_between_replays(tmp_path):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    external_link = tmp_path / "transient-hardlink.json"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        raw = next(
            value
            for value in output.rglob("*.json")
            if value.name not in {"manifest.json", "writer-completion.json"}
        )
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path,
            expected_manifest_file_sha256=result.manifest_file_sha256,
        )
        bundle = adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
        os.link(raw, external_link)
        external_link.unlink()
        with pytest.raises(RuntimeError, match="files or identities changed"):
            adapter.revalidate_bundle(
                bundle=bundle,
                plan=plan,
                contract=v5_k1_phase_c_contract_payload(),
            )
    finally:
        external_link.unlink(missing_ok=True)
        _unlock(output)


def test_consumer_rejects_writable_and_same_byte_replaced_writer_files(tmp_path):
    plan, snapshot = _writer_snapshot(tmp_path)
    for attack in ("writable", "replacement"):
        output = tmp_path / attack
        try:
            result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
            raw = next(
                value
                for value in output.rglob("*.json")
                if value.name not in {"manifest.json", "writer-completion.json"}
            )
            adapter = V5K1PhaseCFilesystemReplayAdapter(
                result.manifest_path,
                expected_manifest_file_sha256=result.manifest_file_sha256,
            )
            bundle = adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
            if attack == "writable":
                raw.chmod(0o600)
                with pytest.raises(ValueError, match="read-only"):
                    adapter.revalidate_bundle(
                        bundle=bundle,
                        plan=plan,
                        contract=v5_k1_phase_c_contract_payload(),
                    )
            else:
                raw.parent.chmod(0o700)
                replacement = raw.with_suffix(".replacement")
                replacement.write_bytes(raw.read_bytes())
                replacement.chmod(0o400)
                os.replace(replacement, raw)
                raw.parent.chmod(0o500)
                with pytest.raises(RuntimeError, match="files or identities changed"):
                    adapter.revalidate_bundle(
                        bundle=bundle,
                        plan=plan,
                        contract=v5_k1_phase_c_contract_payload(),
                    )
        finally:
            _unlock(output)


def test_writer_capability_rejects_direct_init_subclass_pickle_replay_and_reuse(
    tmp_path, monkeypatch
):
    with pytest.raises(TypeError, match="no public constructor"):
        _V5K1PhaseCWriterCapability()
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _ForbiddenCapabilitySubclass(_V5K1PhaseCWriterCapability):
            pass

    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        fake_formal = _promote_for_capability_unit_test(result, plan)
        monkeypatch.setattr(
            "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
            "k1_phase_c_writer_capability_v5.build_v5_k1_phase_c_plan",
            lambda *, formal: fake_formal,
        )
        with pytest.raises(TypeError, match="exact live writer capability"):
            V5K1PhaseCFilesystemReplayAdapter(
                result.manifest_path,
                expected_manifest_file_sha256=sha256(result.manifest_path.read_bytes()).hexdigest(),
                writer_capability=json.loads(result.receipt_path.read_bytes()),
            )
        capability = verify_v5_k1_phase_c_writer_receipt(result.receipt_path)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(capability)
        with pytest.raises(RuntimeError, match="already minted"):
            verify_v5_k1_phase_c_writer_receipt(result.receipt_path)
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path,
            expected_manifest_file_sha256=sha256(result.manifest_path.read_bytes()).hexdigest(),
            writer_capability=capability,
        )
        with pytest.raises(RuntimeError, match="invalid, reused"):
            V5K1PhaseCFilesystemReplayAdapter(
                result.manifest_path,
                expected_manifest_file_sha256=sha256(result.manifest_path.read_bytes()).hexdigest(),
                writer_capability=capability,
            )
        replay = adapter.load_bundle(plan=fake_formal, contract=v5_k1_phase_c_contract_payload())
        assert len(replay.parents) == 1
    finally:
        _unlock(output)


def test_writer_receipt_verifier_rejects_an_unlisted_physical_file(tmp_path, monkeypatch):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        fake_formal = _promote_for_capability_unit_test(result, plan)
        monkeypatch.setattr(
            "utils.ML_Fitting_1D_GISAXS.PosteriorV8."
            "k1_phase_c_writer_capability_v5.build_v5_k1_phase_c_plan",
            lambda *, formal: fake_formal,
        )
        _unlock(output)
        extra = output / "unlisted.json"
        _write_canonical(extra, {"unlisted": True})
        for directory in sorted(
            (value for value in output.rglob("*") if value.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        output.chmod(0o500)
        with pytest.raises(ValueError, match="missing or extra physical files"):
            verify_v5_k1_phase_c_writer_receipt(result.receipt_path)
    finally:
        _unlock(output)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("duplicate_id", "file IDs must be unique"),
        ("duplicate_path", "paths must be unique"),
        ("missing_path", "path is missing"),
        ("extra_file", "unreferenced files"),
        ("symlink", "must not contain symbolic links"),
    ),
)
def test_consumer_rejects_manifest_identity_and_path_attacks(tmp_path, attack, message):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        _unlock(output)
        manifest = json.loads(result.manifest_path.read_bytes())
        first = manifest["files"][0]
        if attack == "duplicate_id":
            manifest_sha = _rewrite_manifest(
                result.manifest_path,
                lambda value: value["files"].append(
                    {
                        **value["files"][0],
                        "relative_path": "duplicate-id.json",
                    }
                ),
            )
        elif attack == "duplicate_path":
            manifest_sha = _rewrite_manifest(
                result.manifest_path,
                lambda value: value["files"].append(
                    {**value["files"][0], "file_id": "duplicate-path"}
                ),
            )
        elif attack == "missing_path":
            manifest_sha = _rewrite_manifest(
                result.manifest_path,
                lambda value: value["files"][0].update(relative_path="missing.json"),
            )
        elif attack == "extra_file":
            extra = output / "extra.json"
            extra_sha = _write_canonical(extra, {"unexpected": True})
            manifest_sha = _rewrite_manifest(
                result.manifest_path,
                lambda value: value["files"].append(
                    {
                        "file_id": "extra",
                        "role": "evaluator_config",
                        "relative_path": "extra.json",
                        "sha256": extra_sha,
                    }
                ),
            )
        else:
            raw = output / first["relative_path"]
            external = tmp_path / "symlink-target.json"
            external.write_bytes(raw.read_bytes())
            raw.unlink()
            raw.symlink_to(external)
            manifest_sha = sha256(result.manifest_path.read_bytes()).hexdigest()
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path, expected_manifest_file_sha256=manifest_sha
        )
        with pytest.raises(ValueError, match=message):
            adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    finally:
        _unlock(output)


@pytest.mark.parametrize(
    ("role", "mutate", "message"),
    (
        (
            "method_exact_call_trace",
            lambda payload: payload["exact_forward_calls"].pop(),
            "complete, ordered, and contiguous",
        ),
        (
            "representative_payload",
            lambda payload: payload["parameter"].get("exact_intensity", []).__setitem__(0, 2.0),
            "exact-intensity SHA-256",
        ),
    ),
)
def test_consumer_rejects_exact_trace_gaps_and_payload_mismatch(tmp_path, role, mutate, message):
    plan, snapshot = _writer_snapshot(tmp_path)
    output = tmp_path / "written"
    try:
        result = write_v5_k1_phase_c_lossless_snapshot(snapshot, output)
        _unlock(output)
        if role == "representative_payload":
            manifest = json.loads(result.manifest_path.read_bytes())
            rows = [value for value in manifest["files"] if value["role"] == role]
            target = next(
                value
                for value in rows
                if "exact_intensity"
                in json.loads((output / value["relative_path"]).read_bytes())["parameter"]
            )
            payload_path = output / target["relative_path"]
            payload = json.loads(payload_path.read_bytes())
            mutate(payload)
            file_sha = _write_canonical(payload_path, payload)

            def update(value):
                selected = next(
                    item for item in value["files"] if item["file_id"] == target["file_id"]
                )
                selected["sha256"] = file_sha

            manifest_sha = _rewrite_manifest(result.manifest_path, update)
        else:
            manifest_sha = _rewrite_bound_file(result, role, mutate)
        adapter = V5K1PhaseCFilesystemReplayAdapter(
            result.manifest_path, expected_manifest_file_sha256=manifest_sha
        )
        with pytest.raises(ValueError, match=message):
            adapter.load_bundle(plan=plan, contract=v5_k1_phase_c_contract_payload())
    finally:
        _unlock(output)
