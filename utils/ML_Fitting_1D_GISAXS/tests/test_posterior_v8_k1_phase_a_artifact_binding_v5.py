from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_grouped_dataset_v5 import (
    build_tiny_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_dataset_v5 import (
    write_v5_grouped_dataset,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_cross_platform_v5 import (
    gate_claim_sha256,
    validate_v5_k1_cross_platform_marker,
    validate_v5_k1_cross_platform_marker_file,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_dataset_binding_v5 import (
    publish_v5_k1_phase_a_dataset_binding,
    validate_v5_k1_phase_a_dataset_binding_file,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_capability_v7 import (
    _mint_phase_a_capability,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_contract_v7 import (
    PHASE_A_LAUNCH_BINDING_SCHEMA,
    portable_identity,
    self_hashed,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_staging_files_v5 import (
    read_only_identity,
)


_SOURCE = {
    "source_archive_sha256": "1" * 64,
    "source_manifest_sha256": "2" * 64,
    "source_tree_sha256": "3" * 64,
}
_REFERENCE_SHA256 = "4" * 64
_REFERENCE_BYTE_COUNT = 123456
_MANIFEST_SHA256 = "5" * 64
_SCIENTIFIC_SHA256 = "6" * 64
_COMPARISON_SHA256 = "7" * 64


def _marker_expected() -> dict[str, object]:
    values = {
        "expected_source": dict(_SOURCE),
        "reference_file_sha256": _REFERENCE_SHA256,
        "reference_file_byte_count": _REFERENCE_BYTE_COUNT,
        "reference_manifest_sha256": _MANIFEST_SHA256,
        "scientific_content_sha256": _SCIENTIFIC_SHA256,
        "comparison_result_sha256": _COMPARISON_SHA256,
    }
    return {
        **values,
        "expected_gate_claim_sha256": gate_claim_sha256(
            source=values["expected_source"],
            reference_file_sha256=values["reference_file_sha256"],
            reference_file_byte_count=values["reference_file_byte_count"],
            reference_manifest_sha256=values["reference_manifest_sha256"],
            scientific_content_sha256=values["scientific_content_sha256"],
            comparison_result_sha256=values["comparison_result_sha256"],
        ),
    }


def _marker_text() -> str:
    core = {
        "schema": "gisaxs.posterior_v8.sobol_cross_platform_pass_marker/v1",
        "version": "posterior_v8_atomic_exclusive_pass_after_exact_compare_v1",
        "status": "PASS",
        "comparison_result_sha256": _COMPARISON_SHA256,
        "reference_manifest_sha256": _MANIFEST_SHA256,
        "candidate_manifest_sha256": _MANIFEST_SHA256,
        "scientific_content_sha256": _SCIENTIFIC_SHA256,
        "source": dict(_SOURCE),
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    payload = {**core, "marker_sha256": sha256(canonical.encode()).hexdigest()}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_only_file(path: Path, encoded: str) -> Path:
    path.write_text(encoded, encoding="utf-8")
    path.chmod(0o400)
    return path


def _checked_dataset(path: Path, *, base_seed: int = 20260903) -> Path:
    write_v5_grouped_dataset(
        build_tiny_v5_grouped_dataset(base_seed=base_seed),
        path,
    )
    path.chmod(0o400)
    return path


def _phase_a_authority(root: Path, *, stage: str, job_id: str):
    staged = _read_only_file(
        root / f".{stage}-{job_id}-staged-input.json",
        json.dumps({"stage": stage, "job_id": job_id}, sort_keys=True),
    )
    launch = self_hashed(
        {
            "schema": PHASE_A_LAUNCH_BINDING_SCHEMA,
            "status": "VALIDATED",
            "stage": stage,
            "slurm_job_id": job_id,
            "plan_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "release_sha256": "c" * 64,
            "transaction_files": {},
            "upstream": None,
        },
        "binding_sha256",
    )
    identity = portable_identity(read_only_identity(staged, "test staged input"))
    capability = _mint_phase_a_capability(
        launch,
        [{"role": "test_staged_input", "path": str(staged), "identity": identity}],
    )
    return launch, capability


def test_marker_binding_requires_exact_source_reference_and_launch_claim(tmp_path):
    expected = _marker_expected()
    parsed = validate_v5_k1_cross_platform_marker(_marker_text(), **expected)

    assert parsed["status"] == "PASS"
    assert parsed["source"] == _SOURCE
    assert parsed["gate_claim_sha256"] == expected["expected_gate_claim_sha256"]

    wrong_source = dict(expected)
    wrong_source["expected_source"] = {**_SOURCE, "source_tree_sha256": "8" * 64}
    with pytest.raises(ValueError, match="source identity"):
        validate_v5_k1_cross_platform_marker(_marker_text(), **wrong_source)

    with pytest.raises(ValueError, match="gate claim"):
        validate_v5_k1_cross_platform_marker(
            _marker_text(),
            **{**expected, "expected_gate_claim_sha256": "9" * 64},
        )

    payload = json.loads(_marker_text())
    payload["untrusted_extension"] = True
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        validate_v5_k1_cross_platform_marker(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", **expected
        )


def test_marker_file_rejects_writable_and_symlink_paths(tmp_path):
    marker = tmp_path / "PASS.json"
    marker.write_text(_marker_text(), encoding="utf-8")
    expected = _marker_expected()

    with pytest.raises(ValueError, match="read-only"):
        validate_v5_k1_cross_platform_marker_file(marker, **expected)
    marker.chmod(0o400)

    linked = tmp_path / "linked-PASS.json"
    linked.symlink_to(marker)
    with pytest.raises(ValueError, match="symlink"):
        validate_v5_k1_cross_platform_marker_file(linked, **expected)


def test_completion_last_binding_accepts_only_exact_local_dataset_and_marker(tmp_path):
    marker = _read_only_file(tmp_path / "PASS.json", _marker_text())
    original = _checked_dataset(tmp_path / "original.gvd5")
    binding = tmp_path / "original.gvd5.binding-v1.json"
    expected = _marker_expected()
    launch, capability = _phase_a_authority(
        tmp_path, stage="smoke_dataset", job_id="24390001"
    )

    published = publish_v5_k1_phase_a_dataset_binding(
        original,
        marker,
        binding,
        original_dataset_path=str(original),
        marker_expected=expected,
        launch_binding=launch,
        capability=capability,
    )

    assert binding.stat().st_mode & 0o222 == 0
    assert published["status"] == "COMPLETE"
    assert published["dataset"]["original_path"] == str(original)
    assert published["dataset"]["byte_count"] == original.stat().st_size
    assert published["source"] == _SOURCE
    assert published["cross_platform_gate"]["gate_claim_sha256"] == expected[
        "expected_gate_claim_sha256"
    ]

    local_dataset = tmp_path / "worker-input.gvd5"
    shutil.copyfile(original, local_dataset)
    local_dataset.chmod(0o400)
    local_binding = tmp_path / "worker-input.binding.json"
    shutil.copyfile(binding, local_binding)
    local_binding.chmod(0o400)
    assert validate_v5_k1_phase_a_dataset_binding_file(
        local_binding,
        dataset_path=local_dataset,
        marker_path=marker,
        expected_original_dataset_path=str(original),
        marker_expected=expected,
        expected_launch_binding=launch,
    ) == published

    overwrite_launch, overwrite_capability = _phase_a_authority(
        tmp_path, stage="smoke_dataset", job_id="24390002"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_v5_k1_phase_a_dataset_binding(
            original,
            marker,
            binding,
            original_dataset_path=str(original),
            marker_expected=expected,
            launch_binding=overwrite_launch,
            capability=overwrite_capability,
        )


def test_binding_rejects_old_data_swaps_tamper_and_hidden_fields(tmp_path):
    marker = _read_only_file(tmp_path / "PASS.json", _marker_text())
    dataset = _checked_dataset(tmp_path / "dataset.gvd5")
    binding = tmp_path / "dataset.binding.json"
    expected = _marker_expected()
    launch, capability = _phase_a_authority(
        tmp_path, stage="full_dataset", job_id="24390003"
    )
    publish_v5_k1_phase_a_dataset_binding(
        dataset,
        marker,
        binding,
        original_dataset_path=str(dataset),
        marker_expected=expected,
        launch_binding=launch,
        capability=capability,
    )

    old_dataset = _checked_dataset(tmp_path / "old-unbound.gvd5", base_seed=7)
    with pytest.raises(FileNotFoundError):
        validate_v5_k1_phase_a_dataset_binding_file(
            tmp_path / "old-unbound.gvd5.binding.json",
            dataset_path=old_dataset,
            marker_path=marker,
            expected_original_dataset_path=str(old_dataset),
            marker_expected=expected,
            expected_launch_binding=launch,
        )

    with pytest.raises(ValueError, match="consumed dataset"):
        validate_v5_k1_phase_a_dataset_binding_file(
            binding,
            dataset_path=old_dataset,
            marker_path=marker,
            expected_original_dataset_path=str(dataset),
            marker_expected=expected,
            expected_launch_binding=launch,
        )

    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["dataset"]["hidden_legacy_field"] = "accepted-by-old-reader"
    tampered = _read_only_file(
        tmp_path / "tampered.binding.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        validate_v5_k1_phase_a_dataset_binding_file(
            tampered,
            dataset_path=dataset,
            marker_path=marker,
            expected_original_dataset_path=str(dataset),
            marker_expected=expected,
            expected_launch_binding=launch,
        )

    wrong_claim = {**expected, "expected_gate_claim_sha256": "a" * 64}
    with pytest.raises(ValueError, match="gate claim"):
        validate_v5_k1_phase_a_dataset_binding_file(
            binding,
            dataset_path=dataset,
            marker_path=marker,
            expected_original_dataset_path=str(dataset),
            marker_expected=wrong_claim,
            expected_launch_binding=launch,
        )
