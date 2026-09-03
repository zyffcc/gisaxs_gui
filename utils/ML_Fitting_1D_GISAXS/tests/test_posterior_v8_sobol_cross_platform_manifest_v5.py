from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    sobol_cross_platform_manifest_v5 as manifest_module,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_sobol_cross_platform_gate_v5 import (
    main as gate_main,
    validate_v5_sobol_cross_platform_pass_marker,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_cross_platform_manifest_v5 import (
    V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES,
    V5SobolCrossPlatformComparison,
    V5SobolCrossPlatformManifest,
    build_v5_sobol_cross_platform_manifest,
    compare_v5_sobol_cross_platform_manifests,
)


_ARCHIVE_SHA = "a" * 64
_SOURCE_MANIFEST_SHA = "b" * 64
_TREE_SHA = "c" * 64


@pytest.fixture(scope="module")
def frozen_manifest() -> V5SobolCrossPlatformManifest:
    return build_v5_sobol_cross_platform_manifest(
        source_archive_sha256=_ARCHIVE_SHA,
        source_manifest_sha256=_SOURCE_MANIFEST_SHA,
        source_tree_sha256=_TREE_SHA,
    )


def _manifest_from_core(core: dict[str, object]) -> V5SobolCrossPlatformManifest:
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return V5SobolCrossPlatformManifest(
        canonical_json=canonical,
        sha256=sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _refresh_layers(core: dict[str, object]) -> None:
    core["layer_sha256"] = manifest_module._layer_hashes(  # noqa: SLF001
        design=core["design"],
        scope=core["scope"],
        generating_entries=core["generating_entries"],
        probe_entries=core["topology_probe_entries"],
    )


def test_full_frozen_scope_is_layered_runtime_free_and_strictly_round_trips(
    frozen_manifest,
):
    payload = frozen_manifest.payload
    assert payload["scope"] == {
        "prefix_start": 0,
        "prefix_stop": 1024,
        "prefix_count": 1024,
        "probe_indices": [0, 1, 17, 118, 233, 511, 589, 700, 753, 1023],
        "selected_topology_ids": list(range(34)),
        "probe_topology_query_count": 340,
        "assigned_split": "train",
        "ood_label": None,
        "clean_group_policy": (
            "sha256(runtime_free_semantic_design_sha256,NUL,decimal_sobol_index);"
            "neutral_train_attestation_parent_not_a_dataset_split_assignment_v1"
        ),
    }
    assert len(payload["generating_entries"]) == 1024
    assert all(
        {
            "geometry_bounds_embedding_sha256",
            "amplitude_embedding_ref1_sha256",
            "amplitude_embedding_ref1e4_sha256",
        }
        <= set(entry)
        for entry in payload["generating_entries"]
    )
    assert [value["sobol_index"] for value in payload["generating_entries"]] == list(
        range(1024)
    )
    assert [value["sobol_index"] for value in payload["topology_probe_entries"]] == list(
        V5_SOBOL_CROSS_PLATFORM_PROBE_INDICES
    )
    assert all(
        [query["topology_id"] for query in probe["topology_queries"]]
        == list(range(34))
        for probe in payload["topology_probe_entries"]
    )
    assert all(
        "amplitude_embedding_ref1e4_sha256" in query
        for probe in payload["topology_probe_entries"]
        for query in probe["topology_queries"]
    )
    assert set(payload["layer_sha256"]) == manifest_module._LAYER_FIELDS  # noqa: SLF001
    semantic = payload["design"]["semantic_payload"]
    assert "scipy_version" not in semantic
    assert "numpy_version" not in semantic
    assert "python_version" not in semantic
    assert V5SobolCrossPlatformManifest.from_json(frozen_manifest.to_json()) == frozen_manifest


def test_builder_fails_if_prefix_and_formal_gray_code_random_access_disagree(monkeypatch):
    original = manifest_module._random_access_sobol_points  # noqa: SLF001

    def one_bit_wrong(design, indices):
        values = original(design, indices)
        values[0, 0] = np.nextafter(values[0, 0], np.inf)
        return values

    monkeypatch.setattr(
        manifest_module,
        "_random_access_sobol_points",
        one_bit_wrong,
    )
    with pytest.raises(RuntimeError, match="random access"):
        build_v5_sobol_cross_platform_manifest(
            source_archive_sha256=_ARCHIVE_SHA,
            source_manifest_sha256=_SOURCE_MANIFEST_SHA,
            source_tree_sha256=_TREE_SHA,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("source_archive_sha256", "d" * 64),
        ("source_manifest_sha256", "e" * 64),
        ("source_tree_sha256", "f" * 64),
    ],
)
def test_comparison_fails_closed_on_each_source_identity_mismatch(
    frozen_manifest,
    field,
    replacement,
):
    core = frozen_manifest.payload
    core["source"][field] = replacement
    candidate = _manifest_from_core(core)

    result = compare_v5_sobol_cross_platform_manifests(frozen_manifest, candidate)

    assert not result.passed
    assert result.first_mismatch_path == f"$.source.{field}"
    assert (
        result.reference_scientific_content_sha256
        == result.candidate_scientific_content_sha256
    )


def test_one_bit_scientific_tamper_is_rejected_without_rehash_and_compared_if_rehashed(
    frozen_manifest,
):
    original = frozen_manifest.payload
    tampered = deepcopy(original)
    digest = tampered["generating_entries"][18]["geometry_query_sha256"]
    replacement = ("0" if digest[0] != "0" else "1") + digest[1:]
    tampered["generating_entries"][18]["geometry_query_sha256"] = replacement

    with pytest.raises(ValueError, match="layer SHA-256"):
        _manifest_from_core(tampered)

    _refresh_layers(tampered)
    candidate = _manifest_from_core(tampered)
    result = compare_v5_sobol_cross_platform_manifests(frozen_manifest, candidate)
    assert not result.passed
    assert result.first_mismatch_path == (
        "$.generating_entries[18].geometry_query_sha256"
    )
    assert (
        result.reference_scientific_content_sha256
        != result.candidate_scientific_content_sha256
    )


def test_parser_rejects_duplicate_unknown_reordered_and_noncanonical_json(frozen_manifest):
    encoded = frozen_manifest.to_json()
    duplicate = encoded.replace(
        '  "schema":',
        '  "schema": "duplicate",\n  "schema":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        V5SobolCrossPlatformManifest.from_json(duplicate)

    unknown = frozen_manifest.payload
    unknown["unexpected"] = True
    unknown["manifest_sha256"] = frozen_manifest.sha256
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        V5SobolCrossPlatformManifest.from_json(
            json.dumps(unknown, indent=2, sort_keys=True) + "\n"
        )

    reordered = frozen_manifest.payload
    reordered["generating_entries"][0], reordered["generating_entries"][1] = (
        reordered["generating_entries"][1],
        reordered["generating_entries"][0],
    )
    _refresh_layers(reordered)
    with pytest.raises(ValueError, match="ordered Sobol indices"):
        _manifest_from_core(reordered)

    with pytest.raises(ValueError, match="not canonical"):
        V5SobolCrossPlatformManifest.from_json(encoded.rstrip() + "  \n")


def test_runner_writes_atomic_pass_marker_only_after_exact_match_and_refuses_existing(
    frozen_manifest,
    tmp_path: Path,
):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    result = tmp_path / "result.json"
    marker = tmp_path / "PASS.json"
    reference.write_text(frozen_manifest.to_json(), encoding="utf-8")
    candidate.write_text(frozen_manifest.to_json(), encoding="utf-8")

    assert gate_main(
        [
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--result-output",
            str(result),
            "--pass-marker",
            str(marker),
        ]
    ) == 0
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    assert V5SobolCrossPlatformComparison.from_json(result.read_text(encoding="utf-8"))
    marker_payload = validate_v5_sobol_cross_platform_pass_marker(
        marker.read_text(encoding="utf-8")
    )
    assert result_payload["passed"] is True
    assert marker_payload["status"] == "PASS"
    assert marker_payload["comparison_result_sha256"] == result_payload["result_sha256"]
    assert marker_payload["source"] == frozen_manifest.payload["source"]

    with pytest.raises(FileExistsError, match="already exists"):
        gate_main(
            [
                "compare",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--result-output",
                str(result),
                "--pass-marker",
                str(marker),
            ]
        )


def test_runner_records_failed_compare_without_leaving_pass_marker(
    frozen_manifest,
    tmp_path: Path,
):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    result = tmp_path / "result.json"
    marker = tmp_path / "PASS.json"
    reference.write_text(frozen_manifest.to_json(), encoding="utf-8")
    candidate_core = frozen_manifest.payload
    candidate_core["source"]["source_tree_sha256"] = "d" * 64
    candidate.write_text(_manifest_from_core(candidate_core).to_json(), encoding="utf-8")

    assert gate_main(
        [
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--result-output",
            str(result),
            "--pass-marker",
            str(marker),
        ]
    ) == 1
    assert json.loads(result.read_text(encoding="utf-8"))["passed"] is False
    assert not marker.exists()


def test_runner_rejects_existing_build_output_before_materialization(tmp_path: Path):
    output = tmp_path / "manifest.json"
    output.write_text("user-owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        gate_main(
            [
                "build",
                "--source-archive-sha256",
                _ARCHIVE_SHA,
                "--source-manifest-sha256",
                _SOURCE_MANIFEST_SHA,
                "--source-tree-sha256",
                _TREE_SHA,
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "user-owned"
