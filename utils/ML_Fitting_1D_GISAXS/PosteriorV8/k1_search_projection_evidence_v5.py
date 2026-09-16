"""Strict versioned evidence for source-to-search-parent identity preservation."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .k1_training_identity_contract_v5 import V5K1TrainingIdentityAuthorization


_SCHEMA = "gisaxs.posterior_v8.k1_search_projection_evidence/v1"
_ROW_FIELDS = frozenset((
    "array_task_id", "role", "source_path", "source_artifact_sha256",
    "source_manifest_sha256", "projected_path", "projected_artifact_sha256",
    "projected_manifest_sha256", "clean_parent_count",
    "ordered_recipe_branch_split_group_arrays_equal",
))


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("projection evidence requires lowercase SHA-256 identities")
    return value


@dataclass(frozen=True, kw_only=True)
class V5K1SearchProjectionEvidence:
    plan_sha256: str
    inventory_file_sha256: str
    completion_file_sha256: str
    original_identity_authorization_sha256: str | None
    mappings: tuple[Mapping[str, object], ...]

    def __post_init__(self):
        for name in ("plan_sha256", "inventory_file_sha256", "completion_file_sha256"):
            _digest(getattr(self, name))
        if self.original_identity_authorization_sha256 is not None:
            _digest(self.original_identity_authorization_sha256)
        rows = []
        paths = set()
        for index, value in enumerate(self.mappings):
            if not isinstance(value, Mapping) or set(value) != _ROW_FIELDS:
                raise ValueError("projection evidence mapping fields differ")
            row = dict(value)
            if type(row["array_task_id"]) is not int or row["array_task_id"] != index:
                raise ValueError("projection evidence task order differs")
            if row["role"] not in ("train", "tuning_validation"):
                raise ValueError("projection evidence role differs")
            if type(row["clean_parent_count"]) is not int or row["clean_parent_count"] <= 0:
                raise ValueError("projection evidence parent count is invalid")
            if row["ordered_recipe_branch_split_group_arrays_equal"] is not True:
                raise ValueError("projection evidence must preserve actual population arrays")
            for name in ("source_artifact_sha256", "source_manifest_sha256",
                         "projected_artifact_sha256", "projected_manifest_sha256"):
                _digest(row[name])
            for name in ("source_path", "projected_path"):
                raw = row[name]
                if (not isinstance(raw, str) or "\0" in raw
                        or not PurePosixPath(raw).is_absolute() or ".." in PurePosixPath(raw).parts
                        or str(PurePosixPath(raw)) != raw):
                    raise ValueError("projection evidence path is not absolute and contained")
                if raw in paths:
                    raise ValueError("projection evidence paths must be unique")
                paths.add(raw)
            rows.append(MappingProxyType(row))
        if not rows or {row["role"] for row in rows} != {"train", "tuning_validation"}:
            raise ValueError("projection evidence must contain both populations")
        object.__setattr__(self, "mappings", tuple(rows))

    def to_payload(self):
        core = {
            "schema": _SCHEMA, "plan_sha256": self.plan_sha256,
            "inventory_file_sha256": self.inventory_file_sha256,
            "completion_file_sha256": self.completion_file_sha256,
            "original_identity_authorization_sha256": self.original_identity_authorization_sha256,
            "mappings": [dict(row) for row in self.mappings],
            "gradient_training_authorized": False, "phase_c_disjointness_authorized": False,
            "scientific_acceptance_evidence": False,
        }
        return {**core, "evidence_sha256": sha256(canonical_json(core).encode()).hexdigest()}

    def validate_population_binding(self, authorization, artifacts, parent_hashes):
        """Check inventory linkage; serialized evidence is not a filesystem replay."""
        if not isinstance(authorization, V5K1TrainingIdentityAuthorization):
            raise TypeError("projection requires the original typed identity authorization")
        if self.original_identity_authorization_sha256 != authorization.sha256:
            raise ValueError("projection original identity authorization differs")
        for role in ("train", "tuning_validation"):
            rows = tuple(row for row in self.mappings if row["role"] == role)
            population = authorization.population(role)
            actual = tuple(artifacts[role])
            if (
                tuple(sorted(row["source_artifact_sha256"] for row in rows))
                != population.artifact_sha256s
                or tuple(sorted(row["source_manifest_sha256"] for row in rows))
                != population.manifest_sha256s
                or sum(row["clean_parent_count"] for row in rows) != population.clean_parent_count
                or parent_hashes[role] != population.clean_group_set_sha256
            ):
                raise ValueError(f"projection original {role} population differs")
            expected = sorted((row["projected_path"], row["projected_artifact_sha256"],
                               row["projected_manifest_sha256"], row["clean_parent_count"])
                              for row in rows)
            observed = sorted((item.path, item.artifact_sha256, item.manifest_sha256,
                               item.clean_parent_count) for item in actual)
            if observed != expected:
                raise ValueError(f"projection training {role} artifact binding differs")

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, Mapping):
            raise TypeError("projection evidence must be an object")
        for claim in ("gradient_training_authorized", "phase_c_disjointness_authorized",
                      "scientific_acceptance_evidence"):
            if payload.get(claim) is not False:
                raise ValueError("projection evidence cannot grant authority")
        value = cls(**{name: payload[name] for name in (
            "plan_sha256", "inventory_file_sha256", "completion_file_sha256",
            "original_identity_authorization_sha256", "mappings",
        )})
        if value.to_payload() != dict(payload):
            raise ValueError("projection evidence schema, claims or self hash differs")
        return value
