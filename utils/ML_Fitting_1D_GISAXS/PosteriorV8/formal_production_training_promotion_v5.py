"""File-backed, role-isolated promotion of formal search evidence.

The pure membership proof intentionally performs no I/O.  This module is the
promotion boundary used by a trainer: it first replays the receipt and every
executor evidence file, then proves global-plan membership, and finally checks
that the requested consumer role is the one derived from the frozen split.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Callable

from .formal_production_search_membership_v5 import (
    V5FormalProductionMembershipProof,
    verify_v5_formal_production_receipt_membership,
)
from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchPlan,
    V5_FORMAL_PRODUCTION_CONSUMER_ROLES,
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    authorize_v5_formal_production_search_shard,
)
from .grouped_artifact_v5 import canonical_json
from .search_evidence_receipt_v5 import (
    V5SearchEvidenceReceipt,
    read_v5_search_evidence_receipt,
)


V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_evidence_promotion/v1"
)
V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_VERSION = (
    "posterior_v8_file_replay_plan_membership_source_split_role_gate_v1"
)
V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_input_snapshot/v1"
)
V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_VERSION = (
    "private_read_only_parent_sidecar_receipt_copy_v1"
)

FormalProductionSourceGuard = Callable[[], str]


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_regular_path(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain parent traversal")
    lexical = path if path.is_absolute() else Path.cwd() / path
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise ValueError(f"{name} must not contain symbolic-link components")
    if not lexical.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return lexical.resolve(strict=True)


def _checked_directory_path(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain parent traversal")
    lexical = path if path.is_absolute() else Path.cwd() / path
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise ValueError(f"{name} must not contain symbolic-link components")
    if not lexical.is_dir():
        raise ValueError(f"{name} must be a directory with no symlink components")
    return lexical.resolve(strict=True)


def _copy_verified_file(source: Path, target: Path, expected_sha256: str) -> str:
    expected = _digest(expected_sha256, "snapshot source SHA-256")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, os.O_RDONLY | nofollow)
    target_fd = -1
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("snapshot source must be a regular file")
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o400,
        )
        digest = sha256()
        while chunk := os.read(source_fd, 1024 * 1024):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written < 1:  # pragma: no cover - regular-file defensive guard
                    raise OSError("private snapshot write made no progress")
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        observed = digest.hexdigest()
        if identity_before != identity_after or observed != expected:
            raise RuntimeError("formal evidence changed while its private copy was made")
    finally:
        os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
    if _file_sha256(target) != expected:
        raise RuntimeError("private formal-evidence copy failed SHA-256 verification")
    return expected


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> str:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return sha256(encoded).hexdigest()


def _verify_source_guard(
    guard: FormalProductionSourceGuard,
    *,
    expected_sha256: str,
) -> None:
    if not callable(guard):
        raise TypeError("pre_consume_source_guard must be callable")
    if guard() != expected_sha256:
        raise RuntimeError("formal-production source bundle changed before consumption")


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FormalProductionEvidencePromotion:
    """Checked identity; mutable source paths are never direct training inputs."""

    receipt_path: Path
    parent_dataset_path: Path
    sidecar_path: Path
    receipt_file_sha256: str
    parent_file_sha256: str
    sidecar_file_sha256: str
    membership_proof: V5FormalProductionMembershipProof
    consumer_role: str
    schema: str = V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_VERSION

    def __post_init__(self) -> None:
        if self.schema != V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_SCHEMA or (
            self.version != V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_VERSION
        ):
            raise ValueError("unsupported formal-production promotion contract")
        if self.consumer_role not in V5_FORMAL_PRODUCTION_CONSUMER_ROLES:
            raise ValueError("consumer_role is unsupported")
        if not isinstance(self.membership_proof, V5FormalProductionMembershipProof):
            raise TypeError("membership_proof has an invalid type")
        if self.membership_proof.authorized_consumer_role != self.consumer_role:
            raise ValueError("promotion role disagrees with its membership proof")
        for name in (
            "receipt_file_sha256",
            "parent_file_sha256",
            "sidecar_file_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("receipt_path", "parent_dataset_path", "sidecar_path"):
            object.__setattr__(
                self,
                name,
                _checked_regular_path(getattr(self, name), name),
            )
        if (
            _file_sha256(self.receipt_path) != self.receipt_file_sha256
            or _file_sha256(self.parent_dataset_path) != self.parent_file_sha256
            or _file_sha256(self.sidecar_path) != self.sidecar_file_sha256
        ):
            raise RuntimeError("formal evidence changed while promotion was issued")

    @property
    def gradient_training_permitted(self) -> bool:
        return self.consumer_role == V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "receipt_path": os.fspath(self.receipt_path),
            "parent_dataset_path": os.fspath(self.parent_dataset_path),
            "sidecar_path": os.fspath(self.sidecar_path),
            "receipt_file_sha256": self.receipt_file_sha256,
            "parent_file_sha256": self.parent_file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "membership_proof": self.membership_proof.audit_payload(),
            "membership_proof_sha256": self.membership_proof.sha256,
            "consumer_role": self.consumer_role,
            "evidence_files_verified": True,
            "gradient_training_permitted": self.gradient_training_permitted,
            "tuning_validation_can_supply_gradients": False,
            "mutable_source_paths_are_training_inputs": False,
            "requires_private_snapshot_before_consumption": True,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def _input_snapshot_completion_core(
    *,
    promotion_sha256: str,
    membership_proof_sha256: str,
    consumer_role: str,
    parent_file_sha256: str,
    sidecar_file_sha256: str,
    evidence_receipt_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema": V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_SCHEMA,
        "version": V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_VERSION,
        "status": "complete_private_formal_input_snapshot",
        "promotion_sha256": promotion_sha256,
        "membership_proof_sha256": membership_proof_sha256,
        "consumer_role": consumer_role,
        "gradient_training_permitted": (
            consumer_role == V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
        ),
        "source_paths_are_training_inputs": False,
        "files": {
            "parent": {
                "relative_path": "grouped-parent.gvd5",
                "file_sha256": parent_file_sha256,
            },
            "sidecar": {
                "relative_path": "search-supervision.gvd5",
                "file_sha256": sidecar_file_sha256,
            },
            "source_evidence_receipt_copy": {
                "relative_path": "source-evidence-receipt.json",
                "file_sha256": evidence_receipt_file_sha256,
                "audit_copy_only_relative_executor_paths_are_source_relative": True,
            },
        },
        "publication_rule": "completion_written_after_copy_hashes_and_second_full_replay",
    }


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FormalProductionInputSnapshot:
    """Private read-only copies that a trainer may open instead of source paths."""

    snapshot_root: Path
    parent_dataset_path: Path
    sidecar_path: Path
    evidence_receipt_copy_path: Path
    parent_file_sha256: str
    sidecar_file_sha256: str
    evidence_receipt_file_sha256: str
    promotion_sha256: str
    membership_proof_sha256: str
    consumer_role: str
    completion_path: Path
    completion_sha256: str
    completion_file_sha256: str
    schema: str = V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.schema != V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_SCHEMA or (
            self.version != V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_VERSION
        ):
            raise ValueError("unsupported formal input-snapshot contract")
        if self.consumer_role not in V5_FORMAL_PRODUCTION_CONSUMER_ROLES:
            raise ValueError("snapshot consumer_role is unsupported")
        root = _checked_directory_path(self.snapshot_root, "snapshot_root")
        object.__setattr__(self, "snapshot_root", root)
        for name in (
            "parent_file_sha256",
            "sidecar_file_sha256",
            "evidence_receipt_file_sha256",
            "promotion_sha256",
            "membership_proof_sha256",
            "completion_sha256",
            "completion_file_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        checked_paths = {}
        for name in (
            "parent_dataset_path",
            "sidecar_path",
            "evidence_receipt_copy_path",
            "completion_path",
        ):
            selected = _checked_regular_path(getattr(self, name), name)
            if root not in selected.parents:
                raise ValueError("snapshot artifact escaped its private root")
            checked_paths[name] = selected
            object.__setattr__(self, name, selected)
        if (
            checked_paths["parent_dataset_path"] != root / "grouped-parent.gvd5"
            or checked_paths["sidecar_path"] != root / "search-supervision.gvd5"
            or checked_paths["evidence_receipt_copy_path"]
            != root / "source-evidence-receipt.json"
            or checked_paths["completion_path"] != root / "completion.json"
        ):
            raise ValueError("snapshot files do not use the canonical private layout")
        expected = (
            (checked_paths["parent_dataset_path"], self.parent_file_sha256),
            (checked_paths["sidecar_path"], self.sidecar_file_sha256),
            (
                checked_paths["evidence_receipt_copy_path"],
                self.evidence_receipt_file_sha256,
            ),
            (checked_paths["completion_path"], self.completion_file_sha256),
        )
        if any(_file_sha256(path) != digest for path, digest in expected):
            raise RuntimeError("private formal input snapshot changed after publication")
        try:
            completion = json.loads(self.completion_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("formal input snapshot completion is unreadable") from exc
        if not isinstance(completion, dict):
            raise ValueError("formal input snapshot completion must be an object")
        core = dict(completion)
        supplied = core.pop("completion_sha256", None)
        expected_core = _input_snapshot_completion_core(
            promotion_sha256=self.promotion_sha256,
            membership_proof_sha256=self.membership_proof_sha256,
            consumer_role=self.consumer_role,
            parent_file_sha256=self.parent_file_sha256,
            sidecar_file_sha256=self.sidecar_file_sha256,
            evidence_receipt_file_sha256=self.evidence_receipt_file_sha256,
        )
        if (
            core != expected_core
            or supplied != self.completion_sha256
            or supplied
            != sha256(canonical_json(expected_core).encode("utf-8")).hexdigest()
        ):
            raise ValueError("formal input snapshot completion SHA-256 does not reproduce")
        if stat.S_IMODE(root.stat().st_mode) & 0o222 or any(
            stat.S_IMODE(path.stat().st_mode) & 0o222
            for path in checked_paths.values()
        ):
            raise ValueError("formal input snapshot must be read-only after publication")

    @property
    def gradient_training_permitted(self) -> bool:
        return self.consumer_role == V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE


def promote_v5_formal_production_search_evidence(
    plan: V5FormalProductionSearchPlan,
    *,
    output_relative_path: str,
    receipt_path: str | os.PathLike[str],
    parent_dataset_path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    consumer_role: str,
    pre_consume_source_guard: FormalProductionSourceGuard,
) -> V5FormalProductionEvidencePromotion:
    """Replay evidence and issue one role-specific, non-persistent capability."""

    if not isinstance(plan, V5FormalProductionSearchPlan):
        raise TypeError("plan must be a V5FormalProductionSearchPlan")
    if consumer_role not in V5_FORMAL_PRODUCTION_CONSUMER_ROLES:
        raise ValueError("consumer_role is unsupported")
    _verify_source_guard(
        pre_consume_source_guard,
        expected_sha256=plan.source.bundle_sha256,
    )
    selected_receipt = _checked_regular_path(receipt_path, "receipt_path")
    selected_parent = _checked_regular_path(parent_dataset_path, "parent_dataset_path")
    selected_sidecar = _checked_regular_path(sidecar_path, "sidecar_path")
    checked = read_v5_search_evidence_receipt(
        selected_receipt,
        parent_dataset_path=selected_parent,
        sidecar_path=selected_sidecar,
        require_training_eligible=True,
        expected_consumer_role=consumer_role,
    )
    proof = verify_v5_formal_production_receipt_membership(
        plan,
        output_relative_path=output_relative_path,
        receipt=checked,
    )
    authorization = authorize_v5_formal_production_search_shard(
        plan, shard_plan_sha256=proof.shard_plan_sha256
    )
    if (
        authorization.consumer_role != consumer_role
        or proof.authorized_consumer_role != consumer_role
        or proof.authorization_sha256 != authorization.sha256
    ):
        raise ValueError("formal evidence consumer role escaped the planned split")
    _verify_source_guard(
        pre_consume_source_guard,
        expected_sha256=plan.source.bundle_sha256,
    )
    receipt_file_sha = _file_sha256(selected_receipt)
    if receipt_file_sha != checked.file_sha256:
        raise RuntimeError("formal evidence receipt changed during promotion")
    return V5FormalProductionEvidencePromotion(
        receipt_path=selected_receipt,
        parent_dataset_path=selected_parent,
        sidecar_path=selected_sidecar,
        receipt_file_sha256=receipt_file_sha,
        parent_file_sha256=_file_sha256(selected_parent),
        sidecar_file_sha256=_file_sha256(selected_sidecar),
        membership_proof=proof,
        consumer_role=consumer_role,
    )


def reverify_v5_formal_production_evidence_promotion(
    plan: V5FormalProductionSearchPlan,
    promotion: V5FormalProductionEvidencePromotion,
    *,
    pre_consume_source_guard: FormalProductionSourceGuard,
) -> V5FormalProductionEvidencePromotion:
    """Detect intervening change; this does not pin mutable pathnames for use."""

    if not isinstance(promotion, V5FormalProductionEvidencePromotion):
        raise TypeError("promotion has an invalid type")
    replay = promote_v5_formal_production_search_evidence(
        plan,
        output_relative_path=promotion.membership_proof.output_relative_path,
        receipt_path=promotion.receipt_path,
        parent_dataset_path=promotion.parent_dataset_path,
        sidecar_path=promotion.sidecar_path,
        consumer_role=promotion.consumer_role,
        pre_consume_source_guard=pre_consume_source_guard,
    )
    if replay.audit_payload() != promotion.audit_payload() or replay.sha256 != promotion.sha256:
        raise RuntimeError("formal evidence changed after promotion")
    return replay


def materialize_v5_formal_production_input_snapshot(
    plan: V5FormalProductionSearchPlan,
    promotion: V5FormalProductionEvidencePromotion,
    *,
    snapshot_root: str | os.PathLike[str],
    pre_consume_source_guard: FormalProductionSourceGuard,
) -> V5FormalProductionInputSnapshot:
    """Copy verified training inputs to a new private root and publish completion last."""

    checked = reverify_v5_formal_production_evidence_promotion(
        plan,
        promotion,
        pre_consume_source_guard=pre_consume_source_guard,
    )
    target = Path(snapshot_root)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to reuse a formal input-snapshot root")
    parent = _checked_directory_path(target.parent, "snapshot parent")
    target = parent / target.name
    target.mkdir(mode=0o700)
    parent_copy = target / "grouped-parent.gvd5"
    sidecar_copy = target / "search-supervision.gvd5"
    receipt_copy = target / "source-evidence-receipt.json"
    _copy_verified_file(
        checked.parent_dataset_path,
        parent_copy,
        checked.parent_file_sha256,
    )
    _copy_verified_file(
        checked.sidecar_path,
        sidecar_copy,
        checked.sidecar_file_sha256,
    )
    _copy_verified_file(
        checked.receipt_path,
        receipt_copy,
        checked.receipt_file_sha256,
    )
    # Replay the source evidence again after copying.  Subsequent source-path
    # replacement cannot change the private copies handed to the trainer.
    reverify_v5_formal_production_evidence_promotion(
        plan,
        checked,
        pre_consume_source_guard=pre_consume_source_guard,
    )
    core = _input_snapshot_completion_core(
        promotion_sha256=checked.sha256,
        membership_proof_sha256=checked.membership_proof.sha256,
        consumer_role=checked.consumer_role,
        parent_file_sha256=checked.parent_file_sha256,
        sidecar_file_sha256=checked.sidecar_file_sha256,
        evidence_receipt_file_sha256=checked.receipt_file_sha256,
    )
    completion_sha = sha256(canonical_json(core).encode("utf-8")).hexdigest()
    completion_path = target / "completion.json"
    completion_file_sha = _write_json_exclusive(
        completion_path,
        {**core, "completion_sha256": completion_sha},
    )
    for path in (parent_copy, sidecar_copy, receipt_copy, completion_path):
        path.chmod(0o400)
    target.chmod(0o500)
    return V5FormalProductionInputSnapshot(
        snapshot_root=target,
        parent_dataset_path=parent_copy,
        sidecar_path=sidecar_copy,
        evidence_receipt_copy_path=receipt_copy,
        parent_file_sha256=checked.parent_file_sha256,
        sidecar_file_sha256=checked.sidecar_file_sha256,
        evidence_receipt_file_sha256=checked.receipt_file_sha256,
        promotion_sha256=checked.sha256,
        membership_proof_sha256=checked.membership_proof.sha256,
        consumer_role=checked.consumer_role,
        completion_path=completion_path,
        completion_sha256=completion_sha,
        completion_file_sha256=completion_file_sha,
    )


__all__ = [
    "FormalProductionSourceGuard",
    "V5FormalProductionEvidencePromotion",
    "V5FormalProductionInputSnapshot",
    "V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_SCHEMA",
    "V5_FORMAL_PRODUCTION_EVIDENCE_PROMOTION_VERSION",
    "V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_SCHEMA",
    "V5_FORMAL_PRODUCTION_INPUT_SNAPSHOT_VERSION",
    "materialize_v5_formal_production_input_snapshot",
    "promote_v5_formal_production_search_evidence",
    "reverify_v5_formal_production_evidence_promotion",
]
