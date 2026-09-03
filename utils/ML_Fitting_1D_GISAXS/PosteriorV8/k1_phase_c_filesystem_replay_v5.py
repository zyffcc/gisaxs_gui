"""Production filesystem adapter for K1 Phase-C raw replay evidence.

Every input is a canonical, read-only JSON file named by one externally
supplied manifest-file SHA-256.  Loading and each revalidation reopen every
file with ``O_NOFOLLOW``, verify bytes and filesystem identity, and rebuild the
entire typed replay bundle.  The designed formal authorization also requires
an audited writer capability; until that producer exists it fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_contract_v5 import digest
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan
from .k1_phase_c_raw_artifacts_v5 import (
    V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA,
    V5_K1_PHASE_C_RAW_MANIFEST_VERSION,
    V5K1PhaseCQueryDistanceMatcher,
    V5K1PhaseCRawFile,
    build_v5_k1_phase_c_bundle_from_raw_files,
)
from .k1_phase_c_replay_contract_v5 import V5K1PhaseCReplayBundle


V5_K1_PHASE_C_FILESYSTEM_ADAPTER_ID = "v5_k1_phase_c_production_filesystem_replay"
V5_K1_PHASE_C_FILESYSTEM_ADAPTER_VERSION = (
    "canonical_read_only_nofollow_per_file_sha_inode_double_revalidation_v1"
)
V5_K1_PHASE_C_RAW_PRODUCER_BLOCKER = (
    "production_phase_c_lossless_artifact_writer_not_implemented_or_audited"
)

_MAX_MANIFEST_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_FILE_FIELDS = {"file_id", "role", "relative_path", "sha256"}
_ALLOWED_ROLES = {
    "artifact_binding",
    "proposal_execution_policy",
    "split_receipt",
    "evaluator_config",
    "parent_provenance",
    "distance_context",
    "branch_search_sidecar",
    "reference_bank",
    "reference_search_trace",
    "method_exact_call_trace",
    "representative_payload",
}
_FORMAL_CAPABILITY_SEAL = object()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int


def _identity(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _read_immutable_json(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    name: str,
) -> tuple[dict[str, object], _FileIdentity]:
    expected = digest(expected_sha256, f"{name} SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{name} must be an existing non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > maximum_bytes
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise ValueError(f"{name} must be a bounded read-only regular file")
        chunks = []
        observed = sha256()
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            observed.update(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise ValueError(f"{name} exceeds its bounded size")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"{name} changed while it was read")
        encoded = b"".join(chunks)
        if observed.hexdigest() != expected:
            raise ValueError(f"{name} file SHA-256 does not match its manifest binding")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or canonical_json(payload).encode("utf-8") != encoded:
        raise ValueError(f"{name} must contain exactly one canonical JSON object")
    return payload, _identity(after)


def _checked_manifest_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("manifest path must not contain parent traversal")
    lexical = path if path.is_absolute() else Path.cwd() / path
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise ValueError("manifest path must not contain symbolic links")
    if not lexical.is_file():
        raise ValueError("manifest path must identify a regular file")
    return lexical.resolve(strict=True)


def _artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError("raw artifact path must be a non-empty POSIX relative path")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("raw artifact path must be normalized and below the manifest root")
    selected = root.joinpath(*relative.parts)
    probe = root
    for part in relative.parts:
        probe /= part
        if probe.is_symlink():
            raise ValueError("raw artifact path must not contain symbolic links")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as exc:
        raise ValueError("raw artifact path is missing") from exc
    if root not in resolved.parents or resolved == root:
        raise ValueError("raw artifact escaped the manifest root")
    return resolved


@dataclass(frozen=True)
class _FilesystemSnapshot:
    manifest: Mapping[str, object]
    manifest_file_sha256: str
    identities: Mapping[str, _FileIdentity]
    bundle: V5K1PhaseCReplayBundle
    matcher: V5K1PhaseCQueryDistanceMatcher


def _load_filesystem_snapshot(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object],
) -> _FilesystemSnapshot:
    manifest, manifest_identity = _read_immutable_json(
        manifest_path,
        expected_sha256=expected_manifest_file_sha256,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        name="K1 Phase-C raw manifest",
    )
    if (
        manifest.get("schema") != V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA
        or manifest.get("version") != V5_K1_PHASE_C_RAW_MANIFEST_VERSION
    ):
        raise ValueError("unsupported K1 Phase-C raw manifest schema/version")
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("K1 Phase-C raw manifest requires a non-empty file table")
    root = manifest_path.parent.resolve(strict=True)
    files: dict[str, V5K1PhaseCRawFile] = {}
    identities: dict[str, _FileIdentity] = {"@manifest": manifest_identity}
    paths: set[Path] = {manifest_path}
    for index, value in enumerate(raw_rows):
        if not isinstance(value, dict) or set(value) != _FILE_FIELDS:
            raise ValueError(f"raw manifest files[{index}] fields are incomplete or unsupported")
        file_id = value["file_id"]
        role = value["role"]
        if not isinstance(file_id, str) or not file_id.strip() or file_id != file_id.strip():
            raise ValueError("raw file_id must be a non-empty stripped string")
        if file_id in files:
            raise ValueError("raw manifest file IDs must be unique")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"unsupported K1 Phase-C raw artifact role: {role!r}")
        expected_sha = digest(value["sha256"], f"files[{index}].sha256")
        path = _artifact_path(root, value["relative_path"])
        if path in paths:
            raise ValueError("raw manifest paths must be unique and cannot name the manifest")
        paths.add(path)
        payload, identity = _read_immutable_json(
            path,
            expected_sha256=expected_sha,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
            name=f"raw artifact {file_id!r}",
        )
        files[file_id] = V5K1PhaseCRawFile(
            file_id=file_id,
            role=role,
            relative_path=value["relative_path"],
            file_sha256=expected_sha,
            payload=payload,
        )
        identities[file_id] = identity
    bundle, matcher = build_v5_k1_phase_c_bundle_from_raw_files(
        manifest=manifest,
        files=files,
        plan=plan,
        contract=contract,
    )
    return _FilesystemSnapshot(
        manifest=manifest,
        manifest_file_sha256=expected_manifest_file_sha256,
        identities=identities,
        bundle=bundle,
        matcher=matcher,
    )


@dataclass(frozen=True, eq=False)
class _V5K1PhaseCFormalReplayCapability:
    adapter: object = field(repr=False, compare=False)
    plan_sha256: str
    contract_sha256: str
    bundle_sha256: str
    manifest_file_sha256: str
    matcher_identity_sha256: str
    revalidation_count: int
    _seal: object = field(repr=False, compare=False)


class V5K1PhaseCFilesystemReplayAdapter:
    """Read-only production adapter; it never manufactures raw evidence."""

    adapter_id = V5_K1_PHASE_C_FILESYSTEM_ADAPTER_ID
    adapter_version = V5_K1_PHASE_C_FILESYSTEM_ADAPTER_VERSION

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        expected_manifest_file_sha256: str,
    ) -> None:
        self._manifest_path = _checked_manifest_path(manifest_path)
        self._expected_manifest_file_sha256 = digest(
            expected_manifest_file_sha256, "expected_manifest_file_sha256"
        )
        self._snapshot: _FilesystemSnapshot | None = None
        self._loaded_plan_sha256: str | None = None
        self._loaded_contract_sha256: str | None = None
        self._revalidation_count = 0

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def manifest_file_sha256(self) -> str:
        return self._expected_manifest_file_sha256

    @property
    def equivalence_distance_matcher(self) -> V5K1PhaseCQueryDistanceMatcher:
        if self._snapshot is None:
            raise RuntimeError("raw evidence must be loaded before requesting its distance matcher")
        return self._snapshot.matcher

    def load_bundle(
        self, *, plan: V5K1PhaseCPlan, contract: dict[str, object]
    ) -> V5K1PhaseCReplayBundle:
        snapshot = _load_filesystem_snapshot(
            self._manifest_path,
            expected_manifest_file_sha256=self._expected_manifest_file_sha256,
            plan=plan,
            contract=contract,
        )
        self._snapshot = snapshot
        self._loaded_plan_sha256 = plan.sha256
        self._loaded_contract_sha256 = str(contract["contract_sha256"])
        self._revalidation_count = 0
        return snapshot.bundle

    def revalidate_bundle(
        self,
        *,
        bundle: V5K1PhaseCReplayBundle,
        plan: V5K1PhaseCPlan,
        contract: dict[str, object],
    ) -> None:
        original = self._snapshot
        if original is None or bundle is not original.bundle:
            raise ValueError("adapter can only revalidate the exact bundle it loaded")
        if (
            self._loaded_plan_sha256 != plan.sha256
            or self._loaded_contract_sha256 != contract.get("contract_sha256")
        ):
            raise ValueError("adapter replay identity changed after bundle loading")
        fresh = _load_filesystem_snapshot(
            self._manifest_path,
            expected_manifest_file_sha256=self._expected_manifest_file_sha256,
            plan=plan,
            contract=contract,
        )
        if (
            fresh.identities != original.identities
            or dict(fresh.manifest) != dict(original.manifest)
            or fresh.bundle.sha256 != original.bundle.sha256
            or fresh.bundle.audit_payload() != original.bundle.audit_payload()
            or fresh.matcher.identity_sha256 != original.matcher.identity_sha256
        ):
            raise RuntimeError("K1 Phase-C raw files or identities changed during replay")
        self._revalidation_count += 1

    def _mint_formal_capability(
        self,
        *,
        bundle: V5K1PhaseCReplayBundle,
        plan: V5K1PhaseCPlan,
        contract: Mapping[str, object],
    ) -> _V5K1PhaseCFormalReplayCapability:
        snapshot = self._snapshot
        if (
            type(self) is not V5K1PhaseCFilesystemReplayAdapter
            or snapshot is None
            or bundle is not snapshot.bundle
            or plan.formal is not True
            or self._loaded_plan_sha256 != plan.sha256
            or self._loaded_contract_sha256 != contract.get("contract_sha256")
            or self._revalidation_count < 2
        ):
            raise RuntimeError("formal Phase-C capability requires two complete filesystem replays")
        _require_audited_v5_k1_phase_c_production_writer()
        return _V5K1PhaseCFormalReplayCapability(
            adapter=self,
            plan_sha256=plan.sha256,
            contract_sha256=str(contract["contract_sha256"]),
            bundle_sha256=bundle.sha256,
            manifest_file_sha256=self._expected_manifest_file_sha256,
            matcher_identity_sha256=snapshot.matcher.identity_sha256,
            revalidation_count=self._revalidation_count,
            _seal=_FORMAL_CAPABILITY_SEAL,
        )


def _is_v5_k1_phase_c_production_filesystem_adapter(value: object) -> bool:
    return type(value) is V5K1PhaseCFilesystemReplayAdapter


def _require_audited_v5_k1_phase_c_production_writer() -> None:
    """Fail closed until a real writer receipt verifier is implemented.

    This deliberate hard stop is not configurable and accepts no serialized
    substitute.  A future audited production writer must add its own opaque
    receipt/capability verification here before formal minting can proceed.
    """

    raise RuntimeError(V5_K1_PHASE_C_RAW_PRODUCER_BLOCKER)


def _authorize_v5_k1_phase_c_formal_filesystem_replay(
    port: object,
    *,
    bundle: V5K1PhaseCReplayBundle,
    plan: V5K1PhaseCPlan,
    contract: Mapping[str, object],
) -> None:
    """Consume an opaque capability minted only by the exact production adapter."""

    if type(port) is not V5K1PhaseCFilesystemReplayAdapter:
        raise RuntimeError("formal K1 Phase-C replay requires the production filesystem adapter")
    capability = port._mint_formal_capability(bundle=bundle, plan=plan, contract=contract)
    if (
        type(capability) is not _V5K1PhaseCFormalReplayCapability
        or capability._seal is not _FORMAL_CAPABILITY_SEAL
        or capability.adapter is not port
        or capability.plan_sha256 != plan.sha256
        or capability.contract_sha256 != contract.get("contract_sha256")
        or capability.bundle_sha256 != bundle.sha256
        or capability.manifest_file_sha256 != port.manifest_file_sha256
        or capability.matcher_identity_sha256
        != port.equivalence_distance_matcher.identity_sha256
        or capability.revalidation_count != port._revalidation_count
    ):
        raise RuntimeError("formal K1 Phase-C filesystem capability did not validate")


__all__ = [
    "V5_K1_PHASE_C_FILESYSTEM_ADAPTER_ID",
    "V5_K1_PHASE_C_FILESYSTEM_ADAPTER_VERSION",
    "V5_K1_PHASE_C_RAW_PRODUCER_BLOCKER",
    "V5K1PhaseCFilesystemReplayAdapter",
]
