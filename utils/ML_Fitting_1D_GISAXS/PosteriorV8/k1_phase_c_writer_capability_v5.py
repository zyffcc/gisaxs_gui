"""Verify a writer completion receipt and mint one live adapter capability."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_METHOD_IDS,
    digest,
    v5_k1_phase_c_contract_payload,
)
from .k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan
from .k1_phase_c_production_writer_v5 import (
    V5_K1_PHASE_C_MANIFEST_FILENAME,
    V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME,
    V5_K1_PHASE_C_WRITER_RECEIPT_SCHEMA,
    V5_K1_PHASE_C_WRITER_RECEIPT_VERSION,
)


_PROCESS_ID = os.getpid()
_MINT_SEAL = object()
_ATTESTATION_SEAL = object()
_SEEN_RECEIPTS: set[tuple[int, int, int, str]] = set()
_LIVE_CAPABILITIES: dict[object, "_ReceiptState"] = {}
_MAX_RECEIPT_BYTES = 256 * 1024 * 1024
_RECEIPT_FIELDS = {
    "schema",
    "version",
    "status",
    "production_eligible",
    "formal",
    "plan_sha256",
    "contract_sha256",
    "manifest_relative_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "raw_file_count",
    "raw_file_inventory_sha256",
    "parent_count",
    "output_bundle_sha256",
    "matcher_identity_sha256",
    "upstream_file_count",
    "upstream_provenance",
    "upstream_provenance_sha256",
    "writer_policy",
    "receipt_sha256",
}
_UPSTREAM_FIELDS = {"scope", "parent_sha256", "role", "file_sha256"}
_REQUIRED_GLOBAL_UPSTREAM_ROLES = {
    "source_archive",
    "source_manifest",
    "cross_platform_gate_claim",
    "phase_a_launch_receipt",
    "model_artifact",
    "model_weights",
    "model_training_result",
    "split_receipt",
}
_REQUIRED_PARENT_UPSTREAM_ROLES = {
    "search_sidecar",
    "search_evidence_receipt",
    "reference_bank",
    "reference_trace",
}
_WRITER_POLICY = {
    "typed_live_inputs_only_for_production": True,
    "audit_summaries_are_not_lossless_inputs": True,
    "canonical_json_without_trailing_bytes": True,
    "exclusive_non_overwrite_atomic_publication": True,
    "regular_files_mode": "0400",
    "directories_mode_after_completion": "0500",
    "unique_link_count_required": 1,
    "exact_emitted_intensity_dtype": "<f8",
    "exact_emitted_intensity_finite_and_positive": True,
    "receipt_written_after_manifest_and_full_typed_replay": True,
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    link_count: int


def _identity(metadata: os.stat_result) -> _Identity:
    return _Identity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        mode=stat.S_IMODE(metadata.st_mode),
        link_count=metadata.st_nlink,
    )


def _safe_receipt_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.name != V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME or ".." in path.parts:
        raise ValueError("writer receipt must use its canonical completion-last filename")
    lexical = path if path.is_absolute() else Path.cwd() / path
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise ValueError("writer receipt path must not contain symbolic links")
    return lexical.resolve(strict=True)


def _read_canonical_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    maximum_bytes: int = _MAX_RECEIPT_BYTES,
) -> tuple[dict[str, object], str, _Identity]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size > maximum_bytes
        ):
            raise ValueError("writer artifact must be a bounded uniquely linked 0400 file")
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > maximum_bytes:
            raise ValueError("writer artifact exceeds its bounded size")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError("writer artifact changed while being verified")
    finally:
        os.close(descriptor)
    file_sha = sha256(raw).hexdigest()
    if expected_sha256 is not None and file_sha != digest(
        expected_sha256, "writer artifact SHA-256"
    ):
        raise ValueError("writer artifact file SHA-256 changed")
    try:
        payload = json.loads(
            bytes(raw),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("writer artifact is not strict JSON") from exc
    if not isinstance(payload, dict) or canonical_json(payload).encode("utf-8") != bytes(raw):
        raise ValueError("writer artifact must contain one canonical JSON object")
    return payload, file_sha, _identity(after)


def _relative_files(root: Path) -> tuple[dict[str, _Identity], int]:
    root_metadata = root.stat(follow_symlinks=False)
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o500
    ):
        raise ValueError("writer output root must be a non-symlink mode-0500 directory")
    files: dict[str, _Identity] = {}
    latest_nonreceipt_mtime = 0
    inode_keys = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("writer output must not contain symbolic links")
        relative = path.relative_to(root).as_posix()
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o500:
                raise ValueError("writer output directories must be mode 0500")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_nlink != 1
        ):
            raise ValueError("writer output files must be unique regular mode-0400 files")
        identity = _identity(metadata)
        inode_key = (identity.device, identity.inode)
        if inode_key in inode_keys:
            raise ValueError("writer output files must not share a hard-linked inode")
        inode_keys.add(inode_key)
        files[relative] = identity
        if relative != V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME:
            latest_nonreceipt_mtime = max(latest_nonreceipt_mtime, identity.mtime_ns)
    return files, latest_nonreceipt_mtime


@dataclass(frozen=True)
class _ReceiptState:
    root: Path
    receipt_path: Path
    manifest_path: Path
    receipt_identity: _Identity
    manifest_identity: _Identity
    file_identities: Mapping[str, _Identity]
    manifest_file_sha256: str
    plan_sha256: str
    contract_sha256: str
    bundle_sha256: str
    matcher_identity_sha256: str


class _V5K1PhaseCWriterCapability:
    __slots__ = ("_nonce",)

    def __new__(cls, seal=None):
        if cls is not _V5K1PhaseCWriterCapability or seal is not _MINT_SEAL:
            raise TypeError("writer capability has no public constructor")
        return super().__new__(cls)

    def __init__(self, seal=None) -> None:
        self._nonce = object()

    def __init_subclass__(cls, **kwargs):
        raise TypeError("writer capability cannot be subclassed")

    def __reduce__(self):
        raise TypeError("writer capability cannot be serialized")

    def __reduce_ex__(self, _protocol):
        raise TypeError("writer capability cannot be serialized")


@dataclass(frozen=True)
class _WriterAttestation:
    state: _ReceiptState
    adapter: object
    process_id: int
    _seal: object


def _receipt_key(identity: _Identity, receipt_sha256: str) -> tuple[int, int, int, str]:
    return (identity.device, identity.inode, identity.ctime_ns, receipt_sha256)


def verify_v5_k1_phase_c_writer_receipt(
    receipt_path: str | os.PathLike[str],
) -> object:
    """Fully replay one production receipt and return one opaque live capability."""

    if os.getpid() != _PROCESS_ID:
        raise RuntimeError("writer receipt verifier is bound to its original process")
    selected = _safe_receipt_path(receipt_path)
    receipt, _, receipt_identity = _read_canonical_file(selected)
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("writer receipt fields are incomplete or unsupported")
    core = dict(receipt)
    supplied_sha = digest(core.pop("receipt_sha256"), "writer receipt SHA-256")
    if supplied_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("writer receipt semantic SHA-256 does not reproduce")
    key = _receipt_key(receipt_identity, supplied_sha)
    if key in _SEEN_RECEIPTS:
        raise RuntimeError("writer receipt capability was already minted in this process")
    plan = build_v5_k1_phase_c_plan(formal=True)
    contract = v5_k1_phase_c_contract_payload()
    if (
        core["schema"] != V5_K1_PHASE_C_WRITER_RECEIPT_SCHEMA
        or core["version"] != V5_K1_PHASE_C_WRITER_RECEIPT_VERSION
        or core["status"] != "complete"
        or core["production_eligible"] is not True
        or core["formal"] is not True
        or core["plan_sha256"] != plan.sha256
        or core["contract_sha256"] != contract["contract_sha256"]
        or plan.formal is not True
        or core["writer_policy"] != _WRITER_POLICY
        or core["manifest_relative_path"] != V5_K1_PHASE_C_MANIFEST_FILENAME
    ):
        raise ValueError("writer receipt is not an eligible frozen formal production receipt")
    raw_file_count = _nonnegative_int(core["raw_file_count"], "raw_file_count")
    parent_count = _nonnegative_int(core["parent_count"], "parent_count")
    upstream_file_count = _nonnegative_int(core["upstream_file_count"], "upstream_file_count")
    if raw_file_count < 1 or parent_count < 1 or upstream_file_count < 1:
        raise ValueError("writer receipt counts must be positive")
    upstream = core["upstream_provenance"]
    if not isinstance(upstream, list) or any(
        not isinstance(value, Mapping) or set(value) != _UPSTREAM_FIELDS for value in upstream
    ):
        raise ValueError("writer upstream provenance is incomplete")
    if (
        len(upstream) != upstream_file_count
        or upstream
        != sorted(
            upstream,
            key=lambda value: (
                str(value["scope"]),
                str(value["parent_sha256"]),
                str(value["role"]),
                str(value["file_sha256"]),
            ),
        )
        or sha256(canonical_json(upstream).encode("utf-8")).hexdigest()
        != core["upstream_provenance_sha256"]
    ):
        raise ValueError("writer upstream provenance inventory does not reproduce")
    identities = set()
    global_roles: dict[str, set[str]] = {}
    parent_roles: dict[str, dict[str, set[str]]] = {}
    for value in upstream:
        scope = value["scope"]
        role = value["role"]
        parent_sha = value["parent_sha256"]
        file_sha = digest(value["file_sha256"], "upstream provenance file SHA-256")
        if not isinstance(role, str) or not role.strip() or role != role.strip():
            raise ValueError("upstream provenance role must be non-empty stripped text")
        identity = (scope, parent_sha, role, file_sha)
        if identity in identities:
            raise ValueError("upstream provenance contains a duplicate binding")
        identities.add(identity)
        if scope == "global" and parent_sha is None:
            global_roles.setdefault(role, set()).add(file_sha)
        elif scope == "parent" and isinstance(parent_sha, str):
            digest(parent_sha, "upstream parent SHA-256")
            parent_roles.setdefault(parent_sha, {}).setdefault(role, set()).add(file_sha)
        else:
            raise ValueError("upstream provenance scope/parent binding is invalid")
    if set(global_roles) != _REQUIRED_GLOBAL_UPSTREAM_ROLES or any(
        len(hashes) != 1 for hashes in global_roles.values()
    ):
        raise ValueError("writer receipt lacks exact global source/model/split provenance")
    if len(parent_roles) != parent_count:
        raise ValueError("writer receipt lacks per-parent upstream provenance")
    required_method_roles = {f"method_trace:{method_id}" for method_id in K1_PHASE_C_METHOD_IDS}
    for roles in parent_roles.values():
        role_names = set(roles)
        if not _REQUIRED_PARENT_UPSTREAM_ROLES <= role_names:
            raise ValueError("writer receipt lacks search/reference parent provenance")
        if {
            role for role in role_names if role.startswith("method_trace:")
        } != required_method_roles:
            raise ValueError("writer receipt lacks all three method trace sources")
        allowed = (
            _REQUIRED_PARENT_UPSTREAM_ROLES
            | required_method_roles
            | {
                role
                for role in role_names
                if role.startswith("representative_payload:")
                and role.removeprefix("representative_payload:").strip()
                == role.removeprefix("representative_payload:")
                and role.removeprefix("representative_payload:")
            }
        )
        if role_names != allowed or any(len(hashes) != 1 for hashes in roles.values()):
            raise ValueError("writer receipt contains unsupported parent provenance")
    manifest_path = selected.parent / V5_K1_PHASE_C_MANIFEST_FILENAME
    manifest, manifest_file_sha, manifest_identity = _read_canonical_file(
        manifest_path, expected_sha256=core["manifest_file_sha256"]
    )
    manifest_core = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256") != core["manifest_sha256"]
        or sha256(canonical_json(manifest_core).encode("utf-8")).hexdigest()
        != core["manifest_sha256"]
    ):
        raise ValueError("writer manifest semantic SHA-256 does not reproduce")
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list) or (
        len(raw_rows) != raw_file_count
        or sha256(canonical_json(raw_rows).encode("utf-8")).hexdigest()
        != core["raw_file_inventory_sha256"]
    ):
        raise ValueError("writer raw-file inventory does not reproduce")
    files, latest_nonreceipt = _relative_files(selected.parent)
    expected_paths = {
        V5_K1_PHASE_C_MANIFEST_FILENAME,
        V5_K1_PHASE_C_WRITER_RECEIPT_FILENAME,
    }
    for row in raw_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("relative_path"), str):
            raise ValueError("writer manifest raw-file row is invalid")
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("writer manifest contains an unsafe raw-file path")
        expected_paths.add(relative.as_posix())
    if set(files) != expected_paths:
        raise ValueError("writer output has missing or extra physical files")
    if receipt_identity.mtime_ns < latest_nonreceipt:
        raise ValueError("writer receipt was not published after every data file")

    from .k1_phase_c_filesystem_replay_v5 import _load_filesystem_snapshot

    loaded = _load_filesystem_snapshot(
        manifest_path,
        expected_manifest_file_sha256=manifest_file_sha,
        plan=plan,
        contract=contract,
    )
    binding = loaded.bundle.artifact_binding
    expected_global_hashes = {
        "source_archive": binding.source_archive_sha256,
        "source_manifest": binding.source_manifest_sha256,
        "cross_platform_gate_claim": binding.cross_platform_gate_claim_sha256,
        "phase_a_launch_receipt": binding.phase_a_launch_receipt_sha256,
        "model_artifact": binding.model_artifact_sha256,
        "model_weights": binding.model_weights_sha256,
        "model_training_result": binding.model_training_result_sha256,
    }
    if any(
        global_roles[role] != {expected_sha}
        for role, expected_sha in expected_global_hashes.items()
    ):
        raise ValueError("writer global provenance disagrees with the typed artifact binding")
    manifest_parents = manifest.get("parents")
    if not isinstance(manifest_parents, list) or any(
        not isinstance(value, Mapping) or not isinstance(value.get("clean_parent_sha256"), str)
        for value in manifest_parents
    ):
        raise ValueError("writer manifest parent inventory is invalid")
    manifest_parent_sha256s = {
        digest(value["clean_parent_sha256"], "manifest clean parent SHA-256")
        for value in manifest_parents
    }
    if len(manifest_parent_sha256s) != len(manifest_parents) or manifest_parent_sha256s != set(
        parent_roles
    ):
        raise ValueError("writer receipt parent provenance does not match the manifest")
    for parent in loaded.bundle.parents:
        clean_parent_sha = parent.provenance.clean_parent_sha256
        expected_representative_roles = {
            f"representative_payload:{emission.candidate_id}"
            for method in parent.methods
            for emission in method.trace.candidate_emissions
        }
        observed_representative_roles = {
            role
            for role in parent_roles[clean_parent_sha]
            if role.startswith("representative_payload:")
        }
        if observed_representative_roles != expected_representative_roles:
            raise ValueError("writer representative provenance does not match typed emissions")
    if (
        len(loaded.bundle.parents) != parent_count
        or len(loaded.bundle.parents) != plan.total_parent_count
        or loaded.bundle.sha256 != core["output_bundle_sha256"]
        or loaded.matcher.identity_sha256 != core["matcher_identity_sha256"]
    ):
        raise ValueError("writer receipt does not reproduce its exact typed output bundle")
    state = _ReceiptState(
        root=selected.parent,
        receipt_path=selected,
        manifest_path=manifest_path,
        receipt_identity=receipt_identity,
        manifest_identity=manifest_identity,
        file_identities=dict(files),
        manifest_file_sha256=manifest_file_sha,
        plan_sha256=plan.sha256,
        contract_sha256=str(contract["contract_sha256"]),
        bundle_sha256=loaded.bundle.sha256,
        matcher_identity_sha256=loaded.matcher.identity_sha256,
    )
    capability = _V5K1PhaseCWriterCapability(_MINT_SEAL)
    _LIVE_CAPABILITIES[capability] = state
    _SEEN_RECEIPTS.add(key)
    return capability


def _revalidate_state(state: _ReceiptState) -> None:
    files, _ = _relative_files(state.root)
    if dict(files) != dict(state.file_identities):
        raise RuntimeError("writer output files or identities changed after receipt verification")
    _, _, receipt_identity = _read_canonical_file(state.receipt_path)
    _, manifest_sha, manifest_identity = _read_canonical_file(
        state.manifest_path, expected_sha256=state.manifest_file_sha256
    )
    if (
        receipt_identity != state.receipt_identity
        or manifest_identity != state.manifest_identity
        or manifest_sha != state.manifest_file_sha256
    ):
        raise RuntimeError("writer receipt or manifest changed after verification")


def _consume_v5_k1_phase_c_writer_capability(
    capability: object,
    *,
    adapter: object,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> _WriterAttestation:
    if os.getpid() != _PROCESS_ID or type(capability) is not _V5K1PhaseCWriterCapability:
        raise TypeError("adapter requires an exact live writer capability")
    state = _LIVE_CAPABILITIES.pop(capability, None)
    if state is None:
        raise RuntimeError("writer capability is invalid, reused, or from another process")
    if state.manifest_path != manifest_path or state.manifest_file_sha256 != manifest_file_sha256:
        raise ValueError("writer capability does not bind this exact manifest")
    _revalidate_state(state)
    return _WriterAttestation(state, adapter, _PROCESS_ID, _ATTESTATION_SEAL)


def _validate_v5_k1_phase_c_writer_attestation(
    value: object,
    *,
    adapter: object,
    manifest_path: Path,
    manifest_file_sha256: str,
    plan_sha256: str,
    contract_sha256: str,
    bundle_sha256: str | None = None,
) -> _ReceiptState:
    if (
        type(value) is not _WriterAttestation
        or value._seal is not _ATTESTATION_SEAL
        or value.adapter is not adapter
        or value.process_id != os.getpid()
    ):
        raise RuntimeError("formal replay has no live audited writer attestation")
    state = value.state
    if (
        state.manifest_path != manifest_path
        or state.manifest_file_sha256 != manifest_file_sha256
        or state.plan_sha256 != plan_sha256
        or state.contract_sha256 != contract_sha256
        or (bundle_sha256 is not None and state.bundle_sha256 != bundle_sha256)
    ):
        raise RuntimeError("writer attestation escaped its manifest/plan/contract/bundle")
    _revalidate_state(state)
    return state


__all__ = ["verify_v5_k1_phase_c_writer_receipt"]
