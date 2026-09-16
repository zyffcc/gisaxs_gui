"""Completion-last, read-only publication for audited K1 Phase-B results."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .k1_phase_a_cross_platform_v5 import file_identity as strict_file_identity
from .k1_phase_b_contract_v5 import (
    CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS,
    V5_K1_PHASE_B_RESULT_FILENAME,
    V5_K1_PHASE_B_SCHEMA,
    V5_K1_PHASE_B_VERSION,
    cross_node_stable_file_identity,
    digest,
)
from .k1_phase_b_capability_v5 import (
    V5K1PhaseBInputCapability,
    _claim_phase_b_completion,
    _consumed_phase_b_capability_payload,
    _recheck_consumed_phase_b_capability,
)


V5_K1_PHASE_B_COMPLETION_SCHEMA = "gisaxs.posterior_v8.k1_phase_b_completion/v11"
V5_K1_PHASE_B_COMPLETION_VERSION = (
    "posterior_v8_closed_interval_tolerance_result_then_completion_last_v11"
)
V5_K1_PHASE_B_COMPLETION_FILENAME = "phase-b-complete-v11.json"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_PUBLISHED_FILENAMES = {
    V5_K1_PHASE_B_RESULT_FILENAME,
    V5_K1_PHASE_B_COMPLETION_FILENAME,
}


def _publish_read_only_json(
    target: Path, payload: Mapping[str, object], *, name: str
) -> dict[str, object]:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {name}: {target}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    target_created = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        pending = strict_file_identity(
            temporary, name=f"pending {name}", require_read_only=True
        )
        if pending["mode"] != 0o400 or pending["nlink"] != 1:
            raise RuntimeError(f"{name} pending permissions/link count are invalid")
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {name}: {target}") from None
        target_created = True
        temporary.unlink()
        identity = strict_file_identity(target, name=name, require_read_only=True)
        if identity["mode"] != 0o400 or identity["nlink"] != 1:
            raise RuntimeError(f"{name} final permissions/link count are invalid")
        return identity
    except BaseException:
        if target_created:
            target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _completion_payload(
    *,
    output_dir: Path,
    result_identity: Mapping[str, object],
    result_payload: Mapping[str, object],
    launch_binding: Mapping[str, object],
    capability_payload: Mapping[str, object],
) -> dict[str, object]:
    result_digest = digest(
        result_payload.get("result_payload_sha256"), "Phase-B result payload SHA-256"
    )
    result_core = dict(result_payload)
    result_core.pop("result_payload_sha256", None)
    if sha256(canonical_json(result_core).encode("utf-8")).hexdigest() != result_digest:
        raise ValueError("Phase-B result payload SHA-256 does not reproduce")
    if (
        result_payload.get("schema_version") != V5_K1_PHASE_B_SCHEMA
        or result_payload.get("version") != V5_K1_PHASE_B_VERSION
        or result_payload.get("launch_chain") != dict(launch_binding)
        or result_payload.get("job_local_capability") != dict(capability_payload)
    ):
        raise ValueError("Phase-B result schema or launch-chain binding is incompatible")
    stage = launch_binding.get("launch_stage")
    job_id = launch_binding.get("slurm_job_id")
    plan = launch_binding.get("launch_plan")
    if stage not in {"engineering_smoke", "formal_gate"}:
        raise ValueError("completion launch stage is unsupported")
    if not isinstance(job_id, str) or not job_id.isdigit() or job_id.startswith("0"):
        raise ValueError("completion Slurm job id is malformed")
    if not isinstance(plan, Mapping):
        raise ValueError("completion launch-plan evidence is missing")
    stable_result_identity = cross_node_stable_file_identity(result_identity)
    if stable_result_identity is None:
        raise ValueError("Phase-B result file identity is incomplete")
    portable_result = {
        **stable_result_identity,
        "path": str(output_dir / V5_K1_PHASE_B_RESULT_FILENAME),
        "result_payload_sha256": result_digest,
    }
    core = {
        "schema_version": V5_K1_PHASE_B_COMPLETION_SCHEMA,
        "version": V5_K1_PHASE_B_COMPLETION_VERSION,
        "status": "COMPLETE",
        "consumer_contract": "completion_marker_required_result_alone_is_incomplete",
        "launch_stage": stage,
        "slurm_job_id": job_id,
        "output_dir": str(output_dir),
        "result": portable_result,
        "result_identity_policy": {
            "bound_fields": list(CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS),
            "device_field": "mount_namespace_local_not_cross_node_bound",
            "canonical_evidence_excludes_device": True,
        },
        "launch_plan": dict(plan),
        "job_local_capability": dict(capability_payload),
        "formal_prerequisite_evidence_sha256": launch_binding.get(
            "formal_prerequisite_evidence_sha256"
        ),
        "publication": "exclusive_directory_result_then_live_recheck_then_completion_last",
    }
    return {
        **core,
        "completion_payload_sha256": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def _verify_directory_protection(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("published Phase-B output is not a real directory")
    if stat.S_IMODE(path.stat().st_mode) & _WRITE_BITS:
        raise RuntimeError("published Phase-B output directory remains writable")


def _remove_owned_publication(path: Path) -> None:
    """Remove only the exact private staging/publication shape created here."""

    if not path.exists() or path.is_symlink() or not path.is_dir():
        return
    path.chmod(0o700)
    children = tuple(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        return
    if any(child.name not in _PUBLISHED_FILENAMES for child in children):
        return
    for child in children:
        child.unlink()
    path.rmdir()


def publish_v5_k1_phase_b_completed_result(
    output_dir: Path,
    result_payload: Mapping[str, object],
    *,
    launch_binding: Mapping[str, object],
    capability: V5K1PhaseBInputCapability,
) -> dict[str, object]:
    """Publish result plus completion marker only after all final rechecks pass."""

    output = output_dir.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase-B output: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("Phase-B output parent must be a real directory")
    capability_payload = _consumed_phase_b_capability_payload(capability)
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        result_identity = _publish_read_only_json(
            output / V5_K1_PHASE_B_RESULT_FILENAME,
            result_payload,
            name="Phase-B result",
        )
        _recheck_consumed_phase_b_capability(capability)
        completion_capability = _claim_phase_b_completion(capability)
        if completion_capability != capability_payload:
            raise RuntimeError("Phase-B completion capability changed before publication")
        completion = _completion_payload(
            output_dir=output,
            result_identity=result_identity,
            result_payload=result_payload,
            launch_binding=launch_binding,
            capability_payload=completion_capability,
        )
        completion_identity = _publish_read_only_json(
            output / V5_K1_PHASE_B_COMPLETION_FILENAME,
            completion,
            name="Phase-B completion marker",
        )
        output.chmod(0o500)
        output_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        _verify_directory_protection(output)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        final_result_identity = strict_file_identity(
            output / V5_K1_PHASE_B_RESULT_FILENAME,
            name="published Phase-B result",
            require_read_only=True,
        )
        final_completion_identity = strict_file_identity(
            output / V5_K1_PHASE_B_COMPLETION_FILENAME,
            name="published Phase-B completion marker",
            require_read_only=True,
        )
        if final_result_identity != result_identity or (
            final_completion_identity != completion_identity
        ):
            raise RuntimeError("published Phase-B result/completion identity drift detected")
        _verify_directory_protection(output)
        return {
            "result": dict(result_payload),
            "result_identity": final_result_identity,
            "completion": completion,
            "completion_identity": final_completion_identity,
        }
    except BaseException:
        _remove_owned_publication(output)
        raise


__all__ = [
    "V5_K1_PHASE_B_COMPLETION_FILENAME",
    "V5_K1_PHASE_B_COMPLETION_SCHEMA",
    "V5_K1_PHASE_B_COMPLETION_VERSION",
    "publish_v5_k1_phase_b_completed_result",
]
