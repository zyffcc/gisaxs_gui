"""Opaque, process-live authority for one verified K1 trainer invocation."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import hmac
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence
from weakref import WeakKeyDictionary

from .k1_input_closure_v5 import staged_binding_rows
from .k1_job_staging_v5 import rebuild_job_staging_proof
from .k1_staging_files_v5 import lexical_no_symlinks
from .k1_training_chain_contract_v5 import canonical_json
from .k1_training_chain_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_training_chain_fingerprints,
    validate_v5_k1_training_chain_plan,
)


V5_JOB_LOCAL_INPUT_CAPABILITY_SCHEMA = (
    "gisaxs.posterior_v8.grouped_training_job_local_input_capability/v2"
)
V5_JOB_LOCAL_INPUT_CAPABILITY_VERSION = (
    "k1_wrapper_minted_single_use_live_original_local_source_binding_v2"
)


class V5JobLocalInputCapability:
    """Non-serializable object identity backed by this module's live registry."""

    __slots__ = ("_nonce", "__weakref__")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        raise TypeError(
            "V5JobLocalInputCapability is opaque; the verified K1 wrapper mints it"
        )

    def __init_subclass__(cls, **kwargs):
        del cls, kwargs
        raise TypeError("V5JobLocalInputCapability cannot be subclassed")

    def __repr__(self) -> str:
        return "<V5JobLocalInputCapability opaque>"

    def __reduce__(self):
        raise TypeError("V5JobLocalInputCapability cannot be serialized")

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError("V5JobLocalInputCapability cannot be serialized")

    def __copy__(self):
        raise TypeError("V5JobLocalInputCapability cannot be copied")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("V5JobLocalInputCapability cannot be copied")

    def audit_payload(self) -> dict[str, object]:
        return deepcopy(_live_record(self)["audit"])


_CAPABILITY_SEAL_KEY = secrets.token_bytes(32)
_LIVE_CAPABILITIES: WeakKeyDictionary[
    V5JobLocalInputCapability, dict[str, object]
] = WeakKeyDictionary()


def _sealed_bytes(record: Mapping[str, object]) -> bytes:
    return canonical_json(
        {
            "audit": record["audit"],
            "verification": record["verification"],
            "state": record["state"],
            "pre_training_rehash_sha256": record[
                "pre_training_rehash_sha256"
            ],
            "post_training_rehash_sha256": record[
                "post_training_rehash_sha256"
            ],
        }
    ).encode("utf-8")


def _live_record(capability: V5JobLocalInputCapability) -> dict[str, object]:
    if type(capability) is not V5JobLocalInputCapability:
        raise TypeError("job-local input capability has an unsupported type")
    record = _LIVE_CAPABILITIES.get(capability)
    if record is None:
        raise RuntimeError(
            "job-local input capability was not minted by the verified K1 wrapper"
        )
    expected = hmac.digest(
        _CAPABILITY_SEAL_KEY,
        capability._nonce + _sealed_bytes(record),
        "sha256",
    )
    if not hmac.compare_digest(record["seal"], expected):
        raise RuntimeError("job-local input capability live seal is invalid")
    return record


def _reseal(
    capability: V5JobLocalInputCapability, record: dict[str, object]
) -> None:
    record["seal"] = hmac.digest(
        _CAPABILITY_SEAL_KEY,
        capability._nonce + _sealed_bytes(record),
        "sha256",
    )


def _bound_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: str(environment.get(name, ""))
        for name in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_TMPDIR",
            "TMPDIR",
            "POSTERIOR_V8_SCRATCH_BASE",
            "POSTERIOR_V8_JOB_TMP_ROOT",
        )
    }


def _require_live_process_environment(expected: Mapping[str, str]) -> None:
    for name, value in expected.items():
        if os.environ.get(name, "") != value:
            raise RuntimeError(f"live {name} differs from the wrapper capability binding")


def _consume_wrapper_mint(common: Mapping[str, object]):
    """Turn the wrapper's exclusive token file into an unlinked live descriptor."""

    claimed = common["wrapper_mint"]
    path = lexical_no_symlinks(Path(claimed["path"]), "wrapper one-shot mint")
    before = os.stat(path, follow_symlinks=False)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        status = os.fstat(descriptor)
        def identity(value):
            return {
                "path": str(path.resolve(strict=True)),
                "byte_count": value.st_size,
                "mode_octal": f"{stat.S_IMODE(value.st_mode):04o}",
                "device": value.st_dev,
                "inode": value.st_ino,
                "uid": value.st_uid,
                "link_count": value.st_nlink,
                "mtime_ns": value.st_mtime_ns,
                "ctime_ns": value.st_ctime_ns,
            }

        observed = {
            **identity(status),
        }
        token = os.pread(descriptor, 33, 0)
        if (
            not stat.S_ISREG(status.st_mode)
            or identity(before) != observed
            or observed != claimed
            or len(token) != 32
            or observed["mode_octal"] != "0400"
            or identity(os.fstat(descriptor)) != observed
            or identity(os.stat(path, follow_symlinks=False)) != observed
        ):
            raise RuntimeError("wrapper one-shot mint changed before consumption")
        path.unlink()
        unlinked = os.fstat(descriptor)
        if unlinked.st_nlink != 0 or path.exists() or path.is_symlink():
            raise RuntimeError("wrapper one-shot mint was not consumed")
        live_identity = {
            "byte_count": unlinked.st_size,
            "mode_octal": f"{stat.S_IMODE(unlinked.st_mode):04o}",
            "device": unlinked.st_dev,
            "inode": unlinked.st_ino,
            "uid": unlinked.st_uid,
            "link_count": unlinked.st_nlink,
            "mtime_ns": unlinked.st_mtime_ns,
            "ctime_ns": unlinked.st_ctime_ns,
            "sha256": sha256(token).hexdigest(),
        }
        return os.fdopen(descriptor, "rb"), live_identity
    except Exception:
        os.close(descriptor)
        raise


def _validate_consumed_wrapper_mint(record: Mapping[str, object]) -> None:
    stream = record["wrapper_mint_stream"]
    claimed = record["verification"]["common"]["wrapper_mint"]
    if stream.closed:
        raise RuntimeError("wrapper one-shot mint live descriptor is closed")
    status = os.fstat(stream.fileno())
    token = os.pread(stream.fileno(), 33, 0)
    expected = record["verification"]["wrapper_mint_live_identity"]
    observed = {
        "byte_count": status.st_size,
        "mode_octal": f"{stat.S_IMODE(status.st_mode):04o}",
        "device": status.st_dev,
        "inode": status.st_ino,
        "uid": status.st_uid,
        "link_count": status.st_nlink,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
        "sha256": sha256(token).hexdigest(),
    }
    if (
        not stat.S_ISREG(status.st_mode)
        or observed != expected
        or status.st_nlink != 0
        or status.st_size != claimed["byte_count"]
        or status.st_dev != claimed["device"]
        or status.st_ino != claimed["inode"]
        or status.st_uid != claimed["uid"]
        or f"{stat.S_IMODE(status.st_mode):04o}" != claimed["mode_octal"]
        or Path(claimed["path"]).exists()
        or Path(claimed["path"]).is_symlink()
    ):
        raise RuntimeError("consumed wrapper one-shot mint identity changed")


def _mint_job_local_input_capability(
    plan: Mapping[str, object],
    common_staging: Mapping[str, object],
    staged_inputs: Mapping[str, object],
    *,
    environment: Mapping[str, str],
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> V5JobLocalInputCapability:
    """Private wrapper entry: mint only after replaying every live binding."""

    value = validate_v5_k1_training_chain_plan(plan)
    if (
        not isinstance(common_staging, Mapping)
        or common_staging.get("worker_kind") != "training_seed"
    ):
        raise ValueError("trainer capability requires a training-seed staging proof")
    expected_environment = _bound_environment(environment)
    _require_live_process_environment(expected_environment)
    allowed = allowed_root.resolve(strict=True)
    verification: dict[str, object] = {
        "plan": deepcopy(value),
        "common": deepcopy(dict(common_staging)),
        "staged_inputs": deepcopy(dict(staged_inputs)),
        "environment": expected_environment,
        "allowed_root": str(allowed),
    }
    if rebuild_job_staging_proof(
        value,
        common_staging,
        environment=expected_environment,
        allowed_root=allowed,
    ) != dict(common_staging):
        raise RuntimeError("job-private staging identity changed before capability mint")
    replay = replay_v5_k1_training_chain_fingerprints(value, allowed_root=allowed)
    live_plan = {**value, "_live_staging_root": common_staging["staging_root"]}
    rows, trainer_paths = staged_binding_rows(
        staged_inputs,
        plan=live_plan,
        inventory=replay["inventory"],
        allowed_root=allowed,
    )
    pre_training_rehash = sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    mint_stream, mint_live_identity = _consume_wrapper_mint(common_staging)
    verification.update(
        binding_rows=deepcopy(rows),
        trainer_paths=trainer_paths,
        pre_training_rehash_sha256=pre_training_rehash,
        wrapper_mint_live_identity=mint_live_identity,
    )
    core = {
        "schema": V5_JOB_LOCAL_INPUT_CAPABILITY_SCHEMA,
        "version": V5_JOB_LOCAL_INPUT_CAPABILITY_VERSION,
        "worker_kind": "training_seed",
        "slurm_job_id": common_staging["slurm_job_id"],
        "slurm_array_task_id": common_staging["slurm_array_task_id"],
        "scratch_base": common_staging["scratch_base"],
        "scratch_base_source": common_staging["scratch_base_source"],
        "scratch_base_device": common_staging["scratch_base_device"],
        "scratch_base_inode": common_staging["scratch_base_inode"],
        "job_tmp_root": common_staging["job_tmp_root"],
        "job_tmp_root_device": common_staging["job_tmp_root_device"],
        "job_tmp_root_inode": common_staging["job_tmp_root_inode"],
        "job_tmp_root_uid": common_staging["job_tmp_root_uid"],
        "job_tmp_root_mode_octal": common_staging["job_tmp_root_mode_octal"],
        "job_tmp_root_owner_private": common_staging[
            "job_tmp_root_owner_private"
        ],
        "wrapper_mint_binding": {
            **deepcopy(common_staging["wrapper_mint"]),
            "consumed_into_live_registry": True,
            "token_or_token_digest_disclosed": False,
        },
        "runtime_cache_root": common_staging["runtime_cache_root"],
        "staging_root": common_staging["staging_root"],
        "staging_root_device": common_staging["staging_root_device"],
        "staging_root_inode": common_staging["staging_root_inode"],
        "staging_root_owner_private": common_staging[
            "staging_root_owner_private"
        ],
        "scientific_plan_sha256": value["plan_sha256"],
        "plan_binding": deepcopy(common_staging["plan"]),
        "source_binding": deepcopy(common_staging["source"]),
        "input_inventory_sha256": value["input_inventory"]["inventory_sha256"],
        "original_to_local_inputs": deepcopy(rows),
        "trainer_argument_local_paths": list(trainer_paths),
        "pre_training_rehash_sha256": pre_training_rehash,
        "authorization": {
            "live_registry_required": True,
            "single_use": True,
            "serialized_payload_authorizes_use": False,
            "shared_scratch_base_alone_authorizes_use": False,
            "minted_after_complete_live_staging_reverification": True,
        },
    }
    audit = {
        **core,
        "capability_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    capability = object.__new__(V5JobLocalInputCapability)
    capability._nonce = secrets.token_bytes(32)
    record = {
        "audit": audit,
        "verification": verification,
        "wrapper_mint_stream": mint_stream,
        "state": "minted",
        "pre_training_rehash_sha256": None,
        "post_training_rehash_sha256": None,
    }
    _reseal(capability, record)
    _LIVE_CAPABILITIES[capability] = record
    return capability


def _revalidate_record(
    record: Mapping[str, object], *, datasets: Sequence[Path], phase: str
) -> str:
    verification = record["verification"]
    environment = verification["environment"]
    _require_live_process_environment(environment)
    _validate_consumed_wrapper_mint(record)
    if rebuild_job_staging_proof(
        verification["plan"],
        verification["common"],
        environment=environment,
        allowed_root=Path(verification["allowed_root"]),
        wrapper_mint_consumed=True,
    ) != verification["common"]:
        raise RuntimeError(f"job-private source/plan identity changed {phase}")
    plan = verification["plan"]
    allowed = Path(verification["allowed_root"])
    replay = replay_v5_k1_training_chain_fingerprints(plan, allowed_root=allowed)
    live_plan = {
        **plan,
        "_live_staging_root": verification["common"]["staging_root"],
    }
    rows, trainer_paths = staged_binding_rows(
        verification["staged_inputs"],
        plan=live_plan,
        inventory=replay["inventory"],
        allowed_root=allowed,
    )
    supplied_paths = tuple(
        str(lexical_no_symlinks(Path(path), "trainer input").resolve(strict=True))
        for path in datasets
    )
    if supplied_paths != trainer_paths:
        raise RuntimeError(
            "trainer received paths other than the verified job-local argument set"
        )
    if rows != verification["binding_rows"]:
        raise RuntimeError(f"original/job-local input identity changed {phase}")
    observed = sha256(canonical_json(rows).encode("utf-8")).hexdigest()
    if observed != verification["pre_training_rehash_sha256"]:
        raise RuntimeError(f"original/job-local input bytes changed {phase}")
    return observed


def _validate_job_local_input_capability(
    datasets: Sequence[Path],
    capability: V5JobLocalInputCapability,
    *,
    phase: str,
) -> str:
    if phase not in {"pre_training", "post_training"}:
        raise ValueError("job-local capability phase is unsupported")
    record = _live_record(capability)
    expected_state = "minted" if phase == "pre_training" else "pre_validated"
    if record["state"] != expected_state:
        raise RuntimeError(
            "job-local input capability single-use state is stale, reused, or out of order"
        )
    observed = _revalidate_record(record, datasets=datasets, phase=phase)
    if phase == "pre_training":
        record["pre_training_rehash_sha256"] = observed
        record["state"] = "pre_validated"
        _reseal(capability, record)
    else:
        if observed != record["pre_training_rehash_sha256"]:
            raise RuntimeError("job-local input bytes changed across training")
        record["post_training_rehash_sha256"] = observed
        record["state"] = "post_validated"
        record["wrapper_mint_stream"].close()
        _reseal(capability, record)
    return observed


def _job_local_capability_post_validation_payload(
    capability: V5JobLocalInputCapability,
) -> dict[str, object]:
    record = _live_record(capability)
    if record["state"] != "post_validated":
        raise RuntimeError("job-local input capability has no post-training validation")
    return {
        "capability": deepcopy(record["audit"]),
        "pre_training_rehash_sha256": record["pre_training_rehash_sha256"],
        "post_training_rehash_sha256": record["post_training_rehash_sha256"],
        "pre_post_byte_identity_equal": (
            record["pre_training_rehash_sha256"]
            == record["post_training_rehash_sha256"]
        ),
    }


__all__ = [
    "V5JobLocalInputCapability",
    "V5_JOB_LOCAL_INPUT_CAPABILITY_SCHEMA",
    "V5_JOB_LOCAL_INPUT_CAPABILITY_VERSION",
]
