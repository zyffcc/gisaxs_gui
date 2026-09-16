"""Plan or submit the immutable Maxwell V5.2 K1 Phase-A dependency DAG.

The launcher performs only validation, hashing, exclusive audit publication,
and ``sbatch`` calls on max-wgs.  Training and dataset construction remain in
Slurm jobs.  Dry-run is the default and performs no writes or submissions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

sys.dont_write_bytecode = True

from .k1_phase_a_contract_v7 import (
    DEPENDENCY,
    PHASE_A_FAILURE_SCHEMA,
    PHASE_A_PLAN_SCHEMA,
    PHASE_A_PLAN_VERSION,
    PHASE_A_RECEIPT_SCHEMA,
    PHASE_A_RELEASE_SCHEMA,
    STAGES,
    self_hashed,
    validate_plan,
    validate_receipt,
    validate_release,
)
from .k1_phase_a_cross_platform_v5 import (
    inspect_v5_k1_cross_platform_reference,
)
from .package_source_snapshot_v5 import verify_extracted_source_snapshot
from .k1_staging_files_v5 import read_only_bytes_identity


V5_K1_PHASE_A_LAUNCH_SCHEMA = PHASE_A_PLAN_SCHEMA
V5_K1_PHASE_A_LAUNCH_VERSION = PHASE_A_PLAN_VERSION
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")
CURRENT_RUN_ROOT_NAME = "GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2"
_PHASE_PUBLICATION_GENERATION = "v28"
PHASE_ROOT_NAME = f"k1_phase_a_v5_2_dag_{_PHASE_PUBLICATION_GENERATION}"
_LOG_PREFIX = f"k1-phase-a-{_PHASE_PUBLICATION_GENERATION}"
PLAN_FILENAME = "launch-plan-v7.json"
RECEIPT_FILENAME = "submission-receipt-v7.json"
RELEASE_COMPLETION_FILENAME = "release-completion-v7.json"
FAILURE_AUDIT_FILENAME = "launch-failure-v7.json"
_POSTERIOR_ROOT = Path("utils/ML_Fitting_1D_GISAXS/PosteriorV8")
_REGRESSION_WRAPPER = _POSTERIOR_ROOT / "slurm/regression_cpu.sbatch"
_CROSS_PLATFORM_WRAPPER = (
    _POSTERIOR_ROOT / "slurm/v5_sobol_cross_platform_gate_cpu.sbatch"
)
_DATASET_WRAPPER = _POSTERIOR_ROOT / "slurm/v5_k1_memorization_dataset_cpu.sbatch"
_GATE_WRAPPER = _POSTERIOR_ROOT / "slurm/v5_k1_memorization_gpu.sbatch"
_REQUIRED_SOURCE = (
    _POSTERIOR_ROOT / "launch_k1_phase_a_dag_v5.py",
    _POSTERIOR_ROOT / "package_source_snapshot_v5.py",
    _POSTERIOR_ROOT / "sobol_cross_platform_contract_v5.py",
    _POSTERIOR_ROOT / "sobol_cross_platform_manifest_io_v5.py",
    _POSTERIOR_ROOT / "sobol_cross_platform_manifest_v5.py",
    _POSTERIOR_ROOT / "run_sobol_cross_platform_gate_v5.py",
    _POSTERIOR_ROOT / "k1_phase_a_cross_platform_v5.py",
    _POSTERIOR_ROOT / "k1_phase_a_dataset_binding_v5.py",
    _POSTERIOR_ROOT / "k1_phase_a_capability_v7.py",
    _POSTERIOR_ROOT / "k1_phase_a_contract_v7.py",
    _POSTERIOR_ROOT / "k1_phase_a_publication_v7.py",
    _POSTERIOR_ROOT / "k1_phase_a_runtime_v7.py",
    _POSTERIOR_ROOT / "k1_staging_files_v5.py",
    _POSTERIOR_ROOT / "build_k1_memorization_dataset_v5.py",
    _POSTERIOR_ROOT / "run_k1_memorization_gate_v5.py",
    _POSTERIOR_ROOT / "memorization_gate_v5.py",
    _POSTERIOR_ROOT / "model_v5.py",
    _POSTERIOR_ROOT / "model_v5_contract.py",
    _POSTERIOR_ROOT / "training_objective_v5.py",
    _POSTERIOR_ROOT / "study_protocol.py",
    _POSTERIOR_ROOT / "amplitude_query_sampling_v5.py",
    _POSTERIOR_ROOT / "amplitude_query_v5.py",
    _POSTERIOR_ROOT / "bounds_query_v5.py",
    _POSTERIOR_ROOT / "grouped_amplitude_join_v5.py",
    _POSTERIOR_ROOT / "sobol_amplitude_recipe_v5.py",
    _POSTERIOR_ROOT / "sobol_geometry_recipe_v5.py",
    _POSTERIOR_ROOT / "sobol_numeric_canonicalization_v5.py",
    _POSTERIOR_ROOT / "sobol_recipe_coordinates_v5.py",
    _POSTERIOR_ROOT / "sobol_recipe_physics_v5.py",
    _POSTERIOR_ROOT / "sobol_recipe_v5.py",
    _POSTERIOR_ROOT / "sobol_universal_query_design_v5.py",
    _REGRESSION_WRAPPER,
    _CROSS_PLATFORM_WRAPPER,
    _DATASET_WRAPPER,
    _GATE_WRAPPER,
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_amplitude_query_sampling_v5.py"),
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_sobol_recipe_v5.py"),
    Path(
        "utils/ML_Fitting_1D_GISAXS/tests/"
        "test_posterior_v8_sobol_universal_query_design_v5.py"
    ),
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_study_protocol.py"),
    Path(
        "utils/ML_Fitting_1D_GISAXS/tests/"
        "test_posterior_v8_sobol_cross_platform_manifest_v5.py"
    ),
    Path(
        "utils/ML_Fitting_1D_GISAXS/tests/"
        "test_posterior_v8_k1_phase_a_artifact_binding_v5.py"
    ),
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_dataset_gate_worker_v5.py"),
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_a_launcher_v5.py"),
    Path("utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_a_runtime_v7.py"),
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PARSABLE_JOB_ID = re.compile(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\Z")
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")
_STAGES = STAGES


@dataclass(frozen=True)
class V5K1PhaseALaunchConfig:
    source_root: Path
    source_archive_path: Path
    run_root: Path
    source_archive_sha256: str
    cross_platform_reference_path: Path
    cross_platform_reference_sha256: str


@dataclass(frozen=True)
class V5CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class V5CommandRequest:
    argv: tuple[str, ...]
    script_bytes: bytes | None = None
    script_sha256: str | None = None

    def __iter__(self):
        return iter(self.argv)

    def __len__(self) -> int:
        return len(self.argv)

    def __getitem__(self, index):
        return self.argv[index]


CommandRunner = Callable[[V5CommandRequest], V5CommandResult]
AuditWriter = Callable[[Path, Mapping[str, object]], None]


class V5K1PhaseALaunchError(RuntimeError):
    """Submission failed after its exclusive audit root was reserved."""

    def __init__(self, message: str, *, receipt_path: Path, failure_path: Path) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path
        self.failure_path = failure_path


def _file_identity(path: Path) -> dict[str, object]:
    _, identity = read_only_bytes_identity(path, f"Phase-A source file {path}")
    return identity


def _lexical_absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("Maxwell paths must be absolute")
    return Path(os.path.abspath(path))


def _under_dust(path: Path, allowed_root: Path, name: str, *, must_exist: bool) -> Path:
    root = allowed_root.resolve(strict=True)
    lexical = _lexical_absolute(path)
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {root}") from exc
    if not relative.parts:
        raise ValueError(f"{name} must not be the dust root itself")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
        if not current.exists():
            break
    resolved = lexical.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} resolves outside {root}")
    return resolved


def _source_identity(
    source_root: Path,
    source_archive_path: Path,
    source_archive_sha256: str,
    allowed_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, object]:
    root = _under_dust(source_root, allowed_root, "source_root", must_exist=True)
    archive = _under_dust(
        source_archive_path,
        allowed_root,
        "source_archive_path",
        must_exist=True,
    )
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("source_archive_path must be a real source archive")
    archive_sha256 = _archive_digest(source_archive_sha256)
    identity = verify_extracted_source_snapshot(
        archive,
        root,
        expected_archive_sha256=archive_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_tree_sha256=expected_source_tree_sha256,
    )
    required_identity: dict[str, object] = {}
    for relative in _REQUIRED_SOURCE:
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(f"missing immutable required source: {target}")
        required_identity[relative.as_posix()] = _file_identity(target)
    return {
        **identity,
        "snapshot_name": root.name,
        "required_file_identity": required_identity,
    }


def _reference_identity(
    reference_path: Path,
    reference_sha256: str,
    source: Mapping[str, object],
    allowed_root: Path,
) -> dict[str, object]:
    reference = _under_dust(
        reference_path,
        allowed_root,
        "cross_platform_reference_path",
        must_exist=True,
    )
    expected_source = {
        "source_archive_sha256": source["archive_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_tree_sha256": source["source_tree_sha256"],
    }
    binding, file_state = inspect_v5_k1_cross_platform_reference(
        reference,
        expected_file_sha256=_digest(
            reference_sha256, "cross_platform_reference_sha256"
        ),
        expected_source=expected_source,
        require_read_only=True,
    )
    return {
        **binding.audit_payload(),
        "file_identity": file_state,
        "verification": (
            "strict_manifest_replay_external_file_hash_and_source_snapshot_binding"
        ),
    }


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _archive_digest(value: str) -> str:
    return _digest(value, "source_archive_sha256")


def _safe_export_value(value: object, name: str) -> str:
    result = str(value)
    if any(token in result for token in (",", "\n", "\r", "\0")):
        raise ValueError(f"{name} cannot be represented in the Slurm export contract")
    return result


def _export_argument(environment: Mapping[str, object]) -> str:
    assignments = [
        f"{name}={_safe_export_value(value, name)}" for name, value in environment.items()
    ]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _job_command(
    job: Mapping[str, object],
    dependency_job_id: str | None,
    *,
    transaction_environment: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    command = ["sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes"]
    if dependency_job_id is not None:
        command.append(f"--dependency=afterok:{dependency_job_id}")
    command.extend(
        (
            f"--job-name={job['job_name']}",
            f"--output={job['stdout']}",
            f"--error={job['stderr']}",
            _export_argument(
                {
                    **job["environment"],
                    **({} if transaction_environment is None else transaction_environment),
                }
            ),
        )
    )
    return tuple(command)


def build_v5_k1_phase_a_launch_plan(
    config: V5K1PhaseALaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate immutable inputs and return a deterministic, write-free DAG."""

    if not isinstance(config, V5K1PhaseALaunchConfig):
        raise TypeError("config must be V5K1PhaseALaunchConfig")
    source = _source_identity(
        config.source_root,
        config.source_archive_path,
        config.source_archive_sha256,
        allowed_root,
    )
    run_root = _under_dust(config.run_root, allowed_root, "run_root", must_exist=True)
    expected = (allowed_root.resolve(strict=True) / "MaxwellRuns" / CURRENT_RUN_ROOT_NAME).resolve(
        strict=True
    )
    if run_root != expected or not run_root.is_dir():
        raise ValueError(f"run_root must be exactly {expected}")
    logs = run_root / "logs"
    if not logs.is_dir() or logs.is_symlink():
        raise FileNotFoundError("the current versioned run root must contain a real logs directory")
    reference = _reference_identity(
        config.cross_platform_reference_path,
        config.cross_platform_reference_sha256,
        source,
        allowed_root,
    )
    source_root = Path(str(source["source_root"]))
    source_archive = Path(str(source["archive_path"]))
    reference_path = Path(str(reference["path"]))
    if (
        source_root == run_root
        or run_root.is_relative_to(source_root)
        or source_archive == run_root
        or source_archive.is_relative_to(run_root)
        or reference_path == run_root
        or reference_path.is_relative_to(run_root)
        or reference_path == source_root
        or reference_path.is_relative_to(source_root)
        or reference_path == source_archive
    ):
        raise ValueError("source, reference, and writable run outputs must be separate")

    phase_root = run_root / PHASE_ROOT_NAME
    data_root = phase_root / "datasets"
    model_root = phase_root / "models"
    audit_root = phase_root / "audit"
    layout = {
        "phase_root": str(phase_root),
        "data_root": str(data_root),
        "model_root": str(model_root),
        "audit_root": str(audit_root),
        "plan": str(audit_root / PLAN_FILENAME),
        "receipt": str(audit_root / RECEIPT_FILENAME),
        "release_completion": str(audit_root / RELEASE_COMPLETION_FILENAME),
        "failure_audit": str(audit_root / FAILURE_AUDIT_FILENAME),
        "cross_platform_candidate": str(
            audit_root / "sobol-cross-platform-linux-candidate-v2.json"
        ),
        "cross_platform_result": str(
            audit_root / "sobol-cross-platform-gate-result-v1.json"
        ),
        "cross_platform_pass_marker": str(
            audit_root / "sobol-cross-platform-PASS-v1.json"
        ),
        "smoke_dataset": str(data_root / "k1-v5-2-phase-a-v5-smoke-r2-sphere-v0.gvd5"),
        "smoke_dataset_binding": str(
            data_root / "k1-v5-2-phase-a-v5-smoke-r2-sphere-v0.gvd5.binding-v1.json"
        ),
        "full_dataset": str(data_root / "k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5"),
        "full_dataset_binding": str(
            data_root / "k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5.binding-v1.json"
        ),
        "smoke_model": str(model_root / "k1-v5-2-phase-a-v5-smoke-steps2"),
        "full_model": str(model_root / "k1-v5-2-phase-a-v5-full-steps18000"),
        "logs": str(logs),
    }
    for stage in _STAGES:
        layout[f"{stage}_completion"] = str(
            audit_root / f"{stage.replace('_', '-')}-completion-v1.json"
        )
    targets = tuple(Path(value) for name, value in layout.items() if name != "logs")
    if phase_root.exists() or phase_root.is_symlink() or any(path.exists() for path in targets):
        raise FileExistsError("refusing to reuse a K1 Phase-A target, receipt, or job manifest")
    if any(logs.glob(f"{_LOG_PREFIX}-*.out")) or any(
        logs.glob(f"{_LOG_PREFIX}-*.err")
    ):
        raise FileExistsError("refusing to reuse existing K1 Phase-A Slurm logs")

    wrapper = {
        "regression": source_root / _REGRESSION_WRAPPER,
        "cross_platform_gate": source_root / _CROSS_PLATFORM_WRAPPER,
        "dataset": source_root / _DATASET_WRAPPER,
        "gate": source_root / _GATE_WRAPPER,
    }
    smoke_dataset, full_dataset = layout["smoke_dataset"], layout["full_dataset"]
    source_environment = {
        "POSTERIOR_V8_SOURCE_ROOT": str(source_root),
        "POSTERIOR_V8_V5_SOURCE_ARCHIVE": str(source_archive),
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256": source["archive_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256": source["manifest_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256": source["source_tree_sha256"],
    }
    cross_platform_environment = {
        "POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER": layout[
            "cross_platform_pass_marker"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256": reference[
            "file_sha256"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT": reference[
            "file_byte_count"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256": reference[
            "manifest_sha256"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256": reference[
            "scientific_content_sha256"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256": reference[
            "comparison_result_sha256"
        ],
        "POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256": reference[
            "gate_claim_sha256"
        ],
    }
    downstream_environment = {**source_environment, **cross_platform_environment}
    jobs = {
        "regression": {
            "job_name": "gisaxs-v5-2-k1-v5-regression",
            "wrapper": str(wrapper["regression"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-regression-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-regression-%j.err"),
            "environment": dict(source_environment),
            "depends_on": None,
        },
        "cross_platform_gate": {
            "job_name": "gisaxs-v5-2-k1-v5-cross-platform",
            "wrapper": str(wrapper["cross_platform_gate"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-cross-platform-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-cross-platform-%j.err"),
            "environment": {
                **source_environment,
                **cross_platform_environment,
                "POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE": str(reference_path),
                "POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT": layout[
                    "cross_platform_candidate"
                ],
                "POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT": layout[
                    "cross_platform_result"
                ],
            },
            "depends_on": "regression",
        },
        "smoke_dataset": {
            "job_name": "gisaxs-v5-2-k1-v5-smoke-data",
            "wrapper": str(wrapper["dataset"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-smoke-dataset-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-smoke-dataset-%j.err"),
            "environment": {
                **downstream_environment,
                "POSTERIOR_V8_V5_K1_RECIPE_COUNT": 2,
                "POSTERIOR_V8_V5_K1_TOPOLOGY": "sphere",
                "POSTERIOR_V8_V5_K1_BASE_SEED": 20260903,
                "POSTERIOR_V8_V5_K1_VIEW_INDICES": "0",
                "POSTERIOR_V8_V5_K1_PATTERN_ID": 0,
                "POSTERIOR_V8_V5_K1_DATASET_OUTPUT": smoke_dataset,
                "POSTERIOR_V8_V5_K1_DATASET_BINDING": layout[
                    "smoke_dataset_binding"
                ],
            },
            "depends_on": "cross_platform_gate",
        },
        "smoke_gate": {
            "job_name": "gisaxs-v5-2-k1-v5-smoke-gate",
            "wrapper": str(wrapper["gate"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-smoke-gate-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-smoke-gate-%j.err"),
            "environment": {
                **downstream_environment,
                "POSTERIOR_V8_V5_K1_DATASET": smoke_dataset,
                "POSTERIOR_V8_V5_K1_DATASET_BINDING": layout[
                    "smoke_dataset_binding"
                ],
                "POSTERIOR_V8_V5_K1_GATE_OUTPUT": layout["smoke_model"],
                "POSTERIOR_V8_V5_K1_STEPS": 2,
                "POSTERIOR_V8_V5_K1_LEARNING_RATE": 0.003,
                "POSTERIOR_V8_V5_K1_FINAL_LEARNING_RATE": 0.00003,
                "POSTERIOR_V8_V5_K1_LEARNING_RATE_SCHEDULE": "cosine_decay",
                "POSTERIOR_V8_V5_K1_LOCAL_MDN_WEIGHT": 1.0,
                "POSTERIOR_V8_V5_K1_LOCAL_COVERAGE_WEIGHT": 100.0,
                "POSTERIOR_V8_V5_K1_OPERATIONAL_TOP_L_ALIGNMENT_WEIGHT": 1.0,
                "POSTERIOR_V8_V5_K1_WIDTH": 32,
                "POSTERIOR_V8_V5_K1_ENCODER_BLOCKS": 1,
                "POSTERIOR_V8_V5_K1_SMOKE": 1,
            },
            "depends_on": "smoke_dataset",
        },
        "full_dataset": {
            "job_name": "gisaxs-v5-2-k1-v5-full-data",
            "wrapper": str(wrapper["dataset"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-full-dataset-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-full-dataset-%j.err"),
            "environment": {
                **downstream_environment,
                "POSTERIOR_V8_V5_K1_RECIPE_COUNT": 512,
                "POSTERIOR_V8_V5_K1_TOPOLOGY": "sphere",
                "POSTERIOR_V8_V5_K1_BASE_SEED": 20260903,
                "POSTERIOR_V8_V5_K1_VIEW_INDICES": "0",
                "POSTERIOR_V8_V5_K1_PATTERN_ID": 0,
                "POSTERIOR_V8_V5_K1_DATASET_OUTPUT": full_dataset,
                "POSTERIOR_V8_V5_K1_DATASET_BINDING": layout[
                    "full_dataset_binding"
                ],
            },
            "depends_on": "smoke_gate",
        },
        "full_gate": {
            "job_name": "gisaxs-v5-2-k1-v5-full-gate",
            "wrapper": str(wrapper["gate"]),
            "stdout": str(logs / f"{_LOG_PREFIX}-full-gate-%j.out"),
            "stderr": str(logs / f"{_LOG_PREFIX}-full-gate-%j.err"),
            "environment": {
                **downstream_environment,
                "POSTERIOR_V8_V5_K1_DATASET": full_dataset,
                "POSTERIOR_V8_V5_K1_DATASET_BINDING": layout[
                    "full_dataset_binding"
                ],
                "POSTERIOR_V8_V5_K1_GATE_OUTPUT": layout["full_model"],
                "POSTERIOR_V8_V5_K1_STEPS": 18000,
                "POSTERIOR_V8_V5_K1_LEARNING_RATE": 0.003,
                "POSTERIOR_V8_V5_K1_FINAL_LEARNING_RATE": 0.00003,
                "POSTERIOR_V8_V5_K1_LEARNING_RATE_SCHEDULE": "cosine_decay",
                "POSTERIOR_V8_V5_K1_LOCAL_MDN_WEIGHT": 1.0,
                "POSTERIOR_V8_V5_K1_LOCAL_COVERAGE_WEIGHT": 100.0,
                "POSTERIOR_V8_V5_K1_OPERATIONAL_TOP_L_ALIGNMENT_WEIGHT": 1.0,
                "POSTERIOR_V8_V5_K1_WIDTH": 128,
                "POSTERIOR_V8_V5_K1_ENCODER_BLOCKS": 6,
            },
            "depends_on": "full_dataset",
        },
    }
    preview_ids = {stage: f"{stage.upper()}_JOB_ID" for stage in _STAGES}
    preview_transaction = {
        "POSTERIOR_V8_V7_PHASE_A_PLAN": layout["plan"],
        "POSTERIOR_V8_V7_PHASE_A_PLAN_SHA256": "PLAN_SHA256",
        "POSTERIOR_V8_V7_PHASE_A_RECEIPT": layout["receipt"],
        "POSTERIOR_V8_V7_PHASE_A_RELEASE": layout["release_completion"],
        "POSTERIOR_V8_V7_PHASE_A_STAGE": "STAGE",
        "POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION": "UPSTREAM_COMPLETION_OR_NONE",
        "POSTERIOR_V8_V7_PHASE_A_COMPLETION": "STAGE_COMPLETION",
    }
    commands = {
        stage: list(
            _job_command(
                jobs[stage],
                None
                if jobs[stage]["depends_on"] is None
                else preview_ids[jobs[stage]["depends_on"]],
                transaction_environment={
                    **preview_transaction,
                    "POSTERIOR_V8_V7_PHASE_A_STAGE": stage,
                    "POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION": (
                        "NONE"
                        if DEPENDENCY[stage] is None
                        else layout[f"{DEPENDENCY[stage]}_completion"]
                    ),
                    "POSTERIOR_V8_V7_PHASE_A_COMPLETION": layout[f"{stage}_completion"],
                },
            )
        )
        for stage in _STAGES
    }
    core = {
        "schema_version": V5_K1_PHASE_A_LAUNCH_SCHEMA,
        "version": V5_K1_PHASE_A_LAUNCH_VERSION,
        "scientific_role": "phase_a_single_branch_capacity_and_wiring_not_model_acceptance",
        "phase_a_scope": {
            "topology": ["sphere"],
            "branch_pattern_id": 0,
            "single_branch_capacity_diagnostic": True,
            "balanced_all_k1_topologies_and_legal_branches": False,
        },
        "protocol_k1_memorization_stage_complete": False,
        "full_k1_all_legal_branches_pending_fail_closed": True,
        "pending_balanced_k1_scope": {
            "topologies": ["sphere", "cylinder", "vertical_cylinder"],
            "legal_d_resolution_patterns_per_topology": 4,
            "total_topology_pattern_branches": 12,
        },
        "phase_b_status": "not_part_of_this_launcher_fail_closed_pending",
        "run_root": str(run_root),
        "source": source,
        "source_archive": {
            "path": source["archive_path"],
            "sha256": source["archive_sha256"],
            "byte_count": source["archive_byte_count"],
            "manifest_sha256": source["manifest_sha256"],
            "source_tree_sha256": source["source_tree_sha256"],
            "verification": "archive_manifest_and_exact_read_only_extracted_tree_recomputed",
        },
        "cross_platform_reference": reference,
        "wrapper_identity": {
            name: source["required_file_identity"][path.as_posix()]
            for name, path in (
                ("regression", _REGRESSION_WRAPPER),
                ("cross_platform_gate", _CROSS_PLATFORM_WRAPPER),
                ("dataset", _DATASET_WRAPPER),
                ("gate", _GATE_WRAPPER),
            )
        },
        "layout": layout,
        "stage_order": list(_STAGES),
        "jobs": jobs,
        "submission_preview": commands,
        "login_node_heavy_compute_allowed": False,
    }
    return {**core, "plan_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite launch audit: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    raw, identity = read_only_bytes_identity(path, f"published Phase-A audit {path.name}")
    if raw != encoded or identity["mode_octal"] != "0400" or identity["link_count"] != 1:
        raise RuntimeError(f"published Phase-A audit identity failed: {path}")


def _default_runner(request: V5CommandRequest) -> V5CommandResult:
    completed = subprocess.run(  # noqa: S603
        request.argv,
        input=request.script_bytes,
        check=False,
        capture_output=True,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    return V5CommandResult(completed.returncode, stdout, stderr)


def _job_id(result: V5CommandResult, stage: str) -> str:
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} submission failed with return code {result.returncode}; "
            "scheduler output is retained only by digest"
        )
    match = _PARSABLE_JOB_ID.fullmatch(result.stdout.strip())
    if match is None:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job id")
    return match.group(1)


def _text_identity(value: str) -> dict[str, object]:
    encoded = value.encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "byte_count": len(encoded)}


def _scheduler_snapshot(
    result: V5CommandResult,
    *,
    stage: str,
    expected_job_id: str,
    expected_dependency_job_id: str | None,
    expected_held: bool,
) -> dict[str, object]:
    """Validate the scheduler's own state instead of trusting submitted argv."""

    if not isinstance(result, V5CommandResult) or result.returncode != 0:
        raise RuntimeError(f"could not read back Slurm state for {stage}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Slurm returned an ambiguous state for {stage}")
    fields: dict[str, str] = {}
    for token in shlex.split(lines[0]):
        if "=" in token:
            name, value = token.split("=", 1)
            fields[name] = value
    if fields.get("JobId") != expected_job_id:
        raise RuntimeError(f"Slurm state job id drifted for {stage}")
    state = fields.get("JobState")
    reason = fields.get("Reason")
    dependency = fields.get("Dependency", "")
    normalized_dependency = (
        None if dependency in {"", "(null)", "None"} else dependency
    )
    if expected_held:
        if state != "PENDING" or reason != "JobHeldUser":
            raise RuntimeError(f"Slurm did not retain {stage} in a user hold")
    elif state not in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "COMPLETED"}:
        raise RuntimeError(f"Slurm returned an unsupported released state for {stage}")
    elif reason == "JobHeldUser":
        raise RuntimeError(f"Slurm still reports {stage} as user-held after release")
    if expected_dependency_job_id is None:
        if normalized_dependency is not None:
            raise RuntimeError(f"Slurm attached an unexpected dependency to {stage}")
    elif not str(normalized_dependency).startswith(
        f"afterok:{expected_dependency_job_id}"
    ):
        raise RuntimeError(f"Slurm dependency drifted for {stage}")
    return {
        "job_id": expected_job_id,
        "job_state": state,
        "reason": reason,
        "dependency": normalized_dependency,
        "response": _text_identity(result.stdout),
    }


def launch_v5_k1_phase_a_dag(
    config: V5K1PhaseALaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    receipt_writer: AuditWriter | None = None,
) -> dict[str, object]:
    """Dry-run by default; transactionally submit and release the v7 DAG."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    if allowed_root == MAXWELL_DUST_ROOT and any(
        value is not None for value in (runner, hostname, receipt_writer)
    ):
        raise RuntimeError("production Maxwell submission forbids injected test seams")
    plan = build_v5_k1_phase_a_launch_plan(config, allowed_root=allowed_root)
    if not submit:
        return {"status": "dry_run", "writes_performed": False, "plan": plan}
    host = socket.gethostname() if hostname is None else hostname
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 Phase-A DAG submission is restricted to max-wgs")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("K1 Phase-A launcher must not run inside a Slurm allocation")

    layout = plan["layout"]
    phase_root = Path(layout["phase_root"])
    phase_root.mkdir(mode=0o700, exist_ok=False)
    for name in ("data_root", "model_root", "audit_root"):
        Path(layout[name]).mkdir(mode=0o700, exist_ok=False)
    plan_path = Path(layout["plan"])
    receipt_path = Path(layout["receipt"])
    release_path = Path(layout["release_completion"])
    failure_path = Path(layout["failure_audit"])
    _write_json_exclusive(plan_path, plan)

    command_runner = _default_runner if runner is None else runner
    attempts: list[dict[str, object]] = []
    release_attempts: list[dict[str, object]] = []
    held_scheduler_snapshots: dict[str, dict[str, object]] = {}
    cancellation_attempts: list[dict[str, object]] = []
    job_ids: dict[str, str] = {}
    receipt: dict[str, object] | None = None
    stage = _STAGES[0]
    try:
        for stage in _STAGES:
            replay = _source_identity(
                config.source_root,
                config.source_archive_path,
                config.source_archive_sha256,
                allowed_root,
                expected_manifest_sha256=str(plan["source"]["manifest_sha256"]),
                expected_source_tree_sha256=str(plan["source"]["source_tree_sha256"]),
            )
            if replay != plan["source"]:
                raise RuntimeError("immutable source snapshot changed during DAG submission")
            reference_replay = _reference_identity(
                config.cross_platform_reference_path,
                config.cross_platform_reference_sha256,
                replay,
                allowed_root,
            )
            if reference_replay != plan["cross_platform_reference"]:
                raise RuntimeError(
                    "immutable cross-platform reference changed during DAG submission"
                )
            dependency = plan["jobs"][stage]["depends_on"]
            command = _job_command(
                plan["jobs"][stage],
                None if dependency is None else job_ids[dependency],
                transaction_environment={
                    "POSTERIOR_V8_V7_PHASE_A_PLAN": str(plan_path),
                    "POSTERIOR_V8_V7_PHASE_A_PLAN_SHA256": plan["plan_sha256"],
                    "POSTERIOR_V8_V7_PHASE_A_RECEIPT": str(receipt_path),
                    "POSTERIOR_V8_V7_PHASE_A_RELEASE": str(release_path),
                    "POSTERIOR_V8_V7_PHASE_A_STAGE": stage,
                    "POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION": (
                        "NONE" if dependency is None else layout[f"{dependency}_completion"]
                    ),
                    "POSTERIOR_V8_V7_PHASE_A_COMPLETION": layout[f"{stage}_completion"],
                },
            )
            wrapper_path = Path(str(plan["jobs"][stage]["wrapper"]))
            script_bytes, script_identity = read_only_bytes_identity(
                wrapper_path, f"Phase-A {stage} wrapper"
            )
            expected_wrapper = plan["wrapper_identity"][
                "dataset"
                if stage in {"smoke_dataset", "full_dataset"}
                else "gate"
                if stage in {"smoke_gate", "full_gate"}
                else stage
            ]
            if script_identity != expected_wrapper:
                raise RuntimeError("authenticated Phase-A wrapper changed before stdin submission")
            request = V5CommandRequest(
                argv=command,
                script_bytes=script_bytes,
                script_sha256=str(script_identity["sha256"]),
            )
            attempt = {
                "stage": stage,
                "argv": list(command),
                "script_identity": script_identity,
                "script_bytes_recorded": False,
            }
            attempts.append(attempt)
            result = command_runner(request)
            if isinstance(result, V5CommandResult):
                attempt.update(
                    {
                        "returncode": result.returncode,
                        "stdout": _text_identity(result.stdout),
                        "stderr": _text_identity(result.stderr),
                    }
                )
            job_ids[stage] = _job_id(result, stage)
            dependency_job_id = None if dependency is None else job_ids[dependency]
            scheduler_request = V5CommandRequest(
                argv=("scontrol", "show", "job", "--oneliner", job_ids[stage])
            )
            scheduler_result = command_runner(scheduler_request)
            held_scheduler_snapshots[stage] = _scheduler_snapshot(
                scheduler_result,
                stage=stage,
                expected_job_id=job_ids[stage],
                expected_dependency_job_id=dependency_job_id,
                expected_held=True,
            )

        receipt_core = {
            "schema": PHASE_A_RECEIPT_SCHEMA,
            "status": "ALL_JOBS_HELD",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_path": str(plan_path),
            "receipt_path": str(receipt_path),
            "job_ids": dict(job_ids),
            "dependency_edges": [
                [DEPENDENCY[item], item, job_ids[DEPENDENCY[item]], job_ids[item]]
                for item in _STAGES
                if DEPENDENCY[item] is not None
            ],
            "submission_attempts": attempts,
            "held_scheduler_snapshots": held_scheduler_snapshots,
            "all_jobs_submitted_held": True,
            "secret_environment_captured": False,
        }
        receipt = self_hashed(receipt_core, "receipt_sha256")
        validate_receipt(receipt, plan)
        selected_writer = _write_json_exclusive if receipt_writer is None else receipt_writer
        selected_writer(receipt_path, receipt)

        for released_stage in reversed(_STAGES):
            request = V5CommandRequest(
                argv=("scontrol", "release", job_ids[released_stage])
            )
            result = command_runner(request)
            dependency = DEPENDENCY[released_stage]
            scheduler_request = V5CommandRequest(
                argv=("scontrol", "show", "job", "--oneliner", job_ids[released_stage])
            )
            scheduler_result = command_runner(scheduler_request)
            release_attempt = {
                "stage": released_stage,
                "job_id": job_ids[released_stage],
                "argv": list(request.argv),
                "returncode": result.returncode,
                "stdout": _text_identity(result.stdout),
                "stderr": _text_identity(result.stderr),
                "scheduler_snapshot": _scheduler_snapshot(
                    scheduler_result,
                    stage=released_stage,
                    expected_job_id=job_ids[released_stage],
                    expected_dependency_job_id=(
                        None if dependency is None else job_ids[dependency]
                    ),
                    expected_held=False,
                ),
            }
            release_attempts.append(release_attempt)
            if result.returncode != 0:
                raise RuntimeError(f"release failed for {released_stage}")

        release_core = {
            "schema": PHASE_A_RELEASE_SCHEMA,
            "status": "ALL_JOBS_RELEASED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "release_order": list(reversed(_STAGES)),
            "released_job_ids": [job_ids[item] for item in reversed(_STAGES)],
            "release_attempts": release_attempts,
        }
        release = self_hashed(release_core, "release_sha256")
        validate_release(release, plan, receipt)
        _write_json_exclusive(release_path, release)
        return {
            "status": "released",
            "writes_performed": True,
            "plan": plan,
            "submission_receipt": receipt,
            "release_completion": release,
        }
    except (Exception, KeyboardInterrupt) as exc:
        for cancelled_stage in reversed(tuple(job_ids)):
            request = V5CommandRequest(argv=("scancel", job_ids[cancelled_stage]))
            try:
                result = command_runner(request)
                cancellation_attempts.append(
                    {
                        "stage": cancelled_stage,
                        "job_id": job_ids[cancelled_stage],
                        "argv": list(request.argv),
                        "returncode": result.returncode,
                        "stdout": _text_identity(result.stdout),
                        "stderr": _text_identity(result.stderr),
                    }
                )
            except BaseException as cancel_exc:  # best effort must continue
                cancellation_attempts.append(
                    {
                        "stage": cancelled_stage,
                        "job_id": job_ids[cancelled_stage],
                        "argv": list(request.argv),
                        "exception_type": type(cancel_exc).__name__,
                        "exception_message": str(cancel_exc)[:2000],
                    }
                )
        failure_core = {
            "schema": PHASE_A_FAILURE_SCHEMA,
            "status": "LAUNCH_FAILED_NO_RELEASE_COMPLETION",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "failed_stage": stage,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:2000],
            "known_job_ids": dict(job_ids),
            "submission_attempts": attempts,
            "release_attempts": release_attempts,
            "cancellation_attempts": cancellation_attempts,
            "submission_receipt_published": receipt_path.is_file(),
            "release_completion_published": False,
        }
        failure = self_hashed(failure_core, "failure_sha256")
        try:
            _write_json_exclusive(failure_path, failure)
        except Exception as audit_exc:
            raise V5K1PhaseALaunchError(
                f"K1 Phase-A launch failed and failure audit publication failed: {audit_exc}",
                receipt_path=receipt_path,
                failure_path=failure_path,
            ) from exc
        raise V5K1PhaseALaunchError(
            f"K1 Phase-A DAG launch failed during {stage}; failure audit preserved",
            receipt_path=receipt_path,
            failure_path=failure_path,
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--cross-platform-reference", required=True, type=Path)
    parser.add_argument("--cross-platform-reference-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = launch_v5_k1_phase_a_dag(
        V5K1PhaseALaunchConfig(
            source_root=args.source_root,
            source_archive_path=args.source_archive,
            run_root=args.run_root,
            source_archive_sha256=args.source_archive_sha256,
            cross_platform_reference_path=args.cross_platform_reference,
            cross_platform_reference_sha256=args.cross_platform_reference_sha256,
        ),
        submit=args.submit and not args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CURRENT_RUN_ROOT_NAME",
    "MAXWELL_DUST_ROOT",
    "PHASE_ROOT_NAME",
    "PLAN_FILENAME",
    "RECEIPT_FILENAME",
    "V5CommandResult",
    "V5CommandRequest",
    "V5K1PhaseALaunchConfig",
    "V5K1PhaseALaunchError",
    "V5_K1_PHASE_A_LAUNCH_SCHEMA",
    "V5_K1_PHASE_A_LAUNCH_VERSION",
    "build_v5_k1_phase_a_launch_plan",
    "launch_v5_k1_phase_a_dag",
    "main",
]
