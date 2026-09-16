"""Worker-only, write-free replay of a sealed all-K1 search collection."""

from pathlib import Path
import os
import socket

from . import k1_balanced_full_search_collector_v5 as collector
from .grouped_artifact_v5 import canonical_json
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_identity, read_only_json
from .k1_balanced_dataset_launch_plan_v5 import (
    replay_v5_k1_balanced_dataset_launch_inputs, validate_v5_k1_balanced_dataset_launch_plan,
)
from .k1_balanced_full_search_worker_v5 import _task_input_identity_after_source_replay


def _source_input_identity(*, balanced_dataset_launch_plan_path=None, **arguments):
    if balanced_dataset_launch_plan_path is None:
        return collector._source_input_identity(**arguments)
    plan = arguments["plan"]
    path = lexical_no_symlinks(Path(balanced_dataset_launch_plan_path), "producer launch plan").resolve(strict=True)
    if not path.is_relative_to(collector.MAXWELL_DUST_ROOT):
        raise ValueError("producer launch plan escaped the Maxwell dust root")
    payload, identity = read_only_json(path, "producer launch plan")
    if (identity["mode_octal"] != "0400"
            or identity["sha256"] != plan["balanced_dataset"]["launch_plan_file_sha256"]):
        raise ValueError("producer launch-plan file differs from the frozen search binding")
    producer = validate_v5_k1_balanced_dataset_launch_plan(payload)
    source = producer["source"]
    if (producer["plan_sha256"] != plan["balanced_dataset"]["launch_plan_sha256"]
            or str(path) != producer["layout"]["plan"]
            or source["root"] != str(arguments["source_root"])
            or source["archive_path"] != str(arguments["source_archive"])
            or source["archive_sha256"] != plan["source"]["archive_sha256"]
            or source["bundle_sha256"] != plan["source"]["bundle_sha256"]):
        raise ValueError("producer source identity differs from the frozen search binding")
    # This replays the producer's sealed file inventory AND its complete archive
    # tree. It does not invent a new bundle hash using the consumer's file list.
    replay_v5_k1_balanced_dataset_launch_inputs(producer, allowed_root=collector.MAXWELL_DUST_ROOT)
    rows = tuple(_task_input_identity_after_source_replay(
        plan_path=arguments["plan_path"], source_archive=arguments["source_archive"],
        schedule_path=arguments["schedule_path"], calibration_path=arguments["calibration_path"],
        selected=selected, plan=plan, source_bundle_sha256=source["bundle_sha256"],
    ) for selected in plan["parents"])
    if read_only_identity(path, "producer launch plan") != identity:
        raise RuntimeError("producer launch plan changed during historical source replay")
    return rows


def _publication(path, expected_file_sha256, hash_field):
    value, identity = read_only_json(path, "full-search publication")
    if identity["mode_octal"] != "0400" or identity["sha256"] != expected_file_sha256:
        raise ValueError("full-search publication mode or planned file SHA differs")
    collector._self_hashed(value, field=hash_field, name="full-search publication")
    return value, identity


def replay_v5_k1_balanced_full_search_collection(
    plan_path, *, expected_plan_sha256, expected_plan_file_sha256,
    expected_inventory_file_sha256, expected_completion_file_sha256,
    source_root, source_archive, local_sobol_schedule_path, calibration_path,
    balanced_dataset_launch_plan_path=None,
):
    """Recheck all task evidence without publishing files or granting gradients.

    Paths and file digests must come from the caller's frozen launch binding.
    This is computational replay, so a login-node dry-run escape is not offered.
    """
    collector._worker_guard(dry_run=False, hostname=socket.gethostname(), environment=os.environ)
    producer_identity = None
    if balanced_dataset_launch_plan_path is not None:
        balanced_dataset_launch_plan_path = lexical_no_symlinks(
            Path(balanced_dataset_launch_plan_path), "producer launch plan",
        ).resolve(strict=True)
        if not balanced_dataset_launch_plan_path.is_relative_to(collector.MAXWELL_DUST_ROOT):
            raise ValueError("producer launch plan escaped the Maxwell dust root")
        producer_identity = read_only_identity(balanced_dataset_launch_plan_path, "producer launch plan")
    paths = {
        "plan": Path(plan_path), "source_root": Path(source_root),
        "source_archive": Path(source_archive), "schedule": Path(local_sobol_schedule_path),
        "calibration": Path(calibration_path),
    }
    for name, path in paths.items():
        paths[name] = lexical_no_symlinks(path, name).resolve(strict=True)
        if not paths[name].is_relative_to(collector.MAXWELL_DUST_ROOT):
            raise ValueError("search collection inputs must remain under the Maxwell dust root")
    plan = collector.read_v5_k1_balanced_full_search_plan_file(
        paths["plan"], expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
    )
    completion_path = Path(plan["layout"]["completion"])
    inventory_path = completion_path.parent / collector.V5_K1_BALANCED_FULL_SEARCH_INVENTORY_FILENAME
    for path in (completion_path, inventory_path):
        checked = lexical_no_symlinks(path, "collection output").resolve(strict=True)
        if not checked.is_relative_to(collector.MAXWELL_DUST_ROOT):
            raise ValueError("search collection output escaped the Maxwell dust root")
    inventory, inventory_identity = _publication(
        inventory_path, expected_inventory_file_sha256, "inventory_sha256",
    )
    completion, completion_identity = _publication(
        completion_path, expected_completion_file_sha256, "completion_sha256",
    )
    if completion_identity["mtime_ns"] < inventory_identity["mtime_ns"]:
        raise ValueError("search collection completion predates its sealed inventory")
    before = _source_input_identity(
        balanced_dataset_launch_plan_path=balanced_dataset_launch_plan_path,
        plan=plan, plan_path=paths["plan"], source_root=paths["source_root"],
        source_archive=paths["source_archive"], schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
    )
    rows = tuple(collector._checked_task(plan, selected) for selected in plan["parents"])
    if len(rows) != 60 or tuple(row["array_task_id"] for row in rows) != tuple(range(60)):
        raise ValueError("search collection is not the complete ordered 60-task cohort")
    totals = collector._aggregate(rows)
    expected_inventory = {
        "schema": collector.V5_K1_BALANCED_FULL_SEARCH_INVENTORY_SCHEMA,
        "version": collector.V5_K1_BALANCED_FULL_SEARCH_INVENTORY_VERSION,
        "status": "PASS", "plan_sha256": plan["plan_sha256"],
        "source_bundle_sha256": plan["source"]["bundle_sha256"],
        "identity_authorization_sha256": plan["identity_authorization_sha256"],
        "task_count": 60, "totals": totals, "tasks": list(rows),
        "immutable_input_identity_pre": list(before),
        "immutable_input_identity_post": list(before),
        "all_task_bound_search_evidence_replayed_twice": True,
        "full_search_supervision_complete": True,
        "tuning_exact_budget_summary_complete": False,
        "gradient_training_authorized": False, "scientific_acceptance_evidence": False,
    }
    actual_inventory = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if canonical_json(actual_inventory) != canonical_json(expected_inventory):
        raise ValueError("sealed search inventory differs from actual task and source replay")
    expected_completion = {
        "schema": collector.V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_SCHEMA,
        "version": collector.V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_VERSION,
        "status": "PASS", "plan_sha256": plan["plan_sha256"],
        "inventory": {"path": str(inventory_path), "file_sha256": expected_inventory_file_sha256,
                      "inventory_sha256": inventory["inventory_sha256"]},
        "task_count": 60, "totals": totals,
        "all_task_bound_search_evidence_replayed_twice": True,
        "full_search_supervision_complete": True, "tuning_exact_budget_summary_complete": False,
        "gradient_training_authorized": False, "phase_c_model_acceptance": False,
        "paper_model_acceptance": False, "inventory_sealed_before_completion": True,
        "completion_written_last": True,
    }
    actual_completion = {key: value for key, value in completion.items()
                         if key not in ("completion_sha256", "slurm_job_id", "hostname")}
    if actual_completion != expected_completion:
        raise ValueError("search collection completion lost its inventory or claim binding")
    host = completion.get("hostname")
    job = completion.get("slurm_job_id")
    if (not isinstance(host, str) or not host.strip()
            or host.split(".", 1)[0].startswith(("max-wgs", "max-fs-display"))
            or not isinstance(job, str) or not job.isdecimal() or int(job) < 1):
        raise ValueError("search collection completion has no valid worker provenance")
    after = _source_input_identity(
        balanced_dataset_launch_plan_path=balanced_dataset_launch_plan_path,
        plan=plan, plan_path=paths["plan"], source_root=paths["source_root"],
        source_archive=paths["source_archive"], schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
    )
    replay_rows = tuple(collector._checked_task(plan, selected) for selected in plan["parents"])
    if (after != before or replay_rows != rows
            or read_only_identity(inventory_path, "search inventory") != inventory_identity
            or read_only_identity(completion_path, "search completion") != completion_identity):
        raise RuntimeError("search collection changed during read-only replay")
    if producer_identity is not None and read_only_identity(
        balanced_dataset_launch_plan_path, "producer launch plan",
    ) != producer_identity:
        raise RuntimeError("producer launch plan changed across collection replay")
    return {
        "plan": plan,
        "producer_launch_plan_file_identity": producer_identity,
        "inventory": inventory, "completion": completion,
        "inventory_file_identity": inventory_identity,
        "completion_file_identity": completion_identity,
        "writes_performed": False, "gradient_training_authorized": False,
        "scientific_acceptance_evidence": False,
    }
