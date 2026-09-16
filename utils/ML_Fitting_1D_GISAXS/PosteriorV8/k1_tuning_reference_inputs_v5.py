"""Worker publication of the complete collection-bound K1 tuning reference cohort."""

from dataclasses import asdict
import os
from pathlib import Path
import socket

from .calibrated_search_threshold_v5 import read_v5_checked_compatibility_calibration
from .formal_production_search_worker_contract_v5 import replay_v5_formal_production_stage_from_artifacts
from .frozen_search_pipeline_v5 import _catalog_artifact_id, _search_specs
from .grouped_dataset_v5 import read_v5_grouped_dataset
from .k1_balanced_full_search_authorization_v5 import (
    authorize_v5_k1_balanced_full_search_task, build_v5_k1_balanced_full_search_task_membership,
)
from .k1_balanced_full_search_collector_v5 import _worker_guard
from .k1_balanced_full_search_parent_v5 import project_v5_k1_balanced_full_search_parent
from .k1_balanced_full_search_replay_v5 import replay_v5_k1_balanced_full_search_collection
from .k1_balanced_full_search_runtime_v5 import V5K1BalancedFullSearchExecutableTask
from .k1_balanced_full_search_worker_v5 import _read_parent
from .k1_search_training_inputs_v5 import read_v5_k1_search_training_inputs
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_identity, read_regular_bytes
from .k1_training_chain_dataset_audit_v5 import k1_parent_set_sha256
from .paper_budget_evaluator_v5 import V5PaperBudgetEvaluationConfig
from .paper_checkpoint_selector_v5 import V5TuningQueryCohort, V5TuningQueryCohortMember
from .read_only_json_publication_v5 import publish_read_only_canonical_json
from .search_evidence_receipt_v5 import _ordered_tasks, read_v5_search_evidence_receipt
from .tuning_reference_executor_v5 import publish_v5_query_executor_reference


MAXWELL_DUST_ROOT = Path('/data/dust/user/zhaiyufe')
COHORT_SCHEMA = 'gisaxs.posterior_v8.k1_collection_tuning_reference_cohort/v1'
COMPLETION_SCHEMA = 'gisaxs.posterior_v8.k1_collection_tuning_reference_completion/v1'


def _stage(plan, binding):
    stage_binding = plan['k1_stage']
    identity = stage_binding['payload']['protocol']['calibration_identity']
    calibration = read_v5_checked_compatibility_calibration(
        binding['calibration_path'], expected_artifact_sha256=identity['artifact_sha256'],
        expected_file_sha256=identity['file_sha256'])
    stage = replay_v5_formal_production_stage_from_artifacts(
        stage_payload={**stage_binding['payload'], 'stage_sha256': stage_binding['stage_sha256']},
        schedule_path=binding['local_sobol_schedule_path'], calibration=calibration)
    return stage, calibration


def _query_entries(plan, selected, row, stage, calibration):
    """Reconstruct original tasks; never execute a new search or write old roots."""
    if selected['role'] != 'tuning_validation' or row['role'] != 'tuning_validation':
        raise ValueError('reference cohort may consume only tuning_validation shards')
    parent, _ = _read_parent(selected)
    projection = project_v5_k1_balanced_full_search_parent(
        parent, source_parent_artifact_sha256=selected['parent']['artifact_sha256'],
        expected_balanced_dataset_plan_sha256=plan['identity_authorization']['populations'][
            'tuning_validation']['plan_sha256'],
        expected_balanced_sobol_block_sha256=selected['balanced_sobol_block_sha256'],
        expected_role=selected['role'], expected_split_id=selected['split_id'],
        candidate_view_indices=tuple(plan['configuration']['candidate_view_indices']))
    membership = build_v5_k1_balanced_full_search_task_membership(
        projection, array_task_id=selected['array_task_id'], shard_index=selected['shard_index'],
        split_offset=selected['split_offset'], source_selection_sha256=selected['selection_sha256'])
    runtime = V5K1BalancedFullSearchExecutableTask(
        projection=projection, task_authorization=authorize_v5_k1_balanced_full_search_task(plan, membership),
        topology_schedule=stage.topology_schedule)
    persisted, parent_receipt = read_v5_grouped_dataset(row['projected_parent_path'])
    if parent_receipt.artifact_sha256 != row['projected_parent_artifact_sha256']:
        raise ValueError('tuning projected parent differs from the collection binding')
    bindings, catalog_files = {}, []
    source_root = Path(selected['search_output_root'])
    for design in runtime.query_designs:
        path = lexical_no_symlinks(source_root / 'query-catalogs' / f'{design.clean_group_id}.json',
                                   'tuning query catalog')
        before = read_only_identity(path, 'tuning query catalog')
        if before['mode_octal'] != '0400' or read_regular_bytes(path, 'tuning query catalog') != (
                design.to_json().encode('utf-8')):
            raise ValueError('persisted tuning query catalog does not reproduce')
        catalog_files.append((path, before))
        bindings[design.clean_group_id] = (_catalog_artifact_id(design), design.sha256)
    specs = _search_specs(runtime, persisted, runtime.recipes, bindings, calibration)
    tasks = _ordered_tasks(specs, stage.protocol, persisted)
    receipt = read_v5_search_evidence_receipt(
        row['evidence_receipt_path'], parent_dataset_path=row['projected_parent_path'],
        sidecar_path=row['sidecar_path'], require_training_eligible=True,
        expected_consumer_role='tuning_checkpoint_selection')
    if receipt.file_sha256 != row['evidence_receipt_file_sha256']:
        raise ValueError('tuning evidence receipt escaped its collection binding')
    evidence = receipt.manifest['branch_evidence']
    if len(tasks) != len(evidence) or len(specs) != row['recipe_count']:
        raise ValueError('tuning receipt does not cover every selected query and branch')
    grouped = {}
    receipt_root = Path(row['evidence_receipt_path']).parent.resolve(strict=True)
    for task, item in zip(tasks, evidence, strict=True):
        if not task.full_training_label_permitted or (
            task.audit_sha256 != item['task_audit_sha256']
            or task.query_index != item['query_index']
            or task.branch.global_key.wire_key != item['global_branch_key']
        ):
            raise ValueError('tuning task does not reproduce calibrated full-search receipt')
        path = lexical_no_symlinks(receipt_root / item['relative_path'], 'tuning executor').resolve(strict=True)
        if not path.is_relative_to(receipt_root):
            raise ValueError('tuning executor escaped its sealed search root')
        grouped.setdefault(task.query_index, []).append((task, path, item['artifact_sha256']))
    if set(grouped) != set(range(len(specs))):
        raise ValueError('tuning query indices are incomplete')
    for path, before in catalog_files:
        if read_only_identity(path, 'tuning query catalog') != before:
            raise RuntimeError('tuning query catalog changed during reconstruction')
    return tuple(tuple(grouped[index]) for index in range(len(specs)))


def publish_v5_k1_tuning_reference_cohort(plan_path, *, output_root, config,
                                         identity_publication_paths, **collection_binding):
    """Publish all plan-selected tuning references with completion last.

    This supplies evaluator inputs, not a model, checkpoint selection, gradient
    authorization or scientific acceptance. Failed partial roots stay immutable.
    """
    _worker_guard(dry_run=False, hostname=socket.gethostname(), environment=os.environ)
    if type(config) is not V5PaperBudgetEvaluationConfig:
        raise TypeError('tuning references require the typed evaluator configuration')
    root = lexical_no_symlinks(Path(output_root), 'tuning reference root')
    if not root.is_relative_to(MAXWELL_DUST_ROOT) or root == MAXWELL_DUST_ROOT:
        raise ValueError('tuning reference root must remain below Maxwell dust root')
    if root.exists():
        raise FileExistsError('refusing to reuse a tuning reference root')
    root.parent.resolve(strict=True)
    prepared = read_v5_k1_search_training_inputs(plan_path,
        identity_publication_paths=identity_publication_paths, **collection_binding)
    if prepared['training_inventory'] is None or not prepared['original_identity_publication_files']:
        raise ValueError('tuning reference publication requires original exclusion publications')
    checked = replay_v5_k1_balanced_full_search_collection(plan_path, **collection_binding)
    if checked['inventory_file_identity']['sha256'] != prepared['collection_inventory_file_sha256']:
        raise RuntimeError('collection changed before tuning reference preparation')
    plan = checked['plan']
    population = plan['identity_authorization']['populations']['tuning_validation']
    stage, calibration = _stage(plan, collection_binding)
    by_task = {row['array_task_id']: row for row in checked['inventory']['tasks']}
    root.mkdir(mode=0o700)
    references, members, reference_files = [], [], []
    for selected in plan['parents']:
        if selected['role'] != 'tuning_validation':
            continue
        row = by_task[selected['array_task_id']]
        for entries in _query_entries(plan, selected, row, stage, calibration):
            task = entries[0][0]
            path = root / f'query-{len(references):06d}.json'
            reference = publish_v5_query_executor_reference(
                path, query_task=task, branch_artifacts=entries,
                reference_set_id=f'k1-tuning-reference/{task.observation_id}',
                comparison_protocol_id=config.comparison_protocol_id,
                comparison_protocol_sha256=config.comparison_protocol_sha256)
            references.append(reference)
            members.append(V5TuningQueryCohortMember(
                query_id=reference.query_id, pairing_unit_id=reference.pairing_unit_id,
                reference_set_id=reference.reference_set_id, reference_set_sha256=reference.reference_set_sha256,
                query_context_sha256=reference.query_context_sha256,
                reference_representative_payload_set_sha256=reference.representative_payload_set_sha256))
            reference_files.append((path, read_only_identity(path, 'tuning reference')))
    groups = [value.pairing_unit_id for value in members]
    if (len(members) != population['clean_parent_count'] or len(set(groups)) != len(groups)
            or k1_parent_set_sha256(groups) != population['clean_group_set_sha256']):
        raise ValueError('tuning references do not cover the exact original clean-parent population')
    # Recheck all original publication files, all search shards and their source
    # after the final query, not merely after each individual branch.
    after = read_v5_k1_search_training_inputs(plan_path,
        identity_publication_paths=identity_publication_paths, **collection_binding)
    if after != prepared:
        raise RuntimeError('training/search identity changed during tuning reference publication')
    for path, identity in reference_files:
        if read_only_identity(path, 'tuning reference') != identity:
            raise RuntimeError('a published tuning reference changed before cohort completion')
    payload = {
        'schema': COHORT_SCHEMA, 'selection_split': 'tuning_validation',
        'cohort_id': f"k1-tuning/{plan['plan_sha256']}",
        'collection_binding': {key: str(value) for key, value in collection_binding.items()},
        'identity_publication_paths': {key: str(value) for key, value in identity_publication_paths.items()},
        'original_identity_authorization_sha256': prepared['original_identity_authorization_sha256'],
        'configuration': asdict(config),
        'members': [asdict(value) for value in sorted(members, key=lambda member: member.query_id)],
        'reference_files': [{'path': str(path), 'file_sha256': identity['sha256']}
                            for path, identity in reference_files],
        'gradient_training_authorized': False, 'scientific_acceptance_evidence': False,
    }
    cohort_path = root / 'tuning-reference-cohort-v1.json'
    digest = publish_read_only_canonical_json(cohort_path, payload)
    cohort = V5TuningQueryCohort(cohort_id=payload['cohort_id'], cohort_artifact_sha256=digest,
                               members=tuple(members))
    completion_path = root / 'completion-v1.json'
    publish_read_only_canonical_json(completion_path, {
        'schema': COMPLETION_SCHEMA, 'status': 'PASS', 'cohort_path': str(cohort_path),
        'cohort_file_sha256': digest, 'query_count': len(members),
        'source_input_pre_post_equal': True, 'completion_written_last': True,
        'gradient_training_authorized': False, 'scientific_acceptance_evidence': False,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
    })
    return cohort, tuple(references), completion_path
