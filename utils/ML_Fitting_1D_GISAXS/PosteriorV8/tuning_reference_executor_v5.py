"""Recover typed reference parameters from task-replayed executor artifacts."""

from pathlib import Path
from hashlib import sha256
import os
import socket

from .evaluation import LinearSolutionSnapshot, ReferenceMode
from .exact_search_executor_v5 import read_v5_exact_search_executor_artifact
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_identity
from .paper_representative_payload_v5 import (
    V5PaperParameterRepresentativePayload, V5_REFERENCE_REPRESENTATIVE_ROLE,
)
from .search_supervision_contract_v5 import V5FrozenSearchTask
from .grouped_artifact_v5 import canonical_json
from .paper_budget_evaluator_v5 import V5FrozenReferenceRepresentative, V5FrozenReferenceSet
from .k1_staging_files_v5 import read_regular_bytes
from .read_only_json_publication_v5 import publish_read_only_canonical_json


MAXWELL_DUST_ROOT = Path('/data/dust/user/zhaiyufe')
QUERY_REFERENCE_SCHEMA = 'gisaxs.posterior_v8.tuning_query_executor_reference/v1'


def read_v5_executor_reference_payloads(path, *, task, expected_artifact_sha256):
    """Return all representatives of one completed branch, without selecting any.

    The existing executor reader replays every exact curve, metric, physical
    constraint and clustering decision against the actual task. Production
    callers run this on workers and must separately bind the complete tuning
    cohort, all legal branches and the formal calibrated protocol. An empty
    tuple means a completed negative branch, not a missing observation.
    """
    if type(task) is not V5FrozenSearchTask:
        raise TypeError("reference extraction requires the original typed search task")
    selected = lexical_no_symlinks(Path(path), "reference executor artifact")
    before = read_only_identity(selected, "reference executor artifact")
    if before["mode_octal"] != "0400" or before["sha256"] != expected_artifact_sha256:
        raise ValueError("reference executor artifact differs from its sealed binding")
    artifact = read_v5_exact_search_executor_artifact(selected, task=task)
    if artifact.receipt.artifact_sha256 != expected_artifact_sha256:
        raise ValueError("reference executor reader returned another artifact")
    if artifact.manifest["completed"] is not True:
        raise ValueError("incomplete search cannot supply tuning references")
    values = []
    arrays = artifact.arrays
    for row, index in zip(artifact.manifest["representatives"],
                          arrays["representative_candidate_index"], strict=True):
        index = int(index)
        components, resolution = task.codec.decode(arrays["candidate_local_unit"][index])
        linear = LinearSolutionSnapshot(
            background=arrays["candidate_background"][index],
            particle_amplitudes=tuple(arrays["candidate_particle_amplitudes"][index, :len(components)]),
            resolution_amplitude=arrays["candidate_resolution_amplitude"][index],
            k=arrays["candidate_k"][index],
        )
        identifier = row["artifact_id"]
        values.append(V5PaperParameterRepresentativePayload(
            representative_id=identifier, role=V5_REFERENCE_REPRESENTATIVE_ROLE,
            parameter=ReferenceMode(reference_id=identifier, topology_id=task.codec.topology_id,
                                    components=components, resolution=resolution, linear_solution=linear),
            global_branch_key=task.branch.global_key.wire_key,
            query_context_sha256=task.universal_context.audit_sha256,
            source_artifact_sha256=expected_artifact_sha256,
        ))
    if read_only_identity(selected, "reference executor artifact") != before:
        raise RuntimeError("reference executor artifact changed during parameter extraction")
    return tuple(values)


def read_v5_query_executor_reference_payloads(*, query_task, branch_artifacts):
    """Replay every legal branch of one query, including completed negatives.

    Entries are (original task, sealed artifact path, external file SHA). This
    reader does not publish a reference bank or grant training authority.
    """
    if type(query_task) is not V5FrozenSearchTask:
        raise TypeError("query reference extraction requires an original typed task")
    entries = tuple(branch_artifacts)
    count = query_task.universal_context.branch_count
    if len(entries) != count or count < 1:
        raise ValueError("query reference extraction requires all legal branches")
    # Per-branch context hashes legitimately differ; the universal context,
    # observation, protocol and calibrated threshold must remain identical.
    def query_identity(task):
        payload = task.audit_payload()
        payload.pop("global_branch_key")
        payload.pop("context_sha256")
        payload["query_catalog_artifact_id"] = task.query_catalog_artifact_id
        return payload

    expected = query_identity(query_task)
    indices, paths, identities = set(), set(), []
    for task, path, digest in entries:
        if type(task) is not V5FrozenSearchTask:
            raise TypeError("every branch requires its original typed task")
        if type(task.branch_index) is not int or not 0 <= task.branch_index < count:
            raise ValueError("invalid query reference branch index")
        if task.branch_index in indices or query_identity(task) != expected:
            raise ValueError("duplicate branch or mixed query reference inputs")
        indices.add(task.branch_index)
        selected = lexical_no_symlinks(Path(path), "query reference executor artifact")
        if selected in paths:
            raise ValueError("query reference branches must bind distinct artifact paths")
        paths.add(selected)
        identity = read_only_identity(selected, "query reference executor artifact")
        if identity["mode_octal"] != "0400" or identity["sha256"] != digest:
            raise ValueError("query reference artifact differs from its sealed binding")
        identities.append((selected, identity))
    values = tuple(
        value
        for task, path, digest in sorted(entries, key=lambda entry: entry[0].branch_index)
        for value in read_v5_executor_reference_payloads(
            path, task=task, expected_artifact_sha256=digest)
    )
    for path, identity in identities:
        if read_only_identity(path, "query reference executor artifact") != identity:
            raise RuntimeError("query reference input changed during all-branch replay")
    if not values:
        raise ValueError("completed query has no compatible reference; cannot silently omit it")
    if len({value.representative_id for value in values}) != len(values):
        raise ValueError("query reference representatives have duplicate identities")
    return values


def _query_reference_document(query_task, entries, reference_set_id,
                              comparison_protocol_id, comparison_protocol_sha256):
    values = read_v5_query_executor_reference_payloads(
        query_task=query_task, branch_artifacts=entries)
    payload = {
        'schema': QUERY_REFERENCE_SCHEMA,
        'query_id': query_task.observation_id,
        'pairing_unit_id': query_task.clean_group_id,
        'reference_set_id': reference_set_id,
        'comparison_protocol_id': comparison_protocol_id,
        'comparison_protocol_sha256': comparison_protocol_sha256,
        'branches': [
            {'task_sha256': task.audit_sha256, 'branch_index': task.branch_index,
             'artifact_path': str(Path(path).absolute()), 'artifact_sha256': digest}
            for task, path, digest in sorted(entries, key=lambda entry: entry[0].branch_index)
        ],
        'representatives': [value.audit_payload() for value in values],
        'scientific_acceptance_evidence': False,
        'training_authorized': False,
    }
    digest = sha256(canonical_json(payload).encode('utf-8')).hexdigest()
    reference = V5FrozenReferenceSet(
        query_id=query_task.observation_id, pairing_unit_id=query_task.clean_group_id,
        reference_set_id=reference_set_id, reference_set_sha256=digest,
        comparison_protocol_id=comparison_protocol_id,
        comparison_protocol_sha256=comparison_protocol_sha256,
        representatives=tuple(V5FrozenReferenceRepresentative(
            representative_id=value.representative_id, payload=value) for value in values),
    )
    return payload, reference


def publish_v5_query_executor_reference(path, *, query_task, branch_artifacts,
                                       reference_set_id, comparison_protocol_id,
                                       comparison_protocol_sha256):
    """Publish one task-replayed query reference on a worker, exclusively."""
    from .k1_balanced_full_search_collector_v5 import _worker_guard

    _worker_guard(dry_run=False, hostname=socket.gethostname(), environment=os.environ)
    selected = lexical_no_symlinks(Path(path), 'query reference publication')
    if not selected.is_relative_to(MAXWELL_DUST_ROOT) or selected == MAXWELL_DUST_ROOT:
        raise ValueError('query reference publication must remain below Maxwell dust root')
    if selected.exists():
        raise FileExistsError('refusing to reuse a query reference publication')
    payload, reference = _query_reference_document(
        query_task, tuple(branch_artifacts), reference_set_id,
        comparison_protocol_id, comparison_protocol_sha256)
    digest = publish_read_only_canonical_json(selected, payload)
    identity = read_only_identity(selected, 'query reference publication')
    if digest != reference.reference_set_sha256 or identity['sha256'] != digest:
        raise RuntimeError('published query reference bytes differ from reconstructed reference')
    return reference


def read_v5_query_executor_reference(path, *, expected_file_sha256, query_task,
                                    branch_artifacts, reference_set_id,
                                    comparison_protocol_id, comparison_protocol_sha256):
    """Return an evaluator-ready reference only after replaying its full source."""
    selected = lexical_no_symlinks(Path(path), 'query reference publication')
    before = read_only_identity(selected, 'query reference publication')
    if before['mode_octal'] != '0400' or before['sha256'] != expected_file_sha256:
        raise ValueError('query reference publication escaped its external binding')
    payload, reference = _query_reference_document(
        query_task, tuple(branch_artifacts), reference_set_id,
        comparison_protocol_id, comparison_protocol_sha256)
    if (read_regular_bytes(selected, 'query reference publication')
            != canonical_json(payload).encode('utf-8')
            or reference.reference_set_sha256 != expected_file_sha256):
        raise ValueError('query reference publication does not reproduce from all branch sources')
    if read_only_identity(selected, 'query reference publication') != before:
        raise RuntimeError('query reference publication changed during source replay')
    return reference
